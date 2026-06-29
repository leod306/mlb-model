#!/usr/bin/env python3
"""
grade_player_props.py

Scores player prop picks against actual MLB boxscore stats and writes
WIN / LOSS / PUSH back to the player_props table.

Usage:
    python scripts/grade_player_props.py              # grade yesterday
    python scripts/grade_player_props.py --date 2026-06-28
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import requests
from sqlalchemy import text
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if os.getenv("DYNO") is None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass

from app.db import engine

BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

# prop_type -> (stat_side, mlb_api_key)
PROP_STAT_MAP = {
    "batter_hits":        ("batting",  "hits"),
    "batter_total_bases": ("batting",  "totalBases"),
    "batter_home_runs":   ("batting",  "homeRuns"),
    "batter_rbis":        ("batting",  "rbi"),
    "batter_runs_scored": ("batting",  "runs"),
    "batter_stolen_bases":("batting",  "stolenBases"),
    "batter_walks":       ("batting",  "baseOnBalls"),
    "pitcher_strikeouts": ("pitching", "strikeOuts"),
    "pitcher_outs":       ("pitching", "outs"),
}


def ensure_columns():
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE player_props
            ADD COLUMN IF NOT EXISTS result       VARCHAR(10),
            ADD COLUMN IF NOT EXISTS actual_value FLOAT,
            ADD COLUMN IF NOT EXISTS graded_at    TIMESTAMP
        """))


def fetch_boxscore(game_pk: int) -> dict:
    url = BOXSCORE_URL.format(game_pk=game_pk)
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def extract_player_stats(boxscore: dict) -> dict[str, dict]:
    """Returns {full_name_lower: {'batting': {...}, 'pitching': {...}}}"""
    stats: dict[str, dict] = {}
    for side in ("home", "away"):
        players = boxscore.get("teams", {}).get(side, {}).get("players", {})
        for _, pdata in players.items():
            name = pdata.get("person", {}).get("fullName", "")
            if not name:
                continue
            stats[name.lower()] = {
                "batting":  pdata.get("stats", {}).get("batting",  {}),
                "pitching": pdata.get("stats", {}).get("pitching", {}),
            }
    return stats


def fuzzy_match(name_key: str, player_stats: dict[str, dict]) -> dict | None:
    """Try last-name match as fallback."""
    last = name_key.split()[-1]
    matches = [v for k, v in player_stats.items() if k.split()[-1] == last]
    return matches[0] if len(matches) == 1 else None


def grade_result(actual: float, line: float, pick: str) -> str:
    if actual > line:
        return "WIN" if pick == "OVER" else "LOSS"
    elif actual < line:
        return "WIN" if pick == "UNDER" else "LOSS"
    else:
        return "PUSH"


def grade_date(target_date: date) -> None:
    ensure_columns()

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, game_pk, player_name, prop_type, line, pick
            FROM player_props
            WHERE prop_date = :d
              AND pick IN ('OVER', 'UNDER')
              AND result IS NULL
        """), {"d": target_date}).fetchall()

    if not rows:
        print(f"No ungraded props found for {target_date}.")
        return

    print(f"Grading {len(rows)} props for {target_date}...")

    # Group by game_pk to minimise API calls
    games: dict[int, list] = {}
    for row in rows:
        games.setdefault(row.game_pk, []).append(row)

    wins = losses = pushes = skipped = 0

    for game_pk, game_rows in games.items():
        try:
            boxscore = fetch_boxscore(game_pk)
        except Exception as e:
            print(f"  game {game_pk}: fetch error — {e}")
            skipped += len(game_rows)
            continue

        player_stats = extract_player_stats(boxscore)

        for row in game_rows:
            name_key  = row.player_name.lower()
            stat_info = PROP_STAT_MAP.get(row.prop_type)
            if not stat_info:
                skipped += 1
                continue

            side, stat_key = stat_info
            pdata = player_stats.get(name_key) or fuzzy_match(name_key, player_stats)
            if pdata is None:
                skipped += 1
                continue

            raw = pdata.get(side, {}).get(stat_key)
            if raw is None:
                skipped += 1
                continue

            actual = float(raw)
            result = grade_result(actual, float(row.line), row.pick)

            if result == "WIN":   wins   += 1
            elif result == "LOSS": losses += 1
            else:                  pushes += 1

            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE player_props
                       SET result = :result,
                           actual_value = :actual,
                           graded_at = NOW()
                     WHERE id = :id
                """), {"result": result, "actual": actual, "id": row.id})

            icon = "✓" if result == "WIN" else ("✗" if result == "LOSS" else "~")
            print(f"  {icon} {row.player_name} {row.pick} {row.line} {row.prop_type}: "
                  f"actual={actual} → {result}")

    total = wins + losses + pushes
    pct   = round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0
    print(f"\n{'='*50}")
    print(f"Results for {target_date}: {wins}W / {losses}L / {pushes}P")
    print(f"Win rate: {pct}%  (skipped {skipped} — no boxscore data)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade player prop picks vs actual stats.")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    print(f"{'='*50}")
    print(f"grade_player_props.py | target: {target}")
    print(f"{'='*50}")
    grade_date(target)
