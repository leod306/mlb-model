"""
load_rotowire_lineups.py
------------------------
Scrapes RotoWire's free projected lineup page and stores batting orders
in the lineups table with source='rotowire'.

Official MLB lineups (source='mlb') always win — never overwritten.
Falls back to league avg OPS if no lineups available.

Run order: BEFORE mlb_engine_daily.py
"""
from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if os.getenv("DYNO") is None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass

from app.db import engine


# =============================================================================
# CONFIG
# =============================================================================

ROTOWIRE_URL    = "https://www.rotowire.com/baseball/daily-lineups.php"
REQUEST_TIMEOUT = 20
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TEAM_MAP = {
    # Full names
    "Arizona Diamondbacks": "ARI",   "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",      "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",           "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",        "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",       "Detroit Tigers": "DET",
    "Houston Astros": "HOU",         "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",     "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",          "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",        "New York Mets": "NYM",
    "New York Yankees": "NYY",       "Oakland Athletics": "ATH",
    "Athletics": "ATH",              "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",     "San Diego Padres": "SD",
    "San Francisco Giants": "SF",    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",          "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
    # Short names
    "D-backs": "ARI",   "Diamondbacks": "ARI",
    "Braves": "ATL",    "Orioles": "BAL",
    "Red Sox": "BOS",   "Cubs": "CHC",
    "White Sox": "CWS", "Reds": "CIN",
    "Guardians": "CLE", "Rockies": "COL",
    "Tigers": "DET",    "Astros": "HOU",
    "Royals": "KC",     "Angels": "LAA",
    "Dodgers": "LAD",   "Marlins": "MIA",
    "Brewers": "MIL",   "Twins": "MIN",
    "Mets": "NYM",      "Yankees": "NYY",
    "Phillies": "PHI",  "Pirates": "PIT",
    "Padres": "SD",     "Giants": "SF",
    "Mariners": "SEA",  "Cardinals": "STL",
    "Rays": "TB",       "Rangers": "TEX",
    "Blue Jays": "TOR", "Nationals": "WSH",
    # Abbreviations
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC":  "KC",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "ATH",
    "ATH": "ATH", "PHI": "PHI", "PIT": "PIT", "SD":  "SD",
    "SF":  "SF",  "SEA": "SEA", "STL": "STL", "TB":  "TB",
    "TEX": "TEX", "TOR": "TOR", "WSH": "WSH",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def should_skip(raw: str) -> bool:
    """
    Return True if this text is NOT a real batter.
    Filters: SP name+ERA lines, placeholder text, DK salaries.
    """
    t = raw.strip().replace("\xa0", " ").replace("\u00a0", " ")
    if not t or len(t) < 2:
        return True
    # Pitcher line — contains ERA anywhere in the string
    if "ERA" in t:
        return True
    # Placeholder rows
    if t in ("Expected Lineup", "Confirmed Lineup", "Projected Lineup"):
        return True
    # DraftKings salary rows like $4,100
    if re.match(r"^\$[\d,]+$", t):
        return True
    return False


def clean_name(raw: str) -> str:
    """Strip handedness suffix (L/R/S at end) and whitespace."""
    t = raw.strip().replace("\xa0", " ").replace("\u00a0", " ")
    # Remove trailing handedness indicator
    t = re.sub(r"\s*[LRS]$", "", t).strip()
    return t


# =============================================================================
# DB HELPERS
# =============================================================================

def ensure_db_ready() -> None:
    """Add source column and make player_id nullable if needed."""
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE lineups
            ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'mlb'
        """))
        conn.execute(text("""
            ALTER TABLE lineups
            ALTER COLUMN player_id DROP NOT NULL
        """))
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS lineups_game_team_order_idx
            ON lineups (game_pk, team, batting_order)
        """))


def get_official_game_pks(target_date: date) -> set:
    try:
        with engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT game_pk FROM lineups
                WHERE official_date = :d AND source = 'mlb'
            """), {"d": target_date}).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def get_game_pk(home_team: str, away_team: str, target_date: date) -> Optional[int]:
    try:
        with engine.begin() as conn:
            row = conn.execute(text("""
                SELECT game_pk FROM games
                WHERE official_date = :d
                  AND home_team = :home
                  AND away_team = :away
                  AND season = :season
                LIMIT 1
            """), {
                "d": target_date,
                "home": home_team,
                "away": away_team,
                "season": target_date.year,
            }).fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def upsert_rows(rows: list) -> int:
    """Insert rows game by game — one failure won't cascade."""
    if not rows:
        return 0

    sql = text("""
        INSERT INTO lineups (
            game_pk, official_date, team, side,
            batting_order, player_id, player_name, position, source
        ) VALUES (
            :game_pk, :official_date, :team, :side,
            :batting_order, :player_id, :player_name, :position, 'rotowire'
        )
        ON CONFLICT (game_pk, team, batting_order) DO UPDATE SET
            player_name = EXCLUDED.player_name,
            position    = EXCLUDED.position,
            source      = CASE
                            WHEN lineups.source = 'mlb' THEN 'mlb'
                            ELSE 'rotowire'
                          END,
            updated_at  = NOW()
        WHERE lineups.source != 'mlb'
    """)

    # Group by game_pk so each game is its own transaction
    by_game: dict = {}
    for r in rows:
        by_game.setdefault(r["game_pk"], []).append(r)

    inserted = 0
    for game_pk, game_rows in by_game.items():
        try:
            with engine.begin() as conn:
                conn.execute(sql, game_rows)
            inserted += len(game_rows)
            log(f"  ✅ {game_rows[0]['team']}/{game_rows[-1]['team']} ({game_pk}): {len(game_rows)} rows")
        except Exception as e:
            log(f"  ⚠️  game {game_pk} failed: {e}")

    return inserted


# =============================================================================
# SCRAPER
# =============================================================================

def fetch_page() -> Optional[str]:
    try:
        resp = requests.get(ROTOWIRE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        log(f"❌ Failed to fetch RotoWire: {e}")
        return None


def norm_team(name: str) -> Optional[str]:
    name = name.strip()
    if name in TEAM_MAP:
        return TEAM_MAP[name]
    for k, v in TEAM_MAP.items():
        if len(k) > 3 and k.lower() in name.lower():
            return v
    return None


def parse_lineups(html: str, target_date: date) -> list:
    soup = BeautifulSoup(html, "html.parser")
    boxes = soup.select(".lineup__box")
    log(f"  Found {len(boxes)} lineup boxes")

    all_rows = []
    for box in boxes:
        try:
            rows = _parse_box(box, target_date)
            all_rows.extend(rows)
        except Exception as e:
            log(f"  ⚠️  Box parse error: {e}")
    return all_rows


def _parse_box(box, target_date: date) -> list:
    rows = []

    # Team names
    team_els = box.select(".lineup__team")
    if len(team_els) < 2:
        return []

    away_raw  = team_els[0].get_text(strip=True)
    home_raw  = team_els[1].get_text(strip=True)
    away_abbr = norm_team(away_raw)
    home_abbr = norm_team(home_raw)

    if not away_abbr or not home_abbr:
        log(f"  ⚠️  Could not map teams: {away_raw} @ {home_raw}")
        return []

    game_pk = get_game_pk(home_abbr, away_abbr, target_date)
    if not game_pk:
        log(f"  ⚠️  No game_pk: {away_abbr} @ {home_abbr}")
        return []

    # Two batting order lists — first = away, second = home
    player_lists = box.select(".lineup__list")
    if len(player_lists) < 2:
        return []

    for side_idx, (side, team_abbr) in enumerate([("away", away_abbr), ("home", home_abbr)]):
        if side_idx >= len(player_lists):
            continue

        batter_els  = player_lists[side_idx].select("li")
        batting_order = 0

        for el in batter_els:
            raw = el.get_text(strip=True)

            # ── FILTER non-batter rows BEFORE assigning order ──
            if should_skip(raw):
                continue

            batting_order += 1
            if batting_order > 9:
                break

            # Extract position from nested span
            position = None
            pos_el   = el.select_one(".lineup__pos")
            if pos_el:
                position   = pos_el.get_text(strip=True)
                player_name = clean_name(raw.replace(pos_el.get_text(strip=True), ""))
            else:
                player_name = clean_name(raw)

            if not player_name or len(player_name) < 2:
                batting_order -= 1  # don't count empty rows
                continue

            rows.append({
                "game_pk":       game_pk,
                "official_date": target_date,
                "team":          team_abbr,
                "side":          side,
                "batting_order": batting_order,
                "player_id":     None,
                "player_name":   player_name,
                "position":      position,
            })

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    today = date.today()
    log("=" * 55)
    log(f"  load_rotowire_lineups.py  |  {today}")
    log("=" * 55)

    log("\nEnsuring DB is ready...")
    try:
        ensure_db_ready()
    except Exception as e:
        log(f"  ⚠️  DB setup warning (may already exist): {e}")

    log("\nFetching RotoWire lineup page...")
    html = fetch_page()
    if not html:
        log("❌ Could not fetch RotoWire. Skipping.")
        return

    official_pks = get_official_game_pks(today)
    if official_pks:
        log(f"  Official MLB lineups exist for {len(official_pks)} games — will be preserved")

    log("\nParsing lineups...")
    rows = parse_lineups(html, today)

    if not rows:
        log("⚠️  No lineup rows parsed.")
        return

    # Skip games with official lineups
    if official_pks:
        before = len(rows)
        rows   = [r for r in rows if r["game_pk"] not in official_pks]
        if before - len(rows):
            log(f"  Skipped {before - len(rows)} rows (official lineups exist)")

    games = {r["game_pk"] for r in rows}
    log(f"  Parsed {len(rows)} batter rows across {len(games)} games\n")

    inserted = upsert_rows(rows)
    log(f"\n✅ Upserted {inserted} RotoWire lineup rows for {today}")


if __name__ == "__main__":
    main()