"""
load_pitcher_game_logs.py
--------------------------
Fetches per-game pitching data from the MLB Stats API for 2024 + 2025 + 2026.
Stores in pitcher_game_log table which is used to compute:
  - Pitcher rest days since last start
  - Pitcher pitch count last start
  - Bullpen IP last 4 days
  - Home/away win% splits

Run once to backfill, then daily as part of mlb_quick_update.py.

Usage:
  python scripts/load_pitcher_game_logs.py           # backfill all seasons
  python scripts/load_pitcher_game_logs.py 2026      # single season only
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timezone
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import execute_values

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)

GAMES_TABLE  = os.getenv("MLB_GAMES_TABLE", "games")
LOG_TABLE    = "pitcher_game_log"

HTTP_TIMEOUT = 20
SLEEP        = float(os.getenv("REQUEST_SLEEP_SECONDS", "0.1"))

MLB_BASE     = "https://statsapi.mlb.com/api/v1"
SEASONS      = [2024, 2025, 2026]


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def conn():
    return psycopg2.connect(DATABASE_URL)


def ensure_tables(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {LOG_TABLE} (
            game_pk         BIGINT  NOT NULL,
            official_date   DATE    NOT NULL,
            season          INT     NOT NULL,
            pitcher_id      BIGINT  NOT NULL,
            pitcher_name    TEXT,
            team            TEXT,
            side            TEXT,        -- 'home' or 'away'
            role            TEXT,        -- 'SP' or 'RP'
            innings_pitched DOUBLE PRECISION,
            pitch_count     INT,
            batters_faced   INT,
            strikes         INT,
            balls           INT,
            hits_allowed    INT,
            runs_allowed    INT,
            er_allowed      INT,
            strikeouts      INT,
            walks           INT,
            home_runs       INT,
            win             BOOLEAN,
            loss            BOOLEAN,
            save_           BOOLEAN,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (game_pk, pitcher_id)
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_pgl_date    ON {LOG_TABLE}(official_date);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_pgl_pitcher ON {LOG_TABLE}(pitcher_id);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_pgl_team    ON {LOG_TABLE}(team);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_pgl_season  ON {LOG_TABLE}(season);")


def already_loaded(cur, game_pk: int) -> bool:
    cur.execute(f"SELECT 1 FROM {LOG_TABLE} WHERE game_pk = %s LIMIT 1", (game_pk,))
    return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# MLB API
# ---------------------------------------------------------------------------

def get_json(url: str, params: dict = None) -> dict:
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_game_pitcher_logs(game_pk: int, official_date, season: int,
                             home_team: str, away_team: str) -> list[dict]:
    """
    Fetch all pitcher lines from a completed game's boxscore.
    Returns list of dicts, one per pitcher appearance.
    """
    try:
        data = get_json(f"{MLB_BASE}/game/{game_pk}/boxscore")
    except Exception as e:
        print(f"    boxscore fetch failed for {game_pk}: {e}")
        return []

    teams = data.get("teams") or {}
    rows  = []

    for side in ("home", "away"):
        team_data = teams.get(side) or {}
        team_name = home_team if side == "home" else away_team
        pitchers  = (team_data.get("pitchers") or [])
        players   = team_data.get("players") or {}

        # First pitcher listed is the SP
        for idx, pid in enumerate(pitchers):
            key    = f"ID{pid}"
            player = players.get(key) or {}
            person = player.get("person") or {}
            stats  = (player.get("stats") or {}).get("pitching") or {}

            # Parse innings pitched (stored as "5.2" = 5 and 2/3)
            ip_str = stats.get("inningsPitched", "0.0")
            try:
                ip_parts = str(ip_str).split(".")
                ip = int(ip_parts[0]) + (int(ip_parts[1]) / 3 if len(ip_parts) > 1 else 0)
            except Exception:
                ip = 0.0

            role = "SP" if idx == 0 else "RP"

            # Wins/losses/saves from game info
            note = player.get("gameStatus") or {}

            rows.append({
                "game_pk":         game_pk,
                "official_date":   official_date,
                "season":          season,
                "pitcher_id":      int(pid),
                "pitcher_name":    person.get("fullName") or "",
                "team":            team_name,
                "side":            side,
                "role":            role,
                "innings_pitched": round(ip, 4),
                "pitch_count":     stats.get("pitchesThrown"),
                "batters_faced":   stats.get("battersFaced"),
                "strikes":         stats.get("strikes"),
                "balls":           stats.get("balls"),
                "hits_allowed":    stats.get("hits"),
                "runs_allowed":    stats.get("runs"),
                "er_allowed":      stats.get("earnedRuns"),
                "strikeouts":      stats.get("strikeOuts"),
                "walks":           stats.get("baseOnBalls"),
                "home_runs":       stats.get("homeRuns"),
                "win":             note.get("isWin", False),
                "loss":            note.get("isLoss", False),
                "save_":           note.get("isSave", False),
            })

    return rows


def upsert_logs(cur, rows: list[dict]) -> int:
    if not rows:
        return 0

    data = [
        (
            r["game_pk"], r["official_date"], r["season"],
            r["pitcher_id"], r["pitcher_name"], r["team"], r["side"], r["role"],
            r["innings_pitched"], r["pitch_count"], r["batters_faced"],
            r["strikes"], r["balls"], r["hits_allowed"], r["runs_allowed"],
            r["er_allowed"], r["strikeouts"], r["walks"], r["home_runs"],
            r["win"], r["loss"], r["save_"],
            datetime.now(timezone.utc),
        )
        for r in rows
    ]

    sql = f"""
    INSERT INTO {LOG_TABLE} (
        game_pk, official_date, season,
        pitcher_id, pitcher_name, team, side, role,
        innings_pitched, pitch_count, batters_faced,
        strikes, balls, hits_allowed, runs_allowed,
        er_allowed, strikeouts, walks, home_runs,
        win, loss, save_, updated_at
    ) VALUES %s
    ON CONFLICT (game_pk, pitcher_id) DO UPDATE SET
        pitcher_name    = EXCLUDED.pitcher_name,
        team            = EXCLUDED.team,
        innings_pitched = EXCLUDED.innings_pitched,
        pitch_count     = EXCLUDED.pitch_count,
        batters_faced   = EXCLUDED.batters_faced,
        strikes         = EXCLUDED.strikes,
        balls           = EXCLUDED.balls,
        hits_allowed    = EXCLUDED.hits_allowed,
        runs_allowed    = EXCLUDED.runs_allowed,
        er_allowed      = EXCLUDED.er_allowed,
        strikeouts      = EXCLUDED.strikeouts,
        walks           = EXCLUDED.walks,
        home_runs       = EXCLUDED.home_runs,
        win             = EXCLUDED.win,
        loss            = EXCLUDED.loss,
        save_           = EXCLUDED.save_,
        updated_at      = NOW();
    """
    execute_values(cur, sql, data, page_size=500)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seasons = SEASONS
    if len(sys.argv) > 1:
        try:
            seasons = [int(sys.argv[1])]
        except ValueError:
            print(f"Invalid season argument: {sys.argv[1]}")
            return

    print(f"{'='*55}")
    print(f"  load_pitcher_game_logs.py")
    print(f"  Seasons: {seasons}")
    print(f"{'='*55}\n")

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_tables(cur)
            c.commit()
            print("Tables ready.\n")

            total_games    = 0
            total_skipped  = 0
            total_pitchers = 0

            for season in seasons:
                print(f"--- Season {season} ---")

                # Get all completed games for this season from our games table
                cur.execute(f"""
                    SELECT game_pk, official_date, home_team, away_team
                    FROM {GAMES_TABLE}
                    WHERE season = %s
                      AND home_score IS NOT NULL
                      AND away_score IS NOT NULL
                      AND game_type = 'R'
                    ORDER BY official_date, game_pk
                """, (season,))
                games = cur.fetchall()
                print(f"  Found {len(games)} completed regular season games\n")

                for game_pk, official_date, home_team, away_team in games:
                    if already_loaded(cur, int(game_pk)):
                        total_skipped += 1
                        continue

                    rows = fetch_game_pitcher_logs(
                        int(game_pk), official_date, season,
                        home_team, away_team
                    )

                    if rows:
                        n = upsert_logs(cur, rows)
                        total_pitchers += n
                        total_games    += 1
                        print(f"  {official_date} game {game_pk}: {n} pitcher lines")
                    else:
                        print(f"  {official_date} game {game_pk}: no data")

                    c.commit()

                    if SLEEP:
                        time.sleep(SLEEP)

                print(f"\n  Season {season} done.\n")

        print(f"{'='*55}")
        print(f"  Games processed  : {total_games}")
        print(f"  Games skipped    : {total_skipped} (already cached)")
        print(f"  Pitcher lines    : {total_pitchers}")
        print(f"{'='*55}")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()