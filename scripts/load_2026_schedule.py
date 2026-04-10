from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict

import requests
import psycopg2
from psycopg2.extras import execute_values

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)

SEASON      = int(os.getenv("MLB_SEASON", "2026"))
GAME_TYPES  = os.getenv("MLB_GAME_TYPES", "R")
GAMES_TABLE = os.getenv("MLB_GAMES_TABLE", "games")

MLB_BASE     = "https://statsapi.mlb.com/api/v1"
HTTP_TIMEOUT = 30


def conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)


def get_json(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def utc_dt(iso_z: str) -> datetime:
    return datetime.fromisoformat(iso_z.replace("Z", "+00:00")).astimezone(timezone.utc)


def ensure_games(cur):
    cur.execute(f"CREATE TABLE IF NOT EXISTS {GAMES_TABLE} (game_pk BIGINT PRIMARY KEY);")

    cols = [
        ("official_date", "DATE"),
        ("game_date_utc", "TIMESTAMPTZ"),
        ("season",        "INT"),
        ("game_type",     "TEXT"),
        ("status",        "TEXT"),
        ("home_team_id",  "INT"),
        ("away_team_id",  "INT"),
        ("home_team",     "TEXT"),
        ("away_team",     "TEXT"),
        ("home_score",    "INT"),
        ("away_score",    "INT"),
    ]
    for col, typ in cols:
        cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS {col} {typ};")

    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{GAMES_TABLE}_official_date ON {GAMES_TABLE}(official_date);")


def team_map(season: int) -> Dict[int, str]:
    data = get_json(f"{MLB_BASE}/teams", {"sportId": 1, "season": season})
    out: Dict[int, str] = {}
    for t in data.get("teams", []) or []:
        tid  = t.get("id")
        abbr = t.get("abbreviation") or t.get("teamCode") or t.get("fileCode") or t.get("name")
        if tid and abbr:
            out[int(tid)] = str(abbr)
    return out


def main():
    tmap = team_map(SEASON)

    data = get_json(
        f"{MLB_BASE}/schedule",
        {
            "sportId":   1,
            "season":    SEASON,
            "gameTypes": GAME_TYPES,
            "hydrate":   "team",
        },
    )

    rows_by_pk: dict = {}
    now = datetime.now(timezone.utc)

    for d in data.get("dates", []) or []:
        for g in d.get("games", []) or []:
            teams     = g.get("teams") or {}
            home_data = teams.get("home") or {}
            away_data = teams.get("away") or {}
            home_team = home_data.get("team") or {}
            away_team = away_data.get("team") or {}

            if not home_team.get("id") or not away_team.get("id"):
                continue

            home_score = home_data.get("score")
            away_score = away_data.get("score")

            game_pk = int(g["gamePk"])

            # Deduplicate by game_pk — last write wins
            # (handles doubleheaders where same game appears twice in API response)
            rows_by_pk[game_pk] = (
                game_pk,
                (g.get("officialDate") or d.get("date")),
                utc_dt(g["gameDate"]),
                SEASON,
                str(g.get("gameType") or ""),
                str((g.get("status") or {}).get("detailedState") or ""),
                int(home_team["id"]),
                int(away_team["id"]),
                str(home_team.get("abbreviation") or tmap.get(int(home_team["id"])) or ""),
                str(away_team.get("abbreviation") or tmap.get(int(away_team["id"])) or ""),
                home_score,
                away_score,
                now,
            )

    rows = list(rows_by_pk.values())

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_games(cur)

            if rows:
                sql = f"""
                INSERT INTO {GAMES_TABLE} (
                  game_pk, official_date, game_date_utc, season, game_type, status,
                  home_team_id, away_team_id, home_team, away_team,
                  home_score, away_score, updated_at
                )
                VALUES %s
                ON CONFLICT (game_pk) DO UPDATE SET
                  official_date = EXCLUDED.official_date,
                  game_date_utc = EXCLUDED.game_date_utc,
                  season        = EXCLUDED.season,
                  game_type     = EXCLUDED.game_type,
                  status        = EXCLUDED.status,
                  home_team_id  = EXCLUDED.home_team_id,
                  away_team_id  = EXCLUDED.away_team_id,
                  home_team     = EXCLUDED.home_team,
                  away_team     = EXCLUDED.away_team,
                  home_score    = EXCLUDED.home_score,
                  away_score    = EXCLUDED.away_score,
                  updated_at    = NOW();
                """
                execute_values(cur, sql, rows, page_size=500)

            c.commit()

        print(f"Upserted {len(rows)} games for season={SEASON} (game_type={GAME_TYPES})")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()