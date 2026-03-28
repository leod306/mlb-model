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

SEASON = int(os.getenv("MLB_SEASON", "2026"))
GAME_TYPES = os.getenv("MLB_GAME_TYPES", "R")
GAMES_TABLE = os.getenv("MLB_GAMES_TABLE", "games")

MLB_BASE = "https://statsapi.mlb.com/api/v1"
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
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS official_date DATE;")
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS game_date_utc TIMESTAMPTZ;")
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS season INT;")
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS game_type TEXT;")
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS status TEXT;")
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS home_team_id INT;")
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS away_team_id INT;")
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS home_team TEXT;")
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS away_team TEXT;")
    cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{GAMES_TABLE}_official_date ON {GAMES_TABLE}(official_date);")


def team_map(season: int) -> Dict[int, str]:
    data = get_json(f"{MLB_BASE}/teams", {"sportId": 1, "season": season})
    out: Dict[int, str] = {}
    for t in data.get("teams", []) or []:
        tid = t.get("id")
        abbr = t.get("abbreviation") or t.get("teamCode") or t.get("fileCode") or t.get("name")
        if tid and abbr:
            out[int(tid)] = str(abbr)
    return out


def main():
    tmap = team_map(SEASON)
    data = get_json(
        f"{MLB_BASE}/schedule",
        {"sportId": 1, "season": SEASON, "gameTypes": GAME_TYPES, "hydrate": "team"},
    )

    rows = []
    now = datetime.now(timezone.utc)

    for d in data.get("dates", []) or []:
        for g in d.get("games", []) or []:
            home = (((g.get("teams") or {}).get("home") or {}).get("team") or {})
            away = (((g.get("teams") or {}).get("away") or {}).get("team") or {})
            if not home.get("id") or not away.get("id"):
                continue

            rows.append(
                (
                    int(g["gamePk"]),
                    (g.get("officialDate") or d.get("date")),
                    utc_dt(g["gameDate"]),
                    SEASON,
                    str(g.get("gameType") or ""),
                    str((g.get("status") or {}).get("detailedState") or ""),
                    int(home["id"]),
                    int(away["id"]),
                    str(home.get("abbreviation") or tmap.get(int(home["id"])) or home.get("name") or home["id"]),
                    str(away.get("abbreviation") or tmap.get(int(away["id"])) or away.get("name") or away["id"]),
                    now,
                )
            )

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_games(cur)

            if rows:
                sql = f"""
                INSERT INTO {GAMES_TABLE} (
                  game_pk, official_date, game_date_utc, season, game_type, status,
                  home_team_id, away_team_id, home_team, away_team, updated_at
                )
                VALUES %s
                ON CONFLICT (game_pk) DO UPDATE SET
                  official_date=EXCLUDED.official_date,
                  game_date_utc=EXCLUDED.game_date_utc,
                  season=EXCLUDED.season,
                  game_type=EXCLUDED.game_type,
                  status=EXCLUDED.status,
                  home_team_id=EXCLUDED.home_team_id,
                  away_team_id=EXCLUDED.away_team_id,
                  home_team=EXCLUDED.home_team,
                  away_team=EXCLUDED.away_team,
                  updated_at=NOW();
                """
                execute_values(cur, sql, rows, page_size=1000)

            c.commit()

        print(f"Upserted {len(rows)} games for season={SEASON} into {GAMES_TABLE}")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()