from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import requests
import psycopg2
from psycopg2.extras import execute_values

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)

GAMES_TABLE = os.getenv("MLB_GAMES_TABLE", "games")
PROB_TABLE = os.getenv("MLB_PROBABLES_TABLE", "game_probables")

LOOKBACK_DAYS = int(os.getenv("PROB_LOOKBACK_DAYS", "1"))
LOOKAHEAD_DAYS = int(os.getenv("PROB_LOOKAHEAD_DAYS", "14"))
SLEEP = float(os.getenv("REQUEST_SLEEP_SECONDS", "0.05"))
HTTP_TIMEOUT = 20

MLB_FEED = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"


def conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)


def get_json(url: str) -> dict:
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def resolve_window() -> Tuple[date, date]:
    start_env = os.getenv("START_DATE")
    end_env = os.getenv("END_DATE")
    if start_env and end_env:
        return date.fromisoformat(start_env), date.fromisoformat(end_env)

    today = datetime.now(timezone.utc).date()
    return today - timedelta(days=LOOKBACK_DAYS), today + timedelta(days=LOOKAHEAD_DAYS)


def ensure_probables(cur):
    cur.execute(f"CREATE TABLE IF NOT EXISTS {PROB_TABLE} (game_pk BIGINT PRIMARY KEY);")
    cur.execute(f"ALTER TABLE {PROB_TABLE} ADD COLUMN IF NOT EXISTS official_date DATE;")
    cur.execute(f"ALTER TABLE {PROB_TABLE} ADD COLUMN IF NOT EXISTS home_sp_id INT;")
    cur.execute(f"ALTER TABLE {PROB_TABLE} ADD COLUMN IF NOT EXISTS away_sp_id INT;")
    cur.execute(f"ALTER TABLE {PROB_TABLE} ADD COLUMN IF NOT EXISTS home_sp_name TEXT;")
    cur.execute(f"ALTER TABLE {PROB_TABLE} ADD COLUMN IF NOT EXISTS away_sp_name TEXT;")
    cur.execute(f"ALTER TABLE {PROB_TABLE} ADD COLUMN IF NOT EXISTS status TEXT;")
    cur.execute(f"ALTER TABLE {PROB_TABLE} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{PROB_TABLE}_official_date ON {PROB_TABLE}(official_date);")


def parse_probables(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str], Optional[str]]:
    gd = payload.get("gameData") or {}
    prob = gd.get("probablePitchers") or {}
    players = gd.get("players") or {}
    status = (gd.get("status") or {}).get("detailedState") or None

    def one(side: str) -> Tuple[Optional[int], Optional[str]]:
        p = prob.get(side) or {}
        pid = p.get("id")
        if not pid:
            return None, None
        pid = int(pid)
        pl = players.get(f"ID{pid}") or {}
        name = pl.get("fullName") or p.get("fullName")
        return pid, name

    hid, hname = one("home")
    aid, aname = one("away")
    return hid, hname, aid, aname, status


def main():
    start, end = resolve_window()

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_probables(cur)

            cur.execute(
                f"""
                SELECT game_pk, official_date
                FROM {GAMES_TABLE}
                WHERE official_date BETWEEN %s AND %s
                ORDER BY official_date, game_pk
                """,
                (start, end),
            )
            games = cur.fetchall()

            rows = []
            now = datetime.now(timezone.utc)

            for game_pk, off_date in games:
                try:
                    payload = get_json(MLB_FEED.format(gamePk=int(game_pk)))
                except Exception:
                    continue

                hid, hname, aid, aname, status = parse_probables(payload)
                rows.append((int(game_pk), off_date, hid, aid, hname, aname, status, now))

                if SLEEP:
                    time.sleep(SLEEP)

            if rows:
                sql = f"""
                INSERT INTO {PROB_TABLE} (
                  game_pk, official_date,
                  home_sp_id, away_sp_id,
                  home_sp_name, away_sp_name,
                  status, updated_at
                )
                VALUES %s
                ON CONFLICT (game_pk) DO UPDATE SET
                  official_date=EXCLUDED.official_date,
                  home_sp_id=EXCLUDED.home_sp_id,
                  away_sp_id=EXCLUDED.away_sp_id,
                  home_sp_name=EXCLUDED.home_sp_name,
                  away_sp_name=EXCLUDED.away_sp_name,
                  status=EXCLUDED.status,
                  updated_at=NOW();
                """
                execute_values(cur, sql, rows, page_size=1000)

            c.commit()
            print(f"Upserted {len(rows)} probable starter rows into {PROB_TABLE} ({start} -> {end})")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()