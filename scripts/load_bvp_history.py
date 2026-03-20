"""
load_bvp_history.py
-------------------
ONE-TIME backfill script. Run this manually once before the season starts.

What it does:
  - Pulls every probable starter from your game_probables table for 2024 + 2025 + 2026
  - For each SP, finds all batters who faced them in your lineups table
  - Fetches career Statcast data from Baseball Savant for each pair
  - Saves to batter_vs_pitcher table

After this runs, load_lineups.py will skip all cached pairs and run in seconds.

Expected run time: 30-90 minutes. Commits after each pitcher so it is safe
to interrupt and resume.

Usage:
  python scripts/load_bvp_history.py

  # Resume after interruption (use last pitcher_id printed):
  RESUME_AFTER_PITCHER_ID=434378 python scripts/load_bvp_history.py

  # Force re-fetch all pairs:
  BVP_FORCE_REFRESH=1 python scripts/load_bvp_history.py
"""
from __future__ import annotations

import math
import os
import time
from datetime import date
from typing import Optional

import pandas as pd
import psycopg2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)

PROB_TABLE   = os.getenv("MLB_PROBABLES_TABLE", "game_probables")
LINEUP_TABLE = "lineups"
BVP_TABLE    = "batter_vs_pitcher"

SLEEP            = float(os.getenv("REQUEST_SLEEP_SECONDS", "0.2"))
FORCE_REFRESH    = os.getenv("BVP_FORCE_REFRESH", "0") == "1"
RESUME_AFTER_PID = int(os.getenv("RESUME_AFTER_PITCHER_ID", "0"))

BACKFILL_START = os.getenv("BVP_START_DATE", "2024-03-01")
TODAY_STR      = date.today().isoformat()


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def conn():
    return psycopg2.connect(DATABASE_URL)


def ensure_tables(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {BVP_TABLE} (
            batter_id      BIGINT NOT NULL,
            pitcher_id     BIGINT NOT NULL,
            batter_name    TEXT,
            pitcher_name   TEXT,
            data_start     DATE,
            data_end       DATE,
            pa             INT    DEFAULT 0,
            ab             INT    DEFAULT 0,
            hits           INT    DEFAULT 0,
            doubles        INT    DEFAULT 0,
            triples        INT    DEFAULT 0,
            home_runs      INT    DEFAULT 0,
            strikeouts     INT    DEFAULT 0,
            walks          INT    DEFAULT 0,
            avg            DOUBLE PRECISION,
            obp            DOUBLE PRECISION,
            slg            DOUBLE PRECISION,
            hard_hit_pct   DOUBLE PRECISION,
            avg_exit_velo  DOUBLE PRECISION,
            k_pct          DOUBLE PRECISION,
            bb_pct         DOUBLE PRECISION,
            updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (batter_id, pitcher_id)
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_bvp_batter  ON {BVP_TABLE}(batter_id);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_bvp_pitcher ON {BVP_TABLE}(pitcher_id);")

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {LINEUP_TABLE} (
            game_pk        BIGINT NOT NULL,
            official_date  DATE   NOT NULL,
            team           TEXT   NOT NULL,
            side           TEXT   NOT NULL,
            batting_order  INT    NOT NULL,
            player_id      BIGINT NOT NULL,
            player_name    TEXT   NOT NULL,
            position       TEXT,
            updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (game_pk, side, batting_order)
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_lineups_date ON {LINEUP_TABLE}(official_date);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_lineups_game ON {LINEUP_TABLE}(game_pk);")


def bvp_already_cached(cur, batter_id: int, pitcher_id: int) -> bool:
    cur.execute(
        f"SELECT 1 FROM {BVP_TABLE} WHERE batter_id = %s AND pitcher_id = %s",
        (batter_id, pitcher_id),
    )
    return cur.fetchone() is not None


def upsert_bvp(cur, row: dict):
    cur.execute(f"""
        INSERT INTO {BVP_TABLE} (
            batter_id, pitcher_id, batter_name, pitcher_name,
            data_start, data_end,
            pa, ab, hits, doubles, triples, home_runs, strikeouts, walks,
            avg, obp, slg, hard_hit_pct, avg_exit_velo, k_pct, bb_pct, updated_at
        ) VALUES (
            %(batter_id)s, %(pitcher_id)s, %(batter_name)s, %(pitcher_name)s,
            %(data_start)s, %(data_end)s,
            %(pa)s, %(ab)s, %(hits)s, %(doubles)s, %(triples)s, %(home_runs)s,
            %(strikeouts)s, %(walks)s,
            %(avg)s, %(obp)s, %(slg)s, %(hard_hit_pct)s, %(avg_exit_velo)s,
            %(k_pct)s, %(bb_pct)s, NOW()
        )
        ON CONFLICT (batter_id, pitcher_id) DO UPDATE SET
            batter_name   = EXCLUDED.batter_name,
            pitcher_name  = EXCLUDED.pitcher_name,
            data_start    = EXCLUDED.data_start,
            data_end      = EXCLUDED.data_end,
            pa            = EXCLUDED.pa,
            ab            = EXCLUDED.ab,
            hits          = EXCLUDED.hits,
            doubles       = EXCLUDED.doubles,
            triples       = EXCLUDED.triples,
            home_runs     = EXCLUDED.home_runs,
            strikeouts    = EXCLUDED.strikeouts,
            walks         = EXCLUDED.walks,
            avg           = EXCLUDED.avg,
            obp           = EXCLUDED.obp,
            slg           = EXCLUDED.slg,
            hard_hit_pct  = EXCLUDED.hard_hit_pct,
            avg_exit_velo = EXCLUDED.avg_exit_velo,
            k_pct         = EXCLUDED.k_pct,
            bb_pct        = EXCLUDED.bb_pct,
            updated_at    = NOW();
    """, row)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val, default=None):
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def get_pitchers_from_probables(cur) -> list:
    cur.execute(f"""
        SELECT DISTINCT pitcher_id, pitcher_name FROM (
            SELECT home_sp_id AS pitcher_id, home_sp_name AS pitcher_name
            FROM {PROB_TABLE}
            WHERE home_sp_id IS NOT NULL AND official_date >= '2024-01-01'
            UNION
            SELECT away_sp_id AS pitcher_id, away_sp_name AS pitcher_name
            FROM {PROB_TABLE}
            WHERE away_sp_id IS NOT NULL AND official_date >= '2024-01-01'
        ) sub
        ORDER BY pitcher_id
    """)
    return [{"pitcher_id": int(r[0]), "pitcher_name": r[1] or ""} for r in cur.fetchall()]


def get_batters_for_pitcher(cur, pitcher_id: int) -> list:
    cur.execute(f"""
        SELECT DISTINCT l.player_id, l.player_name
        FROM {LINEUP_TABLE} l
        JOIN {PROB_TABLE} p ON p.game_pk = l.game_pk
        WHERE (p.home_sp_id = %s AND l.side = 'away')
           OR (p.away_sp_id = %s AND l.side = 'home')
        ORDER BY l.player_name
    """, (pitcher_id, pitcher_id))
    rows = cur.fetchall()

    if rows:
        return [{"player_id": int(r[0]), "player_name": r[1] or ""} for r in rows]

    # Fallback: all batters in lineups table
    cur.execute(f"SELECT DISTINCT player_id, player_name FROM {LINEUP_TABLE} ORDER BY player_name")
    return [{"player_id": int(r[0]), "player_name": r[1] or ""} for r in cur.fetchall()]


def fetch_bvp_statcast(batter_id: int, batter_name: str,
                       pitcher_id: int, pitcher_name: str) -> Optional[dict]:
    try:
        from pybaseball import statcast_batter
        df = statcast_batter(BACKFILL_START, TODAY_STR, player_id=batter_id)
    except Exception as e:
        print(f"      statcast failed for {batter_name}: {e}")
        return None

    if df is None or df.empty:
        return None

    if "pitcher" in df.columns:
        df = df[df["pitcher"] == pitcher_id].copy()
    if df.empty:
        return None

    events = df["events"].dropna() if "events" in df.columns else pd.Series()

    pa = int(events.isin([
        "single", "double", "triple", "home_run",
        "strikeout", "strikeout_double_play",
        "walk", "intent_walk", "hit_by_pitch",
        "field_out", "grounded_into_double_play", "force_out",
        "field_error", "fielders_choice", "fielders_choice_out",
        "double_play", "triple_play", "sac_fly", "sac_bunt",
    ]).sum())

    if pa == 0:
        return None

    ab         = int(events.isin(["single","double","triple","home_run","strikeout","strikeout_double_play","field_out","grounded_into_double_play","force_out","field_error","fielders_choice","fielders_choice_out","double_play","triple_play"]).sum())
    hits       = int(events.isin(["single","double","triple","home_run"]).sum())
    doubles    = int((events == "double").sum())
    triples    = int((events == "triple").sum())
    home_runs  = int((events == "home_run").sum())
    strikeouts = int(events.isin(["strikeout","strikeout_double_play"]).sum())
    walks      = int(events.isin(["walk","intent_walk"]).sum())
    tb         = hits + doubles + (2 * triples) + (3 * home_runs)

    hard_hit_pct = avg_exit_velo = None
    if "launch_speed" in df.columns:
        ev = pd.to_numeric(df["launch_speed"], errors="coerce").dropna()
        if len(ev) > 0:
            avg_exit_velo = _safe(round(float(ev.mean()), 1))
            hard_hit_pct  = _safe(round(float((ev >= 95).sum() / len(ev)), 3))

    return {
        "batter_id":     batter_id,      "pitcher_id":    pitcher_id,
        "batter_name":   batter_name,    "pitcher_name":  pitcher_name,
        "data_start":    BACKFILL_START, "data_end":      TODAY_STR,
        "pa": pa, "ab": ab, "hits": hits, "doubles": doubles,
        "triples": triples, "home_runs": home_runs,
        "strikeouts": strikeouts, "walks": walks,
        "avg":  _safe(round(hits / ab, 3)           if ab > 0 else None),
        "obp":  _safe(round((hits + walks) / pa, 3) if pa > 0 else None),
        "slg":  _safe(round(tb / ab, 3)             if ab > 0 else None),
        "hard_hit_pct":  hard_hit_pct,
        "avg_exit_velo": avg_exit_velo,
        "k_pct":  _safe(round(strikeouts / pa, 3) if pa > 0 else None),
        "bb_pct": _safe(round(walks / pa, 3)      if pa > 0 else None),
    }


def empty_bvp_row(batter_id, batter_name, pitcher_id, pitcher_name) -> dict:
    return {
        "batter_id": batter_id, "pitcher_id": pitcher_id,
        "batter_name": batter_name, "pitcher_name": pitcher_name,
        "data_start": BACKFILL_START, "data_end": TODAY_STR,
        "pa": 0, "ab": 0, "hits": 0, "doubles": 0, "triples": 0,
        "home_runs": 0, "strikeouts": 0, "walks": 0,
        "avg": None, "obp": None, "slg": None,
        "hard_hit_pct": None, "avg_exit_velo": None,
        "k_pct": None, "bb_pct": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"{'='*60}")
    print(f"  load_bvp_history.py — ONE-TIME BACKFILL")
    print(f"  Data range    : {BACKFILL_START} -> {TODAY_STR}")
    print(f"  Force refresh : {FORCE_REFRESH}")
    if RESUME_AFTER_PID:
        print(f"  Resuming after pitcher_id > {RESUME_AFTER_PID}")
    print(f"{'='*60}\n")

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:

            ensure_tables(cur)
            c.commit()
            print("Tables ready.\n")

            pitchers = get_pitchers_from_probables(cur)
            if not pitchers:
                print("No pitchers found in game_probables.")
                print("Run load_probable_starters.py first, then retry.")
                return

            if RESUME_AFTER_PID:
                pitchers = [p for p in pitchers if p["pitcher_id"] > RESUME_AFTER_PID]

            print(f"Found {len(pitchers)} unique SPs to process\n")

            total_fetched = total_cached = total_empty = 0

            for i, pitcher in enumerate(pitchers, start=1):
                pid   = pitcher["pitcher_id"]
                pname = pitcher["pitcher_name"]

                print(f"[{i}/{len(pitchers)}] {pname} (id={pid})")

                batters = get_batters_for_pitcher(cur, pid)
                if not batters:
                    print(f"  No batters found, skipping\n")
                    continue

                print(f"  {len(batters)} batters to check")

                for batter in batters:
                    bid   = batter["player_id"]
                    bname = batter["player_name"]

                    if not FORCE_REFRESH and bvp_already_cached(cur, bid, pid):
                        total_cached += 1
                        continue

                    bvp = fetch_bvp_statcast(bid, bname, pid, pname)
                    if bvp:
                        upsert_bvp(cur, bvp)
                        total_fetched += 1
                        print(f"    {bname}: {bvp['pa']} PA | AVG {bvp['avg']} | K% {bvp['k_pct']}")
                    else:
                        upsert_bvp(cur, empty_bvp_row(bid, bname, pid, pname))
                        total_empty += 1

                    if SLEEP:
                        time.sleep(SLEEP)

                c.commit()
                print(f"  Done. [{total_fetched} fetched | {total_cached} cached | {total_empty} empty so far]\n")

        print(f"{'='*60}")
        print(f"  Backfill complete!")
        print(f"  Pairs with data  : {total_fetched}")
        print(f"  Pairs cached     : {total_cached}")
        print(f"  Pairs no history : {total_empty}")
        print(f"{'='*60}")
        print("\nYou can now run load_lineups.py daily — it will be fast.")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()