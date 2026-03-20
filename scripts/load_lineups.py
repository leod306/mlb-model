"""
load_lineups.py
---------------
DAILY script. Run this every day as part of mlb_quick_update.py.

What it does:
  1. Fetches today's batting lineups from the MLB Stats API live feed
  2. For each batter vs opposing SP, checks if BvP data already exists in DB
     - If cached → skips (fast)
     - If new matchup → fetches from Baseball Savant and saves
  3. Stores lineups and any new BvP pairs in Postgres

Run time:
  - First time for a given matchup : ~1-2 min per game (fetches new pairs)
  - After initial backfill          : ~5-10 seconds total (all cached)

To force re-fetch all pairs for today:
  BVP_FORCE_REFRESH=1 python scripts/load_lineups.py

Add to mlb_quick_update.py:
  run_step("Load Lineups", "scripts/load_lineups.py")
"""
from __future__ import annotations

import math
import os
import time
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)

GAMES_TABLE  = os.getenv("MLB_GAMES_TABLE",    "games")
PROB_TABLE   = os.getenv("MLB_PROBABLES_TABLE", "game_probables")
LINEUP_TABLE = "lineups"
BVP_TABLE    = "batter_vs_pitcher"

HTTP_TIMEOUT  = 20
SLEEP         = float(os.getenv("REQUEST_SLEEP_SECONDS", "0.15"))
FORCE_REFRESH = os.getenv("BVP_FORCE_REFRESH", "0") == "1"

MLB_FEED      = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"

# Must match what load_bvp_history.py used so stats are comparable
BACKFILL_START = os.getenv("BVP_START_DATE", "2024-03-01")
TODAY_STR      = date.today().isoformat()


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def conn():
    return psycopg2.connect(DATABASE_URL)


def ensure_tables(cur):
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


# ---------------------------------------------------------------------------
# Cache check
# ---------------------------------------------------------------------------

def bvp_already_cached(cur, batter_id: int, pitcher_id: int) -> bool:
    cur.execute(
        f"SELECT 1 FROM {BVP_TABLE} WHERE batter_id = %s AND pitcher_id = %s",
        (batter_id, pitcher_id),
    )
    return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# MLB API
# ---------------------------------------------------------------------------

def get_json(url: str) -> dict:
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_lineup_from_feed(game_pk: int) -> dict:
    try:
        data = get_json(MLB_FEED.format(gamePk=game_pk))
    except Exception as e:
        print(f"  Could not fetch feed for game {game_pk}: {e}")
        return {"home": [], "away": []}

    live_data = data.get("liveData") or {}
    boxscore  = live_data.get("boxscore") or {}
    teams     = boxscore.get("teams") or {}

    result = {"home": [], "away": []}
    for side in ("home", "away"):
        team_data = teams.get(side) or {}
        batters   = team_data.get("battingOrder") or []
        players   = team_data.get("players") or {}

        for order, pid in enumerate(batters, start=1):
            key    = f"ID{pid}"
            player = players.get(key) or {}
            person = player.get("person") or {}
            pos    = (player.get("position") or {}).get("abbreviation") or ""
            result[side].append({
                "batting_order": order,
                "player_id":     int(pid),
                "player_name":   person.get("fullName") or f"Player {pid}",
                "position":      pos,
            })

    return result


def upsert_lineups(cur, game_pk: int, off_date, team_home: str, team_away: str, lineup: dict) -> int:
    rows = []
    now  = datetime.now(timezone.utc)
    side_team = {"home": team_home, "away": team_away}

    for side, batters in lineup.items():
        for b in batters:
            rows.append((
                game_pk, off_date, side_team[side], side,
                b["batting_order"], b["player_id"], b["player_name"],
                b["position"], now,
            ))

    if not rows:
        return 0

    execute_values(cur, f"""
        INSERT INTO {LINEUP_TABLE}
          (game_pk, official_date, team, side, batting_order, player_id, player_name, position, updated_at)
        VALUES %s
        ON CONFLICT (game_pk, side, batting_order) DO UPDATE SET
          player_id   = EXCLUDED.player_id,
          player_name = EXCLUDED.player_name,
          position    = EXCLUDED.position,
          updated_at  = NOW();
    """, rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Statcast fetch (single batter vs single pitcher)
# ---------------------------------------------------------------------------

def _safe(val, default=None):
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def fetch_bvp_statcast(
    batter_id: int, batter_name: str,
    pitcher_id: int, pitcher_name: str,
) -> Optional[dict]:
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
        "single","double","triple","home_run",
        "strikeout","strikeout_double_play",
        "walk","intent_walk","hit_by_pitch",
        "field_out","grounded_into_double_play","force_out",
        "field_error","fielders_choice","fielders_choice_out",
        "double_play","triple_play","sac_fly","sac_bunt",
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
        "batter_id":     batter_id,    "pitcher_id":   pitcher_id,
        "batter_name":   batter_name,  "pitcher_name": pitcher_name,
        "data_start":    BACKFILL_START, "data_end":   TODAY_STR,
        "pa":  pa,  "ab": ab, "hits": hits, "doubles": doubles,
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
            batter_name=EXCLUDED.batter_name, pitcher_name=EXCLUDED.pitcher_name,
            data_start=EXCLUDED.data_start,   data_end=EXCLUDED.data_end,
            pa=EXCLUDED.pa, ab=EXCLUDED.ab, hits=EXCLUDED.hits,
            doubles=EXCLUDED.doubles, triples=EXCLUDED.triples,
            home_runs=EXCLUDED.home_runs, strikeouts=EXCLUDED.strikeouts,
            walks=EXCLUDED.walks, avg=EXCLUDED.avg, obp=EXCLUDED.obp,
            slg=EXCLUDED.slg, hard_hit_pct=EXCLUDED.hard_hit_pct,
            avg_exit_velo=EXCLUDED.avg_exit_velo,
            k_pct=EXCLUDED.k_pct, bb_pct=EXCLUDED.bb_pct,
            updated_at=NOW();
    """, row)


def empty_bvp_row(batter_id, batter_name, pitcher_id, pitcher_name) -> dict:
    """Saves a zero-row so we never re-fetch a pair with no history."""
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
# Process one side
# ---------------------------------------------------------------------------

def process_side(cur, batters, sp_id, sp_name, label) -> tuple[int, int]:
    """Returns (fetched, cached) counts."""
    if not sp_id or not batters:
        print(f"  {label}: no SP set, skipping")
        return 0, 0

    print(f"  {label}: {sp_name} (id={sp_id})")
    fetched = cached = 0

    for batter in batters:
        bid  = batter["player_id"]
        name = batter["player_name"]

        if not FORCE_REFRESH and bvp_already_cached(cur, bid, int(sp_id)):
            print(f"    {name}: cached ✓")
            cached += 1
            continue

        bvp = fetch_bvp_statcast(bid, name, int(sp_id), sp_name or "")
        if bvp:
            upsert_bvp(cur, bvp)
            fetched += 1
            print(f"    {name}: {bvp['pa']} PA | AVG {bvp['avg']} | K% {bvp['k_pct']} ← saved")
        else:
            upsert_bvp(cur, empty_bvp_row(bid, name, int(sp_id), sp_name or ""))
            print(f"    {name}: no history — saved empty row")

        if SLEEP:
            time.sleep(SLEEP)

    return fetched, cached


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    target = date.today()
    print(f"{'='*55}")
    print(f"  load_lineups.py  |  {target}")
    print(f"  BvP range : {BACKFILL_START} -> {TODAY_STR}")
    print(f"  Force refresh : {FORCE_REFRESH}")
    print(f"{'='*55}\n")

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_tables(cur)

            cur.execute(f"""
                SELECT g.game_pk, g.official_date, g.home_team, g.away_team,
                       p.home_sp_id, p.home_sp_name, p.away_sp_id, p.away_sp_name
                FROM {GAMES_TABLE} g
                LEFT JOIN {PROB_TABLE} p ON p.game_pk = g.game_pk
                WHERE g.official_date = %s
                ORDER BY g.game_pk
            """, (target,))
            games = cur.fetchall()

            if not games:
                print(f"No games found for {target} — tables created, nothing to fetch.")
                c.commit()
                return

            print(f"Found {len(games)} games\n")
            total_lineup = total_fetched = total_cached = 0

            for game_pk, off_date, home_team, away_team, home_sp_id, home_sp_name, away_sp_id, away_sp_name in games:
                print(f"--- {away_team} @ {home_team} (pk={game_pk}) ---")

                lineup = fetch_lineup_from_feed(int(game_pk))
                n = upsert_lineups(cur, int(game_pk), off_date, home_team, away_team, lineup)
                total_lineup += n
                print(f"  Lineup: {len(lineup['away'])} away | {len(lineup['home'])} home batters")

                if SLEEP:
                    time.sleep(SLEEP)

                f, ca = process_side(cur, lineup["away"], home_sp_id, home_sp_name,
                                     f"Away vs {home_sp_name or 'TBD'}")
                total_fetched += f
                total_cached  += ca

                f, ca = process_side(cur, lineup["home"], away_sp_id, away_sp_name,
                                     f"Home vs {away_sp_name or 'TBD'}")
                total_fetched += f
                total_cached  += ca
                print()

            c.commit()

        print(f"{'='*55}")
        print(f"  Lineup rows upserted       : {total_lineup}")
        print(f"  BvP pairs cached (skipped) : {total_cached}")
        print(f"  BvP pairs newly fetched    : {total_fetched}")
        print(f"{'='*55}")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()