"""
load_statcast_batting.py
------------------------
Downloads season Statcast quality-of-contact metrics from Baseball Savant
via pybaseball. One call per metric type — returns all qualifying batters
at once (fast, ~5 seconds total).

Stores in statcast_batting table used by load_player_props.py to apply a
quality-of-contact luck factor:
  xwOBA > wOBA  →  batter is UNLUCKY (good contact, bad results) → boost proj
  xwOBA < wOBA  →  batter is LUCKY   (poor contact, good results) → suppress proj

Modeled on load_bvp_history.py — same psycopg2 pattern, same pybaseball usage.

Install pybaseball:
  pip install pybaseball

Run weekly (Statcast data updates daily but changes slowly):
  python scripts/load_statcast_batting.py           # current season
  python scripts/load_statcast_batting.py --year 2025

Heroku: add to Scheduler weekly or run manually as needed.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date, datetime, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)
    except Exception:
        pass

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)
MLB_SEASON   = int(os.getenv("MLB_SEASON", "2026"))
TABLE        = "statcast_batting"


def log(msg): print(msg, flush=True)


def _sf(val) -> Optional[float]:
    """Safe float — returns None on NaN/inf/non-numeric."""
    try:
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(DATABASE_URL)


def ensure_table(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            player_id       INTEGER PRIMARY KEY,
            player_name     TEXT,
            season          INTEGER,
            pa              INTEGER,
            -- Actual vs expected batting outcomes
            ba              DOUBLE PRECISION,
            xba             DOUBLE PRECISION,   -- xBA  (luck-neutral batting avg)
            xba_diff        DOUBLE PRECISION,   -- xBA - BA  (positive = unlucky)
            obp             DOUBLE PRECISION,
            xobp            DOUBLE PRECISION,
            slg             DOUBLE PRECISION,
            xslg            DOUBLE PRECISION,   -- xSLG (luck-neutral slugging)
            woba            DOUBLE PRECISION,
            xwoba           DOUBLE PRECISION,   -- xwOBA (best single luck metric)
            xwoba_diff      DOUBLE PRECISION,   -- xwOBA - wOBA (positive = unlucky)
            -- Quality of contact
            barrel_pct      DOUBLE PRECISION,   -- % batted balls that are barrels
            avg_exit_velo   DOUBLE PRECISION,   -- avg launch speed mph
            hard_hit_pct    DOUBLE PRECISION,   -- % EV >= 95 mph
            sweet_spot_pct  DOUBLE PRECISION,   -- % launch angle 8-32 degrees
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_sb_season ON {TABLE}(season);")


# ---------------------------------------------------------------------------
# pybaseball fetches
# ---------------------------------------------------------------------------

def fetch_expected_stats(year: int):
    """
    xBA, xOBP, xSLG, xwOBA for all qualifying batters.
    pybaseball.statcast_batter_expected_stats(year, minPA=25)
    Returns DataFrame or None.
    """
    try:
        from pybaseball import statcast_batter_expected_stats
        log(f"  Fetching xwOBA / xBA / xSLG from Baseball Savant (year={year}, minPA=25)...")
        df = statcast_batter_expected_stats(year, minPA=25)
        log(f"    Got {len(df)} rows  |  columns: {list(df.columns)[:8]}...")
        return df
    except ImportError:
        log("  pybaseball not installed — run:  pip install pybaseball")
        return None
    except Exception as e:
        log(f"  Expected stats fetch error: {e}")
        return None


def fetch_exitvelo_barrels(year: int):
    """
    Barrel%, avg exit velocity, hard hit% for all qualifying batters.
    pybaseball.statcast_batter_exitvelo_barrels(year, minBBE=25)
    Returns DataFrame or None.
    """
    try:
        from pybaseball import statcast_batter_exitvelo_barrels
        log(f"  Fetching barrel / exit-velo from Baseball Savant (year={year}, minBBE=25)...")
        df = statcast_batter_exitvelo_barrels(year, minBBE=25)
        log(f"    Got {len(df)} rows  |  columns: {list(df.columns)[:8]}...")
        return df
    except ImportError:
        return None
    except Exception as e:
        log(f"  Exit velo fetch error: {e}")
        return None


# ---------------------------------------------------------------------------
# Normalize column names (pybaseball column names vary slightly by version)
# ---------------------------------------------------------------------------

# Expected stats column aliases
EXP_COL_MAP = {
    # player identity
    "player_id":                  "player_id",
    "mlbam_id":                   "player_id",   # older pybaseball
    "last_name":                  "last_name",
    "first_name":                 "first_name",
    "pa":                         "pa",
    # actual
    "ba":                         "ba",
    "avg":                        "ba",
    "obp":                        "obp",
    "slg":                        "slg",
    "woba":                       "woba",
    # expected
    "est_ba":                     "xba",
    "est_ba_minus_ba_diff":       "xba_diff",
    "est_obp":                    "xobp",
    "est_slg":                    "xslg",
    "est_woba":                   "xwoba",
    "est_woba_minus_woba_diff":   "xwoba_diff",
    # Sometimes the diff is stored as woba - xwoba (sign flipped) — normalised below
}

# Exit velo / barrel column aliases
EV_COL_MAP = {
    "player_id":              "player_id",
    "mlbam_id":               "player_id",
    "avg_hit_speed":          "avg_exit_velo",
    "avg_exit_velocity":      "avg_exit_velo",
    "ev95percent":            "hard_hit_pct",
    "hard_hit_percent":       "hard_hit_pct",
    "brl_percent":            "barrel_pct",
    "barrel_batted_rate":     "barrel_pct",
    "anglesweetspotpercent":  "sweet_spot_pct",
    "sweet_spot_percent":     "sweet_spot_pct",
}


def _rename(df, col_map: dict):
    """Rename df columns using col_map, keeping only mapped columns."""
    import pandas as pd
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df2 = df.rename(columns=rename)
    keep = [v for v in col_map.values() if v in df2.columns]
    # deduplicate keep list (multiple source cols can map to same target)
    seen, unique_keep = set(), []
    for c in keep:
        if c not in seen:
            seen.add(c)
            unique_keep.append(c)
    return df2[unique_keep]


# ---------------------------------------------------------------------------
# Merge & upsert
# ---------------------------------------------------------------------------

def build_rows(exp_df, ev_df, year: int) -> list[dict]:
    import pandas as pd

    base = None

    if exp_df is not None and not exp_df.empty:
        # pybaseball may return "last_name, first_name" as a single combined column
        # e.g. "Trout, Mike" → split to first_name="Mike", last_name="Trout"
        if "last_name, first_name" in exp_df.columns:
            split = exp_df["last_name, first_name"].str.split(", ", n=1, expand=True)
            exp_df = exp_df.copy()
            exp_df["last_name"]  = split[0].str.strip()
            exp_df["first_name"] = split[1].str.strip() if split.shape[1] > 1 else ""
        exp_clean = _rename(exp_df, EXP_COL_MAP)
        # Build player_name: "First Last"
        if "first_name" in exp_clean.columns and "last_name" in exp_clean.columns:
            exp_clean["player_name"] = (exp_clean["first_name"].fillna("").astype(str).str.strip()
                                        + " "
                                        + exp_clean["last_name"].fillna("").astype(str).str.strip())
            exp_clean = exp_clean.drop(columns=["first_name", "last_name"], errors="ignore")
        base = exp_clean

    if ev_df is not None and not ev_df.empty:
        ev_clean = _rename(ev_df, EV_COL_MAP)
        if base is not None and "player_id" in base.columns and "player_id" in ev_clean.columns:
            base = base.merge(ev_clean, on="player_id", how="left")
        elif base is None:
            base = ev_clean

    if base is None or base.empty:
        return []

    rows = []
    for _, r in base.iterrows():
        pid = r.get("player_id")
        if pid is None:
            continue
        try:
            pid = int(pid)
        except (ValueError, TypeError):
            continue

        # Normalise xwoba_diff sign:
        # pybaseball sometimes returns woba - xwoba (negative = unlucky)
        # We want: positive = unlucky = boost
        xwoba_diff = _sf(r.get("xwoba_diff"))
        xwoba      = _sf(r.get("xwoba"))
        woba       = _sf(r.get("woba"))
        if xwoba_diff is None and xwoba is not None and woba is not None:
            xwoba_diff = xwoba - woba
        elif xwoba_diff is not None and xwoba is not None and woba is not None:
            # If sign seems flipped (diff matches woba - xwoba), correct it
            if abs(xwoba_diff - (woba - xwoba)) < abs(xwoba_diff - (xwoba - woba)):
                xwoba_diff = xwoba - woba   # normalise to xwoba - woba

        xba_diff = _sf(r.get("xba_diff"))
        xba      = _sf(r.get("xba"))
        ba       = _sf(r.get("ba"))
        if xba_diff is None and xba is not None and ba is not None:
            xba_diff = xba - ba

        rows.append({
            "player_id":      pid,
            "player_name":    str(r.get("player_name", "")).strip(),
            "season":         year,
            "pa":             int(_sf(r.get("pa")) or 0),
            "ba":             _sf(r.get("ba")),
            "xba":            xba,
            "xba_diff":       xba_diff,
            "obp":            _sf(r.get("obp")),
            "xobp":           _sf(r.get("xobp")),
            "slg":            _sf(r.get("slg")),
            "xslg":           _sf(r.get("xslg")),
            "woba":           woba,
            "xwoba":          xwoba,
            "xwoba_diff":     xwoba_diff,
            "barrel_pct":     _sf(r.get("barrel_pct")),
            "avg_exit_velo":  _sf(r.get("avg_exit_velo")),
            "hard_hit_pct":   _sf(r.get("hard_hit_pct")),
            "sweet_spot_pct": _sf(r.get("sweet_spot_pct")),
        })

    return rows


def upsert_rows(cur, rows: list[dict]) -> int:
    if not rows:
        return 0
    data = [
        (
            r["player_id"], r["player_name"], r["season"], r["pa"],
            r["ba"], r["xba"], r["xba_diff"],
            r["obp"], r["xobp"], r["slg"], r["xslg"],
            r["woba"], r["xwoba"], r["xwoba_diff"],
            r["barrel_pct"], r["avg_exit_velo"], r["hard_hit_pct"], r["sweet_spot_pct"],
            datetime.now(timezone.utc),
        )
        for r in rows
    ]
    sql = f"""
    INSERT INTO {TABLE} (
        player_id, player_name, season, pa,
        ba, xba, xba_diff, obp, xobp, slg, xslg,
        woba, xwoba, xwoba_diff,
        barrel_pct, avg_exit_velo, hard_hit_pct, sweet_spot_pct,
        updated_at
    ) VALUES %s
    ON CONFLICT (player_id) DO UPDATE SET
        player_name     = EXCLUDED.player_name,
        season          = EXCLUDED.season,
        pa              = EXCLUDED.pa,
        ba              = EXCLUDED.ba,
        xba             = EXCLUDED.xba,
        xba_diff        = EXCLUDED.xba_diff,
        obp             = EXCLUDED.obp,
        xobp            = EXCLUDED.xobp,
        slg             = EXCLUDED.slg,
        xslg            = EXCLUDED.xslg,
        woba            = EXCLUDED.woba,
        xwoba           = EXCLUDED.xwoba,
        xwoba_diff      = EXCLUDED.xwoba_diff,
        barrel_pct      = EXCLUDED.barrel_pct,
        avg_exit_velo   = EXCLUDED.avg_exit_velo,
        hard_hit_pct    = EXCLUDED.hard_hit_pct,
        sweet_spot_pct  = EXCLUDED.sweet_spot_pct,
        updated_at      = NOW();
    """
    execute_values(cur, sql, data, page_size=500)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load Statcast batting metrics from Baseball Savant.")
    parser.add_argument("--year", type=int, default=MLB_SEASON,
                        help=f"Season year (default: {MLB_SEASON})")
    args = parser.parse_args()

    log("=" * 60)
    log(f"load_statcast_batting.py | year={args.year} | {date.today()}")
    log("=" * 60)

    c = get_conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_table(cur)
            c.commit()
            log("Table ready.\n")

        exp_df = fetch_expected_stats(args.year)
        ev_df  = fetch_exitvelo_barrels(args.year)

        rows = build_rows(exp_df, ev_df, args.year)
        log(f"\n  Combined: {len(rows)} players to upsert")

        if rows:
            with c.cursor() as cur:
                n = upsert_rows(cur, rows)
            c.commit()
            log(f"  Upserted {n} rows into {TABLE}")
        else:
            log("  No rows — check pybaseball install or Baseball Savant availability.")

        log(f"\n{'='*60}")
        log(f"Done. {len(rows)} players in {TABLE} for {args.year}.")
        log(f"{'='*60}")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()
