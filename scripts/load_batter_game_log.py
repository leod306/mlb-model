"""
load_batter_game_log.py
-----------------------
Fetches per-game hitting stats from the MLB Stats API for active batters.
Stores in batter_game_log table which is used by load_player_props.py to
compute a 14-day recent form factor (hot/cold streak adjustment).

Modeled on load_pitcher_game_log.py — same psycopg2 pattern, same API base.

Run order (in run_full_update.py):
  After  load_lineups.py     (needs player IDs from lineups)
  Before load_player_props.py (props model reads this table)

Usage:
  python scripts/load_batter_game_log.py             # today's lineup players only (fast, daily)
  python scripts/load_batter_game_log.py --mode all  # all season batters (slower, for backfill)
  python scripts/load_batter_game_log.py --days 30   # extend lookback window
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import execute_values

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)
    except Exception:
        pass

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)
GAMES_TABLE  = os.getenv("MLB_GAMES_TABLE", "games")
LOG_TABLE    = "batter_game_log"
MLB_SEASON   = int(os.getenv("MLB_SEASON", "2026"))
MLB_BASE     = "https://statsapi.mlb.com/api/v1"
HTTP_TIMEOUT = 20
SLEEP        = float(os.getenv("REQUEST_SLEEP_SECONDS", "0.15"))
DEFAULT_DAYS = 21   # lookback window for daily mode


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(DATABASE_URL)


def ensure_table(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {LOG_TABLE} (
            player_id     INTEGER     NOT NULL,
            player_name   TEXT        NOT NULL,
            game_pk       BIGINT      NOT NULL,
            official_date DATE        NOT NULL,
            season        INT         NOT NULL,
            team          TEXT,
            hits          INT  DEFAULT 0,
            ab            INT  DEFAULT 0,
            pa            INT  DEFAULT 0,
            doubles       INT  DEFAULT 0,
            triples       INT  DEFAULT 0,
            home_runs     INT  DEFAULT 0,
            rbi           INT  DEFAULT 0,
            runs          INT  DEFAULT 0,
            walks         INT  DEFAULT 0,
            stolen_bases  INT  DEFAULT 0,
            strikeouts    INT  DEFAULT 0,
            total_bases   INT  DEFAULT 0,
            avg           DOUBLE PRECISION,
            obp           DOUBLE PRECISION,
            slg           DOUBLE PRECISION,
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (player_id, game_pk)
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_bgl_player_date ON {LOG_TABLE}(player_id, official_date DESC);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_bgl_date        ON {LOG_TABLE}(official_date);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_bgl_season      ON {LOG_TABLE}(season);")


# ---------------------------------------------------------------------------
# Player ID resolution
# ---------------------------------------------------------------------------

def get_lineup_players(cur) -> list[tuple[int, str]]:
    """Return (player_id, player_name) for players in today's lineups."""
    cur.execute("""
        SELECT DISTINCT l.player_id, l.player_name
        FROM lineups l
        JOIN games g ON l.game_pk = g.game_pk
        WHERE g.official_date = CURRENT_DATE
          AND l.player_id IS NOT NULL
        ORDER BY l.player_name
    """)
    return [(int(r[0]), r[1]) for r in cur.fetchall()]


def get_all_season_players() -> list[tuple[int, str]]:
    """Return (player_id, fullName) from MLB Stats API season batting stats."""
    url = f"{MLB_BASE}/stats"
    params = {
        "stats":      "season",
        "group":      "hitting",
        "season":     MLB_SEASON,
        "gameType":   "R",
        "sportId":    1,
        "playerPool": "All",
        "limit":      2000,
    }
    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        out = []
        for s in splits:
            p  = s.get("player", {})
            pa = int(s.get("stat", {}).get("plateAppearances", 0) or 0)
            pid  = p.get("id")
            name = p.get("fullName", "")
            if pid and name and pa >= 10:
                out.append((int(pid), name))
        return out
    except Exception as e:
        print(f"  season stats fetch failed: {e}")
        return []


# ---------------------------------------------------------------------------
# MLB Stats API game log
# ---------------------------------------------------------------------------

def fetch_game_log(player_id: int, season: int, since: date) -> list[dict]:
    """
    Fetch game-by-game hitting stats for one player from MLB Stats API.
    Returns list of row dicts filtered to games on or after `since`.
    """
    url = f"{MLB_BASE}/people/{player_id}/stats"
    params = {
        "stats":    "gameLog",
        "group":    "hitting",
        "season":   season,
        "gameType": "R",
    }
    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    rows = []
    for stat_group in data.get("stats", []):
        if stat_group.get("type", {}).get("displayName") != "gameLog":
            continue
        for split in stat_group.get("splits", []):
            # Date field — try both common locations
            game_date_str = (split.get("date") or
                             split.get("game", {}).get("gameDate", "")[:10])
            try:
                game_date = date.fromisoformat(game_date_str[:10])
            except Exception:
                continue

            if game_date < since:
                continue

            game_pk = split.get("game", {}).get("gamePk")
            if not game_pk:
                continue

            team = (split.get("team", {}).get("abbreviation") or
                    split.get("team", {}).get("name", ""))
            st = split.get("stat", {})

            pa  = int(st.get("plateAppearances", 0) or 0)
            ab  = int(st.get("atBats", 0) or 0)
            h   = int(st.get("hits", 0) or 0)
            d   = int(st.get("doubles", 0) or 0)
            t   = int(st.get("triples", 0) or 0)
            hr  = int(st.get("homeRuns", 0) or 0)
            rbi = int(st.get("rbi", 0) or 0)
            r_  = int(st.get("runs", 0) or 0)
            bb  = int(st.get("baseOnBalls", 0) or 0)
            sb  = int(st.get("stolenBases", 0) or 0)
            k   = int(st.get("strikeOuts", 0) or 0)
            tb  = int(st.get("totalBases", 0) or 0)
            # Fallback: compute TB from hit components if not provided
            if tb == 0 and (h > 0 or hr > 0):
                tb = (h - d - t - hr) + 2 * d + 3 * t + 4 * hr

            if pa == 0:
                continue

            rows.append({
                "player_id":    player_id,
                "game_pk":      int(game_pk),
                "official_date": game_date,
                "season":       season,
                "team":         team,
                "hits":         h,
                "ab":           ab,
                "pa":           pa,
                "doubles":      d,
                "triples":      t,
                "home_runs":    hr,
                "rbi":          rbi,
                "runs":         r_,
                "walks":        bb,
                "stolen_bases": sb,
                "strikeouts":   k,
                "total_bases":  tb,
                "avg":          round(h / ab, 4) if ab > 0 else None,
                "obp":          round((h + bb) / pa, 4) if pa > 0 else None,
                "slg":          round(tb / ab, 4) if ab > 0 else None,
            })

    return rows


def upsert_rows(cur, player_name: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    data = [
        (
            r["player_id"], player_name, r["game_pk"], r["official_date"], r["season"],
            r["team"],
            r["hits"], r["ab"], r["pa"], r["doubles"], r["triples"],
            r["home_runs"], r["rbi"], r["runs"], r["walks"],
            r["stolen_bases"], r["strikeouts"], r["total_bases"],
            r["avg"], r["obp"], r["slg"],
            datetime.now(timezone.utc),
        )
        for r in rows
    ]
    sql = f"""
    INSERT INTO {LOG_TABLE} (
        player_id, player_name, game_pk, official_date, season,
        team, hits, ab, pa, doubles, triples,
        home_runs, rbi, runs, walks,
        stolen_bases, strikeouts, total_bases,
        avg, obp, slg, updated_at
    ) VALUES %s
    ON CONFLICT (player_id, game_pk) DO UPDATE SET
        hits          = EXCLUDED.hits,
        ab            = EXCLUDED.ab,
        pa            = EXCLUDED.pa,
        doubles       = EXCLUDED.doubles,
        triples       = EXCLUDED.triples,
        home_runs     = EXCLUDED.home_runs,
        rbi           = EXCLUDED.rbi,
        runs          = EXCLUDED.runs,
        walks         = EXCLUDED.walks,
        stolen_bases  = EXCLUDED.stolen_bases,
        strikeouts    = EXCLUDED.strikeouts,
        total_bases   = EXCLUDED.total_bases,
        avg           = EXCLUDED.avg,
        obp           = EXCLUDED.obp,
        slg           = EXCLUDED.slg,
        updated_at    = NOW();
    """
    execute_values(cur, sql, data, page_size=500)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load batter game logs from MLB Stats API.")
    parser.add_argument("--mode", choices=["today", "all"], default="today",
                        help="'today': lineup players only (fast, for daily run). "
                             "'all': all season batters (slower, for backfill).")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Lookback window in days (default: {DEFAULT_DAYS}).")
    args = parser.parse_args()

    since = date.today() - timedelta(days=args.days)

    print("=" * 55)
    print(f"  load_batter_game_log.py | mode={args.mode} | since={since}")
    print("=" * 55)

    c = get_conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_table(cur)
            c.commit()
            print("Table ready.\n")

            if args.mode == "today":
                players = get_lineup_players(cur)
                print(f"  Today's lineup players: {len(players)}")
            else:
                players = get_all_season_players()
                print(f"  All season batters: {len(players)}")

            if not players:
                print("  No players found — check lineups table or MLB API.")
                return

            total_rows = 0
            for i, (pid, pname) in enumerate(players, 1):
                rows = fetch_game_log(pid, MLB_SEASON, since)
                n    = upsert_rows(cur, pname, rows)
                total_rows += n

                if i % 30 == 0 or i == len(players):
                    c.commit()
                    print(f"  {i:>4}/{len(players)} players  |  {total_rows} rows upserted")

                time.sleep(SLEEP)

            c.commit()

        print(f"\n{'='*55}")
        print(f"  Done. {total_rows} batter game-log rows upserted.")
        print(f"  Players processed: {len(players)}")
        print(f"{'='*55}")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()
