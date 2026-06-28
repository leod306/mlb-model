"""
scripts/load_player_props.py
Fetches player prop lines from The Odds API and scores them against
BvP data to find edges.

Props covered:
  batter_hits, batter_total_bases, batter_home_runs,
  pitcher_strikeouts, batter_walks

Run daily after load_odds.py:
    python scripts/load_player_props.py
"""

from __future__ import annotations

import math
import os
from scipy.stats import poisson as _poisson
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
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

MLB_SEASON    = int(os.getenv("MLB_SEASON", "2026"))
ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
HTTP_TIMEOUT  = 20

GAMES_TABLE = os.getenv("MLB_GAMES_TABLE", "games")
PROPS_TABLE = "player_props"

PROP_MARKETS = [
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
    "pitcher_strikeouts",
    "batter_walks",
]

# Bookmakers to average across
BOOKMAKERS = "draftkings,fanduel,betmgm,betonlineag,betrivers"

# Minimum edge to flag as a play — 8% keeps ~10-15% of props as edges
MIN_EDGE = 0.08
# Maximum plausible edge — anything above this is likely a data artifact
MAX_EDGE = 0.20
# Minimum books required on each side to trust the line
MIN_BOOKS_EACH_SIDE = 2

# League average fallbacks
LEAGUE_AVG   = 0.255
LEAGUE_OBP   = 0.320
LEAGUE_SLG   = 0.415
LEAGUE_K_PER_9 = 8.5


def log(msg: str) -> None:
    print(msg, flush=True)


def _safe_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def american_to_prob(american: float) -> Optional[float]:
    try:
        a = float(american)
        if a > 0:
            return 100.0 / (a + 100.0)
        return (-a) / ((-a) + 100.0)
    except Exception:
        return None


# =============================================================================
# DB SETUP
# =============================================================================

def ensure_props_table() -> None:
    sql = f"""
    CREATE TABLE IF NOT EXISTS {PROPS_TABLE} (
        id              SERIAL PRIMARY KEY,
        prop_date       DATE NOT NULL,
        game_pk         BIGINT,
        home_team       TEXT,
        away_team       TEXT,
        player_name     TEXT NOT NULL,
        player_team     TEXT,
        prop_type       TEXT NOT NULL,
        line            DOUBLE PRECISION,
        over_price      INT,
        under_price     INT,
        avg_over_prob   DOUBLE PRECISION,
        projection      DOUBLE PRECISION,
        edge            DOUBLE PRECISION,
        pick            TEXT,
        confidence      TEXT,
        bookmakers_used INT,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        updated_at      TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (prop_date, player_name, prop_type, line)
    );
    CREATE INDEX IF NOT EXISTS idx_player_props_date ON {PROPS_TABLE}(prop_date);
    CREATE INDEX IF NOT EXISTS idx_player_props_game ON {PROPS_TABLE}(game_pk);
    """
    with engine.begin() as conn:
        for stmt in sql.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))


def table_exists(name: str) -> bool:
    sql = "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name=:n)"
    with engine.begin() as conn:
        return bool(conn.execute(text(sql), {"n": name}).scalar())


# =============================================================================
# LOAD TODAY'S GAMES + EVENT IDs FROM ODDS API
# =============================================================================

def get_target_date() -> Optional[date]:
    sql = f"SELECT MIN(official_date) FROM {GAMES_TABLE} WHERE official_date >= CURRENT_DATE AND season = :s"
    with engine.begin() as conn:
        row = conn.execute(text(sql), {"s": MLB_SEASON}).fetchone()
    return row[0] if row and row[0] else None


def fetch_events() -> List[Dict]:
    """Get today's MLB event IDs from Odds API."""
    if not ODDS_API_KEY:
        log("ODDS_API_KEY not set.")
        return []
    url = f"{ODDS_API_BASE}/sports/baseball_mlb/events"
    try:
        r = requests.get(url, params={"apiKey": ODDS_API_KEY}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"Events fetch failed: {e}")
        return []


# =============================================================================
# FETCH PLAYER PROPS PER EVENT
# =============================================================================

def fetch_props_for_event(event_id: str, home_team: str, away_team: str) -> List[Dict]:
    """
    Fetch all prop markets for one game.
    Returns list of raw outcome dicts with player_name, prop_type, line, prices.
    """
    if not ODDS_API_KEY:
        return []

    results = []
    # Fetch one market at a time to stay within URL length limits
    for market in PROP_MARKETS:
        url = f"{ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds"
        params = {
            "apiKey":      ODDS_API_KEY,
            "regions":     "us",
            "markets":     market,
            "bookmakers":  BOOKMAKERS,
            "oddsFormat":  "american",
        }
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log(f"  Props fetch failed ({market}): {e}")
            continue

        # Aggregate prices per player+line across bookmakers
        player_data: Dict[str, Dict] = {}

        for bookmaker in data.get("bookmakers", []):
            for mkt in bookmaker.get("markets", []):
                if mkt.get("key") != market:
                    continue
                for outcome in mkt.get("outcomes", []):
                    player = outcome.get("description", "")
                    side   = outcome.get("name", "")   # Over / Under
                    price  = _safe_float(outcome.get("price"))
                    line   = _safe_float(outcome.get("point"))

                    if not player or price is None or line is None:
                        continue

                    key = f"{player}|{line}"
                    if key not in player_data:
                        player_data[key] = {
                            "player_name": player,
                            "prop_type":   market,
                            "line":        line,
                            "home_team":   home_team,
                            "away_team":   away_team,
                            "over_prices": [],
                            "under_prices": [],
                        }

                    if side == "Over":
                        player_data[key]["over_prices"].append(price)
                    elif side == "Under":
                        player_data[key]["under_prices"].append(price)

        results.extend(player_data.values())

    return results


# =============================================================================
# 2026 SEASON BATTING STATS — primary projection source
# =============================================================================

def load_season_batting_stats() -> pd.DataFrame:
    """
    Pull current season batting stats from MLB Stats API (free, no key).
    Returns DataFrame with player_id, avg, slg, obp, ab, pa, hits, home_runs, walks.
    This is the primary source — far more reliable than sparse BvP data.
    """
    url = "https://statsapi.mlb.com/api/v1/stats"
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
        rows = []
        for s in splits:
            p = s.get("player", {})
            st = s.get("stat", {})
            ab = int(st.get("atBats", 0) or 0)
            pa = int(st.get("plateAppearances", 0) or 0)
            if pa < 10:
                continue
            rows.append({
                "player_id":   p.get("id"),
                "player_name": (p.get("fullName") or "").lower().strip(),
                "avg":         float(st.get("avg", 0) or 0),
                "slg":         float(st.get("slg", 0) or 0),
                "obp":         float(st.get("obp", 0) or 0),
                "ab":          ab,
                "pa":          pa,
                "hits":        int(st.get("hits", 0) or 0),
                "home_runs":   int(st.get("homeRuns", 0) or 0),
                "walks":       int(st.get("baseOnBalls", 0) or 0),
            })
        log(f"  Season stats loaded: {len(rows)} batters")
        return pd.DataFrame(rows)
    except Exception as e:
        log(f"  Season stats fetch failed: {e}")
        return pd.DataFrame()


def load_season_pitching_stats() -> pd.DataFrame:
    """
    Pull current season pitching stats from MLB Stats API.
    Used as fallback when pitcher_game_log has no recent starts.
    """
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats":    "season",
        "group":    "pitching",
        "season":   MLB_SEASON,
        "gameType": "R",
        "sportId":  1,
        "limit":    1000,
    }
    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        rows = []
        for s in splits:
            p = s.get("player", {})
            st = s.get("stat", {})
            ip = float(st.get("inningsPitched", 0) or 0)
            if ip < 5:
                continue
            rows.append({
                "player_id":    p.get("id"),
                "pitcher_name": (p.get("fullName") or "").lower().strip(),
                "k_per_9":      float(st.get("strikeoutsPer9Inn", 0) or 0),
                "innings_pitched": ip,
                "avg_ip_per_start": ip / max(int(st.get("gamesStarted", 1) or 1), 1),
            })
        log(f"  Season pitching stats loaded: {len(rows)} pitchers")
        return pd.DataFrame(rows)
    except Exception as e:
        log(f"  Season pitching stats fetch failed: {e}")
        return pd.DataFrame()


# =============================================================================
# BvP + PITCHER DATA
# =============================================================================

def load_bvp() -> pd.DataFrame:
    """Matchup-specific BvP (batter vs specific pitcher, PA >= 3)."""
    if not table_exists("batter_vs_pitcher"):
        return pd.DataFrame()
    try:
        return pd.read_sql(text("""
            SELECT batter_id, pitcher_id, pa, ab, hits, home_runs,
                   strikeouts, walks, avg, obp, slg, hard_hit_pct, avg_exit_velo
            FROM batter_vs_pitcher WHERE pa >= 3
        """), engine)
    except Exception:
        return pd.DataFrame()


def load_bvp_career() -> pd.DataFrame:
    """
    Career aggregate per batter across ALL pitchers.
    Used as fallback when matchup-specific BvP is unavailable.
    """
    if not table_exists("batter_vs_pitcher"):
        return pd.DataFrame()
    try:
        return pd.read_sql(text("""
            SELECT
                batter_id,
                SUM(pa)                                                  AS pa,
                SUM(ab)                                                  AS ab,
                SUM(hits)                                                AS hits,
                SUM(home_runs)                                           AS home_runs,
                SUM(walks)                                               AS walks,
                CASE WHEN SUM(ab) > 0
                     THEN SUM(hits)::float / SUM(ab) END                AS avg,
                CASE WHEN SUM(pa) > 0
                     THEN (SUM(hits) + SUM(walks))::float / SUM(pa) END AS obp,
                CASE WHEN SUM(ab) > 0
                     THEN (SUM(hits) + SUM(home_runs)*3)::float / SUM(ab) END AS slg
            FROM batter_vs_pitcher
            WHERE pa >= 1
            GROUP BY batter_id
            HAVING SUM(pa) >= 5
        """), engine)
    except Exception:
        return pd.DataFrame()


def load_lineups(game_pks: List[int]) -> pd.DataFrame:
    """
    Load lineups for today's game PKs first. If empty, fall back to the
    full lineups table to build a name→player_id lookup (uses most recent
    entry per player so RotoWire projected lineups are always usable).
    """
    if not table_exists("lineups"):
        return pd.DataFrame()
    try:
        # Try today's games first
        if game_pks:
            df = pd.read_sql(
                text("SELECT game_pk, player_id, player_name, side, batting_order FROM lineups WHERE game_pk = ANY(:pks)"),
                engine, params={"pks": game_pks}
            )
            if not df.empty:
                return df

        # Fall back: latest entry per player across all games
        df = pd.read_sql(text("""
            SELECT DISTINCT ON (player_name) game_pk, player_id, player_name, side, batting_order
            FROM lineups
            ORDER BY player_name, game_pk DESC
        """), engine)
        return df
    except Exception:
        return pd.DataFrame()


def load_pitcher_game_log() -> pd.DataFrame:
    if not table_exists("pitcher_game_log"):
        return pd.DataFrame()
    try:
        df = pd.read_sql(text("""
            SELECT pitcher_name, team, role, official_date,
                   innings_pitched, strikeouts, walks, er_allowed, hits_allowed
            FROM pitcher_game_log
            WHERE role = 'SP' AND official_date >= CURRENT_DATE - INTERVAL '60 days'
            ORDER BY official_date DESC
        """), engine)
        df["pitcher_name_key"] = df["pitcher_name"].str.strip().str.lower()
        return df
    except Exception:
        return pd.DataFrame()


def load_probables() -> pd.DataFrame:
    if not table_exists("game_probables"):
        return pd.DataFrame()
    try:
        return pd.read_sql(text("""
            SELECT game_pk, home_sp_name, away_sp_name, home_sp_id, away_sp_id
            FROM game_probables
        """), engine)
    except Exception:
        return pd.DataFrame()


def load_games_for_date(target_date: date) -> pd.DataFrame:
    try:
        return pd.read_sql(
            text(f"SELECT game_pk, home_team, away_team FROM {GAMES_TABLE} WHERE official_date = :d AND season = :s"),
            engine, params={"d": target_date, "s": MLB_SEASON}
        )
    except Exception:
        return pd.DataFrame()


# =============================================================================
# PROJECTION ENGINE
# =============================================================================

def _get_player_id(player_name: str, lineups_df: pd.DataFrame,
                   season_df: pd.DataFrame = None) -> Optional[int]:
    """
    Resolve player name → player_id.
    Tries lineups table first, then season stats by name.
    """
    name_key = player_name.strip().lower()

    if not lineups_df.empty:
        match = lineups_df[lineups_df["player_name"].str.lower().str.strip() == name_key]
        if not match.empty and match.iloc[0]["player_id"]:
            return int(match.iloc[0]["player_id"])

    if season_df is not None and not season_df.empty:
        match = season_df[season_df["player_name"] == name_key]
        if not match.empty:
            return int(match.iloc[0]["player_id"])

    return None


def _get_season_row(player_name: str, season_df: pd.DataFrame,
                    lineups_df: pd.DataFrame) -> Optional[pd.Series]:
    """Get season stats row by player name or player_id lookup."""
    if season_df is None or season_df.empty:
        return None
    name_key = player_name.strip().lower()

    # Direct name match
    match = season_df[season_df["player_name"] == name_key]
    if not match.empty:
        return match.iloc[0]

    # Try via player_id from lineups
    pid = _get_player_id(player_name, lineups_df)
    if pid is not None:
        match = season_df[season_df["player_id"] == pid]
        if not match.empty:
            return match.iloc[0]

    return None


def project_hits(player_name: str, bvp_df: pd.DataFrame, lineups_df: pd.DataFrame,
                 career_df: pd.DataFrame = None,
                 season_df: pd.DataFrame = None) -> Optional[float]:
    """
    Hits per game ≈ avg × 4.0 PA.
    Priority: season stats → career BvP → matchup BvP → None
    """
    # 1. Season stats (most reliable)
    row = _get_season_row(player_name, season_df, lineups_df)
    if row is not None:
        avg = _safe_float(row.get("avg"))
        pa  = _safe_float(row.get("pa")) or 0
        if avg is not None and avg > 0:
            weight  = min(pa / 200.0, 1.0)
            reg_avg = weight * avg + (1 - weight) * LEAGUE_AVG
            return round(reg_avg * 4.0, 3)

    # 2. Career BvP fallback
    if career_df is None:
        career_df = pd.DataFrame()
    player_id = _get_player_id(player_name, lineups_df)
    if player_id and not career_df.empty:
        r = career_df[career_df["batter_id"] == player_id]
        if not r.empty:
            avg = _safe_float(r.iloc[0].get("avg"))
            pa  = _safe_float(r.iloc[0].get("pa")) or 0
            if avg is not None and pa >= 10:
                weight  = min(pa / 100.0, 1.0)
                reg_avg = weight * avg + (1 - weight) * LEAGUE_AVG
                return round(reg_avg * 4.0, 3)

    return None


def project_total_bases(player_name: str, bvp_df: pd.DataFrame, lineups_df: pd.DataFrame,
                        career_df: pd.DataFrame = None,
                        season_df: pd.DataFrame = None) -> Optional[float]:
    """Total bases per game ≈ slg × 4.0 PA."""
    row = _get_season_row(player_name, season_df, lineups_df)
    if row is not None:
        slg = _safe_float(row.get("slg"))
        pa  = _safe_float(row.get("pa")) or 0
        if slg is not None and slg > 0:
            weight  = min(pa / 200.0, 1.0)
            reg_slg = weight * slg + (1 - weight) * LEAGUE_SLG
            return round(reg_slg * 4.0, 3)

    if career_df is None:
        career_df = pd.DataFrame()
    player_id = _get_player_id(player_name, lineups_df)
    if player_id and not career_df.empty:
        r = career_df[career_df["batter_id"] == player_id]
        if not r.empty:
            slg = _safe_float(r.iloc[0].get("slg"))
            pa  = _safe_float(r.iloc[0].get("pa")) or 0
            if slg is not None and pa >= 10:
                weight  = min(pa / 100.0, 1.0)
                reg_slg = weight * slg + (1 - weight) * LEAGUE_SLG
                return round(reg_slg * 4.0, 3)

    return None


def project_home_runs(player_name: str, bvp_df: pd.DataFrame, lineups_df: pd.DataFrame,
                      career_df: pd.DataFrame = None,
                      season_df: pd.DataFrame = None) -> Optional[float]:
    """HR per game ≈ (HR / PA) × 4.0."""
    LEAGUE_HR_RATE = 0.033 / 4.0  # ~3.3% of PA result in HR league-wide

    row = _get_season_row(player_name, season_df, lineups_df)
    if row is not None:
        hr = _safe_float(row.get("home_runs")) or 0
        pa = _safe_float(row.get("pa")) or 0
        if pa >= 20:
            hr_rate = hr / pa
            weight  = min(pa / 300.0, 1.0)
            reg_hr  = weight * hr_rate + (1 - weight) * LEAGUE_HR_RATE
            return round(reg_hr * 4.0, 4)

    if career_df is None:
        career_df = pd.DataFrame()
    player_id = _get_player_id(player_name, lineups_df)
    if player_id and not career_df.empty:
        r = career_df[career_df["batter_id"] == player_id]
        if not r.empty:
            hr = _safe_float(r.iloc[0].get("home_runs")) or 0
            pa = _safe_float(r.iloc[0].get("pa")) or 0
            if pa >= 10:
                hr_rate = hr / pa
                weight  = min(pa / 100.0, 1.0)
                reg_hr  = weight * hr_rate + (1 - weight) * LEAGUE_HR_RATE
                return round(reg_hr * 4.0, 4)

    return None


def project_walks(player_name: str, bvp_df: pd.DataFrame, lineups_df: pd.DataFrame,
                  career_df: pd.DataFrame = None,
                  season_df: pd.DataFrame = None) -> Optional[float]:
    """Walks per game ≈ (BB / PA) × 4.0."""
    LEAGUE_BB_RATE = LEAGUE_OBP - LEAGUE_AVG  # ~0.065

    row = _get_season_row(player_name, season_df, lineups_df)
    if row is not None:
        walks = _safe_float(row.get("walks")) or 0
        pa    = _safe_float(row.get("pa")) or 0
        if pa >= 20:
            bb_rate = walks / pa
            weight  = min(pa / 200.0, 1.0)
            reg_bb  = weight * bb_rate + (1 - weight) * LEAGUE_BB_RATE
            return round(reg_bb * 4.0, 4)

    if career_df is None:
        career_df = pd.DataFrame()
    player_id = _get_player_id(player_name, lineups_df)
    if player_id and not career_df.empty:
        r = career_df[career_df["batter_id"] == player_id]
        if not r.empty:
            walks = _safe_float(r.iloc[0].get("walks")) or 0
            pa    = _safe_float(r.iloc[0].get("pa")) or 0
            if pa >= 10:
                bb_rate = walks / pa
                weight  = min(pa / 100.0, 1.0)
                reg_bb  = weight * bb_rate + (1 - weight) * LEAGUE_BB_RATE
                return round(reg_bb * 4.0, 4)

    return None


def project_strikeouts(pitcher_name: str, pgl_df: pd.DataFrame,
                       season_pitch_df: pd.DataFrame = None) -> Optional[float]:
    """
    Project pitcher Ks using last 5 starts (primary) or season K/9 (fallback).
    """
    key = pitcher_name.strip().lower()

    # 1. Recent game log (best — actual recent starts)
    if not pgl_df.empty:
        recent = pgl_df[pgl_df["pitcher_name_key"] == key].head(5)
        if not recent.empty:
            total_ip = _safe_float(recent["innings_pitched"].sum()) or 0
            total_k  = _safe_float(recent["strikeouts"].sum())      or 0
            if total_ip >= 1:
                k_per_9 = (total_k / total_ip) * 9.0
                weight  = min(len(recent) / 5.0, 1.0)
                reg_k9  = weight * k_per_9 + (1 - weight) * LEAGUE_K_PER_9
                avg_ip  = total_ip / len(recent)
                return round(reg_k9 / 9.0 * avg_ip, 2)

    # 2. Season stats fallback
    if season_pitch_df is not None and not season_pitch_df.empty:
        match = season_pitch_df[season_pitch_df["pitcher_name"] == key]
        if not match.empty:
            k9  = _safe_float(match.iloc[0].get("k_per_9")) or LEAGUE_K_PER_9
            avg_ip = _safe_float(match.iloc[0].get("avg_ip_per_start")) or 5.5
            return round(k9 / 9.0 * avg_ip, 2)

    return None


# =============================================================================
# SCORE PROPS
# =============================================================================

def score_prop(raw: Dict, lineups_df: pd.DataFrame, bvp_df: pd.DataFrame,
               pgl_df: pd.DataFrame, probables_df: pd.DataFrame,
               game_pk: Optional[int], career_df: pd.DataFrame = None,
               season_df: pd.DataFrame = None,
               season_pitch_df: pd.DataFrame = None) -> Optional[Dict]:
    """
    Compare book line to projection. Return scored prop dict or None.
    """
    player    = raw["player_name"]
    prop_type = raw["prop_type"]
    line      = raw["line"]
    over_px   = raw["over_prices"]
    under_px  = raw["under_prices"]

    if not over_px:
        return None

    # Require enough books on each side for a reliable consensus price
    if len(over_px) < MIN_BOOKS_EACH_SIDE or len(under_px) < MIN_BOOKS_EACH_SIDE:
        return None

    # Average prices across books
    avg_over  = sum(over_px)  / len(over_px)
    avg_under = sum(under_px) / len(under_px)

    # Convert to no-vig probability
    over_prob  = american_to_prob(avg_over)
    under_prob = american_to_prob(avg_under) if avg_under is not None else None

    if over_prob is None:
        return None

    # Remove vig — need both sides to de-vig properly
    # If only one side available, use raw prob but require larger edge buffer
    if under_prob is not None:
        total = over_prob + under_prob
        if total < 0.5 or total > 2.0:  # sanity check
            return None
        over_prob_nv = over_prob / total
    else:
        # One-sided market — can't de-vig, skip to avoid false edges
        return None

    # Project — season stats are primary source; BvP/career are fallbacks
    cd = career_df if career_df is not None else pd.DataFrame()
    sd = season_df if season_df is not None else pd.DataFrame()
    spd = season_pitch_df if season_pitch_df is not None else pd.DataFrame()

    if prop_type == "batter_hits":
        proj = project_hits(player, bvp_df, lineups_df, cd, sd)
    elif prop_type == "batter_total_bases":
        proj = project_total_bases(player, bvp_df, lineups_df, cd, sd)
    elif prop_type == "batter_home_runs":
        proj = project_home_runs(player, bvp_df, lineups_df, cd, sd)
    elif prop_type == "batter_walks":
        proj = project_walks(player, bvp_df, lineups_df, cd, sd)
    elif prop_type == "pitcher_strikeouts":
        proj = project_strikeouts(player, pgl_df, spd)
    else:
        return None

    if proj is None:
        return None

    # Convert projection (Poisson mean) → P(stat > line)
    # For half-integer lines (0.5, 1.5, 2.5), P(X > line) = P(X >= floor(line)+1)
    # = 1 - CDF(floor(line), lambda=proj)
    k = int(math.floor(line))
    try:
        proj_over_prob = float(1.0 - _poisson.cdf(k, max(proj, 0.01)))
    except Exception:
        proj_over_prob = 0.5
    proj_over_prob = min(max(proj_over_prob, 0.02), 0.98)
    edge = round(proj_over_prob - over_prob_nv, 4)

    # Skip HR props with line >= 1.5 (need 2+ HRs — near impossible, always misflagged)
    if prop_type == "batter_home_runs" and line >= 1.5:
        return None

    # Pick direction — cap at MAX_EDGE to filter data artifacts
    if abs(edge) < MIN_EDGE or abs(edge) > MAX_EDGE:
        pick = "PASS"
        confidence = "LOW"
    elif edge > 0:
        pick = "OVER"
        confidence = "HIGH" if edge >= 0.13 else "MED"
    else:
        pick = "UNDER"
        confidence = "HIGH" if edge <= -0.13 else "MED"

    # Determine player team
    player_team = None
    if not lineups_df.empty:
        name_key = player.strip().lower()
        match = lineups_df[lineups_df["player_name"].str.lower().str.strip() == name_key]
        if not match.empty:
            game_pk_match = match.iloc[0].get("game_pk")
            side = match.iloc[0].get("side")
            if game_pk_match and side:
                game_row_df = pd.DataFrame([{"game_pk": game_pk, "home_team": raw["home_team"], "away_team": raw["away_team"]}])
                player_team = raw["home_team"] if side == "home" else raw["away_team"]

    return {
        "prop_date":      date.today(),
        "game_pk":        game_pk,
        "home_team":      raw["home_team"],
        "away_team":      raw["away_team"],
        "player_name":    player,
        "player_team":    player_team,
        "prop_type":      prop_type,
        "line":           line,
        "over_price":     int(round(avg_over)),
        "under_price":    int(round(avg_under)) if avg_under is not None else None,
        "avg_over_prob":  round(over_prob_nv, 4),
        "projection":     round(proj, 3),
        "edge":           edge,
        "pick":           pick,
        "confidence":     confidence,
        "bookmakers_used": len(over_px),
    }


# =============================================================================
# UPSERT
# =============================================================================

def upsert_props(rows: List[Dict], target_date) -> int:
    if not rows:
        return 0

    # Clear today's props first so stale data from previous runs doesn't linger
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {PROPS_TABLE} WHERE prop_date = :d"), {"d": target_date})
    log(f"  Cleared existing props for {target_date}, saving {len(rows)} fresh rows")

    sql = f"""
    INSERT INTO {PROPS_TABLE} (
        prop_date, game_pk, home_team, away_team, player_name, player_team,
        prop_type, line, over_price, under_price, avg_over_prob,
        projection, edge, pick, confidence, bookmakers_used, updated_at
    ) VALUES (
        :prop_date, :game_pk, :home_team, :away_team, :player_name, :player_team,
        :prop_type, :line, :over_price, :under_price, :avg_over_prob,
        :projection, :edge, :pick, :confidence, :bookmakers_used, NOW()
    )
    ON CONFLICT (prop_date, player_name, prop_type, line) DO UPDATE SET
        game_pk        = EXCLUDED.game_pk,
        over_price     = EXCLUDED.over_price,
        under_price    = EXCLUDED.under_price,
        avg_over_prob  = EXCLUDED.avg_over_prob,
        projection     = EXCLUDED.projection,
        edge           = EXCLUDED.edge,
        pick           = EXCLUDED.pick,
        confidence     = EXCLUDED.confidence,
        bookmakers_used = EXCLUDED.bookmakers_used,
        updated_at     = NOW()
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)
    return len(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log("=" * 60)
    log(f"load_player_props.py | {datetime.now(timezone.utc).date()}")
    log("=" * 60)

    if not ODDS_API_KEY:
        log("ERROR: ODDS_API_KEY not set. Add it to .env or Heroku config.")
        return

    ensure_props_table()

    target_date = get_target_date()
    if not target_date:
        log("No upcoming slate found.")
        return
    log(f"Target date: {target_date}")

    # Load supporting data
    games_df      = load_games_for_date(target_date)
    lineups_df    = load_lineups(games_df["game_pk"].tolist() if not games_df.empty else [])
    bvp_df        = load_bvp()
    career_df     = load_bvp_career()
    pgl_df        = load_pitcher_game_log()
    probables_df  = load_probables()
    season_df     = load_season_batting_stats()
    season_pitch_df = load_season_pitching_stats()

    log(f"Games: {len(games_df)}  Lineups: {len(lineups_df)}  "
        f"BvP rows: {len(bvp_df)}  Career rows: {len(career_df)}  "
        f"Season batters: {len(season_df)}  Season pitchers: {len(season_pitch_df)}")

    # Normalize player names in lineups
    if not lineups_df.empty:
        lineups_df["player_name"] = lineups_df["player_name"].astype(str)

    # Fetch today's events from Odds API
    events = fetch_events()
    log(f"Events from Odds API: {len(events)}")

    # Build team → game_pk map
    team_to_pk: Dict[str, int] = {}
    if not games_df.empty:
        for _, row in games_df.iterrows():
            team_to_pk[row["home_team"]] = int(row["game_pk"])
            team_to_pk[row["away_team"]] = int(row["game_pk"])

    TEAM_MAP = {
        "Arizona Diamondbacks": "ARI",  "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL",     "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC",          "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN",       "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL",      "Detroit Tigers": "DET",
        "Houston Astros": "HOU",        "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA",    "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA",         "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN",       "New York Mets": "NYM",
        "New York Yankees": "NYY",      "Athletics": "ATH",
        "Oakland Athletics": "ATH",     "Philadelphia Phillies": "PHI",
        "Pittsburgh Pirates": "PIT",    "San Diego Padres": "SD",
        "San Francisco Giants": "SF",   "Seattle Mariners": "SEA",
        "St. Louis Cardinals": "STL",   "Tampa Bay Rays": "TB",
        "Texas Rangers": "TEX",         "Toronto Blue Jays": "TOR",
        "Washington Nationals": "WSH",
    }

    all_scored: List[Dict] = []
    total_raw  = 0

    for event in events:
        home_team = TEAM_MAP.get(event.get("home_team", ""), event.get("home_team", ""))
        away_team = TEAM_MAP.get(event.get("away_team", ""), event.get("away_team", ""))
        event_id  = event.get("id", "")

        game_pk = team_to_pk.get(home_team)

        log(f"\n  {away_team} @ {home_team} [{event_id[:8]}...]")
        raw_props = fetch_props_for_event(event_id, home_team, away_team)
        log(f"    Raw prop outcomes: {len(raw_props)}")
        total_raw += len(raw_props)

        for raw in raw_props:
            scored = score_prop(
                raw, lineups_df, bvp_df, pgl_df, probables_df, game_pk,
                career_df, season_df, season_pitch_df
            )
            if scored and scored["pick"] != "PASS":
                all_scored.append(scored)

    log(f"\nTotal raw props fetched: {total_raw}")
    log(f"Props with edge (non-PASS): {len(all_scored)}")

    # Save
    n = upsert_props(all_scored, target_date)
    log(f"Saved {n} props to {PROPS_TABLE}")

    # Preview top edges
    if all_scored:
        top = sorted(all_scored, key=lambda x: abs(x["edge"]), reverse=True)[:10]
        log("\nTOP 10 EDGES:")
        log(f"{'Player':<25} {'Type':<22} {'Line':>5} {'Proj':>6} {'Edge':>7} {'Pick':<6} {'Conf'}")
        log("-" * 80)
        for p in top:
            log(f"{p['player_name']:<25} {p['prop_type']:<22} {p['line']:>5.1f} "
                f"{p['projection']:>6.3f} {p['edge']:>+7.3f} {p['pick']:<6} {p['confidence']}")


if __name__ == "__main__":
    main()
