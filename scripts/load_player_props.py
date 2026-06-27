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

# Minimum edge to flag as a play (e.g. 0.05 = 5%)
MIN_EDGE = 0.05

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
# BvP + PITCHER DATA
# =============================================================================

def load_bvp() -> pd.DataFrame:
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


def load_lineups(game_pks: List[int]) -> pd.DataFrame:
    if not game_pks or not table_exists("lineups"):
        return pd.DataFrame()
    try:
        return pd.read_sql(
            text("SELECT game_pk, player_id, player_name, side, batting_order FROM lineups WHERE game_pk = ANY(:pks)"),
            engine, params={"pks": game_pks}
        )
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

def project_hits(player_name: str, bvp_df: pd.DataFrame, lineups_df: pd.DataFrame) -> Optional[float]:
    """
    Project expected hits using BvP avg if available, else league average.
    Hits per game ≈ avg × (plate appearances per game ~4.0)
    """
    if bvp_df.empty or lineups_df.empty:
        return LEAGUE_AVG * 4.0

    name_key = player_name.strip().lower()
    # Try to match player in lineups
    match = lineups_df[lineups_df["player_name"].str.lower().str.strip() == name_key]
    if match.empty:
        return LEAGUE_AVG * 4.0

    player_id = match.iloc[0]["player_id"]
    bvp = bvp_df[bvp_df["batter_id"] == player_id]
    if bvp.empty:
        return LEAGUE_AVG * 4.0

    avg = _safe_float(bvp.iloc[0].get("avg")) or LEAGUE_AVG
    pa  = _safe_float(bvp.iloc[0].get("pa")) or 12
    # Regress toward league avg based on sample size
    weight = min(pa / 50.0, 1.0)
    reg_avg = weight * avg + (1 - weight) * LEAGUE_AVG
    return round(reg_avg * 4.0, 3)


def project_total_bases(player_name: str, bvp_df: pd.DataFrame, lineups_df: pd.DataFrame) -> Optional[float]:
    if bvp_df.empty or lineups_df.empty:
        return (LEAGUE_SLG * 4.0)

    name_key = player_name.strip().lower()
    match = lineups_df[lineups_df["player_name"].str.lower().str.strip() == name_key]
    if match.empty:
        return LEAGUE_SLG * 4.0

    player_id = match.iloc[0]["player_id"]
    bvp = bvp_df[bvp_df["batter_id"] == player_id]
    if bvp.empty:
        return LEAGUE_SLG * 4.0

    slg = _safe_float(bvp.iloc[0].get("slg")) or LEAGUE_SLG
    pa  = _safe_float(bvp.iloc[0].get("pa")) or 12
    weight  = min(pa / 50.0, 1.0)
    reg_slg = weight * slg + (1 - weight) * LEAGUE_SLG
    return round(reg_slg * 4.0, 3)


def project_home_runs(player_name: str, bvp_df: pd.DataFrame, lineups_df: pd.DataFrame) -> Optional[float]:
    if bvp_df.empty or lineups_df.empty:
        return 0.10  # ~1 HR per 10 games league avg

    name_key = player_name.strip().lower()
    match = lineups_df[lineups_df["player_name"].str.lower().str.strip() == name_key]
    if match.empty:
        return 0.10

    player_id = match.iloc[0]["player_id"]
    bvp = bvp_df[bvp_df["batter_id"] == player_id]
    if bvp.empty:
        return 0.10

    hr = _safe_float(bvp.iloc[0].get("home_runs")) or 0
    pa = _safe_float(bvp.iloc[0].get("pa"))        or 1
    hr_rate = hr / pa
    weight  = min(pa / 80.0, 1.0)
    reg_hr  = weight * hr_rate + (1 - weight) * (0.10 / 4.0)
    return round(reg_hr * 4.0, 4)


def project_walks(player_name: str, bvp_df: pd.DataFrame, lineups_df: pd.DataFrame) -> Optional[float]:
    if bvp_df.empty or lineups_df.empty:
        return (LEAGUE_OBP - LEAGUE_AVG) * 4.0  # walks ≈ OBP - AVG

    name_key = player_name.strip().lower()
    match = lineups_df[lineups_df["player_name"].str.lower().str.strip() == name_key]
    if match.empty:
        return (LEAGUE_OBP - LEAGUE_AVG) * 4.0

    player_id = match.iloc[0]["player_id"]
    bvp = bvp_df[bvp_df["batter_id"] == player_id]
    if bvp.empty:
        return (LEAGUE_OBP - LEAGUE_AVG) * 4.0

    walks = _safe_float(bvp.iloc[0].get("walks")) or 0
    pa    = _safe_float(bvp.iloc[0].get("pa"))    or 1
    bb_rate = walks / pa
    weight  = min(pa / 50.0, 1.0)
    reg_bb  = weight * bb_rate + (1 - weight) * ((LEAGUE_OBP - LEAGUE_AVG))
    return round(reg_bb * 4.0, 4)


def project_strikeouts(pitcher_name: str, pgl_df: pd.DataFrame) -> Optional[float]:
    """Project pitcher Ks using last 5 starts."""
    if pgl_df.empty:
        return LEAGUE_K_PER_9 / 9.0 * 6.0  # ~6 IP per start

    key = pitcher_name.strip().lower()
    recent = pgl_df[pgl_df["pitcher_name_key"] == key].head(5)
    if recent.empty:
        return LEAGUE_K_PER_9 / 9.0 * 6.0

    total_ip = _safe_float(recent["innings_pitched"].sum()) or 0
    total_k  = _safe_float(recent["strikeouts"].sum())      or 0
    if total_ip < 1:
        return LEAGUE_K_PER_9 / 9.0 * 6.0

    k_per_9 = (total_k / total_ip) * 9.0
    weight  = min(len(recent) / 5.0, 1.0)
    reg_k9  = weight * k_per_9 + (1 - weight) * LEAGUE_K_PER_9
    avg_ip  = total_ip / len(recent)
    return round(reg_k9 / 9.0 * avg_ip, 2)


# =============================================================================
# SCORE PROPS
# =============================================================================

def score_prop(raw: Dict, lineups_df: pd.DataFrame, bvp_df: pd.DataFrame,
               pgl_df: pd.DataFrame, probables_df: pd.DataFrame,
               game_pk: Optional[int]) -> Optional[Dict]:
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

    # Average prices across books
    avg_over  = sum(over_px)  / len(over_px)
    avg_under = sum(under_px) / len(under_px) if under_px else None

    # Convert to no-vig probability
    over_prob  = american_to_prob(avg_over)
    under_prob = american_to_prob(avg_under) if avg_under is not None else None

    if over_prob is None:
        return None

    # Remove vig
    if under_prob is not None:
        total = over_prob + under_prob
        over_prob_nv  = over_prob  / total
    else:
        over_prob_nv = over_prob

    # Project
    if prop_type == "batter_hits":
        proj = project_hits(player, bvp_df, lineups_df)
    elif prop_type == "batter_total_bases":
        proj = project_total_bases(player, bvp_df, lineups_df)
    elif prop_type == "batter_home_runs":
        proj = project_home_runs(player, bvp_df, lineups_df)
    elif prop_type == "batter_walks":
        proj = project_walks(player, bvp_df, lineups_df)
    elif prop_type == "pitcher_strikeouts":
        # Need pitcher name — match against probables
        proj = project_strikeouts(player, pgl_df)
    else:
        return None

    if proj is None:
        return None

    # Edge: difference between our projection implied prob and book's no-vig prob
    # Simple edge: (proj - line) normalized
    proj_over_prob = min(max(0.5 + (proj - line) * 0.25, 0.05), 0.95)
    edge = round(proj_over_prob - over_prob_nv, 4)

    # Pick direction
    if abs(edge) < MIN_EDGE:
        pick = "PASS"
        confidence = "LOW"
    elif edge > 0:
        pick = "OVER"
        confidence = "HIGH" if edge >= 0.10 else "MED"
    else:
        pick = "UNDER"
        confidence = "HIGH" if edge <= -0.10 else "MED"

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

def upsert_props(rows: List[Dict]) -> int:
    if not rows:
        return 0

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
    games_df    = load_games_for_date(target_date)
    lineups_df  = load_lineups(games_df["game_pk"].tolist() if not games_df.empty else [])
    bvp_df      = load_bvp()
    pgl_df      = load_pitcher_game_log()
    probables_df = load_probables()

    log(f"Games: {len(games_df)}  Lineups: {len(lineups_df)}  BvP rows: {len(bvp_df)}")

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
            scored = score_prop(raw, lineups_df, bvp_df, pgl_df, probables_df, game_pk)
            if scored and scored["pick"] != "PASS":
                all_scored.append(scored)

    log(f"\nTotal raw props fetched: {total_raw}")
    log(f"Props with edge (non-PASS): {len(all_scored)}")

    # Save
    n = upsert_props(all_scored)
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
