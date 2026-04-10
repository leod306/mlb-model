from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
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

MLB_SEASON = int(os.getenv("MLB_SEASON", "2026"))

GAMES_TABLE = os.getenv("MLB_GAMES_TABLE", "games")
PREDICTIONS_TABLE = os.getenv("MLB_PREDICTIONS_TABLE", "predictions")
ODDS_TABLE = os.getenv("MLB_ODDS_TABLE", "market_odds")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
HTTP_TIMEOUT = 20

BOOKMAKERS = "draftkings,fanduel,betmgm,caesars"


# =============================================================================
# TEAM NAME MAP
# =============================================================================

ODDS_TEAM_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


# =============================================================================
# HELPERS
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def table_exists(table_name: str) -> bool:
    sql = """
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = :table_name
    )
    """
    with engine.begin() as conn:
        return bool(conn.execute(text(sql), {"table_name": table_name}).scalar())


def get_table_columns(table_name: str) -> list[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = :table_name
    ORDER BY ordinal_position
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"table_name": table_name}).fetchall()
    return [r[0] for r in rows]


def _safe_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def norm_team(name: str) -> str:
    return ODDS_TEAM_MAP.get(name, name)


def american_to_prob(american: float) -> Optional[float]:
    try:
        american = float(american)
        if american > 0:
            return 100.0 / (american + 100.0)
        return (-american) / ((-american) + 100.0)
    except Exception:
        return None


def prob_to_american(prob: float) -> Optional[int]:
    prob = _safe_float(prob)
    if prob is None or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-(prob / (1.0 - prob)) * 100))
    return int(round(((1.0 - prob) / prob) * 100))


def best_bettor_price(prices: list[float]) -> Optional[int]:
    """
    Picks the most favorable American odds for a bettor:
    - for positive prices: bigger is better (+140 > +120)
    - for negative prices: less negative is better (-110 > -140)
    """
    if not prices:
        return None
    return int(max(prices))


# =============================================================================
# DB SETUP
# =============================================================================

def ensure_odds_table() -> None:
    sql = f"""
    CREATE TABLE IF NOT EXISTS {ODDS_TABLE} (
        game_pk             BIGINT,
        odds_game_id        TEXT,
        official_date       DATE NOT NULL,
        home_team           TEXT NOT NULL,
        away_team           TEXT NOT NULL,

        market_home_ml      INT,
        market_away_ml      INT,
        market_home_prob    DOUBLE PRECISION,
        market_away_prob    DOUBLE PRECISION,

        market_total_line   DOUBLE PRECISION,
        market_ou_direction TEXT,

        best_home_ml        INT,
        best_away_ml        INT,

        bookmakers_used     INT,
        fetched_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        PRIMARY KEY (home_team, away_team, official_date)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(sql))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{ODDS_TABLE}_date ON {ODDS_TABLE}(official_date);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{ODDS_TABLE}_gamepk ON {ODDS_TABLE}(game_pk);"))


def ensure_predictions_odds_columns() -> None:
    if not table_exists(PREDICTIONS_TABLE):
        return

    wanted = {
        "market_home_prob": "DOUBLE PRECISION",
        "market_away_prob": "DOUBLE PRECISION",
        "market_total_line": "DOUBLE PRECISION",
        "market_home_ml": "INT",
        "market_away_ml": "INT",
        "best_home_ml": "INT",
        "best_away_ml": "INT",
        "model_edge": "DOUBLE PRECISION",
    }

    existing = set(get_table_columns(PREDICTIONS_TABLE))
    with engine.begin() as conn:
        for col, dtype in wanted.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE {PREDICTIONS_TABLE} ADD COLUMN {col} {dtype}"))


# =============================================================================
# TARGET SLATE
# =============================================================================

def get_target_date() -> Optional[datetime.date]:
    if not table_exists(GAMES_TABLE):
        raise RuntimeError(f"Missing required table: {GAMES_TABLE}")

    sql = f"""
    SELECT MIN(official_date) AS official_date
    FROM {GAMES_TABLE}
    WHERE official_date >= CURRENT_DATE
      AND season = :season
    """
    with engine.begin() as conn:
        row = conn.execute(text(sql), {"season": MLB_SEASON}).fetchone()

    if not row or not row[0]:
        return None
    return row[0]


# =============================================================================
# FETCH ODDS
# =============================================================================

def fetch_market_odds(target_date) -> list[dict]:
    if not ODDS_API_KEY:
        log("ODDS_API_KEY not set. Skipping odds fetch.")
        return []

    url = f"{ODDS_API_BASE}/sports/baseball_mlb/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,totals",
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        remaining = r.headers.get("x-requests-remaining", "?")
        used = r.headers.get("x-requests-used", "?")
        log(f"Odds API fetched {len(data)} games | remaining={remaining} used={used}")
    except Exception as e:
        log(f"Odds API fetch failed: {e}")
        return []

    results: list[dict] = []

    for game in data:
        home_team = norm_team(game.get("home_team", ""))
        away_team = norm_team(game.get("away_team", ""))
        odds_game_id = game.get("id", "")

        home_ml_prices: list[float] = []
        away_ml_prices: list[float] = []
        total_lines: list[float] = []

        commence = game.get("commence_time")
        official_date = target_date
        if commence:
            try:
                dt = pd.to_datetime(commence, utc=True)
                official_date = dt.date()
            except Exception:
                official_date = target_date

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = market.get("key")

                if key == "h2h":
                    for outcome in market.get("outcomes", []):
                        name = norm_team(outcome.get("name", ""))
                        price = outcome.get("price")
                        if price is None:
                            continue
                        price = float(price)

                        if name == home_team:
                            home_ml_prices.append(price)
                        elif name == away_team:
                            away_ml_prices.append(price)

                elif key == "totals":
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point")
                        if point is not None:
                            total_lines.append(float(point))

        # keep only games for the target slate date
        if official_date != target_date:
            continue

        if not home_ml_prices or not away_ml_prices:
            continue

        home_probs = [p for p in (american_to_prob(x) for x in home_ml_prices) if p is not None]
        away_probs = [p for p in (american_to_prob(x) for x in away_ml_prices) if p is not None]

        if not home_probs or not away_probs:
            continue

        avg_home_prob = sum(home_probs) / len(home_probs)
        avg_away_prob = sum(away_probs) / len(away_probs)

        # vig removal normalization
        total_prob = avg_home_prob + avg_away_prob
        if total_prob > 0:
            avg_home_prob /= total_prob
            avg_away_prob /= total_prob

        avg_home_prob = round(avg_home_prob, 4)
        avg_away_prob = round(avg_away_prob, 4)

        avg_total = round(sum(total_lines) / len(total_lines), 1) if total_lines else None

        results.append({
            "odds_game_id": odds_game_id,
            "official_date": official_date,
            "home_team": home_team,
            "away_team": away_team,
            "market_home_ml": prob_to_american(avg_home_prob),
            "market_away_ml": prob_to_american(avg_away_prob),
            "market_home_prob": avg_home_prob,
            "market_away_prob": avg_away_prob,
            "market_total_line": avg_total,
            "market_ou_direction": None,
            "best_home_ml": best_bettor_price(home_ml_prices),
            "best_away_ml": best_bettor_price(away_ml_prices),
            "bookmakers_used": len(game.get("bookmakers", [])),
        })

    return results


# =============================================================================
# UPSERT / LINK
# =============================================================================

def upsert_market_odds(odds_rows: list[dict]) -> int:
    if not odds_rows:
        return 0

    sql = f"""
    INSERT INTO {ODDS_TABLE} (
        odds_game_id,
        official_date,
        home_team,
        away_team,
        market_home_ml,
        market_away_ml,
        market_home_prob,
        market_away_prob,
        market_total_line,
        market_ou_direction,
        best_home_ml,
        best_away_ml,
        bookmakers_used,
        fetched_at,
        updated_at
    )
    VALUES (
        :odds_game_id,
        :official_date,
        :home_team,
        :away_team,
        :market_home_ml,
        :market_away_ml,
        :market_home_prob,
        :market_away_prob,
        :market_total_line,
        :market_ou_direction,
        :best_home_ml,
        :best_away_ml,
        :bookmakers_used,
        NOW(),
        NOW()
    )
    ON CONFLICT (home_team, away_team, official_date) DO UPDATE SET
        odds_game_id       = EXCLUDED.odds_game_id,
        market_home_ml     = EXCLUDED.market_home_ml,
        market_away_ml     = EXCLUDED.market_away_ml,
        market_home_prob   = EXCLUDED.market_home_prob,
        market_away_prob   = EXCLUDED.market_away_prob,
        market_total_line  = EXCLUDED.market_total_line,
        market_ou_direction= EXCLUDED.market_ou_direction,
        best_home_ml       = EXCLUDED.best_home_ml,
        best_away_ml       = EXCLUDED.best_away_ml,
        bookmakers_used    = EXCLUDED.bookmakers_used,
        updated_at         = NOW()
    """
    with engine.begin() as conn:
        conn.execute(text(sql), odds_rows)

    return len(odds_rows)


def link_odds_to_games(target_date) -> None:
    sql = f"""
    UPDATE {ODDS_TABLE} mo
    SET game_pk = g.game_pk
    FROM {GAMES_TABLE} g
    WHERE mo.official_date = g.official_date
      AND mo.home_team = g.home_team
      AND mo.away_team = g.away_team
      AND mo.official_date = :target_date
    """
    with engine.begin() as conn:
        conn.execute(text(sql), {"target_date": target_date})


def update_predictions_with_odds(target_date) -> int:
    """
    If predictions already exist for the slate, enrich them with market odds.
    Safe to run even if predictions are not there yet.
    """
    if not table_exists(PREDICTIONS_TABLE):
        return 0

    sql = f"""
    UPDATE {PREDICTIONS_TABLE} p
    SET
        market_home_prob = mo.market_home_prob,
        market_away_prob = mo.market_away_prob,
        market_total_line = mo.market_total_line,
        market_home_ml = mo.market_home_ml,
        market_away_ml = mo.market_away_ml,
        best_home_ml = mo.best_home_ml,
        best_away_ml = mo.best_away_ml,
        model_edge = CASE
            WHEN p.home_win_prob IS NOT NULL AND mo.market_home_prob IS NOT NULL
            THEN ROUND((p.home_win_prob - mo.market_home_prob)::numeric, 4)
            ELSE NULL
        END
    FROM {ODDS_TABLE} mo
    WHERE p.official_date = mo.official_date
      AND p.home_team = mo.home_team
      AND p.away_team = mo.away_team
      AND p.official_date = :target_date
    """
    with engine.begin() as conn:
        result = conn.execute(text(sql), {"target_date": target_date})
    return result.rowcount if result.rowcount is not None else 0


def show_preview(target_date) -> None:
    if not table_exists(ODDS_TABLE):
        return

    sql = f"""
    SELECT
        away_team,
        home_team,
        market_home_ml,
        market_away_ml,
        market_total_line,
        best_home_ml,
        best_away_ml
    FROM {ODDS_TABLE}
    WHERE official_date = :target_date
    ORDER BY away_team, home_team
    """
    df = pd.read_sql(text(sql), engine, params={"target_date": target_date})
    if df.empty:
        log("No odds rows saved for this slate.")
        return

    log("")
    log("=" * 60)
    log(f"MARKET ODDS PREVIEW FOR {target_date}")
    log("=" * 60)
    log(df.to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    now_utc = datetime.now(timezone.utc)
    log("=" * 60)
    log(f"load_odds.py | {now_utc.date()}")
    log("=" * 60)

    ensure_odds_table()
    ensure_predictions_odds_columns()

    target_date = get_target_date()
    if target_date is None:
        log("No upcoming slate found in games table.")
        return

    log(f"Target slate date: {target_date}")

    odds_rows = fetch_market_odds(target_date)
    if not odds_rows:
        log("No odds fetched for target slate.")
        return

    n_saved = upsert_market_odds(odds_rows)
    link_odds_to_games(target_date)
    n_updated = update_predictions_with_odds(target_date)

    log(f"Market odds saved: {n_saved}")
    log(f"Predictions updated with odds: {n_updated}")

    show_preview(target_date)


if __name__ == "__main__":
    main()