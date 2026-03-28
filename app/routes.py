from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Optional

import requests
import pandas as pd

from scripts.mlb_engine_daily import (
    conn,
    ensure_tables,
    team_map,
    upsert_schedule,
    upsert_probables,
    build_features_for_date,
    align_to_model,
    load_models,
    save_predictions,
    implied_moneyline,
    DEFAULT_TOTAL_LINE,
    SEASON,
    GAMES_TABLE,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_TABLE    = "market_odds"
HTTP_TIMEOUT  = 15
BOOKMAKERS    = "draftkings,fanduel,betmgm,caesars"

ODDS_TEAM_MAP = {
    "Arizona Diamondbacks":  "ARI",
    "Atlanta Braves":        "ATL",
    "Baltimore Orioles":     "BAL",
    "Boston Red Sox":        "BOS",
    "Chicago Cubs":          "CHC",
    "Chicago White Sox":     "CHW",
    "Cincinnati Reds":       "CIN",
    "Cleveland Guardians":   "CLE",
    "Colorado Rockies":      "COL",
    "Detroit Tigers":        "DET",
    "Houston Astros":        "HOU",
    "Kansas City Royals":    "KC",
    "Los Angeles Angels":    "LAA",
    "Los Angeles Dodgers":   "LAD",
    "Miami Marlins":         "MIA",
    "Milwaukee Brewers":     "MIL",
    "Minnesota Twins":       "MIN",
    "New York Mets":         "NYM",
    "New York Yankees":      "NYY",
    "Athletics":             "ATH",
    "Oakland Athletics":     "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates":    "PIT",
    "San Diego Padres":      "SD",
    "San Francisco Giants":  "SF",
    "Seattle Mariners":      "SEA",
    "St. Louis Cardinals":   "STL",
    "Tampa Bay Rays":        "TB",
    "Texas Rangers":         "TEX",
    "Toronto Blue Jays":     "TOR",
    "Washington Nationals":  "WSH",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        if val is None:
            return None
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else int(f)
    except (TypeError, ValueError):
        return None


def norm_team(name: str) -> str:
    return ODDS_TEAM_MAP.get(name, name)


def american_to_prob(american: float) -> Optional[float]:
    try:
        if american > 0:
            return round(100 / (american + 100), 4)
        else:
            return round(-american / (-american + 100), 4)
    except Exception:
        return None


def prob_to_american(prob: float) -> Optional[int]:
    if prob is None or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-(prob / (1 - prob)) * 100))
    return int(round(((1 - prob) / prob) * 100))


def get_col(row: pd.Series, col: str, default=None):
    """Safely get a value from a pandas Series row."""
    try:
        val = row[col] if col in row.index else default
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return val
    except Exception:
        return default


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def ensure_odds_table(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {ODDS_TABLE} (
            game_pk           BIGINT,
            odds_game_id      TEXT,
            official_date     DATE NOT NULL,
            home_team         TEXT NOT NULL,
            away_team         TEXT NOT NULL,
            market_home_ml    INT,
            market_away_ml    INT,
            market_home_prob  DOUBLE PRECISION,
            market_away_prob  DOUBLE PRECISION,
            market_total_line DOUBLE PRECISION,
            best_home_ml      INT,
            best_away_ml      INT,
            bookmakers_used   INT,
            fetched_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (home_team, away_team, official_date)
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_odds_date ON {ODDS_TABLE}(official_date);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_odds_game ON {ODDS_TABLE}(game_pk);")


def ensure_predictions_odds_columns(cur):
    cols = [
        ("market_home_prob",  "DOUBLE PRECISION"),
        ("market_away_prob",  "DOUBLE PRECISION"),
        ("market_total_line", "DOUBLE PRECISION"),
        ("market_home_ml",    "INT"),
        ("market_away_ml",    "INT"),
        ("best_home_ml",      "INT"),
        ("best_away_ml",      "INT"),
        ("model_edge",        "DOUBLE PRECISION"),
    ]
    for col, dtype in cols:
        cur.execute(f"ALTER TABLE predictions ADD COLUMN IF NOT EXISTS {col} {dtype};")


def link_odds_to_games(cur, target_date):
    cur.execute(f"""
        UPDATE {ODDS_TABLE} mo
        SET game_pk = g.game_pk
        FROM {GAMES_TABLE} g
        WHERE mo.official_date = g.official_date
          AND mo.home_team = g.home_team
          AND mo.away_team = g.away_team
          AND mo.official_date = %s
    """, (target_date,))


def upsert_market_odds(cur, odds_rows: list) -> int:
    if not odds_rows:
        return 0
    for row in odds_rows:
        cur.execute(f"""
            INSERT INTO {ODDS_TABLE} (
                odds_game_id, official_date, home_team, away_team,
                market_home_ml, market_away_ml,
                market_home_prob, market_away_prob,
                market_total_line, best_home_ml, best_away_ml,
                bookmakers_used, fetched_at, updated_at
            ) VALUES (
                %(odds_game_id)s, %(official_date)s, %(home_team)s, %(away_team)s,
                %(market_home_ml)s, %(market_away_ml)s,
                %(market_home_prob)s, %(market_away_prob)s,
                %(market_total_line)s, %(best_home_ml)s, %(best_away_ml)s,
                %(bookmakers_used)s, NOW(), NOW()
            )
            ON CONFLICT (home_team, away_team, official_date) DO UPDATE SET
                market_home_ml    = EXCLUDED.market_home_ml,
                market_away_ml    = EXCLUDED.market_away_ml,
                market_home_prob  = EXCLUDED.market_home_prob,
                market_away_prob  = EXCLUDED.market_away_prob,
                market_total_line = EXCLUDED.market_total_line,
                best_home_ml      = EXCLUDED.best_home_ml,
                best_away_ml      = EXCLUDED.best_away_ml,
                bookmakers_used   = EXCLUDED.bookmakers_used,
                updated_at        = NOW();
        """, row)
    return len(odds_rows)


# ---------------------------------------------------------------------------
# Fetch from The Odds API
# ---------------------------------------------------------------------------

def fetch_market_odds(target_date) -> list:
    if not ODDS_API_KEY:
        print("  ⚠️  ODDS_API_KEY not set — skipping")
        return []

    url = f"{ODDS_API_BASE}/sports/baseball_mlb/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "h2h,totals",
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        remaining = r.headers.get("x-requests-remaining", "?")
        print(f"  Odds API: {len(data)} games | {remaining} requests remaining")
    except Exception as e:
        print(f"  ⚠️  Odds API failed: {e}")
        return []

    results = []
    for game in data:
        home_team    = norm_team(game.get("home_team", ""))
        away_team    = norm_team(game.get("away_team", ""))
        odds_game_id = game.get("id", "")

        home_ml_prices = []
        away_ml_prices = []
        total_lines    = []

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    for outcome in market.get("outcomes", []):
                        name  = norm_team(outcome.get("name", ""))
                        price = outcome.get("price")
                        if price is None:
                            continue
                        if name == home_team:
                            home_ml_prices.append(float(price))
                        elif name == away_team:
                            away_ml_prices.append(float(price))
                elif market["key"] == "totals":
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point")
                        if point is not None:
                            total_lines.append(float(point))

        if not home_ml_prices or not away_ml_prices:
            print(f"  ⚠️  No prices for {away_team} @ {home_team} — skipping")
            continue

        home_probs = [p for p in [american_to_prob(ml) for ml in home_ml_prices] if p]
        away_probs = [p for p in [american_to_prob(ml) for ml in away_ml_prices] if p]

        if not home_probs or not away_probs:
            continue

        avg_home_prob = sum(home_probs) / len(home_probs)
        avg_away_prob = sum(away_probs) / len(away_probs)

        # Remove vig — normalize to 100%
        total_prob = avg_home_prob + avg_away_prob
        if total_prob > 0:
            avg_home_prob = round(avg_home_prob / total_prob, 4)
            avg_away_prob = round(avg_away_prob / total_prob, 4)

        avg_total    = round(sum(total_lines) / len(total_lines), 1) if total_lines else None
        best_home_ml = int(max(home_ml_prices))
        best_away_ml = int(max(away_ml_prices))

        print(f"  {away_team} @ {home_team}: home {avg_home_prob*100:.1f}% | total {avg_total}")

        results.append({
            "odds_game_id":      odds_game_id,
            "official_date":     target_date,
            "home_team":         home_team,
            "away_team":         away_team,
            "market_home_ml":    prob_to_american(avg_home_prob),
            "market_away_ml":    prob_to_american(avg_away_prob),
            "market_home_prob":  avg_home_prob,
            "market_away_prob":  avg_away_prob,
            "market_total_line": avg_total,
            "best_home_ml":      best_home_ml,
            "best_away_ml":      best_away_ml,
            "bookmakers_used":   len(game.get("bookmakers", [])),
        })

    return results


# ---------------------------------------------------------------------------
# Merge odds into features DataFrame
# ---------------------------------------------------------------------------

def merge_market_odds(base: pd.DataFrame, cur, target_date) -> pd.DataFrame:
    cur.execute(f"""
        SELECT home_team, away_team,
               market_home_prob, market_away_prob,
               market_total_line, market_home_ml, market_away_ml,
               best_home_ml, best_away_ml
        FROM {ODDS_TABLE}
        WHERE official_date = %s
    """, (target_date,))
    rows = cur.fetchall()

    odds_cols = [
        "market_home_prob", "market_away_prob", "market_total_line",
        "market_home_ml", "market_away_ml", "best_home_ml", "best_away_ml",
    ]

    if not rows:
        for col in odds_cols:
            base[col] = None
        return base

    odds_df = pd.DataFrame(rows, columns=["home_team", "away_team"] + odds_cols)
    base    = base.merge(odds_df, on=["home_team", "away_team"], how="left")
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    today = datetime.now(timezone.utc).date()

    print("=" * 55)
    print(f"  load_odds.py  |  {today}")
    print("=" * 55)

    win_model, run_model, total_model = load_models()

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_tables(cur)
            ensure_odds_table(cur)
            ensure_predictions_odds_columns(cur)
            c.commit()

            # 1) Schedule + probables
            tmap    = team_map(SEASON)
            n_sched = upsert_schedule(cur, today, today, tmap)
            n_prob  = upsert_probables(cur, today, today)

            # 2) Vegas lines
            print("\nFetching market odds...")
            odds_rows = fetch_market_odds(today)
            n_odds    = upsert_market_odds(cur, odds_rows)
            link_odds_to_games(cur, today)
            c.commit()
            print(f"  Market odds saved: {n_odds} games\n")

            # 3) Model features
            base = build_features_for_date(cur, today)
            if base.empty:
                c.commit()
                print(f"No games found for {today}.")
                return

            # 4) Merge odds into features
            base = merge_market_odds(base, cur, today)

            # 5) Predict
            X_win = align_to_model(base, win_model)
            X_run = align_to_model(base, run_model)
            X_tot = align_to_model(base, total_model)

            if hasattr(win_model, "predict_proba"):
                p             = win_model.predict_proba(X_win)
                home_prob_raw = p[:, 1] if p.shape[1] > 1 else p[:, 0]
            else:
                home_prob_raw = win_model.predict(X_win)

            home_prob  = [_safe_float(x) or 0.5 for x in home_prob_raw]
            away_prob  = [round(1.0 - hp, 6)     for hp in home_prob]
            prediction = [1 if hp >= 0.5 else 0  for hp in home_prob]
            run_diff   = [_safe_float(x) or 0.0  for x in run_model.predict(X_run)]
            total_runs = [_safe_float(x) or 0.0  for x in total_model.predict(X_tot)]

            # Use Vegas total line for O/U when available
            ou_lines = []
            for i in range(len(base)):
                row = base.iloc[i]
                mkt_line = get_col(row, "market_total_line")
                ou_lines.append(float(mkt_line) if mkt_line is not None else DEFAULT_TOTAL_LINE)

            ml_pick      = ["HOME" if hp >= 0.5      else "AWAY"      for hp in home_prob]
            runline_pick = ["HOME -1.5" if rd >= 1.5 else "AWAY +1.5" for rd in run_diff]
            ou_pick      = ["OVER" if tr >= ou_lines[i] else "UNDER"  for i, tr in enumerate(total_runs)]

            # Build output DataFrame with all market columns
            out_data = {
                "game_pk":          base["game_pk"].astype(int),
                "official_date":    pd.to_datetime(base["official_date"]).dt.date,
                "away_team":        base["away_team"].astype(str),
                "home_team":        base["home_team"].astype(str),
                "prediction":       prediction,
                "win_probability":  home_prob,
                "home_win_prob":    home_prob,
                "away_win_prob":    away_prob,
                "home_ml_implied":  [implied_moneyline(x) for x in home_prob],
                "away_ml_implied":  [implied_moneyline(x) for x in away_prob],
                "run_diff_pred":    run_diff,
                "total_runs_pred":  total_runs,
                "ml_pick":          ml_pick,
                "runline_pick":     runline_pick,
                "ou_pick":          ou_pick,
            }

            # Add market columns row by row using safe accessor
            mkt_home_prob  = []
            mkt_away_prob  = []
            mkt_total_line = []
            mkt_home_ml    = []
            mkt_away_ml    = []
            best_home_ml   = []
            best_away_ml   = []
            model_edge     = []

            for i in range(len(base)):
                row  = base.iloc[i]
                mhp  = _safe_float(get_col(row, "market_home_prob"))
                map_ = _safe_float(get_col(row, "market_away_prob"))
                mtl  = _safe_float(get_col(row, "market_total_line"))
                mhml = _safe_int(get_col(row, "market_home_ml"))
                maml = _safe_int(get_col(row, "market_away_ml"))
                bhml = _safe_int(get_col(row, "best_home_ml"))
                baml = _safe_int(get_col(row, "best_away_ml"))
                edge = round(home_prob[i] - mhp, 4) if mhp is not None else None

                mkt_home_prob.append(mhp)
                mkt_away_prob.append(map_)
                mkt_total_line.append(mtl)
                mkt_home_ml.append(mhml)
                mkt_away_ml.append(maml)
                best_home_ml.append(bhml)
                best_away_ml.append(baml)
                model_edge.append(edge)

            out_data["market_home_prob"]  = mkt_home_prob
            out_data["market_away_prob"]  = mkt_away_prob
            out_data["market_total_line"] = mkt_total_line
            out_data["market_home_ml"]    = mkt_home_ml
            out_data["market_away_ml"]    = mkt_away_ml
            out_data["best_home_ml"]      = best_home_ml
            out_data["best_away_ml"]      = best_away_ml
            out_data["model_edge"]        = model_edge

            out     = pd.DataFrame(out_data)
            n_saved = save_predictions(cur, out)
            c.commit()

            # Summary
            print(f"{'='*55}")
            print(f"  PREDICTIONS FOR {today}")
            print(f"{'='*55}")
            for _, row in out.iterrows():
                edge    = row.get("model_edge")
                mkt_ml  = row.get("market_home_ml")
                edge_str = f"  edge: {round(edge*100,1)}%" if edge is not None else ""
                mkt_str  = f"  mkt: {mkt_ml}"             if mkt_ml is not None else "  mkt: —"
                print(f"  {row['away_team']} @ {row['home_team']}")
                print(f"    ML: {row['ml_pick']} | model: {round(row['home_win_prob']*100,1)}%"
                      f"{mkt_str}{edge_str}")
                print(f"    RL: {row['runline_pick']} | O/U: {row['ou_pick']} "
                      f"(line: {row.get('market_total_line') or DEFAULT_TOTAL_LINE})")
                print()

            print(f"Schedule updated:   {n_sched}")
            print(f"Probables updated:  {n_prob}")
            print(f"Market odds saved:  {n_odds}")
            print(f"Predictions saved:  {n_saved}")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()
