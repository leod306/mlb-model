from __future__ import annotations

import os
import math
from pathlib import Path
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
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

MLB_SEASON         = int(os.getenv("MLB_SEASON", "2026"))
WINDOW_DAYS        = int(os.getenv("WINDOW_DAYS", "30"))

GAMES_TABLE        = os.getenv("MLB_GAMES_TABLE",       "games")
PROBABLES_TABLE    = os.getenv("MLB_PROBABLES_TABLE",   "game_probables")
PREDICTIONS_TABLE  = os.getenv("MLB_PREDICTIONS_TABLE", "predictions")
FEATURES_TABLE     = os.getenv("MLB_FEATURES_TABLE",    "game_features")
ODDS_TABLE         = os.getenv("MLB_ODDS_TABLE",        "market_odds")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "ml" / "mlb_model.pkl"
MODEL_PATH         = Path(os.getenv("MLB_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

BLEND_CUTOFF_GAMES = 30
ROLLING_WINDOW     = 10
OU_WINDOW          = 20
ATS_WINDOW         = 20
DEFAULT_TOTAL_LINE = float(os.getenv("DEFAULT_TOTAL_LINE", "8.5"))

# -----------------------------------------------------------------------------
# BETTING LAYER CONFIG
#
# The old approach used the ensemble's seed-to-seed spread as a "confidence
# interval" and hand-tuned fudge constants (TOTAL_CALIBRATION=-1.2). Both are
# gone. Uncertainty now comes from the model bundle's out-of-fold residual
# sigmas, bias correction comes from the measured total_bias, and every pick
# is an edge-vs-price decision.
# -----------------------------------------------------------------------------

# Legacy env overrides — default 0.0 now. The retrained bundle carries a
# measured total_bias which is applied automatically. Only set these env vars
# if you're running an OLD model bundle that lacks sigma/bias fields.
TOTAL_CALIBRATION    = float(os.getenv("TOTAL_CALIBRATION",    "0.0"))
HOME_BIAS_CORRECTION = float(os.getenv("HOME_BIAS_CORRECTION", "0.0"))

# Fallback uncertainty if the bundle predates sigma fields.
# ~4.0 runs is the empirical std dev of an MLB game total; ~3.9 for margin.
FALLBACK_SIGMA_TOTAL = float(os.getenv("FALLBACK_SIGMA_TOTAL", "4.0"))
FALLBACK_SIGMA_RD    = float(os.getenv("FALLBACK_SIGMA_RD",    "3.9"))

# Assumed juice on totals when the odds feed has the line but not the price.
OU_ASSUMED_PRICE = int(os.getenv("OU_ASSUMED_PRICE", "-110"))   # breakeven .5238

# Minimum edges before anything becomes a pick. These are probability-point
# edges over the de-vigged market / breakeven price. 2.5-4% is a sane band:
# lower and you're betting noise, higher and you'll almost never bet.
MIN_ML_EDGE = float(os.getenv("MIN_ML_EDGE", "0.03"))    # model prob vs novig market prob
MIN_OU_EDGE = float(os.getenv("MIN_OU_EDGE", "0.03"))    # P(side) vs breakeven at OU_ASSUMED_PRICE

# Run line thresholds. We do NOT have run line prices in market_odds, so these
# picks are advisory only and never ranked in top plays. Typical prices:
#   favorite -1.5 pays ~+120..+140  (breakeven ~.42-.45)
#   underdog +1.5 costs ~-180..-250 (breakeven ~.64-.71)
# Thresholds below are conservative vs those typical prices.
# TODO: add RL prices to the odds loader — then these become edge-vs-price too.
RL_FAV_MIN_COVER_PROB = float(os.getenv("RL_FAV_MIN_COVER_PROB", "0.50"))
RL_DOG_MIN_COVER_PROB = float(os.getenv("RL_DOG_MIN_COVER_PROB", "0.74"))

# Elo constants (must match retrain.py)
ELO_START    = 1500.0
ELO_K        = 20.0
ELO_HOME_ADV = 35.0

# Ballpark run factors (must match retrain.py)
PARK_FACTORS: Dict[str, float] = {
    "COL": 1.18, "CIN": 1.10, "TEX": 1.07, "BOS": 1.06, "PHI": 1.05,
    "MIL": 1.04, "CHC": 1.04, "HOU": 1.03, "ATL": 1.02, "NYY": 1.02,
    "ARI": 1.01, "LAD": 1.00, "NYM": 1.00, "STL": 1.00, "DET": 0.99,
    "BAL": 0.99, "CWS": 0.99, "TOR": 0.98, "MIA": 0.98, "MIN": 0.98,
    "CLE": 0.97, "KC":  0.97, "TB":  0.97, "WSH": 0.97, "SF":  0.96,
    "SEA": 0.96, "SD":  0.96, "PIT": 0.95, "LAA": 0.95, "ATH": 0.95,
}

# Batting order weights — leadoff matters most, drops toward bottom
BATTING_ORDER_WEIGHTS = {1: 1.0, 2: 0.95, 3: 0.9, 4: 0.85, 5: 0.75,
                         6: 0.65, 7: 0.55, 8: 0.45, 9: 0.35}

# League average OPS fallback when no BvP data
LEAGUE_AVG_OPS = 0.720


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def table_exists(table_name: str) -> bool:
    sql = """
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = :table_name
    )
    """
    with engine.begin() as conn:
        return bool(conn.execute(text(sql), {"table_name": table_name}).scalar())


def get_table_columns(table_name: str) -> List[str]:
    sql = """
    SELECT column_name FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = :table_name
    ORDER BY ordinal_position
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), {"table_name": table_name}).fetchall()
    return [r[0] for r in rows]


def coerce_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def safe_moneyline_from_prob(prob: Optional[float]) -> Optional[int]:
    if prob is None:
        return None
    try:
        p = float(prob)
    except Exception:
        return None
    if not math.isfinite(p):
        return None
    p = max(min(p, 0.999), 0.001)
    if p >= 0.5:
        return int(round(-(p / (1.0 - p)) * 100))
    return int(round(((1.0 - p) / p) * 100))


def normalize_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


# =============================================================================
# BETTING MATH
# All picks are edge-vs-price decisions built on these four functions.
# =============================================================================

def american_to_prob(ml: Any) -> Optional[float]:
    """American odds → implied probability (includes vig)."""
    m = coerce_float(ml)
    if m is None or m == 0:
        return None
    if m < 0:
        return -m / (-m + 100.0)
    return 100.0 / (m + 100.0)


def american_payout(ml: Any) -> Optional[float]:
    """Profit per $1 staked at American odds ml (excluding stake)."""
    m = coerce_float(ml)
    if m is None or m == 0:
        return None
    return (100.0 / -m) if m < 0 else (m / 100.0)


def devig_two_way(p_a: Optional[float], p_b: Optional[float]) -> Optional[float]:
    """Remove vig from a two-way market. Returns fair probability of side A."""
    if p_a is None or p_b is None:
        return None
    denom = p_a + p_b
    if denom <= 0:
        return None
    return p_a / denom


def ev_per_dollar(p: float, ml: Any) -> Optional[float]:
    """Expected profit per $1 staked given win probability p and price ml."""
    payout = american_payout(ml)
    if payout is None:
        return None
    return p * payout - (1.0 - p)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard-library normal CDF — avoids a scipy dependency."""
    if sigma <= 0:
        return 0.5
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def prob_over(pred_total: float, line: float, sigma: float) -> float:
    """
    P(actual total > line) treating the outcome as Normal(pred_total, sigma).
    sigma must be the OUT-OF-FOLD residual sigma from retrain.py — the true
    predictive uncertainty — never the ensemble's seed-to-seed spread.
    MLB totals can't push on .5 lines, so > vs >= doesn't matter there;
    on whole-number lines pushes exist but the normal approx absorbs it.
    """
    return 1.0 - normal_cdf(line, mu=pred_total, sigma=sigma)


def prob_home_covers(rd_pred: float, spread: float, sigma_rd: float) -> float:
    """P(home margin > spread) with margin ~ Normal(rd_pred, sigma_rd)."""
    return 1.0 - normal_cdf(spread, mu=rd_pred, sigma=sigma_rd)


# =============================================================================
# DB SETUP
# =============================================================================

def ensure_predictions_table() -> None:
    sql = f"""
    CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
        game_pk BIGINT PRIMARY KEY,
        official_date DATE,
        away_team TEXT, home_team TEXT,
        away_team_id INT, home_team_id INT,
        away_sp_name TEXT, home_sp_name TEXT,
        era_diff DOUBLE PRECISION, whip_diff DOUBLE PRECISION,
        home_sp_rest_days DOUBLE PRECISION, away_sp_rest_days DOUBLE PRECISION,
        home_bullpen_ip_4d DOUBLE PRECISION, away_bullpen_ip_4d DOUBLE PRECISION,
        home_win_pct_home DOUBLE PRECISION, away_win_pct_away DOUBLE PRECISION,
        home_last10_runs_scored DOUBLE PRECISION, away_last10_runs_scored DOUBLE PRECISION,
        home_last10_runs_allowed DOUBLE PRECISION, away_last10_runs_allowed DOUBLE PRECISION,
        home_last10_run_diff DOUBLE PRECISION, away_last10_run_diff DOUBLE PRECISION,
        sp_fip_diff DOUBLE PRECISION, offense_wrc_diff DOUBLE PRECISION,
        home_wrc_plus DOUBLE PRECISION, away_wrc_plus DOUBLE PRECISION,
        bullpen_fip_diff DOUBLE PRECISION,
        park_run_factor DOUBLE PRECISION,
        temperature_f DOUBLE PRECISION, wind_speed_mph DOUBLE PRECISION,
        run_diff_form_diff DOUBLE PRECISION,
        runs_scored_diff DOUBLE PRECISION,
        runs_allowed_diff DOUBLE PRECISION,
        sp_rest_diff DOUBLE PRECISION,
        bullpen_usage_diff DOUBLE PRECISION,
        win_pct_diff DOUBLE PRECISION,
        home_ou_over_rate DOUBLE PRECISION,
        away_ou_over_rate DOUBLE PRECISION,
        home_last_game_total DOUBLE PRECISION,
        away_last_game_total DOUBLE PRECISION,
        home_ats_cover_rate DOUBLE PRECISION,
        away_ats_cover_rate DOUBLE PRECISION,
        home_lineup_ops_vs_sp DOUBLE PRECISION,
        away_lineup_ops_vs_sp DOUBLE PRECISION,
        lineup_ops_diff DOUBLE PRECISION,
        home_lineup_hard_hit DOUBLE PRECISION,
        away_lineup_hard_hit DOUBLE PRECISION,
        home_win_prob DOUBLE PRECISION, away_win_prob DOUBLE PRECISION,
        home_win_prob_lo DOUBLE PRECISION, home_win_prob_hi DOUBLE PRECISION,
        home_win_prob_std DOUBLE PRECISION,
        home_ml_implied INT, away_ml_implied INT,
        run_diff_pred DOUBLE PRECISION, run_diff_lo DOUBLE PRECISION,
        run_diff_hi DOUBLE PRECISION, run_diff_std DOUBLE PRECISION,
        total_runs_pred DOUBLE PRECISION, total_runs_lo DOUBLE PRECISION,
        total_runs_hi DOUBLE PRECISION, total_runs_std DOUBLE PRECISION,
        ml_pick TEXT, runline_pick TEXT, ou_pick TEXT,
        play_rank INT, play_type TEXT, play_score DOUBLE PRECISION, play_detail TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """
    with engine.begin() as conn:
        conn.execute(text(sql))

    wanted = {
        "sp_fip_diff":            "DOUBLE PRECISION",
        "offense_wrc_diff":       "DOUBLE PRECISION",
        "home_wrc_plus":          "DOUBLE PRECISION",
        "away_wrc_plus":          "DOUBLE PRECISION",
        "bullpen_fip_diff":       "DOUBLE PRECISION",
        "park_run_factor":        "DOUBLE PRECISION",
        "temperature_f":          "DOUBLE PRECISION",
        "wind_speed_mph":         "DOUBLE PRECISION",
        "sp_rest_diff":           "DOUBLE PRECISION",
        "bullpen_usage_diff":     "DOUBLE PRECISION",
        "home_ou_over_rate":      "DOUBLE PRECISION",
        "away_ou_over_rate":      "DOUBLE PRECISION",
        "home_last_game_total":   "DOUBLE PRECISION",
        "away_last_game_total":   "DOUBLE PRECISION",
        "home_ats_cover_rate":    "DOUBLE PRECISION",
        "away_ats_cover_rate":    "DOUBLE PRECISION",
        "home_lineup_ops_vs_sp":  "DOUBLE PRECISION",
        "away_lineup_ops_vs_sp":  "DOUBLE PRECISION",
        "lineup_ops_diff":        "DOUBLE PRECISION",
        "home_lineup_hard_hit":   "DOUBLE PRECISION",
        "away_lineup_hard_hit":   "DOUBLE PRECISION",
        "market_home_prob":       "DOUBLE PRECISION",
        "market_away_prob":       "DOUBLE PRECISION",
        "market_total_line":      "DOUBLE PRECISION",
        "market_home_ml":         "INT",
        "market_away_ml":         "INT",
        "best_home_ml":           "INT",
        "best_away_ml":           "INT",
        "model_edge":             "DOUBLE PRECISION",
        # --- new betting-layer columns (needed for CLV + pick auditing) ---
        "market_home_prob_novig": "DOUBLE PRECISION",
        "home_win_prob_raw":      "DOUBLE PRECISION",  # pre-calibration ensemble prob
        "ml_edge":                "DOUBLE PRECISION",  # calibrated prob - novig market prob
        "p_over":                 "DOUBLE PRECISION",  # P(total > line) via sigma_total
        "ou_edge":                "DOUBLE PRECISION",  # P(picked side) - breakeven
        "rl_home_cover_prob":     "DOUBLE PRECISION",  # P(home margin > 1.5)
        "sigma_total_used":       "DOUBLE PRECISION",
        "sigma_rd_used":          "DOUBLE PRECISION",
    }
    existing = set(get_table_columns(PREDICTIONS_TABLE))
    with engine.begin() as conn:
        for col, col_type in wanted.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE {PREDICTIONS_TABLE} ADD COLUMN {col} {col_type}"))


# =============================================================================
# LOAD DATA
# =============================================================================

def load_upcoming_games(target_date: Optional[date] = None) -> pd.DataFrame:
    if not table_exists(GAMES_TABLE):
        raise RuntimeError(f"Missing required table: {GAMES_TABLE}")

    game_cols         = set(get_table_columns(GAMES_TABLE))
    has_team_ids      = "home_team_id" in game_cols and "away_team_id" in game_cols
    has_game_date_utc = "game_date_utc" in game_cols

    select_parts = ["game_pk", "official_date", "season", "home_team", "away_team"]
    select_parts += ["home_team_id, away_team_id"] if has_team_ids else ["NULL::INT AS home_team_id, NULL::INT AS away_team_id"]
    select_parts += ["game_date_utc"] if has_game_date_utc else ["NULL::TIMESTAMP AS game_date_utc"]
    select_sql = ",\n        ".join(select_parts)

    if target_date is not None:
        sql = f"""
        SELECT {select_sql} FROM {GAMES_TABLE}
        WHERE official_date = :target_date AND season = :season
        ORDER BY game_date_utc NULLS LAST, game_pk
        """
        df = pd.read_sql(text(sql), engine, params={"target_date": target_date, "season": MLB_SEASON})
        return normalize_date_col(df, "official_date")

    sql = f"""
    WITH next_date AS (
        SELECT MIN(official_date) AS official_date FROM {GAMES_TABLE}
        WHERE official_date >= CURRENT_DATE AND season = :season
    )
    SELECT {select_sql} FROM {GAMES_TABLE}
    WHERE official_date = (SELECT official_date FROM next_date) AND season = :season
    ORDER BY game_date_utc NULLS LAST, game_pk
    """
    df = pd.read_sql(text(sql), engine, params={"season": MLB_SEASON})
    return normalize_date_col(df, "official_date")


def load_probables() -> pd.DataFrame:
    if not table_exists(PROBABLES_TABLE):
        return pd.DataFrame(columns=["game_pk", "away_sp_name", "home_sp_name", "home_sp_id", "away_sp_id"])
    cols = set(get_table_columns(PROBABLES_TABLE))
    select_cols = ["game_pk", "away_sp_name", "home_sp_name"]
    if "home_sp_id" in cols: select_cols.append("home_sp_id")
    if "away_sp_id" in cols: select_cols.append("away_sp_id")
    return pd.read_sql(text(f"SELECT {', '.join(select_cols)} FROM {PROBABLES_TABLE}"), engine)


def load_completed_games() -> pd.DataFrame:
    sql = f"""
    SELECT game_pk, official_date, season, home_team, away_team, home_score, away_score
    FROM {GAMES_TABLE}
    WHERE official_date IS NOT NULL AND season IN (2024, 2025, 2026)
      AND home_score IS NOT NULL AND away_score IS NOT NULL
    ORDER BY official_date, game_pk
    """
    df = pd.read_sql(text(sql), engine)
    if df.empty:
        return df
    df = normalize_date_col(df, "official_date")
    df["season"]     = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["official_date", "season", "home_score", "away_score"]).copy()
    df["season"] = df["season"].astype(int)
    return df


def load_pitchers() -> pd.DataFrame:
    if not table_exists("pitchers"):
        return pd.DataFrame(columns=["pitcher_name", "season", "era", "whip"])
    df = pd.read_sql(text("SELECT pitcher_name, season, era, whip FROM pitchers"), engine)
    if df.empty:
        return df
    df["pitcher_name_key"] = df["pitcher_name"].astype(str).str.strip().str.lower()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["era"]    = pd.to_numeric(df["era"],    errors="coerce")
    df["whip"]   = pd.to_numeric(df["whip"],   errors="coerce")
    return df.dropna(subset=["pitcher_name_key", "season"])


def load_pitcher_game_log() -> pd.DataFrame:
    if not table_exists("pitcher_game_log"):
        return pd.DataFrame()
    df = pd.read_sql(text("""
        SELECT official_date, pitcher_name, team, role,
               innings_pitched, er_allowed, hits_allowed, walks, strikeouts
        FROM pitcher_game_log
    """), engine)
    if df.empty:
        return df
    df["official_date"]    = pd.to_datetime(df["official_date"], errors="coerce").dt.date
    df["pitcher_name_key"] = df["pitcher_name"].astype(str).str.strip().str.lower()
    df["innings_pitched"]  = pd.to_numeric(df["innings_pitched"], errors="coerce")
    df["er_allowed"]       = pd.to_numeric(df["er_allowed"],       errors="coerce")
    df["hits_allowed"]     = pd.to_numeric(df["hits_allowed"],     errors="coerce")
    df["walks"]            = pd.to_numeric(df["walks"],            errors="coerce")
    df["strikeouts"]       = pd.to_numeric(df["strikeouts"],       errors="coerce")
    return df.dropna(subset=["official_date"])


def load_game_features(game_pks: list) -> pd.DataFrame:
    if not game_pks or not table_exists(FEATURES_TABLE):
        return pd.DataFrame()
    try:
        sql = text(f"""
            SELECT game_pk,
                   home_wrc_plus,    away_wrc_plus,
                   home_sp_fip,      away_sp_fip,
                   home_bullpen_era, away_bullpen_era,
                   home_bullpen_fip, away_bullpen_fip,
                   sp_fip_diff,      bullpen_fip_diff, offense_wrc_diff,
                   park_run_factor,  park_hr_factor,
                   temperature_f,    wind_speed_mph
            FROM {FEATURES_TABLE}
            WHERE game_pk = ANY(:pks)
        """)
        with engine.begin() as conn:
            return pd.read_sql(sql, conn, params={"pks": game_pks})
    except Exception as e:
        log(f"⚠️  game_features load failed: {e}")
        return pd.DataFrame()


def load_market_odds(game_pks: list) -> pd.DataFrame:
    if not game_pks or not table_exists(ODDS_TABLE):
        return pd.DataFrame()
    try:
        sql = text(f"""
            SELECT game_pk,
                   market_home_ml, market_away_ml,
                   market_home_prob, market_away_prob,
                   market_total_line
            FROM {ODDS_TABLE}
            WHERE game_pk = ANY(:pks)
        """)
        with engine.begin() as conn:
            return pd.read_sql(sql, conn, params={"pks": game_pks})
    except Exception as e:
        log(f"⚠️  market_odds load failed: {e}")
        return pd.DataFrame()


# =============================================================================
# LINEUP QUALITY SCORE
# =============================================================================

def load_lineups_for_games(game_pks: list) -> pd.DataFrame:
    if not game_pks or not table_exists("lineups"):
        return pd.DataFrame()
    try:
        sql = text("""
            SELECT game_pk, team, side, batting_order, player_id, player_name
            FROM lineups
            WHERE game_pk = ANY(:pks)
            ORDER BY game_pk, side, batting_order
        """)
        with engine.begin() as conn:
            return pd.read_sql(sql, conn, params={"pks": game_pks})
    except Exception as e:
        log(f"⚠️  lineups load failed: {e}")
        return pd.DataFrame()


def load_bvp_for_pitchers(pitcher_ids: list) -> pd.DataFrame:
    if not pitcher_ids or not table_exists("batter_vs_pitcher"):
        return pd.DataFrame()
    pitcher_ids = [p for p in pitcher_ids if p is not None]
    if not pitcher_ids:
        return pd.DataFrame()
    try:
        sql = text("""
            SELECT batter_id, pitcher_id, avg, obp, slg, hard_hit_pct, avg_exit_velo, pa
            FROM batter_vs_pitcher
            WHERE pitcher_id = ANY(:pids)
              AND pa >= 2
        """)
        with engine.begin() as conn:
            return pd.read_sql(sql, conn, params={"pids": pitcher_ids})
    except Exception as e:
        log(f"⚠️  bvp load failed: {e}")
        return pd.DataFrame()


def compute_lineup_quality(
    game_pk: int,
    batting_side: str,
    opposing_sp_id: Any,
    lineups_df: pd.DataFrame,
    bvp_df: pd.DataFrame,
) -> Dict[str, float]:
    defaults = {"ops_vs_sp": LEAGUE_AVG_OPS, "hard_hit_pct": 0.35, "coverage": 0.0}

    if lineups_df.empty or opposing_sp_id is None:
        return defaults
    try:
        if math.isnan(float(opposing_sp_id)):
            return defaults
    except (TypeError, ValueError):
        return defaults

    team_lineup = lineups_df[
        (lineups_df["game_pk"] == game_pk) &
        (lineups_df["side"] == batting_side)
    ].sort_values("batting_order")

    if team_lineup.empty or bvp_df.empty:
        return defaults

    try:
        pitcher_bvp = bvp_df[bvp_df["pitcher_id"] == int(opposing_sp_id)]
    except (ValueError, TypeError):
        return defaults
    bvp_lookup = {row["batter_id"]: row for _, row in pitcher_bvp.iterrows()}

    weighted_ops      = 0.0
    weighted_hard_hit = 0.0
    total_weight      = 0.0
    batters_with_data = 0

    for _, batter in team_lineup.iterrows():
        order  = int(batter["batting_order"]) if pd.notna(batter["batting_order"]) else 9
        weight = BATTING_ORDER_WEIGHTS.get(order, 0.3)
        pid    = batter["player_id"]

        if pid in bvp_lookup:
            bvp = bvp_lookup[pid]
            obp = coerce_float(bvp.get("obp"))
            slg = coerce_float(bvp.get("slg"))
            hh  = coerce_float(bvp.get("hard_hit_pct"))
            if obp is not None and slg is not None:
                weighted_ops      += (obp + slg) * weight
                weighted_hard_hit += (hh or 0.35) * weight
                total_weight      += weight
                batters_with_data += 1
        else:
            weighted_ops      += LEAGUE_AVG_OPS * weight
            weighted_hard_hit += 0.35 * weight
            total_weight      += weight

    if total_weight == 0:
        return defaults

    return {
        "ops_vs_sp":    round(weighted_ops / total_weight, 4),
        "hard_hit_pct": round(weighted_hard_hit / total_weight, 4),
        "coverage":     round(batters_with_data / max(len(team_lineup), 1), 3),
    }


# =============================================================================
# TEAM BASELINES / ROLLING FORM
# =============================================================================

def build_team_game_log(completed_games: pd.DataFrame) -> pd.DataFrame:
    if completed_games.empty:
        return pd.DataFrame(columns=["official_date","season","team","side","runs_scored","runs_allowed","run_diff","win"])

    home = completed_games[["official_date","season","home_team","home_score","away_score"]].copy()
    home.columns = ["official_date","season","team","runs_scored","runs_allowed"]
    home["side"] = "home"

    away = completed_games[["official_date","season","away_team","away_score","home_score"]].copy()
    away.columns = ["official_date","season","team","runs_scored","runs_allowed"]
    away["side"] = "away"

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["run_diff"] = team_games["runs_scored"] - team_games["runs_allowed"]
    team_games["win"]      = (team_games["run_diff"] > 0).astype(int)
    return team_games.sort_values(["team","official_date"]).reset_index(drop=True)


def compute_live_elo(completed_games: pd.DataFrame, target_date) -> Dict[str, float]:
    """
    Walk all completed games chronologically up to (not including) target_date
    and return each team's current Elo rating.  No lookahead.
    """
    ratings: Dict[str, float] = {}
    if completed_games.empty:
        return ratings

    hist = completed_games[completed_games["official_date"] < target_date].sort_values("official_date")
    for _, row in hist.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        h_elo = ratings.get(ht, ELO_START)
        a_elo = ratings.get(at, ELO_START)
        exp_h = 1.0 / (1.0 + 10 ** ((a_elo - h_elo - ELO_HOME_ADV) / 400.0))
        actual_h = 1.0 if float(row["home_score"]) > float(row["away_score"]) else 0.0
        delta = ELO_K * (actual_h - exp_h)
        ratings[ht] = h_elo + delta
        ratings[at] = a_elo - delta
    return ratings


def get_team_last5_run_diff(team_games: pd.DataFrame, team: str, target_date, current_season: int = 2026) -> float:
    """Last-5 game average run differential (same season, before target_date)."""
    tg = team_games[
        (team_games["team"] == team) &
        (team_games["season"] == current_season) &
        (team_games["official_date"] < target_date)
    ].sort_values("official_date").tail(5)
    if tg.empty:
        return 0.0
    return float(tg["run_diff"].mean())


def get_team_last_n_stats(team_games: pd.DataFrame, team: str, target_date,
                          n: int = 10, current_season: int = 2026,
                          min_games: int = 5) -> Optional[Dict[str, float]]:
    """
    TRUE rolling last-n stats (same season, before target_date).

    This is what the training data's home_last10_* columns represent. The old
    engine fed the model get_team_blended_form() output instead — a mix of
    current form and 2024/2025 season baselines. If prior seasons ran hotter,
    that inflated offensive inputs at serve time relative to training and
    biased every total upward (the reason TOTAL_CALIBRATION=-1.2 existed).

    Returns None when fewer than min_games are available; caller falls back
    to blended baselines (early season only, when there's no better option).
    """
    tg = team_games[
        (team_games["team"] == team) &
        (team_games["season"] == current_season) &
        (team_games["official_date"] < target_date)
    ].sort_values("official_date").tail(n)
    if len(tg) < min_games:
        return None
    return {
        "runs_scored":  float(tg["runs_scored"].mean()),
        "runs_allowed": float(tg["runs_allowed"].mean()),
        "run_diff":     float(tg["run_diff"].mean()),
        "games":        float(len(tg)),
    }


def build_prior_baselines(team_games: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Tapered weights: 2026=60%, 2025=30%, 2024=10%"""
    if team_games.empty:
        return {"_LEAGUE_DEFAULT_": {"runs_scored":4.5,"runs_allowed":4.5,"run_diff":0.0,"home_win_pct":0.5,"away_win_pct":0.5}}

    season_team = team_games.groupby(["season","team"], as_index=False).agg(
        runs_scored=("runs_scored","mean"),
        runs_allowed=("runs_allowed","mean"),
        run_diff=("run_diff","mean"),
    )
    side_team = team_games.groupby(["season","team","side"], as_index=False).agg(win_pct=("win","mean"))

    league_rs  = float(team_games["runs_scored"].mean())
    league_ra  = float(team_games["runs_allowed"].mean())
    league_rd  = float(team_games["run_diff"].mean())
    league_hwp = float(team_games.loc[team_games["side"]=="home","win"].mean())
    league_awp = float(team_games.loc[team_games["side"]=="away","win"].mean())

    baselines: Dict[str, Dict[str, float]] = {}

    for team in sorted(season_team["team"].dropna().unique()):
        t = season_team[season_team["team"] == team]
        s = side_team[side_team["team"] == team]

        def gwp(ssn, side, fallback):
            r = s[(s["season"]==ssn)&(s["side"]==side)]
            return fallback if r.empty else coerce_float(r.iloc[0]["win_pct"], fallback)

        r26 = t[t["season"]==2026]
        r25 = t[t["season"]==2025]
        r24 = t[t["season"]==2024]

        if not r26.empty and not r25.empty and not r24.empty:
            rs  = 0.6*float(r26.iloc[0]["runs_scored"])  + 0.3*float(r25.iloc[0]["runs_scored"])  + 0.1*float(r24.iloc[0]["runs_scored"])
            ra  = 0.6*float(r26.iloc[0]["runs_allowed"]) + 0.3*float(r25.iloc[0]["runs_allowed"]) + 0.1*float(r24.iloc[0]["runs_allowed"])
            hwp = 0.6*gwp(2026,"home",league_hwp) + 0.3*gwp(2025,"home",league_hwp) + 0.1*gwp(2024,"home",league_hwp)
            awp = 0.6*gwp(2026,"away",league_awp) + 0.3*gwp(2025,"away",league_awp) + 0.1*gwp(2024,"away",league_awp)
        elif not r26.empty and not r25.empty:
            rs  = 0.7*float(r26.iloc[0]["runs_scored"])  + 0.3*float(r25.iloc[0]["runs_scored"])
            ra  = 0.7*float(r26.iloc[0]["runs_allowed"]) + 0.3*float(r25.iloc[0]["runs_allowed"])
            hwp = 0.7*gwp(2026,"home",league_hwp) + 0.3*gwp(2025,"home",league_hwp)
            awp = 0.7*gwp(2026,"away",league_awp) + 0.3*gwp(2025,"away",league_awp)
        elif not r26.empty:
            rs, ra = float(r26.iloc[0]["runs_scored"]), float(r26.iloc[0]["runs_allowed"])
            hwp, awp = gwp(2026,"home",league_hwp), gwp(2026,"away",league_awp)
        elif not r25.empty and not r24.empty:
            rs  = 0.7*float(r25.iloc[0]["runs_scored"])  + 0.3*float(r24.iloc[0]["runs_scored"])
            ra  = 0.7*float(r25.iloc[0]["runs_allowed"]) + 0.3*float(r24.iloc[0]["runs_allowed"])
            hwp = 0.7*gwp(2025,"home",league_hwp) + 0.3*gwp(2024,"home",league_hwp)
            awp = 0.7*gwp(2025,"away",league_awp) + 0.3*gwp(2024,"away",league_awp)
        elif not r25.empty:
            rs, ra = float(r25.iloc[0]["runs_scored"]), float(r25.iloc[0]["runs_allowed"])
            hwp, awp = gwp(2025,"home",league_hwp), gwp(2025,"away",league_awp)
        elif not r24.empty:
            rs, ra = float(r24.iloc[0]["runs_scored"]), float(r24.iloc[0]["runs_allowed"])
            hwp, awp = gwp(2024,"home",league_hwp), gwp(2024,"away",league_awp)
        else:
            rs, ra, hwp, awp = league_rs, league_ra, league_hwp, league_awp

        baselines[team] = {"runs_scored":rs,"runs_allowed":ra,"run_diff":rs-ra,"home_win_pct":hwp,"away_win_pct":awp}

    baselines["_LEAGUE_DEFAULT_"] = {
        "runs_scored":league_rs,"runs_allowed":league_ra,"run_diff":league_rd,
        "home_win_pct":league_hwp,"away_win_pct":league_awp
    }
    return baselines


def get_team_blended_form(team_games, baselines, team, target_date, side,
                          current_season=2026, blend_cutoff_games=30, rolling_window=10):
    default_base = baselines.get(team, baselines.get("_LEAGUE_DEFAULT_", {
        "runs_scored":4.5,"runs_allowed":4.5,"run_diff":0.0,
        "home_win_pct":0.5,"away_win_pct":0.5
    }))
    base_side_wp = float(default_base["home_win_pct"] if side=="home" else default_base["away_win_pct"])

    team_hist = team_games[
        (team_games["team"]==team) &
        (team_games["season"]==current_season) &
        (team_games["official_date"] < target_date)
    ].sort_values("official_date")

    games_played = len(team_hist)
    if games_played == 0:
        return {"runs_scored":float(default_base["runs_scored"]),"runs_allowed":float(default_base["runs_allowed"]),"run_diff":float(default_base["run_diff"]),"side_win_pct":base_side_wp,"games_played":0.0}

    recent          = team_hist.tail(rolling_window)
    current_rs      = float(recent["runs_scored"].mean())
    current_ra      = float(recent["runs_allowed"].mean())
    current_rd      = float(recent["run_diff"].mean())
    side_hist       = team_hist[team_hist["side"]==side]
    current_side_wp = coerce_float(side_hist["win"].mean(), base_side_wp) or base_side_wp

    w = min(math.sqrt(games_played / float(blend_cutoff_games)), 1.0)
    return {
        "runs_scored":  w*current_rs      + (1-w)*float(default_base["runs_scored"]),
        "runs_allowed": w*current_ra      + (1-w)*float(default_base["runs_allowed"]),
        "run_diff":     w*current_rd      + (1-w)*float(default_base["run_diff"]),
        "side_win_pct": w*current_side_wp + (1-w)*base_side_wp,
        "games_played": float(games_played),
    }


# =============================================================================
# PITCHER / REST / BULLPEN
# =============================================================================

def latest_pitcher_stats_map(pitchers_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if pitchers_df.empty:
        return {}
    latest = pitchers_df.sort_values(["pitcher_name_key","season"]).drop_duplicates(subset=["pitcher_name_key"], keep="last")
    return {row["pitcher_name_key"]: {"era": coerce_float(row["era"]), "whip": coerce_float(row["whip"])} for _, row in latest.iterrows()}


def compute_league_pitcher_defaults(pitchers_df: pd.DataFrame) -> Dict[str, float]:
    if pitchers_df.empty:
        return {"era": 4.20, "whip": 1.30}
    return {
        "era":  coerce_float(pitchers_df["era"].mean(),  4.20) or 4.20,
        "whip": coerce_float(pitchers_df["whip"].mean(), 1.30) or 1.30,
    }


def get_pitcher_stats(pitcher_name, pitcher_map, league_defaults):
    if pitcher_name is None or pd.isna(pitcher_name):
        return {"era": league_defaults["era"], "whip": league_defaults["whip"]}
    key  = str(pitcher_name).strip().lower()
    vals = pitcher_map.get(key)
    if not vals:
        return {"era": league_defaults["era"], "whip": league_defaults["whip"]}
    return {
        "era":  coerce_float(vals.get("era"),  league_defaults["era"])  or league_defaults["era"],
        "whip": coerce_float(vals.get("whip"), league_defaults["whip"]) or league_defaults["whip"],
    }


def get_sp_rest_days(pgl_df, pitcher_name, target_date):
    if pgl_df.empty or pitcher_name is None or pd.isna(pitcher_name):
        return 5.0
    key = str(pitcher_name).strip().lower()
    q   = pgl_df[(pgl_df["pitcher_name_key"]==key) & (pgl_df["role"]=="SP") & (pgl_df["official_date"] < target_date)].sort_values("official_date")
    if q.empty:
        return 5.0
    try:
        return min(float((target_date - q.iloc[-1]["official_date"]).days), 10.0)
    except Exception:
        return 5.0


def get_bullpen_ip_4d(pgl_df, team, target_date):
    if pgl_df.empty or team is None:
        return 4.0
    start = target_date - timedelta(days=4)
    q = pgl_df[(pgl_df["team"]==str(team)) & (pgl_df["role"]=="RP") & (pgl_df["official_date"]>=start) & (pgl_df["official_date"]<target_date)]
    return coerce_float(q["innings_pitched"].sum(), 4.0) or 4.0


def get_sp_current_stats(pgl_df: pd.DataFrame, pitcher_name: str,
                          target_date, window_starts: int = 5) -> Dict[str, float]:
    if pgl_df.empty or pitcher_name is None or pd.isna(pitcher_name):
        return {"era": None, "whip": None}

    key = str(pitcher_name).strip().lower()
    sp  = pgl_df[
        (pgl_df["pitcher_name_key"] == key) &
        (pgl_df["role"] == "SP") &
        (pgl_df["official_date"] < target_date)
    ].sort_values("official_date")

    if sp.empty:
        return {"era": None, "whip": None}

    recent = sp.tail(window_starts)
    ip  = coerce_float(recent["innings_pitched"].sum(), 0.0) or 0.0
    er  = coerce_float(recent["er_allowed"].sum(),      0.0) or 0.0
    h   = coerce_float(recent["hits_allowed"].sum(),    0.0) or 0.0
    bb  = coerce_float(recent["walks"].sum(),           0.0) or 0.0

    if ip < 1.0:
        return {"era": None, "whip": None}

    return {
        "era":  round((er / ip) * 9, 2),
        "whip": round((h + bb) / ip, 3),
    }


# =============================================================================
# O/U TENDENCY + ATS COVER RATE
# =============================================================================

def get_ou_over_rate(team_games: pd.DataFrame, team: str, target_date, window: int = OU_WINDOW) -> float:
    tg = team_games[
        (team_games["team"] == team) &
        (team_games["official_date"] < target_date)
    ].tail(window)
    if len(tg) < 3:
        return 0.5
    total = tg["runs_scored"] + tg["runs_allowed"]
    return float((total > DEFAULT_TOTAL_LINE).mean())


def get_last_game_total(team_games: pd.DataFrame, team: str, target_date) -> float:
    tg = team_games[
        (team_games["team"] == team) &
        (team_games["official_date"] < target_date)
    ].sort_values("official_date")
    if tg.empty:
        return DEFAULT_TOTAL_LINE
    last = tg.iloc[-1]
    return float(last["runs_scored"] + last["runs_allowed"])


def get_ats_cover_rate(team_games: pd.DataFrame, team: str, target_date, window: int = ATS_WINDOW) -> float:
    tg = team_games[
        (team_games["team"] == team) &
        (team_games["official_date"] < target_date)
    ].tail(window)
    if len(tg) < 3:
        return 0.5
    run_diff = tg["runs_scored"] - tg["runs_allowed"]
    return float((run_diff > 1.5).mean())


# =============================================================================
# CALIBRATOR
# Must be defined here (not just in retrain.py) so joblib can unpickle it.
# =============================================================================
class PlattCalibrator:
    """
    Sigmoid (Platt scaling) calibrator. Wraps LogisticRegression to expose
    a .predict(proba_array) -> calibrated_proba_array interface.
    """
    def __init__(self, lr):
        self.lr = lr

    def predict(self, raw_proba):
        import numpy as np
        arr = np.asarray(raw_proba).reshape(-1, 1)
        return self.lr.predict_proba(arr)[:, 1]


# =============================================================================
# MODEL
# =============================================================================

def load_model_bundle() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    obj = joblib.load(MODEL_PATH)
    if not isinstance(obj, dict):
        raise RuntimeError("Expected ensemble bundle dict in mlb_model.pkl")
    return {
        "win_models":   obj["win_models"],
        "run_models":   obj["run_diff_models"],
        "total_models": obj["total_runs_models"],
        "feature_cols": obj.get("feature_cols"),
        "n_models":     obj.get("n_models"),
        "calibrator":   obj.get("calibrator"),
        "park_factors": obj.get("park_factors", PARK_FACTORS),
        # Out-of-fold residual stats — the REAL uncertainty of predictions.
        "sigma_total":  coerce_float(obj.get("sigma_total"), FALLBACK_SIGMA_TOTAL) or FALLBACK_SIGMA_TOTAL,
        "sigma_rd":     coerce_float(obj.get("sigma_rd"),    FALLBACK_SIGMA_RD)    or FALLBACK_SIGMA_RD,
        "total_bias":   coerce_float(obj.get("total_bias"),  None),
        "market_fills": obj.get("market_fills", {"market_home_prob_novig": 0.5,
                                                 "market_total_line": DEFAULT_TOTAL_LINE}),
    }


def predict_ml_ensemble(models, X):
    preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    # NOTE: the percentile spread here is seed-to-seed model disagreement.
    # It is kept only for logging/monitoring — it is NOT outcome uncertainty
    # and is never used in pick logic.
    return preds.mean(axis=1), np.percentile(preds,25,axis=1), np.percentile(preds,75,axis=1), preds.std(axis=1)


def predict_regression_ensemble(models, X):
    preds = np.column_stack([m.predict(X) for m in models])
    return preds.mean(axis=1), np.percentile(preds,25,axis=1), np.percentile(preds,75,axis=1), preds.std(axis=1)


# =============================================================================
# FEATURE BUILDING
# =============================================================================

def build_features_for_games(games_df: pd.DataFrame) -> pd.DataFrame:
    completed_games = load_completed_games()
    team_games      = build_team_game_log(completed_games)
    baselines       = build_prior_baselines(team_games)

    # Pre-compute Elo ratings up to today (same target_date for all games in a day)
    if not games_df.empty:
        elo_target = games_df["official_date"].iloc[0]
    else:
        elo_target = date.today()
    elo_ratings = compute_live_elo(completed_games, elo_target)

    probables_df            = load_probables()
    pitchers_df             = load_pitchers()
    pgl_df                  = load_pitcher_game_log()
    pitcher_map             = latest_pitcher_stats_map(pitchers_df)
    league_pitcher_defaults = compute_league_pitcher_defaults(pitchers_df)

    merged = games_df.merge(probables_df, on="game_pk", how="left") if not probables_df.empty else games_df.assign(away_sp_name=None, home_sp_name=None, home_sp_id=None, away_sp_id=None)

    gf_df = load_game_features(merged["game_pk"].tolist())
    if not gf_df.empty:
        merged = merged.merge(gf_df, on="game_pk", how="left")
        log(f"  game_features joined: {len(gf_df)} rows")
    else:
        log("  ⚠️  game_features not available — using defaults")

    odds_df = load_market_odds(merged["game_pk"].tolist())
    if not odds_df.empty:
        merged = merged.merge(odds_df, on="game_pk", how="left")
        log(f"  market_odds joined: {len(odds_df)} rows")
    else:
        log("  ⚠️  market_odds not available — ML picks will be PASS (no price = no bet)")

    lineups_df = load_lineups_for_games(merged["game_pk"].tolist())
    sp_ids = []
    if "home_sp_id" in merged.columns:
        sp_ids += merged["home_sp_id"].dropna().astype(int).tolist()
    if "away_sp_id" in merged.columns:
        sp_ids += merged["away_sp_id"].dropna().astype(int).tolist()
    bvp_df = load_bvp_for_pitchers(sp_ids) if sp_ids else pd.DataFrame()

    if not lineups_df.empty:
        log(f"  lineups loaded: {len(lineups_df)} rows across {lineups_df['game_pk'].nunique()} games")
    else:
        log("  ⚠️  no lineups available — using league avg OPS for lineup quality")

    rows: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        target_date  = row["official_date"]
        home_team    = row["home_team"]
        away_team    = row["away_team"]
        home_sp_name = row.get("home_sp_name")
        away_sp_name = row.get("away_sp_name")
        home_sp_id   = row.get("home_sp_id")
        away_sp_id   = row.get("away_sp_id")

        home_form = get_team_blended_form(team_games, baselines, home_team, target_date, "home", MLB_SEASON, BLEND_CUTOFF_GAMES, ROLLING_WINDOW)
        away_form = get_team_blended_form(team_games, baselines, away_team, target_date, "away", MLB_SEASON, BLEND_CUTOFF_GAMES, ROLLING_WINDOW)

        # ---------------------------------------------------------------------
        # TRUE last-10 rolling stats to match training data (fixes train/serve
        # skew that inflated totals). Blended baselines only as an early-season
        # fallback when a team has < 5 games in the current season.
        # ---------------------------------------------------------------------
        home_l10 = get_team_last_n_stats(team_games, home_team, target_date, ROLLING_WINDOW, MLB_SEASON)
        away_l10 = get_team_last_n_stats(team_games, away_team, target_date, ROLLING_WINDOW, MLB_SEASON)

        if home_l10 is not None:
            home_rs, home_ra, home_rd = home_l10["runs_scored"], home_l10["runs_allowed"], home_l10["run_diff"]
        else:
            home_rs, home_ra, home_rd = float(home_form["runs_scored"]), float(home_form["runs_allowed"]), float(home_form["run_diff"])

        if away_l10 is not None:
            away_rs, away_ra, away_rd = away_l10["runs_scored"], away_l10["runs_allowed"], away_l10["run_diff"]
        else:
            away_rs, away_ra, away_rd = float(away_form["runs_scored"]), float(away_form["runs_allowed"]), float(away_form["run_diff"])

        home_live = get_sp_current_stats(pgl_df, home_sp_name, target_date)
        away_live = get_sp_current_stats(pgl_df, away_sp_name, target_date)

        home_p = {
            "era":  home_live["era"]  or get_pitcher_stats(home_sp_name, pitcher_map, league_pitcher_defaults)["era"],
            "whip": home_live["whip"] or get_pitcher_stats(home_sp_name, pitcher_map, league_pitcher_defaults)["whip"],
        }
        away_p = {
            "era":  away_live["era"]  or get_pitcher_stats(away_sp_name, pitcher_map, league_pitcher_defaults)["era"],
            "whip": away_live["whip"] or get_pitcher_stats(away_sp_name, pitcher_map, league_pitcher_defaults)["whip"],
        }

        home_sp_rest = float(get_sp_rest_days(pgl_df, home_sp_name, target_date))
        away_sp_rest = float(get_sp_rest_days(pgl_df, away_sp_name, target_date))
        home_bp_ip   = float(get_bullpen_ip_4d(pgl_df, home_team, target_date))
        away_bp_ip   = float(get_bullpen_ip_4d(pgl_df, away_team, target_date))
        home_win_pct = float(home_form["side_win_pct"])
        away_win_pct = float(away_form["side_win_pct"])

        home_wrc    = coerce_float(row.get("home_wrc_plus"),    100.0) or 100.0
        away_wrc    = coerce_float(row.get("away_wrc_plus"),    100.0) or 100.0
        sp_fip_diff = coerce_float(row.get("sp_fip_diff"),      0.0)   or 0.0
        bp_fip_diff = coerce_float(row.get("bullpen_fip_diff"), 0.0)   or 0.0
        wrc_diff    = coerce_float(row.get("offense_wrc_diff"), 0.0)   or 0.0
        park_rf     = coerce_float(row.get("park_run_factor"),  1.0)   or 1.0
        temp_f      = coerce_float(row.get("temperature_f"),    72.0)  or 72.0
        wind_mph    = coerce_float(row.get("wind_speed_mph"),   7.0)   or 7.0

        mkt_home_ml   = coerce_float(row.get("market_home_ml"))
        mkt_away_ml   = coerce_float(row.get("market_away_ml"))
        mkt_home_prob = coerce_float(row.get("market_home_prob"))
        mkt_away_prob = coerce_float(row.get("market_away_prob"))
        mkt_total     = coerce_float(row.get("market_total_line"))

        # De-vigged market probability — the market's fair opinion.
        hp = mkt_home_prob if mkt_home_prob is not None else american_to_prob(mkt_home_ml)
        ap = mkt_away_prob if mkt_away_prob is not None else american_to_prob(mkt_away_ml)
        mkt_home_novig = devig_two_way(hp, ap)

        home_ou_rate    = get_ou_over_rate(team_games, home_team, target_date)
        away_ou_rate    = get_ou_over_rate(team_games, away_team, target_date)
        home_last_total = get_last_game_total(team_games, home_team, target_date)
        away_last_total = get_last_game_total(team_games, away_team, target_date)
        home_ats_rate   = get_ats_cover_rate(team_games, home_team, target_date)
        away_ats_rate   = get_ats_cover_rate(team_games, away_team, target_date)

        game_pk = int(row["game_pk"])
        home_lq = compute_lineup_quality(game_pk, "home", away_sp_id, lineups_df, bvp_df)
        away_lq = compute_lineup_quality(game_pk, "away", home_sp_id, lineups_df, bvp_df)

        home_elo = elo_ratings.get(home_team, ELO_START)
        away_elo = elo_ratings.get(away_team, ELO_START)

        home_park_factor = PARK_FACTORS.get(home_team, 1.0)

        home_last5_rd = get_team_last5_run_diff(team_games, home_team, target_date, MLB_SEASON)
        away_last5_rd = get_team_last5_run_diff(team_games, away_team, target_date, MLB_SEASON)
        # form trend = last5 - TRUE last10 (matches retrain.py exactly now)
        home_form_trend = home_last5_rd - home_rd
        away_form_trend = away_last5_rd - away_rd

        home_neutral_win_pct = home_win_pct * 0.5 + (1 - away_win_pct) * 0.5
        away_neutral_win_pct = away_win_pct * 0.5 + (1 - home_win_pct) * 0.5

        rows.append({
            "game_pk":                  game_pk,
            "official_date":            target_date,
            "away_team":                away_team,
            "home_team":                home_team,
            "away_team_id":             row.get("away_team_id"),
            "home_team_id":             row.get("home_team_id"),
            "away_sp_name":             away_sp_name,
            "home_sp_name":             home_sp_name,
            "era_diff":                 float(home_p["era"]  - away_p["era"]),
            "whip_diff":                float(home_p["whip"] - away_p["whip"]),
            "home_sp_rest_days":        home_sp_rest,
            "away_sp_rest_days":        away_sp_rest,
            "home_bullpen_ip_4d":       home_bp_ip,
            "away_bullpen_ip_4d":       away_bp_ip,
            "home_win_pct_home":        home_win_pct,
            "away_win_pct_away":        away_win_pct,
            "home_last10_runs_scored":  home_rs,
            "away_last10_runs_scored":  away_rs,
            "home_last10_runs_allowed": home_ra,
            "away_last10_runs_allowed": away_ra,
            "home_last10_run_diff":     home_rd,
            "away_last10_run_diff":     away_rd,
            "sp_fip_diff":              sp_fip_diff,
            "offense_wrc_diff":         wrc_diff,
            "home_wrc_plus":            home_wrc,
            "away_wrc_plus":            away_wrc,
            "bullpen_fip_diff":         bp_fip_diff,
            "park_run_factor":          park_rf,
            "temperature_f":            temp_f,
            "wind_speed_mph":           wind_mph,
            "run_diff_form_diff":       home_rd    - away_rd,
            "runs_scored_diff":         home_rs    - away_rs,
            "runs_allowed_diff":        home_ra    - away_ra,
            "sp_rest_diff":             home_sp_rest - away_sp_rest,
            "bullpen_usage_diff":       home_bp_ip   - away_bp_ip,
            "win_pct_diff":             home_win_pct - away_win_pct,
            "home_ou_over_rate":        home_ou_rate,
            "away_ou_over_rate":        away_ou_rate,
            "home_last_game_total":     home_last_total,
            "away_last_game_total":     away_last_total,
            "home_ats_cover_rate":      home_ats_rate,
            "away_ats_cover_rate":      away_ats_rate,
            "home_lineup_ops_vs_sp":    home_lq["ops_vs_sp"],
            "away_lineup_ops_vs_sp":    away_lq["ops_vs_sp"],
            "lineup_ops_diff":          home_lq["ops_vs_sp"] - away_lq["ops_vs_sp"],
            "home_lineup_hard_hit":     home_lq["hard_hit_pct"],
            "away_lineup_hard_hit":     away_lq["hard_hit_pct"],
            "elo_diff":                 home_elo - away_elo,
            "home_elo_pre":             home_elo,
            "away_elo_pre":             away_elo,
            "home_park_factor":         home_park_factor,
            "home_last5_run_diff":      home_last5_rd,
            "away_last5_run_diff":      away_last5_rd,
            "home_form_trend":          home_form_trend,
            "away_form_trend":          away_form_trend,
            "form_trend_diff":          home_form_trend - away_form_trend,
            "home_neutral_win_pct":     home_neutral_win_pct,
            "away_neutral_win_pct":     away_neutral_win_pct,
            "neutral_win_pct_diff":     home_neutral_win_pct - away_neutral_win_pct,
            "ou_over_rate_diff":        home_ou_rate - away_ou_rate,
            "last_game_total_diff":     home_last_total - away_last_total,
            "ats_cover_rate_diff":      home_ats_rate - away_ats_rate,
            "market_home_ml":           mkt_home_ml,
            "market_away_ml":           mkt_away_ml,
            "market_home_prob":         mkt_home_prob,
            "market_away_prob":         mkt_away_prob,
            "market_home_prob_novig":   mkt_home_novig,
            "market_total_line":        mkt_total,
        })

    features_df = pd.DataFrame(rows)

    log("Feature nunique:")
    preview_cols = [
        "era_diff","whip_diff",
        "home_win_pct_home","away_win_pct_away","win_pct_diff",
        "home_last10_run_diff","away_last10_run_diff","run_diff_form_diff",
        "home_ou_over_rate","away_ou_over_rate",
        "home_ats_cover_rate","away_ats_cover_rate",
        "home_lineup_ops_vs_sp","away_lineup_ops_vs_sp","lineup_ops_diff",
        "market_home_ml","market_away_ml","market_total_line","market_home_prob_novig",
    ]
    if not features_df.empty:
        log(features_df[[c for c in preview_cols if c in features_df.columns]].nunique(dropna=False).to_string())

    return features_df


# =============================================================================
# PICKS / TOP PLAYS
# Every pick is an edge-vs-price decision:
#   ML  — calibrated model prob vs de-vigged market prob (no price → PASS)
#   O/U — P(over) from Normal(pred, sigma_total) vs breakeven at assumed -110
#   RL  — P(cover) from Normal(rd_pred, sigma_rd) vs conservative thresholds
#         (advisory only until RL prices are in market_odds)
# =============================================================================

def build_pick_columns(df: pd.DataFrame, sigma_total: float, sigma_rd: float) -> pd.DataFrame:
    df = df.copy()

    ou_breakeven = american_to_prob(OU_ASSUMED_PRICE) or 0.5238

    def get_ml(r):
        rd_pred = coerce_float(r.get("run_diff_pred"))
        m_home  = coerce_float(r.get("market_home_prob_novig")) # de-vigged market
        if rd_pred is None or m_home is None:
            return "PASS", None   # no price = no bet
        # P(home wins) = P(run_diff > 0) under Normal(rd_pred, sigma_rd)
        # This ensures ML and RL picks always agree on direction.
        p_home = prob_over(rd_pred, 0.0, sigma_rd)
        edge = p_home - m_home
        if edge >= MIN_ML_EDGE:
            return r["home_team"], edge
        if edge <= -MIN_ML_EDGE:
            return r["away_team"], edge
        return "PASS", edge

    def get_ou(r):
        pred = coerce_float(r.get("total_runs_pred"))
        line = coerce_float(r.get("market_total_line"))
        if pred is None or line is None:
            # No market total for this game — nothing to bet against.
            return "PASS", None, None
        p_ov = prob_over(pred, line, sigma_total)
        if p_ov >= ou_breakeven + MIN_OU_EDGE:
            return "OVER",  p_ov, p_ov - ou_breakeven
        if (1.0 - p_ov) >= ou_breakeven + MIN_OU_EDGE:
            return "UNDER", p_ov, (1.0 - p_ov) - ou_breakeven
        return "PASS", p_ov, None

    def get_rl(r):
        rd_pred = coerce_float(r.get("run_diff_pred"))
        m_home  = coerce_float(r.get("market_home_prob_novig"))
        home    = r["home_team"]
        away    = r["away_team"]
        if rd_pred is None:
            return "PASS", None

        # market favorite by de-vigged prob; fall back to model if odds missing
        if m_home is not None:
            fav_is_home = m_home >= 0.5
        else:
            p_home = coerce_float(r.get("home_win_prob"))
            if p_home is None:
                return "PASS", None
            fav_is_home = p_home >= 0.5

        p_home_covers_15 = prob_home_covers(rd_pred, 1.5, sigma_rd)   # home wins by 2+
        p_away_covers_15 = 1.0 - normal_cdf(-1.5, mu=-rd_pred, sigma=sigma_rd)  # away wins by 2+

        if fav_is_home:
            fav, dog = home, away
            p_fav_cover = p_home_covers_15                 # fav -1.5
            p_dog_cover = 1.0 - p_home_covers_15           # dog +1.5 (home fails to win by 2+)
        else:
            fav, dog = away, home
            p_fav_cover = p_away_covers_15
            p_dog_cover = 1.0 - p_away_covers_15

        if p_fav_cover >= RL_FAV_MIN_COVER_PROB:
            return f"{fav} -1.5", p_home_covers_15
        if p_dog_cover >= RL_DOG_MIN_COVER_PROB:
            return f"{dog} +1.5", p_home_covers_15
        return "PASS", p_home_covers_15

    ml_results = df.apply(get_ml, axis=1)
    df["ml_pick"] = [x[0] for x in ml_results]
    df["ml_edge"] = [x[1] for x in ml_results]

    ou_results = df.apply(get_ou, axis=1)
    df["ou_pick"] = [x[0] for x in ou_results]
    df["p_over"]  = [x[1] for x in ou_results]
    df["ou_edge"] = [x[2] for x in ou_results]

    rl_results = df.apply(get_rl, axis=1)
    df["runline_pick"]       = [x[0] for x in rl_results]
    df["rl_home_cover_prob"] = [x[1] for x in rl_results]

    # legacy column some dashboards read
    df["model_edge"] = df["ml_edge"]
    return df


def build_top_plays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank plays by expected value per $1 staked at the actual (or assumed)
    price. ML plays require real market prices — no price, no play.
    Run line plays are excluded until RL prices are in the odds feed.
    """
    rows = []
    for _, r in df.iterrows():
        away, home = r["away_team"], r["home_team"]

        # --- Moneyline: EV at the actual market price
        ml_pick = r.get("ml_pick")
        p_home  = coerce_float(r.get("home_win_prob"))
        if ml_pick not in (None, "PASS", "") and p_home is not None:
            if ml_pick == home:
                p, price = p_home, coerce_float(r.get("market_home_ml"))
            else:
                p, price = 1.0 - p_home, coerce_float(r.get("market_away_ml"))
            if price is not None:
                ev = ev_per_dollar(p, price)
                if ev is not None and ev > 0:
                    edge = coerce_float(r.get("ml_edge"))
                    rows.append({
                        "game_pk":     r["game_pk"],
                        "bet_type":    "ML",
                        "pick":        ml_pick,
                        "matchup":     f"{away} @ {home}",
                        "score":       round(ev, 4),
                        "model_value": round(p, 3),
                        "extra":       f"p={p:.3f} vs mkt novig={coerce_float(r.get('market_home_prob_novig'), 0.5):.3f}"
                                       f" | edge {edge:+.3f} | price {int(price)}",
                    })

        # --- Totals: EV at assumed juice (line from market, price assumed)
        ou_pick = r.get("ou_pick")
        p_ov    = coerce_float(r.get("p_over"))
        line    = coerce_float(r.get("market_total_line"))
        pred    = coerce_float(r.get("total_runs_pred"))
        if ou_pick in ("OVER", "UNDER") and p_ov is not None and line is not None:
            p  = p_ov if ou_pick == "OVER" else 1.0 - p_ov
            ev = ev_per_dollar(p, OU_ASSUMED_PRICE)
            if ev is not None and ev > 0:
                rows.append({
                    "game_pk":     r["game_pk"],
                    "bet_type":    "O/U",
                    "pick":        ou_pick,
                    "matchup":     f"{away} @ {home}",
                    "score":       round(ev, 4),
                    "model_value": round(pred, 2) if pred is not None else None,
                    "extra":       f"P({ou_pick.lower()})={p:.3f} | pred {pred:.2f} vs line {line}"
                                   f" | assumed {OU_ASSUMED_PRICE}",
                })

    top = pd.DataFrame(rows)
    if top.empty:
        return top
    return top.sort_values("score", ascending=False).head(5).reset_index(drop=True)


# =============================================================================
# SAVE
# =============================================================================

def upsert_predictions(pred_df: pd.DataFrame) -> None:
    if pred_df.empty:
        log("No predictions to save.")
        return

    ensure_predictions_table()

    base_cols = [
        "game_pk","official_date","away_team","home_team",
        "away_team_id","home_team_id","away_sp_name","home_sp_name",
        "era_diff","whip_diff",
        "home_sp_rest_days","away_sp_rest_days",
        "home_bullpen_ip_4d","away_bullpen_ip_4d",
        "home_win_pct_home","away_win_pct_away",
        "home_last10_runs_scored","away_last10_runs_scored",
        "home_last10_runs_allowed","away_last10_runs_allowed",
        "home_last10_run_diff","away_last10_run_diff",
        "sp_fip_diff","offense_wrc_diff",
        "home_wrc_plus","away_wrc_plus",
        "bullpen_fip_diff","park_run_factor",
        "temperature_f","wind_speed_mph",
        "run_diff_form_diff","runs_scored_diff","runs_allowed_diff",
        "sp_rest_diff","bullpen_usage_diff","win_pct_diff",
        "home_ou_over_rate","away_ou_over_rate",
        "home_last_game_total","away_last_game_total",
        "home_ats_cover_rate","away_ats_cover_rate",
        "home_lineup_ops_vs_sp","away_lineup_ops_vs_sp","lineup_ops_diff",
        "home_lineup_hard_hit","away_lineup_hard_hit",
        "home_win_prob","away_win_prob",
        "home_win_prob_lo","home_win_prob_hi","home_win_prob_std",
        "home_ml_implied","away_ml_implied",
        "run_diff_pred","run_diff_lo","run_diff_hi","run_diff_std",
        "total_runs_pred","total_runs_lo","total_runs_hi","total_runs_std",
        "ml_pick","runline_pick","ou_pick",
        "play_rank","play_type","play_score","play_detail",
        "market_home_ml","market_away_ml",
        "market_home_prob","market_away_prob","market_total_line",
        # betting-layer audit columns
        "market_home_prob_novig","home_win_prob_raw",
        "ml_edge","p_over","ou_edge","rl_home_cover_prob",
        "sigma_total_used","sigma_rd_used","model_edge",
    ]

    cols       = [c for c in base_cols if c in pred_df.columns]
    col_str    = ", ".join(cols)
    val_str    = ", ".join(f":{c}" for c in cols)
    update_str = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "game_pk")

    sql = f"""
    INSERT INTO {PREDICTIONS_TABLE} ({col_str}, updated_at)
    VALUES ({val_str}, NOW())
    ON CONFLICT (game_pk) DO UPDATE SET
        {update_str},
        updated_at = NOW()
    """

    def clean_record(rec):
        out = {}
        for k, v in rec.items():
            if isinstance(v, float) and math.isnan(v):
                out[k] = None
            elif k in ("market_home_ml", "market_away_ml") and v is not None:
                try:
                    out[k] = int(v)
                except (TypeError, ValueError):
                    out[k] = None
            else:
                out[k] = v
        return out

    records = [clean_record(r) for r in pred_df[cols].to_dict(orient="records")]
    with engine.begin() as conn:
        conn.execute(text(sql), records)

    log(f"Saved {len(pred_df)} predictions into {PREDICTIONS_TABLE}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log(f"PROJECT_ROOT:     {PROJECT_ROOT}")
    log(f"MODEL_PATH:       {MODEL_PATH}")
    log(f"GAMES_TABLE:      {GAMES_TABLE}")
    log(f"FEATURES_TABLE:   {FEATURES_TABLE}")
    log(f"MLB_SEASON:       {MLB_SEASON}")

    if not table_exists(GAMES_TABLE):
        raise RuntimeError(f"Table not found: {GAMES_TABLE}")

    games_df = load_upcoming_games()
    if games_df.empty:
        log("No upcoming games found.")
        return

    target_date = games_df["official_date"].iloc[0]
    log(f"Building predictions for {target_date} ({len(games_df)} games)")

    features_df = build_features_for_games(games_df)
    if features_df.empty:
        log("No features built.")
        return

    bundle = load_model_bundle()

    sigma_total = bundle["sigma_total"]
    sigma_rd    = bundle["sigma_rd"]
    total_bias  = bundle["total_bias"]
    log(f"Uncertainty: sigma_total={sigma_total:.2f}  sigma_rd={sigma_rd:.2f}  "
        f"total_bias={'n/a (old bundle)' if total_bias is None else f'{total_bias:+.2f}'}")

    feature_cols = bundle.get("feature_cols") or [
        "era_diff", "whip_diff",
        "home_sp_rest_days", "away_sp_rest_days",
        "home_bullpen_ip_4d", "away_bullpen_ip_4d",
        "home_win_pct_home", "away_win_pct_away",
        "home_last10_runs_scored", "away_last10_runs_scored",
        "home_last10_runs_allowed", "away_last10_runs_allowed",
        "home_last10_run_diff", "away_last10_run_diff",
        "sp_rest_diff", "bullpen_usage_diff",
        "home_ou_over_rate", "away_ou_over_rate",
        "home_last_game_total", "away_last_game_total",
        "home_ats_cover_rate", "away_ats_cover_rate",
        "home_lineup_ops_vs_sp", "away_lineup_ops_vs_sp", "lineup_ops_diff",
        "home_lineup_hard_hit", "away_lineup_hard_hit",
    ]

    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    X = features_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Market features get neutral fills (0.5 prob / league-ish line), matching
    # how retrain.py filled them. Everything else falls back to the median.
    market_fills = bundle.get("market_fills", {})
    for col in X.columns:
        if col in market_fills:
            X[col] = X[col].fillna(market_fills[col])
        else:
            fill = X[col].median() if X[col].notna().any() else 0.0
            X[col] = X[col].fillna(fill if not pd.isna(fill) else 0.0)

    log(f"Features ({len(feature_cols)}): {feature_cols}")

    raw_prob,   home_lo,  home_hi,  home_std  = predict_ml_ensemble(bundle["win_models"], X)
    run_pred,   run_lo,   run_hi,   run_std   = predict_regression_ensemble(bundle["run_models"], X)
    total_pred, total_lo, total_hi, total_std = predict_regression_ensemble(bundle["total_models"], X)

    # -------------------------------------------------------------------------
    # CALIBRATION IS MANDATORY.
    # If Platt scaling compresses raw 26%/74% outputs toward 45-58%, that is
    # the honest out-of-sample signal — the raw model was overconfident.
    # Wide raw probabilities feel better but lose to the vig at real prices.
    # -------------------------------------------------------------------------
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        home_prob = np.asarray(calibrator.predict(raw_prob))
        log(f"Calibration applied: raw mean {np.mean(raw_prob):.3f} "
            f"(5-95th: {np.percentile(raw_prob,5):.2f}-{np.percentile(raw_prob,95):.2f})"
            f" → cal mean {np.mean(home_prob):.3f} "
            f"(5-95th: {np.percentile(home_prob,5):.2f}-{np.percentile(home_prob,95):.2f})")
    else:
        home_prob = raw_prob
        log("⚠️  No calibrator in bundle — using raw probabilities. Retrain with new retrain.py.")

    home_prob = home_prob - HOME_BIAS_CORRECTION  # default 0.0; legacy escape hatch

    # Bias correction on totals: measured OOF bias from the bundle, else the
    # legacy env constant (default 0.0).
    bias_shift = -(total_bias) if total_bias is not None else TOTAL_CALIBRATION
    if total_bias is not None:
        log(f"Applying measured total bias correction: {bias_shift:+.2f} runs")
    elif TOTAL_CALIBRATION != 0.0:
        log(f"Applying legacy TOTAL_CALIBRATION: {TOTAL_CALIBRATION:+.2f} runs")

    out = features_df.copy()
    out["home_win_prob_raw"] = raw_prob
    out["home_win_prob"]     = home_prob
    out["away_win_prob"]     = 1.0 - home_prob
    out["home_win_prob_lo"]  = home_lo    # ensemble seed spread — monitoring only
    out["home_win_prob_hi"]  = home_hi
    out["home_win_prob_std"] = home_std
    out["home_ml_implied"]   = out["home_win_prob"].apply(safe_moneyline_from_prob)
    out["away_ml_implied"]   = out["away_win_prob"].apply(safe_moneyline_from_prob)
    out["run_diff_pred"]     = run_pred

    # lo/hi now reflect REAL uncertainty (±1 sigma), not ensemble seed spread.
    out["run_diff_lo"]       = run_pred - sigma_rd
    out["run_diff_hi"]       = run_pred + sigma_rd
    out["run_diff_std"]      = sigma_rd

    out["total_runs_pred"]   = total_pred + bias_shift
    out["total_runs_lo"]     = out["total_runs_pred"] - sigma_total
    out["total_runs_hi"]     = out["total_runs_pred"] + sigma_total
    out["total_runs_std"]    = sigma_total

    out["sigma_total_used"]  = sigma_total
    out["sigma_rd_used"]     = sigma_rd

    out = build_pick_columns(out, sigma_total=sigma_total, sigma_rd=sigma_rd)

    top_plays = build_top_plays(out)
    out["play_rank"]   = None
    out["play_type"]   = None
    out["play_score"]  = None
    out["play_detail"] = None

    if top_plays.empty:
        log("\nTop plays: none qualified (no positive-EV edges today — that's a feature, not a bug)")
    else:
        log("\nTOP 5 PLAYS OF THE DAY  (score = expected profit per $1 staked)")
        log(top_plays.to_string(index=False))
        for i, (_, row) in enumerate(top_plays.iterrows(), start=1):
            mask = out["game_pk"] == row["game_pk"]
            out.loc[mask, "play_rank"]   = i
            out.loc[mask, "play_type"]   = row["bet_type"]
            out.loc[mask, "play_score"]  = row["score"]
            out.loc[mask, "play_detail"] = row["extra"]

    # quick sanity print: pick distribution (a wall of OVERs should never happen now)
    if "ou_pick" in out.columns:
        log(f"\nO/U picks: {out['ou_pick'].value_counts(dropna=False).to_dict()}")
    if "ml_pick" in out.columns:
        n_pass = int((out["ml_pick"] == "PASS").sum())
        log(f"ML picks: {len(out) - n_pass} bets, {n_pass} passes")

    upsert_predictions(out)


if __name__ == "__main__":
    main()