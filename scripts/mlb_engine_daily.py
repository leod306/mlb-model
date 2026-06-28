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

# Calibration offset — set via env; new retrain.py model should need less correction
TOTAL_CALIBRATION  = float(os.getenv("TOTAL_CALIBRATION", "-1.2"))

# Home bias correction — set to 0: isotonic calibration in new model handles this
HOME_BIAS_CORRECTION = float(os.getenv("HOME_BIAS_CORRECTION", "0.0"))

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
        "calibrator":   obj.get("calibrator"),   # isotonic calibration (new model)
        "park_factors": obj.get("park_factors", PARK_FACTORS),
    }


def predict_ml_ensemble(models, X):
    preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
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
        log("  ⚠️  market_odds not available — using model prob + default line")

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
        home_rs      = float(home_form["runs_scored"])
        away_rs      = float(away_form["runs_scored"])
        home_ra      = float(home_form["runs_allowed"])
        away_ra      = float(away_form["runs_allowed"])
        home_rd      = float(home_form["run_diff"])
        away_rd      = float(away_form["run_diff"])

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

        home_ou_rate    = get_ou_over_rate(team_games, home_team, target_date)
        away_ou_rate    = get_ou_over_rate(team_games, away_team, target_date)
        home_last_total = get_last_game_total(team_games, home_team, target_date)
        away_last_total = get_last_game_total(team_games, away_team, target_date)
        home_ats_rate   = get_ats_cover_rate(team_games, home_team, target_date)
        away_ats_rate   = get_ats_cover_rate(team_games, away_team, target_date)

        game_pk = int(row["game_pk"])
        home_lq = compute_lineup_quality(game_pk, "home", away_sp_id, lineups_df, bvp_df)
        away_lq = compute_lineup_quality(game_pk, "away", home_sp_id, lineups_df, bvp_df)

        # --- New features for retrained model ---
        home_elo = elo_ratings.get(home_team, ELO_START)
        away_elo = elo_ratings.get(away_team, ELO_START)

        home_park_factor = PARK_FACTORS.get(home_team, 1.0)

        home_last5_rd = get_team_last5_run_diff(team_games, home_team, target_date, MLB_SEASON)
        away_last5_rd = get_team_last5_run_diff(team_games, away_team, target_date, MLB_SEASON)
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
            # New features (retrained model)
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
        "market_home_ml","market_away_ml","market_total_line",
    ]
    if not features_df.empty:
        log(features_df[[c for c in preview_cols if c in features_df.columns]].nunique(dropna=False).to_string())

    return features_df


# =============================================================================
# PICKS / TOP PLAYS
# =============================================================================

def build_pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def get_ml(r):
        lo   = coerce_float(r.get("home_win_prob_lo"))
        hi   = coerce_float(r.get("home_win_prob_hi"))
        if lo is None or hi is None:
            return "PASS"

        lo_adj = lo - HOME_BIAS_CORRECTION
        hi_adj = hi - HOME_BIAS_CORRECTION

        if lo_adj > 0.48:  return r["home_team"]
        if hi_adj < 0.52:  return r["away_team"]
        return "PASS"

    def get_rl(r, ml):
        rd             = coerce_float(r.get("run_diff_pred"))
        home_prob      = coerce_float(r.get("home_win_prob"))
        market_home_ml = coerce_float(r.get("market_home_ml"))
        market_away_ml = coerce_float(r.get("market_away_ml"))
        home           = r["home_team"]
        away           = r["away_team"]
        if rd is None: return "PASS"

        if market_home_ml is not None and market_away_ml is not None:
            vegas_underdog = away if market_away_ml > 0 else home
            vegas_favorite = home if market_away_ml > 0 else away
        elif home_prob is not None:
            vegas_underdog = away if home_prob >= 0.5 else home
            vegas_favorite = home if home_prob >= 0.5 else away
        else:
            return "PASS"

        if ml and ml not in ("PASS", "") and ml == vegas_underdog:
            return f"{vegas_underdog} +1.5"

        if home_prob is not None:
            underdog_prob = (1 - home_prob) if vegas_underdog == away else home_prob
            if underdog_prob >= 0.40:
                return f"{vegas_underdog} +1.5"

        if ml and ml not in ("PASS", "") and ml == vegas_favorite and rd > 1.5:
            return f"{vegas_favorite} -1.5"

        return "PASS"

    def get_ou(r):
        mean = coerce_float(r.get("total_runs_pred"))
        lo   = coerce_float(r.get("total_runs_lo"))
        hi   = coerce_float(r.get("total_runs_hi"))
        if mean is None or lo is None or hi is None:
            return None
        line = coerce_float(r.get("market_total_line")) or DEFAULT_TOTAL_LINE
        if lo > line:         return "OVER"
        if hi < line:         return "UNDER"
        if mean > line + 0.5: return "OVER"
        if mean < line - 0.5: return "UNDER"
        return "PASS"

    df["ml_pick"]      = df.apply(get_ml, axis=1)
    df["runline_pick"] = df.apply(lambda r: get_rl(r, r["ml_pick"]), axis=1)
    df["ou_pick"]      = df.apply(get_ou, axis=1)
    return df


def build_top_plays(df, total_line=DEFAULT_TOTAL_LINE):
    rows = []
    for _, r in df.iterrows():
        away, home = r["away_team"], r["home_team"]
        lo   = coerce_float(r.get("home_win_prob_lo"))
        hi   = coerce_float(r.get("home_win_prob_hi"))
        mean = coerce_float(r.get("home_win_prob"))

        lo_adj = (lo - HOME_BIAS_CORRECTION) if lo is not None else None
        hi_adj = (hi - HOME_BIAS_CORRECTION) if hi is not None else None

        if lo_adj and lo_adj > 0.5:
            rows.append({"game_pk":r["game_pk"],"bet_type":"ML","pick":home,"matchup":f"{away} @ {home}","score":lo_adj-0.5,"model_value":mean,"extra":f"CI: {lo:.3f} to {hi:.3f}"})
        elif hi_adj and hi_adj < 0.5:
            rows.append({"game_pk":r["game_pk"],"bet_type":"ML","pick":away,"matchup":f"{away} @ {home}","score":0.5-hi_adj,"model_value":1-mean if mean else None,"extra":f"Home CI: {lo:.3f} to {hi:.3f}"})

        tlo       = coerce_float(r.get("total_runs_lo"))
        thi       = coerce_float(r.get("total_runs_hi"))
        tmean     = coerce_float(r.get("total_runs_pred"))
        game_line = coerce_float(r.get("market_total_line")) or total_line

        if tlo and tlo > game_line:
            rows.append({"game_pk":r["game_pk"],"bet_type":"O/U","pick":"OVER","matchup":f"{away} @ {home}","score":tlo-game_line,"model_value":tmean,"extra":f"CI: {tlo:.2f} to {thi:.2f} | line {game_line}"})
        elif thi and thi < game_line:
            rows.append({"game_pk":r["game_pk"],"bet_type":"O/U","pick":"UNDER","matchup":f"{away} @ {home}","score":game_line-thi,"model_value":tmean,"extra":f"CI: {tlo:.2f} to {thi:.2f} | line {game_line}"})

    top = pd.DataFrame(rows)
    return top.sort_values("score", ascending=False).head(5).reset_index(drop=True) if not top.empty else top


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

    # -------------------------------------------------------------------------
    # FEATURE COLUMNS
    # NOTE: run_diff_form_diff, runs_scored_diff, runs_allowed_diff, win_pct_diff
    # were removed — they are derived diffs that nearly duplicate signals already
    # present in the raw home/away features, causing triple-counting of form and
    # win% signal. Removing them frees weight for SP and lineup features.
    # Retrain mlb_model.pkl after making this change.
    # -------------------------------------------------------------------------
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
    for col in X.columns:
        fill = X[col].median() if X[col].notna().any() else 0.0
        X[col] = X[col].fillna(fill if not pd.isna(fill) else 0.0)

    log(f"Features ({len(feature_cols)}): {feature_cols}")

    home_prob,  home_lo,  home_hi,  home_std  = predict_ml_ensemble(bundle["win_models"], X)
    run_pred,   run_lo,   run_hi,   run_std   = predict_regression_ensemble(bundle["run_models"], X)
    total_pred, total_lo, total_hi, total_std = predict_regression_ensemble(bundle["total_models"], X)

    # Apply Platt (sigmoid) calibration — smooth curve, distinct output per game
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        home_prob = calibrator.predict(home_prob)
        home_lo   = calibrator.predict(home_lo)
        home_hi   = calibrator.predict(home_hi)
        log("  Platt calibration applied to win probabilities")

    out = features_df.copy()
    out["home_win_prob"]     = home_prob
    out["away_win_prob"]     = 1.0 - home_prob
    out["home_win_prob_lo"]  = home_lo
    out["home_win_prob_hi"]  = home_hi
    out["home_win_prob_std"] = home_std
    out["home_ml_implied"]   = out["home_win_prob"].apply(safe_moneyline_from_prob)
    out["away_ml_implied"]   = out["away_win_prob"].apply(safe_moneyline_from_prob)
    out["run_diff_pred"]     = run_pred
    out["run_diff_lo"]       = run_lo
    out["run_diff_hi"]       = run_hi
    out["run_diff_std"]      = run_std

    out["total_runs_pred"]   = total_pred + TOTAL_CALIBRATION
    out["total_runs_lo"]     = total_lo   + TOTAL_CALIBRATION
    out["total_runs_hi"]     = total_hi   + TOTAL_CALIBRATION
    out["total_runs_std"]    = total_std

    out = build_pick_columns(out)

    top_plays = build_top_plays(out)
    out["play_rank"]   = None
    out["play_type"]   = None
    out["play_score"]  = None
    out["play_detail"] = None

    if top_plays.empty:
        log("Top plays: none qualified")
    else:
        log("\nTOP 5 PLAYS OF THE DAY")
        log(top_plays.to_string(index=False))
        for i, (_, row) in enumerate(top_plays.iterrows(), start=1):
            mask = out["game_pk"] == row["game_pk"]
            out.loc[mask, "play_rank"]   = i
            out.loc[mask, "play_type"]   = row["bet_type"]
            out.loc[mask, "play_score"]  = row["score"]
            out.loc[mask, "play_detail"] = row["extra"]

    upsert_predictions(out)


if __name__ == "__main__":
    main()