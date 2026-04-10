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

DEFAULT_MODEL_PATH = PROJECT_ROOT / "ml" / "mlb_model.pkl"
MODEL_PATH         = Path(os.getenv("MLB_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

BLEND_CUTOFF_GAMES = 10
ROLLING_WINDOW     = 10
DEFAULT_TOTAL_LINE = float(os.getenv("DEFAULT_TOTAL_LINE", "8.5"))


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
        "sp_fip_diff": "DOUBLE PRECISION",
        "offense_wrc_diff": "DOUBLE PRECISION",
        "home_wrc_plus": "DOUBLE PRECISION",
        "away_wrc_plus": "DOUBLE PRECISION",
        "bullpen_fip_diff": "DOUBLE PRECISION",
        "park_run_factor": "DOUBLE PRECISION",
        "temperature_f": "DOUBLE PRECISION",
        "wind_speed_mph": "DOUBLE PRECISION",
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
        for col, col_type in wanted.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE {PREDICTIONS_TABLE} ADD COLUMN {col} {col_type}"))


# =============================================================================
# LOAD DATA
# =============================================================================

def load_upcoming_games(target_date: Optional[date] = None) -> pd.DataFrame:
    if not table_exists(GAMES_TABLE):
        raise RuntimeError(f"Missing required table: {GAMES_TABLE}")

    game_cols = set(get_table_columns(GAMES_TABLE))
    has_team_ids     = "home_team_id" in game_cols and "away_team_id" in game_cols
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
        return pd.DataFrame(columns=["game_pk", "away_sp_name", "home_sp_name"])
    cols = set(get_table_columns(PROBABLES_TABLE))
    if not {"game_pk", "away_sp_name", "home_sp_name"}.issubset(cols):
        return pd.DataFrame(columns=["game_pk", "away_sp_name", "home_sp_name"])
    return pd.read_sql(text(f"SELECT game_pk, away_sp_name, home_sp_name FROM {PROBABLES_TABLE}"), engine)


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
        return pd.DataFrame(columns=["official_date", "pitcher_name", "team", "role", "innings_pitched"])
    df = pd.read_sql(text(
        "SELECT official_date, pitcher_name, team, role, innings_pitched FROM pitcher_game_log"
    ), engine)
    if df.empty:
        return df
    df["official_date"]    = pd.to_datetime(df["official_date"], errors="coerce").dt.date
    df["pitcher_name_key"] = df["pitcher_name"].astype(str).str.strip().str.lower()
    df["innings_pitched"]  = pd.to_numeric(df["innings_pitched"], errors="coerce")
    return df.dropna(subset=["official_date"])


def load_game_features(game_pks: list) -> pd.DataFrame:
    """Load wRC+, FIP, bullpen, park factors, weather from game_features table."""
    if not game_pks or not table_exists(FEATURES_TABLE):
        return pd.DataFrame()
    try:
        sql = text(f"""
            SELECT game_pk,
                   home_wrc_plus,   away_wrc_plus,
                   home_sp_fip,     away_sp_fip,
                   home_bullpen_era, away_bullpen_era,
                   home_bullpen_fip, away_bullpen_fip,
                   sp_fip_diff,     bullpen_fip_diff,  offense_wrc_diff,
                   park_run_factor, park_hr_factor,
                   temperature_f,   wind_speed_mph
            FROM {FEATURES_TABLE}
            WHERE game_pk = ANY(:pks)
        """)
        with engine.begin() as conn:
            return pd.read_sql(sql, conn, params={"pks": game_pks})
    except Exception as e:
        log(f"⚠️  game_features load failed: {e}")
        return pd.DataFrame()


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


def build_prior_baselines(team_games: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if team_games.empty:
        return {"_LEAGUE_DEFAULT_": {"runs_scored":4.5,"runs_allowed":4.5,"run_diff":0.0,"home_win_pct":0.5,"away_win_pct":0.5}}

    season_team = team_games.groupby(["season","team"], as_index=False).agg(
        runs_scored=("runs_scored","mean"),
        runs_allowed=("runs_allowed","mean"),
        run_diff=("run_diff","mean"),
    )
    side_team = team_games.groupby(["season","team","side"], as_index=False).agg(win_pct=("win","mean"))

    league_rs    = float(team_games["runs_scored"].mean())
    league_ra    = float(team_games["runs_allowed"].mean())
    league_rd    = float(team_games["run_diff"].mean())
    league_hwp   = float(team_games.loc[team_games["side"]=="home","win"].mean())
    league_awp   = float(team_games.loc[team_games["side"]=="away","win"].mean())

    baselines: Dict[str, Dict[str, float]] = {}

    for team in sorted(season_team["team"].dropna().unique()):
        t = season_team[season_team["team"] == team]
        s = side_team[side_team["team"] == team]

        def gwp(season, side, fallback):
            r = s[(s["season"]==season)&(s["side"]==side)]
            return fallback if r.empty else coerce_float(r.iloc[0]["win_pct"], fallback)

        r25 = t[t["season"]==2025]
        r24 = t[t["season"]==2024]

        if not r25.empty and not r24.empty:
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

    baselines["_LEAGUE_DEFAULT_"] = {"runs_scored":league_rs,"runs_allowed":league_ra,"run_diff":league_rd,"home_win_pct":league_hwp,"away_win_pct":league_awp}
    return baselines


def get_team_blended_form(team_games, baselines, team, target_date, side, current_season=2026, blend_cutoff_games=10, rolling_window=10):
    default_base = baselines.get(team, baselines.get("_LEAGUE_DEFAULT_", {"runs_scored":4.5,"runs_allowed":4.5,"run_diff":0.0,"home_win_pct":0.5,"away_win_pct":0.5}))
    base_side_wp = float(default_base["home_win_pct"] if side=="home" else default_base["away_win_pct"])

    team_hist = team_games[
        (team_games["team"]==team) &
        (team_games["season"]==current_season) &
        (team_games["official_date"] < target_date)
    ].sort_values("official_date")

    games_played = len(team_hist)
    if games_played == 0:
        return {"runs_scored":float(default_base["runs_scored"]),"runs_allowed":float(default_base["runs_allowed"]),"run_diff":float(default_base["run_diff"]),"side_win_pct":base_side_wp,"games_played":0.0}

    recent = team_hist.tail(rolling_window)
    current_rs = float(recent["runs_scored"].mean())
    current_ra = float(recent["runs_allowed"].mean())
    current_rd = float(recent["run_diff"].mean())

    side_hist = team_hist[team_hist["side"]==side]
    current_side_wp = coerce_float(side_hist["win"].mean(), base_side_wp) or base_side_wp

    if games_played >= blend_cutoff_games:
        return {"runs_scored":current_rs,"runs_allowed":current_ra,"run_diff":current_rd,"side_win_pct":current_side_wp,"games_played":float(games_played)}

    w = min(games_played/float(blend_cutoff_games), 1.0)
    return {
        "runs_scored":  w*current_rs + (1-w)*float(default_base["runs_scored"]),
        "runs_allowed": w*current_ra + (1-w)*float(default_base["runs_allowed"]),
        "run_diff":     w*current_rd + (1-w)*float(default_base["run_diff"]),
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
    return {"era": coerce_float(pitchers_df["era"].mean(), 4.20) or 4.20, "whip": coerce_float(pitchers_df["whip"].mean(), 1.30) or 1.30}


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
    q = pgl_df[(pgl_df["pitcher_name_key"]==key) & (pgl_df["role"]=="SP") & (pgl_df["official_date"] < target_date)].sort_values("official_date")
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
        "win_models":    obj["win_models"],
        "run_models":    obj["run_diff_models"],
        "total_models":  obj["total_runs_models"],
        "feature_cols":  obj.get("feature_cols"),
        "n_models":      obj.get("n_models"),
    }


def predict_ml_ensemble(models, X):
    preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    return preds.mean(axis=1), np.percentile(preds,5,axis=1), np.percentile(preds,95,axis=1), preds.std(axis=1)


def predict_regression_ensemble(models, X):
    preds = np.column_stack([m.predict(X) for m in models])
    return preds.mean(axis=1), np.percentile(preds,5,axis=1), np.percentile(preds,95,axis=1), preds.std(axis=1)


# =============================================================================
# FEATURE BUILDING — now includes game_features (wRC+, FIP, park, weather)
# =============================================================================

def build_features_for_games(games_df: pd.DataFrame) -> pd.DataFrame:
    completed_games = load_completed_games()
    team_games      = build_team_game_log(completed_games)
    baselines       = build_prior_baselines(team_games)

    probables_df    = load_probables()
    pitchers_df     = load_pitchers()
    pgl_df          = load_pitcher_game_log()

    pitcher_map           = latest_pitcher_stats_map(pitchers_df)
    league_pitcher_defaults = compute_league_pitcher_defaults(pitchers_df)

    merged = games_df.merge(probables_df, on="game_pk", how="left") if not probables_df.empty else games_df.assign(away_sp_name=None, home_sp_name=None)

    # Load game_features (wRC+, FIP, bullpen, park, weather)
    gf_df = load_game_features(merged["game_pk"].tolist())
    if not gf_df.empty:
        merged = merged.merge(gf_df, on="game_pk", how="left")
        log(f"  game_features joined: {len(gf_df)} rows")
    else:
        log("  ⚠️  game_features not available — using defaults")

    rows: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        target_date  = row["official_date"]
        home_team    = row["home_team"]
        away_team    = row["away_team"]
        home_sp_name = row.get("home_sp_name")
        away_sp_name = row.get("away_sp_name")

        home_form = get_team_blended_form(team_games, baselines, home_team, target_date, "home", MLB_SEASON, BLEND_CUTOFF_GAMES, ROLLING_WINDOW)
        away_form = get_team_blended_form(team_games, baselines, away_team, target_date, "away", MLB_SEASON, BLEND_CUTOFF_GAMES, ROLLING_WINDOW)

        home_p = get_pitcher_stats(home_sp_name, pitcher_map, league_pitcher_defaults)
        away_p = get_pitcher_stats(away_sp_name, pitcher_map, league_pitcher_defaults)

        # game_features values (fall back to defaults if not joined)
        home_wrc    = coerce_float(row.get("home_wrc_plus"),   100.0) or 100.0
        away_wrc    = coerce_float(row.get("away_wrc_plus"),   100.0) or 100.0
        sp_fip_diff = coerce_float(row.get("sp_fip_diff"),     0.0)   or 0.0
        bp_fip_diff = coerce_float(row.get("bullpen_fip_diff"),0.0)   or 0.0
        wrc_diff    = coerce_float(row.get("offense_wrc_diff"),0.0)   or 0.0
        park_rf     = coerce_float(row.get("park_run_factor"), 1.0)   or 1.0
        temp_f      = coerce_float(row.get("temperature_f"),   72.0)  or 72.0
        wind_mph    = coerce_float(row.get("wind_speed_mph"),  7.0)   or 7.0

        rows.append({
            "game_pk":         int(row["game_pk"]),
            "official_date":   target_date,
            "away_team":       away_team,
            "home_team":       home_team,
            "away_team_id":    row.get("away_team_id"),
            "home_team_id":    row.get("home_team_id"),
            "away_sp_name":    away_sp_name,
            "home_sp_name":    home_sp_name,
            # Pitcher ERA/WHIP
            "era_diff":        float(home_p["era"]  - away_p["era"]),
            "whip_diff":       float(home_p["whip"] - away_p["whip"]),
            # Rest + bullpen
            "home_sp_rest_days":  float(get_sp_rest_days(pgl_df, home_sp_name, target_date)),
            "away_sp_rest_days":  float(get_sp_rest_days(pgl_df, away_sp_name, target_date)),
            "home_bullpen_ip_4d": float(get_bullpen_ip_4d(pgl_df, home_team, target_date)),
            "away_bullpen_ip_4d": float(get_bullpen_ip_4d(pgl_df, away_team, target_date)),
            # Win% splits
            "home_win_pct_home": float(home_form["side_win_pct"]),
            "away_win_pct_away": float(away_form["side_win_pct"]),
            # Rolling team form
            "home_last10_runs_scored":  float(home_form["runs_scored"]),
            "away_last10_runs_scored":  float(away_form["runs_scored"]),
            "home_last10_runs_allowed": float(home_form["runs_allowed"]),
            "away_last10_runs_allowed": float(away_form["runs_allowed"]),
            "home_last10_run_diff":     float(home_form["run_diff"]),
            "away_last10_run_diff":     float(away_form["run_diff"]),
            # game_features — FIP, wRC+, park, weather
            "sp_fip_diff":      sp_fip_diff,
            "offense_wrc_diff": wrc_diff,
            "home_wrc_plus":    home_wrc,
            "away_wrc_plus":    away_wrc,
            "bullpen_fip_diff": bp_fip_diff,
            "park_run_factor":  park_rf,
            "temperature_f":    temp_f,
            "wind_speed_mph":   wind_mph,
        })

    features_df = pd.DataFrame(rows)

    log("Feature nunique:")
    preview_cols = [
        "era_diff","whip_diff","sp_fip_diff","offense_wrc_diff",
        "home_wrc_plus","away_wrc_plus","bullpen_fip_diff",
        "park_run_factor","temperature_f","wind_speed_mph",
        "home_sp_rest_days","away_sp_rest_days",
        "home_bullpen_ip_4d","away_bullpen_ip_4d",
        "home_win_pct_home","away_win_pct_away",
        "home_last10_runs_scored","away_last10_runs_scored",
        "home_last10_run_diff","away_last10_run_diff",
    ]
    if not features_df.empty:
        log(features_df[[c for c in preview_cols if c in features_df.columns]].nunique(dropna=False).to_string())

    return features_df


# =============================================================================
# PICKS / TOP PLAYS
# =============================================================================

def build_pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def ml_pick(r):
        lo = coerce_float(r.get("home_win_prob_lo"))
        hi = coerce_float(r.get("home_win_prob_hi"))
        if lo is None or hi is None: return None
        if lo > 0.5:  return r["home_team"]
        if hi < 0.5:  return r["away_team"]
        return "PASS"

    def runline_pick(r):
        rd = coerce_float(r.get("run_diff_pred"))
        if rd is None: return None
        if rd >  1.5: return f"{r['home_team']} -1.5"
        if rd < -1.5: return f"{r['away_team']} -1.5"
        return f"{r['away_team']} +1.5"

    def ou_pick(r):
        lo = coerce_float(r.get("total_runs_lo"))
        hi = coerce_float(r.get("total_runs_hi"))
        if lo is None or hi is None: return None
        if lo > DEFAULT_TOTAL_LINE: return "OVER"
        if hi < DEFAULT_TOTAL_LINE: return "UNDER"
        return "PASS"

    df["ml_pick"]      = df.apply(ml_pick, axis=1)
    df["runline_pick"] = df.apply(runline_pick, axis=1)
    df["ou_pick"]      = df.apply(ou_pick, axis=1)
    return df


def build_top_plays(df, total_line=DEFAULT_TOTAL_LINE):
    rows = []
    for _, r in df.iterrows():
        away, home = r["away_team"], r["home_team"]
        lo = coerce_float(r.get("home_win_prob_lo"))
        hi = coerce_float(r.get("home_win_prob_hi"))
        mean = coerce_float(r.get("home_win_prob"))
        if lo and lo > 0.5:
            rows.append({"game_pk":r["game_pk"],"bet_type":"ML","pick":home,"matchup":f"{away} @ {home}","score":lo-0.5,"model_value":mean,"extra":f"CI: {lo:.3f} to {hi:.3f}"})
        elif hi and hi < 0.5:
            rows.append({"game_pk":r["game_pk"],"bet_type":"ML","pick":away,"matchup":f"{away} @ {home}","score":0.5-hi,"model_value":1-mean if mean else None,"extra":f"Home CI: {lo:.3f} to {hi:.3f}"})
        tlo = coerce_float(r.get("total_runs_lo"))
        thi = coerce_float(r.get("total_runs_hi"))
        tmean = coerce_float(r.get("total_runs_pred"))
        if tlo and tlo > total_line:
            rows.append({"game_pk":r["game_pk"],"bet_type":"O/U","pick":"OVER","matchup":f"{away} @ {home}","score":tlo-total_line,"model_value":tmean,"extra":f"CI: {tlo:.2f} to {thi:.2f} | line {total_line}"})
        elif thi and thi < total_line:
            rows.append({"game_pk":r["game_pk"],"bet_type":"O/U","pick":"UNDER","matchup":f"{away} @ {home}","score":total_line-thi,"model_value":tmean,"extra":f"CI: {tlo:.2f} to {thi:.2f} | line {total_line}"})

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

    # Build column list dynamically from what's in pred_df
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
        "home_win_prob","away_win_prob",
        "home_win_prob_lo","home_win_prob_hi","home_win_prob_std",
        "home_ml_implied","away_ml_implied",
        "run_diff_pred","run_diff_lo","run_diff_hi","run_diff_std",
        "total_runs_pred","total_runs_lo","total_runs_hi","total_runs_std",
        "ml_pick","runline_pick","ou_pick",
        "play_rank","play_type","play_score","play_detail",
    ]

    cols = [c for c in base_cols if c in pred_df.columns]
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

    records = pred_df[cols].to_dict(orient="records")
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

    # Default feature cols — updated to include game_features columns
    feature_cols = bundle.get("feature_cols") or [
        "era_diff", "whip_diff",
        "sp_fip_diff", "offense_wrc_diff",
        "home_wrc_plus", "away_wrc_plus",
        "bullpen_fip_diff",
        "park_run_factor", "temperature_f", "wind_speed_mph",
        "home_sp_rest_days", "away_sp_rest_days",
        "home_bullpen_ip_4d", "away_bullpen_ip_4d",
        "home_win_pct_home", "away_win_pct_away",
        "home_last10_runs_scored", "away_last10_runs_scored",
        "home_last10_runs_allowed", "away_last10_runs_allowed",
        "home_last10_run_diff", "away_last10_run_diff",
    ]

    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    X = features_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    for col in X.columns:
        fill = X[col].median() if X[col].notna().any() else 0.0
        X[col] = X[col].fillna(fill if not pd.isna(fill) else 0.0)

    log(f"Features ({len(feature_cols)}): {feature_cols}")

    home_prob, home_lo, home_hi, home_std = predict_ml_ensemble(bundle["win_models"], X)
    run_pred,  run_lo,  run_hi,  run_std  = predict_regression_ensemble(bundle["run_models"], X)
    total_pred,total_lo,total_hi,total_std= predict_regression_ensemble(bundle["total_models"], X)

    out = features_df.copy()
    out["home_win_prob"]    = home_prob
    out["away_win_prob"]    = 1.0 - home_prob
    out["home_win_prob_lo"] = home_lo
    out["home_win_prob_hi"] = home_hi
    out["home_win_prob_std"]= home_std
    out["home_ml_implied"]  = out["home_win_prob"].apply(safe_moneyline_from_prob)
    out["away_ml_implied"]  = out["away_win_prob"].apply(safe_moneyline_from_prob)
    out["run_diff_pred"]    = run_pred
    out["run_diff_lo"]      = run_lo
    out["run_diff_hi"]      = run_hi
    out["run_diff_std"]     = run_std
    out["total_runs_pred"]  = total_pred
    out["total_runs_lo"]    = total_lo
    out["total_runs_hi"]    = total_hi
    out["total_runs_std"]   = total_std

    out = build_pick_columns(out)

    top_plays = build_top_plays(out)
    out["play_rank"] = None
    out["play_type"] = None
    out["play_score"]= None
    out["play_detail"]= None

    if top_plays.empty:
        log("Top plays: none qualified")
    else:
        log("\nTOP 5 PLAYS OF THE DAY")
        log(top_plays.to_string(index=False))
        for i, (_, row) in enumerate(top_plays.iterrows(), start=1):
            mask = out["game_pk"] == row["game_pk"]
            out.loc[mask, "play_rank"]  = i
            out.loc[mask, "play_type"]  = row["bet_type"]
            out.loc[mask, "play_score"] = row["score"]
            out.loc[mask, "play_detail"]= row["extra"]

    upsert_predictions(out)


if __name__ == "__main__":
    main()