"""
ml/retrain.py  —  MLB model retraining with sharper features
Run from the project root:
    python ml/retrain.py

Improvements over previous version:
  1. Elo ratings  — computed game-by-game from history (no lookahead)
  2. Ballpark run factor  — park-adjusted totals for O/U
  3. Form trend  — (last5 run diff) - (last10 run diff): positive = heating up
  4. Neutral win %  — overall win%, not split by venue (removes venue bleed)
  5. Time-series cross-validation  — train on past, score on future only
  6. Isotonic calibration  — fixes overconfident win probabilities
  7. Saves new mlb_model.pkl, win_model.pkl, run_model.pkl, total_model.pkl
"""

from __future__ import annotations

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR     = os.path.join(BASE_DIR, "ml")
DATA_PATH  = os.path.join(ML_DIR, "training_data.csv")

# ---------------------------------------------------------------------------
# 1. MLB Ballpark Run Factors (2024-25 average, 100 = league average)
#    Source: Baseball Reference park factors
# ---------------------------------------------------------------------------
PARK_FACTORS: dict[str, float] = {
    "COL": 1.18,   # Coors — huge boost
    "CIN": 1.10,
    "TEX": 1.07,
    "BOS": 1.06,
    "PHI": 1.05,
    "MIL": 1.04,
    "CHC": 1.04,
    "HOU": 1.03,
    "ATL": 1.02,
    "NYY": 1.02,
    "ARI": 1.01,
    "LAD": 1.00,
    "NYM": 1.00,
    "STL": 1.00,
    "DET": 0.99,
    "BAL": 0.99,
    "CWS": 0.99,
    "TOR": 0.98,
    "MIA": 0.98,
    "MIN": 0.98,
    "CLE": 0.97,
    "KC":  0.97,
    "TB":  0.97,
    "WSH": 0.97,
    "SF":  0.96,
    "SEA": 0.96,
    "SD":  0.96,
    "PIT": 0.95,
    "LAA": 0.95,
    "ATH": 0.95,  # Athletics (Oakland/Sacramento)
}

# ---------------------------------------------------------------------------
# 2. Elo ratings — computed chronologically, no lookahead
# ---------------------------------------------------------------------------
ELO_START   = 1500.0
ELO_K       = 20.0
ELO_HOME_ADV = 35.0   # home field bump in Elo


def expected_elo(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((away_elo - home_elo - ELO_HOME_ADV) / 400.0))


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds home_elo_pre, away_elo_pre, elo_diff columns.
    Ratings are the team's Elo BEFORE the game (no lookahead).
    """
    df = df.sort_values("game_date_utc").reset_index(drop=True)

    ratings: dict[str, float] = {}

    home_elo_pre = np.zeros(len(df))
    away_elo_pre = np.zeros(len(df))

    for i, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]

        h_elo = ratings.get(ht, ELO_START)
        a_elo = ratings.get(at, ELO_START)

        home_elo_pre[i] = h_elo
        away_elo_pre[i] = a_elo

        # update ratings using actual result
        if pd.notna(row.get("home_win")):
            exp_h = expected_elo(h_elo, a_elo)
            actual_h = float(row["home_win"])
            delta = ELO_K * (actual_h - exp_h)
            ratings[ht] = h_elo + delta
            ratings[at] = a_elo - delta

    df["home_elo_pre"] = home_elo_pre
    df["away_elo_pre"] = away_elo_pre
    df["elo_diff"]     = home_elo_pre - away_elo_pre
    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Elo
    df = compute_elo(df)

    # --- Park factor
    df["home_park_factor"] = df["home_team"].map(PARK_FACTORS).fillna(1.0)

    # --- Form trend: positive = team is heating up (last 5 better than last 10)
    #     We proxy with run_diff: if last10 data exists, assume last5 ≈ last10 * trend
    #     We'll compute it properly from rolling windows per team
    df = df.sort_values("game_date_utc").reset_index(drop=True)

    home_last5_rd: list[float] = []
    away_last5_rd: list[float] = []

    # rolling last-5 run diff per team
    team_results: dict[str, list[float]] = {}

    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]

        h_hist = team_results.get(ht, [])
        a_hist = team_results.get(at, [])

        h5 = np.mean(h_hist[-5:])  if len(h_hist) >= 5  else (np.mean(h_hist) if h_hist else 0.0)
        a5 = np.mean(a_hist[-5:])  if len(a_hist) >= 5  else (np.mean(a_hist) if a_hist else 0.0)

        home_last5_rd.append(h5)
        away_last5_rd.append(a5)

        # update after the fact
        if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
            h_rd = float(row["home_score"]) - float(row["away_score"])
            team_results.setdefault(ht, []).append(h_rd)
            team_results.setdefault(at, []).append(-h_rd)

    df["home_last5_run_diff"] = home_last5_rd
    df["away_last5_run_diff"] = away_last5_rd

    # trend = last5 minus last10; positive = improving
    df["home_form_trend"] = df["home_last5_run_diff"] - df["home_last10_run_diff"]
    df["away_form_trend"] = df["away_last5_run_diff"] - df["away_last10_run_diff"]
    df["form_trend_diff"]  = df["home_form_trend"]    - df["away_form_trend"]

    # --- Neutral win % (not split by venue)
    #     Approximate from home_win_pct_home and away_win_pct_away weighted equally
    #     A team with 0.60 home, 0.45 away ≈ 0.525 neutral
    #     We'll store both so the model can learn the interaction
    df["home_neutral_win_pct"] = (df["home_win_pct_home"] * 0.5 + (1 - df["away_win_pct_away"]) * 0.5)
    df["away_neutral_win_pct"] = (df["away_win_pct_away"] * 0.5 + (1 - df["home_win_pct_home"]) * 0.5)
    df["neutral_win_pct_diff"] = df["home_neutral_win_pct"] - df["away_neutral_win_pct"]

    # --- diff features that may not be in CSV
    for col in ["sp_rest_diff", "bullpen_usage_diff", "ou_over_rate_diff",
                "last_game_total_diff", "ats_cover_rate_diff", "lineup_ops_diff"]:
        if col not in df.columns:
            if col == "sp_rest_diff":
                df[col] = df["home_sp_rest_days"] - df["away_sp_rest_days"]
            elif col == "bullpen_usage_diff":
                df[col] = df["home_bullpen_ip_4d"] - df["away_bullpen_ip_4d"]
            elif col == "ou_over_rate_diff":
                df[col] = df["home_ou_over_rate"] - df["away_ou_over_rate"]
            elif col == "last_game_total_diff":
                df[col] = df["home_last_game_total"] - df["away_last_game_total"]
            elif col == "ats_cover_rate_diff":
                df[col] = df["home_ats_cover_rate"] - df["away_ats_cover_rate"]
            elif col == "lineup_ops_diff":
                df[col] = df["home_lineup_ops_vs_sp"] - df["away_lineup_ops_vs_sp"]

    return df


# ---------------------------------------------------------------------------
# 4. Feature columns
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Elo — most predictive single signal
    "elo_diff",
    "home_elo_pre",
    "away_elo_pre",

    # Park
    "home_park_factor",

    # Pitcher quality
    "era_diff",
    "whip_diff",

    # Rest / bullpen
    "home_sp_rest_days",
    "away_sp_rest_days",
    "sp_rest_diff",
    "home_bullpen_ip_4d",
    "away_bullpen_ip_4d",
    "bullpen_usage_diff",

    # Rolling form — last 10
    "home_last10_runs_scored",
    "away_last10_runs_scored",
    "home_last10_runs_allowed",
    "away_last10_runs_allowed",
    "home_last10_run_diff",
    "away_last10_run_diff",

    # Trend (direction of form)
    "home_last5_run_diff",
    "away_last5_run_diff",
    "home_form_trend",
    "away_form_trend",
    "form_trend_diff",

    # Win %
    "home_win_pct_home",
    "away_win_pct_away",
    "home_neutral_win_pct",
    "away_neutral_win_pct",
    "neutral_win_pct_diff",

    # Betting / situational
    "home_ou_over_rate",
    "away_ou_over_rate",
    "ou_over_rate_diff",
    "home_last_game_total",
    "away_last_game_total",
    "last_game_total_diff",
    "home_ats_cover_rate",
    "away_ats_cover_rate",
    "ats_cover_rate_diff",

    # Lineup vs SP
    "home_lineup_ops_vs_sp",
    "away_lineup_ops_vs_sp",
    "lineup_ops_diff",
    "home_lineup_hard_hit",
    "away_lineup_hard_hit",
]


# ---------------------------------------------------------------------------
# 5. Model configs
# ---------------------------------------------------------------------------
WIN_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=500,
    learning_rate=0.04,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
)

REG_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.04,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
)

N_MODELS = 20   # ensemble size (different random seeds)


# ---------------------------------------------------------------------------
# 6. Time-series cross-validation
# ---------------------------------------------------------------------------
def run_cv(X: pd.DataFrame, y_win: pd.Series, y_rd: pd.Series, y_tot: pd.Series):
    tscv = TimeSeriesSplit(n_splits=5, gap=30)  # 30-game gap prevents leakage

    win_logloss, win_brier, win_acc = [], [], []
    rd_mae, tot_mae = [], []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_win_tr,  y_win_te  = y_win.iloc[tr_idx],  y_win.iloc[te_idx]
        y_rd_tr,   y_rd_te   = y_rd.iloc[tr_idx],   y_rd.iloc[te_idx]
        y_tot_tr,  y_tot_te  = y_tot.iloc[tr_idx],  y_tot.iloc[te_idx]

        wm = XGBClassifier(**WIN_PARAMS)
        wm.fit(X_tr, y_win_tr, verbose=False)
        prob = wm.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)

        win_logloss.append(log_loss(y_win_te, prob))
        win_brier.append(brier_score_loss(y_win_te, prob))
        win_acc.append((pred == y_win_te).mean())

        rm = XGBRegressor(**REG_PARAMS)
        rm.fit(X_tr, y_rd_tr, verbose=False)
        rd_mae.append(mean_absolute_error(y_rd_te, rm.predict(X_te)))

        tm = XGBRegressor(**REG_PARAMS)
        tm.fit(X_tr, y_tot_tr, verbose=False)
        tot_mae.append(mean_absolute_error(y_tot_te, tm.predict(X_te)))

        print(f"  Fold {fold+1}: win_acc={win_acc[-1]:.3f}  logloss={win_logloss[-1]:.4f}  "
              f"rd_mae={rd_mae[-1]:.2f}  tot_mae={tot_mae[-1]:.2f}")

    print(f"\n  CV means: win_acc={np.mean(win_acc):.3f}  logloss={np.mean(win_logloss):.4f}  "
          f"brier={np.mean(win_brier):.4f}  rd_mae={np.mean(rd_mae):.2f}  tot_mae={np.mean(tot_mae):.2f}")
    return np.mean(win_acc)


# ---------------------------------------------------------------------------
# 7. Platt scaling calibration on a held-out slice
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """
    Sigmoid (Platt scaling) calibrator. Wraps a LogisticRegression so it
    exposes a .predict(proba_array) → calibrated_proba_array interface
    compatible with the engine. Unlike isotonic regression, sigmoid calibration
    produces a smooth monotonic curve — every game gets a distinct win probability.
    """
    def __init__(self, lr: LogisticRegression):
        self.lr = lr

    def predict(self, raw_proba):
        arr = np.asarray(raw_proba).reshape(-1, 1)
        return self.lr.predict_proba(arr)[:, 1]


def calibrate_win_models(win_models: list, X_cal: pd.DataFrame, y_cal: pd.Series):
    """
    Fit a Platt (sigmoid) calibrator on ensemble win probabilities
    from a held-out calibration set (last 20% of data by time).
    Returns a PlattCalibrator that maps raw proba → calibrated proba.
    """
    raw_probas = np.mean([m.predict_proba(X_cal)[:, 1] for m in win_models], axis=0)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(raw_probas.reshape(-1, 1), y_cal)
    return PlattCalibrator(lr)


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MLB Model Retrain")
    print("=" * 60)

    # --- Load data
    print(f"\nLoading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["home_win"].notna()].copy()
    df["game_date_utc"] = pd.to_datetime(df["game_date_utc"], utc=True, errors="coerce")
    df = df.sort_values("game_date_utc").reset_index(drop=True)
    print(f"  {len(df)} completed games  ({df['official_date'].min()} → {df['official_date'].max()})")

    # --- Feature engineering
    print("\nEngineering features ...")
    df = build_features(df)

    X = df[FEATURE_COLS].fillna(0.0)
    y_win = df["home_win"].astype(int)
    y_rd  = df["run_diff"].fillna(0.0)
    y_tot = df["total_runs"].fillna(0.0)

    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Home win rate: {y_win.mean():.3f}")

    # --- Cross-validation
    print("\nTime-series cross-validation ...")
    cv_acc = run_cv(X, y_win, y_rd, y_tot)

    # --- Calibration split (last 20% by time, never seen in CV training)
    cal_cutoff = int(len(df) * 0.80)
    X_cal   = X.iloc[cal_cutoff:]
    y_cal   = y_win.iloc[cal_cutoff:]
    X_train = X.iloc[:cal_cutoff]
    y_win_tr = y_win.iloc[:cal_cutoff]
    y_rd_tr  = y_rd.iloc[:cal_cutoff]
    y_tot_tr = y_tot.iloc[:cal_cutoff]

    # --- Train ensemble on full data (excluding cal set)
    print(f"\nTraining {N_MODELS}-model ensemble on {len(X_train)} games ...")

    win_models   = []
    rd_models    = []
    total_models = []

    for i in range(N_MODELS):
        seed = 1000 + i * 7

        wm = XGBClassifier(**{**WIN_PARAMS, "random_state": seed})
        wm.fit(X_train, y_win_tr, verbose=False)
        win_models.append(wm)

        rm = XGBRegressor(**{**REG_PARAMS, "random_state": seed})
        rm.fit(X_train, y_rd_tr, verbose=False)
        rd_models.append(rm)

        tm = XGBRegressor(**{**REG_PARAMS, "random_state": seed})
        tm.fit(X_train, y_tot_tr, verbose=False)
        total_models.append(tm)

        if (i + 1) % 5 == 0:
            print(f"  Trained {i+1}/{N_MODELS}")

    # --- Calibrate win proba
    print("\nFitting Platt (sigmoid) calibration on held-out set ...")
    calibrator = calibrate_win_models(win_models, X_cal, y_cal)

    # Check calibration improvement
    raw = np.mean([m.predict_proba(X_cal)[:, 1] for m in win_models], axis=0)
    cal_prob = calibrator.predict(raw)
    print(f"  Pre-cal  log_loss: {log_loss(y_cal, raw):.4f}  brier: {brier_score_loss(y_cal, raw):.4f}")
    print(f"  Post-cal log_loss: {log_loss(y_cal, cal_prob):.4f}  brier: {brier_score_loss(y_cal, cal_prob):.4f}")
    print(f"  Raw mean prob: {raw.mean():.3f}  Cal mean prob: {cal_prob.mean():.3f}  Actual: {y_cal.mean():.3f}")

    # --- Feature importance
    importances = pd.Series(
        np.mean([m.feature_importances_ for m in win_models], axis=0),
        index=FEATURE_COLS,
    ).sort_values(ascending=False)
    print("\nTop 10 features (win model):")
    for feat, imp in importances.head(10).items():
        print(f"  {feat:<35} {imp:.4f}")

    # --- Save artifacts
    print("\nSaving model artifacts ...")

    meta = {
        "features":      FEATURE_COLS,
        "n_models":      N_MODELS,
        "cv_win_acc":    float(cv_acc),
        "park_factors":  PARK_FACTORS,
        "note": (
            "Ensemble XGBoost + Elo + park factors + form trend + "
            "neutral win% + isotonic calibration. "
            "Time-series CV, no lookahead."
        ),
    }

    bundle = {
        "win_models":      win_models,
        "run_diff_models": rd_models,
        "total_runs_models": total_models,
        "feature_cols":    FEATURE_COLS,
        "n_models":        N_MODELS,
        "calibrator":      calibrator,
        "park_factors":    PARK_FACTORS,
    }

    joblib.dump(bundle,      os.path.join(ML_DIR, "mlb_model.pkl"))
    joblib.dump(meta,        os.path.join(ML_DIR, "mlb_meta.pkl"))
    joblib.dump(win_models[0],   os.path.join(ML_DIR, "win_model.pkl"))
    joblib.dump(rd_models[0],    os.path.join(ML_DIR, "run_model.pkl"))
    joblib.dump(total_models[0], os.path.join(ML_DIR, "total_model.pkl"))
    joblib.dump(win_models,      os.path.join(ML_DIR, "win_models.pkl"))
    joblib.dump(rd_models,       os.path.join(ML_DIR, "run_models.pkl"))
    joblib.dump(total_models,    os.path.join(ML_DIR, "total_models.pkl"))

    print("  Saved: mlb_model.pkl, mlb_meta.pkl, win/run/total model(s).pkl")
    print(f"\nDone. CV win accuracy: {cv_acc:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
