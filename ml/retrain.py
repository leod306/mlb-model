"""
ml/retrain.py  —  MLB model retraining with market-aware features
Run from the project root:
    python ml/retrain.py

Changes in this version (betting-layer overhaul):
  1. MARKET FEATURES — de-vigged market win probability and the market total
     line are now model inputs (when coverage in training_data.csv is high
     enough). The model learns to predict the RESIDUAL vs the market instead
     of fighting the closing line from scratch.
  2. OUT-OF-FOLD CALIBRATION — the Platt calibrator is now fit on true
     out-of-fold predictions from the time-series CV (never on data the
     models trained on). The engine MUST apply this calibrator.
  3. RESIDUAL SIGMAS — sigma_total and sigma_rd (std dev of out-of-fold
     residuals) are saved in the bundle. These are the REAL uncertainty of a
     prediction, used by the engine to compute P(over), P(cover), etc.
     (Ensemble seed-spread is NOT uncertainty — 20 same-data models with
     different seeds barely disagree.)
  4. MEASURED TOTAL BIAS — mean(pred - actual) on out-of-fold totals is
     saved as total_bias. The engine subtracts it. This replaces the
     hand-tuned TOTAL_CALIBRATION=-1.2 fudge with a measured number.
  5. FINAL ENSEMBLE TRAINS ON ALL DATA — previously the last 20% was held
     out for calibration and never used for training. OOF calibration frees
     that data back up.
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

# Market features are only included if at least this fraction of training
# rows have real market data (avoids training on a constant default column).
MARKET_COVERAGE_MIN = 0.40

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
ELO_START    = 1500.0
ELO_K        = 20.0
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
# 3. Market feature helpers — de-vig the moneyline
# ---------------------------------------------------------------------------
def american_to_prob(ml) -> float | None:
    """Convert American odds to implied probability (with vig)."""
    try:
        m = float(ml)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(m) or m == 0:
        return None
    if m < 0:
        return -m / (-m + 100.0)
    return 100.0 / (m + 100.0)


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      market_home_prob_novig — vig-removed home win probability
      market_total_line      — already in CSV if odds were joined at build time

    De-vig by normalizing the two implied probabilities so they sum to 1.
    Priority: use market_home_prob/market_away_prob columns if present,
    otherwise derive from market_home_ml/market_away_ml.
    """
    df = df.copy()

    hp = pd.to_numeric(df.get("market_home_prob"), errors="coerce") \
        if "market_home_prob" in df.columns else pd.Series(np.nan, index=df.index)
    ap = pd.to_numeric(df.get("market_away_prob"), errors="coerce") \
        if "market_away_prob" in df.columns else pd.Series(np.nan, index=df.index)

    # fall back to moneylines where prob columns are missing
    if "market_home_ml" in df.columns:
        hp_ml = df["market_home_ml"].map(american_to_prob)
        ap_ml = df["market_away_ml"].map(american_to_prob) if "market_away_ml" in df.columns else np.nan
        hp = hp.fillna(pd.to_numeric(hp_ml, errors="coerce"))
        ap = ap.fillna(pd.to_numeric(ap_ml, errors="coerce"))

    denom = hp + ap
    novig = np.where((denom > 0) & hp.notna() & ap.notna(), hp / denom, np.nan)
    df["market_home_prob_novig"] = novig

    if "market_total_line" in df.columns:
        df["market_total_line"] = pd.to_numeric(df["market_total_line"], errors="coerce")
    else:
        df["market_total_line"] = np.nan

    return df


# ---------------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Elo
    df = compute_elo(df)

    # --- Park factor
    df["home_park_factor"] = df["home_team"].map(PARK_FACTORS).fillna(1.0)

    # --- Market features (de-vigged)
    df = add_market_features(df)

    # --- Form trend: positive = team is heating up (last 5 better than last 10)
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
# 5. Feature columns
# ---------------------------------------------------------------------------
BASE_FEATURE_COLS = [
    # Elo — most predictive single signal
    "elo_diff",
    "home_elo_pre",
    "away_elo_pre",

    # Park
    "home_park_factor",

    # Pitcher quality
    "era_diff",
    "whip_diff",
    # as-of-date FIP from game_features (the backfill) — real pitching signal
    "sp_fip_diff",
    "bullpen_fip_diff",
    "offense_wrc_diff",
    "home_wrc_plus",
    "away_wrc_plus",
    "park_run_factor",


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

    # Weather — from game_weather table (backfilled via backfill_weather.py)
    # wind_out_factor: +ve = wind blowing out (inflates runs), -ve = blowing in
    # Domes are set to wind_out_factor=0, precip=0 before training.
    # NOTE: temp_f is intentionally excluded — it's a seasonal proxy (July is
    # always hot) that causes the total model to over-predict runs in summer.
    # Game-specific weather signals (wind direction, rain, visibility) are kept.
    "wind_out_factor",
    "precip_mm",
    "visibility_m",
]

# Included only if coverage in training data clears MARKET_COVERAGE_MIN.
MARKET_FEATURE_COLS = [
    "market_home_prob_novig",
    "market_total_line",
]

# Neutral fill values for market features when a row is missing odds.
# (0.0 would be a nonsense probability / line — these keep missing rows sane.)
MARKET_FILLS = {
    "market_home_prob_novig": 0.5,
    "market_total_line":      8.5,
}

# Neutral fill values for weather features when a row is missing weather data.
# These represent typical MLB conditions (outdoor, comfortable, calm).
WEATHER_FILLS = {
    "temp_f":           72.0,   # comfortable outdoor temp
    "wind_out_factor":   0.0,   # neutral crosswind
    "precip_mm":         0.0,   # no rain
    "visibility_m":  10000.0,   # clear visibility
}


# ---------------------------------------------------------------------------
# 6. Model configs
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
# 7. Time-series cross-validation
#    Now also collects OUT-OF-FOLD predictions, which drive:
#      - Platt calibration (fit on data the models never saw)
#      - sigma_total / sigma_rd (real predictive uncertainty)
#      - total_bias (measured, replaces the hand-tuned fudge)
# ---------------------------------------------------------------------------
def run_cv(X: pd.DataFrame, y_win: pd.Series, y_rd: pd.Series, y_tot: pd.Series):
    tscv = TimeSeriesSplit(n_splits=5, gap=30)  # 30-game gap prevents leakage

    win_logloss, win_brier, win_acc = [], [], []
    rd_mae, tot_mae = [], []

    oof_idx:      list[np.ndarray] = []
    oof_win_prob: list[np.ndarray] = []
    oof_rd_pred:  list[np.ndarray] = []
    oof_tot_pred: list[np.ndarray] = []

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
        rd_p = rm.predict(X_te)
        rd_mae.append(mean_absolute_error(y_rd_te, rd_p))

        tm = XGBRegressor(**REG_PARAMS)
        tm.fit(X_tr, y_tot_tr, verbose=False)
        tot_p = tm.predict(X_te)
        tot_mae.append(mean_absolute_error(y_tot_te, tot_p))

        oof_idx.append(te_idx)
        oof_win_prob.append(prob)
        oof_rd_pred.append(rd_p)
        oof_tot_pred.append(tot_p)

        print(f"  Fold {fold+1}: win_acc={win_acc[-1]:.3f}  logloss={win_logloss[-1]:.4f}  "
              f"rd_mae={rd_mae[-1]:.2f}  tot_mae={tot_mae[-1]:.2f}")

    print(f"\n  CV means: win_acc={np.mean(win_acc):.3f}  logloss={np.mean(win_logloss):.4f}  "
          f"brier={np.mean(win_brier):.4f}  rd_mae={np.mean(rd_mae):.2f}  tot_mae={np.mean(tot_mae):.2f}")

    oof = {
        "idx":      np.concatenate(oof_idx),
        "win_prob": np.concatenate(oof_win_prob),
        "rd_pred":  np.concatenate(oof_rd_pred),
        "tot_pred": np.concatenate(oof_tot_pred),
    }
    return np.mean(win_acc), oof


# ---------------------------------------------------------------------------
# 8. Platt scaling calibration — fit on OUT-OF-FOLD predictions
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """
    Sigmoid (Platt scaling) calibrator. Wraps a LogisticRegression so it
    exposes a .predict(proba_array) → calibrated_proba_array interface
    compatible with the engine.

    IMPORTANT: the engine must apply this. If the calibrator compresses raw
    26%/74% outputs toward 45-58%, that compression IS the honest signal —
    it means the raw model is overconfident on held-out data. Betting the
    raw probabilities against real prices loses to the vig.
    """
    def __init__(self, lr: LogisticRegression):
        self.lr = lr

    def predict(self, raw_proba):
        arr = np.asarray(raw_proba).reshape(-1, 1)
        return self.lr.predict_proba(arr)[:, 1]


def fit_calibrator_oof(oof_prob: np.ndarray, y_oof: np.ndarray) -> PlattCalibrator:
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(np.asarray(oof_prob).reshape(-1, 1), np.asarray(y_oof))
    return PlattCalibrator(lr)


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MLB Model Retrain (market-aware, OOF-calibrated)")
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

    # --- Decide whether market features make the cut
    feature_cols = list(BASE_FEATURE_COLS)
    market_coverage = float(df["market_home_prob_novig"].notna().mean())
    line_coverage   = float(df["market_total_line"].notna().mean())
    print(f"\nMarket data coverage: novig prob {market_coverage:.1%}, total line {line_coverage:.1%}")

    use_market = market_coverage >= MARKET_COVERAGE_MIN
    if use_market:
        feature_cols += MARKET_FEATURE_COLS
        print("  ✅ Market features INCLUDED — model will learn residual vs the closing line.")
    else:
        print("  ⚠️  Market features EXCLUDED (coverage too low).")
        print("     Backfill market_odds into training_data.csv via build_dataset.py —")
        print("     this is the single highest-value data improvement available.")

    # neutral fills for market and weather cols, 0.0 for everything else
    X = df[feature_cols].copy()
    for col, fill in {**MARKET_FILLS, **WEATHER_FILLS}.items():
        if col in X.columns:
            X[col] = X[col].fillna(fill)

    # Report weather coverage so we know how much signal is real vs filled
    for wcol in ["temp_f", "wind_out_factor", "precip_mm", "visibility_m"]:
        if wcol in df.columns:
            cov = df[wcol].notna().mean()
            if cov < 0.5:
                print(f"  ⚠️  Weather col {wcol}: only {cov:.1%} coverage — run backfill_weather.py")
            else:
                print(f"  ✅ Weather col {wcol}: {cov:.1%} coverage")

    X = X.fillna(0.0)

    y_win = df["home_win"].astype(int)
    y_rd  = df["run_diff"].fillna(0.0)
    y_tot = df["total_runs"].fillna(0.0)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Home win rate: {y_win.mean():.3f}")

    # --- Cross-validation + out-of-fold predictions
    print("\nTime-series cross-validation ...")
    cv_acc, oof = run_cv(X, y_win, y_rd, y_tot)

    idx        = oof["idx"]
    y_win_oof  = y_win.iloc[idx].to_numpy()
    y_rd_oof   = y_rd.iloc[idx].to_numpy()
    y_tot_oof  = y_tot.iloc[idx].to_numpy()

    # --- Real predictive uncertainty from OOF residuals
    rd_resid  = y_rd_oof  - oof["rd_pred"]
    tot_resid = y_tot_oof - oof["tot_pred"]

    sigma_rd    = float(np.std(rd_resid))
    sigma_total = float(np.std(tot_resid))
    total_bias  = float(np.mean(oof["tot_pred"] - y_tot_oof))   # + = model over-predicts runs
    rd_bias     = float(np.mean(oof["rd_pred"]  - y_rd_oof))

    print("\nOut-of-fold residual diagnostics:")
    print(f"  sigma_total = {sigma_total:.2f} runs   (real O/U uncertainty — NOT ensemble spread)")
    print(f"  sigma_rd    = {sigma_rd:.2f} runs   (real margin uncertainty)")
    print(f"  total_bias  = {total_bias:+.2f} runs  (engine subtracts this; replaces TOTAL_CALIBRATION fudge)")
    print(f"  rd_bias     = {rd_bias:+.2f} runs")
    if abs(total_bias) > 0.5:
        print("  ⚠️  Total bias > 0.5 runs — check train/serve feature skew "
              "(blended prior-season baselines vs true rolling stats).")

    # --- Calibrate win probabilities on OOF predictions
    print("\nFitting Platt calibration on out-of-fold predictions ...")
    calibrator = fit_calibrator_oof(oof["win_prob"], y_win_oof)

    cal_prob = calibrator.predict(oof["win_prob"])
    print(f"  Pre-cal  log_loss: {log_loss(y_win_oof, oof['win_prob']):.4f}  "
          f"brier: {brier_score_loss(y_win_oof, oof['win_prob']):.4f}")
    print(f"  Post-cal log_loss: {log_loss(y_win_oof, cal_prob):.4f}  "
          f"brier: {brier_score_loss(y_win_oof, cal_prob):.4f}")
    print(f"  Raw mean prob: {oof['win_prob'].mean():.3f}  Cal mean: {cal_prob.mean():.3f}  "
          f"Actual: {y_win_oof.mean():.3f}")
    print(f"  Raw spread (5th-95th pct): {np.percentile(oof['win_prob'],5):.2f}–{np.percentile(oof['win_prob'],95):.2f}"
          f"  →  Cal: {np.percentile(cal_prob,5):.2f}–{np.percentile(cal_prob,95):.2f}")
    print("  (If calibration compresses the spread a lot, the raw model was overconfident.")
    print("   The compressed probabilities are the ones that pay against real prices.)")

    # --- Train final ensemble on ALL data (OOF calibration freed up the holdout)
    print(f"\nTraining {N_MODELS}-model ensemble on all {len(X)} games ...")

    win_models   = []
    rd_models    = []
    total_models = []

    for i in range(N_MODELS):
        seed = 1000 + i * 7

        wm = XGBClassifier(**{**WIN_PARAMS, "random_state": seed})
        wm.fit(X, y_win, verbose=False)
        win_models.append(wm)

        rm = XGBRegressor(**{**REG_PARAMS, "random_state": seed})
        rm.fit(X, y_rd, verbose=False)
        rd_models.append(rm)

        tm = XGBRegressor(**{**REG_PARAMS, "random_state": seed})
        tm.fit(X, y_tot, verbose=False)
        total_models.append(tm)

        if (i + 1) % 5 == 0:
            print(f"  Trained {i+1}/{N_MODELS}")

    # --- Feature importance
    importances = pd.Series(
        np.mean([m.feature_importances_ for m in win_models], axis=0),
        index=feature_cols,
    ).sort_values(ascending=False)
    print("\nTop 10 features (win model):")
    for feat, imp in importances.head(10).items():
        print(f"  {feat:<35} {imp:.4f}")

    # --- Save artifacts
    print("\nSaving model artifacts ...")

    meta = {
        "features":        feature_cols,
        "n_models":        N_MODELS,
        "cv_win_acc":      float(cv_acc),
        "park_factors":    PARK_FACTORS,
        "sigma_total":     sigma_total,
        "sigma_rd":        sigma_rd,
        "total_bias":      total_bias,
        "rd_bias":         rd_bias,
        "uses_market":     use_market,
        "market_fills":    {**MARKET_FILLS, **WEATHER_FILLS},
        "note": (
            "Ensemble XGBoost + Elo + park factors + form trend + neutral win% "
            "+ de-vigged market features. Platt calibration fit on out-of-fold "
            "time-series CV predictions. sigma_total/sigma_rd are OOF residual "
            "std devs — use for P(over)/P(cover). Engine MUST apply calibrator."
        ),
    }

    bundle = {
        "win_models":        win_models,
        "run_diff_models":   rd_models,
        "total_runs_models": total_models,
        "feature_cols":      feature_cols,
        "n_models":          N_MODELS,
        "calibrator":        calibrator,
        "park_factors":      PARK_FACTORS,
        "sigma_total":       sigma_total,
        "sigma_rd":          sigma_rd,
        "total_bias":        total_bias,
        "rd_bias":           rd_bias,
        "uses_market":       use_market,
        "market_fills":      MARKET_FILLS,
    }

    joblib.dump(bundle,          os.path.join(ML_DIR, "mlb_model.pkl"))
    joblib.dump(meta,            os.path.join(ML_DIR, "mlb_meta.pkl"))
    joblib.dump(win_models[0],   os.path.join(ML_DIR, "win_model.pkl"))
    joblib.dump(rd_models[0],    os.path.join(ML_DIR, "run_model.pkl"))
    joblib.dump(total_models[0], os.path.join(ML_DIR, "total_model.pkl"))
    joblib.dump(win_models,      os.path.join(ML_DIR, "win_models.pkl"))
    joblib.dump(rd_models,       os.path.join(ML_DIR, "run_models.pkl"))
    joblib.dump(total_models,    os.path.join(ML_DIR, "total_models.pkl"))

    print("  Saved: mlb_model.pkl, mlb_meta.pkl, win/run/total model(s).pkl")
    print(f"\nDone. CV win accuracy: {cv_acc:.3f}  |  sigma_total: {sigma_total:.2f}  |  "
          f"total_bias: {total_bias:+.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()