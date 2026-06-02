import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print("PROJECT_ROOT =", PROJECT_ROOT)
print("Loading dataset...")

train_csv = os.path.join(PROJECT_ROOT, "ml", "training_data.csv")
if not os.path.exists(train_csv):
    raise FileNotFoundError(f"Missing training dataset: {train_csv}")

df = pd.read_csv(train_csv)
if len(df) == 0:
    raise ValueError("Dataset is empty. Build dataset first.")

le = LabelEncoder()
if "home_team" in df.columns and "away_team" in df.columns:
    teams = pd.concat([df["home_team"], df["away_team"]]).dropna().unique()
    le.fit(teams)
else:
    le.fit(["placeholder"])


# ---------------------------------------------------------------------------
# Compute differential features inline from CSV columns
# NOTE: run_diff_form_diff, runs_scored_diff, runs_allowed_diff, win_pct_diff
# are intentionally NOT computed or included — they duplicate signals already
# in the raw home/away rolling features and cause triple-counting.
# ---------------------------------------------------------------------------

def add_diff(df, col_a, col_b, new_col, default_a=0.0, default_b=0.0):
    a = pd.to_numeric(df[col_a], errors="coerce").fillna(default_a) if col_a in df.columns else pd.Series(default_a, index=df.index)
    b = pd.to_numeric(df[col_b], errors="coerce").fillna(default_b) if col_b in df.columns else pd.Series(default_b, index=df.index)
    df[new_col] = a - b
    return df

# Keep: bullpen diff, rest diff, O/U diffs, ATS diff — these add directional
# signal not already present as individual raw features
df = add_diff(df, "home_sp_rest_days",    "away_sp_rest_days",    "sp_rest_diff",         5.0, 5.0)
df = add_diff(df, "home_bullpen_ip_4d",   "away_bullpen_ip_4d",   "bullpen_usage_diff",   4.0, 4.0)
df = add_diff(df, "home_ou_over_rate",    "away_ou_over_rate",    "ou_over_rate_diff",    0.5, 0.5)
df = add_diff(df, "home_last_game_total", "away_last_game_total", "last_game_total_diff", 8.5, 8.5)
df = add_diff(df, "home_ats_cover_rate",  "away_ats_cover_rate",  "ats_cover_rate_diff",  0.5, 0.5)

# Lineup OPS diff — computed from BvP-weighted lineup scores
# Falls back to 0.0 (neutral) if not in training data yet
df = add_diff(df, "home_lineup_ops_vs_sp", "away_lineup_ops_vs_sp", "lineup_ops_diff", 0.720, 0.720)


# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------

PITCHER_FEATURES = [
    "era_diff",
    "whip_diff",
    "home_sp_rest_days",
    "away_sp_rest_days",
    "home_bullpen_ip_4d",
    "away_bullpen_ip_4d",
]

SITUATIONAL_FEATURES = [
    "home_win_pct_home",
    "away_win_pct_away",
]

ROLLING_FEATURES = [
    "home_last10_runs_scored",
    "away_last10_runs_scored",
    "home_last10_runs_allowed",
    "away_last10_runs_allowed",
    "home_last10_run_diff",
    "away_last10_run_diff",
]

# Kept diffs: these add directional signal beyond what raw features provide
# Removed: run_diff_form_diff, runs_scored_diff, runs_allowed_diff, win_pct_diff
#   — those were near-perfect duplicates of raw rolling/win_pct features
DIFFERENTIAL_FEATURES = [
    "sp_rest_diff",
    "bullpen_usage_diff",
]

BETTING_FEATURES = [
    "home_ou_over_rate",
    "away_ou_over_rate",
    "ou_over_rate_diff",
    "home_last_game_total",
    "away_last_game_total",
    "last_game_total_diff",
    "home_ats_cover_rate",
    "away_ats_cover_rate",
    "ats_cover_rate_diff",
]

# Lineup quality features — BvP-weighted OPS vs opposing SP
# These will be 0.720 (league avg) for games without lineup data,
# and real matchup values for games with BvP coverage
LINEUP_FEATURES = [
    "home_lineup_ops_vs_sp",
    "away_lineup_ops_vs_sp",
    "lineup_ops_diff",
    "home_lineup_hard_hit",
    "away_lineup_hard_hit",
]

FEATURES = (
    PITCHER_FEATURES +
    SITUATIONAL_FEATURES +
    ROLLING_FEATURES +
    DIFFERENTIAL_FEATURES +
    BETTING_FEATURES +
    LINEUP_FEATURES
)

print(f"\nFeature set ({len(FEATURES)} features):")
for f in FEATURES:
    print(f"  - {f}")
print()


# ---------------------------------------------------------------------------
# Defaults / validation
# ---------------------------------------------------------------------------

defaults = {
    "era_diff":                 0.0,
    "whip_diff":                0.0,
    "home_sp_rest_days":        5.0,
    "away_sp_rest_days":        5.0,
    "home_bullpen_ip_4d":       4.0,
    "away_bullpen_ip_4d":       4.0,
    "home_win_pct_home":        0.5,
    "away_win_pct_away":        0.5,
    "home_last10_runs_scored":  4.5,
    "away_last10_runs_scored":  4.5,
    "home_last10_runs_allowed": 4.5,
    "away_last10_runs_allowed": 4.5,
    "home_last10_run_diff":     0.0,
    "away_last10_run_diff":     0.0,
    "sp_rest_diff":             0.0,
    "bullpen_usage_diff":       0.0,
    "home_ou_over_rate":        0.5,
    "away_ou_over_rate":        0.5,
    "ou_over_rate_diff":        0.0,
    "home_last_game_total":     8.5,
    "away_last_game_total":     8.5,
    "last_game_total_diff":     0.0,
    "home_ats_cover_rate":      0.5,
    "away_ats_cover_rate":      0.5,
    "ats_cover_rate_diff":      0.0,
    # Lineup features — default to league avg / neutral when not available
    "home_lineup_ops_vs_sp":    0.720,
    "away_lineup_ops_vs_sp":    0.720,
    "lineup_ops_diff":          0.0,
    "home_lineup_hard_hit":     0.35,
    "away_lineup_hard_hit":     0.35,
}

for col, default in defaults.items():
    if col not in df.columns:
        print(f"  ⚠️  '{col}' not in dataset — filling with {default}")
        df[col] = default

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns in training_data.csv: {missing}")

required_targets = ["game_date", "run_diff", "total_runs", "home_win"]
for col in required_targets:
    if col not in df.columns:
        raise ValueError(f"training_data.csv must include '{col}'.")

df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
df = df.dropna(subset=["game_date"]).sort_values("game_date").copy()

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[FEATURES] = df[FEATURES].fillna(defaults)

# ---------------------------------------------------------------------------
# Report lineup feature coverage in training data
# ---------------------------------------------------------------------------
lineup_coverage = (df["home_lineup_ops_vs_sp"] != 0.720).mean()
print(f"Lineup feature coverage in training data: {lineup_coverage:.1%}")
if lineup_coverage < 0.05:
    print("  ⚠️  Most lineup features are defaulting to 0.720 (league avg).")
    print("  ⚠️  Model will learn limited lineup signal — coverage improves as BvP data accumulates.")
else:
    print(f"  ✓ {lineup_coverage:.1%} of training rows have real BvP lineup scores")

split_index = int(len(df) * 0.8)
train = df.iloc[:split_index].copy()
test  = df.iloc[split_index:].copy()

X_train = train[FEATURES].copy()
X_test  = test[FEATURES].copy()

y_run_train   = train["run_diff"].copy()
y_run_test    = test["run_diff"].copy()
y_total_train = train["total_runs"].copy()
y_total_test  = test["total_runs"].copy()
y_win_train   = train["home_win"].astype(int).copy()
y_win_test    = test["home_win"].astype(int).copy()

print(f"Training on {len(train)} games, testing on {len(test)} games\n")


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

N_MODELS = 20

def bootstrap_sample(X: pd.DataFrame, y: pd.Series, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=len(X), replace=True)
    return X.iloc[idx], y.iloc[idx]


def train_run_ensemble(X, y, n_models=N_MODELS):
    models = []
    for i in range(n_models):
        X_boot, y_boot = bootstrap_sample(X, y, seed=1000 + i)
        model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=1000 + i,
        )
        model.fit(X_boot, y_boot)
        models.append(model)
        print(f"  Run model {i+1}/{n_models} trained")
    return models


def train_total_ensemble(X, y, n_models=N_MODELS):
    models = []
    for i in range(n_models):
        X_boot, y_boot = bootstrap_sample(X, y, seed=2000 + i)
        model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=2000 + i,
        )
        model.fit(X_boot, y_boot)
        models.append(model)
        print(f"  Total model {i+1}/{n_models} trained")
    return models


def train_win_ensemble(X, y, n_models=N_MODELS):
    models = []
    for i in range(n_models):
        X_boot, y_boot = bootstrap_sample(X, y, seed=3000 + i)
        model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=3000 + i, eval_metric="logloss",
        )
        model.fit(X_boot, y_boot)
        models.append(model)
        print(f"  Win model {i+1}/{n_models} trained")
    return models


def ensemble_regression_predict(models, X):
    preds     = np.column_stack([m.predict(X) for m in models])
    mean_pred = preds.mean(axis=1)
    p05       = np.percentile(preds, 5,  axis=1)
    p95       = np.percentile(preds, 95, axis=1)
    std       = preds.std(axis=1)
    return mean_pred, p05, p95, std


def ensemble_classification_predict(models, X):
    preds     = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    mean_pred = preds.mean(axis=1)
    p05       = np.percentile(preds, 5,  axis=1)
    p95       = np.percentile(preds, 95, axis=1)
    std       = preds.std(axis=1)
    return mean_pred, p05, p95, std


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

print("Training run differential ensemble...")
run_models = train_run_ensemble(X_train, y_run_train)
run_mean, run_lo, run_hi, run_std = ensemble_regression_predict(run_models, X_test)
print(f"Run Diff Ensemble MAE: {round(mean_absolute_error(y_run_test, run_mean), 3)}\n")

print("Training total runs ensemble...")
total_models = train_total_ensemble(X_train, y_total_train)
total_mean, total_lo, total_hi, total_std = ensemble_regression_predict(total_models, X_test)
print(f"Total Runs Ensemble MAE: {round(mean_absolute_error(y_total_test, total_mean), 3)}\n")

print("Training win probability ensemble...")
win_models = train_win_ensemble(X_train, y_win_train)
win_mean, win_lo, win_hi, win_std = ensemble_classification_predict(win_models, X_test)
win_preds = (win_mean >= 0.5).astype(int)
print(f"Win Ensemble Accuracy: {round(accuracy_score(y_win_test, win_preds), 4)}\n")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

print("Feature importances (first win model):")
importances = dict(zip(FEATURES, win_models[0].feature_importances_))
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp * 200)
    print(f"  {feat:<30} {imp:.4f}  {bar}")

print("\nFeature importances (first total model):")
importances_t = dict(zip(FEATURES, total_models[0].feature_importances_))
for feat, imp in sorted(importances_t.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp * 200)
    print(f"  {feat:<30} {imp:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

ml_dir = os.path.join(PROJECT_ROOT, "ml")
os.makedirs(ml_dir, exist_ok=True)

run_models_path   = os.path.join(ml_dir, "run_models.pkl")
total_models_path = os.path.join(ml_dir, "total_models.pkl")
win_models_path   = os.path.join(ml_dir, "win_models.pkl")
meta_path         = os.path.join(ml_dir, "mlb_meta.pkl")
alias_path        = os.path.join(ml_dir, "mlb_model.pkl")

joblib.dump(run_models,   run_models_path)
joblib.dump(total_models, total_models_path)
joblib.dump(win_models,   win_models_path)

meta = {
    "features":      FEATURES,
    "team_classes_": le.classes_.tolist(),
    "note":          "Ensemble — pitcher + situational + rolling + differential + betting + lineup features. "
                     "Removed redundant diffs: run_diff_form_diff, runs_scored_diff, runs_allowed_diff, win_pct_diff.",
    "n_models":      N_MODELS,
}
joblib.dump(meta, meta_path)

bundle = {
    "win_models":        win_models,
    "run_diff_models":   run_models,
    "total_runs_models": total_models,
    "feature_cols":      FEATURES,
    "n_models":          N_MODELS,
}
joblib.dump(bundle, alias_path)

print("\nSaved ensemble models to /ml:")
print(f"  {run_models_path}")
print(f"  {total_models_path}")
print(f"  {win_models_path}")
print(f"  {meta_path}")
print(f"  {alias_path}")

print("\nEnsemble interval preview on test set:")
preview = pd.DataFrame({
    "win_mean":    win_mean[:10],
    "win_lo_5":    win_lo[:10],
    "win_hi_95":   win_hi[:10],
    "total_mean":  total_mean[:10],
    "total_lo_5":  total_lo[:10],
    "total_hi_95": total_hi[:10],
    "run_mean":    run_mean[:10],
})
print(preview.round(3).to_string(index=False))

print(f"\nDone — {len(FEATURES)} features active, {N_MODELS} models per ensemble")