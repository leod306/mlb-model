import os
import joblib
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
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    le.fit(teams)
else:
    le.fit(["placeholder"])

# ---------------------------------------------------------------------------
# Feature set
#
# PHASE 1 (Opening Day through ~April 20):
#   Rolling stats removed — all zeros until 2026 games accumulate.
#   Model uses pitcher quality + situational features only.
#
# PHASE 2 (after April 20 + retrain):
#   Uncomment ROLLING_FEATURES and add back to features list.
#   Run build_dataset.py + train_model.py again.
# ---------------------------------------------------------------------------

PITCHER_FEATURES = [
    "era_diff",               # home SP ERA minus away SP ERA
    "whip_diff",              # home SP WHIP minus away SP WHIP
    "home_sp_rest_days",      # days since home SP last start
    "away_sp_rest_days",      # days since away SP last start
    "home_bullpen_ip_4d",     # home bullpen IP last 4 days (fatigue)
    "away_bullpen_ip_4d",     # away bullpen IP last 4 days (fatigue)
]

SITUATIONAL_FEATURES = [
    "home_win_pct_home",      # home team win% at home (historical)
    "away_win_pct_away",      # away team win% on road (historical)
]

# ROLLING_FEATURES — re-enable after April 20 once 2026 data accumulates
# ROLLING_FEATURES = [
#     "home_last10_runs_scored",
#     "away_last10_runs_scored",
#     "home_last10_runs_allowed",
#     "away_last10_runs_allowed",
#     "home_last10_run_diff",
#     "away_last10_run_diff",
# ]

features = PITCHER_FEATURES + SITUATIONAL_FEATURES
# features = PITCHER_FEATURES + SITUATIONAL_FEATURES + ROLLING_FEATURES  # phase 2

print(f"\nPhase 1 feature set ({len(features)} features):")
for f in features:
    print(f"  - {f}")
print()

# ---------------------------------------------------------------------------
# Fill missing features with sensible defaults
# (covers older training CSVs that don't have new columns yet)
# ---------------------------------------------------------------------------

defaults = {
    "home_sp_rest_days":  5.0,
    "away_sp_rest_days":  5.0,
    "home_bullpen_ip_4d": 4.0,
    "away_bullpen_ip_4d": 4.0,
    "home_win_pct_home":  0.5,
    "away_win_pct_away":  0.5,
}

for col, default in defaults.items():
    if col not in df.columns:
        print(f"  ⚠️  '{col}' not in dataset — filling with {default}")
        df[col] = default

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns in training_data.csv: {missing}")

df = df.dropna(subset=["era_diff", "whip_diff"])

if "game_date" not in df.columns:
    raise ValueError("training_data.csv must include 'game_date'.")

df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
df = df.dropna(subset=["game_date"]).sort_values("game_date")

split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test  = df.iloc[split_index:]

X_train = train[features]
X_test  = test[features]

print(f"Training on {len(train)} games, testing on {len(test)} games\n")

# ---------------------------------------------------------------------------
# Run differential model
# ---------------------------------------------------------------------------
print("Training run differential model...")
if "run_diff" not in df.columns:
    raise ValueError("training_data.csv must include 'run_diff'.")

run_model = XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42
)
run_model.fit(X_train, train["run_diff"])
preds = run_model.predict(X_test)
print(f"Run Diff MAE: {round(mean_absolute_error(test['run_diff'], preds), 3)}")

# ---------------------------------------------------------------------------
# Total runs model
# ---------------------------------------------------------------------------
print("Training total runs model...")
if "total_runs" not in df.columns:
    raise ValueError("training_data.csv must include 'total_runs'.")

total_model = XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42
)
total_model.fit(X_train, train["total_runs"])
preds = total_model.predict(X_test)
print(f"Total Runs MAE: {round(mean_absolute_error(test['total_runs'], preds), 3)}")

# ---------------------------------------------------------------------------
# Win probability model
# ---------------------------------------------------------------------------
print("Training win probability model...")
if "home_win" not in df.columns:
    raise ValueError("training_data.csv must include 'home_win'.")

win_model = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42,
    eval_metric="logloss"
)
win_model.fit(X_train, train["home_win"].astype(int))
preds = win_model.predict(X_test)
print(f"Win Model Accuracy: {round(accuracy_score(test['home_win'].astype(int), preds), 4)}")

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
print("\nFeature importances (win model):")
importances = dict(zip(features, win_model.feature_importances_))
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp * 200)
    print(f"  {feat:<30} {imp:.4f}  {bar}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
ml_dir = os.path.join(PROJECT_ROOT, "ml")
os.makedirs(ml_dir, exist_ok=True)

run_path   = os.path.join(ml_dir, "run_model.pkl")
total_path = os.path.join(ml_dir, "total_model.pkl")
win_path   = os.path.join(ml_dir, "win_model.pkl")
meta_path  = os.path.join(ml_dir, "mlb_meta.pkl")
alias_path = os.path.join(ml_dir, "mlb_model.pkl")

joblib.dump(run_model,   run_path)
joblib.dump(total_model, total_path)
joblib.dump(win_model,   win_path)

meta = {
    "features":     features,
    "phase":        1,
    "team_classes_": le.classes_.tolist(),
    "note":         "Phase 1 — rolling stats excluded until 2026 data accumulates (~April 20)",
}
joblib.dump(meta, meta_path)
joblib.dump(win_model, alias_path)

print("\nSaved models to /ml:")
print(f"  {run_path}")
print(f"  {total_path}")
print(f"  {win_path}")
print(f"  {meta_path}")
print(f"  {alias_path} (engine alias)")
print(f"\nPhase 1 — {len(features)} features active")
print("Reminder: re-enable ROLLING_FEATURES and retrain after April 20")