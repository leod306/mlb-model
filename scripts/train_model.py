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

features = [
    "era_diff",
    "whip_diff",
    "home_last10_runs_scored",
    "away_last10_runs_scored",
    "home_last10_runs_allowed",
    "away_last10_runs_allowed",
    "home_last10_run_diff",
    "away_last10_run_diff",
]

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns in training_data.csv: {missing}")

df = df.dropna(subset=features)

if "game_date" not in df.columns:
    raise ValueError("training_data.csv must include 'game_date'.")

df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
df = df.dropna(subset=["game_date"]).sort_values("game_date")

split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[features]
X_test = test[features]

print("Training run differential model...")
if "run_diff" not in df.columns:
    raise ValueError("training_data.csv must include 'run_diff'.")
y_train = train["run_diff"]
y_test = test["run_diff"]

run_model = XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42
)
run_model.fit(X_train, y_train)
preds = run_model.predict(X_test)
print("Run Diff MAE:", round(mean_absolute_error(y_test, preds), 3))

print("Training total runs model...")
if "total_runs" not in df.columns:
    raise ValueError("training_data.csv must include 'total_runs'.")
y_train = train["total_runs"]
y_test = test["total_runs"]

total_model = XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42
)
total_model.fit(X_train, y_train)
preds = total_model.predict(X_test)
print("Total Runs MAE:", round(mean_absolute_error(y_test, preds), 3))

print("Training win probability model...")
if "home_win" not in df.columns:
    raise ValueError("training_data.csv must include 'home_win'.")
y_train = train["home_win"].astype(int)
y_test = test["home_win"].astype(int)

win_model = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42,
    eval_metric="logloss"
)
win_model.fit(X_train, y_train)
preds = win_model.predict(X_test)
print("Win Model Accuracy:", round(accuracy_score(y_test, preds), 4))

ml_dir = os.path.join(PROJECT_ROOT, "ml")
os.makedirs(ml_dir, exist_ok=True)

run_path = os.path.join(ml_dir, "run_model.pkl")
total_path = os.path.join(ml_dir, "total_model.pkl")
win_path = os.path.join(ml_dir, "win_model.pkl")
meta_path = os.path.join(ml_dir, "mlb_meta.pkl")
alias_path = os.path.join(ml_dir, "mlb_model.pkl")  # engine loads this

joblib.dump(run_model, run_path)
joblib.dump(total_model, total_path)
joblib.dump(win_model, win_path)

meta = {"features": features, "team_classes_": le.classes_.tolist()}
joblib.dump(meta, meta_path)

joblib.dump(win_model, alias_path)

print("\nSaved models to /ml:")
print(" -", run_path)
print(" -", total_path)
print(" -", win_path)
print(" -", meta_path)
print(" -", alias_path, "(engine alias)")