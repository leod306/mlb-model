import pandas as pd
from datetime import date
from app.db import engine
from app.predictor import predict_from_features


today = date.today()

# Fix for TEXT date column
games = pd.read_sql(
    "SELECT * FROM games WHERE game_date::date = CURRENT_DATE",
    engine
)

if len(games) == 0:
    print("No games found for today.")
    exit()

results = []

for _, row in games.iterrows():

    features = pd.DataFrame([{
        "era_diff": row["home_era"] - row["away_era"],
        "whip_diff": row["home_whip"] - row["away_whip"],
        "home_last10_runs_scored": row["home_last10_runs_scored"],
        "away_last10_runs_scored": row["away_last10_runs_scored"],
        "home_last10_runs_allowed": row["home_last10_runs_allowed"],
        "away_last10_runs_allowed": row["away_last10_runs_allowed"],
        "home_last10_run_diff": row["home_last10_run_diff"],
        "away_last10_run_diff": row["away_last10_run_diff"]
    }])

    prediction = predict_from_features(features)

    results.append({
        "game_date": row["game_date"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "predicted_run_diff": prediction["predicted_run_diff"],
        "predicted_total_runs": prediction["predicted_total_runs"],
        "home_win_probability": prediction["home_win_probability"]
    })


pred_df = pd.DataFrame(results)

print(pred_df)

# Save predictions
pred_df.to_sql(
    "predictions",
    engine,
    if_exists="append",
    index=False
)

print("\nPredictions saved.")