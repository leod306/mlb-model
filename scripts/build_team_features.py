import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

engine = create_engine(os.getenv("DATABASE_URL"))

print("Loading games...")
games = pd.read_sql("SELECT * FROM games", engine)

games["game_date"] = pd.to_datetime(games["game_date"])

# Create long format (one row per team per game)
home = games[["game_date","home_team","home_score","away_score"]].copy()
home.columns = ["game_date","team","runs_scored","runs_allowed"]

away = games[["game_date","away_team","away_score","home_score"]].copy()
away.columns = ["game_date","team","runs_scored","runs_allowed"]

df = pd.concat([home, away])
df = df.sort_values(["team","game_date"])

# Rolling features (last 10 games)
df["runs_scored_avg_10"] = (
    df.groupby("team")["runs_scored"]
    .rolling(10, min_periods=3)
    .mean()
    .reset_index(level=0, drop=True)
)

df["runs_allowed_avg_10"] = (
    df.groupby("team")["runs_allowed"]
    .rolling(10, min_periods=3)
    .mean()
    .reset_index(level=0, drop=True)
)

df = df.dropna()

df.to_sql("team_features_daily", engine, if_exists="replace", index=False)

print("Team features created.")
print("Row count:", len(df))