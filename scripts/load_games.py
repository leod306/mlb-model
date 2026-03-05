import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# --------------------------------------------------
# Setup
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")

load_dotenv(env_path)
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

# --------------------------------------------------
# Pull MLB Schedule with Starters
# --------------------------------------------------

YEAR = 2025

print(f"Pulling {YEAR} season from MLB API...")

url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={YEAR}"

response = requests.get(url)
data = response.json()

games_data = []

for date in data["dates"]:
    for game in date["games"]:

        home_team = game["teams"]["home"]["team"]["name"]
        away_team = game["teams"]["away"]["team"]["name"]

        home_score = game["teams"]["home"].get("score", None)
        away_score = game["teams"]["away"].get("score", None)

        # Probable starters (may not exist)
        home_pitcher = None
        away_pitcher = None

        if "probablePitcher" in game["teams"]["home"]:
            home_pitcher = game["teams"]["home"]["probablePitcher"]["fullName"]

        if "probablePitcher" in game["teams"]["away"]:
            away_pitcher = game["teams"]["away"]["probablePitcher"]["fullName"]

        games_data.append({
            "game_date": date["date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "home_pitcher": home_pitcher,
            "away_pitcher": away_pitcher
        })

df = pd.DataFrame(games_data)

print("Saving to database...")

df.to_sql("games", engine, if_exists="replace", index=False)

print("Games table updated with pitchers.")