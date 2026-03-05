import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pybaseball import pitching_stats

# --------------------------------------------------
# Setup
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")

load_dotenv(env_path)
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

# --------------------------------------------------
# Load 2022–2025 Pitching Data
# --------------------------------------------------

print("Pulling pitching stats...")

years = [2022, 2023, 2024, 2025]
all_pitchers = []

for year in years:
    print(f"Loading {year}...")
    df = pitching_stats(year)
    df["season"] = year
    all_pitchers.append(df)

pitchers = pd.concat(all_pitchers, ignore_index=True)

# Keep important columns
pitchers = pitchers[
    [
        "Name",
        "Season",
        "ERA",
        "WHIP",
        "SO",
        "BB",
        "IP",
    ]
]

pitchers.columns = [
    "pitcher_name",
    "season",
    "era",
    "whip",
    "strikeouts",
    "walks",
    "innings_pitched",
]

print("Saving to database...")

pitchers.to_sql("pitchers", engine, if_exists="replace", index=False)

print("Pitcher table created successfully.")