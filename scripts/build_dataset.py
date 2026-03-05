import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")

load_dotenv(env_path)
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

MLB_TEAMS = [
    "Arizona Diamondbacks","Atlanta Braves","Baltimore Orioles",
    "Boston Red Sox","Chicago Cubs","Chicago White Sox",
    "Cincinnati Reds","Cleveland Guardians","Colorado Rockies",
    "Detroit Tigers","Houston Astros","Kansas City Royals",
    "Los Angeles Angels","Los Angeles Dodgers","Miami Marlins",
    "Milwaukee Brewers","Minnesota Twins","New York Mets",
    "New York Yankees","Oakland Athletics","Philadelphia Phillies",
    "Pittsburgh Pirates","San Diego Padres","San Francisco Giants",
    "Seattle Mariners","St. Louis Cardinals","Tampa Bay Rays",
    "Texas Rangers","Toronto Blue Jays","Washington Nationals"
]

def clean_name(name):
    if pd.isna(name):
        return None
    name = str(name).lower().strip()
    if "," in name:
        last, first = name.split(",")
        name = first.strip() + " " + last.strip()
    return name

def add_rolling_features(games):
    games = games.sort_values("game_date")

    for team in MLB_TEAMS:
        team_games = games[
            (games["home_team"] == team) |
            (games["away_team"] == team)
        ].sort_values("game_date").copy()

        if len(team_games) < 10:
            continue

        team_games["runs_scored"] = team_games.apply(
            lambda row: row["home_score"] if row["home_team"] == team else row["away_score"],
            axis=1
        )

        team_games["runs_allowed"] = team_games.apply(
            lambda row: row["away_score"] if row["home_team"] == team else row["home_score"],
            axis=1
        )

        team_games["last10_runs_scored"] = team_games["runs_scored"].shift(1).rolling(10).mean()
        team_games["last10_runs_allowed"] = team_games["runs_allowed"].shift(1).rolling(10).mean()
        team_games["last10_run_diff"] = (
            team_games["last10_runs_scored"] - team_games["last10_runs_allowed"]
        )

        for idx in team_games.index:
            if games.loc[idx, "home_team"] == team:
                games.loc[idx, "home_last10_runs_scored"] = team_games.loc[idx, "last10_runs_scored"]
                games.loc[idx, "home_last10_runs_allowed"] = team_games.loc[idx, "last10_runs_allowed"]
                games.loc[idx, "home_last10_run_diff"] = team_games.loc[idx, "last10_run_diff"]
            else:
                games.loc[idx, "away_last10_runs_scored"] = team_games.loc[idx, "last10_runs_scored"]
                games.loc[idx, "away_last10_runs_allowed"] = team_games.loc[idx, "last10_runs_allowed"]
                games.loc[idx, "away_last10_run_diff"] = team_games.loc[idx, "last10_run_diff"]

    return games


def build_training_dataset():

    print("Loading tables...")
    games = pd.read_sql("SELECT * FROM games", engine)
    pitchers = pd.read_sql("SELECT * FROM pitchers", engine)

    # MLB filter
    games = games[
        games["home_team"].isin(MLB_TEAMS) &
        games["away_team"].isin(MLB_TEAMS)
    ].copy()

    games["game_date"] = pd.to_datetime(games["game_date"])
    games["season"] = games["game_date"].dt.year

    # Clean pitcher names
    games["home_pitcher_clean"] = games["home_pitcher"].apply(clean_name)
    games["away_pitcher_clean"] = games["away_pitcher"].apply(clean_name)
    pitchers["pitcher_name_clean"] = pitchers["pitcher_name"].apply(clean_name)

    # Home pitcher merge
    games = games.merge(
        pitchers,
        how="left",
        left_on=["home_pitcher_clean","season"],
        right_on=["pitcher_name_clean","season"]
    )

    games.rename(columns={"era":"home_era","whip":"home_whip"}, inplace=True)
    games.drop(columns=["pitcher_name","pitcher_name_clean"], errors="ignore", inplace=True)

    # Away pitcher merge
    games = games.merge(
        pitchers,
        how="left",
        left_on=["away_pitcher_clean","season"],
        right_on=["pitcher_name_clean","season"]
    )

    games.rename(columns={"era":"away_era","whip":"away_whip"}, inplace=True)
    games.drop(columns=["pitcher_name","pitcher_name_clean"], errors="ignore", inplace=True)

    # Fill missing
    league_era = pitchers["era"].mean()
    league_whip = pitchers["whip"].mean()

    games["home_era"] = games["home_era"].fillna(league_era)
    games["away_era"] = games["away_era"].fillna(league_era)
    games["home_whip"] = games["home_whip"].fillna(league_whip)
    games["away_whip"] = games["away_whip"].fillna(league_whip)

    # DIFFERENTIALS
    games["era_diff"] = games["home_era"] - games["away_era"]
    games["whip_diff"] = games["home_whip"] - games["away_whip"]

    games = games.dropna(subset=["home_score","away_score"])

    games["run_diff"] = games["home_score"] - games["away_score"]
    games["total_runs"] = games["home_score"] + games["away_score"]
    games["home_win"] = (games["run_diff"] > 0).astype(int)

    print("Adding rolling features...")
    games = add_rolling_features(games)

    games = games.dropna(subset=[
        "home_last10_runs_scored",
        "away_last10_runs_scored"
    ])

    print(f"Rows after feature engineering: {len(games)}")

    output_path = os.path.join(BASE_DIR,"ml","training_data.csv")
    games.to_csv(output_path,index=False)

    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    build_training_dataset()