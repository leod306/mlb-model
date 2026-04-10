import os
from datetime import datetime
import requests
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from app.db import engine

SEASONS = [2024, 2025, 2026]
GAMES_TABLE = os.getenv("MLB_GAMES_TABLE", "games")

TEAM_MAP = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC", 145: "CWS",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",
    108: "LAA", 119: "LAD", 146: "MIA", 158: "MIL", 142: "MIN", 121: "NYM",
    147: "NYY", 133: "ATH", 143: "PHI", 134: "PIT", 135: "SD", 137: "SF",
    136: "SEA", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 120: "WSH",
}


def ensure_games_columns():
    statements = [
        f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS season INT",
        f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS official_date DATE",
        f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS game_date_utc TIMESTAMP",
        f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS home_team TEXT",
        f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS away_team TEXT",
        f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS home_team_id INT",
        f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS away_team_id INT",
        f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS home_score INT",
        f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS away_score INT",
        f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS game_type TEXT",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def fetch_schedule_for_season(season: int) -> pd.DataFrame:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "season": season,
        "gameType": "R",
        "hydrate": "team,linescore",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            home_team = home.get("team", {}) or {}
            away_team = away.get("team", {}) or {}

            home_id = home_team.get("id")
            away_id = away_team.get("id")

            rows.append({
                "game_pk": g.get("gamePk"),
                "season": season,
                "official_date": g.get("officialDate"),
                "game_date_utc": g.get("gameDate"),
                "game_type": g.get("gameType"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team": TEAM_MAP.get(home_id, home_team.get("abbreviation") or home_team.get("name")),
                "away_team": TEAM_MAP.get(away_id, away_team.get("abbreviation") or away_team.get("name")),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["official_date"] = pd.to_datetime(df["official_date"], errors="coerce").dt.date
        df["game_date_utc"] = pd.to_datetime(df["game_date_utc"], errors="coerce", utc=True)

    return df


def upsert_games(df: pd.DataFrame):
    if df.empty:
        print("No rows to upsert.")
        return

    clean = df.copy()

    # convert dates
    clean["official_date"] = pd.to_datetime(clean["official_date"], errors="coerce").dt.date
    clean["game_date_utc"] = pd.to_datetime(clean["game_date_utc"], errors="coerce", utc=True)

    # build clean Python-native records (🔥 critical)
    records = []
    for _, row in clean.iterrows():
        records.append({
            "game_pk": int(row["game_pk"]) if pd.notna(row["game_pk"]) else None,
            "season": int(row["season"]) if pd.notna(row["season"]) else None,
            "official_date": row["official_date"],
            "game_date_utc": (
                row["game_date_utc"].to_pydatetime().replace(tzinfo=None)
                if pd.notna(row["game_date_utc"]) else None
            ),
            "game_type": row["game_type"],
            "home_team_id": int(row["home_team_id"]) if pd.notna(row["home_team_id"]) else None,
            "away_team_id": int(row["away_team_id"]) if pd.notna(row["away_team_id"]) else None,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": int(row["home_score"]) if pd.notna(row["home_score"]) else None,
            "away_score": int(row["away_score"]) if pd.notna(row["away_score"]) else None,
        })

    sql = text(f"""
        INSERT INTO {GAMES_TABLE} (
            game_pk,
            season,
            official_date,
            game_date_utc,
            game_type,
            home_team_id,
            away_team_id,
            home_team,
            away_team,
            home_score,
            away_score
        )
        VALUES (
            :game_pk,
            :season,
            :official_date,
            :game_date_utc,
            :game_type,
            :home_team_id,
            :away_team_id,
            :home_team,
            :away_team,
            :home_score,
            :away_score
        )
        ON CONFLICT (game_pk) DO UPDATE SET
            season = EXCLUDED.season,
            official_date = EXCLUDED.official_date,
            game_date_utc = EXCLUDED.game_date_utc,
            game_type = EXCLUDED.game_type,
            home_team_id = EXCLUDED.home_team_id,
            away_team_id = EXCLUDED.away_team_id,
            home_team = EXCLUDED.home_team,
            away_team = EXCLUDED.away_team,
            home_score = EXCLUDED.home_score,
            away_score = EXCLUDED.away_score
    """)

    # 🔥 FIXED MISSING VARIABLES
    chunk_size = 250
    total = 0

    with engine.begin() as conn:
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            conn.execute(sql, chunk)
            total += len(chunk)

    print(f"Upserted {total} rows")


def main():
    ensure_games_columns()

    for season in SEASONS:
        print(f"\nLoading season {season}...")
        df = fetch_schedule_for_season(season)
        print(f"Fetched {len(df)} rows for {season}")
        upsert_games(df)


if __name__ == "__main__":
    main()