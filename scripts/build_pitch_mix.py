from __future__ import annotations

import os
import sys
from datetime import date, datetime
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pybaseball import statcast_pitcher


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)


def season_start(season: int) -> str:
    return f"{season}-03-01"


def today_str() -> str:
    return date.today().isoformat()


def ensure_pitch_mix_table() -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS pitch_mix (
        pitcher_id BIGINT,
        pitcher_name TEXT,
        season INT,
        pitch_type TEXT,
        usage_pct NUMERIC,
        pitch_count INT,
        PRIMARY KEY (pitcher_id, season, pitch_type)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(sql))


def get_probable_pitchers(target_date: str) -> pd.DataFrame:
    sql = text(
        """
        SELECT DISTINCT home_sp_id AS pitcher_id, home_sp_name AS pitcher_name, official_date
        FROM game_probables
        WHERE official_date = :d
          AND home_sp_id IS NOT NULL

        UNION

        SELECT DISTINCT away_sp_id AS pitcher_id, away_sp_name AS pitcher_name, official_date
        FROM game_probables
        WHERE official_date = :d
          AND away_sp_id IS NOT NULL
        """
    )
    return pd.read_sql(sql, engine, params={"d": target_date})


def fetch_pitch_mix_for_pitcher(pitcher_id: int, pitcher_name: str, season: int) -> pd.DataFrame:
    start_dt = season_start(season)
    end_dt = today_str()

    df = statcast_pitcher(start_dt, end_dt, pitcher_id)

    if df is None or df.empty:
        return pd.DataFrame(
            columns=["pitcher_id", "pitcher_name", "season", "pitch_type", "usage_pct", "pitch_count"]
        )

    # Prefer human-readable pitch_name when available, else pitch_type
    pitch_col = "pitch_name" if "pitch_name" in df.columns else "pitch_type"
    if pitch_col not in df.columns:
        return pd.DataFrame(
            columns=["pitcher_id", "pitcher_name", "season", "pitch_type", "usage_pct", "pitch_count"]
        )

    mix = (
        df[pitch_col]
        .fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("pitch_type")
        .reset_index(name="pitch_count")
    )

    total = int(mix["pitch_count"].sum())
    if total == 0:
        return pd.DataFrame(
            columns=["pitcher_id", "pitcher_name", "season", "pitch_type", "usage_pct", "pitch_count"]
        )

    mix["usage_pct"] = (mix["pitch_count"] / total * 100).round(2)
    mix["pitcher_id"] = int(pitcher_id)
    mix["pitcher_name"] = pitcher_name
    mix["season"] = int(season)

    return mix[["pitcher_id", "pitcher_name", "season", "pitch_type", "usage_pct", "pitch_count"]]


def upsert_pitch_mix(rows: pd.DataFrame) -> None:
    if rows.empty:
        return

    sql = text(
        """
        INSERT INTO pitch_mix (
            pitcher_id,
            pitcher_name,
            season,
            pitch_type,
            usage_pct,
            pitch_count
        )
        VALUES (
            :pitcher_id,
            :pitcher_name,
            :season,
            :pitch_type,
            :usage_pct,
            :pitch_count
        )
        ON CONFLICT (pitcher_id, season, pitch_type)
        DO UPDATE SET
            pitcher_name = EXCLUDED.pitcher_name,
            usage_pct = EXCLUDED.usage_pct,
            pitch_count = EXCLUDED.pitch_count
        """
    )

    payload = rows.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(sql, payload)


def build_pitch_mix(target_date: str, season: int) -> None:
    ensure_pitch_mix_table()

    probables = get_probable_pitchers(target_date)
    if probables.empty:
        print(f"No probable pitchers found for {target_date}")
        return

    print(f"Found {len(probables)} probable pitchers for {target_date}")

    all_rows = []
    for _, row in probables.iterrows():
        pitcher_id = int(row["pitcher_id"])
        pitcher_name = str(row["pitcher_name"]) if pd.notna(row["pitcher_name"]) else f"Pitcher {pitcher_id}"

        print(f"Loading pitch mix for {pitcher_name} ({pitcher_id})...")
        mix_df = fetch_pitch_mix_for_pitcher(pitcher_id, pitcher_name, season)

        if mix_df.empty:
            print(f"  No pitch mix data found for {pitcher_name}")
            continue

        all_rows.append(mix_df)

    if not all_rows:
        print("No pitch mix rows to save.")
        return

    final_df = pd.concat(all_rows, ignore_index=True)
    upsert_pitch_mix(final_df)

    print(f"Saved {len(final_df)} pitch mix rows.")


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    season = int(sys.argv[2]) if len(sys.argv) > 2 else date.today().year

    build_pitch_mix(target_date, season)