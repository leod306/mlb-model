"""
build_pitcher_features.py
--------------------------
Computes 4 new features from pitcher_game_log and game results:
  1. home_sp_rest_days     - days since home SP's last start
  2. away_sp_rest_days     - days since away SP's last start
  3. home_bullpen_ip_4d    - home bullpen IP over last 4 days (fatigue)
  4. away_bullpen_ip_4d    - away bullpen IP over last 4 days (fatigue)
  5. home_win_pct_home     - home team win% at home this season
  6. away_win_pct_away     - away team win% away this season

Adds these as columns to your games table so build_dataset.py
and mlb_engine_daily.py can use them as model features.

Run after load_pitcher_game_logs.py:
  python scripts/build_pitcher_features.py
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone, timedelta

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)

GAMES_TABLE = os.getenv("MLB_GAMES_TABLE", "games")
LOG_TABLE   = "pitcher_game_log"


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def conn():
    return psycopg2.connect(DATABASE_URL)


def ensure_feature_columns(cur):
    """Add new feature columns to games table if they don't exist."""
    cols = [
        ("home_sp_rest_days",  "DOUBLE PRECISION"),
        ("away_sp_rest_days",  "DOUBLE PRECISION"),
        ("home_bullpen_ip_4d", "DOUBLE PRECISION"),
        ("away_bullpen_ip_4d", "DOUBLE PRECISION"),
        ("home_win_pct_home",  "DOUBLE PRECISION"),
        ("away_win_pct_away",  "DOUBLE PRECISION"),
    ]
    for col, dtype in cols:
        cur.execute(f"ALTER TABLE {GAMES_TABLE} ADD COLUMN IF NOT EXISTS {col} {dtype};")


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_sp_rest_days(logs_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, find the SP's previous start date and compute rest days.
    Uses home_starting_pitcher / away_starting_pitcher name matching.
    """
    # Only SP starts
    sp_logs = logs_df[logs_df["role"] == "SP"].copy()
    sp_logs = sp_logs.sort_values("official_date")

    # Build lookup: pitcher_name -> list of (date, game_pk) sorted
    sp_by_name: dict[str, list] = {}
    for _, row in sp_logs.iterrows():
        name = (row["pitcher_name"] or "").lower().strip()
        if name:
            sp_by_name.setdefault(name, []).append(row["official_date"])

    def get_rest_days(pitcher_name, game_date) -> float:
        if not pitcher_name:
            return 5.0  # assume normal rest if unknown
        name = pitcher_name.lower().strip()
        dates = sp_by_name.get(name, [])
        # Find most recent start BEFORE this game
        prev_dates = [d for d in dates if d < game_date]
        if not prev_dates:
            return 5.0  # no prior start found, assume normal
        last_start = max(prev_dates)
        return float((game_date - last_start).days)

    games_df = games_df.copy()
    games_df["home_sp_rest_days"] = games_df.apply(
        lambda r: get_rest_days(r.get("home_starting_pitcher"), r["official_date"]), axis=1
    )
    games_df["away_sp_rest_days"] = games_df.apply(
        lambda r: get_rest_days(r.get("away_starting_pitcher"), r["official_date"]), axis=1
    )
    return games_df


def compute_bullpen_fatigue(logs_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, sum RP innings pitched by team over the prior 4 days.
    Higher = more fatigued bullpen.
    """
    rp_logs = logs_df[logs_df["role"] == "RP"].copy()
    rp_logs = rp_logs.sort_values("official_date")

    # Build lookup: (team, date) -> total RP IP that day
    team_day_ip: dict[tuple, float] = {}
    for _, row in rp_logs.iterrows():
        key = (row["team"], row["official_date"])
        team_day_ip[key] = team_day_ip.get(key, 0.0) + (row["innings_pitched"] or 0.0)

    def bullpen_ip_last_4d(team: str, game_date) -> float:
        total = 0.0
        for d in range(1, 5):  # 1, 2, 3, 4 days before game
            prior_date = game_date - timedelta(days=d)
            total += team_day_ip.get((team, prior_date), 0.0)
        return round(total, 2)

    games_df = games_df.copy()
    games_df["home_bullpen_ip_4d"] = games_df.apply(
        lambda r: bullpen_ip_last_4d(r["home_team"], r["official_date"]), axis=1
    )
    games_df["away_bullpen_ip_4d"] = games_df.apply(
        lambda r: bullpen_ip_last_4d(r["away_team"], r["official_date"]), axis=1
    )
    return games_df


def compute_home_away_win_pct(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling home win% for home team at home, and away win% for away team away.
    Uses shift(1) so we never include the current game.
    """
    games_df = games_df.copy().sort_values("official_date")
    games_df["home_win"] = (games_df["home_score"] > games_df["away_score"]).astype(float)

    # Home team win% at home
    home_win_pct = {}
    for team, grp in games_df.groupby("home_team"):
        grp = grp.sort_values("official_date")
        wins_shifted  = grp["home_win"].shift(1).expanding().mean()
        for idx, val in zip(grp.index, wins_shifted):
            home_win_pct[idx] = val

    # Away team win% away
    games_df["away_win"] = (games_df["away_score"] > games_df["home_score"]).astype(float)
    away_win_pct = {}
    for team, grp in games_df.groupby("away_team"):
        grp = grp.sort_values("official_date")
        wins_shifted = grp["away_win"].shift(1).expanding().mean()
        for idx, val in zip(grp.index, wins_shifted):
            away_win_pct[idx] = val

    games_df["home_win_pct_home"] = games_df.index.map(home_win_pct)
    games_df["away_win_pct_away"] = games_df.index.map(away_win_pct)

    # Fill NaN (first game of season) with 0.5
    games_df["home_win_pct_home"] = games_df["home_win_pct_home"].fillna(0.5)
    games_df["away_win_pct_away"] = games_df["away_win_pct_away"].fillna(0.5)

    return games_df


# ---------------------------------------------------------------------------
# Write features back to games table
# ---------------------------------------------------------------------------

def update_games_features(cur, games_df: pd.DataFrame) -> int:
    """Update the 6 new feature columns in the games table."""
    rows = []
    for _, row in games_df.iterrows():
        rows.append((
            row.get("home_sp_rest_days"),
            row.get("away_sp_rest_days"),
            row.get("home_bullpen_ip_4d"),
            row.get("away_bullpen_ip_4d"),
            row.get("home_win_pct_home"),
            row.get("away_win_pct_away"),
            int(row["game_pk"]),
        ))

    for r in rows:
        cur.execute(f"""
            UPDATE {GAMES_TABLE} SET
                home_sp_rest_days  = %s,
                away_sp_rest_days  = %s,
                home_bullpen_ip_4d = %s,
                away_bullpen_ip_4d = %s,
                home_win_pct_home  = %s,
                away_win_pct_away  = %s
            WHERE game_pk = %s
        """, r)

    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"{'='*55}")
    print(f"  build_pitcher_features.py")
    print(f"  Computing rest days, bullpen fatigue, win% splits")
    print(f"{'='*55}\n")

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:

            ensure_feature_columns(cur)
            c.commit()
            print("Feature columns ready.\n")

            # Load all games with scores
            cur.execute(f"""
                SELECT game_pk, official_date, season, home_team, away_team,
                       home_score, away_score,
                       home_starting_pitcher, away_starting_pitcher
                FROM {GAMES_TABLE}
                WHERE home_score IS NOT NULL AND away_score IS NOT NULL
                ORDER BY official_date, game_pk
            """)
            rows = cur.fetchall()
            games_df = pd.DataFrame(rows, columns=[
                "game_pk", "official_date", "season", "home_team", "away_team",
                "home_score", "away_score",
                "home_starting_pitcher", "away_starting_pitcher",
            ])
            print(f"Loaded {len(games_df)} completed games\n")

            # Load pitcher game logs
            cur.execute(f"""
                SELECT game_pk, official_date, pitcher_id, pitcher_name,
                       team, side, role, innings_pitched, pitch_count
                FROM {LOG_TABLE}
                ORDER BY official_date, game_pk
            """)
            log_rows = cur.fetchall()
            logs_df = pd.DataFrame(log_rows, columns=[
                "game_pk", "official_date", "pitcher_id", "pitcher_name",
                "team", "side", "role", "innings_pitched", "pitch_count",
            ])
            print(f"Loaded {len(logs_df)} pitcher game log rows\n")

            if logs_df.empty:
                print("No pitcher logs found. Run load_pitcher_game_logs.py first.")
                return

            # Compute features
            print("Computing SP rest days...")
            games_df = compute_sp_rest_days(logs_df, games_df)

            print("Computing bullpen fatigue (IP last 4 days)...")
            games_df = compute_bullpen_fatigue(logs_df, games_df)

            print("Computing home/away win% splits...")
            games_df = compute_home_away_win_pct(games_df)

            print(f"\nSample output:")
            print(games_df[[
                "official_date", "home_team", "away_team",
                "home_sp_rest_days", "away_sp_rest_days",
                "home_bullpen_ip_4d", "away_bullpen_ip_4d",
                "home_win_pct_home", "away_win_pct_away",
            ]].tail(5).to_string())

            print(f"\nWriting features to {GAMES_TABLE} table...")
            n = update_games_features(cur, games_df)
            c.commit()

        print(f"\n{'='*55}")
        print(f"  Done. Updated {n} games with new features.")
        print(f"{'='*55}")
        print("\nNext steps:")
        print("  1. Run build_dataset.py to rebuild training CSV")
        print("  2. Run train_model.py to retrain with new features")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()