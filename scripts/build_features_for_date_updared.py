# ---------------------------------------------------------------------------
# REPLACE the existing build_features_for_date function in mlb_engine_daily.py
# with this version. It adds 6 new features at prediction time.
# ---------------------------------------------------------------------------

def build_features_for_date(cur, target) -> "pd.DataFrame":
    import math
    import pandas as pd
    from datetime import timedelta

    cur.execute(
        f"""
        SELECT
            g.game_pk, g.official_date, g.home_team, g.away_team,
            g.home_team_id, g.away_team_id,
            p.home_sp_id, p.away_sp_id, p.home_sp_name, p.away_sp_name,
            g.home_last10_runs_scored,
            g.away_last10_runs_scored,
            g.home_last10_runs_allowed,
            g.away_last10_runs_allowed,
            g.home_last10_run_diff,
            g.away_last10_run_diff,
            hp.era  AS home_era,
            hp.whip AS home_whip,
            ap.era  AS away_era,
            ap.whip AS away_whip,
            -- new feature columns (may be NULL if not yet computed)
            g.home_sp_rest_days,
            g.away_sp_rest_days,
            g.home_bullpen_ip_4d,
            g.away_bullpen_ip_4d,
            g.home_win_pct_home,
            g.away_win_pct_away
        FROM {GAMES_TABLE} g
        LEFT JOIN {PROB_TABLE} p ON p.game_pk = g.game_pk
        LEFT JOIN pitchers hp
            ON LOWER(TRIM(hp.pitcher_name)) = LOWER(TRIM(p.home_sp_name))
            AND hp.season = %s
        LEFT JOIN pitchers ap
            ON LOWER(TRIM(ap.pitcher_name)) = LOWER(TRIM(p.away_sp_name))
            AND ap.season = %s
        WHERE g.official_date = %s
        ORDER BY g.game_pk
        """,
        (SEASON, SEASON, target),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "game_pk", "official_date", "home_team", "away_team",
        "home_team_id", "away_team_id",
        "home_sp_id", "away_sp_id", "home_sp_name", "away_sp_name",
        "home_last10_runs_scored", "away_last10_runs_scored",
        "home_last10_runs_allowed", "away_last10_runs_allowed",
        "home_last10_run_diff", "away_last10_run_diff",
        "home_era", "home_whip", "away_era", "away_whip",
        "home_sp_rest_days", "away_sp_rest_days",
        "home_bullpen_ip_4d", "away_bullpen_ip_4d",
        "home_win_pct_home", "away_win_pct_away",
    ])

    # League-average ERA/WHIP fallback for unknown pitchers
    cur.execute("SELECT AVG(era), AVG(whip) FROM pitchers WHERE season = %s", (SEASON,))
    row = cur.fetchone()
    league_era  = float(row[0]) if row and row[0] is not None else 4.20
    league_whip = float(row[1]) if row and row[1] is not None else 1.30

    df["home_era"]  = pd.to_numeric(df["home_era"],  errors="coerce").fillna(league_era)
    df["away_era"]  = pd.to_numeric(df["away_era"],  errors="coerce").fillna(league_era)
    df["home_whip"] = pd.to_numeric(df["home_whip"], errors="coerce").fillna(league_whip)
    df["away_whip"] = pd.to_numeric(df["away_whip"], errors="coerce").fillna(league_whip)

    df["era_diff"]  = df["home_era"]  - df["away_era"]
    df["whip_diff"] = df["home_whip"] - df["away_whip"]

    # Rolling stats default to 0 if not yet available
    rolling_cols = [
        "home_last10_runs_scored", "away_last10_runs_scored",
        "home_last10_runs_allowed", "away_last10_runs_allowed",
        "home_last10_run_diff",    "away_last10_run_diff",
    ]
    for col in rolling_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # New features — compute live if not stored in games table yet
    # Rest days: compute from pitcher_game_log if column is null
    def get_sp_rest_days(sp_name: str, game_date) -> float:
        if not sp_name:
            return 5.0
        try:
            cur.execute("""
                SELECT MAX(official_date)
                FROM pitcher_game_log
                WHERE LOWER(TRIM(pitcher_name)) = LOWER(TRIM(%s))
                  AND role = 'SP'
                  AND official_date < %s
            """, (sp_name, game_date))
            result = cur.fetchone()
            if result and result[0]:
                return float((game_date - result[0]).days)
        except Exception:
            pass
        return 5.0

    def get_bullpen_ip_4d(team: str, game_date) -> float:
        try:
            cur.execute("""
                SELECT COALESCE(SUM(innings_pitched), 0)
                FROM pitcher_game_log
                WHERE team = %s
                  AND role = 'RP'
                  AND official_date >= %s
                  AND official_date < %s
            """, (team, game_date - timedelta(days=4), game_date))
            result = cur.fetchone()
            if result and result[0] is not None:
                return float(result[0])
        except Exception:
            pass
        return 4.0  # league average ~4 IP per day

    def get_win_pct(team: str, side: str, game_date) -> float:
        """Win% for team as home (side='home') or away (side='away') before this date."""
        try:
            if side == "home":
                cur.execute(f"""
                    SELECT
                        COUNT(*) FILTER (WHERE home_score > away_score)::float /
                        NULLIF(COUNT(*), 0)
                    FROM {GAMES_TABLE}
                    WHERE home_team = %s
                      AND official_date < %s
                      AND home_score IS NOT NULL
                """, (team, game_date))
            else:
                cur.execute(f"""
                    SELECT
                        COUNT(*) FILTER (WHERE away_score > home_score)::float /
                        NULLIF(COUNT(*), 0)
                    FROM {GAMES_TABLE}
                    WHERE away_team = %s
                      AND official_date < %s
                      AND away_score IS NOT NULL
                """, (team, game_date))
            result = cur.fetchone()
            if result and result[0] is not None:
                return float(result[0])
        except Exception:
            pass
        return 0.5

    # For each row, fill in new features live if the stored value is null
    for idx, row in df.iterrows():
        game_date = row["official_date"]
        if isinstance(game_date, str):
            from datetime import date as date_
            game_date = date_.fromisoformat(game_date)

        # Rest days
        if pd.isna(row.get("home_sp_rest_days")):
            df.at[idx, "home_sp_rest_days"] = get_sp_rest_days(
                row.get("home_sp_name"), game_date)
        if pd.isna(row.get("away_sp_rest_days")):
            df.at[idx, "away_sp_rest_days"] = get_sp_rest_days(
                row.get("away_sp_name"), game_date)

        # Bullpen fatigue
        if pd.isna(row.get("home_bullpen_ip_4d")):
            df.at[idx, "home_bullpen_ip_4d"] = get_bullpen_ip_4d(
                row["home_team"], game_date)
        if pd.isna(row.get("away_bullpen_ip_4d")):
            df.at[idx, "away_bullpen_ip_4d"] = get_bullpen_ip_4d(
                row["away_team"], game_date)

        # Win% splits
        if pd.isna(row.get("home_win_pct_home")):
            df.at[idx, "home_win_pct_home"] = get_win_pct(
                row["home_team"], "home", game_date)
        if pd.isna(row.get("away_win_pct_away")):
            df.at[idx, "away_win_pct_away"] = get_win_pct(
                row["away_team"], "away", game_date)

    # Final coerce — ensure no NaN leaks through
    new_feat_defaults = {
        "home_sp_rest_days": 5.0,
        "away_sp_rest_days": 5.0,
        "home_bullpen_ip_4d": 4.0,
        "away_bullpen_ip_4d": 4.0,
        "home_win_pct_home": 0.5,
        "away_win_pct_away": 0.5,
    }
    for col, default in new_feat_defaults.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    df["has_home_sp"] = df["home_sp_id"].notna().astype(int)
    df["has_away_sp"] = df["away_sp_id"].notna().astype(int)

    return df