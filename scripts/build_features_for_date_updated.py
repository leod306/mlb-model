# ---------------------------------------------------------------------------
# REPLACE the existing build_features_for_date function in mlb_engine_daily.py
# with this version. It adds 6 new features at prediction time.
# ---------------------------------------------------------------------------

def build_features_for_date(cur, target) -> "pd.DataFrame":
    import pandas as pd
    from datetime import timedelta

    # Use recent prior seasons too, not just the current season
    LOOKBACK_SEASONS = [SEASON - 2, SEASON - 1, SEASON]
    MIN_SEASON = min(LOOKBACK_SEASONS)

    cur.execute(
        f"""
        WITH pitcher_stats AS (
            SELECT
                LOWER(TRIM(pitcher_name)) AS pitcher_key,
                season,
                era,
                whip,
                ROW_NUMBER() OVER (
                    PARTITION BY LOWER(TRIM(pitcher_name))
                    ORDER BY
                        CASE WHEN season = %s THEN 0 ELSE 1 END,
                        season DESC
                ) AS rn
            FROM pitchers
            WHERE season >= %s
        )
        SELECT
            g.game_pk,
            g.official_date,
            g.home_team,
            g.away_team,
            g.home_team_id,
            g.away_team_id,
            p.home_sp_id,
            p.away_sp_id,
            p.home_sp_name,
            p.away_sp_name,
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
            g.home_sp_rest_days,
            g.away_sp_rest_days,
            g.home_bullpen_ip_4d,
            g.away_bullpen_ip_4d,
            g.home_win_pct_home,
            g.away_win_pct_away
        FROM {GAMES_TABLE} g
        LEFT JOIN {PROB_TABLE} p
            ON p.game_pk = g.game_pk
        LEFT JOIN pitcher_stats hp
            ON hp.pitcher_key = LOWER(TRIM(p.home_sp_name))
           AND hp.rn = 1
        LEFT JOIN pitcher_stats ap
            ON ap.pitcher_key = LOWER(TRIM(p.away_sp_name))
           AND ap.rn = 1
        WHERE g.official_date = %s
        ORDER BY g.game_pk
        """,
        (SEASON, MIN_SEASON, target),
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

    # League-average fallback from recent seasons, not just current season
    cur.execute(
        """
        SELECT AVG(era), AVG(whip)
        FROM pitchers
        WHERE season >= %s
        """,
        (MIN_SEASON,)
    )
    row = cur.fetchone()
    league_era = float(row[0]) if row and row[0] is not None else 4.20
    league_whip = float(row[1]) if row and row[1] is not None else 1.30

    # Convert pitcher stats
    for col in ["home_era", "away_era", "home_whip", "away_whip"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing-pitcher indicators before filling
    df["missing_home_pitcher_stats"] = df["home_era"].isna().astype(int)
    df["missing_away_pitcher_stats"] = df["away_era"].isna().astype(int)

    # Fill pitcher stats with league average only when missing
    df["home_era"] = df["home_era"].fillna(league_era)
    df["away_era"] = df["away_era"].fillna(league_era)
    df["home_whip"] = df["home_whip"].fillna(league_whip)
    df["away_whip"] = df["away_whip"].fillna(league_whip)

    # Derived pitcher edges
    df["era_diff"] = df["home_era"] - df["away_era"]
    df["whip_diff"] = df["home_whip"] - df["away_whip"]

    # Rolling stats: preserve missingness flags so the model can distinguish real zero from missing
    rolling_cols = [
        "home_last10_runs_scored", "away_last10_runs_scored",
        "home_last10_runs_allowed", "away_last10_runs_allowed",
        "home_last10_run_diff", "away_last10_run_diff",
    ]
    for col in rolling_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0.0)

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
        return 4.0

    def get_win_pct(team: str, side: str, game_date) -> float:
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
                      AND away_score IS NOT NULL
                """, (team, game_date))
            else:
                cur.execute(f"""
                    SELECT
                        COUNT(*) FILTER (WHERE away_score > home_score)::float /
                        NULLIF(COUNT(*), 0)
                    FROM {GAMES_TABLE}
                    WHERE away_team = %s
                      AND official_date < %s
                      AND home_score IS NOT NULL
                      AND away_score IS NOT NULL
                """, (team, game_date))
            result = cur.fetchone()
            if result and result[0] is not None:
                return float(result[0])
        except Exception:
            pass
        return 0.5

    # Fill live values if not already stored
    for idx, row in df.iterrows():
        game_date = row["official_date"]
        if isinstance(game_date, str):
            from datetime import date as date_
            game_date = date_.fromisoformat(game_date)

        if pd.isna(row.get("home_sp_rest_days")):
            df.at[idx, "home_sp_rest_days"] = get_sp_rest_days(row.get("home_sp_name"), game_date)
        if pd.isna(row.get("away_sp_rest_days")):
            df.at[idx, "away_sp_rest_days"] = get_sp_rest_days(row.get("away_sp_name"), game_date)

        if pd.isna(row.get("home_bullpen_ip_4d")):
            df.at[idx, "home_bullpen_ip_4d"] = get_bullpen_ip_4d(row["home_team"], game_date)
        if pd.isna(row.get("away_bullpen_ip_4d")):
            df.at[idx, "away_bullpen_ip_4d"] = get_bullpen_ip_4d(row["away_team"], game_date)

        if pd.isna(row.get("home_win_pct_home")):
            df.at[idx, "home_win_pct_home"] = get_win_pct(row["home_team"], "home", game_date)
        if pd.isna(row.get("away_win_pct_away")):
            df.at[idx, "away_win_pct_away"] = get_win_pct(row["away_team"], "away", game_date)

    # Coerce/fill remaining live features
    fill_defaults = {
        "home_sp_rest_days": 5.0,
        "away_sp_rest_days": 5.0,
        "home_bullpen_ip_4d": 4.0,
        "away_bullpen_ip_4d": 4.0,
        "home_win_pct_home": 0.5,
        "away_win_pct_away": 0.5,
    }
    for col, default in fill_defaults.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(default)

    # Presence flags
    df["has_home_sp"] = df["home_sp_id"].notna().astype(int)
    df["has_away_sp"] = df["away_sp_id"].notna().astype(int)

    # Extra differentials to reduce symmetry
    df["run_diff_form_diff"] = df["home_last10_run_diff"] - df["away_last10_run_diff"]
    df["runs_scored_form_diff"] = df["home_last10_runs_scored"] - df["away_last10_runs_scored"]
    df["runs_allowed_form_diff"] = df["home_last10_runs_allowed"] - df["away_last10_runs_allowed"]
    df["sp_rest_diff"] = df["home_sp_rest_days"] - df["away_sp_rest_days"]
    df["bullpen_ip_diff"] = df["home_bullpen_ip_4d"] - df["away_bullpen_ip_4d"]
    df["home_away_winpct_diff"] = df["home_win_pct_home"] - df["away_win_pct_away"]

    # Diagnostics
    print("\n=== FEATURE CHECK ===")
    preview_cols = [c for c in [
        "game_pk", "away_team", "home_team",
        "home_era", "away_era", "era_diff",
        "home_whip", "away_whip", "whip_diff",
        "home_last10_run_diff", "away_last10_run_diff", "run_diff_form_diff",
        "home_sp_rest_days", "away_sp_rest_days", "sp_rest_diff",
        "home_bullpen_ip_4d", "away_bullpen_ip_4d", "bullpen_ip_diff",
        "home_win_pct_home", "away_win_pct_away", "home_away_winpct_diff",
        "missing_home_pitcher_stats", "missing_away_pitcher_stats"
    ] if c in df.columns]
    print(df[preview_cols].head(10).to_string(index=False))

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    nunique_series = df[numeric_cols].nunique(dropna=False).sort_values()
    print("\nLowest-variation numeric columns:")
    print(nunique_series.head(20).to_string())

    # Fail loudly only if the core differentiators are completely flat
    key_diff_cols = [
        "era_diff",
        "whip_diff",
        "run_diff_form_diff",
        "sp_rest_diff",
        "bullpen_ip_diff",
        "home_away_winpct_diff",
    ]
    existing_key_cols = [c for c in key_diff_cols if c in df.columns]
    flat_key_cols = [c for c in existing_key_cols if df[c].nunique(dropna=False) <= 1]

    if len(flat_key_cols) == len(existing_key_cols):
        raise RuntimeError(
            f"Feature collapse detected: all core differentiators are flat for {target}. "
            f"Flat columns: {flat_key_cols}"
        )

    return df