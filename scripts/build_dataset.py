import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ------------------------------------------------------------------
# ENV / DB
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env")
DATABASE_URL = DATABASE_URL.strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
MLB_TEAMS = [
    "ARI","ATL","BAL","BOS","CHC","CWS","CIN","CLE","COL",
    "DET","HOU","KC","LAA","LAD","MIA","MIL","MIN","NYM",
    "NYY","ATH","PHI","PIT","SD","SF","SEA","STL","TB",
    "TEX","TOR","WSH"
]

OU_WINDOW      = 20   # games for O/U tendency
ATS_WINDOW     = 20   # games for ATS cover rate
SP_ROLL_STARTS = 5    # rolling window for SP ERA
LEAGUE_AVG_OPS = 0.720
LEAGUE_AVG_HH  = 0.35


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def clean_name(name):
    if pd.isna(name):
        return None
    name = str(name).lower().strip()
    if "," in name:
        last, first = name.split(",", 1)
        name = first.strip() + " " + last.strip()
    return name


# ------------------------------------------------------------------
# FEATURE BUILDERS
# ------------------------------------------------------------------

def add_rolling_features(games):
    games = games.sort_values("game_date").copy()
    for col in [
        "home_last10_runs_scored","away_last10_runs_scored",
        "home_last10_runs_allowed","away_last10_runs_allowed",
        "home_last10_run_diff","away_last10_run_diff",
    ]:
        if col not in games.columns:
            games[col] = pd.NA

    for team in MLB_TEAMS:
        tg = games[(games.home_team == team) | (games.away_team == team)].copy()
        tg = tg.sort_values("game_date")
        if len(tg) == 0:
            continue

        tg["runs_scored"]  = tg.apply(lambda r: r.home_score if r.home_team == team else r.away_score, axis=1)
        tg["runs_allowed"] = tg.apply(lambda r: r.away_score if r.home_team == team else r.home_score, axis=1)
        tg["rs"] = tg["runs_scored"].shift(1).rolling(10, min_periods=1).mean()
        tg["ra"] = tg["runs_allowed"].shift(1).rolling(10, min_periods=1).mean()
        tg["rd"] = tg["rs"] - tg["ra"]

        for i in tg.index:
            if games.loc[i, "home_team"] == team:
                games.loc[i, "home_last10_runs_scored"]  = tg.loc[i, "rs"]
                games.loc[i, "home_last10_runs_allowed"] = tg.loc[i, "ra"]
                games.loc[i, "home_last10_run_diff"]     = tg.loc[i, "rd"]
            else:
                games.loc[i, "away_last10_runs_scored"]  = tg.loc[i, "rs"]
                games.loc[i, "away_last10_runs_allowed"] = tg.loc[i, "ra"]
                games.loc[i, "away_last10_run_diff"]     = tg.loc[i, "rd"]
    return games


def add_rest_days(games):
    games = games.sort_values("game_date").copy()
    last = {}
    home, away = [], []
    for _, r in games.iterrows():
        d  = r.game_date
        hp = r.get("home_pitcher_clean")
        ap = r.get("away_pitcher_clean")
        home.append((d - last.get(hp, d)).days if hp and hp in last else 5)
        away.append((d - last.get(ap, d)).days if ap and ap in last else 5)
        if hp: last[hp] = d
        if ap: last[ap] = d
    games["home_sp_rest_days"] = home
    games["away_sp_rest_days"] = away
    return games


def add_bullpen(games):
    games = games.sort_values("game_date").copy()
    h_list, a_list = [], []
    for _, r in games.iterrows():
        d      = r.game_date
        window = games[(games.game_date < d) & (games.game_date >= d - pd.Timedelta(days=4))]
        h = window[(window.home_team == r.home_team) | (window.away_team == r.home_team)].shape[0]
        a = window[(window.home_team == r.away_team) | (window.away_team == r.away_team)].shape[0]
        h_list.append(h)
        a_list.append(a)
    games["home_bullpen_ip_4d"] = h_list
    games["away_bullpen_ip_4d"] = a_list
    return games


def add_win_pct(games):
    games = games.sort_values("game_date").copy()
    h_pct, a_pct = [], []
    for _, r in games.iterrows():
        hist   = games[games.game_date < r.game_date]
        h_hist = hist[hist.home_team == r.home_team]
        a_hist = hist[hist.away_team == r.away_team]
        h = h_hist.home_win.mean()       if len(h_hist) > 5 else 0.5
        a = (1 - a_hist.home_win).mean() if len(a_hist) > 5 else 0.5
        h_pct.append(h)
        a_pct.append(a)
    games["home_win_pct_home"] = h_pct
    games["away_win_pct_away"] = a_pct
    return games


def add_ou_tendency(games, window=OU_WINDOW):
    games = games.sort_values("game_date").copy()
    DEFAULT_LINE = 8.5

    for col in ["home_ou_over_rate", "away_ou_over_rate",
                "home_last_game_total", "away_last_game_total"]:
        games[col] = pd.NA

    for team in MLB_TEAMS:
        tg = games[(games.home_team == team) | (games.away_team == team)].copy()
        tg = tg.sort_values("game_date")
        if len(tg) == 0:
            continue

        tg["game_total"] = tg["home_score"] + tg["away_score"]
        tg["went_over"]  = (tg["game_total"] > DEFAULT_LINE).astype(float)
        tg["over_rate"]  = tg["went_over"].shift(1).rolling(window, min_periods=3).mean()
        tg["last_total"] = tg["game_total"].shift(1)

        for i in tg.index:
            if games.loc[i, "home_team"] == team:
                games.loc[i, "home_ou_over_rate"]    = tg.loc[i, "over_rate"]
                games.loc[i, "home_last_game_total"] = tg.loc[i, "last_total"]
            else:
                games.loc[i, "away_ou_over_rate"]    = tg.loc[i, "over_rate"]
                games.loc[i, "away_last_game_total"] = tg.loc[i, "last_total"]

    return games


def add_ats_cover_rate(games, window=ATS_WINDOW):
    games = games.sort_values("game_date").copy()

    for col in ["home_ats_cover_rate", "away_ats_cover_rate"]:
        games[col] = pd.NA

    for team in MLB_TEAMS:
        tg = games[(games.home_team == team) | (games.away_team == team)].copy()
        tg = tg.sort_values("game_date")
        if len(tg) == 0:
            continue

        tg["team_run_diff"] = tg.apply(
            lambda r: (r.home_score - r.away_score) if r.home_team == team
                      else (r.away_score - r.home_score), axis=1
        )
        tg["covered_minus1_5"] = (tg["team_run_diff"] > 1.5).astype(float)
        tg["ats_cover_rate"]   = tg["covered_minus1_5"].shift(1).rolling(window, min_periods=3).mean()

        for i in tg.index:
            if games.loc[i, "home_team"] == team:
                games.loc[i, "home_ats_cover_rate"] = tg.loc[i, "ats_cover_rate"]
            else:
                games.loc[i, "away_ats_cover_rate"] = tg.loc[i, "ats_cover_rate"]

    return games


# ------------------------------------------------------------------
# GAME_FEATURES JOIN  (FIP / wRC+ / park — the backfilled as-of-date signal)
# ------------------------------------------------------------------

def add_market_features(games):
    """
    Join market_odds so the model can learn to predict the RESIDUAL vs the
    closing line instead of fighting it from scratch. This is the single
    highest-value feature addition — without a market anchor the model drifts
    toward 50/50 and calibration compresses everything.

    Brings in:
      market_home_prob_novig  — de-vigged market home win probability
      market_total_line       — the market's total

    De-vig: normalize the two implied probabilities so they sum to 1. Uses
    market_home_prob/away_prob if present, else derives from the moneylines.
    Joins on date+teams (market_odds game_pk is ~90% populated; date+teams is
    the more complete key here).
    """
    print("Joining market_odds (de-vigged prob + total line)...")
    try:
        mo = pd.read_sql("""
            SELECT official_date, home_team, away_team,
                   market_home_ml, market_away_ml,
                   market_home_prob, market_away_prob,
                   market_total_line
            FROM market_odds
        """, engine)
        if mo.empty:
            print("  ⚠️  market_odds empty — market features will be absent")
            games["market_home_prob_novig"] = pd.NA
            games["market_total_line_feat"] = pd.NA
            return games

        mo["official_date"] = pd.to_datetime(mo["official_date"])

        def ml_to_prob(ml):
            try:
                m = float(ml)
            except (TypeError, ValueError):
                return None
            if m == 0 or pd.isna(m):
                return None
            return (-m / (-m + 100.0)) if m < 0 else (100.0 / (m + 100.0))

        hp = pd.to_numeric(mo["market_home_prob"], errors="coerce")
        ap = pd.to_numeric(mo["market_away_prob"], errors="coerce")
        hp = hp.fillna(mo["market_home_ml"].map(ml_to_prob))
        ap = ap.fillna(mo["market_away_ml"].map(ml_to_prob))
        denom = hp + ap
        mo["market_home_prob_novig"] = (hp / denom).where(denom > 0)
        mo["market_total_line_feat"] = pd.to_numeric(mo["market_total_line"], errors="coerce")

        keep = mo[["official_date", "home_team", "away_team",
                   "market_home_prob_novig", "market_total_line_feat"]].drop_duplicates(
                   subset=["official_date", "home_team", "away_team"])

        before = len(games)
        games = games.merge(keep, on=["official_date", "home_team", "away_team"], how="left")
        if len(games) != before:
            print(f"  ⚠️  row count changed {before}->{len(games)} (dup odds rows?)")

        cov = games["market_home_prob_novig"].notna().mean()
        print(f"  market join: {cov:.1%} of games have de-vigged market prob")
    except Exception as e:
        print(f"  ⚠️  market_odds join failed: {e}")
        games["market_home_prob_novig"] = pd.NA
        games["market_total_line_feat"] = pd.NA
    return games


def add_game_features(games):
    """
    Join the game_features table produced by backfill_pitching_offense.py.
    These are strict as-of-date, leak-free, id-based FIP/WHIP, bullpen FIP,
    offense proxy, and park factors — the real pitching/park signal the
    totals model needs. Without this join, none of that reaches training.
    """
    print("Joining game_features (as-of-date FIP / offense / park)...")
    try:
        gf = pd.read_sql("""
            SELECT game_pk,
                   home_sp_fip, away_sp_fip,
                   home_sp_whip, away_sp_whip,
                   home_bullpen_fip, away_bullpen_fip,
                   home_wrc_plus, away_wrc_plus,
                   park_run_factor, park_hr_factor,
                   sp_fip_diff, bullpen_fip_diff, offense_wrc_diff
            FROM game_features
        """, engine)
        if gf.empty:
            print("  ⚠️  game_features is empty — run backfill_pitching_offense.py first!")
            return games

        before = len(games)
        games = games.merge(gf, on="game_pk", how="left")

        cov = games["sp_fip_diff"].notna().mean()
        real = (games["sp_fip_diff"].fillna(0) != 0).mean()
        print(f"  game_features join: {cov:.1%} of games have a feature row; "
              f"{real:.1%} have non-zero sp_fip_diff (real pitching signal)")
        if before != len(games):
            print(f"  ⚠️  row count changed {before}->{len(games)} (duplicate game_pk in game_features?)")
    except Exception as e:
        print(f"  ⚠️  game_features join failed: {e}")
    return games


def add_lineup_features(games):
    """
    Pull lineup OPS and hard-hit features from the predictions table.
    The engine saves these after every run, so coverage grows over time.
    Falls back to league avg for games without lineup data.

    Joins on game_pk — only games the engine has already run will have
    real values. Historical games (2024-2025) will get league avg defaults,
    which is fine since lineup signal in historical data is thin anyway.
    """
    print("Loading lineup features from predictions table...")
    try:
        pred_cols = pd.read_sql("""
            SELECT game_pk,
                   home_lineup_ops_vs_sp,
                   away_lineup_ops_vs_sp,
                   home_lineup_hard_hit,
                   away_lineup_hard_hit
            FROM predictions
            WHERE home_lineup_ops_vs_sp IS NOT NULL
              AND home_lineup_ops_vs_sp != 0.720
        """, engine)

        if pred_cols.empty:
            print("  ⚠️  No real lineup data in predictions table yet — using league avg")
            games["home_lineup_ops_vs_sp"] = LEAGUE_AVG_OPS
            games["away_lineup_ops_vs_sp"] = LEAGUE_AVG_OPS
            games["home_lineup_hard_hit"]  = LEAGUE_AVG_HH
            games["away_lineup_hard_hit"]  = LEAGUE_AVG_HH
            return games

        before = len(games)
        games = games.merge(pred_cols, on="game_pk", how="left")

        # Fill unmatched rows with league avg
        games["home_lineup_ops_vs_sp"] = games["home_lineup_ops_vs_sp"].fillna(LEAGUE_AVG_OPS)
        games["away_lineup_ops_vs_sp"] = games["away_lineup_ops_vs_sp"].fillna(LEAGUE_AVG_OPS)
        games["home_lineup_hard_hit"]  = games["home_lineup_hard_hit"].fillna(LEAGUE_AVG_HH)
        games["away_lineup_hard_hit"]  = games["away_lineup_hard_hit"].fillna(LEAGUE_AVG_HH)

        real_coverage = (games["home_lineup_ops_vs_sp"] != LEAGUE_AVG_OPS).sum()
        print(f"  Lineup coverage: {real_coverage}/{before} games ({real_coverage/before:.1%}) have real BvP scores")
        print(f"  Remaining {before - real_coverage} games use league avg ({LEAGUE_AVG_OPS})")

    except Exception as e:
        print(f"  ⚠️  Lineup feature join failed: {e} — using league avg")
        games["home_lineup_ops_vs_sp"] = LEAGUE_AVG_OPS
        games["away_lineup_ops_vs_sp"] = LEAGUE_AVG_OPS
        games["home_lineup_hard_hit"]  = LEAGUE_AVG_HH
        games["away_lineup_hard_hit"]  = LEAGUE_AVG_HH

    return games


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def build_training_dataset():
    print("Loading tables...")
    games    = pd.read_sql("SELECT * FROM games", engine)
    pitchers = pd.read_sql("SELECT * FROM pitchers", engine)

    print(f"Initial rows: {len(games)}")

    games = games[
        games.home_team.isin(MLB_TEAMS) &
        games.away_team.isin(MLB_TEAMS)
    ].copy()
    print(f"Rows after MLB filter: {len(games)}")

    games["game_date"] = pd.to_datetime(games.get("game_date", games.get("official_date")))
    games["season"]    = games.game_date.dt.year
    games = games.dropna(subset=["home_score", "away_score"])

    # Drop spring training / preseason. Regular seasons open late March; ST
    # games (Feb + early March) have no prior FIP history AND unrepresentative
    # scoring, so they poison a totals model. Keep ~Mar 18 onward each year.
    before_st = len(games)
    doy = games["game_date"].dt.strftime("%m-%d")
    games = games[doy >= "03-18"].copy()
    print(f"Rows after dropping spring training (< Mar 18): {len(games)} "
          f"(removed {before_st - len(games)})")

    # League average fallbacks
    pitchers["pitcher_name_clean"] = pitchers["pitcher_name"].apply(clean_name)
    league_era  = float(pitchers["era"].mean())  if not pitchers.empty else 4.20
    league_whip = float(pitchers["whip"].mean()) if not pitchers.empty else 1.30
    print(f"League avg ERA={league_era:.2f} WHIP={league_whip:.2f}")

    # ------------------------------------------------------------------
    # SP ERA — join directly from pitcher_game_log by game_pk + team
    # ------------------------------------------------------------------
    print("Computing SP ERA from pitcher_game_log by game_pk + team...")
    try:
        sp_log = pd.read_sql("""
            SELECT game_pk, pitcher_name, team, role,
                   innings_pitched, er_allowed, hits_allowed, walks
            FROM pitcher_game_log
            WHERE role = 'SP'
        """, engine)

        sp_log["pitcher_name_clean"] = sp_log["pitcher_name"].apply(clean_name)
        sp_log["innings_pitched"]    = pd.to_numeric(sp_log["innings_pitched"], errors="coerce")
        sp_log["er_allowed"]         = pd.to_numeric(sp_log["er_allowed"],      errors="coerce").fillna(0)
        sp_log["hits_allowed"]       = pd.to_numeric(sp_log["hits_allowed"],    errors="coerce").fillna(0)
        sp_log["walks"]              = pd.to_numeric(sp_log["walks"],           errors="coerce").fillna(0)
        sp_log = sp_log.dropna(subset=["innings_pitched"])
        sp_log = sp_log.sort_values(["pitcher_name_clean", "game_pk"])

        g = sp_log.groupby("pitcher_name_clean")
        sp_log["rolling_ip"] = g["innings_pitched"].transform(
            lambda x: x.shift(1).rolling(SP_ROLL_STARTS, min_periods=1).sum()
        )
        sp_log["rolling_er"] = g["er_allowed"].transform(
            lambda x: x.shift(1).rolling(SP_ROLL_STARTS, min_periods=1).sum()
        )
        sp_log["rolling_h"]  = g["hits_allowed"].transform(
            lambda x: x.shift(1).rolling(SP_ROLL_STARTS, min_periods=1).sum()
        )
        sp_log["rolling_bb"] = g["walks"].transform(
            lambda x: x.shift(1).rolling(SP_ROLL_STARTS, min_periods=1).sum()
        )

        sp_log["sp_era"]  = (sp_log["rolling_er"] / sp_log["rolling_ip"].clip(lower=0.1)) * 9
        sp_log["sp_whip"] = (sp_log["rolling_h"] + sp_log["rolling_bb"]) / sp_log["rolling_ip"].clip(lower=0.1)

        home_sp = sp_log[["game_pk","team","sp_era","sp_whip","pitcher_name_clean"]].copy()
        home_sp.columns = ["game_pk","home_team","home_era","home_whip","home_pitcher_clean"]
        games = games.merge(home_sp, on=["game_pk","home_team"], how="left")

        away_sp = sp_log[["game_pk","team","sp_era","sp_whip","pitcher_name_clean"]].copy()
        away_sp.columns = ["game_pk","away_team","away_era","away_whip","away_pitcher_clean"]
        games = games.merge(away_sp, on=["game_pk","away_team"], how="left")

        games["home_era"]  = games["home_era"].fillna(league_era)
        games["away_era"]  = games["away_era"].fillna(league_era)
        games["home_whip"] = games["home_whip"].fillna(league_whip)
        games["away_whip"] = games["away_whip"].fillna(league_whip)

        sp_found = (games["home_era"] != league_era).sum()
        print(f"  Games with real SP ERA: {sp_found}/{len(games)} ({sp_found/len(games):.1%})")

    except Exception as e:
        print(f"  ⚠️  SP ERA join failed: {e} — using league avg")
        games["home_era"]           = league_era
        games["away_era"]           = league_era
        games["home_whip"]          = league_whip
        games["away_whip"]          = league_whip
        games["home_pitcher_clean"] = None
        games["away_pitcher_clean"] = None

    # Targets
    games["run_diff"]   = games.home_score - games.away_score
    games["total_runs"] = games.home_score + games.away_score
    games["home_win"]   = (games.run_diff > 0).astype(int)

    # Base features
    games["era_diff"]  = games.home_era  - games.away_era
    games["whip_diff"] = games.home_whip - games.away_whip

    print(f"\nERA diff stats:")
    print(f"  std={games['era_diff'].std():.3f}  min={games['era_diff'].min():.2f}  max={games['era_diff'].max():.2f}")
    print(f"  % of games with |era_diff| > 1.0: {(games['era_diff'].abs() > 1.0).mean():.1%}")

    print("\nAdding rolling features...")
    games = add_rolling_features(games)

    print("Adding rest days...")
    games = add_rest_days(games)

    print("Adding bullpen usage...")
    games = add_bullpen(games)

    print("Adding win pct...")
    games = add_win_pct(games)

    print(f"Adding O/U tendency (last {OU_WINDOW} games)...")
    games = add_ou_tendency(games)

    print(f"Adding ATS cover rate (last {ATS_WINDOW} games)...")
    games = add_ats_cover_rate(games)

    print("Adding lineup features from predictions table...")
    games = add_lineup_features(games)

    print("Joining backfilled game_features...")
    games = add_game_features(games)

    print("Joining market_odds features...")
    games = add_market_features(games)
    # retrain.py's MARKET_FEATURE_COLS expects 'market_total_line'; expose it
    # under that name (the raw games column, if any, is unused by the model).
    if "market_total_line_feat" in games.columns:
        games["market_total_line"] = games["market_total_line_feat"]

    # Prefer real as-of-date FIP diff over the crude name-based ERA diff.
    # Keep era_diff for backward compatibility, but sp_fip_diff is the better
    # pitching signal and is what retrain.py should weight for totals.
    if "sp_fip_diff" in games.columns:
        games["sp_fip_diff"] = pd.to_numeric(games["sp_fip_diff"], errors="coerce").fillna(0.0)
    if "bullpen_fip_diff" in games.columns:
        games["bullpen_fip_diff"] = pd.to_numeric(games["bullpen_fip_diff"], errors="coerce").fillna(0.0)
    if "offense_wrc_diff" in games.columns:
        games["offense_wrc_diff"] = pd.to_numeric(games["offense_wrc_diff"], errors="coerce").fillna(0.0)
    for c in ["home_wrc_plus", "away_wrc_plus"]:
        if c in games.columns:
            games[c] = pd.to_numeric(games[c], errors="coerce").fillna(100.0)
    for c in ["park_run_factor", "park_hr_factor"]:
        if c in games.columns:
            games[c] = pd.to_numeric(games[c], errors="coerce").fillna(1.0)

    # ------------------------------------------------------------------
    # Weather — join from game_weather table
    # wind_out_factor: +ve = wind blowing out (adds runs), -ve = blowing in
    # Domes get is_dome=True; their weather cols are left NULL and filled below.
    # ------------------------------------------------------------------
    try:
        weather = pd.read_sql("""
            SELECT game_pk,
                   temp_f, wind_speed_mph, wind_out_factor,
                   precip_mm, visibility_m, cloud_cover_pct, is_dome
            FROM game_weather
        """, engine)
        if not weather.empty:
            games = games.merge(weather, on="game_pk", how="left")
            coverage = games["temp_f"].notna().mean()
            print(f"Weather joined: {len(weather)} rows  coverage={coverage:.1%}")
        else:
            print("⚠️  game_weather is empty — run backfill_weather.py first")
    except Exception as e:
        print(f"⚠️  Weather join failed: {e} — continuing without weather")

    # Fill defaults
    games = games.fillna({
        "home_last10_runs_scored":  4.5,
        "away_last10_runs_scored":  4.5,
        "home_last10_runs_allowed": 4.5,
        "away_last10_runs_allowed": 4.5,
        "home_last10_run_diff":     0.0,
        "away_last10_run_diff":     0.0,
        "home_sp_rest_days":        5.0,
        "away_sp_rest_days":        5.0,
        "home_bullpen_ip_4d":       4.0,
        "away_bullpen_ip_4d":       4.0,
        "home_win_pct_home":        0.5,
        "away_win_pct_away":        0.5,
        "home_ou_over_rate":        0.5,
        "away_ou_over_rate":        0.5,
        "home_last_game_total":     8.5,
        "away_last_game_total":     8.5,
        "home_ats_cover_rate":      0.5,
        "away_ats_cover_rate":      0.5,
        "home_lineup_ops_vs_sp":    LEAGUE_AVG_OPS,
        "away_lineup_ops_vs_sp":    LEAGUE_AVG_OPS,
        "home_lineup_hard_hit":     LEAGUE_AVG_HH,
        "away_lineup_hard_hit":     LEAGUE_AVG_HH,
        # Weather defaults: league-average conditions, dome=False
        "temp_f":           72.0,    # comfortable room-ish temp
        "wind_speed_mph":    5.0,
        "wind_out_factor":   0.0,    # neutral (crosswind)
        "precip_mm":         0.0,
        "visibility_m":  10000.0,    # clear visibility
        "cloud_cover_pct":  50.0,
        "is_dome":          False,
    })

    # For domes: zero out weather effects (dome overrides any fetched values)
    if "is_dome" in games.columns:
        dome_mask = games["is_dome"].astype(bool)
        games.loc[dome_mask, "wind_out_factor"] = 0.0
        games.loc[dome_mask, "temp_f"]          = 72.0   # controlled temp
        games.loc[dome_mask, "precip_mm"]       = 0.0
        games.loc[dome_mask, "visibility_m"]    = 10000.0

    print(f"\nFinal rows: {len(games)}")
    print(f"ERA diff variance: std={games['era_diff'].std():.3f}")

    # Summary of lineup coverage in final dataset
    lineup_coverage = (games["home_lineup_ops_vs_sp"] != LEAGUE_AVG_OPS).mean()
    print(f"Lineup coverage in final dataset: {lineup_coverage:.1%}")

    out = os.path.join(BASE_DIR, "ml", "training_data.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    games.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    build_training_dataset()