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

OU_WINDOW  = 20   # games for O/U tendency
ATS_WINDOW = 20   # games for ATS cover rate


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
        hp = r.home_pitcher_clean
        ap = r.away_pitcher_clean
        home.append((d - last.get(hp, d)).days if hp in last else 5)
        away.append((d - last.get(ap, d)).days if ap in last else 5)
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
        h = h_hist.home_win.mean()          if len(h_hist) > 5 else 0.5
        a = (1 - a_hist.home_win).mean()    if len(a_hist) > 5 else 0.5
        h_pct.append(h)
        a_pct.append(a)
    games["home_win_pct_home"] = h_pct
    games["away_win_pct_away"] = a_pct
    return games


def add_ou_tendency(games, ou_line_col="total_runs", window=OU_WINDOW):
    """
    For each game compute rolling O/U hit rate for both teams
    over their last N games using a default line of 8.5.
    This tells the model whether a team tends to play high or low scoring games.
    """
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

        tg["game_total"]   = tg["home_score"] + tg["away_score"]
        tg["went_over"]    = (tg["game_total"] > DEFAULT_LINE).astype(float)
        tg["over_rate"]    = tg["went_over"].shift(1).rolling(window, min_periods=3).mean()
        tg["last_total"]   = tg["game_total"].shift(1)

        for i in tg.index:
            if games.loc[i, "home_team"] == team:
                games.loc[i, "home_ou_over_rate"]    = tg.loc[i, "over_rate"]
                games.loc[i, "home_last_game_total"] = tg.loc[i, "last_total"]
            else:
                games.loc[i, "away_ou_over_rate"]    = tg.loc[i, "over_rate"]
                games.loc[i, "away_last_game_total"] = tg.loc[i, "last_total"]

    return games


def add_ats_cover_rate(games, window=ATS_WINDOW):
    """
    For each game compute rolling ATS cover rate for both teams
    over their last N games using -1.5 as the spread.
    Positive run diff > 1.5 = covered -1.5.
    """
    games = games.sort_values("game_date").copy()

    for col in ["home_ats_cover_rate", "away_ats_cover_rate"]:
        games[col] = pd.NA

    for team in MLB_TEAMS:
        tg = games[(games.home_team == team) | (games.away_team == team)].copy()
        tg = tg.sort_values("game_date")
        if len(tg) == 0:
            continue

        # From this team's perspective: did they cover -1.5?
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

    # Pitcher columns
    if "home_pitcher" not in games.columns:
        games["home_pitcher"] = None
    if "away_pitcher" not in games.columns:
        games["away_pitcher"] = None

    games["home_pitcher_clean"] = games["home_pitcher"].apply(clean_name)
    games["away_pitcher_clean"] = games["away_pitcher"].apply(clean_name)
    pitchers["pitcher_name_clean"] = pitchers["pitcher_name"].apply(clean_name)

    # Merge pitcher stats
    home_pitchers = pitchers[["pitcher_name_clean","season","era","whip"]].copy()
    home_pitchers = home_pitchers.rename(columns={"era":"home_era","whip":"home_whip"})
    games = games.merge(home_pitchers, how="left",
                        left_on=["home_pitcher_clean","season"],
                        right_on=["pitcher_name_clean","season"])
    games = games.drop(columns=["pitcher_name_clean"], errors="ignore")

    away_pitchers = pitchers[["pitcher_name_clean","season","era","whip"]].copy()
    away_pitchers = away_pitchers.rename(columns={"era":"away_era","whip":"away_whip"})
    games = games.merge(away_pitchers, how="left",
                        left_on=["away_pitcher_clean","season"],
                        right_on=["pitcher_name_clean","season"])
    games = games.drop(columns=["pitcher_name_clean"], errors="ignore")

    league_era  = pitchers["era"].mean()
    league_whip = pitchers["whip"].mean()
    games["home_era"]  = games["home_era"].fillna(league_era)
    games["away_era"]  = games["away_era"].fillna(league_era)
    games["home_whip"] = games["home_whip"].fillna(league_whip)
    games["away_whip"] = games["away_whip"].fillna(league_whip)

    # Targets
    games["run_diff"]   = games.home_score - games.away_score
    games["total_runs"] = games.home_score + games.away_score
    games["home_win"]   = (games.run_diff > 0).astype(int)

    # Base features
    games["era_diff"]  = games.home_era  - games.away_era
    games["whip_diff"] = games.home_whip - games.away_whip

    print("Adding rolling features...")
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
    })

    print(f"Final rows: {len(games)}")

    # Feature check
    new_cols = [
        "home_ou_over_rate","away_ou_over_rate",
        "home_last_game_total","away_last_game_total",
        "home_ats_cover_rate","away_ats_cover_rate",
    ]
    print("\nNew feature stats:")
    print(games[new_cols].describe().round(3))

    out = os.path.join(BASE_DIR, "ml", "training_data.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    games.to_csv(out, index=False)

    print(games[["game_date","home_team","away_team","home_pitcher","away_pitcher"]].head(10))
    print(games[["home_era","away_era","home_whip","away_whip"]].isna().mean())
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    build_training_dataset()