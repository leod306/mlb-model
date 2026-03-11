from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values

# Free/public data
from pybaseball import batting_stats, pitching_stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)

GAMES_TABLE = os.getenv("MLB_GAMES_TABLE", "games")
PROB_TABLE = os.getenv("MLB_PROBABLES_TABLE", "game_probables")
FEATURES_TABLE = os.getenv("MLB_FEATURES_TABLE", "game_features")

SEASON = int(os.getenv("MLB_SEASON", "2026"))
LOOKAHEAD_DAYS = int(os.getenv("FEATURES_LOOKAHEAD_DAYS", "2"))
HTTP_TIMEOUT = int(os.getenv("FEATURES_HTTP_TIMEOUT", "25"))

# ---------------------------------------------
# Park factors (baseline hand-tuned defaults)
# Replace with a more exact table later if you want
# ---------------------------------------------
PARK_FACTORS = {
    "ARI": {"run": 1.00, "hr": 1.03},
    "ATH": {"run": 0.96, "hr": 0.94},
    "ATL": {"run": 1.02, "hr": 1.06},
    "BAL": {"run": 1.01, "hr": 1.04},
    "BOS": {"run": 1.04, "hr": 1.03},
    "CHC": {"run": 1.01, "hr": 1.03},
    "CHW": {"run": 0.99, "hr": 1.00},
    "CIN": {"run": 1.07, "hr": 1.12},
    "CLE": {"run": 0.97, "hr": 0.98},
    "COL": {"run": 1.24, "hr": 1.19},
    "DET": {"run": 0.95, "hr": 0.92},
    "HOU": {"run": 0.99, "hr": 1.03},
    "KC":  {"run": 1.00, "hr": 0.96},
    "LAA": {"run": 1.00, "hr": 1.01},
    "LAD": {"run": 0.99, "hr": 0.98},
    "MIA": {"run": 0.94, "hr": 0.91},
    "MIL": {"run": 1.02, "hr": 1.05},
    "MIN": {"run": 1.01, "hr": 1.02},
    "NYM": {"run": 0.98, "hr": 0.96},
    "NYY": {"run": 1.03, "hr": 1.11},
    "OAK": {"run": 0.96, "hr": 0.94},
    "PHI": {"run": 1.05, "hr": 1.09},
    "PIT": {"run": 0.98, "hr": 0.95},
    "SD":  {"run": 0.96, "hr": 0.94},
    "SEA": {"run": 0.95, "hr": 0.93},
    "SF":  {"run": 0.93, "hr": 0.89},
    "STL": {"run": 0.99, "hr": 0.97},
    "TB":  {"run": 0.97, "hr": 0.95},
    "TEX": {"run": 1.04, "hr": 1.07},
    "TOR": {"run": 1.03, "hr": 1.06},
    "WSH": {"run": 1.00, "hr": 1.01},
}

# ---------------------------------------------
# Stadium coordinates for weather
# keyed by HOME team abbreviation in your DB
# ---------------------------------------------
STADIUM_COORDS = {
    "ARI": (33.4453, -112.0667),
    "ATH": (38.1480, -121.8170),  # Sacramento temp placeholder for A's move period
    "ATL": (33.8907, -84.4677),
    "BAL": (39.2840, -76.6218),
    "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),
    "CHW": (41.8300, -87.6338),
    "CIN": (39.0979, -84.5083),
    "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942),
    "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),
    "KC":  (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827),
    "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),
    "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2776),
    "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),
    "PHI": (39.9057, -75.1665),
    "PIT": (40.4469, -80.0057),
    "SD":  (32.7073, -117.1566),
    "SEA": (47.5914, -122.3325),
    "SF":  (37.7786, -122.3893),
    "STL": (38.6226, -90.1928),
    "TB":  (27.7682, -82.6534),
    "TEX": (32.7513, -97.0825),
    "TOR": (43.6414, -79.3894),
    "WSH": (38.8730, -77.0074),
}

TEAM_ALIASES = {
    "AZ": "ARI",
    "ARZ": "ARI",
    "KC": "KC",
    "KCR": "KC",
    "SD": "SD",
    "SDP": "SD",
    "SF": "SF",
    "SFG": "SF",
    "TB": "TB",
    "TBR": "TB",
    "WSH": "WSH",
    "WAS": "WSH",
    "CWS": "CHW",
    "CHW": "CHW",
    "ATH": "ATH",
    "OAK": "ATH",  # map old A's key into current app key
}

# ---------------------------------------------
# Helpers
# ---------------------------------------------
def conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)


def norm_team(team: str) -> str:
    t = (team or "").strip().upper()
    return TEAM_ALIASES.get(t, t)


def norm_name(name: str) -> str:
    s = (name or "").strip().lower()
    for ch in [".", ",", "'", "-", "`"]:
        s = s.replace(ch, "")
    s = " ".join(s.split())
    return s


def pick_col(df: pd.DataFrame, choices: list[str]) -> Optional[str]:
    for c in choices:
        if c in df.columns:
            return c
    return None


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def weighted_avg(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if value_col not in df.columns or weight_col not in df.columns or df.empty:
        return 0.0
    work = df[[value_col, weight_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
    work = work.dropna()
    if work.empty or work[weight_col].sum() == 0:
        return 0.0
    return float((work[value_col] * work[weight_col]).sum() / work[weight_col].sum())


def ensure_features_table(cur) -> None:
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {FEATURES_TABLE} (
            game_pk BIGINT PRIMARY KEY,
            official_date DATE NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,

            home_sp_fip DOUBLE PRECISION,
            away_sp_fip DOUBLE PRECISION,
            home_sp_xfip DOUBLE PRECISION,
            away_sp_xfip DOUBLE PRECISION,
            home_sp_siera DOUBLE PRECISION,
            away_sp_siera DOUBLE PRECISION,
            home_sp_kbb DOUBLE PRECISION,
            away_sp_kbb DOUBLE PRECISION,
            home_sp_era DOUBLE PRECISION,
            away_sp_era DOUBLE PRECISION,
            home_sp_whip DOUBLE PRECISION,
            away_sp_whip DOUBLE PRECISION,

            home_bullpen_era DOUBLE PRECISION,
            away_bullpen_era DOUBLE PRECISION,
            home_bullpen_fip DOUBLE PRECISION,
            away_bullpen_fip DOUBLE PRECISION,
            home_bullpen_xfip DOUBLE PRECISION,
            away_bullpen_xfip DOUBLE PRECISION,

            home_wrc_plus DOUBLE PRECISION,
            away_wrc_plus DOUBLE PRECISION,
            home_iso DOUBLE PRECISION,
            away_iso DOUBLE PRECISION,
            home_bb_pct DOUBLE PRECISION,
            away_bb_pct DOUBLE PRECISION,
            home_k_pct DOUBLE PRECISION,
            away_k_pct DOUBLE PRECISION,

            park_run_factor DOUBLE PRECISION,
            park_hr_factor DOUBLE PRECISION,

            temperature_f DOUBLE PRECISION,
            wind_speed_mph DOUBLE PRECISION,
            wind_direction_deg DOUBLE PRECISION,
            humidity_pct DOUBLE PRECISION,

            sp_fip_diff DOUBLE PRECISION,
            sp_xfip_diff DOUBLE PRECISION,
            bullpen_fip_diff DOUBLE PRECISION,
            offense_wrc_diff DOUBLE PRECISION,

            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{FEATURES_TABLE}_official_date ON {FEATURES_TABLE}(official_date);")


def load_games_for_window(cur, start_date: date, end_date: date) -> pd.DataFrame:
    # Pull only what we need; optional columns handled if missing.
    cur.execute(f"""
        SELECT
            g.game_pk,
            g.official_date,
            g.game_date_utc,
            g.home_team,
            g.away_team,
            p.home_sp_id,
            p.away_sp_id,
            p.home_sp_name,
            p.away_sp_name
        FROM {GAMES_TABLE} g
        LEFT JOIN {PROB_TABLE} p
          ON p.game_pk = g.game_pk
        WHERE g.official_date BETWEEN %s AND %s
        ORDER BY g.official_date, g.game_pk
    """, (start_date, end_date))
    rows = cur.fetchall()

    return pd.DataFrame(rows, columns=[
        "game_pk",
        "official_date",
        "game_date_utc",
        "home_team",
        "away_team",
        "home_sp_id",
        "away_sp_id",
        "home_sp_name",
        "away_sp_name",
    ])


def fetch_pitching_df(season: int) -> pd.DataFrame:
    # FanGraphs player pitching via pybaseball
    df = pitching_stats(season, season, qual=0, ind=1)
    if df is None or len(df) == 0:
        raise RuntimeError("pitching_stats returned no rows")
    return df


def fetch_batting_df(season: int) -> pd.DataFrame:
    # FanGraphs player batting via pybaseball
    df = batting_stats(season, season, qual=0, ind=1)
    if df is None or len(df) == 0:
        raise RuntimeError("batting_stats returned no rows")
    return df


def normalize_fg_team(team: str) -> str:
    t = (team or "").strip().upper()
    fg_map = {
        "KCR": "KC",
        "SDP": "SD",
        "SFG": "SF",
        "TBR": "TB",
        "WSN": "WSH",
        "CWS": "CHW",
        "ARI": "ARI",
        "AZ": "ARI",
        "OAK": "ATH",
    }
    return fg_map.get(t, t)


def build_pitcher_lookup(pitch_df: pd.DataFrame) -> Dict[str, dict]:
    name_col = pick_col(pitch_df, ["Name", "PlayerName"])
    team_col = pick_col(pitch_df, ["Team", "Tm"])
    if not name_col or not team_col:
        raise RuntimeError("Could not find required pitcher name/team columns")

    out: Dict[str, dict] = {}
    for _, r in pitch_df.iterrows():
        key = norm_name(str(r[name_col]))
        out[key] = {
            "team": normalize_fg_team(str(r[team_col])),
            "era": safe_float(r.get("ERA")),
            "whip": safe_float(r.get("WHIP")),
            "fip": safe_float(r.get("FIP")),
            "xfip": safe_float(r.get("xFIP")),
            "siera": safe_float(r.get("SIERA")),
            "kbb": safe_float(r.get("K-BB%")),
        }
    return out


def build_bullpen_lookup(pitch_df: pd.DataFrame) -> Dict[str, dict]:
    team_col = pick_col(pitch_df, ["Team", "Tm"])
    ip_col = pick_col(pitch_df, ["IP", "IP_float"])
    gs_col = pick_col(pitch_df, ["GS"])
    if not team_col or not ip_col:
        raise RuntimeError("Could not find required bullpen columns")

    work = pitch_df.copy()
    work["team_norm"] = work[team_col].astype(str).map(normalize_fg_team)
    work[ip_col] = pd.to_numeric(work[ip_col], errors="coerce").fillna(0)

    # Treat GS == 0 as relievers when available; otherwise use all pitchers
    if gs_col and gs_col in work.columns:
        work[gs_col] = pd.to_numeric(work[gs_col], errors="coerce").fillna(0)
        rel = work[work[gs_col] == 0].copy()
        if rel.empty:
            rel = work.copy()
    else:
        rel = work.copy()

    out: Dict[str, dict] = {}
    for team, grp in rel.groupby("team_norm"):
        out[team] = {
            "era": weighted_avg(grp, "ERA", ip_col),
            "fip": weighted_avg(grp, "FIP", ip_col),
            "xfip": weighted_avg(grp, "xFIP", ip_col),
        }
    return out


def build_team_offense_lookup(bat_df: pd.DataFrame) -> Dict[str, dict]:
    team_col = pick_col(bat_df, ["Team", "Tm"])
    pa_col = pick_col(bat_df, ["PA"])
    if not team_col or not pa_col:
        raise RuntimeError("Could not find required batting team/PA columns")

    work = bat_df.copy()
    work["team_norm"] = work[team_col].astype(str).map(normalize_fg_team)
    work[pa_col] = pd.to_numeric(work[pa_col], errors="coerce").fillna(0)

    out: Dict[str, dict] = {}
    for team, grp in work.groupby("team_norm"):
        out[team] = {
            "wrc_plus": weighted_avg(grp, "wRC+", pa_col),
            "iso": weighted_avg(grp, "ISO", pa_col),
            "bb_pct": weighted_avg(grp, "BB%", pa_col),
            "k_pct": weighted_avg(grp, "K%", pa_col),
        }
    return out


def get_park_factors(home_team: str) -> dict:
    t = norm_team(home_team)
    return PARK_FACTORS.get(t, {"run": 1.00, "hr": 1.00})


def fetch_weather_for_game(home_team: str, game_dt_utc: Any) -> dict:
    t = norm_team(home_team)
    coords = STADIUM_COORDS.get(t)
    if not coords:
        return {
            "temp_f": 72.0,
            "wind_mph": 7.0,
            "wind_dir": 0.0,
            "humidity": 50.0,
        }

    if pd.isna(game_dt_utc):
        game_time = datetime.now(timezone.utc)
    elif isinstance(game_dt_utc, datetime):
        game_time = game_dt_utc.astimezone(timezone.utc)
    else:
        try:
            game_time = pd.to_datetime(game_dt_utc, utc=True).to_pydatetime()
        except Exception:
            game_time = datetime.now(timezone.utc)

    lat, lon = coords
    start_date = game_time.date().isoformat()
    end_date = game_time.date().isoformat()

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
        "start_date": start_date,
        "end_date": end_date,
    }

    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        wind = hourly.get("wind_speed_10m", [])
        wind_dir = hourly.get("wind_direction_10m", [])

        if not times:
            raise RuntimeError("No hourly weather returned")

        target_hour = game_time.replace(minute=0, second=0, microsecond=0)
        best_i = 0
        best_delta = None

        for i, ts in enumerate(times):
            dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            delta = abs((dt - target_hour).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_i = i

        return {
            "temp_f": safe_float(temps[best_i], 72.0),
            "wind_mph": safe_float(wind[best_i], 7.0),
            "wind_dir": safe_float(wind_dir[best_i], 0.0),
            "humidity": safe_float(humidity[best_i], 50.0),
        }
    except Exception:
        return {
            "temp_f": 72.0,
            "wind_mph": 7.0,
            "wind_dir": 0.0,
            "humidity": 50.0,
        }


def build_feature_rows(games_df: pd.DataFrame,
                       pitcher_lookup: Dict[str, dict],
                       bullpen_lookup: Dict[str, dict],
                       offense_lookup: Dict[str, dict]) -> list[tuple]:
    rows: list[tuple] = []

    for _, g in games_df.iterrows():
        home_team = norm_team(str(g["home_team"]))
        away_team = norm_team(str(g["away_team"]))

        home_sp = pitcher_lookup.get(norm_name(str(g["home_sp_name"] or "")), {})
        away_sp = pitcher_lookup.get(norm_name(str(g["away_sp_name"] or "")), {})

        home_bp = bullpen_lookup.get(home_team, {})
        away_bp = bullpen_lookup.get(away_team, {})

        home_off = offense_lookup.get(home_team, {})
        away_off = offense_lookup.get(away_team, {})

        park = get_park_factors(home_team)
        weather = fetch_weather_for_game(home_team, g["game_date_utc"])

        home_sp_fip = safe_float(home_sp.get("fip"))
        away_sp_fip = safe_float(away_sp.get("fip"))
        home_sp_xfip = safe_float(home_sp.get("xfip"))
        away_sp_xfip = safe_float(away_sp.get("xfip"))

        home_bp_fip = safe_float(home_bp.get("fip"))
        away_bp_fip = safe_float(away_bp.get("fip"))

        home_wrc = safe_float(home_off.get("wrc_plus"), 100.0)
        away_wrc = safe_float(away_off.get("wrc_plus"), 100.0)

        rows.append((
            int(g["game_pk"]),
            g["official_date"],
            home_team,
            away_team,

            safe_float(home_sp.get("fip")),
            safe_float(away_sp.get("fip")),
            safe_float(home_sp.get("xfip")),
            safe_float(away_sp.get("xfip")),
            safe_float(home_sp.get("siera")),
            safe_float(away_sp.get("siera")),
            safe_float(home_sp.get("kbb")),
            safe_float(away_sp.get("kbb")),
            safe_float(home_sp.get("era")),
            safe_float(away_sp.get("era")),
            safe_float(home_sp.get("whip")),
            safe_float(away_sp.get("whip")),

            safe_float(home_bp.get("era")),
            safe_float(away_bp.get("era")),
            safe_float(home_bp.get("fip")),
            safe_float(away_bp.get("fip")),
            safe_float(home_bp.get("xfip")),
            safe_float(away_bp.get("xfip")),

            safe_float(home_off.get("wrc_plus"), 100.0),
            safe_float(away_off.get("wrc_plus"), 100.0),
            safe_float(home_off.get("iso")),
            safe_float(away_off.get("iso")),
            safe_float(home_off.get("bb_pct")),
            safe_float(away_off.get("bb_pct")),
            safe_float(home_off.get("k_pct")),
            safe_float(away_off.get("k_pct")),

            safe_float(park["run"], 1.0),
            safe_float(park["hr"], 1.0),

            safe_float(weather["temp_f"]),
            safe_float(weather["wind_mph"]),
            safe_float(weather["wind_dir"]),
            safe_float(weather["humidity"]),

            home_sp_fip - away_sp_fip,
            home_sp_xfip - away_sp_xfip,
            home_bp_fip - away_bp_fip,
            home_wrc - away_wrc,

            datetime.now(timezone.utc),
        ))

    return rows


def upsert_feature_rows(cur, rows: list[tuple]) -> int:
    if not rows:
        return 0

    sql = f"""
    INSERT INTO {FEATURES_TABLE} (
        game_pk,
        official_date,
        home_team,
        away_team,

        home_sp_fip,
        away_sp_fip,
        home_sp_xfip,
        away_sp_xfip,
        home_sp_siera,
        away_sp_siera,
        home_sp_kbb,
        away_sp_kbb,
        home_sp_era,
        away_sp_era,
        home_sp_whip,
        away_sp_whip,

        home_bullpen_era,
        away_bullpen_era,
        home_bullpen_fip,
        away_bullpen_fip,
        home_bullpen_xfip,
        away_bullpen_xfip,

        home_wrc_plus,
        away_wrc_plus,
        home_iso,
        away_iso,
        home_bb_pct,
        away_bb_pct,
        home_k_pct,
        away_k_pct,

        park_run_factor,
        park_hr_factor,

        temperature_f,
        wind_speed_mph,
        wind_direction_deg,
        humidity_pct,

        sp_fip_diff,
        sp_xfip_diff,
        bullpen_fip_diff,
        offense_wrc_diff,

        updated_at
    )
    VALUES %s
    ON CONFLICT (game_pk) DO UPDATE SET
        official_date = EXCLUDED.official_date,
        home_team = EXCLUDED.home_team,
        away_team = EXCLUDED.away_team,

        home_sp_fip = EXCLUDED.home_sp_fip,
        away_sp_fip = EXCLUDED.away_sp_fip,
        home_sp_xfip = EXCLUDED.home_sp_xfip,
        away_sp_xfip = EXCLUDED.away_sp_xfip,
        home_sp_siera = EXCLUDED.home_sp_siera,
        away_sp_siera = EXCLUDED.away_sp_siera,
        home_sp_kbb = EXCLUDED.home_sp_kbb,
        away_sp_kbb = EXCLUDED.away_sp_kbb,
        home_sp_era = EXCLUDED.home_sp_era,
        away_sp_era = EXCLUDED.away_sp_era,
        home_sp_whip = EXCLUDED.home_sp_whip,
        away_sp_whip = EXCLUDED.away_sp_whip,

        home_bullpen_era = EXCLUDED.home_bullpen_era,
        away_bullpen_era = EXCLUDED.away_bullpen_era,
        home_bullpen_fip = EXCLUDED.home_bullpen_fip,
        away_bullpen_fip = EXCLUDED.away_bullpen_fip,
        home_bullpen_xfip = EXCLUDED.home_bullpen_xfip,
        away_bullpen_xfip = EXCLUDED.away_bullpen_xfip,

        home_wrc_plus = EXCLUDED.home_wrc_plus,
        away_wrc_plus = EXCLUDED.away_wrc_plus,
        home_iso = EXCLUDED.home_iso,
        away_iso = EXCLUDED.away_iso,
        home_bb_pct = EXCLUDED.home_bb_pct,
        away_bb_pct = EXCLUDED.away_bb_pct,
        home_k_pct = EXCLUDED.home_k_pct,
        away_k_pct = EXCLUDED.away_k_pct,

        park_run_factor = EXCLUDED.park_run_factor,
        park_hr_factor = EXCLUDED.park_hr_factor,

        temperature_f = EXCLUDED.temperature_f,
        wind_speed_mph = EXCLUDED.wind_speed_mph,
        wind_direction_deg = EXCLUDED.wind_direction_deg,
        humidity_pct = EXCLUDED.humidity_pct,

        sp_fip_diff = EXCLUDED.sp_fip_diff,
        sp_xfip_diff = EXCLUDED.sp_xfip_diff,
        bullpen_fip_diff = EXCLUDED.bullpen_fip_diff,
        offense_wrc_diff = EXCLUDED.offense_wrc_diff,

        updated_at = NOW();
    """
    execute_values(cur, sql, rows, page_size=500)
    return len(rows)


def main():
    today = datetime.now(timezone.utc).date()
    end_date = today + timedelta(days=LOOKAHEAD_DAYS)

    print("Building enriched game features...")
    print("Window:", today, "->", end_date)

    c = conn()
    try:
        c.autocommit = False
        with c.cursor() as cur:
            ensure_features_table(cur)
            games_df = load_games_for_window(cur, today, end_date)

            if games_df.empty:
                c.commit()
                print("No games found in window.")
                return

            print(f"Games loaded: {len(games_df)}")
            print("Pulling pitcher stats...")
            pitch_df = fetch_pitching_df(SEASON)

            print("Pulling batting stats...")
            bat_df = fetch_batting_df(SEASON)

            pitcher_lookup = build_pitcher_lookup(pitch_df)
            bullpen_lookup = build_bullpen_lookup(pitch_df)
            offense_lookup = build_team_offense_lookup(bat_df)

            rows = build_feature_rows(
                games_df=games_df,
                pitcher_lookup=pitcher_lookup,
                bullpen_lookup=bullpen_lookup,
                offense_lookup=offense_lookup,
            )

            n = upsert_feature_rows(cur, rows)
            c.commit()

            print(f"Feature rows upserted: {n}")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()