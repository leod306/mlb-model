"""
build_team_features.py
----------------------
Builds enriched game features using Baseball Reference (via pybaseball).
Replaces FanGraphs which blocks automated requests.

Data sources:
  - pitching_stats_bref : ERA, WHIP, IP, BB, SO, HR per pitcher
  - batting_stats_bref  : OBP, SLG, OPS, BB%, K% per team
  - Park factors        : hand-tuned constants (no API needed)
  - Weather             : Open-Meteo free API

Computed features saved to game_features table:
  - home/away_sp_fip    : FIP proxy computed from BBRef stats
  - home/away_sp_era    : ERA from BBRef
  - home/away_sp_whip   : WHIP from BBRef
  - home/away_bullpen_* : Bullpen ERA/FIP proxy
  - home/away_wrc_plus  : OPS-based offense proxy (OPS * 100 / league_avg_ops)
  - sp_fip_diff, bullpen_fip_diff, offense_wrc_diff : differentials
  - park_run_factor, park_hr_factor : park constants
  - temperature_f, wind_speed_mph   : weather from Open-Meteo
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values
from pybaseball import pitching_stats_bref, batting_stats_bref

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getenv("DYNO") is None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=False)

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2://", "postgresql://", 1)

GAMES_TABLE    = os.getenv("MLB_GAMES_TABLE",    "games")
PROB_TABLE     = os.getenv("MLB_PROBABLES_TABLE", "game_probables")
FEATURES_TABLE = os.getenv("MLB_FEATURES_TABLE", "game_features")

SEASON          = int(os.getenv("MLB_SEASON", "2026"))
LOOKAHEAD_DAYS  = int(os.getenv("FEATURES_LOOKAHEAD_DAYS", "2"))
HTTP_TIMEOUT    = int(os.getenv("FEATURES_HTTP_TIMEOUT", "25"))

# FIP constant (league average HR/FB environment)
FIP_CONSTANT = 3.10

# League average OPS for wRC+ proxy (updated periodically)
LEAGUE_AVG_OPS = 0.720

# ---------------------------------------------
# Park factors
# ---------------------------------------------
PARK_FACTORS = {
    "ARI": {"run": 1.00, "hr": 1.03}, "ATH": {"run": 0.96, "hr": 0.94},
    "ATL": {"run": 1.02, "hr": 1.06}, "BAL": {"run": 1.01, "hr": 1.04},
    "BOS": {"run": 1.04, "hr": 1.03}, "CHC": {"run": 1.01, "hr": 1.03},
    "CHW": {"run": 0.99, "hr": 1.00}, "CIN": {"run": 1.07, "hr": 1.12},
    "CLE": {"run": 0.97, "hr": 0.98}, "COL": {"run": 1.24, "hr": 1.19},
    "DET": {"run": 0.95, "hr": 0.92}, "HOU": {"run": 0.99, "hr": 1.03},
    "KC":  {"run": 1.00, "hr": 0.96}, "LAA": {"run": 1.00, "hr": 1.01},
    "LAD": {"run": 0.99, "hr": 0.98}, "MIA": {"run": 0.94, "hr": 0.91},
    "MIL": {"run": 1.02, "hr": 1.05}, "MIN": {"run": 1.01, "hr": 1.02},
    "NYM": {"run": 0.98, "hr": 0.96}, "NYY": {"run": 1.03, "hr": 1.11},
    "OAK": {"run": 0.96, "hr": 0.94}, "PHI": {"run": 1.05, "hr": 1.09},
    "PIT": {"run": 0.98, "hr": 0.95}, "SD":  {"run": 0.96, "hr": 0.94},
    "SEA": {"run": 0.95, "hr": 0.93}, "SF":  {"run": 0.93, "hr": 0.89},
    "STL": {"run": 0.99, "hr": 0.97}, "TB":  {"run": 0.97, "hr": 0.95},
    "TEX": {"run": 1.04, "hr": 1.07}, "TOR": {"run": 1.03, "hr": 1.06},
    "WSH": {"run": 1.00, "hr": 1.01},
}

STADIUM_COORDS = {
    "ARI": (33.4453, -112.0667), "ATH": (38.5833, -121.5233),
    "ATL": (33.8907,  -84.4677), "BAL": (39.2840,  -76.6218),
    "BOS": (42.3467,  -71.0972), "CHC": (41.9484,  -87.6553),
    "CHW": (41.8300,  -87.6338), "CIN": (39.0979,  -84.5083),
    "CLE": (41.4962,  -81.6852), "COL": (39.7559, -104.9942),
    "DET": (42.3390,  -83.0485), "HOU": (29.7573,  -95.3555),
    "KC":  (39.0517,  -94.4803), "LAA": (33.8003, -117.8827),
    "LAD": (34.0739, -118.2400), "MIA": (25.7781,  -80.2197),
    "MIL": (43.0280,  -87.9712), "MIN": (44.9817,  -93.2776),
    "NYM": (40.7571,  -73.8458), "NYY": (40.8296,  -73.9262),
    "PHI": (39.9057,  -75.1665), "PIT": (40.4469,  -80.0057),
    "SD":  (32.7073, -117.1566), "SEA": (47.5914, -122.3325),
    "SF":  (37.7786, -122.3893), "STL": (38.6226,  -90.1928),
    "TB":  (27.7682,  -82.6534), "TEX": (32.7513,  -97.0825),
    "TOR": (43.6414,  -79.3894), "WSH": (38.8730,  -77.0074),
}

TEAM_ALIASES = {
    "AZ": "ARI", "ARZ": "ARI", "KCR": "KC", "SDP": "SD",
    "SFG": "SF", "TBR": "TB", "WSN": "WSH", "WAS": "WSH",
    "CWS": "CHW", "OAK": "ATH",
}

BREF_TEAM_MAP = {
    "Arizona Diamondbacks": "ARI",   "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",      "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",           "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",        "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",       "Detroit Tigers": "DET",
    "Houston Astros": "HOU",         "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",     "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",          "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",        "New York Mets": "NYM",
    "New York Yankees": "NYY",       "Oakland Athletics": "ATH",
    "Athletics": "ATH",              "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",     "San Diego Padres": "SD",
    "San Francisco Giants": "SF",    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",          "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
    # Short codes BBRef uses
    "Arizona": "ARI",   "Atlanta": "ATL",    "Baltimore": "BAL",
    "Boston": "BOS",    "Chi Cubs": "CHC",   "Chi White Sox": "CHW",
    "Cincinnati": "CIN","Cleveland": "CLE",  "Colorado": "COL",
    "Detroit": "DET",   "Houston": "HOU",    "Kansas City": "KC",
    "LA Angels": "LAA", "LA Dodgers": "LAD", "Miami": "MIA",
    "Milwaukee": "MIL", "Minnesota": "MIN",  "NY Mets": "NYM",
    "NY Yankees": "NYY","Oakland": "ATH",    "Philadelphia": "PHI",
    "Pittsburgh": "PIT","San Diego": "SD",   "San Francisco": "SF",
    "Seattle": "SEA",   "St. Louis": "STL",  "Tampa Bay": "TB",
    "Texas": "TEX",     "Toronto": "TOR",    "Washington": "WSH",
}


# =============================================================================
# HELPERS
# =============================================================================

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
    return " ".join(s.split())


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def bref_norm_team(tm: str) -> str:
    """Normalize a BBRef team name/abbreviation to our internal code."""
    if not tm:
        return ""
    # Try full name map first
    if tm in BREF_TEAM_MAP:
        return BREF_TEAM_MAP[tm]
    # Try upper abbreviation
    upper = tm.strip().upper()
    return TEAM_ALIASES.get(upper, upper)


# =============================================================================
# DATA FETCHERS
# =============================================================================

def fetch_pitching_df(season: int) -> pd.DataFrame:
    """Pull BBRef pitching stats. Returns empty df on failure."""
    try:
        df = pitching_stats_bref(season)
        if df is None or df.empty:
            print(f"⚠️  BBRef pitching returned no rows for {season}")
            return pd.DataFrame()
        print(f"✅ BBRef pitching loaded: {df.shape}")
        return df
    except Exception as e:
        print(f"⚠️  BBRef pitching failed: {e}")
        return pd.DataFrame()


def fetch_batting_df(season: int) -> pd.DataFrame:
    """Pull BBRef batting stats. Returns empty df on failure."""
    try:
        df = batting_stats_bref(season)
        if df is None or df.empty:
            print(f"⚠️  BBRef batting returned no rows for {season}")
            return pd.DataFrame()
        print(f"✅ BBRef batting loaded: {df.shape}")
        return df
    except Exception as e:
        print(f"⚠️  BBRef batting failed: {e}")
        return pd.DataFrame()


def compute_fip(era: float, bb: float, so: float, hr: float,
                ip: float) -> float:
    """
    FIP proxy from BBRef stats.
    Formula: FIP = ((13*HR + 3*BB - 2*SO) / IP) + FIP_constant
    Falls back to ERA if IP is too low.
    """
    if ip < 1.0:
        return era
    return ((13 * hr + 3 * bb - 2 * so) / ip) + FIP_CONSTANT


def build_pitcher_lookup(pitch_df: pd.DataFrame) -> Dict[str, dict]:
    """
    Build name -> stats dict from BBRef pitching data.
    Computes FIP proxy from available columns.
    """
    if pitch_df.empty:
        return {}

    # BBRef columns: Name, Tm, G, GS, IP, H, ER, BB, SO, HR, ERA, WHIP
    out: Dict[str, dict] = {}
    for _, r in pitch_df.iterrows():
        name = norm_name(str(r.get("Name", "")))
        if not name:
            continue

        ip  = safe_float(r.get("IP",  0))
        bb  = safe_float(r.get("BB",  0))
        so  = safe_float(r.get("SO",  0))
        hr  = safe_float(r.get("HR",  0))
        era = safe_float(r.get("ERA", 4.20))
        whip= safe_float(r.get("WHIP",1.30))
        gs  = safe_float(r.get("GS",  0))
        fip = compute_fip(era, bb, so, hr, ip)

        out[name] = {
            "team": bref_norm_team(str(r.get("Tm", ""))),
            "era":  era,
            "whip": whip,
            "fip":  round(fip, 3),
            "ip":   ip,
            "gs":   gs,
        }

    print(f"  Pitcher lookup entries: {len(out)}")
    return out


def build_bullpen_lookup(pitch_df: pd.DataFrame) -> Dict[str, dict]:
    """
    Build team -> bullpen stats from relievers (GS == 0).
    Weighted by IP.
    """
    if pitch_df.empty:
        return {}

    work = pitch_df.copy()
    work["gs_num"] = pd.to_numeric(work.get("GS", 0), errors="coerce").fillna(0)
    work["ip_num"] = pd.to_numeric(work.get("IP", 0), errors="coerce").fillna(0)
    relievers = work[work["gs_num"] == 0].copy()
    if relievers.empty:
        relievers = work.copy()

    relievers["team_norm"] = relievers["Tm"].astype(str).apply(bref_norm_team)

    out: Dict[str, dict] = {}
    for team, grp in relievers.groupby("team_norm"):
        if not team:
            continue
        total_ip = grp["ip_num"].sum()
        if total_ip == 0:
            continue

        def wav(col, default=4.50):
            vals = pd.to_numeric(grp.get(col, default), errors="coerce").fillna(default)
            return float((vals * grp["ip_num"]).sum() / total_ip)

        era  = wav("ERA")
        whip = wav("WHIP", 1.30)

        # Compute weighted FIP
        bb = pd.to_numeric(grp.get("BB", 0), errors="coerce").fillna(0).sum()
        so = pd.to_numeric(grp.get("SO", 0), errors="coerce").fillna(0).sum()
        hr = pd.to_numeric(grp.get("HR", 0), errors="coerce").fillna(0).sum()
        fip = compute_fip(era, bb, so, hr, total_ip)

        out[team] = {
            "era":  round(era, 3),
            "whip": round(whip, 3),
            "fip":  round(fip, 3),
        }

    print(f"  Bullpen lookup entries: {len(out)}")
    return out


def build_team_offense_lookup(bat_df: pd.DataFrame) -> Dict[str, dict]:
    """
    Build team -> offense stats from BBRef batting.
    Uses OPS as wRC+ proxy: wrc_plus_proxy = (team_OPS / LEAGUE_AVG_OPS) * 100
    Also computes BB% and K% from PA, BB, SO.
    """
    if bat_df.empty:
        return {}

    work = bat_df.copy()
    work["team_norm"] = work["Tm"].astype(str).apply(bref_norm_team)
    work["pa_num"]    = pd.to_numeric(work.get("PA", 0), errors="coerce").fillna(0)

    out: Dict[str, dict] = {}
    for team, grp in work.groupby("team_norm"):
        if not team:
            continue
        total_pa = grp["pa_num"].sum()
        if total_pa == 0:
            continue

        def wav(col, default=0.0):
            vals = pd.to_numeric(grp.get(col, default), errors="coerce").fillna(default)
            return float((vals * grp["pa_num"]).sum() / total_pa)

        ops = wav("OPS", LEAGUE_AVG_OPS)
        obp = wav("OBP", 0.320)
        slg = wav("SLG", 0.400)

        total_bb = pd.to_numeric(grp.get("BB", 0), errors="coerce").fillna(0).sum()
        total_so = pd.to_numeric(grp.get("SO", 0), errors="coerce").fillna(0).sum()

        # wRC+ proxy: scale OPS to 100 baseline
        wrc_proxy = round((ops / LEAGUE_AVG_OPS) * 100, 1) if ops > 0 else 100.0

        out[team] = {
            "wrc_plus": wrc_proxy,
            "ops":      round(ops, 3),
            "obp":      round(obp, 3),
            "slg":      round(slg, 3),
            "bb_pct":   round(total_bb / total_pa, 3) if total_pa > 0 else 0.08,
            "k_pct":    round(total_so / total_pa, 3) if total_pa > 0 else 0.22,
        }

    print(f"  Offense lookup entries: {len(out)}")
    return out


# =============================================================================
# PARK / WEATHER
# =============================================================================

def get_park_factors(home_team: str) -> dict:
    t = norm_team(home_team)
    return PARK_FACTORS.get(t, {"run": 1.00, "hr": 1.00})


def fetch_weather_for_game(home_team: str, game_dt_utc: Any) -> dict:
    t      = norm_team(home_team)
    coords = STADIUM_COORDS.get(t)
    default = {"temp_f": 72.0, "wind_mph": 7.0, "wind_dir": 0.0, "humidity": 50.0}

    if not coords:
        return default

    try:
        if pd.isna(game_dt_utc):
            game_time = datetime.now(timezone.utc)
        elif isinstance(game_dt_utc, pd.Timestamp):
            game_time = game_dt_utc.tz_localize("UTC").to_pydatetime() if game_dt_utc.tzinfo is None else game_dt_utc.tz_convert("UTC").to_pydatetime()
        elif isinstance(game_dt_utc, datetime):
            game_time = game_dt_utc.replace(tzinfo=timezone.utc) if game_dt_utc.tzinfo is None else game_dt_utc.astimezone(timezone.utc)
        else:
            parsed = pd.to_datetime(game_dt_utc, errors="coerce")
            game_time = datetime.now(timezone.utc) if pd.isna(parsed) else (parsed.tz_localize("UTC").to_pydatetime() if parsed.tzinfo is None else parsed.tz_convert("UTC").to_pydatetime())
    except Exception:
        game_time = datetime.now(timezone.utc)

    lat, lon = coords
    day = game_time.date().isoformat()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "UTC", "start_date": day, "end_date": day,
    }

    try:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data    = r.json()
        hourly  = data.get("hourly", {})
        times   = hourly.get("time", [])
        temps   = hourly.get("temperature_2m", [])
        humidity= hourly.get("relative_humidity_2m", [])
        wind    = hourly.get("wind_speed_10m", [])
        wind_dir= hourly.get("wind_direction_10m", [])

        if not times:
            return default

        target_hour = game_time.replace(minute=0, second=0, microsecond=0)
        best_i, best_delta = 0, None
        for i, ts in enumerate(times):
            dt    = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            delta = abs((dt - target_hour).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta, best_i = delta, i

        return {
            "temp_f":   safe_float(temps[best_i],    72.0),
            "wind_mph": safe_float(wind[best_i],      7.0),
            "wind_dir": safe_float(wind_dir[best_i],  0.0),
            "humidity": safe_float(humidity[best_i], 50.0),
        }
    except Exception:
        return default


# =============================================================================
# DB
# =============================================================================

def ensure_features_table(cur) -> None:
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {FEATURES_TABLE} (
            game_pk BIGINT PRIMARY KEY,
            official_date DATE NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_sp_fip DOUBLE PRECISION, away_sp_fip DOUBLE PRECISION,
            home_sp_xfip DOUBLE PRECISION, away_sp_xfip DOUBLE PRECISION,
            home_sp_siera DOUBLE PRECISION, away_sp_siera DOUBLE PRECISION,
            home_sp_kbb DOUBLE PRECISION, away_sp_kbb DOUBLE PRECISION,
            home_sp_era DOUBLE PRECISION, away_sp_era DOUBLE PRECISION,
            home_sp_whip DOUBLE PRECISION, away_sp_whip DOUBLE PRECISION,
            home_bullpen_era DOUBLE PRECISION, away_bullpen_era DOUBLE PRECISION,
            home_bullpen_fip DOUBLE PRECISION, away_bullpen_fip DOUBLE PRECISION,
            home_bullpen_xfip DOUBLE PRECISION, away_bullpen_xfip DOUBLE PRECISION,
            home_wrc_plus DOUBLE PRECISION, away_wrc_plus DOUBLE PRECISION,
            home_iso DOUBLE PRECISION, away_iso DOUBLE PRECISION,
            home_bb_pct DOUBLE PRECISION, away_bb_pct DOUBLE PRECISION,
            home_k_pct DOUBLE PRECISION, away_k_pct DOUBLE PRECISION,
            park_run_factor DOUBLE PRECISION, park_hr_factor DOUBLE PRECISION,
            temperature_f DOUBLE PRECISION, wind_speed_mph DOUBLE PRECISION,
            wind_direction_deg DOUBLE PRECISION, humidity_pct DOUBLE PRECISION,
            sp_fip_diff DOUBLE PRECISION, sp_xfip_diff DOUBLE PRECISION,
            bullpen_fip_diff DOUBLE PRECISION, offense_wrc_diff DOUBLE PRECISION,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{FEATURES_TABLE}_date ON {FEATURES_TABLE}(official_date);")


def load_games_for_window(cur, start_date: date, end_date: date) -> pd.DataFrame:
    cur.execute(f"""
        SELECT g.game_pk, g.official_date, g.game_date_utc,
               g.home_team, g.away_team,
               p.home_sp_id, p.away_sp_id,
               p.home_sp_name, p.away_sp_name
        FROM {GAMES_TABLE} g
        LEFT JOIN {PROB_TABLE} p ON p.game_pk = g.game_pk
        WHERE g.official_date BETWEEN %s AND %s
        ORDER BY g.official_date, g.game_pk
    """, (start_date, end_date))
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=[
        "game_pk","official_date","game_date_utc",
        "home_team","away_team",
        "home_sp_id","away_sp_id","home_sp_name","away_sp_name",
    ])


def build_feature_rows(games_df, pitcher_lookup, bullpen_lookup, offense_lookup):
    rows = []
    for _, g in games_df.iterrows():
        game_pk = g.get("game_pk")
        if pd.isna(game_pk):
            continue

        home_team    = norm_team(str(g.get("home_team", "")))
        away_team    = norm_team(str(g.get("away_team", "")))
        home_sp_name = "" if pd.isna(g.get("home_sp_name")) else str(g.get("home_sp_name"))
        away_sp_name = "" if pd.isna(g.get("away_sp_name")) else str(g.get("away_sp_name"))

        home_sp = pitcher_lookup.get(norm_name(home_sp_name), {})
        away_sp = pitcher_lookup.get(norm_name(away_sp_name), {})
        home_bp = bullpen_lookup.get(home_team, {})
        away_bp = bullpen_lookup.get(away_team, {})
        home_off= offense_lookup.get(home_team, {})
        away_off= offense_lookup.get(away_team, {})

        park    = get_park_factors(home_team)
        weather = fetch_weather_for_game(home_team, g.get("game_date_utc"))

        home_sp_fip  = safe_float(home_sp.get("fip",  4.20))
        away_sp_fip  = safe_float(away_sp.get("fip",  4.20))
        home_sp_era  = safe_float(home_sp.get("era",  4.20))
        away_sp_era  = safe_float(away_sp.get("era",  4.20))
        home_sp_whip = safe_float(home_sp.get("whip", 1.30))
        away_sp_whip = safe_float(away_sp.get("whip", 1.30))
        home_bp_fip  = safe_float(home_bp.get("fip",  4.20))
        away_bp_fip  = safe_float(away_bp.get("fip",  4.20))
        home_bp_era  = safe_float(home_bp.get("era",  4.20))
        away_bp_era  = safe_float(away_bp.get("era",  4.20))
        home_wrc     = safe_float(home_off.get("wrc_plus", 100.0), 100.0)
        away_wrc     = safe_float(away_off.get("wrc_plus", 100.0), 100.0)

        rows.append((
            int(game_pk),
            g.get("official_date"),
            home_team, away_team,
            home_sp_fip,  away_sp_fip,
            home_sp_fip,  away_sp_fip,   # xfip proxy = fip
            None, None,                   # siera — not available from BBRef
            None, None,                   # kbb  — not available from BBRef
            home_sp_era,  away_sp_era,
            home_sp_whip, away_sp_whip,
            home_bp_era,  away_bp_era,
            home_bp_fip,  away_bp_fip,
            home_bp_fip,  away_bp_fip,   # xfip proxy = fip
            home_wrc,     away_wrc,
            None, None,                   # iso — not in BBRef
            safe_float(home_off.get("bb_pct", 0.08)),
            safe_float(away_off.get("bb_pct", 0.08)),
            safe_float(home_off.get("k_pct",  0.22)),
            safe_float(away_off.get("k_pct",  0.22)),
            safe_float(park["run"], 1.0),
            safe_float(park["hr"],  1.0),
            safe_float(weather["temp_f"],   72.0),
            safe_float(weather["wind_mph"],  7.0),
            safe_float(weather["wind_dir"],  0.0),
            safe_float(weather["humidity"], 50.0),
            home_sp_fip - away_sp_fip,
            home_sp_fip - away_sp_fip,    # xfip diff = fip diff
            home_bp_fip - away_bp_fip,
            home_wrc    - away_wrc,
            datetime.now(timezone.utc),
        ))

    return rows


def upsert_feature_rows(cur, rows) -> int:
    if not rows:
        return 0

    sql = f"""
    INSERT INTO {FEATURES_TABLE} (
        game_pk, official_date, home_team, away_team,
        home_sp_fip, away_sp_fip, home_sp_xfip, away_sp_xfip,
        home_sp_siera, away_sp_siera, home_sp_kbb, away_sp_kbb,
        home_sp_era, away_sp_era, home_sp_whip, away_sp_whip,
        home_bullpen_era, away_bullpen_era, home_bullpen_fip, away_bullpen_fip,
        home_bullpen_xfip, away_bullpen_xfip,
        home_wrc_plus, away_wrc_plus, home_iso, away_iso,
        home_bb_pct, away_bb_pct, home_k_pct, away_k_pct,
        park_run_factor, park_hr_factor,
        temperature_f, wind_speed_mph, wind_direction_deg, humidity_pct,
        sp_fip_diff, sp_xfip_diff, bullpen_fip_diff, offense_wrc_diff,
        updated_at
    ) VALUES %s
    ON CONFLICT (game_pk) DO UPDATE SET
        home_sp_fip=EXCLUDED.home_sp_fip, away_sp_fip=EXCLUDED.away_sp_fip,
        home_sp_xfip=EXCLUDED.home_sp_xfip, away_sp_xfip=EXCLUDED.away_sp_xfip,
        home_sp_era=EXCLUDED.home_sp_era, away_sp_era=EXCLUDED.away_sp_era,
        home_sp_whip=EXCLUDED.home_sp_whip, away_sp_whip=EXCLUDED.away_sp_whip,
        home_bullpen_era=EXCLUDED.home_bullpen_era, away_bullpen_era=EXCLUDED.away_bullpen_era,
        home_bullpen_fip=EXCLUDED.home_bullpen_fip, away_bullpen_fip=EXCLUDED.away_bullpen_fip,
        home_wrc_plus=EXCLUDED.home_wrc_plus, away_wrc_plus=EXCLUDED.away_wrc_plus,
        home_bb_pct=EXCLUDED.home_bb_pct, away_bb_pct=EXCLUDED.away_bb_pct,
        home_k_pct=EXCLUDED.home_k_pct, away_k_pct=EXCLUDED.away_k_pct,
        park_run_factor=EXCLUDED.park_run_factor, park_hr_factor=EXCLUDED.park_hr_factor,
        temperature_f=EXCLUDED.temperature_f, wind_speed_mph=EXCLUDED.wind_speed_mph,
        wind_direction_deg=EXCLUDED.wind_direction_deg, humidity_pct=EXCLUDED.humidity_pct,
        sp_fip_diff=EXCLUDED.sp_fip_diff, bullpen_fip_diff=EXCLUDED.bullpen_fip_diff,
        offense_wrc_diff=EXCLUDED.offense_wrc_diff,
        updated_at=NOW();
    """
    execute_values(cur, sql, rows, page_size=500)
    return len(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    today    = datetime.now(timezone.utc).date()
    end_date = today + timedelta(days=LOOKAHEAD_DAYS)

    print("Building enriched game features (BBRef)...")
    print(f"Window: {today} -> {end_date}")

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

            print("Pulling BBRef pitching stats...")
            pitch_df = fetch_pitching_df(SEASON)

            print("Pulling BBRef batting stats...")
            bat_df = fetch_batting_df(SEASON)

            pitcher_lookup = build_pitcher_lookup(pitch_df)
            bullpen_lookup = build_bullpen_lookup(pitch_df)
            offense_lookup = build_team_offense_lookup(bat_df)

            # Show SP coverage
            covered = sum(
                1 for _, g in games_df.iterrows()
                if norm_name(str(g.get("home_sp_name", ""))) in pitcher_lookup
                or norm_name(str(g.get("away_sp_name", ""))) in pitcher_lookup
            )
            print(f"  SP coverage: {covered}/{len(games_df)} games have at least one SP in lookup")

            rows = build_feature_rows(games_df, pitcher_lookup, bullpen_lookup, offense_lookup)
            n    = upsert_feature_rows(cur, rows)
            c.commit()
            print(f"Feature rows upserted: {n}")

    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()