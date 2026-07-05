#!/usr/bin/env python3
"""
load_weather.py

Fetches hourly weather forecasts (or actuals) from Open-Meteo for each game's
stadium and stores the conditions at game-time in the game_weather table.

No API key required — Open-Meteo is fully free.

Usage:
    python scripts/load_weather.py                  # today's games
    python scripts/load_weather.py --date 2026-07-04
    python scripts/load_weather.py --days 3         # today + next 2 days
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if os.getenv("DYNO") is None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except Exception:
        pass

from app.db import engine

# ---------------------------------------------------------------------------
# Stadium coordinates + home-plate orientation
# orientation_deg = compass bearing that points FROM home plate TO center field.
# Wind blowing IN that direction = blowing IN (suppresses runs).
# Wind blowing FROM that direction = blowing OUT (adds runs).
# ---------------------------------------------------------------------------
STADIUMS: dict[str, dict] = {
    # team abbrev → {lat, lon, orientation_deg, name}
    "ARI": {"lat": 33.4453, "lon": -112.0667, "orientation_deg": 345, "name": "Chase Field"},
    "ATL": {"lat": 33.8908, "lon": -84.4678,  "orientation_deg":  60, "name": "Truist Park"},
    "BAL": {"lat": 39.2838, "lon": -76.6218,  "orientation_deg": 100, "name": "Oriole Park"},
    "BOS": {"lat": 42.3467, "lon": -71.0972,  "orientation_deg":  96, "name": "Fenway Park"},
    "CHC": {"lat": 41.9484, "lon": -87.6553,  "orientation_deg":  95, "name": "Wrigley Field"},
    "CWS": {"lat": 41.8300, "lon": -87.6339,  "orientation_deg": 137, "name": "Guaranteed Rate Field"},
    "CIN": {"lat": 39.0979, "lon": -84.5082,  "orientation_deg":  70, "name": "Great American Ball Park"},
    "CLE": {"lat": 41.4962, "lon": -81.6852,  "orientation_deg": 166, "name": "Progressive Field"},
    "COL": {"lat": 39.7559, "lon": -104.9942, "orientation_deg": 293, "name": "Coors Field"},
    "DET": {"lat": 42.3390, "lon": -83.0485,  "orientation_deg": 170, "name": "Comerica Park"},
    "HOU": {"lat": 29.7573, "lon": -95.3555,  "orientation_deg": 343, "name": "Minute Maid Park"},
    "KC":  {"lat": 39.0514, "lon": -94.4803,  "orientation_deg": 313, "name": "Kauffman Stadium"},
    "LAA": {"lat": 33.8003, "lon": -117.8827, "orientation_deg": 175, "name": "Angel Stadium"},
    "LAD": {"lat": 34.0739, "lon": -118.2400, "orientation_deg":  42, "name": "Dodger Stadium"},
    "MIA": {"lat": 25.7781, "lon": -80.2197,  "orientation_deg": 340, "name": "loanDepot park"},
    "MIL": {"lat": 43.0280, "lon": -87.9712,  "orientation_deg": 190, "name": "American Family Field"},
    "MIN": {"lat": 44.9817, "lon": -93.2778,  "orientation_deg": 358, "name": "Target Field"},
    "NYM": {"lat": 40.7571, "lon": -73.8458,  "orientation_deg":  15, "name": "Citi Field"},
    "NYY": {"lat": 40.8296, "lon": -73.9262,  "orientation_deg":  30, "name": "Yankee Stadium"},
    "OAK": {"lat": 37.7516, "lon": -122.2005, "orientation_deg": 330, "name": "Oakland Coliseum"},
    "ATH": {"lat": 36.1699, "lon": -115.1398, "orientation_deg": 340, "name": "Las Vegas Ballpark"},  # Athletics in Vegas
    "PHI": {"lat": 39.9061, "lon": -75.1665,  "orientation_deg": 330, "name": "Citizens Bank Park"},
    "PIT": {"lat": 40.4469, "lon": -80.0058,  "orientation_deg": 358, "name": "PNC Park"},
    "SD":  {"lat": 32.7073, "lon": -117.1566, "orientation_deg": 305, "name": "Petco Park"},
    "SF":  {"lat": 37.7786, "lon": -122.3893, "orientation_deg":  35, "name": "Oracle Park"},
    "SEA": {"lat": 47.5914, "lon": -122.3325, "orientation_deg":  30, "name": "T-Mobile Park"},
    "STL": {"lat": 38.6226, "lon": -90.1928,  "orientation_deg": 270, "name": "Busch Stadium"},
    "TB":  {"lat": 27.7683, "lon": -82.6534,  "orientation_deg": 325, "name": "Tropicana Field"},
    "TEX": {"lat": 32.7473, "lon": -97.0822,  "orientation_deg":  36, "name": "Globe Life Field"},
    "TOR": {"lat": 43.6414, "lon": -79.3894,  "orientation_deg":  15, "name": "Rogers Centre"},
    "WSH": {"lat": 38.8730, "lon": -77.0074,  "orientation_deg":  75, "name": "Nationals Park"},
}

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE  = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_probability",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "visibility",
    "cloud_cover",
    "weather_code",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angular_diff(a: float, b: float) -> float:
    """Signed angular difference a - b, wrapped to [-180, 180]."""
    d = (a - b) % 360
    return d - 360 if d > 180 else d


def wind_component(wind_dir_deg: float, orientation_deg: float) -> float:
    """
    Returns a value in [-1, 1]:
      +1  = wind blowing straight OUT (from CF toward home plate)
      -1  = wind blowing straight IN  (from home plate toward CF)
       0  = crosswind

    wind_dir_deg = direction wind is coming FROM (meteorological convention)
    orientation_deg = bearing from home plate to center field
    """
    # Direction wind is going TO:
    wind_going_to = (wind_dir_deg + 180) % 360
    # How aligned is that with the CF direction?
    diff = _angular_diff(wind_going_to, orientation_deg)
    return math.cos(math.radians(diff))


def ensure_table() -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS game_weather (
                game_pk          BIGINT PRIMARY KEY,
                fetched_at       TIMESTAMPTZ DEFAULT NOW(),
                stadium_name     TEXT,
                game_time_utc    TIMESTAMPTZ,
                temp_f           FLOAT,
                humidity_pct     FLOAT,
                precip_prob      FLOAT,
                precip_mm        FLOAT,
                wind_speed_mph   FLOAT,
                wind_dir_deg     FLOAT,
                wind_out_factor  FLOAT,
                visibility_m     FLOAT,
                cloud_cover_pct  FLOAT,
                weather_code     INT,
                is_dome          BOOLEAN DEFAULT FALSE
            )
        """))


# Domed/retractable-roof stadiums where weather doesn't matter
DOME_TEAMS = {"ARI", "HOU", "MIA", "MIL", "MIN", "SEA", "TB", "TOR", "TEX"}


def fetch_weather_for_game(
    game_pk: int,
    home_team: str,
    game_time_utc: datetime,
) -> Optional[dict]:
    """Fetch weather for a single game from Open-Meteo."""
    stadium = STADIUMS.get(home_team)
    if stadium is None:
        print(f"  ⚠️  No stadium data for {home_team} (game {game_pk})")
        return None

    is_dome = home_team in DOME_TEAMS
    target_dt = game_time_utc
    target_date_str = target_dt.date().isoformat()

    # Use archive API for past dates, forecast for future/today
    today = date.today()
    game_date = target_dt.date()

    if game_date < today:
        base_url = OPEN_METEO_ARCHIVE
    else:
        base_url = OPEN_METEO_FORECAST

    params = {
        "latitude":  stadium["lat"],
        "longitude": stadium["lon"],
        "hourly":    ",".join(HOURLY_VARS),
        "start_date": target_date_str,
        "end_date":   target_date_str,
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    }
    # Archive API doesn't support precipitation_probability
    if game_date < today:
        params["hourly"] = ",".join(
            v for v in HOURLY_VARS if v != "precipitation_probability"
        )

    try:
        r = requests.get(base_url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  ✗ Open-Meteo error for {home_team} game {game_pk}: {e}")
        return None

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])

    # Find the hour closest to game time
    game_hour_str = target_dt.strftime("%Y-%m-%dT%H:00")
    if game_hour_str in times:
        idx = times.index(game_hour_str)
    elif times:
        # Fallback: find closest hour
        game_ts = target_dt.replace(minute=0, second=0, microsecond=0)
        closest = min(
            range(len(times)),
            key=lambda i: abs(
                datetime.fromisoformat(times[i]).replace(tzinfo=timezone.utc) - game_ts
            ),
        )
        idx = closest
    else:
        print(f"  ✗ No hourly data returned for {home_team} game {game_pk}")
        return None

    def get_val(key: str) -> Optional[float]:
        vals = hourly.get(key, [])
        v = vals[idx] if idx < len(vals) else None
        return float(v) if v is not None else None

    wind_speed = get_val("wind_speed_10m")
    wind_dir   = get_val("wind_direction_10m")
    orientation = stadium["orientation_deg"]

    # wind_out_factor: +1 blowing out, -1 blowing in, 0 crosswind
    if wind_speed is not None and wind_dir is not None:
        wof = wind_component(wind_dir, orientation) * (wind_speed / 10.0)
        # Scale: 10 mph straight out → +1.0, 20 mph out → +2.0, etc.
    else:
        wof = None

    return {
        "game_pk":         game_pk,
        "stadium_name":    stadium["name"],
        "game_time_utc":   target_dt,
        "temp_f":          get_val("temperature_2m"),
        "humidity_pct":    get_val("relative_humidity_2m"),
        "precip_prob":     get_val("precipitation_probability"),
        "precip_mm":       get_val("precipitation"),
        "wind_speed_mph":  wind_speed,
        "wind_dir_deg":    wind_dir,
        "wind_out_factor": wof,
        "visibility_m":    get_val("visibility"),
        "cloud_cover_pct": get_val("cloud_cover"),
        "weather_code":    int(get_val("weather_code") or 0) or None,
        "is_dome":         is_dome,
    }


def upsert_weather(row: dict) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO game_weather (
                game_pk, stadium_name, game_time_utc,
                temp_f, humidity_pct, precip_prob, precip_mm,
                wind_speed_mph, wind_dir_deg, wind_out_factor,
                visibility_m, cloud_cover_pct, weather_code, is_dome,
                fetched_at
            ) VALUES (
                :game_pk, :stadium_name, :game_time_utc,
                :temp_f, :humidity_pct, :precip_prob, :precip_mm,
                :wind_speed_mph, :wind_dir_deg, :wind_out_factor,
                :visibility_m, :cloud_cover_pct, :weather_code, :is_dome,
                NOW()
            )
            ON CONFLICT (game_pk) DO UPDATE SET
                stadium_name     = EXCLUDED.stadium_name,
                game_time_utc    = EXCLUDED.game_time_utc,
                temp_f           = EXCLUDED.temp_f,
                humidity_pct     = EXCLUDED.humidity_pct,
                precip_prob      = EXCLUDED.precip_prob,
                precip_mm        = EXCLUDED.precip_mm,
                wind_speed_mph   = EXCLUDED.wind_speed_mph,
                wind_dir_deg     = EXCLUDED.wind_dir_deg,
                wind_out_factor  = EXCLUDED.wind_out_factor,
                visibility_m     = EXCLUDED.visibility_m,
                cloud_cover_pct  = EXCLUDED.cloud_cover_pct,
                weather_code     = EXCLUDED.weather_code,
                is_dome          = EXCLUDED.is_dome,
                fetched_at       = NOW()
        """), row)


def load_weather_for_date(target_date: date) -> None:
    ensure_table()

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT game_pk, home_team, game_date_utc
            FROM games
            WHERE official_date = :d
              AND status NOT IN ('Final', 'Cancelled', 'Postponed')
               OR (official_date = :d AND game_date_utc IS NOT NULL)
            ORDER BY game_date_utc
        """), {"d": target_date}).fetchall()

    if not rows:
        # Fallback: fetch all games for date including completed ones
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT game_pk, home_team, game_date_utc
                FROM games
                WHERE official_date = :d
                  AND game_date_utc IS NOT NULL
                ORDER BY game_date_utc
            """), {"d": target_date}).fetchall()

    if not rows:
        print(f"No games found for {target_date}.")
        return

    print(f"Fetching weather for {len(rows)} games on {target_date}...")

    ok = 0
    for row in rows:
        game_pk   = row.game_pk
        home_team = row.home_team
        game_dt   = row.game_date_utc

        if game_dt is None:
            print(f"  ⚠️  game {game_pk} has no game_date_utc — skipping")
            continue

        # Ensure tz-aware
        if game_dt.tzinfo is None:
            game_dt = game_dt.replace(tzinfo=timezone.utc)

        weather = fetch_weather_for_game(game_pk, home_team, game_dt)
        if weather is None:
            continue

        upsert_weather(weather)

        dome_tag = " [DOME]" if weather["is_dome"] else ""
        wof = weather["wind_out_factor"]
        wof_str = f"{wof:+.1f}" if wof is not None else "n/a"
        vis = weather["visibility_m"]
        vis_str = f"{vis/1000:.1f}km" if vis is not None else "n/a"
        print(
            f"  ✓ {home_team:3s} | {weather['stadium_name']}{dome_tag} | "
            f"{weather['temp_f']:.0f}°F | "
            f"wind {weather['wind_speed_mph']:.0f}mph out_factor={wof_str} | "
            f"precip={weather['precip_prob'] or 0:.0f}% | "
            f"vis={vis_str}"
        )
        ok += 1

    print(f"\nWeather loaded for {ok}/{len(rows)} games.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--days", type=int, default=1, help="Number of days starting from --date")
    args = parser.parse_args()

    start = date.fromisoformat(args.date) if args.date else date.today()
    for i in range(args.days):
        d = start + timedelta(days=i)
        print(f"\n{'='*55}")
        print(f"load_weather.py | {d}")
        print(f"{'='*55}")
        load_weather_for_date(d)
