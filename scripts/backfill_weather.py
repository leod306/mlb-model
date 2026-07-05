#!/usr/bin/env python3
"""
backfill_weather.py

Backfills game_weather for all historical games using Open-Meteo archive API.
Skips games already in game_weather. Groups by date to minimize API calls
(one call per stadium per date instead of one per game).

Usage:
    python scripts/backfill_weather.py            # all missing games
    python scripts/backfill_weather.py --season 2025
    python scripts/backfill_weather.py --from-date 2024-04-01
    python scripts/backfill_weather.py --dry-run   # show what would be fetched
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
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

# Re-use stadium data and helpers from load_weather
from scripts.load_weather import (
    STADIUMS, DOME_TEAMS, HOURLY_VARS,
    wind_component, ensure_table, upsert_weather,
)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
RATE_LIMIT_SLEEP   = 0.25   # seconds between API calls (Open-Meteo is generous)
BATCH_SIZE         = 50     # games per progress print


def get_missing_games(from_date: Optional[date] = None, season: Optional[int] = None) -> list:
    """
    Returns all games that have game_date_utc but no entry in game_weather yet,
    ordered by date ascending.
    """
    conditions = ["g.game_date_utc IS NOT NULL", "g.home_score IS NOT NULL"]
    params: dict = {}

    if season:
        conditions.append("g.season = :season")
        params["season"] = season
    elif from_date:
        conditions.append("CAST(g.official_date AS DATE) >= :from_date")
        params["from_date"] = from_date

    where = " AND ".join(conditions)

    sql = text(f"""
        SELECT g.game_pk, g.home_team, g.game_date_utc, g.official_date
        FROM games g
        LEFT JOIN game_weather w ON w.game_pk = g.game_pk
        WHERE {where}
          AND w.game_pk IS NULL
        ORDER BY g.official_date ASC, g.game_pk ASC
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    return rows


def fetch_archive_weather_for_date(
    lat: float,
    lon: float,
    target_date: str,       # YYYY-MM-DD
    is_dome: bool,
) -> Optional[dict]:
    """
    Fetch all hourly data for one stadium on one date from Open-Meteo archive.
    Returns the full hourly dict keyed by hour string, or None on error.
    """
    if is_dome:
        return {}   # dome — no real weather needed

    archive_vars = [v for v in HOURLY_VARS if v != "precipitation_probability"]
    params = {
        "latitude":          lat,
        "longitude":         lon,
        "hourly":            ",".join(archive_vars),
        "start_date":        target_date,
        "end_date":          target_date,
        "wind_speed_unit":   "mph",
        "temperature_unit":  "fahrenheit",
        "timezone":          "UTC",
    }

    try:
        r = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    ⚠️  Archive fetch failed ({lat},{lon} on {target_date}): {e}")
        return None

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])

    # Build a lookup: hour_string → {var: value}
    result = {}
    for i, t in enumerate(times):
        result[t] = {
            var: (hourly[var][i] if i < len(hourly.get(var, [])) else None)
            for var in archive_vars
        }
    return result


def extract_hour_weather(
    hourly_data: dict,
    game_dt: datetime,
) -> dict:
    """Pick the closest hour from the fetched hourly data."""
    game_hour_str = game_dt.strftime("%Y-%m-%dT%H:00")

    if game_hour_str in hourly_data:
        return hourly_data[game_hour_str]

    # Fallback: nearest hour
    if not hourly_data:
        return {}

    def ts(s: str) -> datetime:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

    closest = min(hourly_data.keys(), key=lambda s: abs(ts(s) - game_dt))
    return hourly_data[closest]


def build_weather_row(
    game_pk: int,
    home_team: str,
    game_dt: datetime,
    hour_vals: dict,
    is_dome: bool,
) -> dict:
    stadium = STADIUMS.get(home_team, {})
    orientation = stadium.get("orientation_deg", 0)
    stadium_name = stadium.get("name", home_team)

    def gv(key: str) -> Optional[float]:
        v = hour_vals.get(key)
        return float(v) if v is not None else None

    wind_speed = gv("wind_speed_10m")
    wind_dir   = gv("wind_direction_10m")

    if wind_speed is not None and wind_dir is not None and not is_dome:
        wof = wind_component(wind_dir, orientation) * (wind_speed / 10.0)
    else:
        wof = None

    return {
        "game_pk":         game_pk,
        "stadium_name":    stadium_name,
        "game_time_utc":   game_dt,
        "temp_f":          gv("temperature_2m"),
        "humidity_pct":    gv("relative_humidity_2m"),
        "precip_prob":     None,   # archive doesn't have precip_probability
        "precip_mm":       gv("precipitation"),
        "wind_speed_mph":  wind_speed,
        "wind_dir_deg":    wind_dir,
        "wind_out_factor": wof,
        "visibility_m":    gv("visibility"),
        "cloud_cover_pct": gv("cloud_cover"),
        "weather_code":    int(gv("weather_code") or 0) or None,
        "is_dome":         is_dome,
    }


def backfill(
    from_date: Optional[date] = None,
    season: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    ensure_table()

    rows = get_missing_games(from_date=from_date, season=season)
    if not rows:
        print("✅ No missing games — game_weather is fully backfilled.")
        return

    print(f"Found {len(rows)} games to backfill.")
    if dry_run:
        for r in rows[:20]:
            print(f"  {r.official_date} | {r.home_team} | game_pk={r.game_pk}")
        if len(rows) > 20:
            print(f"  ... and {len(rows) - 20} more")
        return

    # Group by (date, home_team) to batch API calls
    # Each unique (date, home_team) → one archive API call for all games that day
    by_stadium_date: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (str(r.official_date), r.home_team)
        by_stadium_date[key].append(r)

    total_api_calls = len(by_stadium_date)
    print(f"API calls needed: {total_api_calls} (one per stadium-date)")
    print()

    ok = skip = err = 0
    call_num = 0

    for (date_str, home_team), games_on_day in sorted(by_stadium_date.items()):
        call_num += 1
        is_dome = home_team in DOME_TEAMS
        stadium = STADIUMS.get(home_team)

        if stadium is None:
            print(f"  ⚠️  No stadium for {home_team} — skipping {len(games_on_day)} game(s)")
            skip += len(games_on_day)
            continue

        if call_num % 20 == 1:
            print(f"  [{call_num}/{total_api_calls}] {date_str} {home_team}...")

        if is_dome:
            hourly_data = {}
        else:
            hourly_data = fetch_archive_weather_for_date(
                stadium["lat"], stadium["lon"], date_str, is_dome
            )
            if hourly_data is None:
                err += len(games_on_day)
                continue
            time.sleep(RATE_LIMIT_SLEEP)

        for row in games_on_day:
            game_dt = row.game_date_utc
            if game_dt.tzinfo is None:
                game_dt = game_dt.replace(tzinfo=timezone.utc)

            if is_dome:
                hour_vals = {}
            else:
                hour_vals = extract_hour_weather(hourly_data, game_dt)

            weather_row = build_weather_row(
                game_pk=row.game_pk,
                home_team=home_team,
                game_dt=game_dt,
                hour_vals=hour_vals,
                is_dome=is_dome,
            )
            try:
                upsert_weather(weather_row)
                ok += 1
            except Exception as e:
                print(f"    ✗ DB error for game {row.game_pk}: {e}")
                err += 1

    print(f"\n{'='*50}")
    print(f"Backfill complete: {ok} loaded, {skip} skipped (no stadium), {err} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical game weather")
    parser.add_argument("--season",    type=int, default=None, help="Season year (e.g. 2025)")
    parser.add_argument("--from-date", default=None,           help="Start date YYYY-MM-DD")
    parser.add_argument("--dry-run",   action="store_true",    help="Show what would be fetched without doing it")
    args = parser.parse_args()

    from_date = date.fromisoformat(args.from_date) if args.from_date else None

    print("="*50)
    print("backfill_weather.py")
    if args.season:
        print(f"Season: {args.season}")
    elif from_date:
        print(f"From: {from_date}")
    else:
        print("All missing games")
    print("="*50)

    backfill(from_date=from_date, season=args.season, dry_run=args.dry_run)
