"""
run_full_update.py
------------------
Full morning update — runs all scripts in correct order.
Schedule this at ~8am daily via Heroku Scheduler.

This file lives in scripts/ so:
  __file__ = /path/to/project/scripts/run_full_update.py
  SCRIPTS_DIR = /path/to/project/scripts/
  PROJECT_ROOT = /path/to/project/

Run order:
  1. Load Schedule          — saves yesterday's scores, loads today's schedule
  2. Load Probable Starters — today's probable SPs  ← must be before features
  3. Load Pitcher Game Logs — rest days + bullpen IP + FIP histories
  4. Build Team Features    — as-of-date FIP, bullpen FIP, offense, park
                              (from our own pitcher_game_log; no scraping)
  5. Load RotoWire Lineups  — projected lineups (early signal, non-critical)
  6. Load Official Lineups  — official MLB lineups (overwrites RotoWire)
  7. Load Odds              — Vegas lines (must be BEFORE engine)
 7b. Load Weather           — game-time temp/wind/rain from Open-Meteo (free)
  8. Run MLB Engine         — generates predictions using all of the above
  9. Daily Picks Tracker    — evaluates yesterday + saves today's picks
 10. Load Player Props      — fetches hits/TB/HR/K/BB edges from Odds API

NOTE on Build Team Features (step 4):
  - Now runs backfill_pitching_offense.py for today's slate only.
  - Computes as-of-date FIP / WHIP / bullpen FIP / offense proxy / park
    entirely from pitcher_game_log + games — no FanGraphs, no BBRef, no
    name-matching, no weather API. Runs in seconds and can't be killed by
    an external outage, so it's required=True.
  - This is the SAME code that builds training_data.csv, so the features the
    model trains on and predicts on are identical (no train/serve skew).
  - Pitcher game logs (step 3) now load BEFORE features so today's FIP
    histories include the most recent starts.
  - Weather (temp_f / wind_speed_mph / wind_out_factor / precip_prob / visibility)
    is fetched in step 7b via load_weather.py (Open-Meteo, no API key needed).
    The engine joins game_weather before prediction and stores it in predictions.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

# Ensure app module is importable from any working directory
ENV = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}


def run_step(name: str, script_name: str, required: bool = True,
             args: "list[str] | None" = None) -> float:
    """Run a script step. Returns elapsed seconds."""
    print(f"\n=== Running: {name} ===")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), *(args or [])],
        cwd=PROJECT_ROOT,
        env=ENV,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        if required:
            raise RuntimeError(
                f"{name} failed with exit code {result.returncode} "
                f"(after {elapsed:.1f}s)"
            )
        else:
            print(f"⚠️  {name} failed (non-critical, {elapsed:.1f}s) — continuing...")
    else:
        print(f"=== Finished: {name} ({elapsed:.1f}s) ===")

    return elapsed


def main():
    total_start = time.time()
    timings: list[tuple[str, float]] = []

    # ── 1. Schedule ──────────────────────────────────────────────────────────
    t = run_step("Load Schedule",          "load_2026_schedule.py")
    timings.append(("Load Schedule", t))

    # ── 2. Probable Starters ─────────────────────────────────────────────────
    # Must run BEFORE build_team_features so SP names are in game_probables
    t = run_step("Load Probable Starters", "load_probable_starters.py")
    timings.append(("Load Probable Starters", t))

    # ── 3. Pitcher Game Logs ─────────────────────────────────────────────────
    # Load BEFORE features so today's FIP histories include the latest starts.
    # Also provides rest days + bullpen IP.
    t = run_step("Load Pitcher Game Logs", "load_pitcher_game_log.py")
    timings.append(("Load Pitcher Game Logs", t))

    # ── 4. Team Features (as-of-date FIP / bullpen FIP / offense / park) ──────
    # Runs backfill_pitching_offense.py for TODAY's slate only. Computes
    # everything from our own pitcher_game_log + games — no FanGraphs/BBRef,
    # no name-matching, no weather API. Populates game_features:
    #   sp_fip_diff, bullpen_fip_diff, offense_wrc_diff, park_run_factor, ...
    # Same code as build_dataset.py → zero train/serve skew.
    # required=True: it's pure DB now, so it's reliable and the totals model
    # depends on it.
    t = run_step("Build Team Features",    "backfill_pitching_offense.py",
                 required=True, args=["today"])
    timings.append(("Build Team Features", t))

    # ── 5. RotoWire Lineups ──────────────────────────────────────────────────
    # Early lineup signal — available before official lineups drop (~10am)
    t = run_step("Load RotoWire Lineups",  "load_rotowire_lineups.py", required=False)
    timings.append(("Load RotoWire Lineups", t))

    # ── 6. Official Lineups ──────────────────────────────────────────────────
    # Overwrites RotoWire rows when official lineups are confirmed (~3-4pm)
    t = run_step("Load Official Lineups",  "load_lineups.py",          required=False)
    timings.append(("Load Official Lineups", t))

    # ── 7. Odds ──────────────────────────────────────────────────────────────
    # Must run BEFORE engine so market lines feed into pick logic
    t = run_step("Load Odds",              "load_odds.py")
    timings.append(("Load Odds", t))

    # ── 7b. Weather ──────────────────────────────────────────────────────────
    # Fetch game-time weather from Open-Meteo (free, no API key).
    # Non-critical — engine runs fine without it.
    t = run_step("Load Weather",           "load_weather.py",             required=False)
    timings.append(("Load Weather", t))

    # ── 8. MLB Engine ────────────────────────────────────────────────────────
    # Uses everything above: game_features + lineups + BvP + odds + weather + form
    t = run_step("Run MLB Engine",         "mlb_engine_daily.py")
    timings.append(("Run MLB Engine", t))

    # ── 9. Daily Picks Tracker ───────────────────────────────────────────────
    # Always last — evaluates yesterday's picks against actual scores
    t = run_step("Daily Picks Tracker",    "daily_picks_tracker.py")
    timings.append(("Daily Picks Tracker", t))

    # ── 10. Player Props ─────────────────────────────────────────────────────
    # Fetches hits/TB/HR/K/BB props from Odds API and scores edges
    # Depends on: lineups (step 6), pitcher game logs (step 4)
    t = run_step("Load Player Props",      "load_player_props.py",     required=False)
    timings.append(("Load Player Props", t))

    # ── Summary ──────────────────────────────────────────────────────────────
    total = time.time() - total_start
    print("\n" + "="*50)
    print("✅ Full update complete.")
    print(f"   Total time: {total:.1f}s ({total/60:.1f} min)")
    print("\n   Step timings:")
    for name, secs in timings:
        bar = "█" * int(secs / 5)  # 1 block per 5 seconds
        print(f"   {name:<28} {secs:>6.1f}s  {bar}")
    print("="*50)


if __name__ == "__main__":
    main()