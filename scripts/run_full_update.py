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
  3. Build Team Features    — FIP, wRC+, park, weather (needs SP names from step 2)
  4. Load Pitcher Game Logs — rest days + bullpen IP
  5. Load RotoWire Lineups  — projected lineups (early signal, non-critical)
  6. Load Official Lineups  — official MLB lineups (overwrites RotoWire)
  7. Load Odds              — Vegas lines (must be BEFORE engine)
  8. Run MLB Engine         — generates predictions using all of the above
  9. Daily Picks Tracker    — evaluates yesterday + saves today's picks

NOTE on build_team_features.py timing:
  - Fetches FanGraphs pitching + batting via pybaseball (~30-60s)
  - Hits Open-Meteo weather API once per game (~15 calls on a full slate)
  - Total: 2-4 min on a normal day; can be longer if FanGraphs is slow
  - marked required=False so a FanGraphs outage won't kill the full pipeline
    but the engine will fall back to default values for FIP/wRC+/park/weather
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


def run_step(name: str, script_name: str, required: bool = True) -> float:
    """Run a script step. Returns elapsed seconds."""
    print(f"\n=== Running: {name} ===")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name)],
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

    # ── 3. Team Features (FIP / wRC+ / park / weather) ───────────────────────
    # Depends on: game_probables (step 2) for SP name → FIP lookup
    # Populates: game_features table → sp_fip_diff, bullpen_fip_diff,
    #            offense_wrc_diff, park_run_factor, temperature_f, wind_speed_mph
    # Without this step those 6 features default to 0/league-avg every day.
    # Non-critical: FanGraphs outages shouldn't kill the full pipeline.
    t = run_step("Build Team Features",    "build_team_features.py", required=False)
    timings.append(("Build Team Features", t))

    # ── 4. Pitcher Game Logs ─────────────────────────────────────────────────
    # Needed for: rest days, bullpen IP, live ERA/WHIP from recent starts
    t = run_step("Load Pitcher Game Logs", "load_pitcher_game_log.py")
    timings.append(("Load Pitcher Game Logs", t))

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

    # ── 8. MLB Engine ────────────────────────────────────────────────────────
    # Uses everything above: game_features + lineups + BvP + odds + form
    t = run_step("Run MLB Engine",         "mlb_engine_daily.py")
    timings.append(("Run MLB Engine", t))

    # ── 9. Daily Picks Tracker ───────────────────────────────────────────────
    # Always last — evaluates yesterday's picks against actual scores
    t = run_step("Daily Picks Tracker",    "daily_picks_tracker.py")
    timings.append(("Daily Picks Tracker", t))

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