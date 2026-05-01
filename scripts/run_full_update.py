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
  2. Load Probable Starters — today's probable SPs
  3. Load Pitcher Game Logs — rest days + bullpen IP
  4. Load RotoWire Lineups  — projected lineups (early signal, non-critical)
  5. Load Official Lineups  — official MLB lineups (overwrites RotoWire)
  6. Load Odds              — Vegas lines (must be BEFORE engine)
  7. Run MLB Engine         — generates predictions using all of the above
  8. Daily Picks Tracker    — evaluates yesterday + saves today's picks
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent


def run_step(name: str, script_name: str, required: bool = True) -> None:
    print(f"\n=== Running: {name} ===")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name)],
        cwd=PROJECT_ROOT,
        text=True,
    )
    if result.returncode != 0:
        if required:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}")
        else:
            print(f"⚠️  {name} failed (non-critical) — continuing...")
    else:
        print(f"=== Finished: {name} ===")


def main():
    run_step("Load Schedule",          "load_2026_schedule.py")
    run_step("Load Probable Starters", "load_probable_starters.py")
    run_step("Load Pitcher Game Logs", "load_pitcher_game_log.py")
    run_step("Load RotoWire Lineups",  "load_rotowire_lineups.py", required=False)  # early lineup signal
    run_step("Load Official Lineups",  "load_lineups.py",          required=False)  # overwrites RotoWire
    run_step("Load Odds",              "load_odds.py")             # ← must be before engine
    run_step("Run MLB Engine",         "mlb_engine_daily.py")      # ← uses Vegas lines + lineups
    run_step("Daily Picks Tracker",    "daily_picks_tracker.py")   # ← always last
    print("\n✅ Full update complete.")


if __name__ == "__main__":
    main()