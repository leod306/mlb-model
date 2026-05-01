"""
mlb_quick_update.py
-------------------
Noon refresh — re-runs engine after lineups are posted.
Run manually or schedule at ~11:30am ET daily.

Run order:
  1. Load RotoWire Lineups  — projected lineups (early signal)
  2. Load Official Lineups  — official MLB lineups (overwrites RotoWire when available)
  3. Load Odds              — refresh Vegas lines
  4. Run MLB Engine         — re-predict with real lineups + BvP matchup scores
  5. Daily Picks Tracker    — overwrite today's picks with updated predictions
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
    # RotoWire lineups first — non-critical, don't stop if it fails
    run_step("Load RotoWire Lineups",  "load_rotowire_lineups.py", required=False)
    # Official MLB lineups overwrite RotoWire where available
    run_step("Load Official Lineups",  "load_lineups.py",          required=False)
    run_step("Load Odds",              "load_odds.py")
    run_step("Run MLB Engine",         "mlb_engine_daily.py")
    run_step("Daily Picks Tracker",    "daily_picks_tracker.py")
    print("\n✅ Quick update complete.")


if __name__ == "__main__":
    main()