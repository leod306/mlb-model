import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(name: str, script_name: str) -> None:
    print(f"\n=== Running: {name} ===")
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=PROJECT_ROOT,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    print(f"=== Finished: {name} ===")


def main():
    run_step("Load Schedule",          "load_2026_schedule.py")
    run_step("Load Probable Starters", "load_probable_starters.py")
    run_step("Load Pitcher Game Logs", "load_pitcher_game_logs.py")
    run_step("Load Lineups",           "load_lineups.py")
    run_step("Load Odds",              "load_odds.py")
    run_step("Build Features",         "build_team_features.py")
    run_step("Run MLB Engine",         "mlb_engine_daily.py")
    run_step("Daily Picks Tracker",    "daily_picks_tracker.py")
    print("\n✅ Full update complete.")


if __name__ == "__main__":
    main()