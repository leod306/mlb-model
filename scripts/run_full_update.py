import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(name: str, script_path: str) -> None:
    print(f"\n=== Running: {name} ===")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    print(f"=== Finished: {name} ===")


def main():
    run_step("Load Schedule", "scripts/load_2026_schedule.py")
    run_step("Load Probable Starters", "scripts/load_probable_starters.py")
    run_step("Load Odds", "scripts/load_odds.py")
    run_step("Build Features", "scripts/build_team_features.py")
    run_step("Run MLB Engine", "scripts/mlb_engine_daily.py")
    print("\n✅ Full update complete.")


if __name__ == "__main__":
    main()