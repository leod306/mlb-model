import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(name: str, script_path: str) -> None:
    print(f"\n=== Running: {name} ===")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    print(f"=== Finished: {name} ===")


def main():
    run_step("Load Probable Starters",  "load_probable_starters.py")
    run_step("Load Lineups",            "load_lineups.py")
    run_step("Load Odds / Predict",     "load_odds.py")
    print("\n✅ Quick update complete.")


if __name__ == "__main__":
    main()