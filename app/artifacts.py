import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml")

RUN_MODEL_PATH = os.path.join(ML_DIR, "run_model.pkl")
TOTAL_MODEL_PATH = os.path.join(ML_DIR, "total_model.pkl")
WIN_MODEL_PATH = os.path.join(ML_DIR, "win_model.pkl")


def _safe_load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return joblib.load(path)


run_model = _safe_load(RUN_MODEL_PATH)
total_model = _safe_load(TOTAL_MODEL_PATH)
win_model = _safe_load(WIN_MODEL_PATH)