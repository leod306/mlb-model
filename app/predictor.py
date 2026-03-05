# app/predictor.py
"""
MLB Predictor module

This file provides a stable import for:
    from app.predictor import predict_from_features

It supports:
- Loading a scikit-learn style model (joblib pickle)
- Optional preprocessing pipeline (if you saved a full Pipeline as the model, that's fine too)
- Predicting from a dict, list[dict], or pandas DataFrame
- Returning predictions + probabilities (when available)

ENV options:
- MLB_MODEL_PATH (default: models/mlb_model.pkl)
- MLB_POSITIVE_CLASS (default: 1)  # which class probability to return if model has classes_
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import pandas as pd


# -----------------------------
# Config
# -----------------------------
MODEL_PATH = os.getenv("MLB_MODEL_PATH", os.path.join("models", "mlb_model.pkl"))
POSITIVE_CLASS = int(os.getenv("MLB_POSITIVE_CLASS", "1"))


# -----------------------------
# Types
# -----------------------------
FeaturesInput = Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]


@dataclass
class PredictionResult:
    """
    A single-game prediction result.
    """
    prediction: Any
    proba: Optional[float]
    details: Dict[str, Any]


# -----------------------------
# Model loading (cached)
# -----------------------------
_MODEL: Any = None


def _load_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Set MLB_MODEL_PATH or place your model there."
        )

    _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


# -----------------------------
# Utilities
# -----------------------------
def _to_dataframe(features: FeaturesInput) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.copy()

    if isinstance(features, dict):
        return pd.DataFrame([features])

    if isinstance(features, list):
        if not features:
            return pd.DataFrame([])
        if not isinstance(features[0], dict):
            raise TypeError("If passing a list, it must be a list of dicts.")
        return pd.DataFrame(features)

    raise TypeError("features must be a dict, list[dict], or pandas DataFrame.")


def _get_positive_class_index(model: Any) -> Optional[int]:
    """
    If model has classes_ (sklearn classifiers), return the column index for POSITIVE_CLASS.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None

    try:
        classes_list = list(classes)
        return classes_list.index(POSITIVE_CLASS)
    except Exception:
        # If POSITIVE_CLASS isn't present, default to last column
        return len(classes) - 1


def _safe_predict_proba(model: Any, X: pd.DataFrame) -> Optional[List[float]]:
    """
    Return probability of the "positive" class if available.
    """
    if not hasattr(model, "predict_proba"):
        return None

    proba = model.predict_proba(X)
    if proba is None:
        return None

    # proba expected shape: (n, n_classes)
    idx = _get_positive_class_index(model)
    if idx is None:
        # Unexpected for predict_proba, but safe
        return None

    return [float(row[idx]) for row in proba]


def _safe_predict(model: Any, X: pd.DataFrame) -> List[Any]:
    """
    Always try predict(); raise a clear error if missing.
    """
    if not hasattr(model, "predict"):
        raise AttributeError("Loaded model does not have a .predict() method.")
    preds = model.predict(X)
    return list(preds)


# -----------------------------
# Public API
# -----------------------------
def predict_from_features(features: FeaturesInput) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Predict outcomes from feature rows.

    Input:
      - dict (one game)
      - list[dict] (many games)
      - DataFrame (many games)

    Output:
      - dict for single input
      - list[dict] for multi-row input

    Returned fields:
      - prediction
      - win_probability (if model supports predict_proba)
      - model_path
      - n_features_used
    """
    model = _load_model()
    X = _to_dataframe(features)

    if X.empty:
        return [] if isinstance(features, (list, pd.DataFrame)) else {
            "prediction": None,
            "win_probability": None,
            "model_path": MODEL_PATH,
            "n_features_used": 0,
            "warning": "No features provided."
        }

    preds = _safe_predict(model, X)
    probas = _safe_predict_proba(model, X)

    results: List[Dict[str, Any]] = []
    for i, pred in enumerate(preds):
        win_proba = probas[i] if probas is not None else None
        results.append(
            {
                "prediction": int(pred) if _is_int_like(pred) else pred,
                "win_probability": win_proba,
                "model_path": MODEL_PATH,
                "n_features_used": int(X.shape[1]),
            }
        )

    # Return shape matches input shape
    if isinstance(features, dict):
        return results[0]
    return results


def _is_int_like(x: Any) -> bool:
    try:
        # numpy scalars, ints, bools
        return float(x).is_integer()
    except Exception:
        return False