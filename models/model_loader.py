"""
models/model_loader.py
-----------------------
Central utility for loading the trained RandomForest model and
StandardScaler from disk.  Import this wherever inference is needed.
"""

import logging
import os

import joblib

logger = logging.getLogger(__name__)

_BASE = os.path.dirname(__file__)

DEFAULT_MODEL_PATH  = os.path.join(_BASE, "rf_model.pkl")
DEFAULT_SCALER_PATH = os.path.join(_BASE, "scaler.pkl")
DEFAULT_FEATURE_COLS_PATH = os.path.join(_BASE, "feature_columns.pkl")


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load and return the trained sklearn model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run  python training/train_model.py  first."
        )
    model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
    return model


def load_scaler(scaler_path: str = DEFAULT_SCALER_PATH):
    """Load and return the fitted StandardScaler."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Run  python training/train_model.py  first."
        )
    scaler = joblib.load(scaler_path)
    logger.info("Scaler loaded from %s", scaler_path)
    return scaler


def load_feature_columns(feature_cols_path: str = DEFAULT_FEATURE_COLS_PATH):
    """Load and return the list of feature column names used during training."""
    if not os.path.exists(feature_cols_path):
        logger.warning("Feature columns file not found at %s. Using defaults.", feature_cols_path)
        return None
    feature_cols = joblib.load(feature_cols_path)
    logger.info("Feature columns loaded from %s", feature_cols_path)
    return feature_cols


def load_artifacts(
    model_path=DEFAULT_MODEL_PATH,
    scaler_path=DEFAULT_SCALER_PATH,
    feature_cols_path=DEFAULT_FEATURE_COLS_PATH
):
    """Convenience: load model, scaler, and feature columns."""
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    feature_cols = load_feature_columns(feature_cols_path)
    return model, scaler, feature_cols
