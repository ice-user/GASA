"""
training/train_model.py
------------------------
Trains a RandomForestClassifier on the balanced dataset,
runs 5-fold stratified cross-validation, and saves the model.

Usage:
    # Train on default dataset
    python training/train_model.py

    # Train on custom dataset
    python training/train_model.py --data path/to/custom_dataset.csv
"""

import argparse
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

from dataset_builder.dataset_builder import build_dataset

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rf_model.pkl")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators  = 300,
    class_weight  = "balanced",
    random_state  = 42,
    n_jobs        = -1,
)


def train(
    data_path: str = None,
    model_path: str = MODEL_PATH
):
    """
    Full training pipeline:
      1. Load processed dataset
      2. Train RandomForest
      3. Cross-validate on training set
      4. Save model artifact

    Parameters
    ----------
    data_path : str, optional
        Path to custom CSV dataset. If None, uses default dataset.
    model_path : str
        Where to save the trained model.

    Returns
    -------
    model, cv_scores
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "features_dataset.csv"
        )
    
    logger.info("Training configuration:")
    logger.info("  Data path: %s", data_path)
    logger.info("  Model params: %s", RF_PARAMS)

    X_train, X_test, y_train, y_test = build_dataset(raw_csv=data_path)

    logger.info("Training RandomForestClassifier  params=%s", RF_PARAMS)
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # ── Cross-validation ─────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring="f1_macro", n_jobs=-1)
    logger.info("CV F1 scores : %s", np.round(cv_scores, 3))
    logger.info("Mean CV F1   : %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved → %s", model_path)

    return model, cv_scores


def main():
    parser = argparse.ArgumentParser(
        description="Train RandomForest on system metrics dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV dataset (default: data/features_dataset.csv)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to save trained model"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    train(
        data_path=args.data,
        model_path=args.model_path
    )


if __name__ == "__main__":
    main()
