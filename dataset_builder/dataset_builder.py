"""
training/dataset_builder.py
-----------------------------
Loads raw CSV data, applies feature engineering row-by-row,
applies SMOTE to balance classes, and saves train/test splits.

Expected CSV columns (from your data collection):
    timestamp, cpu_percent, mem_percent, disk_read_bytes,
    disk_write_bytes, cpu_rolling_mean, mem_rolling_mean,
    disk_read_diff, disk_write_diff, label
    (cpu_rolling_std is derived here if missing, otherwise used as-is)

Supports custom datasets with column name mapping via CUSTOM_COLUMN_MAPPING.
"""

import logging
import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

RAW_CSV      = os.path.join(os.path.dirname(__file__), "..", "data", "features_dataset.csv")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
SCALER_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_columns.pkl")

FEATURE_COLS = [
    "cpu_percent",
    "mem_percent",
    "disk_read_bytes",
    "disk_write_bytes",
    "cpu_rolling_mean",
    "cpu_rolling_std",
    "mem_rolling_mean",
    "disk_read_diff",
    "disk_write_diff",
]


def build_dataset(
    raw_csv: str  = RAW_CSV,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Full pipeline: load → clean → scale → SMOTE → split → save.

    Parameters
    ----------
    raw_csv : str
        Path to the CSV file
    test_size : float
        Fraction for test set
    random_state : int
        Random seed

    Returns
    -------
    X_train, X_test, y_train, y_test  (numpy arrays, scaled)
    """
    logger.info("Loading dataset from %s", raw_csv)
    df = pd.read_csv(raw_csv)
    logger.info("Raw shape: %s", df.shape)
    logger.info("Raw columns: %s", list(df.columns))

    logger.info("Class distribution:\n%s", df["label"].value_counts())

    # ── Auto-detect feature columns: use all numeric columns except label & timestamp
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in all_numeric if c not in {"timestamp"}]
    logger.info("Using feature columns: %s", feature_cols)

    # ── Handle missing disk_read/write_bytes columns ─────────────────────
    # If only diffs are available, compute cumulative sums
    if "disk_read_bytes" not in df.columns and "disk_read_diff" in df.columns:
        logger.info("Computing disk_read_bytes from disk_read_diff (cumulative sum)")
        df["disk_read_bytes"] = df["disk_read_diff"].fillna(0).cumsum()
        feature_cols.append("disk_read_bytes")
    
    if "disk_write_bytes" not in df.columns and "disk_write_diff" in df.columns:
        logger.info("Computing disk_write_bytes from disk_write_diff (cumulative sum)")
        df["disk_write_bytes"] = df["disk_write_diff"].fillna(0).cumsum()
        feature_cols.append("disk_write_bytes")

    # ── Extract features and labels ──────────────────────────────────────
    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values

    # ── Train / test split (stratified) ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))

    # ── Feature scaling ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaler saved → %s", SCALER_PATH)

    # ── SMOTE (applied only to training set) ────────────────────────────
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)
    logger.info("After SMOTE – train size: %d", len(X_train_bal))
    logger.info("Balanced class counts:\n%s",
                pd.Series(y_train_bal).value_counts().to_string())

    # ── Save processed splits ────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_bal)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test_s)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train_bal)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)
    logger.info("Processed splits saved to %s", OUTPUT_DIR)

    # ── Save feature column names for reproducibility ────────────────────
    os.makedirs(os.path.dirname(FEATURE_COLS_PATH), exist_ok=True)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    logger.info("Feature columns saved → %s", FEATURE_COLS_PATH)

    return X_train_bal, X_test_s, y_train_bal, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    build_dataset()
