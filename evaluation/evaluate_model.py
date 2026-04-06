"""
evaluation/evaluate_model.py
-----------------------------
Loads saved model + test split and prints:
  • classification report (precision, recall, F1 per class)
  • confusion matrix
  • prediction latency (mean & p99)
  • agent loop overhead estimate

Run:
    python evaluation/evaluate_model.py
"""

import logging
import os
import sys
import time

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.model_loader import load_artifacts

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def evaluate():
    # ── Load artifacts ───────────────────────────────────────────────────
    model, scaler, feature_cols = load_artifacts()

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"), allow_pickle=True)

    n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else X_test.shape[1]
    logger.info("Model trained on %d features", n_features)
    if feature_cols:
        logger.info("Feature columns: %s", feature_cols)

    # ── Predictions ──────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Reports ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Classification Report")
    print("="*55)
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print()

    # ── Latency benchmark ────────────────────────────────────────────────
    # Measure single-sample inference latency over 1000 calls
    N = 1000
    sample = X_test[:1]
    latencies = []
    for _ in range(N):
        t0 = time.perf_counter()
        model.predict(sample)
        latencies.append((time.perf_counter() - t0) * 1e6)   # → µs

    latencies = np.array(latencies)
    print("="*55)
    print("  Inference Latency  (µs, N=1000)")
    print("="*55)
    print(f"  Mean  : {latencies.mean():.1f} µs")
    print(f"  Median: {np.median(latencies):.1f} µs")
    print(f"  P99   : {np.percentile(latencies, 99):.1f} µs")
    print(f"  Max   : {latencies.max():.1f} µs")
    print()


if __name__ == "__main__":
    evaluate()
