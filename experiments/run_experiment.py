"""
experiments/run_experiment.py
------------------------------
One-command reproducibility script.

Steps
-----
1. Build processed dataset from raw CSV
2. Train model + cross-validate
3. Evaluate on test set + print metrics
4. Generate all plots

Usage
-----
    python experiments/run_experiment.py
    python experiments/run_experiment.py --raw data/features_dataset.csv
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/features_dataset.csv")
    args = parser.parse_args()

    # ── Step 1: Dataset ──────────────────────────────────────────────────
    logger.info("=== Step 1/4: Building dataset ===")
    from training.dataset_builder import build_dataset
    build_dataset(raw_csv=args.raw)

    # ── Step 2: Training ─────────────────────────────────────────────────
    logger.info("=== Step 2/4: Training model ===")
    from training.train_model import train
    model, cv_scores = train()
    logger.info("Mean CV F1 = %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    # ── Step 3: Evaluation ───────────────────────────────────────────────
    logger.info("=== Step 3/4: Evaluating model ===")
    from evaluation.evaluate_model import evaluate
    evaluate()

    # ── Step 4: Plots ────────────────────────────────────────────────────
    logger.info("=== Step 4/4: Generating plots ===")
    from visualization.plot_all import main as plot_main
    plot_main()

    logger.info("Experiment complete. Figures saved to asa/figures/")


if __name__ == "__main__":
    main()
