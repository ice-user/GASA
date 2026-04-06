#!/usr/bin/env python
"""
training/train_custom_dataset.py
----------------------------------
Convenience script to train on custom datasets with automatic or manual column mapping.

Usage
-----
    # Auto-detect common column name variations
    python training/train_custom_dataset.py data/my_custom_dataset.csv

    # With manual column mapping
    python training/train_custom_dataset.py \\
        data/my_custom_dataset.csv \\
        --map cpu_usage cpu_percent mem_usage mem_percent \\
             cpu_rolling_5 cpu_rolling_mean mem_rolling_5 mem_rolling_mean \\
             disk_read_diff disk_read_diff disk_write_diff disk_write_diff
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_model import train, parse_column_mapping

logger = logging.getLogger(__name__)

# Common column name variations to auto-detect
COMMON_VARIATIONS = {
    # CPU metrics
    "cpu_usage": "cpu_percent",
    "cpu": "cpu_percent",
    "cpu_load": "cpu_percent",
    "cpu_util": "cpu_percent",
    "cpu_core_avg": "cpu_percent",
    
    # Memory metrics
    "mem_usage": "mem_percent",
    "memory": "mem_percent",
    "mem": "mem_percent",
    "memory_percent": "mem_percent",
    "mem_pressure": "mem_percent",
    
    # Rolling averages
    "cpu_rolling_5": "cpu_rolling_mean",
    "cpu_rolling_10": "cpu_rolling_mean",
    "cpu_rolling_avg": "cpu_rolling_mean",
    "mem_rolling_5": "mem_rolling_mean",
    "mem_rolling_10": "mem_rolling_mean",
    "mem_rolling_avg": "mem_rolling_mean",
    
    # Disk metrics (usually match already)
}


def auto_detect_mapping(csv_path: str) -> dict:
    """Auto-detect column mapping from CSV headers."""
    import pandas as pd
    
    df = pd.read_csv(csv_path, nrows=0)  # Just read headers
    columns = df.columns.tolist()
    
    mapping = {}
    found = set()
    
    logger.info("Auto-detecting column mapping...")
    logger.info("Available columns: %s", columns)
    
    for col in columns:
        if col in COMMON_VARIATIONS:
            standard_name = COMMON_VARIATIONS[col]
            mapping[col] = standard_name
            found.add(standard_name)
            logger.info("  %s → %s", col, standard_name)
    
    return mapping, found


def main():
    parser = argparse.ArgumentParser(
        description="Train model on custom dataset with optional column mapping"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to CSV dataset"
    )
    parser.add_argument(
        "--map",
        nargs="+",
        default=[],
        help="Manual column mapping (alternating old new pairs)"
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        default=True,
        help="Try to auto-detect column names (default: True)"
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable auto-detection"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save trained model"
    )

    args = parser.parse_args()

    # Check dataset exists
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    # Build column mapping
    column_mapping = {}

    # Try auto-detection first if not disabled
    if args.no_auto_detect:
        auto_detect = False
    else:
        auto_detect = args.auto_detect

    if auto_detect:
        detected_mapping, found_cols = auto_detect_mapping(args.dataset)
        column_mapping.update(detected_mapping)
        logger.info("Auto-detected mapping: %s", detected_mapping)

    # Apply manual overrides
    if args.map:
        if len(args.map) % 2 != 0:
            print("ERROR: Column mapping must have alternating pairs (old new old new ...)")
            sys.exit(1)
        
        for i in range(0, len(args.map), 2):
            old_col, new_col = args.map[i], args.map[i + 1]
            column_mapping[old_col] = new_col
            logger.info("Manual mapping: %s → %s", old_col, new_col)

    logger.info("Final column mapping: %s", column_mapping if column_mapping else "None (using defaults)")

    # Train
    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "rf_model.pkl"
        )

    try:
        train(
            data_path=args.dataset,
            column_mapping=column_mapping if column_mapping else None,
            model_path=model_path
        )
        print("\n✓ Training completed successfully!")
        print(f"  Model saved to: {model_path}")
    except Exception as e:
        logger.error("Training failed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
