"""
experiments/collect_dataset.py
--------------------------------
Automated data collection script.

Runs three simulated workload phases back-to-back and logs system
metrics + labels to a CSV file.

Phases
------
  idle      : Sleep for `idle_duration` seconds
  cpu_bound : Spin all CPU cores for `cpu_duration` seconds
  io_bound  : Read/write temporary files for `io_duration`  seconds

Usage
-----
    python experiments/collect_dataset.py \
        --output data/features_dataset.csv \
        --idle   30 \
        --cpu    30 \
        --io     30 \
        --poll   0.05
"""

import argparse
import csv
import logging
import math
import multiprocessing
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collector.metric_collector   import MetricCollector
from features.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

FIELDNAMES = [
    "timestamp", "cpu_percent", "mem_percent",
    "disk_read_bytes", "disk_write_bytes",
    "cpu_rolling_mean", "cpu_rolling_std",
    "mem_rolling_mean", "disk_read_diff", "disk_write_diff",
    "label",
]


# ── Workload generators ───────────────────────────────────────────────────────

def _cpu_worker(stop_event):
    """Spin a single core until stop_event is set."""
    x = 0.0
    while not stop_event.is_set():
        x += math.sqrt(abs(math.sin(x + 1)))


def start_cpu_load(n_cores: int = None):
    stop = threading.Event()
    n = n_cores or multiprocessing.cpu_count()
    threads = [threading.Thread(target=_cpu_worker, args=(stop,), daemon=True)
               for _ in range(n)]
    for t in threads:
        t.start()
    return stop


def run_io_load(stop_event, chunk=1024 * 1024):
    """Write then read 1 MB chunks repeatedly until stop_event."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fname = f.name
    try:
        data = b"X" * chunk
        while not stop_event.is_set():
            with open(fname, "wb") as f:
                f.write(data)
            with open(fname, "rb") as f:
                _ = f.read()
    finally:
        if os.path.exists(fname):
            os.remove(fname)


# ── Collection loop ───────────────────────────────────────────────────────────

def collect_phase(collector, fe, writer, label, duration, poll):
    logger.info("Phase: %-10s  duration=%.0fs", label, duration)
    end = time.monotonic() + duration
    while time.monotonic() < end:
        t0       = time.monotonic()
        snapshot = collector.collect()
        features = fe.transform(snapshot)

        if features is not None:
            row = {
                "timestamp"        : snapshot["timestamp"],
                "cpu_percent"      : features[0],
                "mem_percent"      : features[1],
                "disk_read_bytes"  : features[2],
                "disk_write_bytes" : features[3],
                "cpu_rolling_mean" : features[4],
                "cpu_rolling_std"  : features[5],
                "mem_rolling_mean" : features[6],
                "disk_read_diff"   : features[7],
                "disk_write_diff"  : features[8],
                "label"            : label,
            }
            writer.writerow(row)

        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, poll - elapsed))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASA Dataset Collector")
    parser.add_argument("--output", default="data/features_dataset.csv")
    parser.add_argument("--idle",   type=float, default=30.0)
    parser.add_argument("--cpu",    type=float, default=30.0)
    parser.add_argument("--io",     type=float, default=30.0)
    parser.add_argument("--poll",   type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    collector = MetricCollector()
    fe        = FeatureEngineer(window=10)

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        # ── Idle phase ────────────────────────────────────────────────
        collect_phase(collector, fe, writer, "idle", args.idle, args.poll)

        # ── CPU-bound phase ───────────────────────────────────────────
        stop = start_cpu_load()
        try:
            collect_phase(collector, fe, writer, "cpu_bound", args.cpu, args.poll)
        finally:
            stop.set()

        # ── I/O-bound phase ───────────────────────────────────────────
        stop = threading.Event()
        io_thread = threading.Thread(target=run_io_load, args=(stop,), daemon=True)
        io_thread.start()
        try:
            collect_phase(collector, fe, writer, "io_bound", args.io, args.poll)
        finally:
            stop.set()
            io_thread.join(timeout=2)

    logger.info("Dataset saved → %s", args.output)


if __name__ == "__main__":
    main()
