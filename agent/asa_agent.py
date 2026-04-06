"""
agent/asa_agent.py
-------------------
Real-time Adaptive Scheduling Agent main loop.

Pipeline per tick
-----------------
1. Collect system metrics          (MetricCollector)
2. Engineer feature vector         (FeatureEngineer)
3. Scale features                  (StandardScaler)
4. Predict workload class          (RandomForest)
5. Push to vote buffer             (MajorityVoter)
6. Compute weighted vote winner    (MajorityVoter)
7. Maybe switch scheduler          (SchedulerRouter)

Run:
    python agent/asa_agent.py [--duration 60] [--poll 0.05]
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# ── Path setup (allow running from any directory) ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collector.metric_collector  import MetricCollector
from features.feature_engineering import FeatureEngineer
from models.model_loader          import load_artifacts
from agent.majority_voting        import MajorityVoter
from scheduler.scheduler_router   import SchedulerRouter

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_agent(duration_s: float = 60.0, poll_s: float = 0.05, simulation_mode: bool = False):
    """
    Run the ASA agent loop for `duration_s` seconds.

    Parameters
    ----------
    duration_s : float – how long to run (seconds); -1 = run forever
    poll_s     : float – polling interval in seconds (default 50 ms)
    simulation_mode : bool – if True, don't apply real scheduler changes (default: False)
    """
    # ── Load artifacts ───────────────────────────────────────────────────
    model, scaler, feature_cols = load_artifacts()

    # ── Instantiate modules ──────────────────────────────────────────────
    collector = MetricCollector()
    fe        = FeatureEngineer(window=10)
    voter     = MajorityVoter(k=5, alpha=0.3)
    router    = SchedulerRouter(
        cooldown_s=0.5,
        enable_real_control=not simulation_mode,  # Disable if in simulation
        simulation_mode=simulation_mode,           # Force simulation if flag set
    )

    # ── Get expected number of features from scaler ──────────────────────
    n_expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 9
    logger.info("Model expects %d features from StandardScaler", n_expected_features)

    # ── Telemetry log ────────────────────────────────────────────────────
    log_rows = []   # (timestamp, cpu, mem, raw_pred, voted_pred, switched)

    mode_str = "SIMULATION" if simulation_mode else "REAL"
    logger.info("ASA agent starting [%s mode].  duration=%.0fs  poll=%.0fms",
                mode_str, duration_s, poll_s * 1000)

    start = time.monotonic()

    try:
        while True:
            tick_start = time.monotonic()

            # ── 1. Collect ───────────────────────────────────────────────
            snapshot = collector.collect()

            # ── 2. Engineer features ─────────────────────────────────────
            features = fe.transform(snapshot)
            if features is None:
                time.sleep(poll_s)
                continue   # window not yet full

            # ── 3. Handle feature count mismatch ──────────────────────────
            # If model expects more features than FeatureEngineer produces,
            # pad with zeros (e.g., if trained on dataset with extra computed columns)
            if len(features) < n_expected_features:
                features = np.pad(features, (0, n_expected_features - len(features)), mode='constant')
                logger.debug("Padded features from %d to %d", len(features) - (n_expected_features - len(features)), n_expected_features)

            # ── 4. Scale ─────────────────────────────────────────────────
            features_scaled = scaler.transform(features.reshape(1, -1))

            # ── 5. Predict ───────────────────────────────────────────────
            raw_pred = model.predict(features_scaled)[0]

            # ── 6 & 7. Vote ──────────────────────────────────────────────
            voter.push(raw_pred)
            voted = voter.vote()

            # ── 8. Maybe switch ──────────────────────────────────────────
            switched = router.maybe_switch(voted)

            # ── Log ──────────────────────────────────────────────────────
            log_rows.append({
                "timestamp"   : snapshot["timestamp"],
                "cpu_percent" : snapshot["cpu_percent"],
                "mem_percent" : snapshot["mem_percent"],
                "raw_pred"    : raw_pred,
                "voted_pred"  : voted,
                "active_sched": router.active_class,
                "switched"    : switched,
            })

            logger.info(
                "cpu=%.1f%%  mem=%.1f%%  raw=%-10s  voted=%-10s  sched=%s%s",
                snapshot["cpu_percent"],
                snapshot["mem_percent"],
                raw_pred,
                voted,
                router.active_class,
                "  [SWITCH]" if switched else "",
            )

            # ── Timing ───────────────────────────────────────────────────
            elapsed_total = time.monotonic() - start
            if duration_s > 0 and elapsed_total >= duration_s:
                break

            tick_elapsed = time.monotonic() - tick_start
            sleep_time   = max(0.0, poll_s - tick_elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    # ── Summary ──────────────────────────────────────────────────────────
    summary = router.summary()
    logger.info("Agent finished.  Total switches: %d  Final sched: %s",
                summary["total_switches"], summary["active_class"])

    # ── Save agent run log ───────────────────────────────────────────────
    if log_rows:
        import pandas as pd
        log_df = pd.DataFrame(log_rows)
        log_path = os.path.join(os.path.dirname(__file__), "..", "data", "agent_run_log.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_df.to_csv(log_path, index=False)
        logger.info("Agent run log saved → %s", log_path)

    return log_rows, summary


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASA Real-time Agent")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Runtime in seconds (-1 = forever)")
    parser.add_argument("--poll",     type=float, default=0.05,
                        help="Polling interval in seconds (default 0.05 = 50ms)")
    parser.add_argument("--sim",      action="store_true",
                        help="Run in simulation mode (no real scheduler changes)")
    args = parser.parse_args()
    run_agent(duration_s=args.duration, poll_s=args.poll, simulation_mode=args.sim)
