"""
features/feature_engineering.py
---------------------------------
Converts a stream of raw metric snapshots into a 9-element feature vector.

Features (in order)
-------------------
0  cpu_percent
1  mem_percent
2  disk_read_bytes
3  disk_write_bytes
4  cpu_rolling_mean   (window W=10)
5  cpu_rolling_std    (window W=10)
6  mem_rolling_mean   (window W=10)
7  disk_read_diff     (current - previous)
8  disk_write_diff    (current - previous)
"""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

WINDOW_SIZE = 10
FEATURE_NAMES = [
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


class FeatureEngineer:
    """
    Maintains an internal rolling window and computes the feature vector
    from the latest metric snapshot.

    Usage
    -----
        fe = FeatureEngineer()
        # feed snapshots one at a time; returns None until window is full
        vec = fe.transform(snapshot)
    """

    def __init__(self, window: int = WINDOW_SIZE):
        self.window = window
        self._cpu_buf   = deque(maxlen=window)
        self._mem_buf   = deque(maxlen=window)
        self._prev_read  = None
        self._prev_write = None
        logger.info("FeatureEngineer initialised (window=%d).", window)

    def transform(self, snapshot: dict):
        """
        Parameters
        ----------
        snapshot : dict   output of MetricCollector.collect()

        Returns
        -------
        numpy.ndarray of shape (9,)  or  None if window not yet full.
        """
        cpu   = snapshot["cpu_percent"]
        mem   = snapshot["mem_percent"]
        d_rd  = snapshot["disk_read_bytes"]
        d_wr  = snapshot["disk_write_bytes"]

        # Rolling window update
        self._cpu_buf.append(cpu)
        self._mem_buf.append(mem)

        # Disk diffs
        rd_diff = d_rd  - self._prev_read  if self._prev_read  is not None else 0.0
        wr_diff = d_wr  - self._prev_write if self._prev_write is not None else 0.0
        self._prev_read  = d_rd
        self._prev_write = d_wr

        if len(self._cpu_buf) < self.window:
            logger.debug("Window not full yet (%d/%d).", len(self._cpu_buf), self.window)
            return None

        cpu_arr = np.array(self._cpu_buf)
        mem_arr = np.array(self._mem_buf)

        features = np.array([
            cpu,
            mem,
            d_rd,
            d_wr,
            float(cpu_arr.mean()),
            float(cpu_arr.std()),
            float(mem_arr.mean()),
            rd_diff,
            wr_diff,
        ], dtype=np.float64)

        logger.debug("Feature vector: %s", features)
        return features

    @staticmethod
    def feature_names():
        return FEATURE_NAMES.copy()
