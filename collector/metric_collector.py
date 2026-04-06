"""
collector/metric_collector.py
------------------------------
Collects raw system metrics using psutil at a fixed polling interval.
Returns a snapshot dict each call; caller is responsible for windowing.
"""

import time
import logging
import psutil

logger = logging.getLogger(__name__)


class MetricCollector:
    """
    Polls system metrics once per call.

    Usage:
        mc = MetricCollector()
        snapshot = mc.collect()
    """

    def __init__(self):
        # Prime psutil so first cpu_percent call returns a valid reading
        psutil.cpu_percent(interval=None)
        self._last_disk = psutil.disk_io_counters()
        self._last_ts   = time.monotonic()
        logger.info("MetricCollector initialised.")

    def collect(self) -> dict:
        """
        Returns a single metric snapshot.

        Keys
        ----
        timestamp       : float   – wall-clock epoch seconds
        cpu_percent     : float   – CPU utilisation  0-100
        mem_percent     : float   – RAM utilisation  0-100
        disk_read_bytes : float   – cumulative bytes read  since boot
        disk_write_bytes: float   – cumulative bytes written since boot
        """
        ts       = time.time()
        cpu_pct  = psutil.cpu_percent(interval=None)
        mem_pct  = psutil.virtual_memory().percent

        disk = psutil.disk_io_counters()
        disk_read  = float(disk.read_bytes)  if disk else 0.0
        disk_write = float(disk.write_bytes) if disk else 0.0

        snapshot = {
            "timestamp"        : ts,
            "cpu_percent"      : cpu_pct,
            "mem_percent"      : mem_pct,
            "disk_read_bytes"  : disk_read,
            "disk_write_bytes" : disk_write,
        }
        logger.debug("Snapshot: %s", snapshot)
        return snapshot
