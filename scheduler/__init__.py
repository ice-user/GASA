"""
scheduler package
------------------
Adaptive scheduler router and real Linux scheduler control.

Modules:
  - scheduler_router.py  – Routes workload classes to scheduling policies
  - scheduler_control.py – Implements real Linux scheduling control (optional)
"""

from .scheduler_router import SchedulerRouter, POLICIES

__all__ = ["SchedulerRouter", "POLICIES"]
