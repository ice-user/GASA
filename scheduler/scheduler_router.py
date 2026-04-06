"""
scheduler/scheduler_router.py
------------------------------
Adaptive scheduler router with real Linux control.

Maps each workload class (cpu_bound, io_bound, idle) to actual Linux scheduling
policies using system calls (sched_setscheduler, sched_setaffinity, nice, etc.).

Integrates SchedulerControl from scheduler_control.py to apply real changes
while supporting simulation mode for testing on non-Linux systems.

Usage:
    router = SchedulerRouter(enable_real_control=True, simulation_mode=False)
    switched = router.maybe_switch("cpu_bound")
"""

import logging
import os
import platform
import time
from typing import Optional

from .scheduler_control import SchedulerControl, validate_environment

logger = logging.getLogger(__name__)

# ── Policy definitions ────────────────────────────────────────────────────────
POLICIES = {
    "cpu_bound": {
        "name"           : "throughput_scheduler",
        "time_slice_ms"  : 20,       # longer quanta to reduce context switches
        "priority_nice"  : -5,       # slightly elevated priority
        "description"    : "Optimised for compute throughput; large time slices.",
        "control_method" : "apply_cpu_bound",
    },
    "io_bound": {
        "name"           : "io_aware_scheduler",
        "time_slice_ms"  : 4,        # short quanta; yields quickly on I/O wait
        "priority_nice"  : 5,        # lower priority for I/O
        "description"    : "Short time slices; yields promptly on I/O wait.",
        "control_method" : "apply_io_bound",
    },
    "idle": {
        "name"           : "default_scheduler",
        "time_slice_ms"  : 10,       # OS default
        "priority_nice"  : 0,        # default priority
        "description"    : "Default OS policy; minimal interference.",
        "control_method" : "apply_idle",
    },
}

SWITCH_COOLDOWN_S = 0.5   # 500 ms


class SchedulerRouter:
    """
    Adaptive scheduler router with real Linux control integration.

    Decides when to switch schedulers and executes actual scheduling changes
    using SchedulerControl (real) or simulation mode (testing).

    Parameters
    ----------
    cooldown_s : float
        Minimum seconds between switches (default 0.5)
    enable_real_control : bool
        If True, apply actual scheduling changes. If False, only log (default: True)
    simulation_mode : bool
        If True, don't execute system calls (useful for testing) (default: False)
    pid : int, optional
        PID to control. If None, controls current process (default: None)

    Usage
    -----
        router = SchedulerRouter(enable_real_control=True)
        switched = router.maybe_switch("cpu_bound")
        
        # Check if a switch actually happened
        print(router.active_class)
        print(router.summary())
    """

    def __init__(
        self,
        cooldown_s: float = SWITCH_COOLDOWN_S,
        enable_real_control: bool = True,
        simulation_mode: bool = False,
        pid: Optional[int] = None,
    ):
        self.cooldown_s = cooldown_s
        self.active_class = "idle"
        self._last_switch = 0.0
        self._switch_count = 0
        self._switch_log = []  # list of (timestamp, from, to, success, message)
        
        # Real control integration
        self.enable_real_control = enable_real_control
        self.simulation_mode = simulation_mode
        self.scheduler_control: Optional[SchedulerControl] = None
        
        # Initialize SchedulerControl if enabled
        if enable_real_control:
            is_valid, env_msg = validate_environment()
            if not is_valid and not simulation_mode:
                logger.warning(
                    "Real control disabled: %s. Falling back to simulation mode.", env_msg
                )
                self.simulation_mode = True
            
            self.scheduler_control = SchedulerControl(
                simulation_mode=self.simulation_mode,
                pid=pid,
                auto_fallback=True,
            )
            self.scheduler_control.save_context()

            env_report = self.scheduler_control.environment_report()
            logger.info("Environment report: %s", env_report)
            logger.info(
                "SchedulerRouter initialized (real_control=%s, sim=%s)",
                enable_real_control,
                self.simulation_mode,
            )
        else:
            logger.info("SchedulerRouter initialized (simulation mode only)")

    @property
    def active_policy(self) -> dict:
        return POLICIES[self.active_class]

    def maybe_switch(self, voted_class: str) -> bool:
        """
        Switch to voted_class if:
          • it differs from the current active class, AND
          • the cooldown period has elapsed.

        If enable_real_control is True, applies actual scheduling changes.

        Returns
        -------
        bool – True if a switch was performed.
        """
        now = time.monotonic()

        if voted_class == self.active_class:
            return False

        if (now - self._last_switch) < self.cooldown_s:
            logger.debug(
                "Cooldown active (%.2fs remaining). No switch.",
                self.cooldown_s - (now - self._last_switch),
            )
            return False

        self._execute_switch(voted_class, now)
        return True

    def _execute_switch(self, target: str, ts: float):
        """Execute the actual switch with real control or simulation."""
        prev = self.active_class
        self.active_class = target
        self._last_switch = ts
        self._switch_count += 1
        
        policy = POLICIES[target]
        
        # Apply real scheduling changes if enabled
        success = True
        message = "(simulation)"
        
        if self.enable_real_control and self.scheduler_control:
            control_method_name = policy.get("control_method")
            if control_method_name:
                try:
                    method = getattr(self.scheduler_control, control_method_name)
                    success, message = method()
                except AttributeError:
                    logger.error("Unknown control method: %s", control_method_name)
                    message = f"Unknown method: {control_method_name}"
                    success = False
                except Exception as e:
                    logger.error("Control method failed: %s", e)
                    message = str(e)
                    success = False
        
        # Log the switch
        self._switch_log.append((ts, prev, target, success, message))
        
        logger.info(
            "SCHEDULER SWITCH  %s → %s  |  policy=%s  time_slice=%dms  [%s]",
            prev,
            target,
            policy["name"],
            policy["time_slice_ms"],
            "SUCCESS" if success else "FAILED",
        )
        
        if message != "(simulation)":
            logger.info("  Details: %s", message)

    def summary(self) -> dict:
        """Return switch history and current state."""
        return {
            "total_switches": self._switch_count,
            "active_class": self.active_class,
            "switch_log": self._switch_log,
            "real_control_enabled": self.enable_real_control,
            "simulation_mode": self.simulation_mode,
            "environment": self.get_environment_report(),
        }

    def get_environment_report(self) -> dict:
        """Return environment and scheduler control status."""
        if self.scheduler_control:
            return self.scheduler_control.environment_report()

        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "is_linux": os.name == "posix" and platform.system() == "Linux",
            "simulation_mode": self.simulation_mode,
            "real_control_enabled": self.enable_real_control,
        }

    def get_context(self):
        """Get current scheduling context if real control is enabled."""
        if self.scheduler_control:
            return self.scheduler_control.get_current_context()
        return None
    
    def restore_initial_context(self) -> bool:
        """Restore the scheduling context from before router initialization."""
        if self.scheduler_control:
            return self.scheduler_control.restore_context()
        return False
