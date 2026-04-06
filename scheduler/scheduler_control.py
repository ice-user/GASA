"""
scheduler/scheduler_control.py
------------------------------
Real Linux scheduler control layer.

Implements actual system-level scheduling changes using Linux system calls:
- os.sched_setscheduler()  – change scheduling policy (SCHED_FIFO, SCHED_RR, SCHED_BATCH)
- os.sched_setaffinity()   – bind process to CPU cores
- os.nice()                – adjust process priority
- psutil                   – process inspection and management

Features:
  • Policy application with safety fallbacks
  • Permission-aware (graceful degradation)
  • Per-process and global control
  • Dry-run/simulation mode
  • Comprehensive logging

Author: Adaptive Scheduler Assistant
"""

import errno
import logging
import os
import platform
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Platform & Availability Checks ────────────────────────────────────────────
_IS_LINUX = platform.system() == "Linux"
_HAS_PSUTIL = False
_PYTHON_VERSION = sys.version_info[:2]


def is_wsl() -> bool:
    """Detect WSL (Windows Subsystem for Linux)."""
    if not _IS_LINUX:
        return False
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            version = f.read()
        return "Microsoft" in version or "WSL" in version
    except Exception:
        return False


# resource module is Unix-only
try:
    import resource
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    logger.warning("psutil not available; process inspection disabled")

# ── Scheduler policies (Linux constants) ──────────────────────────────────────
try:
    SCHED_NORMAL = os.SCHED_OTHER if hasattr(os, 'SCHED_OTHER') else 0
    SCHED_FIFO = os.SCHED_FIFO if hasattr(os, 'SCHED_FIFO') else 1
    SCHED_RR = os.SCHED_RR if hasattr(os, 'SCHED_RR') else 2
    SCHED_BATCH = os.SCHED_BATCH if hasattr(os, 'SCHED_BATCH') else 3
    SCHED_IDLE = os.SCHED_IDLE if hasattr(os, 'SCHED_IDLE') else 5
except AttributeError:
    SCHED_NORMAL = 0
    SCHED_FIFO = 1
    SCHED_RR = 2
    SCHED_BATCH = 3
    SCHED_IDLE = 5

SCHED_NAMES = {
    SCHED_NORMAL: "SCHED_OTHER",
    SCHED_FIFO: "SCHED_FIFO",
    SCHED_RR: "SCHED_RR",
    SCHED_BATCH: "SCHED_BATCH",
    SCHED_IDLE: "SCHED_IDLE",
}


@dataclass
class SchedulingContext:
    """Captures state of a process's scheduling configuration."""
    policy: str
    priority: int
    nice: int
    cpu_affinity: Optional[List[int]]
    
    def __repr__(self) -> str:
        affinity_str = f"CPUs={self.cpu_affinity}" if self.cpu_affinity else "CPUs=all"
        return f"SchedulingContext(policy={self.policy}, nice={self.nice:+d}, {affinity_str})"


class SchedulerControl:
    """
    Real-world Linux scheduler control implementation.
    
    Applies actual scheduling policies to processes. Supports simulation mode
    for testing and safe fallback when permissions are insufficient.
    
    Parameters
    ----------
    simulation_mode : bool
        If True, log actions without executing them (default: False)
    pid : int, optional
        Process ID to control. If None, controls current process (default: None)
    auto_fallback : bool
        If True, gracefully degrade on permission errors (default: True)
    
    Attributes
    ----------
    simulation_mode : bool
        Current simulation mode state
    controlled_pid : int
        PID being controlled
    changes_applied : int
        Count of successfully applied changes
    """
    
    def __init__(
        self,
        simulation_mode: bool = False,
        pid: Optional[int] = None,
        auto_fallback: bool = True,
    ):
        self.simulation_mode = simulation_mode
        self.controlled_pid = pid if pid is not None else os.getpid()
        self.auto_fallback = auto_fallback
        self.changes_applied = 0
        self._previous_context: Optional[SchedulingContext] = None
        
        # Platform validation
        if not _IS_LINUX:
            logger.warning(
                "Not running on Linux. System calls may not work. Platform: %s",
                platform.system(),
            )
        
        if not self.simulation_mode:
            logger.info(
                "SchedulerControl initialized (PID=%d, simulation=%s, fallback=%s)",
                self.controlled_pid,
                self.simulation_mode,
                self.auto_fallback,
            )
        else:
            logger.info(
                "SchedulerControl in SIMULATION MODE (PID=%d)",
                self.controlled_pid,
            )

    def environment_report(self) -> dict:
        """Return runtime environment and mode details."""
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "is_linux": _IS_LINUX,
            "is_wsl": is_wsl(),
            "python_version": sys.version,
            "has_psutil": _HAS_PSUTIL,
            "supports_sched_setscheduler": hasattr(os, "sched_setscheduler"),
            "supports_sched_setaffinity": hasattr(os, "sched_setaffinity"),
            "is_root": is_running_as_root(),
            "simulation_mode": self.simulation_mode,
            "auto_fallback": self.auto_fallback,
            "controlled_pid": self.controlled_pid,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Core Control Methods
    # ──────────────────────────────────────────────────────────────────────────
    
    def apply_cpu_bound(self) -> Tuple[bool, str]:
        """
        Apply CPU-bound workload scheduling:
          • Increase priority (nice = -5)
          • Set SCHED_BATCH if available, else SCHED_FIFO
          • Allow all CPU cores
        
        Returns
        -------
        tuple : (success: bool, message: str)
        """
        logger.info("Applying CPU_BOUND policy...")
        
        try:
            # Priority
            self._set_nice(-5)
            
            # Scheduling policy: prefer SCHED_BATCH for batch workloads
            if hasattr(os, 'SCHED_BATCH'):
                self._set_policy(SCHED_BATCH)
            else:
                logger.warning("SCHED_BATCH unavailable; falling back to SCHED_FIFO")
                self._set_policy(SCHED_FIFO)
            
            # CPU affinity: use all cores
            self._set_cpu_affinity(None)  # None = all cores
            
            self.changes_applied += 1
            msg = "CPU_BOUND applied: nice=-5, policy=BATCH, affinity=all_cores"
            logger.info(msg)
            return True, msg
        
        except PermissionError as e:
            return self._handle_permission_error("CPU_BOUND", e)
        except Exception as e:
            return self._handle_general_error("CPU_BOUND", e)
    
    def apply_io_bound(self) -> Tuple[bool, str]:
        """
        Apply I/O-bound workload scheduling:
          • Reduce priority (nice = +5)
          • Keep SCHED_NORMAL for responsiveness
          • Allow all CPU cores (no restriction)
        
        Returns
        -------
        tuple : (success: bool, message: str)
        """
        logger.info("Applying IO_BOUND policy...")
        
        try:
            # Priority: lower priority for I/O workloads
            self._set_nice(5)
            
            # Policy: SCHED_NORMAL allows for quick wake-ups
            self._set_policy(SCHED_NORMAL)
            
            # Affinity: unrestricted
            self._set_cpu_affinity(None)
            
            self.changes_applied += 1
            msg = "IO_BOUND applied: nice=+5, policy=NORMAL, affinity=all_cores"
            logger.info(msg)
            return True, msg
        
        except PermissionError as e:
            return self._handle_permission_error("IO_BOUND", e)
        except Exception as e:
            return self._handle_general_error("IO_BOUND", e)
    
    def apply_idle(self) -> Tuple[bool, str]:
        """
        Apply idle workload scheduling (default):
          • Reset priority (nice = 0)
          • Use SCHED_NORMAL
          • Allow all CPU cores
        
        Returns
        -------
        tuple : (success: bool, message: str)
        """
        logger.info("Applying IDLE policy...")
        
        try:
            # Reset to defaults
            self._set_nice(0)
            self._set_policy(SCHED_NORMAL)
            self._set_cpu_affinity(None)
            
            self.changes_applied += 1
            msg = "IDLE applied: nice=0, policy=NORMAL, affinity=all_cores"
            logger.info(msg)
            return True, msg
        
        except PermissionError as e:
            return self._handle_permission_error("IDLE", e)
        except Exception as e:
            return self._handle_general_error("IDLE", e)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Low-level Control Operations
    # ──────────────────────────────────────────────────────────────────────────
    
    def _set_nice(self, nice_value: int) -> None:
        """
        Set process nice priority.
        
        Parameters
        ----------
        nice_value : int
            Nice value (-20 highest priority to +19 lowest priority)
        
        Raises
        ------
        PermissionError : if insufficient permissions
        """
        if self.simulation_mode:
            logger.debug(
                "[SIM] Would set nice(%d) for PID %d",
                nice_value,
                self.controlled_pid,
            )
            return
        
        try:
            if self.controlled_pid == os.getpid():
                # Use os.nice() for current process
                current_nice = os.nice(0)
                adjustment = nice_value - current_nice
                os.nice(adjustment)
                logger.debug("Set nice=%d (adjustment=%d)", nice_value, adjustment)
            else:
                # For other processes, would need psutil (requires root)
                if not _HAS_PSUTIL:
                    raise PermissionError("psutil required for controlling other PIDs")
                
                p = psutil.Process(self.controlled_pid)
                p.nice(nice_value)
                logger.debug("Set PID %d nice=%d", self.controlled_pid, nice_value)
        
        except PermissionError:
            raise
        except Exception as e:
            logger.error("Failed to set nice: %s", e)
            raise
    
    def _set_policy(self, policy: int) -> None:
        """
        Set process scheduling policy.
        
        Parameters
        ----------
        policy : int
            Scheduling policy constant (SCHED_NORMAL, SCHED_FIFO, etc.)
        
        Raises
        ------
        PermissionError : if insufficient permissions
        """
        if self.simulation_mode:
            logger.debug(
                "[SIM] Would set policy %s for PID %d",
                SCHED_NAMES.get(policy, policy),
                self.controlled_pid,
            )
            return
        
        try:
            # os.sched_setscheduler(pid, policy, param)
            # param is an os.sched_param(priority)
            param = os.sched_param(0)
            os.sched_setscheduler(self.controlled_pid, policy, param)
            logger.debug(
                "Set PID %d policy=%s",
                self.controlled_pid,
                SCHED_NAMES.get(policy, policy),
            )
        
        except (OSError, AttributeError) as e:
            if isinstance(e, OSError) and e.errno == errno.EPERM:
                raise PermissionError(f"sched_setscheduler: {e}") from e
            raise
    
    def _set_cpu_affinity(self, cpus: Optional[List[int]]) -> None:
        """
        Set process CPU affinity.
        
        Parameters
        ----------
        cpus : list of int, optional
            List of CPU indices to allow. If None, allow all CPUs.
        
        Raises
        ------
        PermissionError : if insufficient permissions
        """
        if self.simulation_mode:
            cpus_str = str(cpus) if cpus else "all"
            logger.debug(
                "[SIM] Would set affinity %s for PID %d",
                cpus_str,
                self.controlled_pid,
            )
            return
        
        try:
            if cpus is None:
                # Get all available CPUs
                try:
                    cpus = list(range(os.cpu_count() or 1))
                except Exception:
                    cpus = list(range(1))
            
            # os.sched_setaffinity(pid, cpus)
            os.sched_setaffinity(self.controlled_pid, set(cpus))
            logger.debug(
                "Set PID %d affinity to CPUs %s",
                self.controlled_pid,
                cpus,
            )
        
        except (OSError, AttributeError) as e:
            if isinstance(e, OSError) and e.errno == errno.EPERM:
                raise PermissionError(f"sched_setaffinity: {e}") from e
            raise
    
    # ──────────────────────────────────────────────────────────────────────────
    # Inspection & Context
    # ──────────────────────────────────────────────────────────────────────────
    
    def get_current_context(self) -> SchedulingContext:
        """
        Inspect current scheduling configuration of the process.
        
        Returns
        -------
        SchedulingContext : current scheduling state
        """
        try:
            # Get scheduling policy
            if hasattr(os, 'sched_getscheduler'):
                policy_int = os.sched_getscheduler(self.controlled_pid)
                policy_name = SCHED_NAMES.get(policy_int, f"UNKNOWN({policy_int})")
            else:
                policy_name = "UNAVAILABLE"
            
            # Get nice priority
            try:
                if self.controlled_pid == os.getpid():
                    nice = os.nice(0)
                else:
                    if _HAS_PSUTIL:
                        nice = psutil.Process(self.controlled_pid).nice()
                    else:
                        nice = None
            except Exception:
                nice = None
            
            # Get CPU affinity
            try:
                if hasattr(os, 'sched_getaffinity'):
                    affinity = list(os.sched_getaffinity(self.controlled_pid))
                else:
                    affinity = None
            except Exception:
                affinity = None
            
            return SchedulingContext(
                policy=policy_name,
                priority=0,  # SCHED_FIFO/RR priority (not typically used)
                nice=nice,
                cpu_affinity=affinity,
            )
        
        except Exception as e:
            logger.warning("Failed to get scheduling context: %s", e)
            return SchedulingContext(policy="UNKNOWN", priority=0, nice=None, cpu_affinity=None)
    
    def save_context(self) -> None:
        """Save current scheduling context for later restoration."""
        self._previous_context = self.get_current_context()
        logger.debug("Saved scheduling context: %s", self._previous_context)
    
    def restore_context(self) -> bool:
        """Restore previously saved scheduling context."""
        if self._previous_context is None:
            logger.warning("No saved context to restore")
            return False
        
        logger.info("Restoring scheduling context: %s", self._previous_context)
        try:
            if self._previous_context.nice is not None:
                self._set_nice(self._previous_context.nice)
            if self._previous_context.cpu_affinity:
                self._set_cpu_affinity(self._previous_context.cpu_affinity)
            return True
        except Exception as e:
            logger.error("Failed to restore context: %s", e)
            return False
    
    # ──────────────────────────────────────────────────────────────────────────
    # Error Handling
    # ──────────────────────────────────────────────────────────────────────────
    
    def _handle_permission_error(
        self,
        policy_name: str,
        error: PermissionError,
    ) -> Tuple[bool, str]:
        """Handle permission errors gracefully."""
        msg = f"{policy_name} DENIED: {error}"
        
        if self.auto_fallback:
            logger.warning(msg + " (falling back to simulation)")
            self.simulation_mode = True
            return False, msg
        else:
            logger.error(msg)
            raise
    
    def _handle_general_error(
        self,
        policy_name: str,
        error: Exception,
    ) -> Tuple[bool, str]:
        """Handle general errors."""
        msg = f"{policy_name} ERROR: {error}"
        
        if self.auto_fallback:
            logger.warning(msg + " (continuing)")
            return False, msg
        else:
            logger.error(msg)
            raise


# ──────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_cpu_count() -> int:
    """Get number of CPUs available."""
    return os.cpu_count() or 1


def is_running_as_root() -> bool:
    """Check if running with root/admin privileges."""
    if _IS_LINUX:
        return os.geteuid() == 0
    return False


def validate_environment() -> Tuple[bool, str]:
    """
    Validate that the environment supports scheduler control.
    
    Returns
    -------
    tuple : (is_valid: bool, message: str)
    """
    if not _IS_LINUX:
        return False, f"Not Linux (found: {platform.system()})"
    
    if _PYTHON_VERSION < (3, 10):
        return False, f"Python < 3.10 (found: {sys.version_info.major}.{sys.version_info.minor})"
    
    has_sched = hasattr(os, 'sched_setscheduler')
    has_affinity = hasattr(os, 'sched_setaffinity')
    
    if not (has_sched and has_affinity):
        return False, "Missing os.sched_* functions"
    
    if not is_running_as_root():
        return (
            True,
            "Non-root: some policies will require permissions (SCHED_FIFO/RR)",
        )
    
    return True, "Environment valid; running as root"
