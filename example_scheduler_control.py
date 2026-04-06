"""
example_scheduler_control.py
------------------------------
Example usage of the real Linux scheduler control system.

Demonstrates:
  1. Basic scheduler switching with real control
  2. Simulation mode (for testing)
  3. Permission handling
  4. Context inspection
  5. Integration with workload classification
  6. Error handling and fallback

Run:
    # Real control (requires root on most systems)
    import sys
    if sys.platform == "linux":
        python example_scheduler_control.py --real
    
    # Simulation (always works)
    python example_scheduler_control.py --sim
    
    # With specific PID
    python example_scheduler_control.py --sim --pid 1234
"""

import argparse
import logging
import sys
import time
from typing import List

# ── Configure logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)8s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Import modules ────────────────────────────────────────────────────────────
try:
    from scheduler.scheduler_router import SchedulerRouter
    from scheduler.scheduler_control import (
        SchedulerControl,
        validate_environment,
        is_running_as_root,
        get_cpu_count,
    )
except ImportError as e:
    logger.error("Failed to import scheduler modules: %s", e)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Example 1: Basic Real Control
# ──────────────────────────────────────────────────────────────────────────────

def example_real_control():
    """Example of applying real scheduling policies."""
    logger.info("=== Example 1: Real Scheduler Control ===")
    
    # Validate environment
    is_valid, msg = validate_environment()
    logger.info("Environment: %s", msg)
    
    if not is_valid:
        logger.warning("Skipping real control example (not Linux or missing functions)")
        return
    
    if not is_running_as_root():
        logger.warning("Not running as root. Some policies may require elevated privileges.")
    
    # Create control instance
    control = SchedulerControl(simulation_mode=False, auto_fallback=True)
    
    # Inspect current context
    logger.info("Initial scheduling context: %s", control.get_current_context())
    
    # Apply each policy
    logger.info("\n-- Applying CPU_BOUND policy --")
    success, msg = control.apply_cpu_bound()
    logger.info("Result: %s", msg)
    logger.info("Context after CPU_BOUND: %s", control.get_current_context())
    
    time.sleep(1)
    
    logger.info("\n-- Applying IO_BOUND policy --")
    success, msg = control.apply_io_bound()
    logger.info("Result: %s", msg)
    logger.info("Context after IO_BOUND: %s", control.get_current_context())
    
    time.sleep(1)
    
    logger.info("\n-- Applying IDLE policy --")
    success, msg = control.apply_idle()
    logger.info("Result: %s", msg)
    logger.info("Context after IDLE: %s", control.get_current_context())
    
    # Restore original context
    logger.info("\n-- Restoring original context --")
    restored = control.restore_context()
    logger.info("Restoration %s", "successful" if restored else "failed")


# ──────────────────────────────────────────────────────────────────────────────
# Example 2: Router with Real Control
# ──────────────────────────────────────────────────────────────────────────────

def example_router_real_control():
    """Example of using SchedulerRouter with real control."""
    logger.info("=== Example 2: Router with Real Control ===")
    
    # Create router with real control enabled
    router = SchedulerRouter(
        enable_real_control=True,
        simulation_mode=False,
        pid=None,  # Current process
    )
    
    # Simulate workload predictions
    predictions = ["idle", "cpu_bound", "cpu_bound", "io_bound", "idle", "cpu_bound"]
    
    logger.info("Simulating %d workload predictions...", len(predictions))
    
    for i, pred in enumerate(predictions):
        logger.info(f"\n-- Prediction {i+1}: {pred} --")
        switched = router.maybe_switch(pred)
        
        if switched:
            logger.info("✓ Scheduler switched to %s", pred)
            context = router.get_context()
            if context:
                logger.info("Current context: %s", context)
        else:
            logger.info("✗ No switch (cooldown or same policy)")
        
        time.sleep(0.3)
    
    # Print summary
    logger.info("\n=== Router Summary ===")
    summary = router.summary()
    logger.info("Total switches: %d", summary["total_switches"])
    logger.info("Active class: %s", summary["active_class"])
    logger.info("Real control enabled: %s", summary["real_control_enabled"])
    logger.info("Simulation mode: %s", summary["simulation_mode"])


# ──────────────────────────────────────────────────────────────────────────────
# Example 3: Simulation Mode
# ──────────────────────────────────────────────────────────────────────────────

def example_simulation():
    """Example using simulation mode (works on any platform)."""
    logger.info("=== Example 3: Simulation Mode ===")
    
    control = SchedulerControl(simulation_mode=True)
    logger.info("Created SchedulerControl in simulation mode")
    
    logger.info("\nApplying CPU_BOUND (simulated)...")
    success, msg = control.apply_cpu_bound()
    logger.info("Result: %s", msg)
    
    logger.info("\nApplying IO_BOUND (simulated)...")
    success, msg = control.apply_io_bound()
    logger.info("Result: %s", msg)
    
    logger.info("\nApplying IDLE (simulated)...")
    success, msg = control.apply_idle()
    logger.info("Result: %s", msg)


# ──────────────────────────────────────────────────────────────────────────────
# Example 4: Router Simulation
# ──────────────────────────────────────────────────────────────────────────────

def example_router_simulation():
    """Example of router in simulation mode."""
    logger.info("=== Example 4: Router in Simulation Mode ===")
    
    router = SchedulerRouter(
        enable_real_control=True,
        simulation_mode=True,  # Force simulation
    )
    
    # Test multiple switches with cooldown
    logger.info("Testing scheduler switches with cooldown...\n")
    
    workload_sequence = [
        "cpu_bound",
        "cpu_bound",  # Should not switch (same policy)
        "io_bound",
        "io_bound",   # Should not switch (same policy)
        "cpu_bound",
        "idle",
    ]
    
    for pred in workload_sequence:
        logger.info(f"Attempting switch to: {pred}")
        switched = router.maybe_switch(pred)
        logger.info(f"  → {"Switched" if switched else "No switch"}\n")
        time.sleep(0.1)
    
    # Print detailed switch log
    logger.info("\n=== Detailed Switch Log ===")
    summary = router.summary()
    
    for ts, prev, target, success, msg in summary["switch_log"]:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(
            "  %s → %s  [%s]  %s",
            prev,
            target,
            status,
            msg,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Example 5: Permission Handling
# ──────────────────────────────────────────────────────────────────────────────

def example_permission_handling():
    """Example showing graceful permission error handling."""
    logger.info("=== Example 5: Permission Error Handling ===")
    
    # Create control with auto_fallback=True (graceful degradation)
    control = SchedulerControl(
        simulation_mode=False,
        auto_fallback=True,  # Gracefully fall back to simulation
    )
    
    logger.info("Attempting to apply policies with auto_fallback=True...")
    logger.info("If permission errors occur, will continue in simulation mode.\n")
    
    logger.info("Applying CPU_BOUND...")
    success, msg = control.apply_cpu_bound()
    logger.info(f"  Result: {msg}\n")
    
    logger.info("Applying IO_BOUND...")
    success, msg = control.apply_io_bound()
    logger.info(f"  Result: {msg}\n")
    
    logger.info("Current simulation mode: %s", control.simulation_mode)


# ──────────────────────────────────────────────────────────────────────────────
# Example 6: Process Inspection
# ──────────────────────────────────────────────────────────────────────────────

def example_inspection():
    """Example of inspecting scheduling configuration."""
    logger.info("=== Example 6: Scheduling Inspection ===")
    
    control = SchedulerControl(simulation_mode=False, auto_fallback=True)
    
    logger.info("Current scheduling context:")
    context = control.get_current_context()
    logger.info("  Policy: %s", context.policy)
    logger.info("  Nice: %s", context.nice)
    logger.info("  CPU Affinity: %s", context.cpu_affinity)
    logger.info("  CPUs available: %d", get_cpu_count())


# ──────────────────────────────────────────────────────────────────────────────
# Example 7: Integration with Agent
# ──────────────────────────────────────────────────────────────────────────────

def example_agent_integration():
    """Example showing integration with ML workload classification."""
    logger.info("=== Example 7: Agent Integration ===")
    
    # Simulate workload classification updates
    class_predictions = [
        ("cpu_bound", 0.92),
        ("cpu_bound", 0.88),
        ("io_bound", 0.85),
        ("io_bound", 0.78),
        ("cpu_bound", 0.91),
        ("idle", 0.95),
    ]
    
    router = SchedulerRouter(
        enable_real_control=True,
        simulation_mode=True,  # Use simulation for demo
    )
    
    logger.info("Simulating agent predictions and automatic scheduler switching:\n")
    
    for workload_class, confidence in class_predictions:
        logger.info(f"Agent: {workload_class} (confidence: {confidence:.0%})")
        
        switched = router.maybe_switch(workload_class)
        
        if switched:
            logger.info(
                "  ✓ Scheduler adapted to %s workload",
                workload_class,
            )
            context = router.get_context()
            if context:
                logger.info("  Active policy: %s", context.policy)
        else:
            logger.info("  - No change (cooldown or redundant)")
        
        time.sleep(0.2)
    
    logger.info("\nAgent Statistics:")
    logger.info(f"  Total switches: {router.summary()['total_switches']}")
    logger.info(f"  Current policy: {router.active_class}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Examples of Linux scheduler control"
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 8),
        default=4,
        help="Run specific example (1-7, default: 4)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Force real control (may fail on non-Linux)",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Force simulation mode",
    )
    parser.add_argument(
        "--pid",
        type=int,
        help="Control specific process ID",
    )
    
    args = parser.parse_args()
    
    # Choose appropriate example
    examples = {
        1: example_real_control,
        2: example_router_real_control,
        3: example_simulation,
        4: example_router_simulation,
        5: example_permission_handling,
        6: example_inspection,
        7: example_agent_integration,
    }
    
    try:
        logger.info("Python scheduler control examples")
        logger.info("=" * 60)
        examples[args.example]()
        logger.info("\n" + "=" * 60)
        logger.info("Example completed successfully")
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Example failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
