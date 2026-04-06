"""
ASA: Adaptive Scheduler Assistant
==================================

TLDR: ML workload classifier that adapts Linux scheduling policies in real-time.

┌─────────────────────────────────────────────────────────────────────────────┐
│                              WHAT THIS DOES                                  │
└─────────────────────────────────────────────────────────────────────────────┘

1. Collects system metrics (CPU, memory, I/O) every 50ms
2. Engineers 9-10 features from raw metrics
3. Classifies workload: cpu_bound | io_bound | idle
4. Applies appropriate Linux scheduling policy
5. Logs predictions and switches to CSV
6. Generates 9 diagnostic visualizations

Result: Adaptive scheduling that matches workload type automatically.

┌─────────────────────────────────────────────────────────────────────────────┐
│                              OPERATING MODES                                 │
└─────────────────────────────────────────────────────────────────────────────┘

MODE 1: SIMULATION MODE (All Platforms) ⭐ Recommended for Testing
─────────────────────────────────────────────────────────────────────────────
  Where: Windows, macOS, Linux
  Command: python agent/asa_agent.py --duration 30 --sim
  Effect: Logs all decisions WITHOUT applying real system changes
  Permissions: None required
  Use Case: Testing, development, cross-platform validation

MODE 2: REAL CONTROL (Linux Only)
──────────────────────────────────────────────────────────────────────────────
  2A. As Root
      Command: sudo python agent/asa_agent.py --duration 30
      Effect: Real system calls, all policies available
      Permissions: Must run with sudo
  
  2B. With CAP_SYS_NICE Capability
      Setup: sudo setcap cap_sys_nice=ep $(which python3)
      Command: python agent/asa_agent.py --duration 30
      Effect: Real system calls, limited permissions
      Permissions: No sudo needed (one-time setup)
  
  2C. Graceful Fallback (Any Platform, Any Permission)
      Command: python agent/asa_agent.py --duration 30
      Effect: Tries real control; falls back to simulation on error
      Permissions: None required
      Use Case: Production deployment (auto-adapts to environment)

┌─────────────────────────────────────────────────────────────────────────────┐
│                          PLATFORM SUPPORT                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Linux      ✅ Full Support (real control + simulation)
macOS      ⚠️  Simulation only (no real system calls)
Windows    ⚠️  Simulation only (no real system calls)

Python: 3.10+ required (for os.sched_* functions)
Dependencies: See requirements.txt (+ optional: psutil)

┌─────────────────────────────────────────────────────────────────────────────┐
│                        HOW TO RUN (Quick Start)                             │
└─────────────────────────────────────────────────────────────────────────────┘

0. Install Dependencies (One-Time)
   ──────────────────────────────────────────────────────────────────────────
   pip install -r requirements.txt

1. Run Agent (30 seconds, Simulation Mode - Works Everywhere)
   ──────────────────────────────────────────────────────────────────────────
   python agent/asa_agent.py --duration 30 --sim
   
   Output:
   [INFO] ASA agent starting [SIMULATION mode].  duration=30s  poll=50ms
   [INFO] SCHEDULER SWITCH  idle → cpu_bound  |  policy=throughput_scheduler
   [INFO] Agent finished.  Total switches: 5  Final sched: io_bound
   [INFO] Agent run log saved → data/agent_run_log.csv

2. Generate Visualizations (Plot all statistics)
   ──────────────────────────────────────────────────────────────────────────
   python visualization/plot_all.py
   
   Output: 9 plots saved to figures/ directory
   
3. View Results
   ──────────────────────────────────────────────────────────────────────────
   ls figures/
   cat data/agent_run_log.csv  # Detailed log of every decision

┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCHEDULING POLICIES (What Gets Applied)                  │
└─────────────────────────────────────────────────────────────────────────────┘

POLICY 1: CPU_BOUND (Compute-Intensive Workloads)
──────────────────────────────────────────────────────────────────────────────
  When Applied: High CPU usage, low I/O
  System Changes:
    • nice = -5 (higher priority)
    • scheduler = SCHED_BATCH (long time slices, reduced context switching)
    • CPU affinity = all cores
  Effect: Faster computation, fewer context switches
  Example: Matrix operations, sorting, ML inference

POLICY 2: IO_BOUND (I/O-Intensive Workloads)
──────────────────────────────────────────────────────────────────────────────
  When Applied: Heavy I/O operations (disk, network, database)
  System Changes:
    • nice = +5 (lower priority)
    • scheduler = SCHED_NORMAL (responsive, short time slices)
    • CPU affinity = all cores
  Effect: Quick wake-up on I/O completion, responsive to events
  Example: File operations, network calls, database queries

POLICY 3: IDLE (Background/Minimal Workload)
──────────────────────────────────────────────────────────────────────────────
  When Applied: System mostly idle
  System Changes:
    • nice = 0 (default priority)
    • scheduler = SCHED_NORMAL
    • CPU affinity = all cores
  Effect: Standard Linux scheduling
  Example: Monitoring, background tasks

┌─────────────────────────────────────────────────────────────────────────────┐
│                     OUTPUTS: VISUALIZATIONS & STATS                         │
└─────────────────────────────────────────────────────────────────────────────┘

DATA FILES CREATED:
──────────────────────────────────────────────────────────────────────────────
data/agent_run_log.csv  – Detailed log, every decision:
                         timestamp, cpu_percent, mem_percent,
                         raw_pred, voted_pred, scheduler_switches, active_sched

PLOTS GENERATED (9 Total):
──────────────────────────────────────────────────────────────────────────────

1. agent_predictions_timeline.png
   What: Agent predictions over execution time
   Shows: Every prediction colored by class (cpu_bound=red, io_bound=blue, idle=green)
   Reveals: How often predictions change, classification stability

2. agent_switches_timeline.png
   What: Scheduler policy switches with annotations
   Shows: When switches occurred + which scheduler was activated (with stars)
   Reveals: Switch frequency, policy distribution, decision patterns

3. agent_prediction_agreement.png
   What: Individual classifier vs majority voting consensus
   Shows: Confusion matrix of raw_pred vs voted_pred
   Reveals: Voting effectiveness, classifier agreement level

4. agent_resource_trace.png
   What: CPU & memory usage colored by predicted workload
   Shows: System resource consumption with prediction overlay
   Reveals: Correlation between metrics and predictions

5. cpu_usage_over_time.png
   What: Raw CPU usage timeline
   Shows: CPU % trend throughout execution
   Reveals: Workload intensity distribution

6. mem_usage_over_time.png
   What: Raw memory usage timeline
   Shows: Memory % trend throughout execution
   Reveals: Memory pressure throughout run

7. confusion_matrix.png
   What: Prediction vs actual labels
   Shows: Normalized confusion matrix with counts
   Reveals: Per-class accuracy (cpu_bound F1, io_bound F1, idle F1)

8. feature_importance.png
   What: Which features drive predictions
   Shows: Ranked feature importance from Random Forest
   Reveals: Most influential metrics (CPU, memory, disk I/O patterns)

9. workload_phase_timeline.png
   What: Ground truth workload phases
   Shows: Actual class labels over time (compared to predictions)
   Reveals: Dataset label distribution

STATISTICS PRINTED TO CONSOLE:
──────────────────────────────────────────────────────────────────────────────
from evaluation/evaluate_model.py:
  • Test Accuracy: 90%
  • Macro F1 Score: 0.90
  • Per-Class Precision/Recall/F1: cpu_bound, io_bound, idle
  • Latency: ms per prediction, total overhead

from agent/asa_agent.py:
  • Total Scheduler Switches: N
  • Active Scheduling Class: (current policy)
  • Policy Distribution: % on cpu_bound, io_bound, idle

┌─────────────────────────────────────────────────────────────────────────────┐
│                      TESTING SCENARIOS                                       │
└─────────────────────────────────────────────────────────────────────────────┘

QUICK TEST (2 minutes)
──────────────────────────────────────────────────────────────────────────────
python agent/asa_agent.py --duration 10 --sim && python visualization/plot_all.py

FULL VALIDATION (10 minutes)
──────────────────────────────────────────────────────────────────────────────
python training/train_model.py --data data/features_dataset.csv
python evaluation/evaluate_model.py
python agent/asa_agent.py --duration 30 --sim
python visualization/plot_all.py

LINUX REAL CONTROL (with root)
──────────────────────────────────────────────────────────────────────────────
sudo python agent/asa_agent.py --duration 30
# Verify actual scheduling changes:
ps -p $$ -o pid,nice,cmd
taskset -p $$

SCHEDULER CONTROL EXAMPLES (Detailed Testing)
──────────────────────────────────────────────────────────────────────────────
python example_scheduler_control.py --example 1  # Real control (Linux)
python example_scheduler_control.py --example 3  # Pure simulation
python example_scheduler_control.py --example 4  # Router simulation
python example_scheduler_control.py --example 7  # Agent integration

┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMMAND REFERENCE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

AGENT:
  python agent/asa_agent.py [--duration SECONDS] [--poll INTERVAL] [--sim]
  
  Options:
    --duration   How long to run (default 60s, -1 = forever)
    --poll       Polling interval (default 0.05s = 50ms)
    --sim        Simulation mode (safe, works everywhere)

VISUALIZATION:
  python visualization/plot_all.py
  (generates 9 plots to figures/ directory)

EVALUATION:
  python evaluation/evaluate_model.py
  (reports accuracy, F1, per-class metrics)

TRAINING:
  python training/train_model.py --data <csv_file>
  (trains Random Forest classifier)

DATA COLLECTION:
  python experiments/collect_dataset.py
  (creates features_dataset.csv from system metrics)

EXAMPLES:
  python example_scheduler_control.py --example N
  (7 examples: 1=real, 2=router, 3=sim, 4=router-sim, 5=perms, 6=inspect, 7=agent)

┌─────────────────────────────────────────────────────────────────────────────┐
│                      FILE STRUCTURE                                          │
└─────────────────────────────────────────────────────────────────────────────┘

CORE CODE:
  agent/asa_agent.py                    – Main agent loop
  scheduler/scheduler_router.py         – Routing decisions
  scheduler/scheduler_control.py        – Real Linux control (NEW)
  models/model_loader.py                – Model artifacts
  features/feature_engineering.py       – Feature extraction
  collector/metric_collector.py         – System metrics
  agent/majority_voting.py              – Voting consensus

SUPPORTING:
  training/train_model.py               – Train classifier
  evaluation/evaluate_model.py          – Test accuracy
  visualization/plot_all.py             – Generate plots
  experiments/collect_dataset.py        – Data collection

DATA:
  data/features_dataset.csv             – Raw dataset
  data/X_train.npy, X_test.npy          – Training/test splits
  data/y_train.npy, y_test.npy          – Labels
  data/agent_run_log.csv                – Agent execution log
  models/rf_model.pkl                   – Trained classifier
  models/scaler.pkl                     – StandardScaler artifact
  figures/*.png                         – Visualizations (9 plots)

DOCUMENTATION:
  README.md                             – Start here (project overview)
  COMPLETE_GUIDE.md                     – This file (modes, commands, outputs)

┌─────────────────────────────────────────────────────────────────────────────┐
│                      QUICK REFERENCE TABLE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────┬──────────┬─────────────────┬─────────────────────────┐
│ Use Case           │ Platform │ Command         │ Permissions             │
├────────────────────┼──────────┼─────────────────┼─────────────────────────┤
│ Quick test         │ All      │ --sim           │ None                    │
│ Real control       │ Linux    │ (no flags)      │ Root or CAP_SYS_NICE    │
│ Graceful fallback  │ All      │ (no flags)      │ None (auto-degrades)    │
│ Development        │ All      │ --sim           │ None                    │
│ Production         │ Linux    │ (no flags)      │ Root or CAP_SYS_NICE    │
│ Cross-platform     │ All      │ --sim           │ None                    │
└────────────────────┴──────────┴─────────────────┴─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXPECTED OUTPUTS                                        │
└─────────────────────────────────────────────────────────────────────────────┘

AGENT LOG (Console):
  [INFO] ASA agent starting [SIMULATION mode].  duration=30s  poll=50ms
  [INFO] Model expects 10 features from StandardScaler
  [INFO] SCHEDULER SWITCH  idle → cpu_bound  |  policy=throughput_scheduler  [SUCCESS]
  [INFO] SCHEDULER SWITCH  cpu_bound → io_bound  |  policy=io_aware_scheduler [SUCCESS]
  [INFO] Agent finished.  Total switches: 5  Final sched: idle
  [INFO] Agent run log saved → data/agent_run_log.csv

PLOTS CREATED:
  ✓ agent_predictions_timeline.png
  ✓ agent_switches_timeline.png
  ✓ agent_prediction_agreement.png
  ✓ agent_resource_trace.png
  ✓ cpu_usage_over_time.png
  ✓ mem_usage_over_time.png
  ✓ confusion_matrix.png
  ✓ feature_importance.png
  ✓ workload_phase_timeline.png

CSV LOG (data/agent_run_log.csv):
  timestamp,cpu_percent,mem_percent,raw_pred,voted_pred,active_sched,switched
  2024-01-15 10:30:45.123,45.2,32.1,cpu_bound,cpu_bound,cfs,False
  2024-01-15 10:30:45.173,48.9,31.8,cpu_bound,cpu_bound,cfs,False
  2024-01-15 10:30:45.223,52.3,32.5,io_bound,cpu_bound,cfs,False
  ...

═════════════════════════════════════════════════════════════════════════════
🚀 QUICK START: python agent/asa_agent.py --duration 30 --sim && python visualization/plot_all.py
📊 DATA: Check data/agent_run_log.csv for detailed log
📈 PLOTS: 9 visualizations in figures/ directory
═════════════════════════════════════════════════════════════════════════════
"""
