# General Adaptive Scheduling Agent (ASA)

A machine-learning-based scheduler router that observes system telemetry,
classifies the current workload type (cpu_bound / io_bound / idle), and
dynamically selects an appropriate scheduling policy.

## Project Structure

```
asa/
├── collector/          # System metric collection (psutil)
├── features/           # Feature engineering & rolling stats
├── training/           # Dataset builder + model trainer
├── models/             # Saved model artifacts
├── agent/              # Real-time ASA agent loop
├── scheduler/          # Adaptive scheduler router + real Linux control
├── evaluation/         # Accuracy, F1, latency benchmarks
├── visualization/      # Plots: CPU, memory, confusion matrix, agent behavior
├── experiments/        # Reproducibility scripts
└── data/               # Raw and processed datasets
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect data (or use existing dataset)
```bash
python experiments/collect_dataset.py
```

### 3. Train the model
```bash
python training/train_model.py --data data/features_dataset.csv
```

### 4. Evaluate the model
```bash
python evaluation/evaluate_model.py
```

### 5. Run the live ASA agent (Simulation Mode - Works Everywhere)
```bash
python agent/asa_agent.py --duration 30 --sim
```

### 6. Run the live ASA agent (Real Control - Linux + Permissions Required)
```bash
# With root access
sudo python agent/asa_agent.py --duration 30

# With CAP_SYS_NICE capability
python agent/asa_agent.py --duration 30
```

### 7. Generate all plots
```bash
python visualization/plot_all.py
```

## Results Summary

| Metric              | Score  |
|---------------------|--------|
| Test Accuracy       | 90%    |
| Macro F1            | 0.90   |
| CV F1 (5-fold)      | 0.922 ± 0.013 |
| cpu_bound F1        | 0.90   |
| io_bound F1         | 0.90   |
| idle F1             | 0.90   |

## Hardware Tested On
- Intel Core Ultra 5 × 2 (Windows 11)
- Intel Core i7 8th Gen × 1 (Windows 11)
- Apple M4 × 1 (macOS)

---

## Testing Guide

### Running in Simulation Mode (Recommended for Testing)

Simulation mode works on **any platform** (Windows, macOS, Linux) and doesn't require permissions.
All scheduling operations are logged but not executed.

#### Quick Test
```bash
python agent/asa_agent.py --duration 30 --sim
```

Expected output:
```
[INFO] ASA agent starting [SIMULATION mode].  duration=30s  poll=50ms
[INFO] Model expects 10 features from StandardScaler
[INFO] SCHEDULER SWITCH  idle → cpu_bound  |  policy=throughput_scheduler  time_slice=20ms  [SUCCESS] (simulation)
[INFO] Agent finished.  Total switches: 5  Final sched: io_bound
[INFO] Agent run log saved → data/agent_run_log.csv
```

### Testing the Full Pipeline (Simulation Mode)

```bash
# 1. Run agent for 60 seconds in simulation
python agent/asa_agent.py --duration 60 --sim

# 2. Generate visualizations
python visualization/plot_all.py

# 3. Plots created:
#   - figures/agent_predictions_timeline.png      (Agent's predictions over time)
#   - figures/agent_switches_timeline.png         (Scheduler switches with events)
#   - figures/agent_prediction_agreement.png      (Voting consensus analysis)
#   - figures/agent_resource_trace.png            (CPU & memory with predictions)
#   - figures/cpu_usage_over_time.png             (Raw CPU telemetry)
#   - figures/mem_usage_over_time.png             (Raw memory telemetry)
#   - figures/confusion_matrix.png                (Fresh predictions)
#   - figures/feature_importance.png              (Model feature weights)
#   - figures/workload_phase_timeline.png         (True labels over time)
```

### Testing Scheduler Control Features (Simulation)

Testing the new scheduler control layer without real system changes:

```bash
# Example 1: Simulation + Router Integration
python example_scheduler_control.py --example 4 --sim

# Example 2: Inspection (CPU info, process state)
python example_scheduler_control.py --example 6

# Example 3: Agent Integration Pattern
python example_scheduler_control.py --example 7
```

### Testing on Linux with Real Control (Optional)

Real scheduling changes apply only on Linux with proper permissions:

#### Option A: Run as Root
```bash
sudo python agent/asa_agent.py --duration 30
# Real system calls executed; all policies available
```

#### Option B: Set CAP_SYS_NICE (Recommended)
```bash
# One-time setup
sudo setcap cap_sys_nice=ep $(which python3)

# Now run without sudo
python agent/asa_agent.py --duration 30
# Real system calls executed with limited permissions
```

#### Option C: Test Permission Handling
```bash
# As non-root user on Linux (graceful fallback)
python agent/asa_agent.py --duration 30
# Will print permission errors and continue in simulation mode
```

### Testing Environment Validation

Verify your environment supports scheduler control:

```bash
python -c "
from scheduler.scheduler_control import validate_environment, is_running_as_root
is_valid, msg = validate_environment()
print(f'Valid: {is_valid}')
print(f'Message: {msg}')
print(f'Running as root: {is_running_as_root()}')
"
```

Expected outputs:
- Linux + root: `"Environment valid; running as root"`
- Linux + user: `"Non-root: some policies will require permissions (SCHED_FIFO/RR)"`
- Windows/macOS: `"Not Linux (found: Windows)"` or similar

### Running Complete Test Suite

```bash
# 1. Environment check
python -c "from scheduler.scheduler_control import validate_environment; print(validate_environment()[1])"

# 2. Simulation examples (any OS)
python example_scheduler_control.py --example 3  # Pure simulation
python example_scheduler_control.py --example 4  # Router simulation
python example_scheduler_control.py --example 7  # Agent integration

# 3. Full pipeline test
python agent/asa_agent.py --duration 30 --sim
python visualization/plot_all.py

# 4. Evaluate model
python evaluation/evaluate_model.py

# 5. Check outputs
ls -lh figures/
ls -lh data/agent_run_log.csv
```

### Performance Validation

Measure the impact of scheduling policies:

```bash
# Baseline (default scheduling)
time python evaluation/evaluate_model.py

# With real control (Linux)
time sudo python evaluation/evaluate_model.py

# Expected: No significant difference (scheduler overhead << 1%)
```

## Key Features

### New: Real Linux Scheduler Control

The project now supports **actual Linux scheduling policy changes**:

- **CPU_BOUND**: Sets nice=-5, SCHED_BATCH, all CPUs
- **IO_BOUND**: Sets nice=+5, SCHED_NORMAL, all CPUs
- **IDLE**: Sets nice=0, SCHED_NORMAL, all CPUs

**System calls used**:
- `os.sched_setscheduler()` – Scheduling policy
- `os.sched_setaffinity()` – CPU binding
- `os.nice()` – Process priority

**Simulation Mode**: All operations logged without system calls (works on any OS)

### Documentation

New comprehensive documentation:
- [LINUX_SCHEDULER_CONTROL.md](LINUX_SCHEDULER_CONTROL.md) – Full technical reference
- [SCHEDULER_CONTROL_SETUP.md](SCHEDULER_CONTROL_SETUP.md) – Setup and integration guide
- [example_scheduler_control.py](example_scheduler_control.py) – 7 runnable examples

## Backward Compatibility

✅ **Fully backward compatible**:
- Original agent functionality unchanged
- All original plots still generated
- Existing workflows still work
- Simulation mode works on any OS
- Real control is opt-in (with --sim flag for simulation)

## Troubleshooting

### "PermissionError: Operation not permitted"
Run in simulation mode (no permissions needed):
```bash
python agent/asa_agent.py --duration 30 --sim
```

### "Not enough features/data"
Make sure dataset exists:
```bash
python experiments/collect_dataset.py
```

### "Cannot import module"
Verify package structure:
```bash
ls -la scheduler/
# Should have: __init__.py, scheduler_router.py, scheduler_control.py
```

### Plots not generating
Check that agent ran successfully:
```bash
ls -la data/agent_run_log.csv
```

## Next Steps

1. **Try the simulation mode**: `python agent/asa_agent.py --duration 30 --sim`
2. **Generate plots**: `python visualization/plot_all.py`
3. **Explore examples**: `python example_scheduler_control.py --help`
4. **Read docs**: See [LINUX_SCHEDULER_CONTROL.md](LINUX_SCHEDULER_CONTROL.md)
5. **Deploy on Linux**: Follow [SCHEDULER_CONTROL_SETUP.md](SCHEDULER_CONTROL_SETUP.md)

---

**Last Updated**: 2024
**Status**: Complete with real scheduler control + full backward compatibility

