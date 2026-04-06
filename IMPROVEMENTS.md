# ASA: 15 Proposed Improvements

---

## 1. eBPF-Based Metric Collection
**What:** Replace psutil with eBPF kernel probes (via bcc or libbpf) to capture
scheduler tracepoints, block I/O completions, and cache misses at sub-millisecond
resolution directly from the kernel.
**Why it helps:** psutil polls from userspace and adds ~1-5 ms latency. eBPF
captures events in-kernel with near-zero overhead and gives richer signals
(context-switch rate, run-queue depth) that psutil cannot provide.

---

## 2. sched_ext Kernel Integration (Linux 6.12+)
**What:** Replace the simulated SchedulerRouter with real sched_ext BPF programs
that implement the three policies (throughput, IO-aware, default) at the kernel level.
**Why it helps:** The current simulation only logs decisions; actual sched_ext
integration would produce measurable P99 latency improvements (as shown in related
work: 37–50% reduction).

---

## 3. Online Learning with Incremental Random Forest
**What:** Use scikit-learn's `HoeffdingTreeClassifier` (River library) or
`SGDClassifier` with partial_fit to update the model from live predictions.
**Why it helps:** The current model is static — trained once and frozen. On new
machines or workloads it has never seen, accuracy degrades. Online learning
adapts continuously without retraining from scratch.

---

## 4. LSTM / Temporal Classifier
**What:** Replace Random Forest with an LSTM or GRU network that takes the last
W=20 timesteps as input rather than point-in-time features.
**Why it helps:** Workload transitions are sequential. CPU usage at t=5 combined
with the trajectory from t=1..4 is far more informative than the value at t=5
alone. LSTMs capture this temporal dependency.

---

## 5. Confidence-Threshold Fallback
**What:** Use `predict_proba` and fall back to the current scheduler if
max(probability) < threshold (e.g. 0.70).
**Why it helps:** Prevents thrashing when the model is uncertain (e.g. during
workload transitions). Currently the model always picks a class even when all
three are equally likely, causing unnecessary switches.

---

## 6. Reinforcement Learning Policy Layer
**What:** Add an RL agent (e.g. PPO via Stable-Baselines3) on top of the
classifier. State = feature vector + current scheduler + recent switch history.
Action = keep / switch. Reward = negative P99 latency or throughput improvement.
**Why it helps:** The current rule (vote wins → switch) is hand-coded. RL learns
the optimal switch policy, including when NOT to switch even if the voted class
differs (e.g. brief transient spikes).

---

## 7. Multi-Machine Federated Training
**What:** Train a shared base model on each machine independently, then aggregate
weights using Federated Averaging (FedAvg).
**Why it helps:** Different hardware (M4 vs i7 8th Gen) has different baseline
metric distributions. Federated training lets each machine personalise the model
while sharing general knowledge.

---

## 8. Richer Feature Set
**What:** Add: context-switch rate, run-queue depth, LLC (Last Level Cache) miss
rate, network I/O bytes, per-core variance, and process count.
**Why it helps:** The current 9 features are mostly CPU/memory/disk totals. Cache
miss rate and context-switch frequency are strong discriminators between cpu_bound
and io_bound that the current feature set misses.

---

## 9. Anomaly / Unknown Class Detection
**What:** Train an Isolation Forest or use OCSVM as an outlier detector alongside
the classifier. If the sample is flagged as anomalous, fall back to the default
scheduler.
**Why it helps:** Prevents incorrect switches under never-before-seen workloads
(e.g. a video encoding task that looks like neither pure CPU nor pure IO).

---

## 10. Hyperparameter Tuning with Optuna
**What:** Replace the hard-coded RF params with an Optuna study that optimises
n_estimators, max_depth, min_samples_leaf, and SMOTE k_neighbors.
**Why it helps:** Current params were chosen manually. Automated tuning typically
recovers 1–3% accuracy and reduces overfitting on small datasets.

---

## 11. Real P99 Latency Measurement
**What:** Use a co-located latency probe (a simple ping-pong thread measuring
wakeup latency) while the agent runs, logging measured P99 per scheduler phase.
**Why it helps:** Currently all latency numbers are projected from related work.
Actual measured P99 per phase would be the most important experiment result for
the final report.

---

## 12. Per-Process Classification
**What:** Classify individual processes (by PID) rather than the whole system.
Use per-process CPU/IO stats from `/proc/[pid]/stat`.
**Why it helps:** The system-wide approach confounds multiple co-running processes.
Per-process classification enables fine-grained routing where different processes
get different scheduling policies simultaneously.

---

## 13. Docker / Reproducible Environment
**What:** Add a `Dockerfile` and `docker-compose.yml` so the full pipeline
(collect → train → evaluate → visualise) runs in a container.
**Why it helps:** Eliminates the "works on my machine" problem across the team's
4 different hardware/OS configurations.

---

## 14. Real-Time Dashboard
**What:** Add a Flask or FastAPI server that serves a live HTML dashboard showing
CPU/memory timelines, current scheduler phase, and switch events in real time
(using Server-Sent Events or WebSocket).
**Why it helps:** Makes the agent's behaviour visible and demystifies the
decision loop for presentation to a teacher or examiner.

---

## 15. Cross-Platform Scheduler Actions
**What:** On Windows, map scheduling policies to `SetPriorityClass` +
`SetThreadPriority` Win32 calls. On macOS, use `setpriority()`/`pthread_setschedparam`.
**Why it helps:** The current SchedulerRouter is simulation-only. Cross-platform
OS calls would make actual scheduling changes on the 4 test machines, turning
this into a real prototype rather than a pure ML project.
