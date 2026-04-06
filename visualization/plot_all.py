"""
visualization/plot_all.py
--------------------------
Generates and saves all diagnostic plots:

  1. cpu_usage_over_time.png          – CPU % timeline from raw dataset
  2. mem_usage_over_time.png          – Memory % timeline
  3. workload_phase_timeline.png      – Predicted workload phase (colour-coded)
  4. confusion_matrix.png             – Normalised confusion matrix heatmap
  5. feature_importance.png           – Random Forest feature importances

Run:
    python visualization/plot_all.py
"""

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.model_loader import load_artifacts
from features.feature_engineering import FEATURE_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_CSV = os.path.join(DATA_DIR, "features_dataset.csv")
AGENT_RUN_LOG = os.path.join(DATA_DIR, "agent_run_log.csv")

CLASS_COLORS = {
    "cpu_bound": "#E84040",
    "io_bound" : "#3A86FF",
    "idle"     : "#6FCF97",
}


# ─────────────────────────────────────────────────────────────────────────────
def load_raw():
    df = pd.read_csv(RAW_CSV)
    if "cpu_rolling_std" not in df.columns:
        df["cpu_rolling_std"] = 0.0
    return df


def find_column(df, keywords):
    """
    Find a column in df that matches any of the keywords (case-insensitive).
    
    Parameters
    ----------
    df : pandas.DataFrame
    keywords : list of str
        List of column names or substrings to search for
    
    Returns
    -------
    str or None
        The column name if found, None otherwise
    """
    df_cols_lower = {col.lower(): col for col in df.columns}
    for keyword in keywords:
        if keyword.lower() in df_cols_lower:
            return df_cols_lower[keyword.lower()]
    return None


# ─────────────────────────────────────────────────────────────────────────────
def plot_cpu(df):
    # Find CPU column dynamically
    cpu_col = find_column(df, ["cpu_percent", "cpu_usage", "cpu"])
    if cpu_col is None:
        logger.warning("CPU column not found. Skipping CPU plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df.index, df[cpu_col], color="#E84040", linewidth=0.8, alpha=0.85)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("CPU %")
    ax.set_title("CPU Utilisation Over Time")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cpu_usage_over_time.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_mem(df):
    # Find memory column dynamically
    mem_col = find_column(df, ["mem_percent", "mem_usage", "memory", "mem"])
    if mem_col is None:
        logger.warning("Memory column not found. Skipping memory plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df.index, df[mem_col], color="#3A86FF", linewidth=0.8, alpha=0.85)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Memory %")
    ax.set_title("Memory Utilisation Over Time")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "mem_usage_over_time.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_phase_timeline(df):
    """Colour each sample by its true label to show workload phases."""
    fig, ax = plt.subplots(figsize=(14, 2.5))
    for label, colour in CLASS_COLORS.items():
        mask = df["label"] == label
        ax.scatter(df.index[mask], [1] * mask.sum(),
                   c=colour, s=6, alpha=0.7, label=label)
    ax.set_yticks([])
    ax.set_xlabel("Sample index")
    ax.set_title("Workload Phase Timeline (True Labels)")
    patches = [mpatches.Patch(color=c, label=l) for l, c in CLASS_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "workload_phase_timeline.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    classes = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Normalised Confusion Matrix")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_feature_importance(model, feature_cols=None):
    # Use loaded feature columns if available, otherwise fall back to hardcoded names
    if feature_cols is None or len(feature_cols) != len(model.feature_importances_):
        feature_cols = FEATURE_NAMES[:len(model.feature_importances_)]
        logger.warning("Using %d feature names (may be truncated)", len(feature_cols))
    
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(importances.index, importances.values,
                   color="#3A86FF", edgecolor="white")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest Feature Importances")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_agent_predictions_timeline():
    """Plot agent's real-time predictions over execution duration."""
    if not os.path.exists(AGENT_RUN_LOG):
        logger.warning("Agent run log not found: %s", AGENT_RUN_LOG)
        return
    
    log_df = pd.read_csv(AGENT_RUN_LOG)
    if log_df.empty:
        logger.warning("Agent run log is empty")
        return
    
    # Convert timestamp to numeric index (seconds from start)
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], errors='coerce')
    log_df['elapsed_sec'] = (log_df['timestamp'] - log_df['timestamp'].iloc[0]).dt.total_seconds()
    
    # Map predictions to colors
    pred_colors = log_df['voted_pred'].map({'cpu_bound': '#E84040', 'io_bound': '#3A86FF', 'idle': '#6FCF97'})
    pred_numeric = log_df['voted_pred'].map({'cpu_bound': 2, 'io_bound': 1, 'idle': 0})
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot predictions as colored scatter
    scatter = ax.scatter(log_df['elapsed_sec'], pred_numeric, c=pred_colors, s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # Add horizontal lines for each class
    for idx, (val, label) in enumerate([(0, 'idle'), (1, 'io_bound'), (2, 'cpu_bound')]):
        ax.axhline(y=val, color='gray', linestyle='--', alpha=0.2, linewidth=1)
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['idle', 'io_bound', 'cpu_bound'])
    ax.set_xlabel("Elapsed Time (seconds)")
    ax.set_ylabel("Predicted Workload Class")
    ax.set_title("Agent's Real-Time Predictions During Execution")
    ax.grid(axis='x', alpha=0.3)
    ax.set_ylim(-0.5, 2.5)
    
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "agent_predictions_timeline.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_agent_switches_timeline():
    """Plot when agent switched schedulers during execution."""
    if not os.path.exists(AGENT_RUN_LOG):
        logger.warning("Agent run log not found: %s", AGENT_RUN_LOG)
        return
    
    log_df = pd.read_csv(AGENT_RUN_LOG)
    if log_df.empty:
        logger.warning("Agent run log is empty")
        return
    
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], errors='coerce')
    log_df['elapsed_sec'] = (log_df['timestamp'] - log_df['timestamp'].iloc[0]).dt.total_seconds()
    
    # Get switch events
    switches_df = log_df[log_df['switched'] == True].copy()
    
    if switches_df.empty:
        logger.info("No scheduler switches recorded")
        return
    
    scheduler_colors = {
        'cfs': '#E84040', 
        'rt': '#3A86FF', 
        'deadline': '#6FCF97'
    }
    switch_colors = switches_df['active_sched'].map(scheduler_colors)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot all predictions as light background
    pred_numeric = log_df['voted_pred'].map({'cpu_bound': 2, 'io_bound': 1, 'idle': 0})
    ax.scatter(log_df['elapsed_sec'], pred_numeric, c='lightgray', s=15, alpha=0.3, label='All predictions')
    
    # Highlight switches
    switch_numeric = switches_df['voted_pred'].map({'cpu_bound': 2, 'io_bound': 1, 'idle': 0})
    scatter = ax.scatter(switches_df['elapsed_sec'], switch_numeric, c=switch_colors, s=150, 
                        alpha=0.8, marker='*', edgecolors='black', linewidths=1, label='Scheduler switch')
    
    # Add labels for switch events
    for idx, row in switches_df.iterrows():
        pred_val = {'cpu_bound': 2, 'io_bound': 1, 'idle': 0}[row['voted_pred']]
        ax.annotate(row['active_sched'], 
                   xy=(row['elapsed_sec'], pred_val), 
                   xytext=(0, 10), 
                   textcoords='offset points',
                   ha='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=scheduler_colors.get(row['active_sched'], 'white'), alpha=0.7))
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['idle', 'io_bound', 'cpu_bound'])
    ax.set_xlabel("Elapsed Time (seconds)")
    ax.set_ylabel("Predicted Workload Class")
    ax.set_title(f"Scheduler Switches During Agent Execution ({len(switches_df)} switches)")
    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_ylim(-0.5, 2.5)
    
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "agent_switches_timeline.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s with %d switches", out, len(switches_df))


def plot_agent_prediction_agreement():
    """Plot agreement between raw predictions and majority voting."""
    if not os.path.exists(AGENT_RUN_LOG):
        logger.warning("Agent run log not found: %s", AGENT_RUN_LOG)
        return
    
    log_df = pd.read_csv(AGENT_RUN_LOG)
    if log_df.empty:
        logger.warning("Agent run log is empty")
        return
    
    # Build confusion matrix: raw vs voted
    from sklearn.metrics import confusion_matrix
    
    class_labels = sorted(log_df['raw_pred'].unique().tolist() + log_df['voted_pred'].unique().tolist())
    cm = confusion_matrix(log_df['raw_pred'], log_df['voted_pred'], labels=class_labels)
    
    # Normalize
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", interpolation="nearest")
    
    # Add text annotations
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            text = ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                          ha="center", va="center", color="black" if cm_norm[i, j] < 0.5 else "white",
                          fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Proportion")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Voted Prediction (Majority Vote)")
    ax.set_ylabel("Raw Prediction (Individual Classifier)")
    ax.set_title("Voting Consensus: Individual vs Majority Vote")
    
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "agent_prediction_agreement.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_agent_resource_trace():
    """Plot CPU and memory resources with prediction overlays."""
    if not os.path.exists(AGENT_RUN_LOG):
        logger.warning("Agent run log not found: %s", AGENT_RUN_LOG)
        return
    
    log_df = pd.read_csv(AGENT_RUN_LOG)
    if log_df.empty:
        logger.warning("Agent run log is empty")
        return
    
    log_df['timestamp'] = pd.to_datetime(log_df['timestamp'], errors='coerce')
    log_df['elapsed_sec'] = (log_df['timestamp'] - log_df['timestamp'].iloc[0]).dt.total_seconds()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # CPU usage with prediction coloring
    for pred_class, color in CLASS_COLORS.items():
        mask = log_df['voted_pred'] == pred_class
        ax1.scatter(log_df[mask]['elapsed_sec'], log_df[mask]['cpu_percent'], 
                   label=pred_class, color=color, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    
    ax1.plot(log_df['elapsed_sec'], log_df['cpu_percent'], color='gray', alpha=0.3, linewidth=1)
    ax1.set_ylabel("CPU Usage (%)")
    ax1.set_title("Agent's Resource Monitoring During Execution")
    ax1.legend(loc='upper left', ncol=3, fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-5, 105)
    
    # Memory usage with prediction coloring
    for pred_class, color in CLASS_COLORS.items():
        mask = log_df['voted_pred'] == pred_class
        ax2.scatter(log_df[mask]['elapsed_sec'], log_df[mask]['mem_percent'],
                   label=pred_class, color=color, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    
    ax2.plot(log_df['elapsed_sec'], log_df['mem_percent'], color='gray', alpha=0.3, linewidth=1)
    ax2.set_xlabel("Elapsed Time (seconds)")
    ax2.set_ylabel("Memory Usage (%)")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-5, 105)
    
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "agent_resource_trace.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    model, scaler, feature_cols = load_artifacts()
    df = load_raw()

    # ── Generate predictions on FRESH raw data (not saved test split) ────
    # Select numeric columns that actually exist in raw data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" in numeric_cols:
        numeric_cols.remove("label")
    
    # Use feature columns from training, but only those that exist in raw data
    if feature_cols:
        cols_to_use = [col for col in feature_cols if col in df.columns]
        # If we're missing disk_read/write_bytes but have the diffs, compute them
        if "disk_read_bytes" not in cols_to_use and "disk_read_diff" in df.columns:
            df["disk_read_bytes"] = df["disk_read_diff"].fillna(0).cumsum()
            cols_to_use.append("disk_read_bytes")
        if "disk_write_bytes" not in cols_to_use and "disk_write_diff" in df.columns:
            df["disk_write_bytes"] = df["disk_write_diff"].fillna(0).cumsum()
            cols_to_use.append("disk_write_bytes")
    else:
        cols_to_use = numeric_cols
    
    logger.info("Using %d features: %s", len(cols_to_use), cols_to_use)
    
    # Extract features from raw data
    X_raw = df[cols_to_use].values.astype(np.float64)
    
    # Handle feature count mismatch (pad if needed)
    n_expected = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else len(cols_to_use)
    if X_raw.shape[1] < n_expected:
        X_raw = np.pad(X_raw, ((0, 0), (0, n_expected - X_raw.shape[1])), mode='constant')
        logger.info("Padded raw data from %d to %d features", X_raw.shape[1] - (n_expected - X_raw.shape[1]), n_expected)
    
    # Scale and predict
    X_raw_scaled = scaler.transform(X_raw)
    y_pred = model.predict(X_raw_scaled)
    
    # Get true labels from raw data
    y_true = df["label"].values if "label" in df.columns else None

    plot_cpu(df)
    plot_mem(df)
    plot_phase_timeline(df)
    
    # Use fresh predictions on raw data, not saved test split
    if y_true is not None:
        plot_confusion_matrix(model, X_raw_scaled, y_true)
    else:
        logger.warning("No label column found. Skipping confusion matrix.")
    
    plot_feature_importance(model, feature_cols)
    
    # ── Plot agent runtime behavior (if log exists) ────
    plot_agent_predictions_timeline()
    plot_agent_switches_timeline()
    plot_agent_prediction_agreement()
    plot_agent_resource_trace()

    logger.info("All plots saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
