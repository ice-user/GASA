# Custom Dataset Training Guide

## Overview

The training pipeline has been updated to support custom datasets with flexible column mapping. The system can automatically detect common column name variations or use manual mapping.

## Quick Start

### Option 1: Auto Detection (Recommended for similar datasets)

```bash
python training/train_custom_dataset.py data/your_custom_dataset.csv
```

This will:
- Automatically detect and map common column name variations
- Train on your dataset
- Save the model to `models/rf_model.pkl`

### Option 2: Manual Column Mapping

If auto-detection doesn't work or you need specific mapping:

```bash
python training/train_custom_dataset.py data/your_dataset.csv \
    --map old_col_name standard_col_name old_col_name2 standard_col_name2
```

Example with your custom dataset:

```bash
python training/train_custom_dataset.py data/your_custom_dataset.csv \
    --map cpu_usage cpu_percent \
         mem_usage mem_percent \
         cpu_rolling_5 cpu_rolling_mean \
         mem_rolling_5 mem_rolling_mean
```

### Option 3: Direct train_model.py with arguments

```bash
python training/train_model.py \
    --data data/your_dataset.csv \
    --column-map cpu_usage cpu_percent mem_usage mem_percent
```

## Supported Auto-Detection Mappings

The system automatically recognizes these common column variations:

| Variation Names | Maps to |
|---|---|
| `cpu_usage`, `cpu`, `cpu_load`, `cpu_util`, `cpu_core_avg` | `cpu_percent` |
| `mem_usage`, `memory`, `mem`, `memory_percent`, `mem_pressure` | `mem_percent` |
| `cpu_rolling_5`, `cpu_rolling_10`, `cpu_rolling_avg` | `cpu_rolling_mean` |
| `mem_rolling_5`, `mem_rolling_10`, `mem_rolling_avg` | `mem_rolling_mean` |

## Required Dataset Format

Your CSV must contain these 9 features (in any column order):

1. **cpu_percent** - CPU usage percentage (0-100)
2. **mem_percent** - Memory usage percentage (0-100)
3. **disk_read_bytes** - Disk read bytes (cumulative)
4. **disk_write_bytes** - Disk write bytes (cumulative)
5. **cpu_rolling_mean** - Rolling average of CPU over window (default 10 samples)
6. **cpu_rolling_std** - Rolling std dev of CPU over window (can be 0 if not available)
7. **mem_rolling_mean** - Rolling average of memory over window
8. **disk_read_diff** - Change in disk reads since last sample
9. **disk_write_diff** - Change in disk writes since last sample

And must have:
- **label** column with class labels (e.g., "idle", "cpu_bound", "io_bound")

## Your Custom Dataset

Your dataset has these columns:
```
timestamp, cpu_usage, cpu_core_avg, mem_usage, mem_pressure, 
cpu_rolling_5, mem_rolling_5, disk_read_diff, disk_write_diff, label
```

To train with it, use:

```bash
python training/train_custom_dataset.py data/features_dataset.csv
```

The system will auto-detect:
- `cpu_usage` → `cpu_percent`
- `mem_usage` → `mem_percent`
- `cpu_rolling_5` → `cpu_rolling_mean`
- `mem_rolling_5` → `mem_rolling_mean`

**Note:** If `disk_read_bytes` and `disk_write_bytes` are missing, you may need to compute them or the training will fail.

## Training Process

1. **Load**: Reads your CSV file
2. **Map**: Applies column name mappings
3. **Validate**: Checks all required columns exist
4. **Split**: 80% train / 20% test (stratified by label)
5. **Scale**: Normalizes features using StandardScaler
6. **Balance**: Applies SMOTE to balance class distribution
7. **Train**: Trains RandomForest with cross-validation
8. **Save**: Saves model and scaler to `models/` directory

## Handling Missing Columns

If your dataset is missing some computed features:

### Missing `disk_read_bytes` and `disk_write_bytes`?

Add them as cumulative sums or constants if not available:

```python
import pandas as pd

df = pd.read_csv("your_dataset.csv")
if "disk_read_bytes" not in df.columns:
    df["disk_read_bytes"] = df.index * 1024  # or compute from disk_read_diff
if "disk_write_bytes" not in df.columns:
    df["disk_write_bytes"] = df.index * 512
df.to_csv("your_dataset_fixed.csv", index=False)
```

### Missing `cpu_rolling_std`?

The system automatically fills it with 0 if missing. Or compute it:

```python
df["cpu_rolling_std"] = df["cpu_percent"].rolling(window=10).std()
```

## Troubleshooting

### Error: "Missing required columns after mapping"

**Solution**: Check your CSV columns and provide correct mapping:

```bash
# First, see what columns you have
python -c "import pandas as pd; print(pd.read_csv('data/your_dataset.csv').columns.tolist())"

# Then use those column names in the mapping
python training/train_custom_dataset.py data/your_dataset.csv \
    --map your_col_name standard_name ...
```

### Error: "label column not found"

**Solution**: Your CSV must have a class label column. Rename it if needed:

```python
df = pd.read_csv("your_dataset.csv")
df.rename(columns={"class": "label"}, inplace=True)
df.to_csv("your_dataset.csv", index=False)
```

### Class imbalance warning

This is normal. SMOTE will balance the classes during training.

## Expected Output

```
2026-03-31 12:34:56  INFO  Loading dataset from data/features_dataset.csv
2026-03-31 12:34:56  INFO  Raw shape: (5000, 10)
2026-03-31 12:34:56  INFO  Auto-detecting column mapping...
2026-03-31 12:34:56  INFO    cpu_usage → cpu_percent
2026-03-31 12:34:56  INFO    mem_usage → mem_percent
2026-03-31 12:34:56  INFO  Training RandomForestClassifier  params=...
2026-03-31 12:34:58  INFO  Training complete.
2026-03-31 12:34:59  INFO  CV F1 scores : [0.87 0.89 0.88 0.86 0.87]
2026-03-31 12:34:59  INFO  Mean CV F1   : 0.874 ± 0.010
2026-03-31 12:34:59  INFO  Model saved → models/rf_model.pkl

✓ Training completed successfully!
  Model saved to: models/rf_model.pkl
```

## Advanced: Programmatic Usage

```python
from training.train_model import train

# Train with custom dataset and mapping
model, cv_scores = train(
    data_path="data/my_dataset.csv",
    column_mapping={
        "cpu_usage": "cpu_percent",
        "mem_usage": "mem_percent",
        "cpu_rolling_5": "cpu_rolling_mean",
        "mem_rolling_5": "mem_rolling_mean",
    }
)

print(f"Mean CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

## Evaluation on Custom Dataset

After training, evaluate on test set:

```bash
python evaluation/evaluate_model.py
```

This will show:
- Classification report (precision, recall, F1 per class)
- Confusion matrix
- Inference latency
- Agent loop overhead

---

**Need help?** Check the logs for detailed error messages showing exactly which columns are missing or mismatched.
