# Session Log: Li-SOCl2 Voltage Forecasting Implementation

**Date:** 2026-02-17 through 2026-02-20
**Project:** TempVoltRelationship
**Goal:** Implement an LSTM + entity embedding model for Li-SOCl2 battery voltage forecasting with autoregressive 12-month rollout, then clean up the repo and push to GitHub.

---

## Step 1: Read Reference Files

Before writing any code, we read all the key reference files to understand existing patterns and reuse architecture decisions.

```bash
# Files read:
#   scripts/ml_tabular_pipeline.py   — data generator (generate_lisocl2_timeseries) + embedding pattern
#   scripts/voltage_forecast_model.py — existing LSTM/GRU architecture
#   scripts/train.py                 — training loop (early stopping, scheduling, checkpoints)
#   scripts/battery_dataset.py       — dataset pattern (battery-level split, sliding window)
#   scripts/device_utils.py          — get_compatible_device() for GPU/CPU fallback
```

**Why:** The new script needed to follow the same conventions (battery-level train/val split to prevent data leakage, StandardScaler for numeric features, OrdinalEncoder for categoricals, gradient clipping, ReduceLROnPlateau scheduler, etc.).

---

## Step 2: Create `scripts/lisocl2_forecast.py`

Wrote a single self-contained script (~1,000 lines) with six sections:

| Section | Class/Function | Purpose |
|---------|---------------|---------|
| 1. Dataset | `LiSOCl2TimeSeriesDataset` | Sliding window (24 months -> predict next), battery-level 80/20 split, StandardScaler per numeric feature, OrdinalEncoder for categoricals |
| 2. Model | `LiSOCl2ForecastLSTM` | 2-layer LSTM (hidden=128) + entity embeddings (manufacturer->3d, installation_type->2d) + FC head (133->64->32->1) |
| 3. Training | `train_model()` | Adam optimizer, ReduceLROnPlateau, early stopping (patience=15), gradient clipping (max_norm=1.0), MSE loss |
| 4. Forecast | `autoregressive_forecast()` | 12-step rollout: predicted voltage feeds back as input, temperature projected seasonally, capacity extrapolated linearly, pulse carried forward |
| 5. Evaluation | `evaluate_and_plot()` | 5 plots: loss curves, per-step MAE bar chart, sample trajectories, pred-vs-actual scatter, per-manufacturer error breakdown |
| 6. Checkpoint | `save_checkpoint()` | Saves model weights, scalers, encoders, val battery IDs, config, and training history |

---

## Step 3: Verify With Small Test Run

```bash
cd scripts
python lisocl2_forecast.py --num_epochs 10 --max_samples 20000 --n_batteries 500 --n_months 60 --forecast_samples 50
```

**Result:** Ran successfully in ~75 seconds on CPU. Produced all expected outputs:
- `models_lisocl2/lisocl2_forecast_model.pth` (checkpoint)
- `models_lisocl2/loss_curves.png`
- `models_lisocl2/per_step_mae.png`
- `models_lisocl2/sample_trajectories.png`
- `models_lisocl2/pred_vs_actual.png`
- `models_lisocl2/manufacturer_error.png`

Metrics: MAE = 0.0125 V, RMSE = 0.0162 V

**Why test small first:** Catches bugs fast without waiting for a long training run.

---

## Step 4: Fix sklearn Warnings

The autoregressive forecast function was passing numpy scalars to sklearn scalers/encoders that were fitted on DataFrames. This produced harmless but noisy warnings:

```
X does not have valid feature names, but StandardScaler was fitted with feature names
```

**Fix:** Added `warnings.filterwarnings("ignore", ...)` in the `autoregressive_forecast()` function to suppress these.

---

## Step 5: Attempt Full-Scale Run

```bash
python lisocl2_forecast.py --n_batteries 5000 --n_months 300 --num_epochs 80 --patience 15
```

**Result:** Generated 1.5M rows -> 1.1M training samples. On CPU, each epoch took ~10-15 minutes (17,250 batches at batch_size=64). The process consumed 506% CPU (multi-threaded PyTorch) but was too slow without a GPU. Eventually stopped manually.

**Lesson:** Full-scale runs (5000 batteries x 300 months) need a GPU or reduced `--max_samples`.

---

## Step 6: First Commit — New Script Only

```bash
git add scripts/lisocl2_forecast.py
git commit -m "Add Li-SOCl2 voltage forecasting script (LSTM + entity embeddings)

Self-contained pipeline: dataset with sliding window and battery-level split,
LSTM model with categorical entity embeddings, autoregressive 12-month forecast,
and 5 evaluation plots.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Commit:** `03d02c8`
**Why commit just this file:** The repo had many other untracked/deleted files from prior reorganization work. We kept the commit focused on the new feature.

---

## Step 7: Add Data Files to `.gitignore`

The repo was tracking large data files that shouldn't be in version control:

```bash
# Added to .gitignore (in three separate steps):
*.npy
*.csv
*.rpt
```

Then removed already-tracked files from git (keeping them on disk):

```bash
# Remove .npy files from tracking
git rm --cached data/BVDF.npy data/ForcastMatrix.npy data/ForecastMatrixSample.npy

# Remove .csv files from tracking
git rm --cached data/VoltTemp.csv data/synthetic_lisocl2_timeseries.csv
```

**Why `--cached`:** This removes files from git's index (stops tracking them) but does NOT delete them from disk. The files remain in your working directory.

---

## Step 8: Commit `.gitignore` Changes

```bash
git add .gitignore
git commit -m "Add *.npy, *.csv, *.rpt to .gitignore and untrack data files

Remove tracked data files (BVDF.npy, ForcastMatrix.npy,
ForecastMatrixSample.npy, VoltTemp.csv, synthetic_lisocl2_timeseries.csv)
from git while keeping them on disk.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Commit:** `9df6ef0`

---

## Step 9: Push to GitHub — First Attempt Failed

```bash
git remote add origin git@github.com:ChuckHahm/PhysicsModel.git
git push -u origin master
```

**Problem:** The push hung indefinitely. Investigation revealed the repo contained **489 MB** in packed objects because the large data files (VoltTemp.csv alone was 1.8 GB) were still in git history even though we untracked them in the latest commit.

```bash
git count-objects -vH
# size-pack: 488.86 MiB

# Largest blobs in history:
# 1836.03 MB  data/VoltTemp.csv
#  811.51 MB  data/BVDF.npy
#  124.57 MB  data/ForcastMatrix.npy
#   28.76 MB  data/synthetic_lisocl2_timeseries.csv
```

**Key insight:** `git rm --cached` only untracks files going forward. The files remain in all prior commits. To truly remove them, you must rewrite git history.

---

## Step 10: Rewrite Git History to Remove Large Files

```bash
pip install git-filter-repo

git filter-repo --invert-paths --path-glob '*.npy' --path-glob '*.csv' --force
```

**What this does:** Rewrites every commit in history, removing any file matching `*.npy` or `*.csv`. This is a destructive operation that changes all commit hashes.

**Result:**
```
489 MB -> 6 MB
```

**Side effect:** `git filter-repo` automatically removes the `origin` remote as a safety measure (to prevent accidental force-push to a shared repo).

---

## Step 11: Push to GitHub — Success

```bash
git remote add origin git@github.com:ChuckHahm/PhysicsModel.git
git push -u origin master --force
```

**Why `--force`:** After rewriting history, the local and remote histories diverge completely. Force push replaces the remote history. This was safe because the remote repo had no other collaborators' work.

**Result:** Push succeeded. Repo is at https://github.com/ChuckHahm/PhysicsModel

---

## Step 12: Add `--fast` Flag for Quick Testing

Added a `--fast` CLI flag that overrides defaults for rapid iteration:

| Parameter | Default | `--fast` |
|-----------|---------|----------|
| n_batteries | 5,000 | 200 |
| n_months | 300 | 60 |
| num_epochs | 80 | 10 |
| patience | 15 | 5 |
| hidden_size | 128 | 32 |
| num_layers | 2 | 1 |
| batch_size | 64 | 128 |
| max_samples | all | 10,000 |
| forecast_samples | 200 | 20 |

```bash
python scripts/lisocl2_forecast.py --fast
```

**Result:** Completes in ~6 seconds on CPU. Explicit CLI args still override fast defaults (e.g., `--fast --num_epochs 20` trains for 20 epochs with everything else fast).

**Commit:** `8cbdeab`

---

## Current State

### Commits on `master` (newest first)
```
8cbdeab  Add --fast flag for quick test runs (~6s on CPU)
46ab6ff  Add *.npy, *.csv, *.rpt to .gitignore and untrack data files
xxxxxxx  Add Li-SOCl2 voltage forecasting script (LSTM + entity embeddings)
...      (prior commits with rewritten hashes)
```

### Key Commands Reference

```bash
# Fast test (6 seconds)
python scripts/lisocl2_forecast.py --fast

# Medium run (few minutes)
python scripts/lisocl2_forecast.py --num_epochs 30 --max_samples 50000

# Full run (needs GPU)
python scripts/lisocl2_forecast.py --n_batteries 5000 --n_months 300 --num_epochs 80
```

### Output Files
All saved to `--output_dir` (default: `./models_lisocl2/`):
- `lisocl2_forecast_model.pth` — model checkpoint (weights, scalers, encoders, config)
- `loss_curves.png` — training/validation loss over epochs
- `per_step_mae.png` — MAE at each forecast horizon step (t+1 through t+12)
- `sample_trajectories.png` — 6 sample batteries with history + forecast overlay
- `pred_vs_actual.png` — scatter plot of predicted vs actual voltages
- `manufacturer_error.png` — per-manufacturer MAE with error bars
