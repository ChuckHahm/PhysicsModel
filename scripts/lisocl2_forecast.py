"""
=============================================================================
Li-SOCl2 Voltage Forecasting — LSTM + Entity Embeddings
=============================================================================

Self-contained script: dataset, model, training, autoregressive 12-month
forecast, and evaluation plots.

Uses generate_lisocl2_timeseries() from ml_tabular_pipeline.py to create
synthetic Li-SOCl2 battery data, then trains an LSTM with categorical
entity embeddings to predict next-month voltage.

Usage:
    python scripts/lisocl2_forecast.py --num_epochs 30 --max_samples 50000
    python scripts/lisocl2_forecast.py  # full training (80 epochs, all data)

Author: Chuck
=============================================================================
"""

import argparse
import logging
import os
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Device utils (must import before torch to filter CUDA warnings)
from device_utils import get_compatible_device

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml_tabular_pipeline import generate_lisocl2_timeseries

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lisocl2_forecast")


# =============================================================================
# SECTION 1 — DATASET
# =============================================================================

# Time-varying features fed to the LSTM (order matters)
TIME_FEATURES = [
    "voltage",
    "temperature",
    "cumulative_capacity_used_pct",
    "pulse_usage_min_per_day",
    "month_sin",
    "month_cos",
]
# Categorical features for entity embeddings
CAT_FEATURES = ["device_manufacturer", "installation_type"]
# Embedding config: (num_categories, embedding_dim)
CAT_EMBED_CONFIG = {
    "device_manufacturer": (5, 3),
    "installation_type": (2, 2),
}
TOTAL_EMB_DIM = sum(v[1] for v in CAT_EMBED_CONFIG.values())  # 5


def _add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add month_sin and month_cos from the month column."""
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


class LiSOCl2TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for Li-SOCl2 voltage forecasting.

    Each sample: 24 months of time-varying features -> predict next month voltage.
    Battery-level 80/20 train/val split to prevent leakage.

    __getitem__ returns:
        {"time_features": (seq_len, 6), "cat_features": (2,), "target": (1,)}
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 24,
        train: bool = True,
        train_split: float = 0.8,
        seed: int = 42,
        max_samples: Optional[int] = None,
        # Pass fitted encoders/scalers for val set
        feature_scalers: Optional[dict[str, StandardScaler]] = None,
        cat_encoders: Optional[dict[str, OrdinalEncoder]] = None,
        val_battery_ids: Optional[set] = None,
        train_battery_ids: Optional[set] = None,
    ):
        self.seq_len = seq_len
        self.train = train

        # Add cyclical month features
        df = _add_cyclical_month(df)

        # ── Battery-level split ──────────────────────────────────────────
        all_ids = df["battery_id"].unique()
        if train_battery_ids is not None and val_battery_ids is not None:
            self._train_ids = train_battery_ids
            self._val_ids = val_battery_ids
        else:
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(all_ids)
            n_train = int(len(shuffled) * train_split)
            self._train_ids = set(shuffled[:n_train])
            self._val_ids = set(shuffled[n_train:])

        split_ids = self._train_ids if train else self._val_ids
        df = df[df["battery_id"].isin(split_ids)].copy()
        logger.info(
            "%s set: %d batteries, %d rows",
            "Train" if train else "Val",
            len(split_ids),
            len(df),
        )

        # ── Fit / apply scalers ──────────────────────────────────────────
        if feature_scalers is None:
            self.feature_scalers = {}
            for col in TIME_FEATURES:
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]]).astype(np.float32)
                self.feature_scalers[col] = scaler
        else:
            self.feature_scalers = feature_scalers
            for col in TIME_FEATURES:
                df[col] = self.feature_scalers[col].transform(df[[col]]).astype(
                    np.float32
                )

        # ── Fit / apply categorical encoders ─────────────────────────────
        if cat_encoders is None:
            self.cat_encoders = {}
            for col in CAT_FEATURES:
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                df[col] = enc.fit_transform(df[[col]]).astype(np.int64).ravel()
                self.cat_encoders[col] = enc
        else:
            self.cat_encoders = cat_encoders
            for col in CAT_FEATURES:
                df[col] = (
                    self.cat_encoders[col]
                    .transform(df[[col]])
                    .astype(np.int64)
                    .ravel()
                )

        # ── Build sliding-window samples ─────────────────────────────────
        self.samples: list[dict] = []
        for _bid, grp in df.groupby("battery_id"):
            grp = grp.sort_values("month").reset_index(drop=True)
            if len(grp) < seq_len + 1:
                continue

            time_arr = grp[TIME_FEATURES].values.astype(np.float32)
            cat_arr = grp[CAT_FEATURES].values.astype(np.int64)
            # Target is the raw (scaled) voltage column
            volt_idx = TIME_FEATURES.index("voltage")

            for i in range(len(grp) - seq_len):
                self.samples.append(
                    {
                        "time_features": time_arr[i : i + seq_len],
                        "cat_features": cat_arr[i],  # static per battery
                        "target": np.float32(time_arr[i + seq_len, volt_idx]),
                    }
                )

        # ── Optionally limit samples ─────────────────────────────────────
        if max_samples is not None and len(self.samples) > max_samples:
            rng2 = np.random.default_rng(seed)
            idxs = rng2.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in idxs]

        logger.info(
            "%s dataset: %d samples (seq_len=%d)",
            "Train" if train else "Val",
            len(self.samples),
            seq_len,
        )

    # Expose split IDs so we can pass them to the val dataset
    @property
    def train_battery_ids(self) -> set:
        return self._train_ids

    @property
    def val_battery_ids(self) -> set:
        return self._val_ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "time_features": torch.from_numpy(s["time_features"]),
            "cat_features": torch.from_numpy(s["cat_features"]),
            "target": torch.tensor(s["target"]),
        }


# =============================================================================
# SECTION 2 — MODEL
# =============================================================================


class LiSOCl2ForecastLSTM(nn.Module):
    """
    LSTM + entity embeddings for Li-SOCl2 voltage forecasting.

    Architecture:
        Time-varying features (seq_len, 6) -> 2-layer LSTM (hidden=128)
            -> last hidden state
        Static categoricals -> Embeddings (5 dims total)
            -> concat with LSTM output
        -> FC(133->64) + ReLU + Dropout
        -> FC(64->32)  + ReLU + Dropout
        -> FC(32->1)   -> predicted next voltage
    """

    def __init__(
        self,
        n_time_features: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        cat_config: Optional[dict[str, tuple[int, int]]] = None,
    ):
        super().__init__()
        if cat_config is None:
            cat_config = CAT_EMBED_CONFIG

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM for time-varying features
        self.lstm = nn.LSTM(
            input_size=n_time_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Entity embeddings for categoricals
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for name, (n_cats, emb_dim) in cat_config.items():
            self.embeddings[name] = nn.Embedding(
                num_embeddings=n_cats + 1, embedding_dim=emb_dim  # +1 for unknown
            )
            total_emb_dim += emb_dim

        fc_input = hidden_size + total_emb_dim  # 128 + 5 = 133

        # MLP head
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "LiSOCl2ForecastLSTM: time_features=%d, hidden=%d, layers=%d, "
            "emb_dim=%d, fc_input=%d, params=%d",
            n_time_features,
            hidden_size,
            num_layers,
            total_emb_dim,
            fc_input,
            total_params,
        )

    def forward(
        self,
        time_features: torch.Tensor,
        cat_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            time_features: (batch, seq_len, n_time_features) float
            cat_features:  (batch, n_cat) long

        Returns:
            (batch, 1) predicted voltage (scaled)
        """
        # LSTM
        lstm_out, _ = self.lstm(time_features)
        h_last = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Embeddings
        emb_parts = []
        for i, name in enumerate(self.embeddings):
            emb_parts.append(self.embeddings[name](cat_features[:, i]))
        emb_cat = torch.cat(emb_parts, dim=1)  # (batch, total_emb_dim)

        # Concat and MLP
        combined = torch.cat([h_last, emb_cat], dim=1)
        return self.fc(combined)  # (batch, 1)


# =============================================================================
# SECTION 3 — TRAINING
# =============================================================================


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 0.001,
    num_epochs: int = 80,
    patience: int = 15,
    output_dir: str = "./models_lisocl2",
) -> tuple[nn.Module, dict]:
    """Train the model and return (model, history)."""
    os.makedirs(output_dir, exist_ok=True)

    model = LiSOCl2ForecastLSTM(
        n_time_features=len(TIME_FEATURES),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for batch in train_loader:
            tf = batch["time_features"].to(device)
            cf = batch["cat_features"].to(device)
            y = batch["target"].to(device)

            optimizer.zero_grad()
            pred = model(tf, cf).squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(y)
            train_n += len(y)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                tf = batch["time_features"].to(device)
                cf = batch["cat_features"].to(device)
                y = batch["target"].to(device)

                pred = model(tf, cf).squeeze(-1)
                loss = criterion(pred, y)
                val_loss_sum += loss.item() * len(y)
                val_n += len(y)

        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{num_epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"lr={lr_now:.2e} | {elapsed:.1f}s"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.6f})")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    logger.info("Best val_loss=%.6f", best_val_loss)

    return model, history


# =============================================================================
# SECTION 4 — AUTOREGRESSIVE 12-MONTH FORECAST
# =============================================================================


@torch.no_grad()
def autoregressive_forecast(
    model: nn.Module,
    df_raw: pd.DataFrame,
    battery_ids: list[str],
    feature_scalers: dict[str, StandardScaler],
    cat_encoders: dict[str, OrdinalEncoder],
    device: torch.device,
    seq_len: int = 24,
    horizon: int = 12,
) -> dict[str, dict]:
    """
    For each battery, run 12-step autoregressive forecast from its last
    available history window.

    Returns dict keyed by battery_id with:
        - "actual_voltage": array of last seq_len+horizon actual voltages (raw)
        - "forecast_voltage": array of horizon forecasted voltages (raw)
        - "history_voltage": array of seq_len history voltages (raw)
        - "manufacturer": str
    """
    model.eval()
    df = _add_cyclical_month(df_raw)
    results = {}

    # Suppress sklearn feature-name warnings when passing scalars
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names",
        category=UserWarning,
    )

    volt_scaler = feature_scalers["voltage"]
    temp_scaler = feature_scalers["temperature"]
    cap_scaler = feature_scalers["cumulative_capacity_used_pct"]
    pulse_scaler = feature_scalers["pulse_usage_min_per_day"]

    for bid in battery_ids:
        grp = df[df["battery_id"] == bid].sort_values("month").reset_index(drop=True)
        if len(grp) < seq_len + horizon:
            continue

        # Take the window ending `horizon` months before the end so we have actuals
        end_idx = len(grp) - horizon
        start_idx = end_idx - seq_len
        if start_idx < 0:
            continue

        # Actual future voltages for comparison
        actual_raw = grp["voltage"].values[end_idx : end_idx + horizon]
        history_raw = grp["voltage"].values[start_idx:end_idx]
        manufacturer = grp["device_manufacturer"].iloc[0]

        # Encode categoricals
        cat_vals = []
        for col in CAT_FEATURES:
            enc_val = cat_encoders[col].transform([[grp[col].iloc[0]]])[0, 0]
            cat_vals.append(int(enc_val))
        cat_tensor = torch.tensor([cat_vals], dtype=torch.long, device=device)

        # Build initial scaled time-feature window
        window_df = grp.iloc[start_idx:end_idx].copy()
        time_arr = np.zeros((seq_len, len(TIME_FEATURES)), dtype=np.float32)
        for fi, feat in enumerate(TIME_FEATURES):
            time_arr[:, fi] = feature_scalers[feat].transform(
                window_df[[feat]]
            ).ravel().astype(np.float32)

        # Prepare projection helpers
        # Seasonal temperature: average per calendar month from last 12 months of history
        hist_12 = grp.iloc[max(0, end_idx - 12) : end_idx]
        cal_month_temps = {}
        for _, row in hist_12.iterrows():
            cm = int(row["month"] - 1) % 12
            cal_month_temps.setdefault(cm, []).append(row["temperature"])
        for cm in cal_month_temps:
            cal_month_temps[cm] = np.mean(cal_month_temps[cm])
        overall_temp_mean = np.mean(hist_12["temperature"].values)

        # Average monthly capacity drain
        cap_vals_raw = grp["cumulative_capacity_used_pct"].values[start_idx:end_idx]
        if len(cap_vals_raw) >= 2:
            avg_cap_drain = (cap_vals_raw[-1] - cap_vals_raw[0]) / (len(cap_vals_raw) - 1)
        else:
            avg_cap_drain = 0.0
        last_cap = grp["cumulative_capacity_used_pct"].values[end_idx - 1]

        # Pulse usage (constant)
        pulse_raw = grp["pulse_usage_min_per_day"].values[0]

        # Last month index for cyclical encoding
        last_month = int(grp["month"].values[end_idx - 1])

        # ── Rollout ──────────────────────────────────────────────────────
        forecast_scaled = []
        current_window = time_arr.copy()  # (seq_len, 6)

        for step in range(horizon):
            inp = torch.from_numpy(current_window).unsqueeze(0).to(device)
            pred_scaled = model(inp, cat_tensor).item()
            forecast_scaled.append(pred_scaled)

            # Project features for next step
            next_month_idx = last_month + step + 1
            cal_m = (next_month_idx - 1) % 12

            # Temperature projection
            proj_temp_raw = cal_month_temps.get(cal_m, overall_temp_mean)
            proj_temp_scaled = temp_scaler.transform([[proj_temp_raw]])[0, 0]

            # Capacity projection
            proj_cap_raw = last_cap + avg_cap_drain * (step + 1)
            proj_cap_scaled = cap_scaler.transform([[proj_cap_raw]])[0, 0]

            # Pulse (constant)
            proj_pulse_scaled = pulse_scaler.transform([[pulse_raw]])[0, 0]

            # Cyclical month
            m_sin = np.sin(2 * np.pi * next_month_idx / 12)
            m_cos = np.cos(2 * np.pi * next_month_idx / 12)
            # Scale cyclical (they are already in [-1,1], but match training)
            m_sin_scaled = feature_scalers["month_sin"].transform([[m_sin]])[0, 0]
            m_cos_scaled = feature_scalers["month_cos"].transform([[m_cos]])[0, 0]

            new_row = np.array(
                [
                    pred_scaled,
                    proj_temp_scaled,
                    proj_cap_scaled,
                    proj_pulse_scaled,
                    m_sin_scaled,
                    m_cos_scaled,
                ],
                dtype=np.float32,
            )
            # Shift window
            current_window = np.vstack([current_window[1:], new_row[np.newaxis, :]])

        # Inverse-scale forecasted voltages
        forecast_raw = volt_scaler.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)
        ).ravel()

        results[bid] = {
            "actual_voltage": actual_raw,
            "forecast_voltage": forecast_raw,
            "history_voltage": history_raw,
            "manufacturer": manufacturer,
        }

    logger.info("Forecasted %d batteries with %d-step horizon", len(results), horizon)
    return results


# =============================================================================
# SECTION 5 — EVALUATION & PLOTS
# =============================================================================


def evaluate_and_plot(
    results: dict[str, dict],
    history: dict[str, list[float]],
    output_dir: str,
    horizon: int = 12,
) -> None:
    """Generate all evaluation plots and print metrics."""
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Training / validation loss curves ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved loss_curves.png")

    # Collect per-step errors across all batteries
    per_step_errors = {s: [] for s in range(horizon)}
    all_actual, all_pred = [], []
    mfg_errors: dict[str, list[float]] = {}

    for bid, res in results.items():
        actual = res["actual_voltage"]
        forecast = res["forecast_voltage"]
        mfg = res["manufacturer"]
        n = min(len(actual), len(forecast))
        for s in range(n):
            err = abs(forecast[s] - actual[s])
            per_step_errors[s].append(err)
        all_actual.extend(actual[:n])
        all_pred.extend(forecast[:n])
        mae = np.mean(np.abs(forecast[:n] - actual[:n]))
        mfg_errors.setdefault(mfg, []).append(mae)

    all_actual = np.array(all_actual)
    all_pred = np.array(all_pred)

    # Overall metrics
    overall_mae = np.mean(np.abs(all_pred - all_actual))
    overall_rmse = np.sqrt(np.mean((all_pred - all_actual) ** 2))
    print(f"\n  Forecast Metrics (12-step autoregressive):")
    print(f"    MAE:  {overall_mae:.4f} V")
    print(f"    RMSE: {overall_rmse:.4f} V")

    # ── 2. Per-horizon-step MAE bar chart ────────────────────────────────
    step_maes = [
        np.mean(per_step_errors[s]) if per_step_errors[s] else 0 for s in range(horizon)
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(1, horizon + 1), step_maes, color="steelblue", edgecolor="white")
    ax.set_xlabel("Forecast Step (months ahead)")
    ax.set_ylabel("MAE (V)")
    ax.set_title("Per-Horizon-Step MAE")
    ax.set_xticks(range(1, horizon + 1))
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, step_maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_step_mae.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved per_step_mae.png")

    print("  Per-step MAE:")
    for s in range(horizon):
        print(f"    t+{s+1:2d}: {step_maes[s]:.4f} V")

    # ── 3. Sample battery trajectories ───────────────────────────────────
    sample_ids = list(results.keys())[:6]
    n_plots = min(len(sample_ids), 6)
    if n_plots > 0:
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, bid in enumerate(sample_ids[:n_plots]):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            res = results[bid]
            hist = res["history_voltage"]
            actual = res["actual_voltage"]
            forecast = res["forecast_voltage"]
            n = min(len(actual), len(forecast))

            hist_x = np.arange(len(hist))
            future_x = np.arange(len(hist), len(hist) + n)

            ax.plot(hist_x, hist, "b-", linewidth=1.2, label="History")
            ax.plot(future_x, actual[:n], "g-", linewidth=1.2, label="Actual")
            ax.plot(future_x, forecast[:n], "r--", linewidth=1.2, label="Forecast")
            ax.axvline(len(hist) - 0.5, color="gray", linestyle=":", alpha=0.5)
            ax.set_title(f"{bid} ({res['manufacturer'][:8]}...)", fontsize=9)
            ax.set_xlabel("Month")
            ax.set_ylabel("Voltage (V)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_plots, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].set_visible(False)

        fig.suptitle("Sample Battery Trajectories — History + 12-Month Forecast", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "sample_trajectories.png"), dpi=150)
        plt.close(fig)
        logger.info("Saved sample_trajectories.png")

    # ── 4. Predicted vs actual scatter ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(all_actual, all_pred, alpha=0.15, s=8, color="steelblue")
    lims = [
        min(all_actual.min(), all_pred.min()) - 0.05,
        max(all_actual.max(), all_pred.max()) + 0.05,
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Voltage (V)")
    ax.set_ylabel("Predicted Voltage (V)")
    ax.set_title(f"Predicted vs Actual (MAE={overall_mae:.4f} V, RMSE={overall_rmse:.4f} V)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pred_vs_actual.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved pred_vs_actual.png")

    # ── 5. Per-manufacturer error breakdown ──────────────────────────────
    mfg_names = sorted(mfg_errors.keys())
    mfg_maes = [np.mean(mfg_errors[m]) for m in mfg_names]
    mfg_stds = [np.std(mfg_errors[m]) for m in mfg_names]
    short_names = [m.replace("_", "\n") for m in mfg_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(mfg_names)),
        mfg_maes,
        yerr=mfg_stds,
        capsize=4,
        color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"][: len(mfg_names)],
        edgecolor="white",
    )
    ax.set_xticks(range(len(mfg_names)))
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_xlabel("Manufacturer")
    ax.set_ylabel("MAE (V)")
    ax.set_title("Per-Manufacturer Forecast Error (MAE +/- std)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, mfg_maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "manufacturer_error.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved manufacturer_error.png")

    print(f"\n  Per-manufacturer MAE:")
    for m, mae_val, std_val in zip(mfg_names, mfg_maes, mfg_stds):
        print(f"    {m:30s}  MAE={mae_val:.4f} +/- {std_val:.4f} V")


# =============================================================================
# SECTION 6 — CHECKPOINT SAVE/LOAD
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    feature_scalers: dict[str, StandardScaler],
    cat_encoders: dict[str, OrdinalEncoder],
    val_battery_ids: set,
    history: dict,
    output_dir: str,
) -> str:
    """Save model + preprocessing artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "lisocl2_forecast_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "n_time_features": len(TIME_FEATURES),
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "cat_config": {
                    k: (v.num_embeddings - 1, v.embedding_dim)
                    for k, v in model.embeddings.items()
                },
            },
            "feature_scalers": feature_scalers,
            "cat_encoders": cat_encoders,
            "val_battery_ids": list(val_battery_ids),
            "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
            "history": history,
        },
        path,
    )
    logger.info("Checkpoint saved: %s", path)
    return path


# =============================================================================
# SECTION 7 — MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Li-SOCl2 Voltage Forecasting — LSTM + Entity Embeddings"
    )
    parser.add_argument(
        "--n_batteries", type=int, default=5000, help="Number of batteries to simulate"
    )
    parser.add_argument(
        "--n_months", type=int, default=300, help="Months per battery"
    )
    parser.add_argument(
        "--seq_len", type=int, default=24, help="Sequence length (months of history)"
    )
    parser.add_argument(
        "--horizon", type=int, default=12, help="Forecast horizon (months)"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="LSTM hidden size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="LSTM layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=80, help="Max training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Limit training samples"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./models_lisocl2", help="Output directory"
    )
    parser.add_argument(
        "--forecast_samples", type=int, default=200,
        help="Number of val batteries to forecast",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help="Fast mode for testing: 200 batteries, 60 months, 10 epochs, "
        "hidden=32, 1 LSTM layer, batch=128, 20 forecast samples",
    )
    args = parser.parse_args()

    # ── Fast mode overrides ──────────────────────────────────────────────
    if args.fast:
        FAST_DEFAULTS = {
            "n_batteries": 200,
            "n_months": 60,
            "num_epochs": 10,
            "patience": 5,
            "hidden_size": 32,
            "num_layers": 1,
            "batch_size": 128,
            "forecast_samples": 20,
            "max_samples": 10000,
        }
        for key, val in FAST_DEFAULTS.items():
            # Only override if user didn't explicitly set the flag
            if getattr(args, key) == parser.get_default(key):
                setattr(args, key, val)

    print("=" * 72)
    mode_tag = " [FAST MODE]" if args.fast else ""
    print(f"  Li-SOCl2 Voltage Forecasting — LSTM + Entity Embeddings{mode_tag}")
    print("=" * 72)

    # ── 1. Generate data ─────────────────────────────────────────────────
    print("\n[1] Generating synthetic Li-SOCl2 data...")
    df = generate_lisocl2_timeseries(
        n_batteries=args.n_batteries,
        n_months=args.n_months,
        seed=args.seed,
    )
    print(f"    Shape: {df.shape}")
    print(f"    Batteries: {df['battery_id'].nunique()}")
    print(f"    Months/battery: {args.n_months}")

    # ── 2. Create datasets ───────────────────────────────────────────────
    print("\n[2] Creating train/val datasets (battery-level split)...")
    train_ds = LiSOCl2TimeSeriesDataset(
        df,
        seq_len=args.seq_len,
        train=True,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    val_ds = LiSOCl2TimeSeriesDataset(
        df,
        seq_len=args.seq_len,
        train=False,
        seed=args.seed,
        feature_scalers=train_ds.feature_scalers,
        cat_encoders=train_ds.cat_encoders,
        val_battery_ids=train_ds.val_battery_ids,
        train_battery_ids=train_ds.train_battery_ids,
    )
    print(f"    Train samples: {len(train_ds):,}")
    print(f"    Val samples:   {len(val_ds):,}")

    # ── 3. DataLoaders ───────────────────────────────────────────────────
    device = get_compatible_device(args.device)
    print(f"\n[3] Device: {device}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches:   {len(val_loader)}")

    # ── 4. Train ─────────────────────────────────────────────────────────
    print(f"\n[4] Training (max {args.num_epochs} epochs, patience={args.patience})...")
    print("-" * 72)
    model, history = train_model(
        train_loader,
        val_loader,
        device=device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        num_epochs=args.num_epochs,
        patience=args.patience,
        output_dir=args.output_dir,
    )
    print("-" * 72)

    # ── 5. Save checkpoint ───────────────────────────────────────────────
    print("\n[5] Saving checkpoint...")
    ckpt_path = save_checkpoint(
        model,
        train_ds.feature_scalers,
        train_ds.cat_encoders,
        train_ds.val_battery_ids,
        history,
        args.output_dir,
    )
    print(f"    Saved: {ckpt_path}")

    # ── 6. Autoregressive forecast on val batteries ──────────────────────
    print(f"\n[6] Running {args.horizon}-step autoregressive forecast...")
    val_ids = sorted(train_ds.val_battery_ids)
    forecast_ids = val_ids[: args.forecast_samples]
    results = autoregressive_forecast(
        model,
        df,
        forecast_ids,
        train_ds.feature_scalers,
        train_ds.cat_encoders,
        device,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )
    print(f"    Forecasted {len(results)} batteries")

    # ── 7. Evaluate & plot ───────────────────────────────────────────────
    print(f"\n[7] Evaluation & plots -> {args.output_dir}/")
    evaluate_and_plot(results, history, args.output_dir, horizon=args.horizon)

    print("\n" + "=" * 72)
    print("  Done!")
    print("=" * 72)


if __name__ == "__main__":
    main()
