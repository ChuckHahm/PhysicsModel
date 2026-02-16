"""
=============================================================================
ML Tabular Data Pipeline — Mixed Numerical & Categorical Features
=============================================================================

A production-ready template for building ML pipelines on tabular datasets
containing both numerical and categorical columns.

Architecture:
    1. Synthetic data generation (replace with your real data loader)
    2. Preprocessing with scikit-learn ColumnTransformer
    3. PyTorch Dataset / DataLoader for batched training
    4. Entity embedding model (learnable categorical embeddings + numerics)
    5. Training loop with validation
    6. Inference / prediction utilities

Dependencies:
    pip install torch scikit-learn pandas numpy

Author: Chuck — ML Pipeline Template
=============================================================================
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Optional

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_pipeline")

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠  PyTorch not installed — preprocessing pipeline will still run.")
    print("   Install with: pip install torch")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset simulating a predictive maintenance scenario.

    Features:
        Numerical:  voltage, temperature, cycle_count, impedance, load_current
        Categorical: manufacturer, battery_type, installation_region, failure_mode

    Target:
        remaining_useful_life (continuous, in days)

    The target has a realistic nonlinear relationship with the features so that
    the model actually has something meaningful to learn.
    """
    logger.info("Generating synthetic data: n_samples=%d, seed=%d", n_samples, seed)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # — Categorical features —
    manufacturers   = rng.choice(["MFG_A", "MFG_B", "MFG_C", "MFG_D"], n_samples)
    battery_types   = rng.choice(["LiSOCl2", "LiFePO4", "NiMH", "LeadAcid"], n_samples)
    regions         = rng.choice(["North", "South", "East", "West", "Central"], n_samples)
    failure_modes   = rng.choice(["capacity_fade", "impedance_rise", "thermal", "none"], n_samples,
                                  p=[0.25, 0.20, 0.15, 0.40])

    # — Numerical features —
    voltage       = rng.normal(3.6, 0.3, n_samples)
    temperature   = rng.normal(25.0, 8.0, n_samples)
    cycle_count   = rng.exponential(500, n_samples).astype(int)
    impedance     = rng.lognormal(2.0, 0.5, n_samples)
    load_current  = rng.uniform(0.1, 5.0, n_samples)

    # — Construct target with nonlinear relationships —
    # Base RUL driven by voltage and cycle degradation
    rul = (
        800
        + 200 * (voltage - 3.0)                         # higher voltage → longer life
        - 0.3 * cycle_count                              # more cycles → shorter life
        - 5.0 * np.clip(temperature - 30, 0, None)       # heat penalty
        - 0.8 * impedance                                # impedance rise hurts
        + 50 * (manufacturers == "MFG_A").astype(float)  # manufacturer effect
        - 80 * (failure_modes == "thermal").astype(float) # thermal failures worse
        + rng.normal(0, 30, n_samples)                   # noise
    )
    rul = np.clip(rul, 0, None)  # no negative RUL
    logger.debug("RUL range: [%.1f, %.1f], mean=%.1f", rul.min(), rul.max(), rul.mean())

    df = pd.DataFrame({
        "voltage":             voltage,
        "temperature":         temperature,
        "cycle_count":         cycle_count,
        "impedance":           impedance,
        "load_current":        load_current,
        "manufacturer":        manufacturers,
        "battery_type":        battery_types,
        "installation_region": regions,
        "failure_mode":        failure_modes,
        "remaining_useful_life": rul,
    })

    logger.info("Synthetic data generated in %.3fs — shape %s", time.perf_counter() - t0, df.shape)
    return df


def generate_lisocl2_timeseries(
    n_batteries: int = 5000,
    n_months: int = 60,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate realistic synthetic Li-SOCl2 battery time-series data.

    Simulates n_batteries deployed across the Paris metro area, each with
    n_months of monthly voltage/temperature readings.  Models physical
    Li-SOCl2 degradation including Arrhenius-accelerated self-discharge,
    pulse drain, manufacturer efficiency differences, passivation effects,
    and seasonal temperature cycles.

    All math is numpy-vectorized (no Python loops over batteries/months).

    Args:
        n_batteries: Number of batteries to simulate.
        n_months: Number of monthly readings per battery.
        seed: Random seed for reproducibility.
        save_path: If provided, save the DataFrame to this CSV path.

    Returns:
        DataFrame with n_batteries * n_months rows and 12 columns:
        battery_id, month, date, voltage, temperature, latitude, longitude,
        device_manufacturer, installation_type, pulse_usage_min_per_day,
        cumulative_capacity_used_pct, remaining_useful_life.
    """
    logger.info(
        "Generating Li-SOCl2 time-series: %d batteries x %d months = %d rows",
        n_batteries, n_months, n_batteries * n_months,
    )
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # ── Battery-level attributes (n_batteries,) ──────────────────────────

    # Manufacturers with market shares and drain multipliers
    mfg_names = np.array([
        "Meridian_Instruments", "Vortex_Metering", "Pinnacle_Systems",
        "Axiom_Technologies", "Zenith_Devices",
    ])
    mfg_shares = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
    mfg_drains = np.array([0.90, 0.95, 1.00, 1.08, 1.15])
    mfg_idx = rng.choice(len(mfg_names), size=n_batteries, p=mfg_shares)
    manufacturers = mfg_names[mfg_idx]
    drain_multipliers = mfg_drains[mfg_idx]

    # Installation type: 60% indoor, 40% outdoor
    installation_type = rng.choice(
        ["indoor", "outdoor"], size=n_batteries, p=[0.60, 0.40]
    )
    is_indoor = installation_type == "indoor"

    # GPS coordinates — Normal distribution centred on central Paris
    latitude = np.clip(rng.normal(48.86, 0.06, n_batteries), 48.7, 49.0)
    longitude = np.clip(rng.normal(2.35, 0.10, n_batteries), 2.1, 2.6)

    # Pulse usage — lognormal around 1 min/day
    pulse_usage = np.clip(
        rng.lognormal(np.log(1.0), 0.5, n_batteries), 0.1, 10.0
    )

    # Battery IDs
    battery_ids = np.array([f"BAT_{i + 1:05d}" for i in range(n_batteries)])

    # ── Time axis ────────────────────────────────────────────────────────

    months = np.arange(1, n_months + 1)  # (n_months,)
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    date_strings = dates.strftime("%Y-%m-%d").values

    # ── Temperature model ────────────────────────────────────────────────
    # Paris real monthly averages (°C)
    paris_monthly_avg = np.array(
        [4.1, 5.0, 8.0, 10.8, 14.5, 17.5, 19.8, 19.5, 16.0, 12.0, 7.4, 4.8]
    )
    seasonal_mean = paris_monthly_avg.mean()
    calendar_month_idx = (months - 1) % 12
    seasonal_deviation = paris_monthly_avg[calendar_month_idx] - seasonal_mean

    # Year-to-year climate variation
    n_years = (n_months + 11) // 12
    year_shifts = rng.normal(0, 0.5, n_years)
    year_variation = year_shifts[(months - 1) // 12]

    # Per-battery temperature parameters (indoor vs outdoor)
    temp_offset = np.where(
        is_indoor, rng.uniform(5, 15, n_batteries), rng.uniform(-2, 5, n_batteries)
    )
    temp_amplitude = np.where(
        is_indoor, rng.uniform(0.2, 0.5, n_batteries), rng.uniform(0.8, 1.2, n_batteries)
    )

    # Build (n_batteries, n_months) temperature array — fully vectorized
    temperature = (
        seasonal_mean
        + temp_offset[:, np.newaxis]
        + seasonal_deviation[np.newaxis, :] * temp_amplitude[:, np.newaxis]
        + year_variation[np.newaxis, :]
        + rng.normal(0, 1.5, (n_batteries, n_months))
    )

    # ── Degradation model ────────────────────────────────────────────────

    # Self-discharge: ~1%/year at 25°C, Arrhenius temperature acceleration
    #   k_self(T) = k_ref * exp(E_a/R * (1/T_ref - 1/T))
    k_ref_monthly = 0.01 / 12.0       # 1%/year → monthly fraction at T_ref
    T_ref_K = 273.15 + 25.0           # reference temperature (K)
    E_a_over_R = 5000.0               # activation energy / gas constant (K)

    T_kelvin = temperature + 273.15
    k_self = k_ref_monthly * np.exp(
        E_a_over_R * (1.0 / T_ref_K - 1.0 / T_kelvin)
    )

    # Pulse drain: 30 mA active current, pulse_usage min/day, 3.65 Ah nominal
    #   Monthly fraction = 0.030 A * (pulse_min * 60 s / 3600) * 30 days / 3.65 Ah
    #                    = 0.015 * pulse_usage / 3.65  (~0.42%/month for 1 min/day)
    pulse_drain_monthly = 0.015 * pulse_usage / 3.65  # (n_batteries,)

    # Total monthly drain = (self-discharge + pulse) * manufacturer multiplier
    monthly_drain = (
        (k_self + pulse_drain_monthly[:, np.newaxis])
        * drain_multipliers[:, np.newaxis]
    )  # (n_batteries, n_months)

    cumulative_capacity = np.cumsum(monthly_drain, axis=1)
    soc = np.clip(1.0 - cumulative_capacity, 0.0, 1.0)  # state of charge

    # ── Voltage curve ────────────────────────────────────────────────────
    # Sigmoid SOC-to-voltage: classic Li-SOCl2 profile
    #   Flat plateau ~3.6 V (SOC > 15 %), sharp knee ~8 % SOC, cutoff 2.5 V
    voltage_base = 2.5 + 1.1 / (1.0 + np.exp(-80.0 * (soc - 0.08)))

    # Passivation effect: initial dip in month 1, exponential recovery ~6 months
    passivation_depth = rng.uniform(0.04, 0.12, n_batteries)
    months_2d = np.broadcast_to(months[np.newaxis, :], (n_batteries, n_months))
    passivation_dip = passivation_depth[:, np.newaxis] * np.exp(
        -(months_2d - 1) / 2.0
    )

    # Measurement noise (~15 mV gaussian)
    noise = rng.normal(0, 0.015, (n_batteries, n_months))

    voltage = voltage_base - passivation_dip + noise

    # ── Remaining useful life ────────────────────────────────────────────
    # RUL ≈ months until SOC reaches 0 (voltage ≈ 2.5 V cutoff)
    avg_drain = np.mean(monthly_drain, axis=1, keepdims=True)  # (n_batteries, 1)
    rul = np.where(avg_drain > 0, soc / avg_drain, 0.0)
    rul = np.clip(rul, 0.0, None)

    # ── Assemble DataFrame ───────────────────────────────────────────────

    df = pd.DataFrame({
        "battery_id":                   np.repeat(battery_ids, n_months),
        "month":                        np.tile(months, n_batteries),
        "date":                         np.tile(date_strings, n_batteries),
        "voltage":                      np.round(voltage.ravel(), 4),
        "temperature":                  np.round(temperature.ravel(), 2),
        "latitude":                     np.round(np.repeat(latitude, n_months), 6),
        "longitude":                    np.round(np.repeat(longitude, n_months), 6),
        "device_manufacturer":          np.repeat(manufacturers, n_months),
        "installation_type":            np.repeat(installation_type, n_months),
        "pulse_usage_min_per_day":      np.round(np.repeat(pulse_usage, n_months), 3),
        "cumulative_capacity_used_pct": np.round((cumulative_capacity * 100.0).ravel(), 4),
        "remaining_useful_life":        np.round(rul.ravel(), 2),
    })

    elapsed = time.perf_counter() - t0
    logger.info("Li-SOCl2 data generated in %.3fs — shape %s", elapsed, df.shape)

    if save_path:
        df.to_csv(save_path, index=False)
        size_mb = df.memory_usage(deep=True).sum() / 1e6
        logger.info("Saved to %s (%.1f MB in memory)", save_path, size_mb)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COLUMN DEFINITIONS & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# Define column groups — edit these for your own dataset
NUMERICAL_COLS  = ["voltage", "temperature", "cycle_count", "impedance", "load_current"]
CATEGORICAL_COLS = ["manufacturer", "battery_type", "installation_region", "failure_mode"]
TARGET_COL       = "remaining_useful_life"

# Column groups for the Li-SOCl2 time-series dataset
LISOCL2_NUMERICAL_COLS = [
    "voltage", "temperature", "latitude", "longitude",
    "pulse_usage_min_per_day", "cumulative_capacity_used_pct",
]
LISOCL2_CATEGORICAL_COLS = ["device_manufacturer", "installation_type"]
LISOCL2_TARGET_COL = "remaining_useful_life"
LISOCL2_ID_COLS = ["battery_id", "month", "date"]


def build_preprocessor(
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
      - Standardizes numerical columns (zero mean, unit variance)
      - Ordinal-encodes categorical columns (integer codes for embedding layers)

    Why OrdinalEncoder instead of OneHotEncoder?
    ─────────────────────────────────────────────
    We use ordinal encoding because the downstream PyTorch model will learn
    entity embeddings for each categorical feature. This is far more parameter-
    efficient than one-hot encoding for high-cardinality categoricals (e.g.,
    device IDs, region codes). The embedding approach was popularized by the
    "Entity Embeddings of Categorical Variables" paper (Guo & Berkhahn, 2016)
    and is standard practice in tabular deep learning.

    For tree-based models (LightGBM, XGBoost), ordinal encoding also works
    directly — no one-hot needed.
    """
    logger.info("Building preprocessor: %d numerical cols, %d categorical cols",
                len(numerical_cols), len(categorical_cols))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    logger.info("Preprocessor built: StandardScaler + OrdinalEncoder")
    return preprocessor


def preprocess_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    fit: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply preprocessing and split into (numerical_array, categorical_array, targets).

    Returns:
        num_features : np.ndarray, shape (n, len(NUMERICAL_COLS)), float32
        cat_features : np.ndarray, shape (n, len(CATEGORICAL_COLS)), int64
        targets      : np.ndarray, shape (n,), float32
    """
    mode = "fit_transform" if fit else "transform"
    logger.info("Preprocessing %d rows (mode=%s)", len(df), mode)
    t0 = time.perf_counter()

    if fit:
        transformed = preprocessor.fit_transform(df)
    else:
        transformed = preprocessor.transform(df)

    n_num = len(NUMERICAL_COLS)
    num_features = transformed[:, :n_num].astype(np.float32)
    cat_features = transformed[:, n_num:].astype(np.int64)
    targets      = df[TARGET_COL].values.astype(np.float32)

    logger.info("Preprocessing done in %.3fs — num:%s cat:%s targets:%s",
                time.perf_counter() - t0, num_features.shape, cat_features.shape, targets.shape)
    return num_features, cat_features, targets


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PyTorch DATASET & DATALOADER
# ══════════════════════════════════════════════════════════════════════════════

if not TORCH_AVAILABLE:
    # Provide stubs so the module can be imported without PyTorch
    TabularDataset = None
    TabularEmbeddingModel = None
    Trainer = None
    create_dataloaders = None
    load_and_predict = None
else:
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — PyTorch DATASET & DATALOADER
    # ══════════════════════════════════════════════════════════════════════════

    class TabularDataset(Dataset):
        """
        PyTorch Dataset for mixed numerical + categorical tabular data.

        Each __getitem__ call returns a dict with:
            "numerical"   : FloatTensor  (n_numerical_features,)
            "categorical" : LongTensor   (n_categorical_features,)
            "target"      : FloatTensor  scalar
        """

        def __init__(
            self,
            num_features: np.ndarray,
            cat_features: np.ndarray,
            targets: np.ndarray,
        ):
            logger.info("Creating TabularDataset: %d samples", len(targets))
            self.num_features = torch.tensor(num_features, dtype=torch.float32)
            self.cat_features = torch.tensor(cat_features, dtype=torch.long)
            self.targets      = torch.tensor(targets, dtype=torch.float32)
            logger.debug("TabularDataset tensors — num:%s cat:%s targets:%s",
                         self.num_features.shape, self.cat_features.shape, self.targets.shape)

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, idx: int) -> dict:
            return {
                "numerical":   self.num_features[idx],
                "categorical": self.cat_features[idx],
                "target":      self.targets[idx],
            }


    def create_dataloaders(
        train_dataset: TabularDataset,
        val_dataset: TabularDataset,
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        """Create train and validation DataLoaders."""
        logger.info("Creating DataLoaders: batch_size=%d, num_workers=%d, pin_memory=%s",
                    batch_size, num_workers, torch.cuda.is_available())
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,   # larger batch for eval (no grad needed)
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info("DataLoaders ready — train: %d batches, val: %d batches",
                    len(train_loader), len(val_loader))
        return train_loader, val_loader


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — ENTITY EMBEDDING MODEL
    # ══════════════════════════════════════════════════════════════════════════

    class TabularEmbeddingModel(nn.Module):
        """
        Neural network for tabular data with entity embeddings.

        Architecture:
            ┌──────────────┐    ┌──────────────────┐
            │  Categorical  │    │    Numerical      │
            │   Inputs      │    │    Inputs         │
            └──────┬───────┘    └────────┬──────────┘
                   │                      │
            ┌──────▼───────┐             │
            │  Embedding    │             │
            │  Layers       │             │
            └──────┬───────┘             │
                   │                      │
                   └──────┬───────────────┘
                          │  concatenate
                   ┌──────▼───────┐
                   │  BatchNorm   │
                   └──────┬───────┘
                   ┌──────▼───────┐
                   │  FC + ReLU   │──► Dropout
                   │  FC + ReLU   │──► Dropout
                   │  FC (output) │
                   └──────────────┘

        Args:
            cat_cardinalities : list of int — number of unique values per categorical column
            cat_embedding_dims: list of int — embedding dimension per categorical column
                                (rule of thumb: min(50, cardinality // 2))
            n_numerical       : int — number of numerical input features
            hidden_dims       : list of int — hidden layer sizes
            dropout           : float — dropout rate between hidden layers
            output_dim        : int — output size (1 for regression, n for classification)
        """

        def __init__(
            self,
            cat_cardinalities: list[int],
            cat_embedding_dims: list[int],
            n_numerical: int,
            hidden_dims: list[int] = [256, 128, 64],
            dropout: float = 0.3,
            output_dim: int = 1,
        ):
            super().__init__()
            logger.info("Building TabularEmbeddingModel: %d numerical, %d categorical, hidden=%s, dropout=%.2f",
                        n_numerical, len(cat_cardinalities), hidden_dims, dropout)

            # — Embedding layers for each categorical feature —
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_embeddings=card + 1, embedding_dim=dim)  # +1 for unknown
                for card, dim in zip(cat_cardinalities, cat_embedding_dims)
            ])

            # — Calculate total input dimension —
            total_emb_dim = sum(cat_embedding_dims)
            input_dim = n_numerical + total_emb_dim

            # — Build MLP layers —
            layers = []
            layers.append(nn.BatchNorm1d(input_dim))

            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.Dropout(dropout))
                prev_dim = h_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.mlp = nn.Sequential(*layers)
            total_params = sum(p.numel() for p in self.parameters())
            logger.info("Model built: input_dim=%d, output_dim=%d, total_params=%d",
                        input_dim, output_dim, total_params)

        def forward(
            self,
            x_numerical: torch.Tensor,
            x_categorical: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                x_numerical   : (batch, n_numerical) float tensor
                x_categorical : (batch, n_categorical) long tensor

            Returns:
                (batch, output_dim) predictions
            """
            # Embed each categorical column and concatenate
            emb_outputs = [
                emb(x_categorical[:, i])
                for i, emb in enumerate(self.embeddings)
            ]
            x_cat = torch.cat(emb_outputs, dim=1)

            # Concatenate numerical and embedded categorical features
            x = torch.cat([x_numerical, x_cat], dim=1)

            return self.mlp(x).squeeze(-1)


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — TRAINING ENGINE
    # ══════════════════════════════════════════════════════════════════════════

    class Trainer:
        """
        Encapsulates the training loop with validation, early stopping,
        and basic metrics tracking.
        """

        def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        ):
            logger.info("Initializing Trainer on device=%s", device)
            self.model = model.to(device)
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device
            self.scheduler = scheduler
            self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
            logger.info("Trainer ready — optimizer=%s, criterion=%s",
                        type(optimizer).__name__, type(criterion).__name__)

        def _run_epoch(self, loader: DataLoader, train: bool = True) -> float:
            """Run one epoch (train or eval)."""
            phase = "train" if train else "val"
            logger.debug("Starting %s epoch (%d batches)", phase, len(loader))
            t0 = time.perf_counter()
            if train:
                self.model.train()
            else:
                self.model.eval()

            total_loss = 0.0
            n_samples = 0

            context = torch.no_grad() if not train else torch.enable_grad()
            with context:
                for batch in loader:
                    x_num = batch["numerical"].to(self.device)
                    x_cat = batch["categorical"].to(self.device)
                    y     = batch["target"].to(self.device)

                    preds = self.model(x_num, x_cat)
                    loss = self.criterion(preds, y)

                    if train:
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                    total_loss += loss.item() * len(y)
                    n_samples += len(y)

            avg_loss = total_loss / n_samples
            logger.debug("%s epoch done in %.3fs — loss=%.4f (%d samples)",
                         phase, time.perf_counter() - t0, avg_loss, n_samples)
            return avg_loss

        def fit(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            n_epochs: int = 50,
            early_stopping_patience: int = 10,
            verbose: bool = True,
        ) -> dict[str, list[float]]:
            """
            Full training loop with early stopping.

            Returns the training history dict.
            """
            logger.info("Starting training: n_epochs=%d, early_stopping_patience=%d", n_epochs, early_stopping_patience)
            fit_t0 = time.perf_counter()
            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            for epoch in range(1, n_epochs + 1):
                epoch_t0 = time.perf_counter()
                train_loss = self._run_epoch(train_loader, train=True)
                val_loss   = self._run_epoch(val_loader, train=False)

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)

                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                lr = self.optimizer.param_groups[0]["lr"]
                logger.info("Epoch %3d/%-3d │ train_loss=%.4f │ val_loss=%.4f │ lr=%.2e │ %.2fs",
                            epoch, n_epochs, train_loss, val_loss, lr, time.perf_counter() - epoch_t0)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    logger.info("New best val_loss=%.4f — saving checkpoint", best_val_loss)
                else:
                    patience_counter += 1
                    logger.debug("No improvement — patience %d/%d", patience_counter, early_stopping_patience)

                if verbose and (epoch % 5 == 0 or epoch == 1):
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"  Epoch {epoch:3d} │ Train Loss: {train_loss:.4f} │ "
                          f"Val Loss: {val_loss:.4f} │ LR: {lr:.2e} │ "
                          f"Patience: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    logger.warning("Early stopping triggered at epoch %d (patience=%d)",
                                   epoch, early_stopping_patience)
                    if verbose:
                        print(f"\n  Early stopping at epoch {epoch}")
                    break

            # Restore best weights
            if best_state is not None:
                self.model.load_state_dict(best_state)
                logger.info("Restored best model weights (val_loss=%.4f)", best_val_loss)
                if verbose:
                    print(f"  Restored best model (val_loss={best_val_loss:.4f})")

            elapsed = time.perf_counter() - fit_t0
            logger.info("Training complete: %d epochs in %.1fs (best val_loss=%.4f)",
                        len(self.history["train_loss"]), elapsed, best_val_loss)
            return self.history

        @torch.no_grad()
        def predict(self, loader: DataLoader) -> np.ndarray:
            """Generate predictions for an entire DataLoader."""
            logger.info("Running inference on %d batches", len(loader))
            t0 = time.perf_counter()
            self.model.eval()
            predictions = []
            for batch in loader:
                x_num = batch["numerical"].to(self.device)
                x_cat = batch["categorical"].to(self.device)
                preds = self.model(x_num, x_cat)
                predictions.append(preds.cpu().numpy())
            result = np.concatenate(predictions)
            logger.info("Inference done in %.3fs — %d predictions", time.perf_counter() - t0, len(result))
            return result


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — INFERENCE UTILITY (for loading & predicting on new data)
    # ══════════════════════════════════════════════════════════════════════════

    def load_and_predict(
        new_df: pd.DataFrame,
        checkpoint_path: str = "model_checkpoint.pt",
        preprocessor_path: str = "preprocessor.joblib",
    ) -> np.ndarray:
        """
        Load a saved model + preprocessor and run inference on new data.

        Usage:
            new_data = pd.DataFrame({...})
            predictions = load_and_predict(new_data)
        """
        import joblib

        logger.info("Loading model from %s and preprocessor from %s", checkpoint_path, preprocessor_path)
        t0 = time.perf_counter()

        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        logger.info("Checkpoint loaded — config: %s", config)

        # Reconstruct model
        model = TabularEmbeddingModel(
            cat_cardinalities=config["cat_cardinalities"],
            cat_embedding_dims=config["cat_embedding_dims"],
            n_numerical=config["n_numerical"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Preprocess new data (fit=False — use saved transform)
        num_features, cat_features, _ = preprocess_data(
            new_df.assign(**{TARGET_COL: 0}),  # dummy target
            preprocessor,
            fit=False,
        )

        # Predict
        dataset = TabularDataset(num_features, cat_features, np.zeros(len(new_df)))
        loader = DataLoader(dataset, batch_size=512, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in loader:
                preds = model(batch["numerical"], batch["categorical"])
                predictions.append(preds.numpy())

        result = np.concatenate(predictions)
        logger.info("load_and_predict done in %.3fs — %d predictions", time.perf_counter() - t0, len(result))
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — UTILITY: COMPUTE EMBEDDING DIMENSIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_embedding_dims(
    df: pd.DataFrame,
    cat_cols: list[str],
    max_dim: int = 50,
) -> tuple[list[int], list[int]]:
    """
    Compute cardinalities and embedding dimensions for categorical columns.

    Rule of thumb: embedding_dim = min(max_dim, (cardinality + 1) // 2)
    This balances expressiveness vs. parameter count.

    Returns:
        cardinalities    : list of unique value counts per column
        embedding_dims   : list of embedding dimensions per column
    """
    logger.info("Computing embedding dimensions for %d categorical columns (max_dim=%d)", len(cat_cols), max_dim)
    cardinalities = []
    embedding_dims = []

    for col in cat_cols:
        card = df[col].nunique()
        dim = min(max_dim, (card + 1) // 2)
        dim = max(dim, 2)  # minimum 2 dims
        cardinalities.append(card)
        embedding_dims.append(dim)
        logger.debug("  %s: cardinality=%d -> emb_dim=%d", col, card, dim)

    logger.info("Embedding dims computed: %s -> %s", cardinalities, embedding_dims)
    return cardinalities, embedding_dims


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN: FULL PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("Pipeline starting")
    pipeline_t0 = time.perf_counter()
    print("=" * 72)
    print("  ML Tabular Pipeline — Mixed Numerical & Categorical Features")
    print("=" * 72)

    # ── Step 1: Generate / Load Data ──────────────────────────────────────
    print("\n[1] Generating synthetic data...")
    df = generate_synthetic_data(n_samples=10_000)
    print(f"    Shape: {df.shape}")
    print(f"    Numerical cols:  {NUMERICAL_COLS}")
    print(f"    Categorical cols: {CATEGORICAL_COLS}")
    print(f"    Target: {TARGET_COL}")
    print(f"\n    Sample rows:")
    print(df.head(3).to_string(index=False))

    # ── Step 2: Train/Val Split ───────────────────────────────────────────
    print("\n[2] Splitting train / validation (80/20)...")
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    print(f"    Train: {len(df_train):,}   Val: {len(df_val):,}")

    # ── Step 3: Compute Embedding Dimensions ──────────────────────────────
    cardinalities, embedding_dims = compute_embedding_dims(df_train, CATEGORICAL_COLS)
    print("\n[3] Categorical embedding plan:")
    for col, card, dim in zip(CATEGORICAL_COLS, cardinalities, embedding_dims):
        print(f"    {col:25s}  cardinality={card:3d}  →  emb_dim={dim}")

    # ── Step 4: Preprocess ────────────────────────────────────────────────
    print("\n[4] Fitting preprocessor on training data...")
    preprocessor = build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS)
    train_num, train_cat, train_y = preprocess_data(df_train, preprocessor, fit=True)
    val_num, val_cat, val_y       = preprocess_data(df_val, preprocessor, fit=False)
    print(f"    Train numericals shape: {train_num.shape}  dtype: {train_num.dtype}")
    print(f"    Train categoricals shape: {train_cat.shape}  dtype: {train_cat.dtype}")

    # ── Step 5: Create Datasets & DataLoaders ─────────────────────────────
    if not TORCH_AVAILABLE:
        print("\n[5] ⏭  Skipping PyTorch model (torch not installed)")
        print("    Preprocessing pipeline validated successfully!")
        print("    Install PyTorch (`pip install torch`) to run the full training loop.")
        print("\n" + "=" * 72)
        print("  Preprocessing pipeline validated — install PyTorch for full training.")
        print("=" * 72)
        return None, preprocessor, {}

    print("\n[5] Building PyTorch Datasets & DataLoaders...")
    train_ds = TabularDataset(train_num, train_cat, train_y)
    val_ds   = TabularDataset(val_num, val_cat, val_y)

    BATCH_SIZE = 256
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE)
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    # ── Step 6: Build Model ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[6] Building model on device: {device}")

    model = TabularEmbeddingModel(
        cat_cardinalities=cardinalities,
        cat_embedding_dims=embedding_dims,
        n_numerical=len(NUMERICAL_COLS),
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        output_dim=1,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"\n    Model architecture:\n{model}")

    # ── Step 7: Configure Training ────────────────────────────────────────
    print("\n[7] Configuring optimizer, scheduler, and loss...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    # ── Step 8: Train ─────────────────────────────────────────────────────
    print("\n[8] Training...")
    print("-" * 72)
    trainer = Trainer(model, optimizer, criterion, device, scheduler)
    history = trainer.fit(
        train_loader,
        val_loader,
        n_epochs=80,
        early_stopping_patience=15,
        verbose=True,
    )
    print("-" * 72)

    # ── Step 9: Evaluate ──────────────────────────────────────────────────
    logger.info("Step 9: Evaluating on validation set")
    print("\n[9] Final evaluation on validation set...")
    val_preds = trainer.predict(val_loader)
    val_true  = val_y

    mse  = np.mean((val_preds - val_true) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(val_preds - val_true))
    ss_res = np.sum((val_true - val_preds) ** 2)
    ss_tot = np.sum((val_true - np.mean(val_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    logger.info("Evaluation metrics — MSE=%.2f  RMSE=%.2f  MAE=%.2f  R2=%.4f", mse, rmse, mae, r2)
    print(f"    MSE:  {mse:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    MAE:  {mae:.2f}")
    print(f"    R²:   {r2:.4f}")

    # ── Step 10: Show sample predictions ──────────────────────────────────
    print(f"\n[10] Sample predictions (first 10):")
    print(f"    {'Predicted':>12s}  {'Actual':>12s}  {'Error':>10s}")
    print(f"    {'─'*12}  {'─'*12}  {'─'*10}")
    for pred, true in zip(val_preds[:10], val_true[:10]):
        print(f"    {pred:12.1f}  {true:12.1f}  {pred - true:10.1f}")

    # ── Step 11: Save artifacts ───────────────────────────────────────────
    logger.info("Step 11: Saving artifacts")
    print("\n[11] Saving model checkpoint & preprocessor...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": {
            "cat_cardinalities": cardinalities,
            "cat_embedding_dims": embedding_dims,
            "n_numerical": len(NUMERICAL_COLS),
            "hidden_dims": [256, 128, 64],
            "dropout": 0.3,
            "numerical_cols": NUMERICAL_COLS,
            "categorical_cols": CATEGORICAL_COLS,
        },
    }
    torch.save(checkpoint, "model_checkpoint.pt")
    logger.info("Saved model checkpoint: model_checkpoint.pt")
    print("    Saved: model_checkpoint.pt")

    # Save preprocessor with joblib for later inference
    import joblib
    joblib.dump(preprocessor, "preprocessor.joblib")
    logger.info("Saved preprocessor: preprocessor.joblib")
    print("    Saved: preprocessor.joblib")

    elapsed = time.perf_counter() - pipeline_t0
    logger.info("Pipeline complete in %.1fs", elapsed)
    print("\n" + "=" * 72)
    print(f"  Pipeline complete in {elapsed:.1f}s.")
    print("=" * 72)

    return model, preprocessor, history


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model, preprocessor, history = main()
