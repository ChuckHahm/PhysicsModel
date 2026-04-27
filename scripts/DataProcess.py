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

import numpy as np
import pandas as pd
from typing import Optional

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

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COLUMN DEFINITIONS & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# Define column groups — edit these for your own dataset
NUMERICAL_COLS  = ["voltage", "temperature", "cycle_count", "impedance", "load_current"]
CATEGORICAL_COLS = ["manufacturer", "battery_type", "installation_region", "failure_mode"]
TARGET_COL       = "remaining_useful_life"


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
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
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
    if fit:
        transformed = preprocessor.fit_transform(df)
    else:
        transformed = preprocessor.transform(df)

    n_num = len(NUMERICAL_COLS)
    num_features = transformed[:, :n_num].astype(np.float32)
    cat_features = transformed[:, n_num:].astype(np.int64)
    targets      = df[TARGET_COL].values.astype(np.float32)

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
            self.num_features = torch.tensor(num_features, dtype=torch.float32)
            self.cat_features = torch.tensor(cat_features, dtype=torch.long)
            self.targets      = torch.tensor(targets, dtype=torch.float32)

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
                layers.append(nn.Dropout(dropout))"""
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

import numpy as np
import pandas as pd
from typing import Optional

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

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COLUMN DEFINITIONS & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# Define column groups — edit these for your own dataset
NUMERICAL_COLS  = ["voltage", "temperature", "cycle_count", "impedance", "load_current"]
CATEGORICAL_COLS = ["manufacturer", "battery_type", "installation_region", "failure_mode"]
TARGET_COL       = "remaining_useful_life"


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
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
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
    if fit:
        transformed = preprocessor.fit_transform(df)
    else:
        transformed = preprocessor.transform(df)

    n_num = len(NUMERICAL_COLS)
    num_features = transformed[:, :n_num].astype(np.float32)
    cat_features = transformed[:, n_num:].astype(np.int64)
    targets      = df[TARGET_COL].values.astype(np.float32)

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
            self.num_features = torch.tensor(num_features, dtype=torch.float32)
            self.cat_features = torch.tensor(cat_features, dtype=torch.long)
            self.targets      = torch.tensor(targets, dtype=torch.float32)

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
            self.model = model.to(device)
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device
            self.scheduler = scheduler
            self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        def _run_epoch(self, loader: DataLoader, train: bool = True) -> float:
            """Run one epoch (train or eval)."""
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

            return total_loss / n_samples

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
            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            for epoch in range(1, n_epochs + 1):
                train_loss = self._run_epoch(train_loader, train=True)
                val_loss   = self._run_epoch(val_loader, train=False)

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)

                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if verbose and (epoch % 5 == 0 or epoch == 1):
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"  Epoch {epoch:3d} │ Train Loss: {train_loss:.4f} │ "
                          f"Val Loss: {val_loss:.4f} │ LR: {lr:.2e} │ "
                          f"Patience: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n  ⏹  Early stopping at epoch {epoch}")
                    break

            # Restore best weights
            if best_state is not None:
                self.model.load_state_dict(best_state)
                if verbose:
                    print(f"  ✓ Restored best model (val_loss={best_val_loss:.4f})")

            return self.history

        @torch.no_grad()
        def predict(self, loader: DataLoader) -> np.ndarray:
            """Generate predictions for an entire DataLoader."""
            self.model.eval()
            predictions = []
            for batch in loader:
                x_num = batch["numerical"].to(self.device)
                x_cat = batch["categorical"].to(self.device)
                preds = self.model(x_num, x_cat)
                predictions.append(preds.cpu().numpy())
            return np.concatenate(predictions)


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

        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

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

        return np.concatenate(predictions)


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
    cardinalities = []
    embedding_dims = []

    for col in cat_cols:
        card = df[col].nunique()
        dim = min(max_dim, (card + 1) // 2)
        dim = max(dim, 2)  # minimum 2 dims
        cardinalities.append(card)
        embedding_dims.append(dim)

    return cardinalities, embedding_dims


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN: FULL PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
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
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
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
    print("\n[9] Final evaluation on validation set...")
    val_preds = trainer.predict(val_loader)
    val_true  = val_y

    mse  = np.mean((val_preds - val_true) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(val_preds - val_true))
    ss_res = np.sum((val_true - val_preds) ** 2)
    ss_tot = np.sum((val_true - np.mean(val_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

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
    print("    ✓ Saved: model_checkpoint.pt")

    # Save preprocessor with joblib for later inference
    import joblib
    joblib.dump(preprocessor, "preprocessor.joblib")
    print("    ✓ Saved: preprocessor.joblib")

    print("\n" + "=" * 72)
    print("  Pipeline complete.")
    print("=" * 72)

    return model, preprocessor, history


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model, preprocessor, history = main()
                prev_dim = h_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.mlp = nn.Sequential(*layers)

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
            self.model = model.to(device)
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device
            self.scheduler = scheduler
            self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        def _run_epoch(self, loader: DataLoader, train: bool = True) -> float:
            """Run one epoch (train or eval)."""
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

            return total_loss / n_samples

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
            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            for epoch in range(1, n_epochs + 1):
                train_loss = self._run_epoch(train_loader, train=True)
                val_loss   = self._run_epoch(val_loader, train=False)

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)

                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if verbose and (epoch % 5 == 0 or epoch == 1):
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"  Epoch {epoch:3d} │ Train Loss: {train_loss:.4f} │ "
                          f"Val Loss: {val_loss:.4f} │ LR: {lr:.2e} │ "
                          f"Patience: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n  ⏹  Early stopping at epoch {epoch}")
                    break

            # Restore best weights
            if best_state is not None:
                self.model.load_state_dict(best_state)
                if verbose:
                    print(f"  ✓ Restored best model (val_loss={best_val_loss:.4f})")

            return self.history

        @torch.no_grad()
        def predict(self, loader: DataLoader) -> np.ndarray:
            """Generate predictions for an entire DataLoader."""
            self.model.eval()
            predictions = []
            for batch in loader:
                x_num = batch["numerical"].to(self.device)
                x_cat = batch["categorical"].to(self.device)
                preds = self.model(x_num, x_cat)
                predictions.append(preds.cpu().numpy())
            return np.concatenate(predictions)


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

        # Load preprocessor
        preprocessor = joblib.load(preprocessor_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

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

        return np.concatenate(predictions)


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
    cardinalities = []
    embedding_dims = []

    for col in cat_cols:
        card = df[col].nunique()
        dim = min(max_dim, (card + 1) // 2)
        dim = max(dim, 2)  # minimum 2 dims
        cardinalities.append(card)
        embedding_dims.append(dim)

    return cardinalities, embedding_dims


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN: FULL PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
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
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
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
    print("\n[9] Final evaluation on validation set...")
    val_preds = trainer.predict(val_loader)
    val_true  = val_y

    mse  = np.mean((val_preds - val_true) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(val_preds - val_true))
    ss_res = np.sum((val_true - val_preds) ** 2)
    ss_tot = np.sum((val_true - np.mean(val_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

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
    print("    ✓ Saved: model_checkpoint.pt")

    # Save preprocessor with joblib for later inference
    import joblib
    joblib.dump(preprocessor, "preprocessor.joblib")
    print("    ✓ Saved: preprocessor.joblib")

    print("\n" + "=" * 72)
    print("  Pipeline complete.")
    print("=" * 72)

    return model, preprocessor, history


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model, preprocessor, history = main()