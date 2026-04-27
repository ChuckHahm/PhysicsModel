# PyTorch Time Series Data Architecture for RUL / Failure Analysis

**Context:** Battery fleet remaining useful life (RUL) modeling for Li/SOCl₂ and Li/MnO₂
chemistries. Physical parameters include voltage, temperature, internal resistance, load
current, date of manufacture, and geographic installation zone.

---

## 1. Three-Tier Feature Taxonomy

Features in physical device failure datasets live at different *granularities*. Respecting
this hierarchy avoids memory waste and preserves temporal structure.

| Tier | Examples | Shape per device | PyTorch dtype |
|---|---|---|---|
| **Static covariates** | Chemistry type, manufacture date, geographic zone, lot | `(C,)` | `int64` (categoricals) / `float32` (conts) |
| **Slowly-varying covariates** | Calendar age, cumulative duty cycle, seasonal temp quartile | `(N, F_slow)` | `float32` |
| **Fast time-series** | Voltage, temperature, R_int, I_load at fixed interval | `(N, F_seq)` | `float32` |

Key design decisions:

- **Geographic encoding:** Do not use raw lat/lon. Map to ASHRAE climate zone, a 3-bin ordinal (coastal/inland/desert for SDG&E), or a learned `nn.Embedding` if you have 20+ devices per geographic cell.
- **Normalisation:** Z-score `x_seq` *per device*, using only observations before the current window start (causal — no future leakage). Apply global normalisation only to slow covariates like calendar age.
- **Static categoricals** (chemistry, region, lot) → `nn.Embedding` layers in the model, not one-hot encoding.

---

## 2. Data Contracts

### `DeviceRecord` dataclass

Holds all data for one physical device, pre-loaded into RAM.

```python
@dataclass
class DeviceRecord:
    ts_features:   np.ndarray   # (N, F_seq)  float32  — V, T, R_int, I_load
    slow_features: np.ndarray   # (N, F_slow) float32  — age, duty cycle, season
    static_cats:   np.ndarray   # (C,)        int64    — chemistry, region, lot
    static_conts:  np.ndarray   # (K,)        float32  — nominal capacity, install temp
    timestamps:    np.ndarray   # (N,)        int64    — Unix epoch seconds
    failed:        bool                                 # True = failure observed
    failure_cycle: Optional[int]                        # index of failure; None if censored
    rul_at_obs:    np.ndarray   # (N,)        float32  — NaN in censored tail
    device_id:     str
```

### `RULSample` — one training sample from `__getitem__`

```python
@dataclass
class RULSample:
    x_seq:          Tensor   # (T, F_seq)   sequence window
    x_slow:         Tensor   # (F_slow,)    slow covariates at window end
    x_static_cats:  Tensor   # (C,)         int64 codes for embedding lookup
    x_static_conts: Tensor   # (K,)         continuous statics
    label:          Tensor   # scalar       RUL in cycles; NaN if censored
    censored:       Tensor   # scalar bool  True = device had not failed by window end
    event_observed: Tensor   # scalar bool  True = failure within this window
    device_id:      str
    window_end_idx: int      # position in original series (for debugging)
```

---

## 3. `SlidingWindowDataset`

### Window indexing

```
window_start = burn_in + k * stride,   k = 0, 1, 2, …
window_end   = window_start + window_len - 1
```

`burn_in` guarantees every window has sufficient prefix for recurrent state initialisation.

### Label strategies

| `label_type` | Label value | Censored handling |
|---|---|---|
| `'regression'` | `rul_at_obs[window_end_idx]` | NaN — loss masks these out |
| `'binary'` | 1.0 if failure within `horizon` cycles, 0.0 otherwise | NaN if censored before horizon |
| `'weibull'` | Time to event from window end | Censored flag carries survival contribution |

### Causal per-device normalisation

```python
def _causal_normalise(self, rec, window_start, x_seq):
    prefix = rec.ts_features[:window_start]      # only history before this window
    mu  = prefix.mean(axis=0)
    std = prefix.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)        # guard zero-variance channels
    return (x_seq - mu) / std
```

Without this, normalisation stats from future observations bleed backward — the model
implicitly sees which batteries degrade the most.

### Key constructor parameters

```python
SlidingWindowDataset(
    records,                    # List[DeviceRecord]
    window_len     = 48,        # T — tune per chemistry (SOCl₂ needs longer windows)
    stride         = 12,        # step between windows; sweep with Optuna
    burn_in        = 24,        # prefix before first window
    label_type     = 'regression',
    horizon        = None,      # required for label_type='binary'
    per_device_norm= True,      # causal z-score per device
    drop_censored  = False,     # set True for pure regression; loses survival info
)
```

> **Chemistry note:** Li/SOCl₂ cells show a long flat discharge curve with abrupt end-of-life.
> Longer windows (capturing the inflection) are more diagnostic than for Li/MnO₂. Make
> `window_len` an Optuna search parameter.

---

## 4. Censoring

Right-censored observations (devices still operating at extraction time) must be handled
from the start. Ignoring them biases RUL estimates downward.

```python
# Every __getitem__ call returns:
censored       = True   # device had not failed by window end
event_observed = False  # failure not yet seen

# Loss function usage:
known  = ~torch.isnan(label) & ~censored
loss_r = F.mse_loss(rul_hat[known], label[known])         # regression term
loss_s = survival_loss(rul_hat[~known], censored[~known]) # survival term
```

---

## 5. `collate_fn`

Assembles a list of `RULSample` objects into padded batch tensors.

```python
def rul_collate_fn(batch: List[RULSample]) -> Dict[str, Tensor]:
    B     = len(batch)
    T_max = max(s.x_seq.shape[0] for s in batch)
    F_seq = batch[0].x_seq.shape[1]

    x_seq_padded = torch.zeros(B, T_max, F_seq)
    padding_mask = torch.ones(B, T_max, dtype=torch.bool)   # True = padding

    for i, s in enumerate(batch):
        T = s.x_seq.shape[0]
        x_seq_padded[i, -T:] = s.x_seq     # right-align: recent obs at end
        padding_mask[i, -T:] = False        # real positions

    return {
        'x_seq':           x_seq_padded,           # (B, T_max, F_seq)
        'x_slow':          torch.stack([...]),      # (B, F_slow)
        'x_static_cats':   torch.stack([...]),      # (B, C)       int64
        'x_static_conts':  torch.stack([...]),      # (B, K)
        'label':           torch.stack([...]),      # (B,)
        'censored':        torch.stack([...]),      # (B,)   bool
        'event_observed':  torch.stack([...]),      # (B,)   bool
        'padding_mask':    padding_mask,            # (B, T_max) bool
        'device_ids':      [...],                   # List[str]
    }
```

**Right-alignment** puts the most recent observations adjacent across the batch — correct
for LSTMs (use the final hidden state) and Transformers (`src_key_padding_mask`).

---

## 6. How DataLoader Uses Dataset

The DataLoader only ever calls **two methods** on the Dataset:

| Method | When called | Purpose |
|---|---|---|
| `__len__(self) → int` | Once at startup | Sampler needs index range `[0, N)` |
| `__getitem__(self, idx) → any` | Once per sample per batch | Slice + convert to tensors |

It never reads `.records`, `.ts_features`, or any other attribute.

### Call sequence for one batch

```
1. BatchSampler   — collects batch_size indices from Sampler: [312, 7, 88, 201]
2. fetch loop     — calls dataset[312], dataset[7], … (in parallel if num_workers > 0)
3. __getitem__    — looks up _index[idx], slices numpy, normalises, calls torch.tensor()
4. collate_fn     — pads, stacks, builds padding_mask → batch dict
5. pin_memory     — copies tensors to page-locked RAM (if pin_memory=True)
6. __next__       — yields batch dict to training loop
```

While the model runs forward + backward on batch N, workers are already fetching batch N+1
(prefetch_factor=2 by default).

### Sampler options

```python
RandomSampler(dataset)                    # training: shuffled order
SequentialSampler(dataset)                # validation: reproducible order

WeightedRandomSampler(                    # near-failure oversampling
    weights=compute_rul_weights(dataset), # upweight windows with RUL < threshold
    num_samples=len(dataset),
    replacement=True,
)
```

### DataLoader construction

```python
loader = DataLoader(
    dataset            = ds,
    batch_size         = 32,
    shuffle            = True,           # = RandomSampler under the hood
    collate_fn         = rul_collate_fn,
    num_workers        = 4,              # parallel __getitem__ workers
    pin_memory         = True,           # GPU transfer speed
    drop_last          = True,           # skip final undersized batch
    persistent_workers = True,           # keep workers alive between epochs
)
```

---

## 7. Model Input Layout

```
x_seq   (B, T, F_seq)  → sequence encoder (LSTM / TCN / Transformer)
                          → h_T: (B, hidden_dim)

x_static_cats  (B, C)  → nn.Embedding per categorical → flatten → emb: (B, C×emb_dim)
x_static_conts (B, K)  → normalised, concat directly

fused = torch.cat([h_T, emb, x_slow], dim=-1)   # (B, hidden + C×emb + F_slow)
rul_hat = head(fused)                            # (B, 1)
```

`padding_mask` is passed to the encoder as `src_key_padding_mask` (Transformer) or used
to derive sequence lengths for `pack_padded_sequence` (LSTM).

---

## 8. Memory Footprint Reference

For batch_size=32, window_len=48, F_seq=4:

```
x_seq:   32 × 48 × 4 × 4 bytes  = 24,576 bytes  ≈ 24 KB
x_slow:  32 × 3      × 4 bytes  =    384 bytes
x_cats:  32 × 3      × 8 bytes  =    768 bytes
label:   32          × 4 bytes  =    128 bytes
─────────────────────────────────────────────────────
Total per batch ≈ 25 KB  (model weights dominate GPU memory, not batch data)
```

`float16` / `bfloat16` halves memory and doubles throughput on Tensor Core GPUs — viable
for large fleet training runs.

---

## 9. Unit Test Checklist

Run before any training run to catch shape/dtype bugs early:

```python
# T1: Dataset constructs without error
# T2: __getitem__ returns correct tensor shapes
#     x_seq: (T, F_seq), x_slow: (F_slow,), label: scalar, censored: bool
# T3: collate_fn batch shapes correct
#     x_seq: (B, T, F_seq), padding_mask: (B, T) bool
# T4: padding_mask is left-aligned (no padding after real data)
# T5: causal normalisation varies across windows of same device
# T6: censoring_rate() ∈ [0, 1]
# T7: drop_censored=True leaves zero censored samples
# T8: binary label values ∈ {0.0, 1.0, NaN}
# T9: feature_dims helper returns correct dict
```

---

## 10. Files in This Project

| File | Description |
|---|---|
| `sliding_window_rul_dataset.py` | Complete `SlidingWindowDataset`, `rul_collate_fn`, synthetic data factory, unit tests |
| `pytorch_ts_data_architecture.md` | This document |

---

*Generated from conversation: PyTorch time series data architecture for RUL / failure analysis · April 2026*
