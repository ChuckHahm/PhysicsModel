# Transform Hypotheses — Battery RUL Pipeline
<!-- CLAUDE CODE INSTRUCTIONS (read before acting on any hypothesis)
  - Each hypothesis has a unique ID: HYP-NNN
  - Status values: proposed | implemented | tested | rejected | deferred
  - Before implementing any HYP: read its Statement, Rationale, and Test Condition
  - Name all transforms:  transform_<id_lowercase>()  e.g. transform_hyp001()
  - Name all test files:  test_hyp<NNN>.py
  - Name all test funcs:  test_hyp<NNN>_<short_description>()
  - After tests pass: update Status and add a Notes entry with date and result
  - If a hypothesis is rejected: document WHY in Notes — do not delete the entry
-->

**Domain:** Li/SOCl₂ and Li/MnO₂ battery fleet · RUL prediction · physical time-series  
**Project:** `/home/chuck/GitClones/DataQuality`  
**Last updated:** 2026-04-25

---

## Status Summary

| ID | Short name | Status | Priority |
|---|---|---|---|
| HYP-001 | voltage-slope degradation | proposed | high |
| HYP-002 | resistance inflection | proposed | high |
| HYP-003 | thermal stress accumulation | proposed | high |
| HYP-004 | knee-point detection | proposed | high |
| HYP-005 | causal per-device z-score | proposed | high |
| HYP-006 | chemistry-conditional normalisation | proposed | medium |
| HYP-007 | geographic thermal bias correction | proposed | medium |
| HYP-008 | manufacture-cohort baseline | proposed | medium |
| HYP-009 | censored window imputation | proposed | low |
| HYP-010 | multi-scale window fusion | proposed | low |

---

## HYP-001 · Voltage Slope as Degradation Signal

**Status:** `proposed`  
**Priority:** high  
**Chemistry:** both Li/SOCl₂ and Li/MnO₂

### Statement
The slope of voltage over a sliding window is a stronger degradation feature than
the raw voltage level. As a cell approaches end-of-life, voltage decline accelerates.
A rolling linear regression slope (dV/dt) computed within each window will carry
monotonically increasing predictive signal for RUL.

### Rationale
- Raw voltage is chemistry-dependent and installation-voltage-dependent; slope is
  more transferable across devices
- The acceleration of voltage decline (second derivative) marks the knee point
  that precedes failure in both chemistries
- Slope features are robust to sensor offset drift that affects absolute readings

### Transform Signature
```python
def transform_hyp001(
    x_seq: np.ndarray,     # (T, F) — voltage is column index 0
    window: int = 10,      # rolling window for slope estimate
) -> np.ndarray:           # (T, F+1) — appends dV/dt column
    """
    Appends a rolling voltage slope feature (dV/dt) to x_seq.
    Uses least-squares slope over the last `window` timesteps.
    Pads the first (window-1) rows with the first valid slope value.
    """
```

### Test Condition
- **Positive:** On a synthetic device with linear voltage decay, dV/dt is constant
  and negative throughout; slope magnitude increases as decay rate increases
- **Positive:** On a synthetic device with quadratic decay (accelerating), dV/dt
  becomes more negative over time
- **Negative:** On a flat voltage trace (no degradation), dV/dt ≈ 0.0 within
  numerical tolerance
- **Edge case:** Window larger than sequence length raises `ValueError`
- **Edge case:** NaN in voltage column propagates correctly (does not silently zero)

### Notes
<!-- Add dated entries here as work progresses -->
<!-- Example: 2026-04-25: implemented, all tests pass. Slope feature improves -->
<!--          LightGBM validation MAE by 4.2% on MnO2 held-out fleet.        -->

---

## HYP-002 · Internal Resistance Inflection as Failure Precursor

**Status:** `proposed`  
**Priority:** high  
**Chemistry:** Li/SOCl₂ primary (inflection more pronounced than MnO₂)

### Statement
Internal resistance (R_int) follows a J-curve over device lifetime: slow increase
during normal operation, then rapid acceleration near end-of-life. The point of
maximum curvature (inflection) in R_int is a reliable failure precursor. A transform
that computes the second derivative of R_int and flags when it exceeds a
chemistry-specific threshold will produce a binary precursor feature.

### Rationale
- R_int increase is caused by electrolyte depletion and SEI layer growth —
  physical processes that accelerate as failure approaches
- The inflection point typically occurs 10–20% of remaining life before failure
  in Li/SOCl₂; earlier detection allows operational response
- Second derivative is more specific than slope alone (slope increases throughout
  life; curvature peaks only at the inflection)

### Transform Signature
```python
def transform_hyp002(
    x_seq: np.ndarray,         # (T, F) — R_int is column index 2
    smoothing_window: int = 5, # Savitzky-Golay or rolling mean before diff
    threshold: float = 0.01,   # curvature threshold for binary flag
) -> np.ndarray:               # (T, F+2) — appends d2R/dt2 and binary flag
    """
    Appends the second derivative of R_int and a binary inflection flag.
    Smooths R_int before differencing to suppress noise.
    """
```

### Test Condition
- **Positive:** On J-curve R_int trace, second derivative peaks near the inflection
  point and returns toward zero after the curve flattens
- **Positive:** Binary flag is 0 for the early-life flat region and 1 after
  inflection for a clearly J-curved trace
- **Negative:** On linearly increasing R_int (no inflection), second derivative
  is ≈ 0 throughout and binary flag stays 0
- **Edge case:** Noisy R_int trace (Gaussian noise added) — smoothing window
  should suppress noise without shifting the inflection point by more than
  2 timesteps

### Notes

---

## HYP-003 · Thermal Stress Accumulation

**Status:** `proposed`  
**Priority:** high  
**Chemistry:** both

### Statement
High-temperature excursions accelerate battery degradation non-linearly (Arrhenius
relationship). A cumulative thermal stress index — the running integral of
temperature above a baseline threshold — will be a stronger predictor of RUL than
instantaneous temperature alone.

### Rationale
- Arrhenius: reaction rate ∝ exp(-Ea / kT); small temperature increases cause
  large degradation rate increases
- Cumulative exposure (degree-days above threshold) captures the history of
  thermal abuse that instantaneous T misses
- Relevant to SDG&E geography: coastal vs. inland vs. desert installations have
  very different thermal profiles

### Transform Signature
```python
def transform_hyp003(
    x_seq: np.ndarray,         # (T, F) — temperature is column index 1
    baseline_temp: float = 25.0,  # °C — threshold above which stress accumulates
    clip_negative: bool = True,   # if True, negative (cool) excursions = 0
) -> np.ndarray:               # (T, F+1) — appends cumulative thermal stress index
    """
    Appends the cumulative sum of max(0, T - baseline_temp) up to each timestep.
    Represents total thermal dose experienced by the device.
    """
```

### Test Condition
- **Positive:** Constant temperature at baseline → cumulative index stays 0
- **Positive:** Temperature 10°C above baseline for N steps → index = 10*N
- **Positive:** Temperature oscillating above/below baseline →
  index increases only during hot periods (when clip_negative=True)
- **Negative:** Temperature always below baseline → index stays 0 throughout
- **Edge case:** baseline_temp=0 → index always increases (sanity check)
- **Chemistry note:** Test separately for coastal (low thermal stress) vs.
  desert (high) synthetic profiles

### Notes

---

## HYP-004 · Knee-Point Detection Transform

**Status:** `proposed`  
**Priority:** high  
**Chemistry:** both (knee shape differs)

### Statement
Both chemistries exhibit a "knee point" — a rapid change in the rate of voltage
decline — that marks the transition from normal degradation to accelerated
end-of-life. Detecting this knee point within a window and expressing the
time-since-knee as a feature will provide the model with an explicit
remaining-life reference.

### Rationale
- Knee-point detection is well-studied in Li-ion literature (Dahn group);
  analogous physics apply to primary Li cells
- Time-since-knee is more interpretable than raw voltage slope and more
  robust to between-device variability
- If no knee is detected within the window, the feature is 0 (knee not yet reached)

### Transform Signature
```python
def transform_hyp004(
    x_seq: np.ndarray,         # (T, F) — voltage column index 0
    method: str = 'curvature', # 'curvature' | 'piecewise'
    min_knee_drop: float = 0.05,  # minimum voltage drop to qualify as knee
) -> np.ndarray:               # (T, F+2) — appends knee_detected (bool) and
                               #            time_since_knee (int, 0 if not detected)
    """
    Detects voltage knee point within the window.
    'curvature': maximum curvature (second derivative) of smoothed voltage.
    'piecewise': two-segment linear fit; knee = breakpoint of best fit.
    """
```

### Test Condition
- **Positive (curvature):** Synthetic voltage with known breakpoint at t=30 →
  knee detected within ±3 timesteps of t=30
- **Positive (piecewise):** Two-segment linear trace with known breakpoint →
  knee detected at the exact breakpoint
- **Negative:** Linearly declining voltage with no knee → `knee_detected=False`
  for all timesteps
- **Edge case:** Multiple local curvature maxima → only the largest is selected
- **Comparison test:** Both methods should agree on knee location within 5
  timesteps on a clean synthetic trace

### Notes

---

## HYP-005 · Causal Per-Device Z-Score Normalisation

**Status:** `proposed`  
**Priority:** high  
**Chemistry:** both

### Statement
Normalising time-series features using only observations *before* the current
window start (causal normalisation) prevents future information leakage and
produces device-relative features that are more transferable across devices than
global normalisation. The normalised value answers "how unusual is this reading
relative to this device's own history?"

### Rationale
- Global normalisation conflates chemistry differences with degradation signal:
  a voltage of 3.2V means different things for SOCl₂ vs. MnO₂
- Per-device normalisation captures deviation from each device's baseline,
  which is the physically meaningful signal
- Causal constraint (no future data) is required for deployment; using future
  data during training causes inflated validation metrics that don't hold at inference

### Transform Signature
```python
def transform_hyp005(
    x_seq: np.ndarray,         # (T, F) — full device time series
    window_start: int,         # first index of the current window
    min_prefix: int = 2,       # minimum observations needed; fallback to device stats
    epsilon: float = 1e-8,     # guard for zero-variance channels
) -> np.ndarray:               # (T_window, F) — normalised window slice
    """
    Z-scores the window [window_start : window_start+T] using statistics
    computed only from x_seq[:window_start] (strictly causal).
    Falls back to whole-device stats if prefix is shorter than min_prefix.
    """
```

### Test Condition
- **Positive:** Two consecutive windows from the same device should have
  different normalised means (statistics evolve as window_start advances)
- **Positive:** With a prefix of all-constant values followed by a step change,
  the normalised window correctly identifies the step as a deviation
- **Negative (leakage check):** Shuffling future observations into the prefix
  must change the normalised output — confirms causal constraint is enforced
- **Edge case:** window_start=0 (no prefix) → falls back to device-level stats
- **Edge case:** Zero-variance channel (constant temperature in prefix) →
  normalised output is 0.0 (not NaN, not inf)

### Notes

---

## HYP-006 · Chemistry-Conditional Normalisation Bounds

**Status:** `proposed`  
**Priority:** medium  
**Chemistry:** both — but parameters differ

### Statement
The physically meaningful range of each feature differs by chemistry. Voltage for
Li/SOCl₂ operates in a different band than Li/MnO₂. A chemistry-conditional
clipping transform applied *before* z-scoring will prevent outlier measurements
(sensor glitches, installation anomalies) from distorting device-level statistics.

### Rationale
- Sensor glitches (voltage spike to 99V) in one window corrupt the z-score
  mean/std for all subsequent windows on that device
- Chemistry-specific bounds are known from datasheets; encoding them as a
  transform makes the constraint explicit and auditable
- Clip before normalise (not after) to keep statistics clean

### Transform Signature
```python
CHEMISTRY_BOUNDS = {
    'LiSOCl2': {
        'voltage':     (2.0, 3.9),    # V
        'temperature': (-40.0, 85.0), # °C
        'resistance':  (0.0, 5.0),    # Ω
        'current':     (0.0, 2.0),    # A
    },
    'LiMnO2': {
        'voltage':     (2.0, 3.3),
        'temperature': (-20.0, 60.0),
        'resistance':  (0.0, 3.0),
        'current':     (0.0, 3.0),
    },
}

def transform_hyp006(
    x_seq: np.ndarray,         # (T, F)
    chemistry: str,            # 'LiSOCl2' | 'LiMnO2'
    col_order: list[str],      # e.g. ['voltage','temperature','resistance','current']
) -> np.ndarray:               # (T, F) — clipped in-place copy
    """
    Clips each feature column to chemistry-specific physical bounds.
    Out-of-bounds values are clipped (not zeroed, not NaN'd).
    """
```

### Test Condition
- **Positive:** Voltage spike to 99V in LiSOCl₂ trace → clipped to 3.9V
- **Positive:** Temperature of -60°C → clipped to -40.0°C
- **Negative:** Value within bounds → unchanged
- **Error case:** Unknown chemistry string → raises `ValueError` with message
  listing valid options
- **Audit check:** Fraction of clipped values per column is logged as a
  transform diagnostic (useful for data quality assessment)

### Notes

---

## HYP-007 · Geographic Thermal Bias Correction

**Status:** `proposed`  
**Priority:** medium  
**Chemistry:** both

### Statement
Devices installed in different SDG&E geographic zones (coastal, inland valley,
desert) operate at systematically different ambient temperatures. A geographic
bias correction — subtracting the long-run zone mean temperature — will produce
temperature features that represent *deviation from local normal* rather than
absolute temperature, making the model transferable across zones.

### Rationale
- A device at 30°C in a desert zone is operating normally; the same temperature
  for a coastal device is a thermal excursion — absolute temperature conflates these
- Zone mean temperatures are computable from historical installation data;
  they are stable, slowly-varying covariates not requiring per-device history
- This correction is analogous to anomaly detection in climate science
  (temperature anomaly = observed - climatological mean)

### Transform Signature
```python
ZONE_TEMP_MEANS = {
    'coastal':       18.5,   # °C  (approx San Diego coast)
    'inland_valley': 24.0,   # °C  (approx El Cajon / Santee)
    'desert':        32.0,   # °C  (approx Borrego Springs)
}

def transform_hyp007(
    x_seq: np.ndarray,         # (T, F) — temperature is column index 1
    zone: str,                 # 'coastal' | 'inland_valley' | 'desert'
) -> np.ndarray:               # (T, F) — temperature column replaced with anomaly
    """
    Replaces raw temperature with (T_obs - T_zone_mean).
    Positive values = warmer than zone normal; negative = cooler.
    """
```

### Test Condition
- **Positive:** Coastal device at 18.5°C → temperature anomaly = 0.0
- **Positive:** Desert device at 42.0°C → anomaly = 10.0
- **Positive:** Same absolute temperature, different zone → different anomaly values
- **Error case:** Unknown zone string → raises `ValueError`
- **Integration:** Combined with HYP-003, thermal stress should be computed
  on the *anomaly* temperature (deviation from zone mean), not the raw temperature

### Notes

---

## HYP-008 · Manufacture-Cohort Baseline Correction

**Status:** `proposed`  
**Priority:** medium  
**Chemistry:** both

### Statement
Devices from the same manufacture lot share baseline characteristics (initial
capacity, initial resistance) that differ from other lots due to manufacturing
variation. Subtracting the cohort mean from each device's features at the start
of life will separate manufacturing variation from degradation signal, improving
between-lot generalisation.

### Rationale
- A device with high initial resistance isn't degraded — it was built that way
- Without cohort correction, the model learns lot-specific offsets rather than
  universal degradation patterns
- Cohort means are computable from early-life observations (first N cycles)
  before significant degradation begins

### Transform Signature
```python
def transform_hyp008(
    x_seq: np.ndarray,         # (T, F) — full device time series
    cohort_means: np.ndarray,  # (F,)   — mean of first-N-cycle values for this lot
    apply_cols: list[int],     # column indices to correct (e.g. [0, 2] for V, R)
) -> np.ndarray:               # (T, F) — cohort-corrected copy
    """
    Subtracts cohort_means from specified columns.
    cohort_means are computed externally from lot-level early-life statistics.
    """
```

### Test Condition
- **Positive:** Two devices from different lots with same degradation trajectory
  but different baselines → after correction, their features align
- **Positive:** Device with cohort mean applied → early-life values ≈ 0
- **Negative:** Column not in apply_cols → unchanged
- **Edge case:** cohort_means computed from fewer than 5 devices → warning logged

### Notes

---

## HYP-009 · Censored Window Feature Imputation

**Status:** `proposed`  
**Priority:** low  
**Chemistry:** both

### Statement
Censored devices (still operating at extraction time) provide no direct RUL label,
but their feature windows are still valid training signal for the encoder. A
transform that computes a "projected RUL" estimate for censored windows — based
on the device's degradation trajectory extrapolated forward — may allow these
windows to contribute to regression loss rather than being relegated to the
survival term only.

### Rationale
- Censored windows represent ~25–30% of fleet observations; excluding them from
  regression loss wastes significant training signal
- Extrapolated RUL is noisy but directionally correct; adding it with a lower
  loss weight (e.g. 0.3×) may still improve calibration
- Requires a simple trend model (linear extrapolation of voltage slope) that
  runs as a preprocessing step, not a neural network

### Transform Signature
```python
def transform_hyp009(
    ts_features: np.ndarray,   # (N, F) — full device history up to extraction
    failure_threshold: float,  # voltage level at which failure is declared
    vol_col: int = 0,          # voltage column index
    method: str = 'linear',    # 'linear' extrapolation of recent slope
    recent_window: int = 20,   # cycles to use for slope estimation
) -> float:                    # projected RUL in cycles (may be negative if overdue)
    """
    Estimates RUL for a censored device by extrapolating the voltage trend
    forward until it reaches failure_threshold.
    Returns float('inf') if slope is non-negative (no projected failure).
    """
```

### Test Condition
- **Positive:** Device with linear voltage decay toward threshold → projected RUL
  matches analytical solution within 2 cycles
- **Positive:** Device with flat voltage (slope ≈ 0) → returns `float('inf')`
- **Negative:** Actual failed device (voltage already below threshold) → returns ≤ 0
- **Edge case:** recent_window > available history → uses all available history
  with a warning

### Notes
<!-- This is the most speculative hypothesis — low priority until HYP-001 through -->
<!-- HYP-005 are validated. Flag for review after initial training runs.          -->

---

## HYP-010 · Multi-Scale Window Fusion

**Status:** `proposed`  
**Priority:** low  
**Chemistry:** both

### Statement
Different degradation signals operate at different time scales: voltage slope is
visible over short windows (10–20 cycles), while thermal stress accumulation
and resistance drift require longer context (50–100 cycles). Computing features
at multiple window scales and concatenating them into a single feature vector
will outperform any single window length.

### Rationale
- Standard SlidingWindowDataset uses a single window_len — a forced compromise
- Short windows capture fast signals (load spikes, thermal excursions);
  long windows capture slow drift (capacity fade, R_int trend)
- Multi-scale feature fusion is established in time-series literature (ROCKET,
  multi-scale TCN); the question is whether it helps for this specific failure mode

### Transform Signature
```python
def transform_hyp010(
    x_seq_long: np.ndarray,    # (T_long, F) — full long window
    scales: list[int],         # e.g. [10, 30, 60] — sub-window sizes
    agg: str = 'slope+mean',   # aggregation per scale: 'slope+mean' | 'stats'
) -> np.ndarray:               # (len(scales) * F * n_agg_features,) — flat vector
    """
    Computes per-scale summary statistics over the trailing sub-windows
    of x_seq_long, then concatenates into a flat feature vector.
    """
```

### Test Condition
- **Positive:** Output dimensionality matches `len(scales) * F * n_agg_features`
- **Positive:** Different degradation speeds produce discriminable feature vectors
  at the scale that matches the degradation rate
- **Ablation:** Compare model trained with single scale vs. multi-scale;
  multi-scale should improve validation MAE by ≥ 5% to justify complexity

### Notes
<!-- Implement only after HYP-001 through HYP-005 are validated. -->
<!-- Multi-scale adds complexity — only worth it if simpler transforms are insufficient. -->

---

## Implementation Order

Work through hypotheses in this sequence to build on validated foundations:

```
Phase 1 — Foundation transforms (implement and validate first)
  HYP-005  causal z-score         ← everything else builds on clean normalisation
  HYP-006  chemistry bounds       ← clip outliers before any feature engineering
  HYP-001  voltage slope          ← primary degradation signal

Phase 2 — Physical feature engineering
  HYP-002  resistance inflection  ← failure precursor, SOCl2 priority
  HYP-003  thermal stress         ← cumulative exposure index
  HYP-004  knee-point detection   ← explicit RUL reference feature

Phase 3 — Covariate corrections
  HYP-007  geographic bias        ← zone-relative temperature
  HYP-008  cohort baseline        ← lot-level manufacturing variation

Phase 4 — Advanced / speculative
  HYP-009  censored imputation    ← only if censored fraction > 30%
  HYP-010  multi-scale fusion     ← only if Phase 1–2 leave clear headroom
```

---

## Claude Code Session Prompts (copy-paste ready)

### Start a new hypothesis implementation
```
Read HYPOTHESES.md. Implement the transform for HYP-001 in
src/transforms/hyp001_voltage_slope.py. Write pytest tests in
tests/test_hyp001.py that cover all conditions listed in the
Test Condition section. Use the function signature exactly as
specified. After tests pass, update HYP-001 status to 'implemented'.
```

### Fix a failing test
```
Read HYPOTHESES.md section HYP-002 and the failing test output below.
Diagnose why test_hyp002_inflection_detection is failing, fix the
transform in src/transforms/hyp002_resistance_inflection.py, and
add a note to HYP-002 in HYPOTHESES.md explaining what was wrong.
[paste test output]
```

### Run integration across hypotheses
```
Read HYPOTHESES.md. All Phase 1 hypotheses are now implemented.
Create src/transforms/pipeline.py that applies HYP-005 → HYP-006 → HYP-001
in sequence on a DeviceRecord and returns the transformed x_seq.
Write an integration test in tests/test_pipeline.py that verifies
the output shape and that no NaN or inf values are present.
```

### Update status after validation
```
Read HYPOTHESES.md. HYP-003 tests have passed and a training run
shows the thermal stress feature improves validation MAE by 6.1%.
Update HYP-003 status to 'tested' and add a dated note with this result.
```

---

---

# Part II — Design Rationale and Instructional Material

> **Purpose of this section:** This part documents the design space — alternatives
> considered, the reasoning that connects domain physics to architectural choices,
> and the tradeoffs that make each decision non-obvious. It is intended as the basis
> for instructional content: blog posts, tutorials, or course material explaining
> how to build a production RUL pipeline from first principles.
>
> Each subsection follows the same structure: **the question**, **the options**,
> **why we chose what we chose**, and **what would change the answer**.

---

## DS-001 · Data Structure: How to Represent a Device Fleet

### The Question
A battery fleet is a collection of irregular time series — each device has a
different length, a different start date, and possibly different sampling rates.
What Python data structure should hold this before it enters PyTorch?

### Options Considered

#### Option A · Single wide DataFrame (rejected)
Concatenate all devices into one pandas DataFrame with a `device_id` column and
a `timestamp` column. Sort by device_id and timestamp.

```python
#  device_id  timestamp   voltage  temp   R_int   I_load   RUL
#  dev_0001   2020-01-01  1.00     24.1   0.050   0.51     180
#  dev_0001   2020-01-02  0.97     25.3   0.058   0.49     179
#  dev_0002   2020-01-01  1.01     23.8   0.048   0.50     NaN  ← censored
```

**Why rejected:** Window extraction requires a `groupby(device_id)` per sample,
which is O(N) per `__getitem__` call with N = total rows. With 10,000 windows
and 500 devices this adds up. More importantly, pandas DataFrames are row-major
in memory; slicing a window of 48 rows from a device buried in a 500-device
DataFrame causes cache misses across a wide stride. The structure optimises for
SQL-style queries, not sliding-window iteration.

#### Option B · Dict of DataFrames (partially viable)
```python
fleet: Dict[str, pd.DataFrame] = {
    'dev_0001': df_device_a,
    'dev_0002': df_device_b,
}
```
Better — O(1) device lookup, no groupby. But DataFrames carry pandas overhead
(dtype negotiation, index alignment) that is irrelevant once the data is clean.
Conversion to numpy happens inside every `__getitem__` call unless you pre-convert.

#### Option C · List of DeviceRecord dataclasses ✓ chosen
```python
@dataclass
class DeviceRecord:
    ts_features:   np.ndarray   # (N, F)  C-contiguous float32
    slow_features: np.ndarray   # (N, F_slow)
    static_cats:   np.ndarray   # (C,)    int64
    ...
```
Each device is a typed object holding pre-converted numpy arrays. `__getitem__`
is a pure numpy slice — no type conversion, no index alignment, no groupby.
Arrays are C-contiguous (row-major), so a window slice `arr[start:end]` is a
single contiguous memory read.

**Why this wins:** The working set for one window fits in L2 cache. For a
48-timestep window with 4 float32 features: 48 × 4 × 4 bytes = 768 bytes.
L2 cache on a modern CPU is 256 KB — you can hold ~340 windows simultaneously.
Pandas DataFrames would require dtype unboxing and index overhead on every access.

**What would change the answer:** If the fleet is too large for RAM (millions of
devices, years of history), you would use `Option D` — memory-mapped numpy
arrays (`np.memmap`) or an Arrow IPC file per device, with on-demand loading
in `__getitem__`. The DeviceRecord structure still works; you replace the numpy
arrays with memmap handles.

---

## DS-002 · Data Structure: Tensor Layout for Sequence Data

### The Question
Inside the Dataset, once you have the numpy slice, what tensor shape should the
sequence window have? Two conventions exist in the literature.

### Options Considered

#### Option A · (batch, features, time) — NCL convention
Used by PyTorch's 1-D `Conv1d` and some older recurrent implementations.
```python
x_seq: Tensor  # shape (B, F, T)  — "channels-first"
```

#### Option B · (batch, time, features) — NLC convention ✓ chosen
Used by `nn.LSTM`, `nn.Transformer`, and most modern sequence models.
```python
x_seq: Tensor  # shape (B, T, F)  — "time-first within the sequence"
```

**Why this wins:** `nn.LSTM` expects `(T, B, F)` by default (or `(B, T, F)` with
`batch_first=True`); `nn.TransformerEncoder` expects `(T, B, F)` with
`src_key_padding_mask` of shape `(B, T)`. The NLC layout avoids a `.permute()`
call at the model boundary, which would otherwise force a memory copy to restore
contiguity. `Conv1d` can always receive `.transpose(-1, -2)` — the cost is paid
once in the model, not in every `__getitem__` call.

**What would change the answer:** If your sequence encoder is purely `Conv1d`-based
(e.g. a TCN), NCL is more natural and avoids the transpose. In practice the
transpose is cheap; the more important decision is consistency — pick one layout
and enforce it in `collate_fn` so the model never has to guess.

---

## DS-003 · Data Structure: Static Features — Embedding vs. One-Hot vs. Target Encoding

### The Question
Chemistry type (2 classes), geographic zone (3 classes), and manufacture lot
(potentially hundreds of classes) are categorical. How should they be represented
as model inputs?

### Options Considered

#### Option A · One-hot encoding
```python
# chemistry: [1, 0]  or  [0, 1]
# zone:      [1, 0, 0], [0, 1, 0], [0, 0, 1]
```
Simple, interpretable, no learned parameters. But dimensionality explodes with
cardinality: 200 manufacture lots → 200-dimensional sparse vector. One-hot
vectors are orthogonal — they encode no similarity between categories — which
means the model cannot generalise "lot 47 behaves similarly to lot 46."

#### Option B · Ordinal encoding
```python
# chemistry: 0 or 1
# lot:       0, 1, 2, ... 199
```
Compact but imposes an arbitrary ordering. The model may learn that lot 100 is
"between" lot 50 and lot 150, which is meaningless for manufacturing lots.

#### Option C · Target encoding
Replace each category with the mean RUL of that category across the training set.
```python
# lot 47 → mean_RUL_for_lot_47 = 312.4
```
Compact and carries real signal. Risk: target leakage if not computed carefully
using held-out folds. Loses all information about within-lot variance.

#### Option D · Learned embeddings (nn.Embedding) ✓ chosen
```python
# chemistry_emb = nn.Embedding(num_embeddings=2,  embedding_dim=4)
# zone_emb      = nn.Embedding(num_embeddings=3,  embedding_dim=8)
# lot_emb       = nn.Embedding(num_embeddings=200, embedding_dim=16)
```
The embedding matrix is learned end-to-end with the model. Lots that degrade
similarly will have similar embedding vectors — the geometry of the embedding
space reflects the actual relationships in the data. Works well for high-cardinality
categoricals (lots, geographic cells).

**Why this wins:** Embeddings have been shown to outperform one-hot encoding for
high-cardinality categoricals in tabular deep learning (entity embeddings paper,
Guo & Berkhahn 2016). The embedding dimension is a hyperparameter — a useful
rule of thumb is `min(50, (cardinality // 2) + 1)`. For chemistry (2 classes)
a 4-dimensional embedding is likely overkill; for manufacture lot (200 classes)
a 16–32 dimensional embedding is appropriate.

**What would change the answer:** If you have very few devices per lot (< 5),
embeddings will overfit to lot-level noise. In that regime, target encoding with
cross-validation folding or simply grouping rare lots into an "other" bucket is
more robust.

---

## FM-001 · Forecasting Method: Regression vs. Survival Analysis vs. Classification

### The Question
RUL prediction is not a standard regression problem — many observations are
censored (device still alive), and the distribution of failure times is often
heavy-tailed and asymmetric. What is the right output formulation?

### Options Considered

#### Option A · Direct regression (MSE loss)
Predict RUL directly as a real number. Apply MSE or MAE loss on uncensored
windows; skip censored windows.

```python
rul_hat = head(fused)          # (B, 1)  scalar RUL in cycles
loss    = F.mse_loss(rul_hat[~censored], label[~censored])
```

**Problem:** Dropping censored windows discards ~25–30% of training signal.
More subtly: MSE treats overestimates and underestimates symmetrically, but in
operational contexts underestimating RUL (triggering unnecessary replacement)
and overestimating it (missing failure) have asymmetric costs.

#### Option B · Weibull AFT (Accelerated Failure Time) survival model ✓ chosen for baseline
Parameterise the output as the parameters (η, β) of a Weibull distribution.
The log-likelihood loss correctly handles censored observations.

```python
# Model outputs two scalars: log_eta, log_beta
# For uncensored:  loss = -log(Weibull_pdf(t | eta, beta))
# For censored:    loss = -log(Weibull_survival(t | eta, beta))
```

**Why this wins for the baseline:** Censored observations contribute their
correct statistical weight — they tell the model "this device survived at least
this long," which is real information. The Weibull distribution is physically
motivated for failure processes governed by material fatigue (the Bathtub curve
is a mixture of Weibulls). The model outputs a *distribution* over RUL, not a
point estimate, which enables uncertainty quantification.

#### Option C · Cox Proportional Hazards
A semi-parametric model that estimates the hazard function as a product of a
baseline hazard and a feature-dependent factor. No distributional assumption on
failure times.

```python
# log_hazard = feature_vector · beta
# Loss: partial likelihood (Breslow approximation)
```

**Tradeoff:** More flexible than Weibull (no parametric assumption), but the
partial likelihood requires sorting by event time within each batch, which
complicates batching. Also produces relative risk scores, not absolute RUL
estimates — less directly useful for maintenance scheduling.

#### Option D · Binary classification with sliding horizon
Predict P(failure within next H cycles). Train with binary cross-entropy.

```python
# label = 1 if failure_cycle - window_end <= H else 0
# loss  = F.binary_cross_entropy_with_logits(logit, label)
```

**When this is preferred:** If the operational question is "should I replace
this device now?" rather than "how many cycles remain?", binary classification
with a well-calibrated probability is more actionable than a point RUL estimate.
Pairs naturally with a cost matrix: P(failure) > threshold → replace.

**What would change the answer:** For a fleet with well-understood failure physics
(as in your battery domain), Weibull is the principled choice. For a fleet where
failure mechanisms are unknown or mixed (some devices fail from one cause, others
from another), the Cox model's nonparametric baseline is safer. For operational
alerting with a fixed decision horizon, binary classification is the most directly
deployable.

---

## FM-002 · Forecasting Method: Sequence Encoder Architecture

### The Question
The sequence encoder consumes `x_seq` of shape `(B, T, F)` and produces a
summary vector `h_T` of shape `(B, hidden_dim)`. Three architectures are viable.

### Options Considered

#### Option A · LSTM (Long Short-Term Memory)
```python
lstm = nn.LSTM(input_size=F, hidden_size=H, num_layers=2,
               batch_first=True, dropout=0.2)
h_T, _ = lstm(x_seq)   # take last timestep: h_T[:, -1, :]
```

**Strengths:**
- Well-understood inductive bias: hidden state is a compressed summary of history
- Naturally handles variable-length sequences via `pack_padded_sequence`
- Works well with window lengths T = 20–100 (your likely range)

**Weaknesses:**
- Sequential computation — cannot be parallelised across timesteps
- Gradient vanishing over very long sequences (T > 200)
- Hidden state is a lossy compression; important early events may be forgotten

**When to choose:** Smaller fleets, shorter windows, GPU not available,
or when interpretability of the hidden state matters.

#### Option B · Temporal Convolutional Network (TCN) ✓ recommended for this domain
```python
# Dilated causal convolutions with exponentially increasing dilation
# Receptive field = 2^(num_layers) * kernel_size
tcn = TCN(input_size=F, output_size=H,
          num_channels=[64, 64, 128, 128],
          kernel_size=3, dropout=0.2)
h_T = tcn(x_seq.permute(0, 2, 1))[:, :, -1]  # last timestep output
```

**Strengths:**
- Fully parallelisable — all timesteps computed simultaneously
- Receptive field is explicit and controllable (no implicit forgetting)
- Dilated convolutions efficiently capture multi-scale patterns — well-matched
  to battery degradation which has both short-term noise and long-term drift
- Causal by construction (no future leakage within the sequence)

**Weaknesses:**
- Fixed receptive field — sequences longer than the receptive field are truncated
- Less natural for variable-length sequences than LSTM
- Kernel size and dilation schedule are additional hyperparameters

**Why recommended for this domain:** Battery degradation has a known multi-scale
structure — high-frequency load current variations (cycles to tens of cycles) and
low-frequency capacity fade (hundreds of cycles). Dilated convolutions with
receptive fields tuned to match these scales are a natural fit.

#### Option C · Transformer Encoder
```python
encoder_layer = nn.TransformerEncoderLayer(d_model=F, nhead=4,
                                            dim_feedforward=256,
                                            batch_first=True)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
h = encoder(x_seq, src_key_padding_mask=padding_mask)
h_T = h[:, -1, :]  # last timestep, or mean over non-padded positions
```

**Strengths:**
- Self-attention has no locality bias — can attend to any timestep equally
- Scales well with data volume
- Pre-training on large corpora is possible (transfer from other battery datasets)

**Weaknesses:**
- O(T²) attention complexity — prohibitive for very long sequences
- Requires more data than LSTM/TCN to outperform them; may underfit on small fleets
- Positional encoding must be chosen carefully for irregular time series
  (sinusoidal encoding assumes uniform spacing; your timestamps may not be uniform)

**When to choose:** Fleet size > 10,000 devices, long sequences (T > 200),
or when transfer learning from a related domain is planned.

#### Summary comparison

| | LSTM | TCN | Transformer |
|---|---|---|---|
| Parallelism | sequential | fully parallel | fully parallel |
| Receptive field | implicit | explicit, fixed | full (O(T²)) |
| Small fleet fit | good | good | risky |
| Variable length | natural | needs padding | needs mask |
| Multi-scale | implicit | explicit (dilation) | via attention |
| Recommended for this project | fallback | **primary** | future |

---

## FM-003 · Forecasting Method: Handling the Three-Tier Feature Hierarchy in the Model

### The Question
We have three tiers of features: fast time series (`x_seq`), slow covariates
(`x_slow`), and static features (`x_static`). Several fusion strategies exist
for combining them.

### Options Considered

#### Option A · Flatten everything into x_seq (rejected)
Replicate `x_slow` and `x_static` at every timestep, concatenate with `x_seq`,
feed the combined tensor into the encoder.

```python
# x_slow replicated T times: (B, T, F_slow)
# x_static replicated T times: (B, T, K)
# combined: (B, T, F + F_slow + K)
```

**Why rejected:** Static features contain no temporal information — they are
the same at every timestep. Replicating them T times wastes computation and
memory, and forces the encoder to learn that these features are constant, which
it should not need to do. The encoder's inductive bias (local in time for TCN,
sequential for LSTM) is not well-suited to processing global context.

#### Option B · Late fusion ✓ chosen
Encode the sequence separately, then concatenate the encoding with slow and
static features before the prediction head.

```python
h_T   = encoder(x_seq)                         # (B, H)
emb   = embed(x_static_cats).flatten(1)         # (B, C × emb_dim)
fused = torch.cat([h_T, emb, x_slow], dim=-1)  # (B, H + C×emb + F_slow)
rul   = head(fused)                             # (B, 1)
```

**Why this wins:** Each tier is processed by the architecture best suited to it.
The sequence encoder is free to focus on temporal patterns. Static features are
injected as a learned embedding at the point where their global context is most
useful — just before the prediction head. This is the standard pattern in
industrial time-series models (TFT uses this; DeepAR conditions on static
covariates through the initial hidden state).

#### Option C · FiLM conditioning (Feature-wise Linear Modulation)
Use static features to generate per-channel scale and shift parameters that
modulate the sequence encoder's intermediate activations.

```python
# gamma, beta = static_mlp(x_static)  # (B, H)
# h = gamma * encoder_hidden + beta   # applied at each layer
```

**When this is preferred:** When the static features are expected to have a large
effect on the *shape* of the degradation curve (e.g., chemistry type fundamentally
changes the voltage profile, not just the overall level). FiLM allows the model
to learn chemistry-specific feature detectors, which late fusion cannot do.
This is a Phase 2 architectural experiment — more complex, warranting an ablation
study against late fusion.

---

## FM-004 · Forecasting Method: Loss Function Design

### The Question
Given a batch with mixed censored and uncensored windows, and the asymmetric
cost of RUL underestimation vs. overestimation, how should the loss be composed?

### Components

#### Regression term (uncensored windows)
```python
known = ~torch.isnan(label) & ~censored

# Symmetric: MSE or Huber
loss_reg = F.huber_loss(rul_hat[known], label[known], delta=10.0)

# Asymmetric: penalise overestimation more than underestimation
# (missing a failure is worse than unnecessary replacement)
errors = label[known] - rul_hat[known]
loss_reg = torch.where(errors > 0,
                        1.5 * errors ** 2,   # overestimate penalty
                        0.5 * errors ** 2)   # underestimate penalty
loss_reg = loss_reg.mean()
```

#### Survival term (censored windows)
```python
# Weibull parameterisation: model outputs log_eta, log_beta
eta  = torch.exp(log_eta)
beta = torch.exp(log_beta)
t    = label[censored].clamp(min=1e-6)  # survival time (time observed so far)

# Survival function: P(T > t) = exp(-(t/eta)^beta)
log_survival = -((t / eta) ** beta)
loss_surv    = -log_survival.mean()
```

#### Combined loss
```python
loss = loss_reg + lambda_surv * loss_surv
# lambda_surv: tuned via Optuna; typical range 0.1 – 1.0
```

**Design rationale:** The Huber loss is preferred over MSE because RUL labels
near failure can be noisy (the exact failure cycle is often not precisely known).
Huber's linear tail makes it robust to these label errors while still being
quadratic (fast-learning) near zero. The survival term's weight `lambda_surv`
controls the influence of censored windows — if censoring is rare (< 10%),
`lambda_surv` can be small; if censoring is common (> 30%), it should be larger.

---

## FM-005 · Forecasting Method: Train/Val/Test Split Strategy

### The Question
Random splitting of windows violates temporal causality and inflates validation
metrics. What is the correct split strategy for a device fleet?

### Options Considered

#### Option A · Random window split (rejected)
Randomly assign 80% of windows to train, 10% to val, 10% to test.

**Why rejected:** Windows from the same device appear in both train and val sets.
The model sees future observations from device A during training (in a later
window) and is evaluated on earlier observations from device A in validation.
This is temporal leakage — validation metrics will be optimistic and will not
reflect real deployment performance.

#### Option B · Device-level split ✓ chosen
Assign entire devices to train, val, or test. No device appears in more than one split.

```python
device_ids = [rec.device_id for rec in fleet]
train_ids, valtest_ids = train_test_split(device_ids, test_size=0.2,
                                           stratify=failure_flags,
                                           random_state=42)
val_ids, test_ids = train_test_split(valtest_ids, test_size=0.5,
                                      stratify=[f for id in valtest_ids ...])
```

Stratify on `failed` flag to ensure censored devices are proportionally
represented in each split.

**Why this wins:** Each device has a complete, independent lifetime history.
Device-level splitting ensures that validation measures true generalisation
to unseen devices — which is the real deployment scenario. Windows from
the same device share temporal correlations; treating them as independent
(random split) produces an overoptimistic estimate of model variance.

#### Option C · Temporal split
Train on devices that failed before date D; validate/test on devices that
failed after date D.

**When this is preferred:** If you suspect concept drift — degradation patterns
changing over calendar time due to manufacturing changes, environmental changes,
or policy changes in how devices are operated. For battery fleets, this is
worth checking if lot manufacture dates span multiple years.

---

## DS-004 · Data Quality: What to Log from Every Transform

### The Question
Transforms can silently corrupt data — a NaN propagates, a clip fires on 40%
of rows, a normalisation produces all-zero output. What diagnostics should
every transform emit?

### Recommended Transform Diagnostic Pattern

Every transform function should optionally return a `TransformDiagnostic`
alongside the transformed array:

```python
@dataclass
class TransformDiagnostic:
    transform_id:    str            # e.g. 'HYP-006'
    device_id:       str
    n_rows_in:       int
    n_rows_out:      int
    nan_count_in:    np.ndarray     # (F,) per-column NaN count before transform
    nan_count_out:   np.ndarray     # (F,) per-column NaN count after
    clip_fraction:   np.ndarray     # (F,) fraction of values clipped (HYP-006)
    value_range_out: np.ndarray     # (F, 2) [min, max] after transform
    wall_time_ms:    float          # transform execution time
```

**Why this matters for instructional content:** In a production fleet pipeline,
the most common failure mode is not model error — it is silent data corruption.
A sensor that returns a constant 0.0 for temperature (frozen sensor) will pass
normalisation (producing NaN std → all-zero output) without raising an exception.
The diagnostic pattern makes these failures visible in logs before they reach
the training loop.

This pattern also doubles as a data quality assessment tool for the DataQuality
project — a natural bridge between the `DataQuality` repo and the RUL pipeline.

---

## Instructional Narrative Arc

The following arc organises this material into a teachable sequence, progressing
from concrete to abstract and from simple to production-grade:

```
Chapter 1 · The Problem
  What is RUL prediction?  Why is it hard?
  Physical domain: Li battery degradation mechanisms
  Why standard regression fails (censoring, asymmetric cost, irregular data)

Chapter 2 · Data Architecture
  DS-001  DeviceRecord: representing irregular fleet data
  DS-002  Tensor layout: (B, T, F) and why
  DS-003  Categorical features: one-hot vs. embeddings
  Hands-on: build SlidingWindowDataset, run unit tests

Chapter 3 · Transforms as Hypotheses
  The hypothesis-driven development pattern
  HYP-005 → HYP-006 → HYP-001: foundation transforms
  How to write a falsifiable test condition
  Hands-on: implement voltage slope transform, run tests, interpret results

Chapter 4 · Physical Feature Engineering
  HYP-002, HYP-003, HYP-004: resistance, thermal, knee-point
  Connecting domain physics to transform design
  Hands-on: ablation study — which transforms improve validation MAE?

Chapter 5 · Model Architecture
  FM-002: LSTM vs. TCN vs. Transformer — when and why
  FM-003: Late fusion of three-tier features
  Hands-on: train baseline LSTM, then TCN, compare

Chapter 6 · Survival Analysis and Loss Design
  FM-001: regression vs. Weibull vs. classification
  FM-004: composing the loss function with censoring
  Hands-on: add Weibull head, compare to MSE baseline

Chapter 7 · Evaluation and Deployment
  FM-005: device-level split strategy
  Calibration: does P(failure) = 0.7 mean failure 70% of the time?
  DS-004: transform diagnostics as data quality monitoring
  Hands-on: deploy to a held-out lot, measure real-world performance

Chapter 8 · Reflections
  What would change the architecture for a different domain?
  The spec-driven development workflow with Claude Code
  Extending to other chemistries, other physical systems
```

---

*Part II added: 2026-04-25 · Basis for instructional content development*
