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
