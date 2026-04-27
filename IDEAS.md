# IDEAS.md — Idea Capture and Promotion Log
<!-- CLAUDE CODE INSTRUCTIONS
  DO NOT implement anything from this file directly.
  Ideas here are unvalidated and may be incomplete or contradictory.
  To act on an idea: it must first be promoted to HYPOTHESES.md as a HYP-NNN entry.
  The only action Claude Code should take on this file is:
    - Add new ideas when asked
    - Update Status when asked
    - Add a Promoted-to reference when an idea is promoted
-->

**Project:** `/home/chuck/GitClones/DataQuality`  
**Domain:** Battery RUL · PyTorch time-series · Instructional content  
**Last updated:** 2026-04-25 10AM

---

## How This File Works

```
Stage 1 · Capture     Add any idea here, any time, in any form.
                      Use the category tags and the template below.
                      No idea is too rough — that's the point.

Stage 2 · Refine      Return to raw ideas and sharpen them:
                      add a falsifiable test condition, a function
                      signature, a concrete data source.

Stage 3 · Promote     Move a refined idea to HYPOTHESES.md as a
                      new HYP-NNN entry. Mark it promoted here.
                      The idea stays in this file for traceability.
```

**Status values**

| Status | Meaning |
|---|---|
| `raw` | captured, not yet evaluated |
| `refining` | actively being sharpened |
| `ready` | has test condition + signature → promote to HYPOTHESES.md |
| `promoted` | lives in HYPOTHESES.md as HYP-NNN |
| `parked` | interesting but not actionable right now |
| `rejected` | evaluated and decided against — keep with reason |

**Category tags**

| Tag | Scope |
|---|---|
| `[TRANSFORM]` | Feature engineering, data preprocessing, signal processing |
| `[ARCH]` | Model architecture, encoder design, fusion strategy |
| `[DATA]` | Data sources, quality issues, collection, labeling |
| `[EVAL]` | Evaluation strategy, metrics, calibration, backtesting |
| `[PRODUCT]` | Consulting product, SaaS, API, deployment |
| `[CONTENT]` | Instructional material, blog posts, course, talks |
| `[WORKFLOW]` | Dev process, tooling, Claude Code patterns, project structure |

---

## Idea Template

Copy this block for each new idea:

```
### IDEA-NNN · Short Title
**Tag:** [CATEGORY]
**Status:** raw
**Captured:** YYYY-MM-DD
**Promoted-to:** —

#### Raw Idea
What you're thinking, in whatever form it arrives.
One sentence or ten — doesn't matter at capture time.

#### Why It Might Matter
What problem does it solve? What would be better if this worked?

#### Open Questions
What you don't know yet. What would need to be true for this to work.

#### Refinement Notes
(fill in as the idea develops)

#### Promotion Checklist
- [ ] Falsifiable test condition written
- [ ] Function signature or interface defined
- [ ] Dependency on other HYPs identified
- [ ] Promoted to HYPOTHESES.md
```

---

## Status Summary

| ID | Title | Tag | Status |
|---|---|---|---|
| IDEA-001 | Arrhenius-weighted thermal loss | [TRANSFORM] | raw |
| IDEA-002 | Voltage curve shape fingerprint | [TRANSFORM] | raw |
| IDEA-003 | FiLM conditioning for chemistry | [ARCH] | raw |
| IDEA-004 | SDG&E AMI data as proxy labels | [DATA] | raw |
| IDEA-005 | Calibration layer for Weibull output | [EVAL] | raw |
| IDEA-006 | Battery RUL-as-a-Service API design | [PRODUCT] | raw |
| IDEA-007 | Hypothesis-driven dev as course module | [CONTENT] | raw |
| IDEA-008 | Claude Code session transcript as lab notebook | [WORKFLOW] | raw |
| IDEA-009 | Acoustic SOH diagnosis for saxophones | [TRANSFORM] | parked |
| IDEA-010 | Lot-stratified cross-validation | [EVAL] | raw |

---

## Ideas

---

### IDEA-001 · Arrhenius-Weighted Thermal Loss
**Tag:** [TRANSFORM]  
**Status:** `raw`  
**Captured:** 2026-04-25  
**Promoted-to:** —

#### Raw Idea
HYP-003 computes a linear cumulative thermal stress index (degree-days above
baseline). But Arrhenius says the relationship between temperature and reaction
rate is exponential, not linear. A transform that computes
`sum(exp(T / T_ref))` rather than `sum(max(0, T - T_baseline))` would be
more physically correct.

#### Why It Might Matter
The linear index undercounts the damage done by extreme temperature excursions.
A device that spends 10 hours at 60°C accumulates far more than 2× the damage
of one that spends 10 hours at 30°C, but the linear index says 2×. If the fleet
has significant variance in peak temperatures (coastal vs. desert), the linear
index may mask the real signal.

#### Open Questions
- What is the activation energy (Ea) for Li/SOCl₂ electrolyte decomposition?
  This sets T_ref in the Arrhenius expression.
- Does the exponential index actually improve model performance, or is the
  linear approximation good enough given measurement noise?
- Is the Arrhenius model valid across the full temperature range of desert
  installations (possibly 70°C+)?

#### Refinement Notes

#### Promotion Checklist
- [ ] Falsifiable test condition written
- [ ] Function signature defined
- [ ] Dependency on HYP-003 noted
- [ ] Promoted to HYPOTHESES.md

---

### IDEA-002 · Voltage Curve Shape Fingerprint
**Tag:** [TRANSFORM]  
**Status:** `raw`  
**Captured:** 2026-04-25  
**Promoted-to:** —

#### Raw Idea
Rather than extracting slope and curvature features point-by-point, fit a
parametric model (e.g. a 3-parameter sigmoid or a piecewise linear model)
to the full voltage curve for each device and use the fitted parameters as
static features. The parameters capture the *shape* of degradation — early
plateau length, slope of the decline, final drop rate — in a compact,
physically interpretable form.

#### Why It Might Matter
The sliding-window approach gives the model local views of the degradation curve.
The shape fingerprint gives it global context — where in the lifecycle this device
sits — without requiring a very long window. Particularly useful for Li/SOCl₂
where the flat plateau is very long and the drop is abrupt; the plateau-to-drop
ratio is a strong between-device discriminator.

#### Open Questions
- Does a sigmoid adequately capture the J-curve shape of SOCl₂ discharge?
  May need a custom parameterisation.
- How do you fit the model causally (using only history up to the current window)?
  Early in device life there isn't enough data to fit a reliable curve.
- Are the fitted parameters stable enough across noise levels to be useful
  as training features, or will they overfit to individual noisy traces?

#### Refinement Notes

#### Promotion Checklist
- [ ] Falsifiable test condition written
- [ ] Parameterisation chosen (sigmoid / piecewise / other)
- [ ] Causal fitting strategy defined
- [ ] Promoted to HYPOTHESES.md

---

### IDEA-003 · FiLM Conditioning for Chemistry-Specific Feature Detection
**Tag:** [ARCH]  
**Status:** `raw`  
**Captured:** 2026-04-25  
**Promoted-to:** —

#### Raw Idea
Late fusion (FM-003) concatenates chemistry embedding with the encoder output.
But chemistry type may affect not just the *level* of degradation features but
their *shape* — the voltage curve for SOCl₂ looks fundamentally different from
MnO₂. FiLM (Feature-wise Linear Modulation) lets the chemistry embedding
generate per-channel scale and shift parameters that modulate the encoder's
intermediate activations, allowing the encoder to learn chemistry-specific
feature detectors.

#### Why It Might Matter
If the encoder learns a single set of convolutional filters shared across both
chemistries, it may find filters that are a compromise — good for neither.
FiLM would let the encoder specialise its internal representations by chemistry
without needing separate models per chemistry.

#### Open Questions
- Is the performance gain over late fusion large enough to justify the added
  complexity? Needs an ablation study.
- Where in the encoder should FiLM modulation be applied — every layer, or
  just the first?
- Does FiLM help if only 2 chemistry classes exist? The benefit is clearest
  with higher cardinality conditioning variables.

#### Refinement Notes
Documented as FM-003 Option C in HYPOTHESES.md Part II. Not yet a HYP-NNN
entry because it requires late fusion (FM-003 Option B) to be validated first.

#### Promotion Checklist
- [ ] Late fusion baseline validated first (prerequisite)
- [ ] Falsifiable ablation condition written (FiLM vs. late fusion on held-out lot)
- [ ] Architecture diagram defined
- [ ] Promoted to HYPOTHESES.md

---

### IDEA-004 · SDG&E AMI Data as Proxy Failure Labels
**Tag:** [DATA]  
**Status:** `raw`  
**Captured:** 2026-04-25  
**Promoted-to:** —

#### Raw Idea
AMI meter data contains voltage and current readings at 15-minute or hourly
intervals for the distribution network. Batteries connected to the grid will
show characteristic signatures in the AMI data near end-of-life (voltage
sag, current anomalies, communication dropouts). AMI data could serve as a
weak supervision signal for RUL labeling — augmenting or replacing direct
failure records that may be incomplete.

#### Why It Might Matter
Direct failure records in utility asset management systems are often incomplete:
devices are replaced proactively before failure, failure is attributed to the
wrong cause, or field records are not entered promptly. AMI data is collected
automatically and continuously — it may be a more reliable source of failure
signal than maintenance records.

#### Open Questions
- What AMI signals are most indicative of battery end-of-life vs. grid anomalies?
- How do you align AMI timestamps with device-level maintenance records?
- Are there CPUC data governance constraints on using AMI data for asset
  health modeling? (Privacy implications for residential meters may apply.)
- Is the 15-minute or hourly AMI resolution sufficient, or does RUL detection
  require higher-frequency measurements?

#### Refinement Notes
This is a data sourcing idea, not a transform idea. Needs a data availability
assessment before it can be shaped into a hypothesis.

#### Promotion Checklist
- [ ] Data availability confirmed
- [ ] CPUC/privacy constraints assessed
- [ ] Alignment methodology with maintenance records defined
- [ ] Shaped into a DATA hypothesis (different template than TRANSFORM)

---

### IDEA-005 · Calibration Layer for Weibull Output
**Tag:** [EVAL]  
**Status:** `raw`  
**Captured:** 2026-04-25  
**Promoted-to:** —

#### Raw Idea
The Weibull head outputs a distribution over failure times. But the predicted
distribution may be systematically overconfident or underconfident — the model
says P(failure within 20 cycles) = 0.9, but the actual rate is 0.7. A
post-hoc calibration layer (temperature scaling or isotonic regression applied
to the predicted survival probabilities) would correct this without retraining
the full model.

#### Why It Might Matter
For operational deployment — "replace this device if P(failure within H cycles)
> threshold" — calibration is more important than raw accuracy. An uncalibrated
model with high AUC can still produce systematically wrong probabilities that
lead to either too many or too few replacements.

#### Open Questions
- Does temperature scaling (a single scalar applied to the Weibull log-hazard)
  suffice, or is the miscalibration more complex?
- How do you evaluate calibration for a survival model? Reliability diagrams
  require binning predicted probabilities against observed event rates —
  the censoring complicates this.
- Should calibration be fit on a held-out calibration set separate from
  the test set?

#### Refinement Notes

#### Promotion Checklist
- [ ] Weibull baseline model trained first (prerequisite)
- [ ] Calibration evaluation metric defined (e.g. Expected Calibration Error
      adapted for survival outcomes)
- [ ] Promoted to HYPOTHESES.md as an EVAL hypothesis

---

### IDEA-006 · Battery RUL-as-a-Service API Design
**Tag:** [PRODUCT]  
**Status:** `raw`  
**Captured:** 2026-04-25  
**Promoted-to:** —

#### Raw Idea
Package the RUL pipeline as a REST API: POST a device's recent time-series
measurements, receive a JSON response with predicted RUL distribution,
confidence interval, and replacement recommendation. Utility customers
(other than SDG&E) could call this API without needing to build or maintain
the model themselves.

#### Why It Might Matter
The trained model is the asset; the API is the delivery mechanism. Utilities
have the data but not the modeling expertise. A well-documented API with
a clear pricing model (per-device per-month) maps cleanly to the
Battery RUL-as-a-Service product identified as high-fit in the post-retirement
product analysis.

#### Open Questions
- What is the minimum viable data contract — what fields does the caller
  need to provide per POST request?
- How do you handle chemistry-conditional behavior in the API
  (caller must specify chemistry, or it is inferred)?
- What is the right latency target? Batch inference (nightly fleet sweep)
  vs. real-time (triggered by anomaly event)?
- How do you version the model while maintaining API stability?

#### Refinement Notes
This is a product design idea. Action items are business/architecture, not code.
Consider drafting a one-page API spec (endpoint, request schema, response schema)
as the next concrete step.

#### Promotion Checklist
- [ ] One-page API spec drafted
- [ ] Pricing model sketched
- [ ] MVP scope defined (which chemistries, which features required)
- [ ] Not a HYPOTHESES.md entry — this is a PRODUCT spec document

---

### IDEA-007 · Hypothesis-Driven Development as Standalone Course Module
**Tag:** [CONTENT]  
**Status:** `raw`  
**Captured:** 2026-04-25  
**Promoted-to:** —

#### Raw Idea
The workflow developed here — HYPOTHESES.md as spec, Claude Code as
implementer, test conditions as falsification criteria, status tracking as
lab notebook — is domain-agnostic. It could be packaged as a standalone
course module: "Hypothesis-Driven ML Development with Claude Code," using
the battery RUL pipeline as the worked example.

#### Why It Might Matter
Most ML tutorials teach code. Few teach *process* — how to move from a vague
domain intuition ("temperature probably matters") to a tested, production-ready
feature. The hypothesis-driven pattern is the missing piece. It is also a
natural fit for the instructional narrative arc in HYPOTHESES.md Part II.

#### Open Questions
- What is the target audience? Data scientists who know Python but lack
  structured development process? Utility domain experts who want to
  understand ML workflows?
- What is the right delivery format — Jupyter notebooks, a GitHub template
  repo, a video course, a written guide?
- How much of the instructional value depends on having Claude Code access
  vs. being applicable to any LLM-assisted workflow?

#### Refinement Notes
The HYPOTHESES.md Part II instructional narrative arc (Chapters 1–8) is
the content skeleton for this. The next step is choosing a delivery format
and identifying the first concrete deliverable (e.g. Chapter 1 as a blog post).

#### Promotion Checklist
- [ ] Target audience defined
- [ ] Delivery format chosen
- [ ] Chapter 1 drafted as pilot
- [ ] Not a HYPOTHESES.md entry — this is a CONTENT plan document

---

### IDEA-008 · Claude Code Session Transcript as Lab Notebook
**Tag:** [WORKFLOW]  
**Status:** `raw`  
**Captured:** 2026-04-25  
**Promoted-to:** —

#### Raw Idea
Each Claude Code session that implements or tests a hypothesis produces a
natural language record of what was attempted, what failed, and what succeeded.
If these transcripts are saved (or summarised) alongside the code, they form
a lab notebook — the kind of record that research scientists keep but software
engineers typically don't.

#### Why It Might Matter
Lab notebooks are valuable for: understanding why a design decision was made
months later, writing the Methods section of a paper or report, onboarding
a collaborator, and generating instructional content from real development history.
The HYPOTHESES.md Notes fields are a lightweight version of this — but
full session transcripts would be richer.

#### Open Questions
- Where should transcripts live? A `transcripts/` directory in the repo?
  A separate private repo?
- Should they be raw transcripts or structured summaries?
  Raw transcripts are large; summaries lose detail.
- Is there a Claude Code feature that auto-exports session history?

#### Refinement Notes
A lightweight version of this is already in place: the Notes fields in
HYPOTHESES.md are the designated place to record dated outcomes.
The question is whether richer transcripts add enough value to justify
the storage and curation overhead.

#### Promotion Checklist
- [ ] Decide on storage location and format
- [ ] Try saving one session transcript and evaluate usefulness
- [ ] If valuable: add to WORKFLOW section of HYPOTHESES.md instructions

---

### IDEA-009 · Acoustic State-of-Health Diagnosis for Saxophones
**Tag:** [TRANSFORM]  
**Status:** `parked`  
**Captured:** 2026-04-25  
**Promoted-to:** —

#### Raw Idea
Acoustic resonance properties of a saxophone change as pads wear, corks
degrade, or the body develops micro-cracks. Signal processing techniques
from the battery domain (impedance spectroscopy analogues, resonance peak
tracking) might transfer to acoustic diagnosis of instrument health.

#### Why It Might Matter
A direct connection between the signal processing background (RUL, impedance
analysis) and a long-standing personal interest. Could be a compelling
instructional example — "the same math that predicts battery failure can
diagnose a saxophone" — that makes the course material memorable and
differentiating.

#### Open Questions
- What is the acoustic equivalent of internal resistance? Probably the
  Q-factor of the main resonance peak.
- What failure modes are detectable acoustically vs. requiring physical inspection?
- Is there existing literature on acoustic instrument health monitoring?

#### Refinement Notes
Parked — not relevant to the current project scope. Revisit post-retirement
when time allows for exploration projects. The signal processing connection
is real and worth developing eventually.

#### Promotion Checklist
- [ ] Literature search on acoustic instrument health monitoring
- [ ] Identify a measurable proxy for pad/cork degradation
- [ ] Scope a small exploratory experiment

---

### IDEA-010 · Lot-Stratified Cross-Validation
**Tag:** [EVAL]  
**Status:** `raw`  
**Captured:** 2026-04-25  
**Promoted-to:** —

#### Raw Idea
Standard device-level train/val/test split (FM-005) randomises across lots.
But if the goal is to generalise to *new* lots not seen during training (the
real deployment scenario for a new product generation), the split should
hold out entire lots from validation and test. Lot-stratified CV would measure
true between-lot generalisation.

#### Why It Might Matter
If a model is trained on lots 1–50 and deployed on lot 51, it must generalise
across manufacturing variation it has never seen. Device-level random split
allows the model to implicitly learn lot-specific patterns (since most devices
from a lot appear in training). Lot-held-out evaluation would reveal if the
model is actually generalising or just memorising lot signatures.

#### Open Questions
- How many lots are in the dataset? Lot-held-out CV requires enough lots
  to form meaningful held-out groups.
- Does the cohort baseline correction (HYP-008) reduce the lot-generalisation
  gap? It should — that is its stated purpose.
- How does lot-held-out performance compare to device-random performance?
  The gap size is a measure of how much lot-specific information the model
  is exploiting.

#### Refinement Notes
This is an evaluation methodology idea. The prerequisite is having a dataset
with known lot labels and enough lots (≥ 10) to make cross-validation meaningful.

#### Promotion Checklist
- [ ] Confirm lot label availability and cardinality in dataset
- [ ] Define the CV scheme (leave-one-lot-out vs. k-fold on lots)
- [ ] Promote to HYPOTHESES.md as an EVAL hypothesis (new category)
      after HYP-008 is implemented

---

### IDEA-011 - Relationship between Temp/Voltage timestamp and measurement values
**Tag:** [TRANSFORM]
**Status:** raw
**Captured:** 2026-04-25
**Promoted-to:** —

#### Raw Idea
Investigate the relationship between the time that the voltage/temperature sample was taken and the temp/voltage measurement

#### Why It Might Matter
This can answer several questions about how the physical mechanics of each sample.  

#### Open Questions

1. Is the temp/voltage measurement taken at the time of the timestamp?  The timestamps span a range over the course of the day during which there are daily temperature seasonalities. So, a measurment could be taken at the same time as the timestamp, or the measurement could be taken when the broadcast message was sent over the network, in which case, the measurement was stored and transmitted at the time of the broadcast message that is beleived to be midnight. 

#### Refinement Notes
(fill in as the idea develops)

#### Promotion Checklist
- [ ] Falsifiable test condition written
- [ ] Function signature or interface defined
- [ ] Dependency on other HYPs identified
- [ ] Promoted to HYPOTHESES.md

### IDEA-012 - Geogrpahic patterns of data arrival timestamp and geographic locality.
**Tag:** [TRANSFORM]
**Status:** raw
**Captured:** 2026-04-25
**Promoted-to:** —

#### Raw Idea
Patterns in the arrival time of the data may indicate network health - or the health of each communication path

#### Why It Might Matter
It may aid in the interpretation of volt/temp data. May shed light on the grouping 

#### Open Questions


#### Refinement Notes
(fill in as the idea develops)

#### Promotion Checklist
- [ ] Falsifiable test condition written
- [ ] Function signature or interface defined
- [ ] Dependency on other HYPs identified
- [ ] Promoted to HYPOTHESES.md





## Promotion Log

| Date | IDEA | Promoted to | Notes |
|---|---|---|---|
| — | — | — | — |

<!-- When an idea is promoted, add a row here and update its Status and Promoted-to field above -->

---

## Parking Lot

Ideas that are interesting but intentionally deferred. Revisit quarterly.

| ID | Title | Reason parked | Revisit trigger |
|---|---|---|---|
| IDEA-009 | Acoustic saxophone diagnosis | Out of scope for current project | Post-retirement exploration time |

---

*Capture everything. Promote only what is sharp enough to test.*
