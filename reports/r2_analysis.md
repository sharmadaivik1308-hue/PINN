# Why is the R² Score So Low? — Root Cause Analysis

> **Observed R² = 0.015 (main run) | CV mean = 0.030**
> The model barely beats a mean-prediction baseline (R² = −0.003).

---

## 🔴 Root Cause #1 — The Target is Quasi-Discrete (Most Critical)

The magnitude target has only **11 unique values** (`[3.5, 4.0, 4.2, 4.3, 4.6, 4.8, 5.0, 5.2, 5.3, 5.7, 6.0]`) across 2,497 samples. This is effectively a **classification problem dressed as regression**. R² is fundamentally about explaining *continuous* variance; when 70%+ of samples share a handful of label values and the "true" variance is concentrated at discrete steps, any neural network will struggle to extract a smooth signal that generalises. Even a perfect model that places every prediction exactly at the correct discrete label would show a low R² *if* the label distribution is class-imbalanced (e.g., most samples at M=5.2).

**Implication**: R² near 0 does not mean the model is completely wrong — MAE of 0.49 vs baseline MAE of 0.50 shows it *is* learning something, just not enough to move R² substantially.

---

## 🔴 Root Cause #2 — P-wave Features Have Weak Predictive Signal for Magnitude

The dataset contains P-wave features extracted from the **first 3 seconds** of seismic recordings. The core problem:

- Earthquake magnitude is determined primarily by **total energy release** and **rupture duration**, both of which unfold over *tens of seconds* to *minutes*.
- P-wave amplitudes in the first 3 s encode *initial rupture nucleation*, which is weakly correlated with final magnitude.
- This is a well-known limitation in seismology — early warning systems struggle precisely because the first 3 s are insufficient for reliable magnitude estimation.

**Evidence from the code**: The audit notes `spectral_centroid` has r = −0.058 with magnitude — essentially zero signal. Even `energy` (the primary physics feature) is a proxy (`∫a²dt`), not absolute Joules, making the G-R law only approximately applicable.

---

## 🔴 Root Cause #3 — The Model Stopped at Only 75 Epochs (Early Stopping Too Aggressive)

The model ran **only 75 epochs** out of 500 configured, stopped by early stopping with patience=50. This means:

- The model had not converged — it stopped learning prematurely.
- The validation loss plateau (best val loss = 0.969) is still very high, suggesting the model is barely fitting the training data.
- Training time was just **10.3 seconds**, which for a dataset of this complexity is extremely short.

**Fix**: Increase `patience` from 50 → 150+, or reduce the LR scheduler's aggressiveness (factor=0.5, patience=15 is very rapid decay).

---

## 🟠 Root Cause #4 — Network is Too Small for the Feature Set

Config uses `hidden_layers: [64, 32, 16]` with only **8,778 parameters**. This is a very shallow and narrow network. Given:
- 19 input features
- A complex non-linear mapping from seismic waveform statistics to magnitude
- The physics constraints adding extra gradient pressure

The network likely lacks **representational capacity** to fit even the training data well. The ablation confirms: all experiments (A–D) show R² between −0.32 and +0.02, consistently poor.

---

## 🟠 Root Cause #5 — Physics Constraint is a Misfit

The Gutenberg–Richter energy constraint enforces:
```
log10(E) = 1.5 × M + 4.8
```
But the `energy` feature is `∫a²dt` (P-wave acceleration integral), **not absolute seismic energy in Joules**. The G-R coefficients (a=1.5, b=4.8) apply to absolute seismic moment energy, not a normalized waveform integral. This means:

- The physics loss is pushing the model toward a **wrong physical relationship** for this feature.
- It acts as a **regularizer with incorrect direction**, potentially harming R².

Evidence: Ablation shows experiment C (Energy PINN) gets R²=0.0156 vs experiment B (no physics) R²=0.0069 — the physics slightly helps, but neither is meaningful.

---

## 🟡 Root Cause #6 — LR Scheduler Kills Learning Too Early

```yaml
scheduler: "plateau"   # factor=0.5, patience=15
```

With patience=15 epochs per LR reduction and factor=0.5, by epoch 75 the LR has likely been halved 3–4 times (from 0.001 → 0.0000625). Combined with early stopping at patience=50, the model never has a chance to escape local minima at a reasonable learning rate.

---

## 🟡 Root Cause #7 — No Stratified Split

The `split_data_three_way` function uses standard `train_test_split` **without stratification**. With only 11 class values and imbalanced distribution (most samples cluster around M=5.0–5.3), the test set may not represent all magnitudes. A rare class like M=3.5 or M=6.0 may appear too few times to be learned or evaluated fairly.

---

## Summary Table

| # | Cause | Severity | Fix |
|---|-------|----------|-----|
| 1 | Quasi-discrete target (11 unique values) | 🔴 Critical | Treat as ordinal classification; use cross-entropy |
| 2 | P-wave (3s) features don't encode full magnitude | 🔴 Critical | Use full waveform or longer window features |
| 3 | Early stopping at epoch 75/500 | 🔴 High | Increase patience to 150+; reduce LR decay |
| 4 | Network too small `[64,32,16]` | 🟠 Medium | Use `[256,128,64]` as README suggests |
| 5 | G-R law applied to non-physical proxy energy | 🟠 Medium | Calibrate coefficients empirically or disable |
| 6 | LR scheduler decays too aggressively | 🟡 Medium | Increase scheduler patience or use cosine LR |
| 7 | No stratified split | 🟡 Low | Add `stratify=y` to train_test_split |

---

## Most Impactful Quick Fixes

1. **Switch to ordinal classification** — treat the 11 magnitude classes as ordered categories and use a cross-entropy loss. This converts the task to what the data actually supports.
2. **Increase patience** — `patience: 150`, `scheduler patience: 30` so the model has time to converge.
3. **Increase network size** — change `hidden_layers: [256, 128, 64]` (README mentions this, but config uses a smaller `[64, 32, 16]`).
4. **Disable physics** or recalibrate the energy coefficients empirically using the actual `energy` ↔ `magnitude` correlation from this dataset.
