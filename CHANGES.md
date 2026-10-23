# Codebase Audit and Remediation Changelog (Sections A–D)

This document details every confirmed problem, the file and line numbers modified, and the remediation applied.

---

## Section A: Evaluation Validity

### A1. Event Leakage in Random Splits (CONFIRMED & FIXED)
- **Problem**: Rows were previously partitioned using random uniform splitting (`train_test_split`). Because the dataset contains 2,497 seismograms recorded across only 12 distinct earthquake events, multiple station records of the exact same event (and same ground-truth magnitude label) were simultaneously present in train, validation, and test splits. This resulted in optimistic test and cross-validation performance.
- **Files Changed**:
  - `src/data_loader.py` (Lines 14–22, 28–30, 75–82): Implemented `extract_event_id` to parse the 10-digit timestamp (`YYMMDDHHMM`) from seismic filenames (e.g. `YMG0190509041906.UD` -> `0509041906`). Added `return_groups=True` to `load_dataset`.
  - `src/preprocess.py` (Lines 114–190): Implemented `split_strategy` parameter ("event" vs "random"). When `split_strategy="event"`, partitions data using `GroupShuffleSplit` so all recordings of any event are strictly isolated into train, val, or test sets. Tagged "random" mode with explicit warnings.
  - `src/cross_validation.py` (Lines 52–155): Added `LeaveOneGroupOut` cross-validation (12 folds for 12 events) for honest event generalisation testing, clearly separating it from leaky random K-fold.
  - `main.py` (Lines 262–285, 410–435): Passed `groups` to `split_data_three_way` and `run_kfold_cv`.

### A2. Preprocessing Leakage in `newdata(claude)/` (CHECKED & CONFIRMED CLEAN)
- **Check**: Verified feature distributions across `newdata(claude)/train.csv`, `val.csv`, and `test.csv`.
- **Finding**: In `train.csv`, all features exhibit empirical mean = 0.000 and variance = 1.000. In `val.csv` and `test.csv`, feature means and standard deviations deviate from 0 and 1, confirming that `StandardScaler` was fitted strictly on the training partition without data leakage. No code modification required.

### A3. Ablation Evaluation on Test Set (CONFIRMED & FIXED)
- **Problem**: `src/ablation.py` was evaluating and ranking ablation experiments directly on the held-out test set over a single random seed, violating evaluation protocols.
- **Files Changed**:
  - `src/ablation.py` (Lines 80–235): Refactored `run_ablation_study` to run each ablation configuration across 5 independent random seeds (seeds 0–4), select and rank configurations strictly based on validation R², and evaluate on the test set only once at the conclusion of the study.

### A4. Trivial Baseline Comparison (CONFIRMED & FIXED)
- **Problem**: Reports and evaluation logs lacked explicit comparison against a trivial mean-predictor baseline, obscuring whether the PINN model outperformed simple constant prediction under negative R² regimes.
- **Files Changed**:
  - `src/evaluate.py` (Lines 180–210): Added trivial baseline computation (`mean(y_train)`) and stored baseline MAE, RMSE, and R² alongside PINN metrics in `metrics.json`.
  - `main.py` (Lines 340–385): Added explicit baseline comparisons and printed terminal indicators ("PINN beats mean baseline: YES / NO") for every metric.

---

## Section B: Physics Loss Formulation

### B1. Energy Constraint Formulation Ambiguity (CONFIRMED & FIXED)
- **Problem**: Code computed Pearson correlation between $\log_{10}(E)$ and predicted magnitude, whereas documentation and reports claimed enforcement of the Gutenberg–Richter relation ($\log_{10}(E) = 1.5 M + 4.8$).
- **Files Changed**:
  - `physics/energy_constraint.py` (Lines 32–78): Implemented explicit `energy_mode` supporting `"correlation"` (standardised Pearson correlation proxy) and `"gr_absolute"` ($\log_{10}(E) = a \cdot M + b$). Added numerical stability guards using unbiased variance and bounded square roots to prevent NaN gradients.
  - `physics/physics_loss.py` (Lines 22–85): Passed `energy_mode`, `energy_coeff_a`, and `energy_coeff_b` through to `energy_magnitude_loss`.
  - `config/config.yaml`: Added `energy_mode` option.

### B2. Gutenberg–Richter Intercept Sign Convention (CONFIRMED & FIXED)
- **Problem**: The theoretical Gutenberg–Richter relation $\log_{10}(E_0) = 1.5 M + 4.8$ applies to absolute seismic energy in Joules. The dataset feature `energy` is the 3-second acceleration waveform integral ($\int a^2 dt$), yielding negative values for $\log_{10}(E)$ relative to $1.5 M$.
- **Files Changed**:
  - `physics/energy_constraint.py`: Clarified the sign convention for `gr_absolute` mode ($\log_{10}(E) = 1.5 M + b$, with empirical $b \approx -3.863$).
  - `main.py` (Lines 300–315): Implemented dynamic calculation of intercept $b$ on the training set partition (`b = mean(log10(E_train) - 1.5 * M_train)`), eliminating hardcoded assumptions.

### B3. Physics Loss Mini-batch Alignment (CONFIRMED & VERIFIED)
- **Check**: Verified `train_optimized` in `src/train.py`.
- **Finding**: In Stage 5, `TensorDataset` packages input features, target magnitude, raw energy, and raw frequency into unified mini-batches. Data loss and physics loss are summed into `total_loss`, followed by a single joint `total_loss.backward()` pass. Confirmed functioning properly without gradient wiping.

### B4. Lambda Sweep Script (IMPLEMENTED)
- **Files Created**:
  - `scripts/lambda_sweep.py`: Implemented systematic evaluation over $\lambda_{\text{energy}} \in \{0.0, 0.001, 0.01, 0.05, 0.1, 0.3\}$ across 5 random seeds using event-based splitting.

---

## Section C: Data and Feature Pipeline

### C1. Degenerate Frequency Features (CONFIRMED & FIXED)
- **Problem**: `dominant_freq` and `predominant_period` are zero across all 2,497 samples due to upstream extraction failure.
- **Files Changed**:
  - `src/feature_audit.py` (Lines 85–92): Added detection and explicit warning indicating that raw waveforms are absent from the repository and that `spectral_centroid` serves as the proxy for frequency constraints.

### C2. Missing Distance Features (CONFIRMED & DOCUMENTED)
- **Problem**: Ground motion amplitude and energy decay with hypocentral/epicentral distance. Without distance corrections or coordinates, predicting magnitude from single-station P-wave amplitude is ill-posed.
- **Files Changed**:
  - `README.md` (Known Limitations section): Documented the absence of station coordinates and hypocentral distance in the dataset.

---

## Section D: Codebase Cleanliness and Consistency

### D1. README Architecture Discrepancy (CONFIRMED & FIXED)
- **Problem**: `README.md` documented hidden layers as `[256, 128, 64]`, while `config/config.yaml` specified `[64, 32, 16]`.
- **Files Changed**:
  - `README.md`: Realigned documentation with `config/config.yaml` (19 features, `[64, 32, 16]`, $\lambda_{\text{frequency}} = 0.0$, plateau scheduler patience = 30).

### D2. Hardcoded Feature Count (CONFIRMED & FIXED)
- **Problem**: `main.py` contained hardcoded strings referencing 49 features.
- **Files Changed**:
  - `main.py`: Replaced hardcoded constants with dynamic `len(feature_cols)` (resolves to 47 for `newdata_config.yaml`).

### D3. Static Feature Audit in `main.py` (CONFIRMED & FIXED)
- **Problem**: Audit printouts contained static hardcoded lists of unseen magnitudes.
- **Files Changed**:
  - `main.py`: Replaced with dynamic set difference between split sets (`set(val_mags) - set(train_mags)`).

### D4. Optuna HPO Failures (CONFIRMED & FIXED)
- **Problem**: In `archive/old_scripts/run_optimization.py`, silent exception handling caught runtime errors and returned `float("-inf")` for all trials.
- **Files Changed**:
  - `scripts/run_hpo.py`: Created an updated HPO script implementing event-based splitting, validation loss objective, proper tensor conversions, and informative logging.
