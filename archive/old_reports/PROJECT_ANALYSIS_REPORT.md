# PROJECT ANALYSIS REPORT
## Physics-Informed Neural Network for Earthquake Magnitude Prediction

> [!NOTE]
> This report was generated after a complete analysis of every file in the project — source code, research papers, dataset, DOCX proposal, CSV data, saved model outputs, and folder structure.

---

## 1. Project Objective

### 1.1 Problem Statement
The project aims to predict **earthquake magnitude** from seismic waveform features extracted from the early P-wave phase of strong-motion records. Specifically, it uses **29 engineered features** extracted from the first 3 seconds of P-wave arrivals recorded by K-NET stations in Japan.

### 1.2 Why Earthquake Magnitude Prediction Matters
- **Early Warning Systems (EEW)**: Rapid magnitude estimation within seconds of P-wave arrival enables warnings before destructive S-waves arrive
- **Disaster Mitigation**: Accurate magnitude prediction helps trigger automated safety measures in buildings, infrastructure, and industrial facilities
- **The base paper** (Joshi et al., 2024) demonstrated that XGBoost with 29 features achieves a prediction error of 0.004 ± 0.57 using 3-second P-wave windows — vastly superior to traditional τc and Pd methods

### 1.3 Why Physics-Informed Neural Networks (PINNs)
Traditional neural networks are purely data-driven: they minimize prediction error without any knowledge of seismological physics. This leads to:

| Limitation of Baseline NN | How PINN Addresses It |
|---|---|
| Can predict physically impossible magnitudes | Physics loss penalizes violations of energy–magnitude scaling |
| Ignores known seismological relationships | Encodes Gutenberg–Richter and energy–magnitude laws |
| Treats all features equally | Attention mechanism prioritizes physics-relevant features |
| No uncertainty quantification | Dual-output head provides both magnitude and confidence |
| May overfit to noise | Physics regularization constrains the solution space |

The **total loss** becomes:

```
L_total = L_data + λ₁·L_energy + λ₂·L_frequency
```

Where the physics losses enforce known seismological scaling laws during training.

---

## 2. Dataset Analysis

### 2.1 Overview

| Property | Value |
|---|---|
| **File** | `earthquake_29_features.csv` |
| **Source** | K-NET (Kyoshin Network), NIED Japan |
| **URL** | https://www.kyoshin.bosai.go.jp |
| **Total Samples** | 2,497 |
| **Total Columns** | 31 (filename + 29 features + magnitude) |
| **Features Used** | 29 (after dropping `filename`) |
| **Target Variable** | `magnitude` (JMA scale) |
| **Missing Values** | 0 |
| **Magnitude Range** | 3.5 – 6.0 |

### 2.2 Target Variable Distribution

| Magnitude | Count |
|---|---|
| 3.5 | 34 |
| 4.0 | 48 |
| 4.2 | 114 |
| 4.3 | 176 |
| 4.6 | 224 |
| 4.8 | 320 |
| 5.0 | 271 |
| 5.2 | 180 |
| 5.3 | 434 |
| 5.7 | 326 |
| 6.0 | 370 |

> [!IMPORTANT]
> The magnitude is **discrete** (11 unique values), not continuous. This is typical for binned JMA magnitudes. The model should still treat this as a regression problem, but this limits the granularity of predictions.

### 2.3 Feature Descriptions

The 29 features are engineering quantities extracted from the first 3 seconds of the P-wave phase:

| Category | Features | Description |
|---|---|---|
| **Amplitude** | `pga`, `max_val`, `min_val`, `range`, `envelope_peak` | Peak ground acceleration and amplitude envelope characteristics |
| **Statistical** | `mean`, `std`, `skewness`, `kurtosis`, `median`, `abs_mean` | Central tendency and distribution shape of the waveform |
| **Quantiles** | `p25`, `p75`, `iqr`, `p90`, `p95` | Percentile-based spread measures |
| **Energy** | `energy`, `log_energy`, `arias`, `spectral_energy`, `rms`, `cav` | Cumulative energy measures: Arias intensity, cumulative absolute velocity |
| **Frequency** | `dominant_freq`, `spectral_centroid`, `bandwidth`, `predominant_period` | Spectral characteristics of the waveform |
| **Time-domain** | `zero_crossings`, `duration`, `rise_time` | Temporal characteristics of the P-wave |

### 2.4 Critical Data Quality Issues

> [!CAUTION]
> **Three features are constant/degenerate and carry zero information:**
> 
> | Feature | Issue | Impact |
> |---|---|---|
> | `dominant_freq` | **All zeros** (2497/2497) | **Cannot be used for frequency–magnitude physics constraint** |
> | `predominant_period` | **All zeros** (2497/2497) | Zero variance, no predictive power |
> | `zero_crossings` | Nearly all zeros (2488/2497), std=0.73 | Near-constant, negligible information |
> 
> This is a **critical finding** for the PINN design. The frequency–magnitude constraint from the DOCX proposal uses `dominant_freq`, which is entirely zero in this dataset. This constraint must be redesigned.

### 2.5 Feature–Magnitude Correlations

Correlations with magnitude are **very weak** across all features:

| Feature | Correlation with Magnitude |
|---|---|
| `kurtosis` | −0.105 (strongest) |
| `bandwidth` | −0.074 |
| `log_energy` | +0.062 |
| `spectral_centroid` | −0.058 |
| `skewness` | −0.049 |
| `rms`, `mean`, `cav` | ~+0.044 |
| `energy`, `arias`, `spectral_energy` | ~+0.023 |

> [!WARNING]
> The weak correlations (all |r| < 0.11) explain the baseline model's poor R² = −0.39. The features were **originally engineered for XGBoost** (which handles complex nonlinear interactions via gradient boosting) and are not optimally suited for a simple MLP. Physics constraints may help regularize the model toward physically meaningful solutions.

---

## 3. Baseline Neural Network Analysis

### 3.1 Architecture

```
Input (29) → Linear(128) + ReLU → Linear(64) + ReLU → Linear(32) + ReLU → Linear(1)
```

| Component | Detail |
|---|---|
| **Model Class** | `BaselineMLP` in [model.py](file:///c:/Users/Daivik%20Sharma/OneDrive/Desktop/physics(neural%20networks)/src/model.py) |
| **Hidden Layers** | 3 (128 → 64 → 32) |
| **Activation** | ReLU |
| **Output** | Single neuron (magnitude prediction) |
| **Parameters** | ~12,961 |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | MSE (Mean Squared Error) |
| **Epochs** | 300 |
| **Batch Size** | Full-batch (entire dataset per step) |
| **Train/Test Split** | 80/20, random_state=42 |
| **Scaling** | StandardScaler (fit on train only) |

### 3.2 Training Pipeline

The pipeline in [main.py](file:///c:/Users/Daivik%20Sharma/OneDrive/Desktop/physics(neural%20networks)/main.py):

1. Load CSV → drop `filename` → separate X, y
2. Check/handle missing values (drop NaN rows)
3. Train/test split (80/20)
4. StandardScaler on features
5. Convert to PyTorch float32 tensors
6. Full-batch training for 300 epochs with MSE loss
7. Evaluate: MAE, RMSE, R²
8. Generate plots and save outputs

### 3.3 Baseline Results

| Metric | Value |
|---|---|
| **MAE** | 0.5046 |
| **RMSE** | 0.7060 |
| **R²** | **−0.3944** |

> [!WARNING]
> **R² = −0.39 means the model performs worse than simply predicting the mean magnitude for every sample.** The baseline MLP is failing to learn meaningful patterns from the 29 features.

### 3.4 Strengths

- Clean, modular codebase with separation of concerns
- `compute_loss()` is deliberately separated from the training loop for easy PINN extension
- Comments explicitly plan for PINN integration ("will be added in Stage 2")
- Reproducible (random_state=42, deterministic pipeline)
- Publication-quality plot generation

### 3.5 Weaknesses

- **Full-batch training** — no mini-batches or DataLoaders, limiting scalability
- **No regularization** — no dropout, batch normalization, or weight decay
- **No learning rate scheduling** — fixed lr=0.001 for all 300 epochs
- **No early stopping** — trains for fixed epochs regardless of overfitting
- **Validation = test set** — using the test set as validation during training leaks information
- **No feature engineering** — degenerate features (`dominant_freq`, `predominant_period`, `zero_crossings`) are fed directly to the network
- The architecture (3 hidden layers, ReLU only) is too simple for the weak feature–target relationships

---

## 4. PINN Design Plan

### 4.1 Physics Laws to be Used

Based on the DOCX proposal and scientific literature:

#### 4.1.1 Energy–Magnitude Scaling (Gutenberg–Richter Energy Relation)

The well-established empirical relationship:

```
log₁₀(E) = 1.5M + 4.8
```

Where E = seismic energy and M = magnitude. This is the most fundamental seismological scaling law and is supported by:
- The original Gutenberg & Richter (1956) formulation
- The DOCX proposal (Section 1 and Section 2)
- The dataset contains both `energy` and `log_energy` features

**Implementation**: The predicted magnitude should be consistent with the measured energy values. The physics loss penalizes deviations from this relationship.

> [!IMPORTANT]
> **Critical issue**: The features have been StandardScaler-normalized, so the raw `energy` values are not available during training. The physics constraint must work with either:
> (a) Raw feature values passed alongside the normalized features, OR
> (b) Inverse-transformed energy values recovered from the scaler.
> 
> Approach (a) is cleaner and will be implemented.

#### 4.1.2 Frequency–Magnitude Relation

The DOCX proposal specifies:
```
M ∝ −log(f)
```

Where f = dominant frequency.

> [!CAUTION]
> **This constraint CANNOT be implemented as proposed** because `dominant_freq` is **all zeros** in the dataset. Using it would produce `−log(0+ε)` ≈ a large constant for every sample, providing no meaningful gradient.
> 
> **Scientifically appropriate alternative**: Use `spectral_centroid` (which represents the center of mass of the frequency spectrum and has non-zero values, std=1.54) as a proxy for the dominant frequency. The spectral centroid has a correlation of −0.058 with magnitude, consistent with the physics: larger earthquakes have lower spectral centroids. Alternatively, use `bandwidth` (r=−0.074). This substitution is documented and justified.

### 4.2 How Physics Loss Will Be Integrated

```
L_total = L_prediction + λ₁·L_energy + λ₂·L_frequency
```

- **L_prediction**: MSE between predicted and actual magnitudes
- **L_energy**: Penalizes deviation from log₁₀(E) = 1.5M + 4.8
- **L_frequency**: Penalizes violations of magnitude–frequency inverse scaling
- **λ₁, λ₂**: Configurable weights (default: 0.3 as per DOCX)

### 4.3 Expected Improvements

| Aspect | Baseline | PINN |
|---|---|---|
| Loss function | MSE only | MSE + physics constraints |
| Feature weighting | Equal | Attention-weighted |
| Physical consistency | Not enforced | Energy/frequency laws enforced |
| Regularization | None | Physics acts as regularizer |
| Interpretability | Black box | Attention weights show feature importance |

The DOCX proposal's expected performance table shows PINN improvements from RMSE 0.24 → 0.17–0.19 for the same architecture family.

---

## 5. Project Folder Analysis

### 5.1 Current Structure

```
physics(neural networks)/
├── Base paper Jo bacche padengey.pdf          # Base paper: XGBoost for EEW (Joshi et al., 2024)
├── Takes diagram and data from this 2026.pdf  # Reference: FNO+LSTM EEW (Tsai et al., 2026)
├── Physics inspired Neural Network.docx       # Project proposal with PINN design & code
├── Dataset source.txt                         # K-NET URL reference
├── earthquake_29_features.csv                 # Dataset (2497 × 31)
├── main.py                                    # Baseline pipeline entry point
├── src/
│   ├── __init__.py                            # Package initializer
│   ├── data_loader.py                         # CSV loading and feature/target separation
│   ├── preprocess.py                          # Missing values, split, scale, tensorize
│   ├── model.py                               # BaselineMLP architecture
│   ├── train.py                               # Training loop with separated loss computation
│   ├── evaluate.py                            # Metrics (MAE, RMSE, R²) and 3 plot types
│   └── utils.py                               # Directory setup, save/load utilities
├── data/                                      # Empty (data is in project root)
├── saved_models/
│   ├── baseline_mlp.pth                       # Trained baseline model weights
│   └── scaler.pkl                             # Fitted StandardScaler
└── outputs/
    ├── metrics/
    │   └── baseline_metrics.json              # {MAE, RMSE, R2}
    ├── plots/
    │   ├── loss_curves.png                    # Train/val loss over epochs
    │   ├── predicted_vs_actual.png            # Scatter plot
    │   └── error_distribution.png             # Histogram of errors
    └── predictions/
        └── test_predictions.csv               # Actual vs predicted magnitudes
```

### 5.2 Recommendations for PINN Extension

The following new directories/files will be added:

```
physics(neural networks)/
├── config/
│   └── pinn_config.yaml                # ← NEW: Configurable hyperparameters & loss weights
├── src/
│   ├── ... (existing files unchanged)
│   ├── pinn_model.py                   # ← NEW: PINN architecture (extends BaselineMLP)
│   └── pinn_train.py                   # ← NEW: PINN training loop with combined loss
├── physics/
│   ├── __init__.py                     # ← NEW: Package init
│   ├── energy_constraint.py            # ← NEW: Energy–magnitude law
│   ├── frequency_constraint.py         # ← NEW: Frequency–magnitude law
│   └── physics_loss.py                 # ← NEW: Combined physics loss aggregator
├── main_pinn.py                        # ← NEW: PINN pipeline entry point
└── compare.py                          # ← NEW: Side-by-side model comparison
```

> [!TIP]
> The existing baseline code (`main.py`, `src/model.py`, `src/train.py`) remains **completely untouched** so both models can be independently reproduced and compared.

---

## 6. Implementation Roadmap

### Step 1 — Baseline Neural Network ✅ (Already Completed)
- Dataset loading and preprocessing
- BaselineMLP architecture (29→128→64→32→1)
- Training with MSE loss
- Evaluation metrics and visualization
- **Status**: Complete. Results: MAE=0.50, RMSE=0.71, R²=−0.39

---

### Step 2 — PINN Implementation

#### 2.1 Configuration System
- Create `config/pinn_config.yaml` with all hyperparameters
- Physics loss weights (λ₁, λ₂), learning rate, epochs, architecture params

#### 2.2 Physics Constraints Module
- `physics/energy_constraint.py` — Implements log₁₀(E) = 1.5M + 4.8
- `physics/frequency_constraint.py` — Implements spectral-centroid–magnitude relation
- `physics/physics_loss.py` — Aggregates all constraints with configurable weights

#### 2.3 PINN Model Architecture
- `src/pinn_model.py` — Extends baseline with same hidden layers (128→64→32)
- Add optional attention layer for feature weighting
- Same output dimension (single magnitude prediction) for fair comparison

#### 2.4 PINN Training Loop
- `src/pinn_train.py` — Modified training using L_total = L_data + λ·L_physics
- Pass raw (unnormalized) feature values for physics constraints
- Track individual loss components during training

---

### Step 3 — Training
- Train PINN with same split, same random seed, same epochs as baseline
- Log data loss and physics loss separately
- Save trained model and scaler

---

### Step 4 — Evaluation
- Compute MAE, RMSE, R² on the same test set
- Generate comparison plots:
  - Training loss curves (baseline vs PINN)
  - Predicted vs actual (side-by-side)
  - Error distribution (overlaid)
- Save metrics to `outputs/metrics/pinn_metrics.json`
- Create `compare.py` for automated side-by-side comparison

---

### Step 5 — Future Improvements (Post-Implementation)
- **Attention-based PINN**: Feature-level attention weighting
- **Uncertainty estimation**: Dual-output (μ, σ) with negative log-likelihood loss
- **Additional physics**: Gutenberg–Richter frequency–magnitude distribution law
- **Hyperparameter tuning**: Grid/Bayesian search over λ₁, λ₂, architecture
- **Mini-batch training**: DataLoaders for better generalization
- **Dropout / batch normalization**: Additional regularization
- **Cross-validation**: K-fold for more robust evaluation

---

## Appendix: Document Conflicts & Resolutions

| Conflict | Resolution |
|---|---|
| DOCX uses `dominant_freq` for frequency constraint, but it's all zeros in dataset | Use `spectral_centroid` as scientifically valid proxy. Documented above. |
| DOCX proposes 500 epochs; `main.py` uses 300 | Use 300 epochs for fair baseline comparison; configurable in YAML |
| DOCX includes attention + uncertainty in one monolithic script | Implement modularly: physics first, attention/uncertainty as future work (per user's phased approach) |
| Base paper uses 2960 records from MJMA 3.0–7.4; our dataset has 2497 records from 3.5–6.0 | The dataset is a subset. This is acceptable — the feature engineering methodology is identical. |
| DOCX suggests `lambda_phy = 0.3`; this may not be optimal | Use 0.3 as default, make configurable. Document as hyperparameter. |
