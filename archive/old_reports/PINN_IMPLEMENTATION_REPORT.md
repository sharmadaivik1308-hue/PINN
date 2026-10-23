# PINN Implementation Report
## Physics-Informed Neural Network for Earthquake Magnitude Prediction

---

## 1. What Was Changed from the Baseline

### 1.1 New Components Added

| Component | File | Purpose |
|---|---|---|
| **Physics Module** | `physics/energy_constraint.py` | Gutenberg–Richter energy–magnitude law |
| | `physics/frequency_constraint.py` | Frequency–magnitude inverse scaling |
| | `physics/physics_loss.py` | Aggregator with configurable weights |
| **PINN Model** | `src/pinn_model.py` | Configurable model with attention + uncertainty flags |
| **PINN Trainer** | `src/pinn_train.py` | Training loop with combined data + physics loss |
| **Configuration** | `config/config.yaml` | All hyperparameters in one YAML file |
| **Entry Point** | `main_pinn.py` | Unified PINN pipeline |
| **Experiments** | `run_all_experiments.py` | Runs all 4 model variants |
| **Comparison** | `compare.py` | Side-by-side visualisation |

### 1.2 What Was NOT Changed

All baseline files remain completely untouched:
- `main.py`, `src/model.py`, `src/train.py`, `src/data_loader.py`, `src/preprocess.py`, `src/evaluate.py`, `src/utils.py`

---

## 2. Why Each Modification Was Necessary

### 2.1 Physics Module

**Why**: A purely data-driven neural network ignores known seismological relationships. By encoding these as differentiable constraints in the loss function, we regularise the model toward physically consistent predictions.

**Why modular**: Each constraint is an independent module so that:
- New constraints can be added without modifying existing code
- Individual constraints can be toggled on/off via configuration
- Each constraint can be unit-tested independently

### 2.2 Configurable Model (`EarthquakePINN`)

**Why**: The research requires four model variants (baseline, basic PINN, attention PINN, full PINN) with the same backbone architecture. A single configurable class with `use_attention` and `use_uncertainty` flags avoids code duplication while enabling ablation studies.

### 2.3 PINN Trainer

**Why**: The baseline training loop only computes MSE loss. The PINN trainer must:
- Compute data loss (MSE or NLL depending on uncertainty mode)
- Compute physics loss from raw (un-normalised) features
- Combine them with configurable weights
- Track each loss component separately for analysis

### 2.4 YAML Configuration

**Why**: Hardcoded hyperparameters make experiments difficult to reproduce and compare. A YAML config file allows:
- Changing any parameter without modifying source code
- Documenting exact experiment settings
- Defining named experiment presets

---

## 3. Physics Constraints Implementation

### 3.1 Energy–Magnitude Scaling (Gutenberg–Richter)

**Seismological Law**:
```
log₁₀(E) = 1.5 × M + 4.8
```

**Implementation** (`physics/energy_constraint.py`):
```python
observed_log_energy = log₁₀(|raw_energy| + ε)
expected_log_energy = 1.5 × pred_magnitude + 4.8
loss_energy = mean((observed - expected)²)
```

**Key Design Decision**: The constraint uses **raw (un-normalised) energy values** from the dataset, not the StandardScaler-normalised values fed to the model. This is critical because the physics law operates in the original physical unit space.

### 3.2 Frequency–Magnitude Inverse Scaling

**Seismological Law**:
```
M ∝ −log(f)
```

**Implementation** (`physics/frequency_constraint.py`):
```python
freq_implied_magnitude = −log(raw_spectral_centroid + ε)
loss_frequency = mean((pred_magnitude - freq_implied_magnitude)²)
```

**Data Issue & Resolution**: The `dominant_freq` column is entirely zero (2497/2497 samples), making it unusable. We use `spectral_centroid` (centre of mass of the frequency spectrum) as a scientifically valid proxy. It has non-zero values (std=1.54) and a negative correlation with magnitude (r=−0.058), consistent with the expected physics.

### 3.3 Combined Physics Loss

```
L_physics = λ₁ × L_energy + λ₂ × L_frequency
```

Default weights: λ₁ = λ₂ = 0.3 (from the DOCX proposal). These are configurable via `config/config.yaml`.

---

## 4. Loss Function Design

### 4.1 Standard Mode (Basic PINN, Attention PINN)

```
L_total = L_MSE + λ₁ × L_energy + λ₂ × L_frequency
```

Where L_MSE = (1/N) Σ(yᵢ - ŷᵢ)² is the standard mean squared error.

### 4.2 Uncertainty Mode (Full PINN)

```
L_total = L_NLL + λ₁ × L_energy + λ₂ × L_frequency
```

Where L_NLL is the negative log-likelihood of a Gaussian:

```
L_NLL = (1/N) Σ [ (yᵢ - μᵢ)² / (2σᵢ²) + log(σᵢ) ]
```

The model outputs two values: μ (predicted magnitude) and log(σ) (log-uncertainty). The uncertainty σ is parameterised as exp(log_σ) to ensure positivity.

---

## 5. Training Process Differences

| Aspect | Baseline | PINN Variants |
|---|---|---|
| **Loss** | MSE only | MSE/NLL + physics constraints |
| **Raw features needed** | No | Yes (energy, spectral_centroid) |
| **Loss components tracked** | Total, validation | Total, data, energy, frequency, physics, validation |
| **Feature weighting** | Equal (baseline, basic PINN) | Learned attention (attention PINN, full PINN) |
| **Output dimension** | 1 (magnitude) | 1 or 2 (magnitude + uncertainty) |
| **Configuration** | Hardcoded in `main.py` | YAML-driven |

---

## 6. Model Variants and Advantages

### 6.1 Basic PINN
- **Same architecture** as baseline (29→128→64→32→1)
- **Added**: Physics loss as training regulariser
- **Advantage**: Physics prevents physically impossible predictions

### 6.2 Attention PINN
- **Added**: Sigmoid attention gate before backbone
- **Architecture**: 29→Attention→128→64→32→1
- **Advantage**: Learns which features are most relevant; provides interpretability through attention weights visualisation

### 6.3 Full PINN
- **Added**: Attention + dual-output head (μ, σ)
- **Architecture**: 29→Attention→128→64→32→2
- **Advantage**: Provides confidence estimates alongside magnitude predictions; essential for safety-critical EEW applications

---

## 7. Assumptions Made

1. **StandardScaler is appropriate**: Features have different scales and distributions; z-score normalisation is standard practice
2. **Spectral centroid as frequency proxy**: The `dominant_freq` column being all zeros is a data extraction issue, not a fundamental limitation. `spectral_centroid` captures the same physical concept.
3. **Gutenberg–Richter coefficients**: The standard values (a=1.5, b=4.8) are used without regional calibration. These may not be optimal for the specific magnitude range (3.5–6.0) and region (Eurasian Plate, Japan).
4. **Physics loss weights**: λ₁ = λ₂ = 0.3 is taken from the DOCX proposal. Optimal values likely differ and could be found through hyperparameter search.
5. **Full-batch training**: Following the baseline's approach for consistency. Mini-batch training with DataLoaders would likely improve generalisation.
6. **Single train/test split**: Using a fixed 80/20 split rather than k-fold cross-validation. This follows the baseline methodology.

---

## 8. Suggestions for Future Work

### 8.1 Immediate Improvements
- **Hyperparameter tuning**: Grid or Bayesian search over λ₁, λ₂, learning rate, and architecture
- **Mini-batch training**: Implement DataLoaders for better gradient estimates
- **Early stopping**: Monitor validation loss and stop when it stops improving
- **Learning rate scheduling**: ReduceLROnPlateau or cosine annealing

### 8.2 Advanced Extensions
- **Multi-Physics Constraints**: Add Gutenberg–Richter frequency–magnitude distribution law (log₁₀(N) = a − bM) as a distributional constraint
- **Multi-head Attention**: Replace single-layer sigmoid attention with multi-head self-attention for richer feature interactions
- **Dropout / Batch Normalisation**: Add regularisation between hidden layers
- **Feature selection**: Remove the three degenerate features (`dominant_freq`, `predominant_period`, `zero_crossings`)
- **Cross-validation**: K-fold for more robust evaluation
- **Regional calibration**: Adjust Gutenberg–Richter coefficients for the specific dataset region
- **Ensemble methods**: Train multiple PINNs with different seeds and aggregate predictions
- **Transfer learning**: Pre-train on a larger global earthquake dataset, fine-tune on this regional dataset

### 8.3 Publication Enhancements
- **Ablation study**: Systematically disable each component to quantify its contribution
- **Statistical significance**: Paired t-tests or bootstrap confidence intervals on metric differences
- **Sensitivity analysis**: Vary physics loss weights and plot the resulting metric landscape
- **Comparison with XGBoost**: Compare PINN results against the original XGBoost baseline from the base paper
