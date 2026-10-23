# Baseline Model Report
## Earthquake Magnitude Prediction — Baseline MLP

---

## 1. What Was Implemented

The baseline model is a standard feed-forward Multi-Layer Perceptron (MLP) for regression, designed to predict earthquake magnitude from 29 engineered P-wave features.

### Architecture

```
Input (29 features)
    ↓
Linear(29, 128) + ReLU
    ↓
Linear(128, 64) + ReLU
    ↓
Linear(64, 32) + ReLU
    ↓
Linear(32, 1)
    ↓
Predicted Magnitude
```

| Component | Detail |
|---|---|
| **Input Dimension** | 29 seismic features |
| **Hidden Layers** | 3 (128 → 64 → 32 neurons) |
| **Activation** | ReLU (Rectified Linear Unit) |
| **Output** | 1 neuron (magnitude prediction) |
| **Total Parameters** | ~12,961 |
| **Loss Function** | Mean Squared Error (MSE) |
| **Optimizer** | Adam (lr = 0.001) |
| **Training** | Full-batch, 300 epochs |

---

## 2. Why This Architecture

The baseline serves as the **control experiment** in the research methodology:

- **Same architecture** as the proposed PINN backbone, enabling a fair comparison where the only variable is the loss function
- **Simple enough** to clearly demonstrate the impact of physics constraints when added
- **Standard MLP** is the most common baseline in deep learning for structured data regression
- The hidden layer sizes (128→64→32) match the DOCX proposal's architecture specification

---

## 3. Mathematical Background

### Forward Pass

For an input vector **x** ∈ ℝ²⁹:

```
h₁ = ReLU(W₁x + b₁)    where W₁ ∈ ℝ¹²⁸ˣ²⁹
h₂ = ReLU(W₂h₁ + b₂)   where W₂ ∈ ℝ⁶⁴ˣ¹²⁸
h₃ = ReLU(W₃h₂ + b₃)   where W₃ ∈ ℝ³²ˣ⁶⁴
ŷ  = W₄h₃ + b₄          where W₄ ∈ ℝ¹ˣ³²
```

### Loss Function

Standard MSE loss:

```
L_MSE = (1/N) Σᵢ (yᵢ - ŷᵢ)²
```

### Optimization

Adam optimizer with β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸ and learning rate = 0.001.

---

## 4. Code Structure

| File | Purpose |
|---|---|
| `src/model.py` | `BaselineMLP` class definition |
| `src/train.py` | Training loop with separated `compute_loss()` |
| `src/data_loader.py` | CSV loading and feature/target separation |
| `src/preprocess.py` | Missing values, train/test split, StandardScaler, tensorisation |
| `src/evaluate.py` | MAE, RMSE, R² computation and plot generation |
| `src/utils.py` | Directory setup, model/scaler/predictions/metrics saving |
| `main.py` | Pipeline orchestration |

---

## 5. Data Pipeline

1. **Load**: Read `earthquake_29_features.csv`, drop `filename` column
2. **Clean**: Check for and drop rows with NaN values (none found in this dataset)
3. **Split**: 80/20 train/test split with `random_state=42` for reproducibility
4. **Scale**: `StandardScaler` fit on training data, applied to both train and test
5. **Tensorise**: Convert to PyTorch `float32` tensors

---

## 6. Results

| Metric | Value |
|---|---|
| **MAE** | 0.5046 |
| **RMSE** | 0.7060 |
| **R²** | −0.3944 |

### Interpretation

- **R² = −0.39** indicates the model performs **worse than predicting the mean magnitude** for every sample. This is the "do nothing" baseline.
- **MAE = 0.50** means predictions are off by about half a magnitude unit on average
- The poor performance is expected given:
  - All feature–magnitude correlations are weak (max |r| = 0.105)
  - Features were originally engineered for XGBoost (tree-based), not neural networks
  - No regularisation, no feature selection, full-batch training

---

## 7. Advantages

1. **Clean, modular code** — each component is independently testable
2. **Reproducible** — fixed random seeds, deterministic pipeline
3. **Extensible** — `compute_loss()` deliberately separated for PINN extension
4. **Publication baseline** — provides the mandatory comparison point for the PINN

---

## 8. Limitations

1. **No physics knowledge** — the network must learn everything from data alone
2. **No regularisation** — no dropout, batch norm, or weight decay
3. **Full-batch training** — uses all data per step (no mini-batches)
4. **No early stopping** — trains for fixed 300 epochs regardless of convergence
5. **Degenerate features** — feeds zero-variance columns (`dominant_freq`, `predominant_period`) that waste model capacity
6. **Weak linear correlations** — the 29 features have very weak linear relationships with magnitude

---

## 9. Future Work

The baseline is improved in subsequent stages by adding:
- **Physics constraints** (Basic PINN) — energy and frequency scaling laws
- **Attention mechanism** (Attention PINN) — learnable feature weighting
- **Uncertainty estimation** (Full PINN) — probabilistic predictions with confidence intervals
