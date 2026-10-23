# Final Model Report
## Optimized Physics-Informed Neural Network for Earthquake Magnitude Prediction

---

## 1. Summary

This project implements a progressive series of neural network models for earthquake magnitude prediction from P-wave features, culminating in an **Optimized Physics-Informed Neural Network** with residual connections, layer normalization, GELU activation, scaled attention, and physics-based regularization.

### Model Progression

```
Stage 1: Baseline MLP (data-driven only)
    ↓ + Physics constraints
Stage 2: Basic PINN (physics-regularised)
    ↓ + Feature attention
Stage 3: Attention PINN (physics + learned feature importance)
    ↓ + Uncertainty estimation
Stage 4: Full PINN (physics + attention + uncertainty)
    ↓ + Residual connections, LayerNorm, GELU, mini-batch, early stopping
Stage 5: Optimized PINN ← FINAL MODEL
```

---

## 2. Final Architecture

```
OptimizedPINN(
    input=27,
    hidden=[256, 128, 64],
    act=gelu,
    dropout=0.1,
    residual=ON,
    layernorm=ON,
    attention=ON,
    uncertainty=OFF
)
```

### Training Configuration
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=15)
- **Early Stopping**: patience=50, restores best model
- **Gradient Clipping**: max_norm=1.0
- **Batch Size**: 64
- **Max Epochs**: 500

### Physics Constraints
- **Energy**: λ_energy = 0.05 — Gutenberg–Richter log₁₀(E) ∝ 1.5 × M
- **Frequency**: λ_frequency = 0.05 — M ∝ −log(f) using spectral_centroid

---

## 3. Physics Implementation

### Energy–Magnitude Relation (Gutenberg–Richter)
The `energy` feature is a P-wave integral (∫a²dt), not total seismic energy. The constraint enforces the **proportionality direction** between log₁₀(energy) and predicted magnitude using normalised correlation-based loss.

### Frequency–Magnitude Relation
Uses `spectral_centroid` as proxy (since `dominant_freq` is all zeros). Enforces the inverse relationship: higher frequencies → lower magnitudes.

---

## 4. Previous Experimental Results

| Model | MAE | RMSE | R² | Parameters |
|---|---|---|---|---|
| Baseline MLP | 0.5014 | 0.6974 | −0.3607 | 14,209 |
| Basic PINN | 0.5114 | 0.6705 | −0.2579 | 14,209 |
| Attention PINN | 0.5036 | 0.6388 | −0.1416 | 15,079 |
| Full PINN | 0.4971 | 0.6160 | −0.0617 | 15,112 |
| **Optimized PINN** | **Run `python main.py` to generate** | — | — | ~123K |

> **Note**: The Optimized PINN results are generated fresh on each run. Previous results from the optimization pipeline are archived in `archive/old_outputs/`.

---

## 5. Limitations

1. **Negative R² values**: All models have R² < 0 on this dataset. The 29 features from a 3-second P-wave window have weak linear correlations with magnitude (max |r| ≈ 0.10).
2. **Discrete target**: Only 11 unique magnitude values (3.5–6.0), limiting continuous regression resolution.
3. **Small dataset**: 2,497 samples is relatively small for deep learning.
4. **Feature mismatch**: Features were engineered for XGBoost, not neural networks.
5. **Physics approximation**: Constraints use normalised correlation, not absolute Gutenberg–Richter values.

---

## 6. Outputs

All outputs are saved to `outputs/`:
- `outputs/model/` — Model weights (.pth), scaler (.pkl), feature names (.json)
- `outputs/metrics/` — Metrics, loss history, predictions, config snapshot
- `outputs/plots/` — All visualizations

---

## 7. Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Train and evaluate
python main.py

# With custom parameters
python main.py --lr 0.0005 --epochs 1000
```

Previous experiments are preserved in `archive/` for comparison.
