# MODEL OPTIMIZATION REPORT
## Physics-Informed Neural Network for Earthquake Magnitude Prediction

---

## 1. Original Implementation Weaknesses

| # | Weakness | Evidence |
|---|---------|----------|
| 1 | Full-batch training (no mini-batches) | All 1,997 train samples processed per step |
| 2 | No regularization | No dropout, batch norm, weight decay |
| 3 | Degenerate features fed to model | `dominant_freq`, `predominant_period` are ALL ZEROS |
| 4 | Massive feature redundancy | `pga`=`max_val`=`envelope_peak` (r=1.0); `mean`≈`rms`≈`cav` (r>0.99) |
| 5 | Fixed learning rate (no scheduling) | LR=0.001 for all 300 epochs |
| 6 | No early stopping | Trains for exactly 300 epochs regardless |
| 7 | Physics loss weights not tuned | Fixed λ=0.1 with no search |
| 8 | Simple ReLU activation only | No GELU/SiLU alternatives tested |
| 9 | No skip/residual connections | Poor gradient flow in deeper networks |
| 10 | Validation = test set | Information leakage during training |

---

## 2. Optimizations Attempted

### 2.1 Feature Engineering (Phase 4)
- **Removed 3 degenerate features**: `dominant_freq`, `predominant_period`, `zero_crossings`
- **Removed ~8 redundant features**: Highly correlated duplicates (pga/max_val/envelope_peak, mean/rms/abs_mean/cav, etc.)
- **Added ~10 physics-inspired features**: `log_pga`, `energy_per_duration`, `log_spectral_centroid`, etc.

### 2.2 Architecture Improvements (Phase 5)
- Residual connections for gradient flow
- Layer Normalization for training stability
- GELU/SiLU activations for smoother gradients
- Dropout for regularization
- Improved attention mechanism with learnable temperature
- Xavier weight initialization

### 2.3 Training Pipeline (Phase 8)
- Mini-batch training with DataLoaders
- AdamW optimizer with weight decay
- ReduceLROnPlateau / CosineAnnealing learning rate scheduling
- Gradient clipping (max norm = 1.0)
- Early stopping with patience
- Best model checkpointing

### 2.4 Physics Loss Optimization (Phase 6)
- Systematic search of λ_energy and λ_frequency via Optuna
- Optional physics loss annealing (gradual increase)

### 2.5 Hyperparameter Optimization (Phase 7)
- Optuna-based Bayesian optimization
- Tuned: learning rate, batch size, hidden layers, dropout, activation, physics weights

---

## 3. Final Performance Comparison

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 0.4604 | 0.5703 | 0.0902 |
| XGBoost | 0.4629 | 0.5774 | 0.0672 |
| LightGBM | 0.4632 | 0.5778 | 0.0661 |
| Optimized PINN | 0.4711 | 0.5784 | 0.0640 |
| full_pinn | 0.4971 | 0.6160 | -0.0617 |
| attention_pinn | 0.5036 | 0.6388 | -0.1416 |
| Linear Regression | 0.4938 | 0.6665 | -0.2427 |
| basic_pinn | 0.5114 | 0.6705 | -0.2579 |
| baseline_mlp | 0.5014 | 0.6974 | -0.3607 |

---

## 4. Cross-Validation Results (5 Folds)

| Metric | Mean | Std |
|---|---|---|
| MAE | 0.472352 | 0.019714 |
| RMSE | 0.577906 | 0.024849 |
| R² | 0.042458 | 0.036534 |

---

## 5. Final Architecture

```
OptimizedPINN(input=27, hidden=[256, 128, 64], act=gelu, dropout=0.1, residual=ON, layernorm=ON, attention=ON, uncertainty=OFF)
Parameters: 99,114
```

### Best Hyperparameters

```json
{
    "model": {
        "input_dim": 27,
        "hidden_layers": [
            256,
            128,
            64
        ],
        "activation": "gelu",
        "dropout": 0.1,
        "use_layer_norm": true,
        "use_batch_norm": false,
        "use_residual": true,
        "use_attention": true,
        "use_uncertainty": false
    },
    "train": {
        "epochs": 500,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "grad_clip": 1.0,
        "patience": 50,
        "print_every": 20,
        "use_physics": true,
        "use_uncertainty": false,
        "lambda_energy": 0.05,
        "lambda_frequency": 0.05,
        "scheduler": "plateau",
        "physics_annealing": false
    }
}
```

---

## 6. Key Findings

1. **Tree-based models significantly outperform neural networks** on this tabular dataset. This is consistent with published ML literature for small tabular datasets.

2. **Feature engineering provides substantial improvement**: Removing degenerate features and adding derived features improved all models.

3. **Training improvements (mini-batch, LR scheduling, early stopping)** are critical for neural network performance on this dataset.

4. **Physics constraints provide modest regularization benefit** but cannot overcome the fundamental limitation of weak feature-target correlations (max |r| < 0.11).

5. **The dataset's discrete target** (11 unique magnitude values) and small size (2,497 samples) inherently limit neural network regression performance.

---

## 7. Remaining Limitations

1. All feature-target correlations are weak (|r| < 0.11)
2. Target is discrete (11 unique values), not continuous
3. Small dataset (2,497 samples) for deep learning
4. Features were engineered for XGBoost, not neural networks
5. Physics constraints are approximate (correlation-based, not absolute)

---

## 8. Recommendations for Future Work

1. **Use ensemble methods** (XGBoost/LightGBM) as primary prediction model
2. **PINN as regularizer** — use physics constraints to regularize ensemble predictions
3. **Larger dataset** — collect more earthquake records for deep learning
4. **Raw waveform input** — use CNN/LSTM on raw seismograms instead of hand-crafted features
5. **Multi-task learning** — predict magnitude + location + depth jointly
6. **Transfer learning** — pre-train on global earthquake catalog, fine-tune on K-NET

---

## 9. Generated Outputs

### Analysis
- `outputs/plots/analysis/` — Dataset analysis visualizations
- `outputs/plots/feature_engineering/` — Feature importance plots

### Optimized Model
- `outputs/optimized/models/` — Trained model weights
- `outputs/optimized/plots/` — Training curves, evaluation plots
- `outputs/optimized/metrics/` — All metrics and HPO results

### Reports
- `MODEL_OPTIMIZATION_REPORT.md` — This report
