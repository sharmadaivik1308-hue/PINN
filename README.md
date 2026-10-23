# Earthquake Magnitude Prediction — Optimized Physics-Informed Neural Network

A Physics-Informed Neural Network (PINN) for predicting earthquake magnitude from 27 P-wave features extracted during the first 3 seconds of seismic recordings.

## What This Model Does

The Optimized PINN predicts earthquake magnitude by combining:

- **Data-driven learning** — Learns patterns from 27 seismic signal features (energy, spectral centroid, PGA, RMS, etc.)
- **Physics constraints** — Enforces two seismological scaling laws during training:
  - **Gutenberg–Richter energy–magnitude relation**: log₁₀(E) ∝ 1.5 × M
  - **Frequency–magnitude inverse scaling**: M ∝ −log(f)

## Model Architecture

```
Input (27 features)
    ↓
ScaledAttention — learnable feature importance with temperature
    ↓
ResidualBlock(27 → 256)  — Linear + GELU + Dropout(0.1) + LayerNorm + Skip
ResidualBlock(256 → 128) — Linear + GELU + Dropout(0.1) + LayerNorm + Skip
ResidualBlock(128 → 64)  — Linear + GELU + Dropout(0.1) + LayerNorm + Skip
    ↓
Linear(64 → 1)
    ↓
Predicted Magnitude
```

**Key components:**
| Component | Setting |
|---|---|
| Hidden layers | 256 → 128 → 64 |
| Activation | GELU |
| Dropout | 0.1 |
| Normalization | Layer Normalization |
| Residual connections | Enabled |
| Attention | Scaled attention with learnable temperature |
| Uncertainty | Disabled |
| Physics constraints | Energy + Frequency |

## Dataset

- **Source**: Earthquake P-wave recordings (K-NET, Japan)
- **Samples**: 2,497 earthquake records
- **Features**: 27 signal features (after removing degenerate columns)
- **Target**: Earthquake magnitude (discrete values: 3.5–6.0)
- **File**: `data/earthquake_29_features.csv`

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

### Default (uses config/config.yaml)
```bash
python main.py
```

### Override learning rate and epochs from CLI
```bash
python main.py --lr 0.0005 --epochs 1000
python main.py --batch-size 128
```

### Use a different config file
```bash
python main.py --config path/to/my_config.yaml
```

## Configuration

All hyperparameters are controlled from `config/config.yaml`.

### To change the learning rate:
```yaml
# config/config.yaml
training:
  learning_rate: 0.001    # ← Change this value
```

### To change the number of epochs:
```yaml
# config/config.yaml
training:
  epochs: 500             # ← Change this value
```

### To change physics loss weights:
```yaml
# config/config.yaml
physics:
  lambda_energy: 0.05     # ← Energy constraint weight
  lambda_frequency: 0.05  # ← Frequency constraint weight
```

### To change model architecture:
```yaml
# config/config.yaml
model:
  hidden_layers: [256, 128, 64]   # ← Layer sizes
  activation: "gelu"               # ← gelu, silu, relu, leakyrelu
  dropout: 0.1                     # ← Dropout rate
```

## Where Results Are Saved

```
outputs/
├── model/
│   ├── optimized_pinn.pth      # Trained model weights (best validation)
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── feature_names.json      # Feature column names
│
├── metrics/
│   ├── metrics.json            # MAE, RMSE, R², training time
│   ├── loss_history.json       # Per-epoch loss components
│   ├── predictions.csv         # Test set predictions vs actuals
│   └── config_snapshot.yaml    # Config used for this run
│
└── plots/
    ├── loss_curves.png         # Training & validation loss
    ├── physics_losses.png      # Physics loss over training
    ├── predicted_vs_actual.png # Scatter plot
    ├── error_distribution.png  # Error histogram
    ├── attention_weights.png   # Feature importance
    └── learning_rate_schedule.png
```

## Project Structure

```
Earthquake_PINN/
│
├── data/
│   └── earthquake_29_features.csv
│
├── src/
│   ├── data_loader.py          # Dataset loading
│   ├── preprocess.py           # Missing values, scaling, tensors
│   ├── pinn_model.py           # OptimizedPINN architecture
│   ├── train.py                # Training pipeline
│   └── evaluate.py             # Metrics and plotting
│   └── utils.py                # Save/load utilities
│
├── physics/
│   ├── physics_loss.py         # Aggregated physics loss
│   ├── energy_constraint.py    # Gutenberg–Richter constraint
│   └── frequency_constraint.py # Frequency–magnitude constraint
│
├── config/
│   └── config.yaml             # All hyperparameters
│
├── outputs/                    # Generated outputs (model, metrics, plots)
├── reports/                    # Final model report
├── archive/                    # Previous experiments (preserved)
│
├── main.py                     # Entry point
├── requirements.txt
└── README.md
```

## Archive

Previous experimental implementations are preserved in `archive/`:
- `archive/baseline_mlp/` — Original data-only MLP
- `archive/basic_pinn/` — First physics-constrained model
- `archive/attention_pinn/` — Attention + physics
- `archive/full_pinn/` — Attention + uncertainty + physics
- `archive/old_reports/` — Previous analysis reports
- `archive/old_scripts/` — Comparison and optimization scripts

## Training Details

- **Optimizer**: AdamW with weight decay (1e-4)
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=15)
- **Early Stopping**: Patience=50, restores best validation model
- **Gradient Clipping**: Max norm = 1.0
- **Mini-batch**: Batch size 64
- **Physics Loss**: Applied on full training set each epoch
