# Earthquake Magnitude Prediction — Optimized Physics-Informed Neural Network

A Physics-Informed Neural Network (PINN) for predicting earthquake magnitude from P-wave features extracted during the first 3 seconds of seismic recordings (K-NET, Japan).

## What This Model Does

The Optimized PINN predicts earthquake magnitude by combining:

- **Data-driven learning** — Learns patterns from 19 seismic signal features selected after degenerate and correlation audit (energy, spectral centroid, PGA, RMS, etc.)
- **Physics constraints** — Enforces seismological scaling laws during training:
  - **Gutenberg–Richter energy–magnitude relation**: `log₁₀(E) ∝ 1.5 × M` (proportionality constraint via correlation mode or calibrated G-R scaling)
  - **Frequency constraint**: Optional inverse scaling `M ∝ −log(f)` (disabled by default, `λ_freq = 0.0`, due to weak correlation with spectral centroid proxy)

## Model Architecture

```
Input (19 features)
    ↓
ScaledAttention — learnable feature importance with temperature
    ↓
ResidualBlock(19 → 64)  — Linear + GELU + Dropout(0.1) + LayerNorm + Skip
ResidualBlock(64 → 32)  — Linear + GELU + Dropout(0.1) + LayerNorm + Skip
ResidualBlock(32 → 16)  — Linear + GELU + Dropout(0.1) + LayerNorm + Skip
    ↓
Linear(16 → 1)
    ↓
Predicted Magnitude
```

**Key components:**
| Component | Setting |
|---|---|
| Hidden layers | [64, 32, 16] |
| Activation | GELU |
| Dropout | 0.1 |
| Normalization | Layer Normalization |
| Residual connections | Enabled |
| Attention | Scaled attention with learnable temperature |
| Uncertainty | Disabled |
| Physics constraint | Energy constraint (λ_energy = 0.1, λ_freq = 0.0) |

## Dataset

- **Source**: Earthquake P-wave recordings (K-NET, Japan)
- **Samples**: 2,497 seismic recording rows across 12 unique earthquake events
- **Features**: 19 features used after automated audit removes 2 degenerate and 8 collinear features from the original 29 features
- **Target**: Earthquake magnitude (discrete values: 3.5–6.0)
- **File**: `data/earthquake_29_features.csv`
- **Split Strategy**: Event-based (`dataset.split_strategy: "event"` using `GroupShuffleSplit` and `LeaveOneGroupOut`) to prevent cross-station event leakage.

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

### Default (uses config/config.yaml with event-based splits)
```bash
python main.py
```

### Run pre-split model (newdata configuration)
```bash
python main.py --config config/newdata_config.yaml
```

### Override learning rate and epochs from CLI
```bash
python main.py --lr 0.0005 --epochs 500
python main.py --batch-size 64
```

## Configuration

All hyperparameters are controlled from `config/config.yaml` as the single source of truth.

```yaml
# config/config.yaml
dataset:
  split_strategy: "event"  # "event" (valid) | "random" (leaky)
  scale_target: true

model:
  hidden_layers: [64, 32, 16]
  activation: "gelu"
  dropout: 0.1

training:
  epochs: 500
  learning_rate: 0.001
  batch_size: 64
  weight_decay: 0.0001
  patience: 50
  scheduler: "plateau"
  scheduler_plateau_patience: 30

physics:
  enabled: true
  energy_mode: "correlation"  # "correlation" | "gr_absolute"
  lambda_energy: 0.1
  lambda_frequency: 0.0
```

## Training Details

- **Optimizer**: AdamW with weight decay (1e-4)
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=30)
- **Early Stopping**: Patience=50, restores best validation model
- **Gradient Clipping**: Max norm = 1.0
- **Mini-batch**: Batch size 64
- **Physics Loss**: Computed jointly on each mini-batch via `TensorDataset` with a single combined backward pass

## Known Limitations

- **Small number of events**: The dataset contains 2,497 station recordings but only 12 unique earthquake events. Generalisation to unseen earthquake events is inherently challenging.
- **Quasi-discrete target**: Only 11 distinct magnitude values exist across the 12 events.
- **Extraction bugs in frequency features**: `dominant_freq` and `predominant_period` are 0.0 across all 2,497 rows due to upstream feature-extraction failure.
- **TODO (Distance Information)**: Amplitude features (PGA, RMS, energy) are distance-dependent. Epicentral and hypocentral distances are not provided in the current dataset. If station coordinates or event hypocenters become available, add `log10(distance)` and distance-corrected `log_pga` and `log_energy` features to eliminate attenuation bias.
