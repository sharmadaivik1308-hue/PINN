# COMPLETE PROJECT PROGRESS REPORT
## Physics-Informed Neural Networks for Earthquake Magnitude Prediction

> Document type: Internal technical record and research-paper reference
> Generated: 2026-10-14
> Audit scope: All Python source files, configuration files, metric JSON files,
> generated reports, archived scripts, and reference documents.
> Evidence standard: Source code, generated metric files, and auto-generated
> audit reports are primary evidence. Older narrative reports are secondary.

---

## 1. Project Objective

Predict earthquake magnitude from 29 engineered features extracted from the
first 3 seconds of P-wave arrivals recorded by K-NET (Japan) stations.
Physics-Informed Neural Networks encode Gutenberg-Richter energy-magnitude scaling
and frequency-magnitude inverse scaling as differentiable soft constraints.

Dataset: data/earthquake_29_features.csv
Target: magnitude (JMA scale), discrete, 11 unique values (3.5-6.0)
Samples: 2,497 | Missing values: 0 | Unique events: 12 | Unique stations: 563

---

## 2. Dataset Analysis

### Feature Categories (29 total)
- Amplitude: pga, max_val, min_val, range, envelope_peak
- Statistical: mean, std, skewness, kurtosis, median, abs_mean
- Quantiles: p25, p75, iqr, p90, p95
- Energy: energy, log_energy, arias, spectral_energy, rms, cav
- Frequency: dominant_freq, spectral_centroid, bandwidth, predominant_period
- Time-domain: zero_crossings, duration, rise_time

### Confirmed Degenerate Features (outputs/reports/FEATURE_ANALYSIS_REPORT.md)
- dominant_freq: ALL 2,497 values = 0
- predominant_period: ALL 2,497 values = 0

### Feature-Magnitude Correlations (all weak, |r| < 0.11)
kurtosis=-0.105, bandwidth=-0.074, log_energy=+0.062, spectral_centroid=-0.058

### Confirmed Redundant Feature Groups (FEATURE_ANALYSIS_REPORT.md)
Removed 8 features with |r| > 0.95:
rms (=pga), mean (=max_val), std (=range), min_val (=median),
abs_mean (=cav), arias (=energy), p25 (=envelope_peak), p90 (=p75)

Final feature set: 19 features (saved in outputs/model/feature_names.json)

---

## 3. Model Progression

### Stage 1: Baseline MLP [COMPLETED AND VERIFIED]
File: archive/baseline_mlp/model.py
Architecture: Input(29)->Linear(128)+ReLU->Linear(64)+ReLU->Linear(32)+ReLU->Linear(1)
Parameters: ~12,961 | Loss: MSE | Optimizer: Adam lr=0.001
Training: Full-batch, 300 epochs, 80/20 split, NO early stopping, NO validation set
DEFECT: Test set used as validation (leakage)

Verified Results (archive/old_outputs/metrics/baseline_mlp_metrics.json):
  MAE=0.5014, RMSE=0.6974, R2=-0.3607, Time=3.35s

### Stage 2: Basic PINN [COMPLETED AND VERIFIED]
File: archive/basic_pinn/pinn_model.py
Added: L_total = L_MSE + 0.3*L_energy + 0.3*L_frequency (DOCX proposal defaults)
DEFECT: Two separate backward() passes wiped data gradients
DEFECT: Physics not batch-aligned

Verified Results (archive/old_outputs/metrics/basic_pinn_metrics.json):
  MAE=0.5114, RMSE=0.6705, R2=-0.2579, Time=1.4s, Params=14,209

### Stage 3: Attention PINN [COMPLETED AND VERIFIED]
Added: Sigmoid attention gate x_attn = x * sigmoid(Linear(29,29)(x))
Parameters: ~15,079

Verified Results (archive/old_outputs/metrics/all_experiments_comparison.json):
  MAE=0.5036, RMSE=0.6388, R2=-0.1416, Time=2.03s

### Stage 4: Full PINN (Uncertainty) [IMPLEMENTED AND VERIFIED VIA ARTIFACTS]
Added: Dual output head [mu, log_sigma], heteroscedastic NLL + 0.1*MSE_anchor
Parameters: ~15,112
NOTE: Calibration NOT evaluated (no ECE or coverage metrics)

Verified Results (archive/old_outputs/metrics/all_experiments_comparison.json):
  MAE=0.4971, RMSE=0.6160, R2=-0.0617, Time=2.0s

### Stage 5a: Optimized PINN on Original Dataset [EXPERIMENTALLY EVALUATED]
Entry point: python main.py (config/config.yaml)
Dataset: data/earthquake_29_features.csv, 19 features, 70/15/15 split
CRITICAL DISCREPANCY: README/FINAL_MODEL_REPORT say hidden=[256,128,64]
  but config/config.yaml specifies hidden=[64,32,16] and
  outputs/metrics/config_snapshot.yaml CONFIRMS [64,32,16] was used.

Actual architecture (outputs/metrics/config_snapshot.yaml):
  input=19, hidden=[64,32,16], act=gelu, dropout=0.1
  residual=ON, layernorm=ON, attention=ON, uncertainty=OFF
  Parameters: 8,778 (outputs/metrics/metrics.json)

Training: AdamW lr=0.001 wd=1e-4, ReduceLROnPlateau factor=0.5 patience=30
  Early stopping patience=50, batch_size=64, max_epochs=500, grad_clip=1.0
  lambda_energy=0.1, lambda_frequency=0.0

VERIFIED Results (outputs/metrics/metrics.json):
  MAE=0.494007, RMSE=0.606174, R2=0.014694
  Training time=436.49s, Epochs=147, Params=8,778, Device=CPU
  Best val loss=0.966317

Ablation (outputs/reports/ablation_results.md - VERIFIED):
  A: Baseline (no reg, no phys): R2=-0.4383, MAE=0.5891, Time=81.4s
  B: Regularized (no phys): R2=0.0279, MAE=0.4868, Time=10.5s
  C: Energy PINN: R2=0.0102, MAE=0.4905, Time=11.5s
  D: Full PINN (energy+freq): R2=0.0148, MAE=0.4912, Time=11.0s

5-Fold CV (outputs/reports/MODEL_TRAINING_AUDIT.md - VERIFIED):
  R2=0.0374 +/- 0.021, MAE=0.4796 +/- 0.0198, RMSE=0.5748 +/- 0.0208

Mean-prediction baseline:
  PINN R2=0.0207 vs Baseline R2=-0.0026

### Stage 5b: Optimized PINN (Event-Based Split) [EXPERIMENTALLY EVALUATED]
Entry: python main.py --config config/newdata_config.yaml
Dataset: newdata(claude)/train.csv (1,653), val.csv (550), test.csv (294)
Split: Event-based (8 train / 2 val / 2 test events)
  Train events: M=3.5,4.0,4.3,4.8,5.0,5.0,5.3,6.0
  Val events: M=4.6,5.7 (UNSEEN in train)
  Test events: M=4.2,5.2 (UNSEEN in train)

Architecture: input=47, hidden=[256,128,64], dropout=0.15, Params=112,354
Training: patience=150, cosine LR, lambda_energy=0.05, GR c=-3.863

VERIFIED Results (outputs_newdata/metrics/metrics.json):
  MAE=0.454448, RMSE=0.596397, R2=-0.498263
  Epochs=151, Time=28.73s, Params=112,354
  Mean-baseline R2=-0.423867 (also negative - distribution shift problem)

---

## 4. Historical Results (Old Optimization Pipeline)

Source: archive/old_outputs/optimized/metrics/ and MODEL_OPTIMIZATION_REPORT.md
NOTE: Different preprocessing (feature-engineered ~27 features). Cannot
directly compare with current pipeline results.

| Model | MAE | RMSE | R2 | Source |
|---|---|---|---|---|
| Random Forest | 0.4604 | 0.5703 | 0.0902 | baseline_comparison.json |
| XGBoost | 0.4629 | 0.5774 | 0.0672 | baseline_comparison.json |
| LightGBM | 0.4632 | 0.5778 | 0.0661 | baseline_comparison.json |
| Optimized PINN | 0.4711 | 0.5784 | 0.0640 | TEXT ONLY - no JSON artifact |
| Linear Regression | 0.4938 | 0.6665 | -0.2427 | baseline_comparison.json |

Old 5-Fold CV (archive/old_outputs/optimized/metrics/cross_validation.json):
  R2=0.042 +/- 0.037, MAE=0.472 +/- 0.020
  Fold R2: [-0.029, 0.072, 0.049, 0.061, 0.059]

HPO (archive/old_outputs/optimized/metrics/hpo_results.json):
  20 Optuna trials. best_r2=-Infinity. HPO FAILED.

---

## 5. Verified Problems and Fixes

| Problem | Status | Evidence |
|---|---|---|
| Test=validation leakage | FIXED | 3-way split in preprocess.py |
| Physics backward() wiped data grads | FIXED | Single backward() in train.py |
| Physics not batch-aligned | FIXED | TensorDataset includes physics tensors |
| No target scaling | FIXED | StandardScaler on target train-only |
| Degenerate features | FIXED | Auto-removed by feature_audit.py |
| 8 redundant correlated features | FIXED | Auto-removed by feature_audit.py |
| No regularization | FIXED | AdamW, dropout, early stopping |
| Multi-station event leakage | FIXED (newdata) | Event-based split |
| GR intercept b=4.8 inapplicable | PARTIALLY FIXED | Correlation-based loss |
| Network [64,32,16] vs [256,128,64] | UNRESOLVED | config.yaml vs README mismatch |
| HPO did not converge | UNRESOLVED | best_r2=-Infinity |

---

## 6. Status Matrix

| Component | Status | Evidence | Remaining Work |
|---|---|---|---|
| Dataset analysis | Verified complete | FEATURE_ANALYSIS_REPORT.md | Stratified correlation analysis |
| Data preprocessing | Verified complete | preprocess.py, feature_audit.py | Stratified splitting |
| Baseline MLP | Verified complete | archive metrics JSON | None |
| Basic PINN | Verified complete | archive metrics JSON | None |
| Attention PINN | Verified complete | archive metrics JSON | None |
| Full PINN (uncertainty) | Implemented, artifacts present | archive metrics JSON | Calibration evaluation |
| Optimized PINN [64,32,16] | Experimentally evaluated | outputs/metrics/metrics.json | Fix arch mismatch |
| Optimized PINN [256,128,64] | Experimentally evaluated | outputs_newdata/metrics/ | Investigate R2=-0.498 |
| Physics-loss validation | Partially complete | energy_constraint.py | Unit consistency check |
| Ablation study | Verified complete | ablation_results.md | Re-run with correct arch |
| Cross-validation | Verified complete | MODEL_TRAINING_AUDIT.md | Re-run with correct arch |
| Research paper | In progress | -- | All sections |

---

## 7. Next Prioritized Tasks

1. Fix config.yaml: set hidden_layers=[256,128,64], re-run main pipeline
2. Clean baseline comparison: run RF/XGBoost/LightGBM on SAME 19-feature set
3. Add stratified splitting in preprocess.py
4. Evaluate uncertainty calibration (ECE, interval coverage)
5. Sensitivity analysis of lambda_energy (0.001 to 0.3)
