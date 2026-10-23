# MODEL DEVELOPMENT HISTORY
## Chronological Progression: Baseline MLP to Optimized PINN

---

## Stage 1: Baseline MLP

Files: archive/baseline_mlp/main.py, model.py, train.py
Architecture: Input(29) -> Linear(128)+ReLU -> Linear(64)+ReLU -> Linear(32)+ReLU -> Linear(1)
Parameters: ~12,961 | Loss: MSE | Optimizer: Adam lr=0.001
Training: Full-batch, 300 epochs, 80/20 split, no validation set, no regularization
Scaling: StandardScaler on features only (no target scaling)

Verified Results (archive/old_outputs/metrics/baseline_mlp_metrics.json):
  MAE=0.5014, RMSE=0.6974, R2=-0.3607, Time=3.35s

Critical defects:
1. Test set used as validation during training (leakage)
2. Degenerate features (dominant_freq, predominant_period) passed to model
3. No regularization or early stopping
4. Full-batch training (no mini-batches)

---

## Stage 2: Basic PINN

Files: archive/basic_pinn/pinn_model.py, pinn_train.py
Architecture: Same backbone (29->128->64->32->1) plus physics loss
New modules: physics/energy_constraint.py, physics/frequency_constraint.py
Physics loss: L_total = L_MSE + 0.3*L_energy + 0.3*L_frequency
  (weights from DOCX proposal: Physics inspired Neural Network.docx)

NEW DEFECT INTRODUCED:
  Two separate backward() calls (energy loss backward after data loss backward,
  with optimizer.zero_grad() in between). Physics backward() wiped data gradients.

Inherited defects: leakage, degenerate features, no regularization

Verified Results (archive/old_outputs/metrics/basic_pinn_metrics.json):
  MAE=0.5114, RMSE=0.6705, R2=-0.2579, Time=1.4s, Params=14,209

Stage 1->2 R2 improvement: -0.361 -> -0.258 (+28.5%)
Interpretation: Even with the gradient-wipe defect, physics helps as regularizer.

---

## Stage 3: Attention PINN

Files: archive/attention_pinn/main_pinn.py (monolithic)
Architecture: Sigmoid attention gate before backbone
  x_attn = x * sigmoid(Linear(29,29)(x))
Parameters: ~15,079

Attention type: Single-layer sigmoid gate (NOT the improved ScaledAttention
  with learnable temperature in current src/pinn_model.py)

Defects: Same as Stage 2 (still two backward passes, full-batch, leakage)

Verified Results (archive/old_outputs/metrics/all_experiments_comparison.json):
  MAE=0.5036, RMSE=0.6388, R2=-0.1416, Time=2.03s, Params=15,079

Stage 2->3 R2 improvement: -0.258 -> -0.142 (+45.2%)
Interpretation: Feature weighting helps model focus on relevant physics features.

---

## Stage 4: Full PINN with Uncertainty Estimation

Files: archive/full_pinn/main_pinn.py (monolithic)
Architecture: Attention gate + backbone + dual output head [mu, log_sigma]
  sigma = exp(clamp(log_sigma, -2, 2))  [clamped for stability]
Data loss: L_NLL + 0.1*MSE_anchor
  L_NLL = mean((y-mu)^2/(2*sigma^2) + log_sigma)

Parameters: ~15,112
Defects: Same as Stage 3 (still two backward passes, full-batch, leakage)
Calibration: NOT evaluated. No ECE or coverage metrics computed.

Verified Results (archive/old_outputs/metrics/all_experiments_comparison.json):
  MAE=0.4971, RMSE=0.6160, R2=-0.0617, Time=2.0s, Params=15,112

Stage 3->4 R2 improvement: -0.142 -> -0.062 (+56.3%)
Stage 1->4 overall R2 improvement: -0.361 -> -0.062 (82.8%)
Stage 1->4 RMSE improvement: 0.697 -> 0.616 (11.6%)

---

## Stage 5: Major Pipeline Overhaul (OptimizedPINN)

This stage reorganized the entire codebase to a modular package structure and
corrected all known critical defects from Stages 1-4.

### Architecture Changes (src/pinn_model.py)

Old EarthquakePINN (Stage 4) replaced by OptimizedPINN with:
1. ResidualBlock: GELU + LayerNorm + skip connections (vs ReLU, no residuals)
2. ScaledAttention with learnable temperature parameter (vs simple sigmoid gate)
   s = Linear(d->d)(Tanh(Linear(d->d)(x)))
   w = sigmoid(s / (tau + epsilon))  where tau is nn.Parameter
   x_tilde = x * w
3. Xavier weight initialization
4. Fully configurable via model_config dict

### Training Pipeline Fixes (src/train.py)

FIX 1 - Joint backward pass:
  OLD: zero_grad() -> data.backward() -> zero_grad() -> phys.backward()
  NEW: total_loss = data_loss + phys_loss -> single total_loss.backward()

FIX 2 - Batch-aligned physics tensors:
  OLD: physics computed on full dataset separately from mini-batch loop
  NEW: TensorDataset(X, y, raw_energy, raw_frequency) ensures alignment

FIX 3 - Per-component loss logging:
  Energy loss and frequency loss tracked separately every epoch

FIX 4 - Device management (CUDA if available)
FIX 5 - Per-epoch timing
FIX 6 - Gradient clipping (max_norm=1.0)
FIX 7 - AdamW optimizer with weight decay
FIX 8 - ReduceLROnPlateau / CosineAnnealingLR scheduler
FIX 9 - Early stopping with validation restore

### Preprocessing Fixes (src/preprocess.py, src/feature_audit.py)

- 3-way split (70/15/15) replaces 80/20
- Target StandardScaler (train-only) added
- Automatic degenerate feature removal (variance_threshold=1e-6)
- Automatic correlation removal (threshold=0.95, physics-motivated priority)
- Physics features extracted BEFORE scaling (raw tensors alongside normalized)
- 5-fold CV with fresh scalers per fold (no CV leakage)

---

## Sub-Iteration 5a: config.yaml Run (Current Default)

IMPORTANT ARCHITECTURE DISCREPANCY:
  README.md says: hidden=[256,128,64]
  config.yaml says: hidden=[64,32,16]
  outputs/metrics/config_snapshot.yaml CONFIRMS: hidden=[64,32,16] was used
  reports/r2_analysis.md identifies this as root cause of low R2

Parameters actually trained: 8,778 (outputs/metrics/metrics.json)

Verified Results (outputs/metrics/metrics.json):
  MAE=0.4940, RMSE=0.6062, R2=0.0147, Epochs=147, Time=436.49s

First run to achieve positive R2 (albeit marginally positive).
Improvement from Stage 4: R2 moves from -0.062 to +0.015

Ablation (outputs/reports/ablation_results.md - VERIFIED):
  A - No reg, no physics: R2=-0.438
  B - Regularization only: R2=0.028 (BEST)
  C - Energy PINN: R2=0.010
  D - Full PINN: R2=0.015

Finding: Regularization dominates; physics at lambda=0.1 slightly hurts R2.

5-Fold CV (outputs/reports/MODEL_TRAINING_AUDIT.md - VERIFIED):
  R2=0.037 +/- 0.021 (stable positive)

---

## Sub-Iteration 5b: newdata_config.yaml Run (Event-Based Split)

Motivation: Random split allows correlated multi-station recordings of the same
  earthquake event in both training and test sets. Event-based split prevents this.

Dataset reorganization (newdata(claude)/preprocessing_report.json):
  12 events -> 8 train (1,653 samples) / 2 val (550) / 2 test (294)
  All samples from the same event are in the same split

Features: 47 (original + log1p transforms, constants removed upstream)
Architecture: hidden=[256,128,64], 112,354 parameters (CORRECT intended size)
Training: patience=150, lambda_energy=0.05, GR intercept c=-3.863 (fitted)

Verified Results (outputs_newdata/metrics/metrics.json):
  MAE=0.4544, RMSE=0.5964, R2=-0.4983

NOTE: R2=-0.498 does NOT indicate the model is worse than Stage 5a.
  The test set contains magnitudes ABSENT from training (M=4.2, 5.2 largely absent)
  making this an extrapolation scenario. The mean baseline R2=-0.424 also negative.
  This reveals that the event-based split creates a harder, more realistic problem.

---

## Historical Optimization Run (Archived)

Run via: archive/old_scripts/run_optimization.py
Features: ~27 (feature-engineered dataset, different from current 19-feature pipeline)

Best PINN performance reported in MODEL_OPTIMIZATION_REPORT.md:
  MAE=0.4711, RMSE=0.5784, R2=0.0640
  No JSON artifact exists for this result.

ML baselines (archive/old_outputs/optimized/metrics/baseline_comparison.json):
  Random Forest: R2=0.090, XGBoost: R2=0.067, LightGBM: R2=0.066
  NOTE: Features differ from current pipeline; cannot directly compare.

HPO: 20 Optuna trials, best_r2=-Infinity (FAILED).

---

## Complete Results Summary

| Stage | Model | MAE | RMSE | R2 | Params | Pipeline |
|---|---|---|---|---|---|---|
| 1 | Baseline MLP | 0.5014 | 0.6974 | -0.3607 | 12,961 | Old (29 feat, 80/20) |
| 2 | Basic PINN | 0.5114 | 0.6705 | -0.2579 | 14,209 | Old (29 feat, 80/20) |
| 3 | Attention PINN | 0.5036 | 0.6388 | -0.1416 | 15,079 | Old (29 feat, 80/20) |
| 4 | Full PINN | 0.4971 | 0.6160 | -0.0617 | 15,112 | Old (29 feat, 80/20) |
| 5a | Optimized [64,32,16] | 0.4940 | 0.6062 | +0.0147 | 8,778 | Current (19 feat, 70/15/15) |
| 5b | Optimized [256,128,64] | 0.4544 | 0.5964 | -0.4983* | 112,354 | Newdata (47 feat, event) |
| Old | Optimized (old eng.) | 0.4711 | 0.5784 | +0.064** | ~99K | Old opt. (~27 feat) |

* Distribution shift (unseen magnitudes in test), not comparable to other runs
** Text only, no JSON artifact
Stages 1-4: verified from JSON. 5a: verified from JSON. 5b: verified from JSON.
Old optimization: partially verified (ML baselines from JSON; PINN from text).
