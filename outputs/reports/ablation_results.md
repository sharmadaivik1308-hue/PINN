# Ablation Study Results

| Experiment | Physics | Dropout | Weight Decay | Early Stop | R² | MAE | RMSE | Time (s) |
|---|---|---|---|---|---|---|---|---|
| A: Baseline | none | 0.0 | 0e+00 | No | -0.4383 | 0.5891 | 0.7324 | 81.4 |
| B: Regularized | none | 0.1 | 1e-04 | Yes | 0.0279 | 0.4868 | 0.6021 | 10.5 |
| C: Energy PINN | energy | 0.1 | 1e-04 | Yes | 0.0102 | 0.4905 | 0.6076 | 11.5 |
| D: Full PINN | full | 0.1 | 1e-04 | Yes | 0.0148 | 0.4912 | 0.6061 | 11.0 |
