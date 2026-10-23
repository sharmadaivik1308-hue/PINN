# Ablation Study Results

| Rank | Experiment | Physics | Val R² (mean±std) | Val MAE | Val RMSE | Test R² (mean±std) | Test MAE | Test RMSE | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | C: Energy PINN | energy | -4.4152 ± 0.1927 | 0.3867 | 0.4228 | -3.3947 ± 0.9623 | 0.1727 | 0.2082 | 60.3 |
| 2 | D: Full PINN | full | -4.4152 ± 0.1927 | 0.3867 | 0.4228 | -3.3947 ± 0.9623 | 0.1727 | 0.2082 | 77.9 |
| 3 | B: Regularized | none | -4.8898 ± 0.4993 | 0.4020 | 0.4406 | -4.1241 ± 0.7334 | 0.1781 | 0.2258 | 59.5 |
| 4 | A: Baseline | none | -14.0836 ± 1.4587 | 0.5806 | 0.7050 | -28.5290 ± 2.7284 | 0.4306 | 0.5428 | 349.6 |
