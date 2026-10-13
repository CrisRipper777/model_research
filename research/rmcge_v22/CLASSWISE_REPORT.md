# Classwise and degree diagnostics

Metrics are Validation-only and use the named checkpoint. No subgroup result was used to tune this frozen study.

## ele-fashion: Classes 11 and 1

| Checkpoint | Arm | Class | Recall mean ± sample SD (%) | Predicted count mean ± SD | Support mean |
|---|---|---:|---:|---:|---:|
| best_accuracy | M0 | 11 | 15.99 ± 2.95 | 71.3 ± 16.9 | 346.0 |
| best_accuracy | M0 | 1 | 31.99 ± 1.50 | 356.0 ± 31.5 | 868.0 |
| best_accuracy | M1 | 11 | 15.51 ± 2.41 | 83.3 ± 15.3 | 346.0 |
| best_accuracy | M1 | 1 | 31.03 ± 2.78 | 349.3 ± 51.2 | 868.0 |
| best_accuracy | M2 | 11 | 11.95 ± 1.17 | 52.0 ± 6.1 | 346.0 |
| best_accuracy | M2 | 1 | 33.26 ± 3.48 | 390.7 ± 65.4 | 868.0 |
| best_accuracy | M3 | 11 | 14.84 ± 3.47 | 70.7 ± 13.7 | 346.0 |
| best_accuracy | M3 | 1 | 34.68 ± 2.57 | 413.3 ± 49.0 | 868.0 |
| best_accuracy | M4 | 11 | 15.80 ± 0.73 | 92.3 ± 14.2 | 346.0 |
| best_accuracy | M4 | 1 | 32.99 ± 1.21 | 384.0 ± 14.4 | 868.0 |
| best_f1 | M0 | 11 | 35.36 ± 1.59 | 268.0 ± 22.6 | 346.0 |
| best_f1 | M0 | 1 | 42.93 ± 1.57 | 615.0 ± 42.8 | 868.0 |
| best_f1 | M1 | 11 | 27.46 ± 3.33 | 189.7 ± 31.1 | 346.0 |
| best_f1 | M1 | 1 | 36.64 ± 0.72 | 475.0 ± 22.3 | 868.0 |
| best_f1 | M2 | 11 | 28.81 ± 8.30 | 216.0 ± 90.5 | 346.0 |
| best_f1 | M2 | 1 | 41.82 ± 3.27 | 617.0 ± 99.5 | 868.0 |
| best_f1 | M3 | 11 | 33.62 ± 11.24 | 263.0 ± 112.3 | 346.0 |
| best_f1 | M3 | 1 | 39.90 ± 1.73 | 556.0 ± 63.3 | 868.0 |
| best_f1 | M4 | 11 | 34.10 ± 7.09 | 290.0 ± 79.2 | 346.0 |
| best_f1 | M4 | 1 | 36.02 ± 1.92 | 446.7 ± 38.0 | 868.0 |

## Degree groups

`degree_group_metrics.csv` reports Best Accuracy and Best F1 Validation Accuracy/Macro-F1/CE for degree 0, 1, 2–3, 4–7, and ≥8, along with full-graph normalized content RMS for each modality. Degree=1 can receive a nonzero content residual; any class prediction change there is still a trained-checkpoint comparison and may reflect shared-parameter changes.

## Correctness transitions

`prediction_transitions.csv` reports per-seed correct→wrong, wrong→correct, unchanged-correct and unchanged-wrong counts relative to M0 for each class and overall at Best Accuracy. These are paired Validation predictions, not Test performance.
