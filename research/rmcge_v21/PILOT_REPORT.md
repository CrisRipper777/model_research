# RM-CGE V2.1 Pilot and Paired Results

All reported scores are Validation metrics. The first group uses each run's Best Accuracy checkpoint; the second group uses the distinct Best Macro-F1 checkpoint. Entries are mean ± sample SD over seeds 42, 43 and 44.

## Best Accuracy checkpoint

### Validation Accuracy (%)

| Dataset | C0 | C1 | C2 |
|---|---:|---:|---:|
| Movies | 55.87 ± 0.40 | 56.08 ± 0.39 | 55.81 ± 0.24 |
| Grocery | 83.35 ± 0.38 | 83.60 ± 0.44 | 83.38 ± 0.45 |
| ele-fashion | 87.55 ± 0.25 | 87.43 ± 0.14 | 87.32 ± 0.06 |

### Validation Macro-F1 (%)

| Dataset | C0 | C1 | C2 |
|---|---:|---:|---:|
| Movies | 47.20 ± 1.21 | 47.84 ± 0.26 | 49.15 ± 0.41 |
| Grocery | 75.83 ± 1.63 | 76.45 ± 0.52 | 76.46 ± 1.24 |
| ele-fashion | 75.34 ± 0.44 | 73.54 ± 0.51 | 73.61 ± 0.46 |

### Validation CE

| Dataset | C0 | C1 | C2 |
|---|---:|---:|---:|
| Movies | 1.4172 ± 0.0459 | 1.4255 ± 0.0717 | 1.3979 ± 0.0485 |
| Grocery | 0.6721 ± 0.0560 | 0.6662 ± 0.0442 | 0.6601 ± 0.0127 |
| ele-fashion | 0.4363 ± 0.0329 | 0.4149 ± 0.0077 | 0.4129 ± 0.0111 |

## Best Macro-F1 checkpoint

### Validation Accuracy (%)

| Dataset | C0 | C1 | C2 |
|---|---:|---:|---:|
| Movies | 55.33 ± 0.67 | 54.64 ± 0.26 | 55.51 ± 0.54 |
| Grocery | 83.06 ± 0.42 | 82.93 ± 0.46 | 83.03 ± 0.41 |
| ele-fashion | 86.52 ± 0.20 | 86.83 ± 0.30 | 86.77 ± 0.27 |

### Validation Macro-F1 (%)

| Dataset | C0 | C1 | C2 |
|---|---:|---:|---:|
| Movies | 49.89 ± 0.10 | 50.20 ± 0.07 | 49.89 ± 0.41 |
| Grocery | 76.79 ± 1.11 | 76.76 ± 0.73 | 76.59 ± 1.21 |
| ele-fashion | 76.38 ± 0.17 | 75.61 ± 0.31 | 75.45 ± 0.19 |

### Validation CE

| Dataset | C0 | C1 | C2 |
|---|---:|---:|---:|
| Movies | 1.4225 ± 0.0341 | 1.4519 ± 0.0512 | 1.3928 ± 0.0447 |
| Grocery | 0.6864 ± 0.0682 | 0.6878 ± 0.0279 | 0.6704 ± 0.0296 |
| ele-fashion | 0.4509 ± 0.0314 | 0.4060 ± 0.0032 | 0.4092 ± 0.0002 |

## Epoch, generalization-gap and resource summary

| Dataset | Condition | Best Acc epoch | Best F1 epoch | Train−Val Acc gap (pp) | Train−Val F1 gap (pp) | Parameters / classifier | Mean wall time (s) | Peak allocated GPU (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | C0 | 90.7 ± 15.3 | 93.7 ± 9.3 | 19.09 ± 4.70 | 28.27 ± 6.09 | 1606310 / 5140 | 28.0 ± 3.5 | 2.64 |
| Movies | C1 | 95.7 ± 26.7 | 99.7 ± 14.7 | 20.86 ± 6.89 | 30.21 ± 8.67 | 1632267 / 5140 | 41.4 ± 8.5 | 2.76 |
| Movies | C2 | 81.7 ± 20.6 | 81.0 ± 13.5 | 18.40 ± 6.79 | 24.80 ± 9.79 | 1632267 / 5140 | 49.4 ± 8.9 | 2.87 |
| Grocery | C0 | 84.3 ± 19.3 | 91.7 ± 26.0 | 9.74 ± 2.75 | 14.08 ± 4.48 | 1606310 / 5140 | 25.3 ± 4.0 | 2.52 |
| Grocery | C1 | 86.3 ± 11.8 | 94.3 ± 5.1 | 10.65 ± 2.19 | 15.78 ± 3.23 | 1632267 / 5140 | 37.7 ± 3.5 | 2.64 |
| Grocery | C2 | 84.0 ± 16.5 | 89.0 ± 4.4 | 10.68 ± 1.69 | 15.59 ± 1.64 | 1632267 / 5140 | 48.6 ± 7.3 | 2.75 |
| ele-fashion | C0 | 176.0 ± 19.7 | 200.7 ± 21.4 | 4.33 ± 0.41 | 11.74 ± 1.23 | 1344166 / 3084 | 170.9 ± 16.6 | 10.86 |
| ele-fashion | C1 | 124.0 ± 13.1 | 135.0 ± 9.5 | 1.82 ± 0.45 | 6.73 ± 1.06 | 1370123 / 3084 | 198.0 ± 17.3 | 11.52 |
| ele-fashion | C2 | 118.3 ± 8.1 | 144.3 ± 6.5 | 1.88 ± 0.21 | 6.55 ± 0.66 | 1370123 / 3084 | 254.7 ± 13.6 | 12.12 |

Gap is Train minus Validation at the Best Accuracy checkpoint. Epoch and time columns show mean ± sample SD; peak allocated GPU memory is the maximum of the three paired runs. The model and classifier parameter counts are reported separately.

## Paired contrasts (left condition minus right condition)

| Dataset | Contrast | Metric | Mean paired Δ | Sample SD | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---:|---:|---:|---:|---:|
| Movies | C1-C0 | val_acc | +0.21 pp | +0.05 pp | +0.18 pp | +0.27 pp | +0.18 pp |
| Movies | C1-C0 | val_macro_f1 | +0.64 pp | +0.95 pp | +1.72 pp | +0.24 pp | -0.05 pp |
| Movies | C1-C0 | val_ce | +0.0083 | +0.0412 | -0.0110 | -0.0199 | +0.0556 |
| Movies | C2-C1 | val_acc | -0.27 pp | +0.36 pp | +0.15 pp | -0.51 pp | -0.45 pp |
| Movies | C2-C1 | val_macro_f1 | +1.31 pp | +0.56 pp | +1.66 pp | +1.61 pp | +0.67 pp |
| Movies | C2-C1 | val_ce | -0.0275 | +0.1176 | +0.0780 | -0.0063 | -0.1543 |
| Movies | C2-C0 | val_acc | -0.06 pp | +0.34 pp | +0.33 pp | -0.24 pp | -0.27 pp |
| Movies | C2-C0 | val_macro_f1 | +1.95 pp | +1.38 pp | +3.38 pp | +1.85 pp | +0.62 pp |
| Movies | C2-C0 | val_ce | -0.0193 | +0.0830 | +0.0670 | -0.0261 | -0.0987 |
| Grocery | C1-C0 | val_acc | +0.25 pp | +0.06 pp | +0.23 pp | +0.32 pp | +0.20 pp |
| Grocery | C1-C0 | val_macro_f1 | +0.62 pp | +1.19 pp | +0.14 pp | -0.25 pp | +1.98 pp |
| Grocery | C1-C0 | val_ce | -0.0059 | +0.0118 | -0.0032 | +0.0042 | -0.0188 |
| Grocery | C2-C1 | val_acc | -0.22 pp | +0.02 pp | -0.23 pp | -0.20 pp | -0.23 pp |
| Grocery | C2-C1 | val_macro_f1 | +0.01 pp | +0.80 pp | +0.31 pp | +0.62 pp | -0.90 pp |
| Grocery | C2-C1 | val_ce | -0.0060 | +0.0446 | +0.0160 | +0.0233 | -0.0574 |
| Grocery | C2-C0 | val_acc | +0.03 pp | +0.08 pp | +0.00 pp | +0.12 pp | -0.03 pp |
| Grocery | C2-C0 | val_macro_f1 | +0.63 pp | +0.39 pp | +0.45 pp | +0.36 pp | +1.08 pp |
| Grocery | C2-C0 | val_ce | -0.0120 | +0.0561 | +0.0128 | +0.0275 | -0.0762 |
| ele-fashion | C1-C0 | val_acc | -0.12 pp | +0.27 pp | -0.01 pp | -0.43 pp | +0.07 pp |
| ele-fashion | C1-C0 | val_macro_f1 | -1.80 pp | +0.91 pp | -1.96 pp | -2.61 pp | -0.82 pp |
| ele-fashion | C1-C0 | val_ce | -0.0213 | +0.0367 | -0.0043 | -0.0635 | +0.0038 |
| ele-fashion | C2-C1 | val_acc | -0.11 pp | +0.08 pp | -0.06 pp | -0.06 pp | -0.20 pp |
| ele-fashion | C2-C1 | val_macro_f1 | +0.07 pp | +0.27 pp | -0.14 pp | +0.38 pp | -0.04 pp |
| ele-fashion | C2-C1 | val_ce | -0.0020 | +0.0051 | -0.0079 | +0.0011 | +0.0008 |
| ele-fashion | C2-C0 | val_acc | -0.23 pp | +0.23 pp | -0.07 pp | -0.49 pp | -0.13 pp |
| ele-fashion | C2-C0 | val_macro_f1 | -1.73 pp | +0.76 pp | -2.09 pp | -2.24 pp | -0.86 pp |
| ele-fashion | C2-C0 | val_ce | -0.0233 | +0.0348 | -0.0122 | -0.0624 | +0.0046 |

Every seed-level paired difference is retained in `data/paired_comparisons.csv`. Differences are descriptive; three seeds do not support a significance claim. Full run fingerprints, epoch metrics, checkpoint paths and per-run resource records are in `data/`. Mean run wall time across the study was 94.9s; the largest allocated GPU peak was 12.12 GiB. No Test metrics were computed.
