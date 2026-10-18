# ele-fashion finite optimization study

Fashion candidates were selected using Validation only. F0 is the historical 4-hop H2 reference; F1/F2/F4 have three new seeds; F3 aliases the 3-hop H2 cells and was not duplicated. Class-balanced weights are derived from Train counts only, normalized inverse-square-root, clipped to [0.5, 2.0], and applied only to main CE.

| Candidate | n | Val Accuracy | Val Macro-F1 | Val CE | Best-F1 Macro-F1 | Δ Acc vs F0 | Δ F1 vs F0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0_historical_4hop | 3/3 | 87.72% ± 0.06% | 74.65% ± 0.35% | 0.4113 ± 0.0035 | 76.16% | 0.00% | 0.00% |
| F1_fashion_dropout_aux | 3/3 | 87.62% ± 0.15% | 74.87% ± 0.99% | 0.4080 ± 0.0049 | 75.83% | -0.10% | 0.22% |
| F2_fashion_class_balanced_4hop | 3/3 | 87.30% ± 0.17% | 75.12% ± 0.74% | 0.4058 ± 0.0039 | 76.39% | -0.42% | 0.47% |
| F3_fashion_original_3hop | 3/3 | 87.80% ± 0.17% | 75.03% ± 0.91% | 0.4449 ± 0.0262 | 76.56% | 0.08% | 0.38% |
| F4_fashion_class_balanced_3hop | 3/3 | 87.37% ± 0.18% | 75.18% ± 0.89% | 0.4086 ± 0.0061 | 76.81% | -0.35% | 0.52% |

F1 jointly changes main dropout to 0.30 and private auxiliary CE to 0.1; their separate effects are not identifiable. Weight decay remains fixed at 1e-4 for every candidate, so this study estimates no weight-decay effect. F2 isolates class-balanced main CE at 4-hop, while F4 combines class balancing with 3-hop. F3 is the same corrected 3-hop model and data as the hop-depth matrix.

Per-seed checkpoint differences are in `data/fashion_paired_differences.csv`. Train/Validation class-wise metrics (Classes 1 and 11 plus low-frequency Train classes) are reported in this section and stored in `data/fashion_classwise_metrics.csv` and `data/fashion_classwise_summary.csv`. The earlier Fashion attempts with the disabled SameNode path remain preserved and superseded. All conclusions are dataset-specific and Validation-only; Test was not evaluated.

## Train/Validation class-wise diagnostics

Class-wise precision/recall/F1 is from Best-Accuracy checkpoints. Classes 1 and 11 are explicit; the three low-frequency classes are selected separately per seed from Train support counts, excluding Classes 1 and 11. Test labels are never indexed. Per-seed class IDs, counts and metrics are available in `data/fashion_classwise_metrics.csv`; aggregate mean and sample SD are in `data/fashion_classwise_summary.csv`.

| Candidate | Class | Class ID* | Val Recall | Val F1 | Train F1 | n |
|---|---|---:|---:|---:|---:|---:|
| H2_4hop | class_1 | 1 | 34.75% | 47.81% | 60.88% | 3/3 |
| H2_4hop | class_11 | 11 | 12.91% | 21.89% | 26.63% | 3/3 |
| H2_4hop | low_train_support_1 | 3 | 59.26% | 62.54% | 88.00% | 3/3 |
| H2_4hop | low_train_support_2 | 7 | 53.33% | 61.38% | 85.64% | 3/3 |
| H2_4hop | low_train_support_3 | 9 | 93.40% | 89.59% | 94.97% | 3/3 |
| F1 | class_1 | 1 | 34.06% | 46.60% | 60.82% | 3/3 |
| F1 | class_11 | 11 | 16.67% | 26.47% | 33.23% | 3/3 |
| F1 | low_train_support_1 | 3 | 57.04% | 62.74% | 86.78% | 3/3 |
| F1 | low_train_support_2 | 7 | 49.82% | 60.43% | 85.40% | 3/3 |
| F1 | low_train_support_3 | 9 | 94.10% | 89.74% | 95.00% | 3/3 |
| F2 | class_1 | 1 | 33.56% | 45.70% | 56.07% | 3/3 |
| F2 | class_11 | 11 | 14.64% | 23.36% | 29.83% | 3/3 |
| F2 | low_train_support_1 | 3 | 69.63% | 66.44% | 87.94% | 3/3 |
| F2 | low_train_support_2 | 7 | 61.75% | 63.14% | 88.78% | 3/3 |
| F2 | low_train_support_3 | 9 | 96.35% | 90.17% | 94.77% | 3/3 |
| H2_3hop | class_1 | 1 | 36.52% | 48.73% | 69.57% | 3/3 |
| H2_3hop | class_11 | 11 | 14.07% | 23.71% | 28.37% | 3/3 |
| H2_3hop | low_train_support_1 | 3 | 59.26% | 62.01% | 93.37% | 3/3 |
| H2_3hop | low_train_support_2 | 7 | 51.23% | 62.11% | 89.89% | 3/3 |
| H2_3hop | low_train_support_3 | 9 | 93.40% | 90.04% | 96.60% | 3/3 |
| F4 | class_1 | 1 | 35.48% | 47.08% | 60.27% | 3/3 |
| F4 | class_11 | 11 | 15.99% | 24.55% | 30.96% | 3/3 |
| F4 | low_train_support_1 | 3 | 65.93% | 63.02% | 90.62% | 3/3 |
| F4 | low_train_support_2 | 7 | 58.25% | 64.08% | 93.76% | 3/3 |
| F4 | low_train_support_3 | 9 | 96.18% | 89.81% | 95.07% | 3/3 |

*For low-frequency ranks, the selected class ID can differ by seed; inspect the per-seed CSV for the exact Train support rule and class identity.

## Class-weight support audit

Class weights are computed from Train labels only. This Fashion split has a class with zero Train support; the current opt-in implementation assigns zero to such an unobserved output class, while every observed class weight is normalized inverse-square-root frequency clipped to [0.5, 2.0]. Because the absent class never appears as a Train target, its zero entry is not selected by the Train CE loss. This is explicitly reported as the zero-support convention rather than claiming every vector entry is in [0.5, 2.0]. Exact seed-wise class supports and weights are in `data/fashion_train_class_support.csv` and `data/fashion_class_weight_audit.json`.
