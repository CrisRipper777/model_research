# ele-fashion finite optimization study

Fashion candidates were selected using Validation only. F0 is the historical 4-hop H2 reference; F1/F2/F4 have three new seeds; F3 aliases the 3-hop H2 cells and was not duplicated. Class-balanced weights are derived from Train counts only, normalized inverse-square-root, clipped to [0.5, 2.0], and applied only to main CE.

| Candidate | n | Val Accuracy | Val Macro-F1 | Val CE | Best-F1 Macro-F1 | Δ Acc vs F0 | Δ F1 vs F0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F0_historical_4hop | 3/3 | 87.72% ± 0.06% | 74.65% ± 0.35% | 0.4113 ± 0.0035 | 76.16% | 0.00% | 0.00% |
| F1_fashion_dropout_aux | 3/3 | 87.71% ± 0.12% | 74.95% ± 0.65% | 0.4156 ± 0.0192 | 76.22% | -0.01% | 0.30% |
| F2_fashion_class_balanced_4hop | 3/3 | 87.32% ± 0.10% | 75.16% ± 1.17% | 0.4177 ± 0.0238 | 76.68% | -0.40% | 0.51% |
| F3_fashion_original_3hop | 3/3 | 87.55% ± 0.14% | 75.22% ± 0.71% | 0.4123 ± 0.0329 | 76.39% | -0.17% | 0.57% |
| F4_fashion_class_balanced_3hop | 3/3 | 87.37% ± 0.11% | 75.28% ± 0.47% | 0.4067 ± 0.0087 | 76.95% | -0.35% | 0.63% |

F1 jointly changes main dropout to 0.30 and private auxiliary CE to 0.1; their separate effects are not identifiable. Weight decay remains fixed at 1e-4 for every candidate, so this study estimates no weight-decay effect. F2 isolates class-balanced main CE at 4-hop, while F4 combines class balancing with 3-hop. F3 is the same 3-hop model and data as the hop-depth matrix.

Per-seed checkpoint differences are in `data/fashion_paired_differences.csv`. Train/Validation class-wise metrics (Classes 1 and 11 plus low-frequency Train classes) will be appended from the post-training diagnostic artifact when available. All conclusions are dataset-specific and Validation-only; Test was not evaluated.

## Train/Validation class-wise diagnostics

Class-wise precision/recall/F1 is from Best-Accuracy checkpoints. Classes 1 and 11 are explicit; the three low-frequency classes are selected separately per seed from Train support counts, excluding Classes 1 and 11. Test labels are never indexed. Per-seed class IDs, counts and metrics are in `data/fashion_classwise_metrics.csv`; aggregate mean and sample SD are in `data/fashion_classwise_summary.csv`. These Fashion tables contain ele-fashion rows only.

| Candidate | Split | Class | Class ID* | Recall | F1 | Train support | n |
|---|---|---|---:|---:|---:|---:|---:|
| H2_4hop | train | class_1 | 1 | 45.19% | 60.92% | 5369.0 | 3/3 |
| H2_4hop | train | class_11 | 11 | 16.02% | 26.51% | 2118.0 | 3/3 |
| H2_4hop | train | low_train_support_1 | 3 | 87.88% | 88.00% | 308.0 | 3/3 |
| H2_4hop | train | low_train_support_2 | 7 | 81.15% | 85.79% | 520.0 | 3/3 |
| H2_4hop | train | low_train_support_3 | 9 | 96.51% | 94.83% | 1346.0 | 3/3 |
| H2_4hop | validation | class_1 | 1 | 34.72% | 47.74% | 5369.0 | 3/3 |
| H2_4hop | validation | class_11 | 11 | 13.01% | 22.01% | 2118.0 | 3/3 |
| H2_4hop | validation | low_train_support_1 | 3 | 59.26% | 62.54% | 308.0 | 3/3 |
| H2_4hop | validation | low_train_support_2 | 7 | 53.33% | 61.25% | 520.0 | 3/3 |
| H2_4hop | validation | low_train_support_3 | 9 | 93.75% | 89.70% | 1346.0 | 3/3 |
| F1 | train | class_1 | 1 | 47.29% | 62.78% | 5369.0 | 3/3 |
| F1 | train | class_11 | 11 | 16.73% | 28.24% | 2118.0 | 3/3 |
| F1 | train | low_train_support_1 | 3 | 86.90% | 86.97% | 308.0 | 3/3 |
| F1 | train | low_train_support_2 | 7 | 80.00% | 86.38% | 520.0 | 3/3 |
| F1 | train | low_train_support_3 | 9 | 96.43% | 95.27% | 1346.0 | 3/3 |
| F1 | validation | class_1 | 1 | 34.41% | 47.10% | 5369.0 | 3/3 |
| F1 | validation | class_11 | 11 | 14.16% | 23.98% | 2118.0 | 3/3 |
| F1 | validation | low_train_support_1 | 3 | 64.44% | 65.71% | 308.0 | 3/3 |
| F1 | validation | low_train_support_2 | 7 | 50.88% | 60.47% | 520.0 | 3/3 |
| F1 | validation | low_train_support_3 | 9 | 92.88% | 89.39% | 1346.0 | 3/3 |
| F2 | train | class_1 | 1 | 46.79% | 61.12% | 5369.0 | 3/3 |
| F2 | train | class_11 | 11 | 21.59% | 31.88% | 2118.0 | 3/3 |
| F2 | train | low_train_support_1 | 3 | 96.21% | 92.60% | 308.0 | 3/3 |
| F2 | train | low_train_support_2 | 7 | 94.49% | 93.94% | 520.0 | 3/3 |
| F2 | train | low_train_support_3 | 9 | 98.98% | 95.39% | 1346.0 | 3/3 |
| F2 | validation | class_1 | 1 | 35.48% | 46.93% | 5369.0 | 3/3 |
| F2 | validation | class_11 | 11 | 16.96% | 25.66% | 2118.0 | 3/3 |
| F2 | validation | low_train_support_1 | 3 | 62.96% | 64.64% | 308.0 | 3/3 |
| F2 | validation | low_train_support_2 | 7 | 54.39% | 61.47% | 520.0 | 3/3 |
| F2 | validation | low_train_support_3 | 9 | 96.35% | 90.03% | 1346.0 | 3/3 |
| H2_3hop | train | class_1 | 1 | 49.93% | 63.35% | 5369.0 | 3/3 |
| H2_3hop | train | class_11 | 11 | 19.69% | 30.91% | 2118.0 | 3/3 |
| H2_3hop | train | low_train_support_1 | 3 | 83.66% | 83.89% | 308.0 | 3/3 |
| H2_3hop | train | low_train_support_2 | 7 | 78.97% | 83.73% | 520.0 | 3/3 |
| H2_3hop | train | low_train_support_3 | 9 | 97.45% | 94.53% | 1346.0 | 3/3 |
| H2_3hop | validation | class_1 | 1 | 35.98% | 47.56% | 5369.0 | 3/3 |
| H2_3hop | validation | class_11 | 11 | 15.80% | 25.20% | 2118.0 | 3/3 |
| H2_3hop | validation | low_train_support_1 | 3 | 60.00% | 62.62% | 308.0 | 3/3 |
| H2_3hop | validation | low_train_support_2 | 7 | 57.19% | 63.33% | 520.0 | 3/3 |
| H2_3hop | validation | low_train_support_3 | 9 | 94.62% | 89.71% | 1346.0 | 3/3 |
| F4 | train | class_1 | 1 | 42.96% | 58.01% | 5369.0 | 3/3 |
| F4 | train | class_11 | 11 | 26.77% | 38.41% | 2118.0 | 3/3 |
| F4 | train | low_train_support_1 | 3 | 97.08% | 91.33% | 308.0 | 3/3 |
| F4 | train | low_train_support_2 | 7 | 93.65% | 92.11% | 520.0 | 3/3 |
| F4 | train | low_train_support_3 | 9 | 98.71% | 94.60% | 1346.0 | 3/3 |
| F4 | validation | class_1 | 1 | 34.45% | 46.72% | 5369.0 | 3/3 |
| F4 | validation | class_11 | 11 | 19.85% | 29.28% | 2118.0 | 3/3 |
| F4 | validation | low_train_support_1 | 3 | 60.74% | 61.35% | 308.0 | 3/3 |
| F4 | validation | low_train_support_2 | 7 | 56.84% | 63.95% | 520.0 | 3/3 |
| F4 | validation | low_train_support_3 | 9 | 95.83% | 89.47% | 1346.0 | 3/3 |

*For low-frequency ranks, the selected class ID can differ by seed; exact identities and supports are listed in the per-seed CSV.


## Class-weight support audit

Class weights are computed from Train labels only. This Fashion split has a class with zero Train support; the current opt-in implementation assigns zero to such an unobserved output class, while every observed class weight is normalized inverse-square-root frequency clipped to [0.5, 2.0]. Because the absent class never appears as a Train target, its zero entry is not selected by the Train CE loss. This is explicitly reported as the zero-support convention rather than claiming every vector entry is in [0.5, 2.0]. Exact seed-wise class supports and weights are in `data/fashion_train_class_support.csv` and `data/fashion_class_weight_audit.json`.
