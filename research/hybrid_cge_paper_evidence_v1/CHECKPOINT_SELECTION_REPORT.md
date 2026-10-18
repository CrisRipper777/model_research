# Validation checkpoint selection differences

The primary checkpoint is selected by Validation Accuracy. The secondary checkpoint is selected by Validation Macro-F1 using the same frozen stopping trajectory. For each run, `Best-F1 minus Best-Accuracy` is shown for Validation Accuracy, Macro-F1 and CE; Train/Validation gaps at the Best-Accuracy checkpoint are included to expose overfit differences. Exact rows and checkpoint digests are in `data/checkpoint_selection_summary.csv`.

| Model | Condition | n | Δ Val Acc (F1−Acc) | Δ Val Macro-F1 (F1−Acc) | Δ Val CE (F1−Acc) | Train−Val Acc gap | Train−Val F1 gap | Train−Val CE gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dgf | dgf | 15 | -0.36% ± 0.58% | 0.94% ± 1.07% | -0.0090 ± 0.0222 | 6.88% ± 5.01% | 8.45% ± 4.88% | -0.2095 ± 0.1433 |
| dip | dip | 15 | -0.47% ± 0.61% | 0.93% ± 0.91% | 0.0460 ± 0.0446 | 11.98% ± 9.92% | 17.38% ± 11.45% | -0.4327 ± 0.3054 |
| dmgc | dmgc | 15 | -0.52% ± 0.40% | 0.89% ± 0.87% | 0.0229 ± 0.0304 | 14.57% ± 8.46% | 15.74% ± 5.92% | -0.5411 ± 0.3166 |
| gcn | gcn | 15 | -0.36% ± 0.31% | 0.93% ± 0.86% | 0.0476 ± 0.0620 | 7.60% ± 5.34% | 9.91% ± 5.43% | -0.4311 ± 0.3004 |
| hybrid_cge_paper_v1 | A1_no_early_physical_cross | 15 | -0.63% ± 0.74% | 0.94% ± 0.78% | 0.0122 ± 0.0221 | 7.16% ± 5.82% | 10.89% ± 6.80% | -0.2788 ± 0.1962 |
| hybrid_cge_paper_v1 | A2_no_late_completion | 15 | -0.47% ± 0.58% | 0.69% ± 0.67% | 0.0048 ± 0.0289 | 7.59% ± 7.12% | 11.54% ± 8.49% | -0.2991 ± 0.2490 |
| hybrid_cge_paper_v1 | A3_no_moe | 15 | -0.52% ± 0.54% | 0.74% ± 0.64% | 0.0172 ± 0.0158 | 7.23% ± 5.54% | 11.04% ± 6.68% | -0.2872 ± 0.1960 |
| hybrid_cge_paper_v1 | A4_no_same_node | 15 | -0.41% ± 0.39% | 0.62% ± 0.67% | 0.0161 ± 0.0255 | 7.60% ± 4.81% | 11.54% ± 5.50% | -0.3021 ± 0.1686 |
| hybrid_cge_paper_v1 | F1_fashion_dropout_aux | 3 | -1.19% ± 1.05% | 0.96% ± 0.60% | 0.0235 ± 0.0205 | 3.07% ± 0.94% | 9.32% ± 2.15% | -0.1480 ± 0.0394 |
| hybrid_cge_paper_v1 | F2_fashion_class_balanced_4hop | 3 | -0.65% ± 0.71% | 1.26% ± 0.33% | 0.0204 ± 0.0187 | 2.51% ± 1.15% | 8.41% ± 2.62% | -0.1145 ± 0.0477 |
| hybrid_cge_paper_v1 | F4_fashion_class_balanced_3hop | 3 | -1.04% ± 0.44% | 1.63% ± 0.85% | 0.0205 ± 0.0067 | 3.12% ± 0.87% | 10.05% ± 2.41% | -0.1434 ± 0.0385 |
| hybrid_cge_paper_v1 | H2_physical_3hop | 15 | -0.56% ± 0.60% | 0.68% ± 0.68% | 0.0071 ± 0.0250 | 7.51% ± 5.20% | 11.62% ± 6.25% | -0.3095 ± 0.1818 |
| lgmrec | lgmrec | 15 | -0.17% ± 0.32% | 0.39% ± 0.48% | 0.0031 ± 0.0073 | 5.38% ± 3.88% | 7.83% ± 4.73% | -0.2063 ± 0.1300 |
| mgat | mgat | 15 | -0.27% ± 0.19% | 0.92% ± 1.28% | -0.0011 ± 0.0099 | 8.56% ± 5.85% | 9.90% ± 4.29% | -0.2705 ± 0.1650 |
| mmgcn | mmgcn | 15 | -0.35% ± 0.34% | 0.91% ± 1.16% | 0.0712 ± 0.1033 | 14.66% ± 12.42% | 18.30% ± 12.04% | -0.5977 ± 0.4683 |
| rolemag | rolemag | 15 | -0.32% ± 0.28% | 0.80% ± 1.05% | 0.0096 ± 0.0119 | 7.03% ± 4.00% | 9.49% ± 4.11% | -0.2761 ± 0.1417 |
| sage | sage | 15 | -0.45% ± 0.59% | 0.70% ± 0.98% | 0.0788 ± 0.2116 | 17.59% ± 14.44% | 22.08% ± 14.66% | -0.9677 ± 0.8249 |

A positive Δ Macro-F1 means choosing the F1 checkpoint improves Macro-F1 but may trade away Accuracy; CE is recalculated unweighted on Train/Validation. Historical/reused rows are identified in the source CSV and should not be treated as fresh training cells. All values exclude Test.
