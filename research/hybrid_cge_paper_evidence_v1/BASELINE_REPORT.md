# Same-protocol baseline comparison

All nine requested families are represented: six new training adaptations and 45 audited reused SAGE/MMGCN/RoleMAG rows. Every value is from the common Validation-only full-graph node-classification protocol. The project adaptations and their departures from the original method objectives are documented in `BASELINE_AUDIT.md`; no unrelated published benchmark score is mixed into this table.

## Validation results (mean ± sample SD over seeds)

| Dataset | Model | n | Val Accuracy | Val Macro-F1 | Val CE | Best-F1 Macro-F1 |
|---|---|---|---|---|---|---|
| Movies | gcn | 3/3 | 53.39% ± 0.35% | 44.62% ± 1.64% | 1.7269 ± 0.0011 | 46.95% |
| Movies | mgat | 3/3 | 53.65% ± 0.08% | 34.06% ± 2.19% | 1.4276 ± 0.0089 | 37.16% |
| Movies | dip | 3/3 | 56.57% ± 0.68% | 48.33% ± 1.03% | 1.3893 ± 0.0439 | 50.31% |
| Movies | dgf | 3/3 | 52.92% ± 1.03% | 30.49% ± 5.39% | 1.5058 ± 0.0465 | 33.22% |
| Movies | dmgc | 3/3 | 47.49% ± 1.48% | 23.27% ± 3.60% | 1.8308 ± 0.0673 | 25.50% |
| Movies | lgmrec | 3/3 | 56.52% ± 0.40% | 50.13% ± 0.55% | 1.3228 ± 0.0089 | 51.17% |
| Movies | sage | 3/3 | 54.76% ± 0.36% | 48.46% ± 0.97% | 2.2949 ± 0.6920 | 50.38% |
| Movies | mmgcn | 3/3 | 55.69% ± 0.81% | 46.90% ± 2.85% | 1.6308 ± 0.1430 | 49.16% |
| Movies | rolemag | 3/3 | 54.36% ± 0.36% | 42.27% ± 0.63% | 1.3960 ± 0.0254 | 44.74% |
| Grocery | gcn | 3/3 | 81.02% ± 0.52% | 72.05% ± 0.30% | 0.8606 ± 0.0627 | 72.81% |
| Grocery | mgat | 3/3 | 81.33% ± 0.60% | 69.06% ± 0.82% | 0.7047 ± 0.0086 | 69.56% |
| Grocery | dip | 3/3 | 83.95% ± 0.51% | 76.90% ± 1.20% | 0.6475 ± 0.0242 | 77.30% |
| Grocery | dgf | 3/3 | 81.29% ± 0.53% | 68.68% ± 0.17% | 0.7766 ± 0.0107 | 69.23% |
| Grocery | dmgc | 3/3 | 73.27% ± 3.11% | 61.67% ± 2.78% | 1.1264 ± 0.1539 | 61.99% |
| Grocery | lgmrec | 3/3 | 83.14% ± 0.31% | 76.00% ± 1.32% | 0.6467 ± 0.0102 | 76.29% |
| Grocery | sage | 3/3 | 83.32% ± 0.17% | 76.23% ± 0.76% | 1.0564 ± 0.1021 | 76.44% |
| Grocery | mmgcn | 3/3 | 81.80% ± 0.62% | 73.29% ± 0.91% | 0.8474 ± 0.0982 | 73.84% |
| Grocery | rolemag | 3/3 | 80.42% ± 0.59% | 70.28% ± 1.81% | 0.7434 ± 0.0235 | 70.58% |
| ele-fashion | gcn | 3/3 | 85.39% ± 0.13% | 72.35% ± 0.43% | 0.4909 ± 0.0014 | 73.31% |
| ele-fashion | mgat | 3/3 | 86.74% ± 0.09% | 72.06% ± 0.58% | 0.4206 ± 0.0003 | 72.77% |
| ele-fashion | dip | 3/3 | 88.03% ± 0.03% | 76.17% ± 0.27% | 0.4007 ± 0.0161 | 77.21% |
| ele-fashion | dgf | 3/3 | 86.86% ± 0.07% | 71.69% ± 0.60% | 0.4437 ± 0.0035 | 72.30% |
| ele-fashion | dmgc | 3/3 | 86.73% ± 0.16% | 71.30% ± 1.94% | 0.4493 ± 0.0022 | 72.62% |
| ele-fashion | lgmrec | 3/3 | 85.82% ± 0.22% | 70.40% ± 1.28% | 0.4554 ± 0.0133 | 70.73% |
| ele-fashion | sage | 3/3 | 87.68% ± 0.24% | 75.85% ± 2.00% | 0.4803 ± 0.1478 | 76.97% |
| ele-fashion | mmgcn | 3/3 | 87.60% ± 0.04% | 74.97% ± 0.22% | 0.4154 ± 0.0166 | 76.13% |
| ele-fashion | rolemag | 3/3 | 87.08% ± 0.15% | 73.73% ± 0.28% | 0.4122 ± 0.0037 | 74.84% |
| Toys | gcn | 3/3 | 79.33% ± 0.80% | 76.25% ± 0.37% | 0.7933 ± 0.0618 | 76.72% |
| Toys | mgat | 3/3 | 78.20% ± 0.11% | 74.32% ± 0.20% | 0.7644 ± 0.0133 | 74.49% |
| Toys | dip | 3/3 | 80.32% ± 0.43% | 77.29% ± 0.54% | 0.6941 ± 0.0290 | 78.23% |
| Toys | dgf | 3/3 | 79.25% ± 0.55% | 75.59% ± 0.67% | 0.8040 ± 0.0201 | 76.11% |
| Toys | dmgc | 3/3 | 72.03% ± 3.15% | 68.01% ± 3.98% | 1.0824 ± 0.1022 | 68.44% |
| Toys | lgmrec | 3/3 | 80.08% ± 0.18% | 77.24% ± 0.17% | 0.6817 ± 0.0124 | 77.39% |
| Toys | sage | 3/3 | 78.84% ± 0.85% | 76.08% ± 1.21% | 1.1966 ± 0.3539 | 76.29% |
| Toys | mmgcn | 3/3 | 78.86% ± 0.38% | 75.57% ± 0.48% | 0.8225 ± 0.1155 | 76.06% |
| Toys | rolemag | 3/3 | 78.84% ± 0.23% | 75.85% ± 0.14% | 0.7290 ± 0.0106 | 75.86% |
| Reddit-S | gcn | 3/3 | 94.23% ± 0.32% | 90.13% ± 0.50% | 0.2738 ± 0.0063 | 90.29% |
| Reddit-S | mgat | 3/3 | 95.94% ± 0.19% | 91.19% ± 0.81% | 0.1820 ± 0.0114 | 91.30% |
| Reddit-S | dip | 3/3 | 96.53% ± 0.34% | 92.36% ± 0.40% | 0.1420 ± 0.0028 | 92.63% |
| Reddit-S | dgf | 3/3 | 96.20% ± 0.32% | 90.73% ± 0.78% | 0.2723 ± 0.0348 | 91.03% |
| Reddit-S | dmgc | 3/3 | 93.68% ± 0.66% | 88.28% ± 0.32% | 0.2817 ± 0.0287 | 88.45% |
| Reddit-S | lgmrec | 3/3 | 96.10% ± 0.08% | 92.05% ± 0.53% | 0.1569 ± 0.0127 | 92.20% |
| Reddit-S | sage | 3/3 | 95.27% ± 0.33% | 91.24% ± 0.51% | 0.2369 ± 0.0154 | 91.30% |
| Reddit-S | mmgcn | 3/3 | 96.08% ± 0.38% | 92.18% ± 0.34% | 0.1720 ± 0.0030 | 92.28% |
| Reddit-S | rolemag | 3/3 | 95.71% ± 0.40% | 91.17% ± 0.27% | 0.1810 ± 0.0240 | 91.26% |

Best-Accuracy and Best-F1 checkpoints are separately recorded. See `data/checkpoint_selection_summary.csv` for per-run checkpoint differences and Train/Validation gaps. All new baseline rows have finite metrics and intact checkpoint hashes; Test evaluation is false.
