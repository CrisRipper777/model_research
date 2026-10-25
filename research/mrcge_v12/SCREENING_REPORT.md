# MR-CGE V1.2 screening report

Status: **complete**. All 12 core cells and both eligible ele-fashion M bonus cells completed with seed 42. The raw records are in `data/`; every completed run has Best Accuracy and Best Macro-F1 checkpoints, checkpoint SHA checks, per-epoch history, and Validation RNG audit.

## Best-Accuracy and Best-F1 Validation results

Scores are percentages. `Macro-F1 @ Acc` is from the primary Best Validation Accuracy checkpoint; `Best-F1 Macro-F1` is from the secondary checkpoint.

| Dataset | Condition | Best Acc epoch | Val Accuracy | Macro-F1 @ Acc | Best F1 epoch | Best-F1 Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| Movies | S_repaired | 86 | 53.329 | 42.866 | 74 | 45.566 |
| Movies | F_sinit | 72 | 53.629 | 43.102 | 79 | 45.561 |
| Movies | M_repaired | 60 | 55.309 | 46.514 | 71 | 47.990 |
| Movies | M_anchor | 87 | 55.729 | 48.438 | 74 | 49.419 |
| Grocery | S_repaired | 67 | 80.117 | 72.824 | 78 | 73.308 |
| Grocery | F_sinit | 68 | 80.088 | 73.092 | 82 | 73.504 |
| Grocery | M_repaired | 82 | 82.723 | 76.259 | 85 | 76.385 |
| Grocery | M_anchor | 85 | 82.840 | 76.370 | 88 | 76.510 |
| Toys | S_repaired | 92 | 77.338 | 74.673 | 100 | 74.713 |
| Toys | F_sinit | 87 | 77.144 | 74.304 | 112 | 74.956 |
| Toys | M_repaired | 59 | 79.198 | 75.838 | 82 | 76.678 |
| Toys | M_anchor | 52 | 78.956 | 76.142 | 53 | 76.536 |
| ele-fashion | M_repaired | 142 | 87.368 | 73.498 | 125 | 75.042 |
| ele-fashion | M_anchor | 117 | 87.389 | 73.584 | 133 | 74.717 |

All 14 cells completed without a non-finite metric. Best Accuracy and Best F1 checkpoint hashes matched the saved files in all 14 rows. Validation RNG audits passed in all rows; Test evaluation remained false.

## Paired descriptive differences

The deltas below are percentage points, computed within each dataset and seed 42.

| Dataset | F_sinit − S_repaired Accuracy | F_sinit − S_repaired Macro-F1 @ Acc | M_anchor − M_repaired Accuracy | M_anchor − M_repaired Macro-F1 @ Acc |
|---|---:|---:|---:|---:|
| Movies | +0.300 | +0.236 | +0.420 | +1.924 |
| Grocery | -0.029 | +0.268 | +0.117 | +0.111 |
| Toys | -0.193 | -0.369 | -0.242 | +0.304 |
| ele-fashion | — | — | +0.020 | +0.086 |

F_sinit's change from S_repaired is small and mixed across the three datasets. M_anchor has higher Macro-F1 at the Best Accuracy checkpoint than M_repaired on all four datasets; Best Accuracy is higher on Movies, Grocery, and ele-fashion, while Toys favors M_repaired by 0.242 points. This makes M_anchor the leading descriptive candidate for a later three-seed confirmation, with M_repaired retained as its comparator. One seed does not establish superiority.

## Historical H2 and DiP context

These are cross-campaign reference values, not matched controls. H2_hybrid_plus is the three-seed V1.1 pilot; DiP values are three-seed means from the separate shared-protocol baseline report. Toys has no Toys row in the V1.1 H2_hybrid_plus pilot, so its H2 references come from the separate H2 physical-depth study. Sources: `../hybrid_cge_v11/PILOT_REPORT.md`, `../hybrid_cge_paper_evidence_v1/BASELINE_REPORT.md`, and `../hybrid_cge_paper_evidence_v1/data/condition_summary.csv`.

| Dataset | V1.2 M_anchor Accuracy / Macro-F1 | Historical H2 Accuracy / Macro-F1 | Historical DiP Accuracy / Macro-F1 |
|---|---:|---:|---:|
| Movies | 55.73 / 48.44 | H2_hybrid_plus: 56.78 / 49.78 | 56.57 / 48.33 |
| Grocery | 82.84 / 76.37 | H2_hybrid_plus: 83.86 / 75.97 | 83.95 / 76.90 |
| Toys | 78.96 / 76.14 | H2_physical_3hop: 80.27 / 77.05; H2 historical 4-hop: 80.33 / 77.47 | 80.32 / 77.29 |
| ele-fashion | 87.39 / 73.58 | H2_hybrid_plus: 87.72 / 74.65 | 88.03 / 76.17 |

M_anchor did not reach historical H2 or DiP Accuracy on any of the four datasets. Its Movies Macro-F1 is close to the DiP mean (+0.11 points) but below H2; it is below both references on Toys and ele-fashion. Grocery Macro-F1 exceeds the V1.1 H2_hybrid_plus mean but remains below DiP. These one-seed/cross-campaign comparisons are descriptive only; H2/DiP are not strict paired controls.

## Training stability, routing, and resource use

- The 14 successful cells contain 1,576 finite epoch-history rows. Twelve core cells used the frozen training protocol; two ele-fashion M cells used the same model and protocol after their preflight and fresh live guard passed.
- The original Movies/M_repaired attempt 1 was stopped after GPU 1 reported new external PID `4030361`. It produced no epoch or checkpoint. After the user authorized resumption, a fresh baseline passed and Movies/M_repaired attempt 2 completed. The failed attempt and earlier attempt-0 blocks remain in the raw audit trail and link to their successful replacements.
- The M route diagnostics include 48 views. Every expert had nonzero Top-2 selection in every recorded view; the smallest observed share was 8.09%. No dead expert was observed at the Best Accuracy checkpoints. These are checkpoint diagnostics, not causal evidence about expert specialization.
- Full-graph ele-fashion M training peaked at 8.51 GiB allocated and 11.23 GiB reserved for both conditions. The single-step preflight measured 8.49 / 8.95 GiB. The actual reserved peak exceeded the preflight reserved peak, but the live guard maintained at least 2 GiB free throughout. Compared with the preserved V1.1 Fashion M preflight (20.68 / 21.50 GiB allocated/reserved), V1.2 formal-training peaks were lower by about 58.8% allocated and 47.8% reserved.
- The first two completed Movies cells record the freeze commit `9cfe6e491d2ea2a58de01101af5032199c37cf14`. The other 12 completed rows record `b024ae8dbe94259611fadb0165df55a67db631b6`, the intervening data/report-only commit created during the guard stop. The model source hash is identical across all completed rows, and the launcher/model/config/trainer files are byte-identical to the freeze commit. No model, optimizer, or data-processing code changed after freeze.

## Boundary

No Test labels or metrics were accessed. Existing V1.1 files and results were not modified. No other GPU process was stopped, signalled, or modified. No seeds 43/44, Response Strength, or additional tuning was started.
