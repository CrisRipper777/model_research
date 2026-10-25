# MR-CGE V1.3 Static-Strength Calibration Screen

Base SHA: `6d94c14d67e511aec34ff5fdcad3ce1a1cfd0701`.
Training source SHA: `7704c4edd43cddf4f7418a732bf5a6805e10d6d9`.
Physical GPU: `1`. Test evaluation: disabled. Link prediction: disabled.

## Validation comparison

Accuracy/F1 are shown as percentages; CE is the original cross-entropy value. Each row is paired by the same dataset and seed.

| Dataset | Seed | V1.2 Acc/F1/CE | Static Acc/F1/CE | Joint View Acc/F1/CE |
|---|---:|---:|---:|---:|
| Movies | 42 | 55.31 / 46.51 / 1.3718 | 55.13 / 45.55 / 1.4181 | 55.97 / 48.20 / 1.3704 |
| Movies | 43 | 54.77 / 47.67 / 1.4296 | 54.71 / 46.01 / 1.4109 | 55.22 / 49.17 / 1.3798 |
| Movies | 44 | 55.16 / 48.15 / 1.4344 | 54.95 / 45.47 / 1.4126 | 55.85 / 48.03 / 1.3542 |
| Grocery | 42 | 82.72 / 76.26 / 0.6916 | 82.75 / 75.07 / 0.6783 | 83.51 / 76.62 / 0.6392 |
| Grocery | 43 | 82.84 / 76.44 / 0.7041 | 82.99 / 75.63 / 0.6766 | 83.69 / 76.94 / 0.6456 |
| ele-fashion | 42 | 87.37 / 73.50 / 0.4130 | 87.43 / 74.28 / 0.4280 | 87.10 / 73.28 / 0.4011 |
| ele-fashion | 43 | 87.44 / 74.20 / 0.4130 | 87.41 / 74.57 / 0.4822 | 87.50 / 73.79 / 0.4039 |

## Paired differences

Accuracy and Macro-F1 differences are percentage points (pp); CE differences are raw CE units. Means and population SDs are across the listed seeds, with no significance test.

### Movies

| Contrast | Seeds | Δ Accuracy (pp), mean ± SD | Δ Macro-F1 (pp), mean ± SD | Δ CE, mean ± SD |
|---|---|---:|---:|---:|
| Static-Strength minus V1.2 | 42,43,44 | -0.15 ± 0.06 | -1.77 ± 0.70 | +0.0019 ± 0.0314 |
| Joint View minus V1.2 | 42,43,44 | +0.60 ± 0.11 | +1.02 ± 0.81 | -0.0438 ± 0.0324 |
| Joint View minus Static-Strength | 42,43,44 | +0.75 ± 0.17 | +2.79 ± 0.26 | -0.0457 ± 0.0112 |
### Grocery

| Contrast | Seeds | Δ Accuracy (pp), mean ± SD | Δ Macro-F1 (pp), mean ± SD | Δ CE, mean ± SD |
|---|---|---:|---:|---:|
| Static-Strength minus V1.2 | 42,43 | +0.09 ± 0.06 | -0.99 ± 0.19 | -0.0204 ± 0.0071 |
| Joint View minus V1.2 | 42,43 | +0.82 ± 0.03 | +0.43 ± 0.07 | -0.0554 ± 0.0031 |
| Joint View minus Static-Strength | 42,43 | +0.73 ± 0.03 | +1.42 ± 0.12 | -0.0350 ± 0.0040 |
### ele-fashion

| Contrast | Seeds | Δ Accuracy (pp), mean ± SD | Δ Macro-F1 (pp), mean ± SD | Δ CE, mean ± SD |
|---|---|---:|---:|---:|
| Static-Strength minus V1.2 | 42,43 | +0.02 ± 0.05 | +0.58 ± 0.20 | +0.0420 ± 0.0271 |
| Joint View minus V1.2 | 42,43 | -0.10 ± 0.16 | -0.31 ± 0.10 | -0.0106 ± 0.0014 |
| Joint View minus Static-Strength | 42,43 | -0.12 ± 0.21 | -0.89 ± 0.11 | -0.0526 ± 0.0257 |

Across both Movies and Grocery seeds, Joint View exceeds Static-Strength on Validation Accuracy and Macro-F1, with lower CE. On ele-fashion, Static is slightly higher on Accuracy and Macro-F1 in both seeds, while Joint View has lower CE in both. This is a descriptive small-sample pattern, not a significance claim.


## Static mechanism and cost

Strengths are evaluated at each Best-Accuracy checkpoint. `sP` and `sS` are global within modality; effective strengths multiply the learned V1.3-A `rho`/`gamma`. Early Cross residual ratios use the frozen model's calibrated Cross-to-physical-neighbor RMS diagnostic.

| Dataset | Seed | sP Text/Visual | sS Text/Visual | rho·sP Text/Visual | gamma·sS Text/Visual | Model params | Peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | 0.9997 / 1.0315 | 0.9942 / 1.0113 | 0.2498 / 0.2702 | 0.0787 / 0.0826 | 2021741 | 2.51 |
| Movies | 43 | 1.0023 / 1.0312 | 0.9918 / 1.0008 | 0.2515 / 0.2701 | 0.0782 / 0.0802 | 2021741 | 2.51 |
| Movies | 44 | 1.0021 / 1.0318 | 0.9899 / 1.0097 | 0.2513 / 0.2704 | 0.0777 / 0.0822 | 2021741 | 2.51 |
| Grocery | 42 | 1.0212 / 1.0287 | 1.0097 / 1.0238 | 0.2635 / 0.2685 | 0.0822 / 0.0856 | 2021741 | 2.46 |
| Grocery | 43 | 1.0236 / 1.0290 | 1.0131 / 1.0286 | 0.2651 / 0.2686 | 0.0830 / 0.0867 | 2021741 | 2.46 |
| ele-fashion | 42 | 1.0647 / 1.0692 | 1.0651 / 1.0352 | 0.2931 / 0.2963 | 0.0961 / 0.0884 | 1755485 | 12.54 |
| ele-fashion | 43 | 1.0843 / 1.0944 | 1.0827 / 1.0721 | 0.3072 / 0.3147 | 0.1010 / 0.0980 | 1755485 | 12.54 |

### Mean parameter and peak-memory cost

Means are across the authorized seeds for each dataset; peak memory is peak reserved CUDA memory.

| Dataset | Model | Total parameters | Peak reserved (GiB) |
|---|---|---:|---:|
| Movies | V1.2 M_repaired | 2026877 | 2.26 |
| Movies | V1.3-A Joint View | 2067935 | 2.67 |
| Movies | V1.3 Static-Strength | 2026881 | 2.51 |
| Grocery | V1.2 M_repaired | 2026877 | 2.22 |
| Grocery | V1.3-A Joint View | 2067935 | 2.59 |
| Grocery | V1.3 Static-Strength | 2026881 | 2.46 |
| ele-fashion | V1.2 M_repaired | 1758565 | 11.23 |
| ele-fashion | V1.3-A Joint View | 1799623 | 13.30 |
| ele-fashion | V1.3 Static-Strength | 1758569 | 12.54 |

The full per-run Best-F1 metrics, Train metrics/gaps, epochs, wall time, peak allocated/reserved memory and checkpoint hashes are in `data/model_comparison.csv` and `data/runs.csv`. The companion `strength_diagnostics.csv` includes weighted Private/Collaborative Physical RMS, weighted Semantic RMS, Early Cross residual/delta, Late residual ratio, and the frozen Joint View node-wise mean/SD/quantiles. Static strengths remain near 1 in this screen and do not approach the upper bound 2. The result does not identify the pure causal effect of node-wise reliability: Static adds four parameters on top of learned `rho/gamma`, so parameterization and optimization trajectories also differ.

## Fixed-split Hybrid-CGE context

This is performance context for Movies only and is excluded from Static-vs-Joint architectural attribution.

| Seed | Hybrid Acc | Macro-F1 | CE | Total parameters | Peak reserved (GiB) |
|---:|---:|---:|---:|---:|---:|
| 42 | 56.87% | 50.26% | 1.4031 | 2273270 | 4.29 |
| 43 | 57.05% | 49.43% | 1.3445 | 2273270 | 4.29 |
| 44 | 56.93% | 48.84% | 1.3362 | 2273270 | 4.29 |

## Status and interpretation

Formal training: 7/7 successful; failed training attempts: 0; resource-blocked attempts: 0; unstarted cells: 0.
CPU/data preflight recorded 2 failed software attempts in `data/failures.json`; both lookup/audit issues were corrected before the clean CPU and GPU preflights. No formal run launched until all seven GPU preflights passed.

Interpret the paired seed-level pattern across datasets. Do not select a winner from one seed, infer statistical significance, or attribute a Static-vs-Joint difference solely to node-wise reliability. This screen uses two seeds outside Movies and historical reference campaigns; optimization paths and small-sample variation remain limitations.

## Audit constraints

All data use frozen splits and V7B.1 K=8 semantic caches. Test labels were masked before Trainer entry; no Test metrics or link prediction were run. New training was isolated to physical GPU 1 in one-cell subprocesses with a fresh ownership guard and at least 8 GiB free at launch, plus a 2 GiB margin above the measured peak-reserved memory.
