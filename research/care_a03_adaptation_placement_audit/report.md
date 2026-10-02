# CARE-MAG A0.3 — Adaptation Placement Audit

- Base SHA: `a44a556517112f15a8550f81c4b916906756d77f`
- Final campaign status: `complete`
- Design: 2×3 factorial; full-graph NC; best checkpoints selected on validation Accuracy.
- Test evaluation: disabled; analyzer verified validation-only metrics and rejected any `test_*` keys.
- Reported seed summaries are paired descriptive results (mean ± population SD); no p-values.

## Validation performance

Accuracy and Macro-F1 are shown as percentages; CE remains on its original scale.

| Dataset | Cell | Val Accuracy (%) | Macro-F1 (%) | CE | Best epoch |
|---|---|---:|---:|---:|---:|
| Movies | G0 (global/off) | 55.56 ± 0.13 | 47.42 ± 0.72 | 1.4473 ± 0.0209 | 100.3 ± 6.6 |
| Movies | GS (global/static) | 55.61 ± 0.24 | 46.85 ± 0.69 | 1.4492 ± 0.0465 | 97.7 ± 16.1 |
| Movies | GC (global/context) | 55.40 ± 0.16 | 47.30 ± 0.51 | 1.4131 ± 0.0354 | 79.0 ± 12.7 |
| Movies | N0 (node/off) | 55.46 ± 0.24 | 47.58 ± 1.40 | 1.4163 ± 0.0392 | 80.3 ± 7.8 |
| Movies | NS (node/static) | 55.59 ± 0.16 | 46.62 ± 0.82 | 1.4429 ± 0.0357 | 92.0 ± 12.7 |
| Movies | NC (node/context) | 55.38 ± 0.10 | 49.05 ± 0.62 | 1.4075 ± 0.0272 | 73.3 ± 10.3 |
| Grocery | G0 (global/off) | 81.82 ± 0.06 | 75.72 ± 0.09 | 0.7113 ± 0.0266 | 89.0 ± 17.0 |
| Grocery | GS (global/static) | 82.05 ± 0.05 | 75.83 ± 0.26 | 0.7223 ± 0.0274 | 100.7 ± 13.9 |
| Grocery | GC (global/context) | 82.06 ± 0.16 | 75.69 ± 0.14 | 0.7112 ± 0.0170 | 94.7 ± 10.3 |
| Grocery | N0 (node/off) | 81.87 ± 0.06 | 75.57 ± 0.17 | 0.7067 ± 0.0208 | 83.3 ± 9.0 |
| Grocery | NS (node/static) | 82.18 ± 0.16 | 75.84 ± 0.47 | 0.7156 ± 0.0408 | 89.7 ± 19.9 |
| Grocery | NC (node/context) | 82.16 ± 0.15 | 76.04 ± 0.27 | 0.7105 ± 0.0210 | 92.3 ± 17.3 |
| ele-fashion | G0 (global/off) | 87.47 ± 0.05 | 74.49 ± 0.78 | 0.4254 ± 0.0134 | 175.7 ± 20.9 |
| ele-fashion | GS (global/static) | 87.24 ± 0.04 | 73.34 ± 0.69 | 0.4089 ± 0.0034 | 126.0 ± 17.7 |
| ele-fashion | GC (global/context) | 87.24 ± 0.07 | 73.95 ± 0.32 | 0.4236 ± 0.0189 | 124.0 ± 16.9 |
| ele-fashion | N0 (node/off) | 87.38 ± 0.02 | 73.80 ± 0.26 | 0.4144 ± 0.0162 | 137.3 ± 5.2 |
| ele-fashion | NS (node/static) | 87.43 ± 0.17 | 74.46 ± 1.01 | 0.4406 ± 0.0349 | 165.3 ± 43.0 |
| ele-fashion | NC (node/context) | 87.28 ± 0.07 | 74.37 ± 0.28 | 0.4086 ± 0.0032 | 117.7 ± 0.9 |

## Paired retraining contrasts

Each delta is first cell minus second cell, paired by dataset and seed. Accuracy and Macro-F1 are percentage-point differences; CE is unscaled. These rows estimate architecture value from separately retrained models.

| Dataset | Comparison | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE | Acc paired signs (+/−/=) |
|---|---|---:|---:|---:|---:|
| Movies | N0 - G0 | -0.100 ± 0.184 | +0.155 ± 0.871 | -0.03106 ± 0.02473 | 2/1/0 |
| Movies | NS - GS | -0.020 ± 0.086 | -0.231 ± 1.416 | -0.00635 ± 0.01262 | 1/2/0 |
| Movies | NC - GC | -0.020 ± 0.099 | +1.752 ± 0.498 | -0.00559 ± 0.03063 | 1/2/0 |
| Movies | GS - G0 | +0.050 ± 0.135 | -0.574 ± 0.778 | +0.00188 ± 0.02719 | 2/1/0 |
| Movies | NS - N0 | +0.130 ± 0.143 | -0.960 ± 0.940 | +0.02659 ± 0.03769 | 2/0/1 |
| Movies | GC - GS | -0.210 ± 0.321 | +0.453 ± 1.183 | -0.03611 ± 0.01246 | 1/2/0 |
| Movies | NC - NS | -0.210 ± 0.241 | +2.436 ± 0.422 | -0.03534 ± 0.03130 | 1/2/0 |
| Movies | GC - G0 | -0.160 ± 0.187 | -0.121 ± 0.902 | -0.03423 ± 0.01878 | 1/2/0 |
| Movies | NC - N0 | -0.080 ± 0.344 | +1.476 ± 0.793 | -0.00876 ± 0.01206 | 1/2/0 |
| Movies | interaction_context=(NC-NS)-(GC-GS) | -0.000 ± 0.136 | +1.983 ± 1.594 | +0.00077 ± 0.03170 | 2/1/0 |
| Movies | interaction_adapter=(NC-N0)-(GC-G0) | +0.080 ± 0.159 | +1.597 ± 0.828 | +0.02548 ± 0.02178 | 2/1/0 |
| Grocery | N0 - G0 | +0.059 ± 0.024 | -0.147 ± 0.247 | -0.00458 ± 0.04338 | 3/0/0 |
| Grocery | NS - GS | +0.127 ± 0.179 | +0.009 ± 0.733 | -0.00679 ± 0.01987 | 2/1/0 |
| Grocery | NC - GC | +0.098 ± 0.222 | +0.356 ± 0.411 | -0.00062 ± 0.00486 | 2/1/0 |
| Grocery | GS - G0 | +0.234 ± 0.072 | +0.115 ± 0.168 | +0.01103 ± 0.03944 | 3/0/0 |
| Grocery | NS - N0 | +0.303 ± 0.097 | +0.271 ± 0.351 | +0.00881 ± 0.05158 | 3/0/0 |
| Grocery | GC - GS | +0.010 ± 0.136 | -0.145 ± 0.361 | -0.01117 ± 0.01718 | 2/1/0 |
| Grocery | NC - NS | -0.020 ± 0.113 | +0.202 ± 0.682 | -0.00500 ± 0.02238 | 2/1/0 |
| Grocery | GC - G0 | +0.244 ± 0.100 | -0.030 ± 0.215 | -0.00014 ± 0.03985 | 3/0/0 |
| Grocery | NC - N0 | +0.283 ± 0.120 | +0.473 ± 0.330 | +0.00381 ± 0.02927 | 3/0/0 |
| Grocery | interaction_context=(NC-NS)-(GC-GS) | -0.029 ± 0.239 | +0.347 ± 1.042 | +0.00617 ± 0.01526 | 1/2/0 |
| Grocery | interaction_adapter=(NC-N0)-(GC-G0) | +0.039 ± 0.199 | +0.503 ± 0.541 | +0.00395 ± 0.03923 | 2/1/0 |
| ele-fashion | N0 - G0 | -0.085 ± 0.049 | -0.691 ± 0.911 | -0.01100 ± 0.01478 | 0/3/0 |
| ele-fashion | NS - GS | +0.194 ± 0.182 | +1.126 ± 1.597 | +0.03166 ± 0.03830 | 2/1/0 |
| ele-fashion | NC - GC | +0.034 ± 0.078 | +0.410 ± 0.589 | -0.01501 ± 0.02059 | 1/2/0 |
| ele-fashion | GS - G0 | -0.232 ± 0.056 | -1.155 ± 1.345 | -0.01652 ± 0.01627 | 0/3/0 |
| ele-fashion | NS - N0 | +0.048 ± 0.155 | +0.661 ± 1.112 | +0.02614 ± 0.04059 | 2/1/0 |
| ele-fashion | GC - GS | +0.007 ± 0.104 | +0.617 ± 0.888 | +0.01470 ± 0.02229 | 1/2/0 |
| ele-fashion | NC - NS | -0.153 ± 0.093 | -0.099 ± 1.287 | -0.03197 ± 0.03685 | 0/3/0 |
| ele-fashion | GC - G0 | -0.225 ± 0.088 | -0.539 ± 0.479 | -0.00182 ± 0.01408 | 0/3/0 |
| ele-fashion | NC - N0 | -0.106 ± 0.063 | +0.563 ± 0.296 | -0.00583 ± 0.01860 | 0/3/0 |
| ele-fashion | interaction_context=(NC-NS)-(GC-GS) | -0.160 ± 0.158 | -0.715 ± 2.124 | -0.04667 ± 0.05889 | 1/2/0 |
| ele-fashion | interaction_adapter=(NC-N0)-(GC-G0) | +0.119 ± 0.126 | +1.101 ± 0.338 | -0.00401 ± 0.01070 | 3/0/0 |

## Factorial interactions

The interaction rows are constructed separately for each dataset and seed, then summarized. Pooled `ALL_DATASETS` is a descriptive aggregation across the nine paired dataset-seed cells, not an inferential sample.

| Dataset | Interaction | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE | n paired |
|---|---|---:|---:|---:|---:|
| Movies | interaction_context=(NC-NS)-(GC-GS) | -0.000 ± 0.136 | +1.983 ± 1.594 | +0.00077 ± 0.03170 | 3 |
| Grocery | interaction_context=(NC-NS)-(GC-GS) | -0.029 ± 0.239 | +0.347 ± 1.042 | +0.00617 ± 0.01526 | 3 |
| ele-fashion | interaction_context=(NC-NS)-(GC-GS) | -0.160 ± 0.158 | -0.715 ± 2.124 | -0.04667 ± 0.05889 | 3 |
| ALL_DATASETS | interaction_context=(NC-NS)-(GC-GS) | -0.063 ± 0.196 | +0.538 ± 1.986 | -0.01324 ± 0.04618 | 9 |
| Movies | interaction_adapter=(NC-N0)-(GC-G0) | +0.080 ± 0.159 | +1.597 ± 0.828 | +0.02548 ± 0.02178 | 3 |
| Grocery | interaction_adapter=(NC-N0)-(GC-G0) | +0.039 ± 0.199 | +0.503 ± 0.541 | +0.00395 ± 0.03923 | 3 |
| ele-fashion | interaction_adapter=(NC-N0)-(GC-G0) | +0.119 ± 0.126 | +1.101 ± 0.338 | -0.00401 ± 0.01070 | 3 |
| ALL_DATASETS | interaction_adapter=(NC-N0)-(GC-G0) | +0.079 ± 0.168 | +1.067 ± 0.751 | +0.00847 ± 0.02940 | 9 |

## Selected-checkpoint trajectory interventions

`traj_global_mean` replaces node-specific alpha rows by their mean; `traj_node_shuffle` permutes alpha rows with five deterministic seeds per checkpoint. The resulting deltas measure checkpoint reliance on recipient-to-hop correspondence. They are not retrained architecture effects.

| Dataset | Cell | Intervention | Accuracy (%) | Macro-F1 (%) | CE | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Movies | N0 | normal (n=3) | 55.46 ± 0.24 | 47.58 ± 1.40 | 1.4163 ± 0.0392 | +0.000 | +0.000 | +0.00000 |
| Movies | N0 | traj_global_mean (n=3) | 55.40 ± 0.19 | 47.32 ± 1.22 | 1.4191 ± 0.0388 | -0.060 | -0.258 | +0.00279 |
| Movies | N0 | traj_node_shuffle (n=15) | 55.29 ± 0.22 | 47.06 ± 1.31 | 1.4203 ± 0.0388 | -0.168 | -0.517 | +0.00399 |
| Movies | NC | normal (n=3) | 55.38 ± 0.10 | 49.05 ± 0.62 | 1.4075 ± 0.0272 | +0.000 | +0.000 | +0.00000 |
| Movies | NC | traj_global_mean (n=3) | 55.09 ± 0.05 | 48.18 ± 0.54 | 1.4079 ± 0.0264 | -0.290 | -0.878 | +0.00039 |
| Movies | NC | traj_node_shuffle (n=15) | 55.12 ± 0.19 | 48.42 ± 0.60 | 1.4094 ± 0.0265 | -0.258 | -0.635 | +0.00192 |
| Movies | NS | normal (n=3) | 55.59 ± 0.16 | 46.62 ± 0.82 | 1.4429 ± 0.0357 | +0.000 | +0.000 | +0.00000 |
| Movies | NS | traj_global_mean (n=3) | 55.53 ± 0.25 | 46.89 ± 0.91 | 1.4431 ± 0.0351 | -0.060 | +0.275 | +0.00025 |
| Movies | NS | traj_node_shuffle (n=15) | 55.37 ± 0.18 | 46.59 ± 0.90 | 1.4449 ± 0.0350 | -0.222 | -0.031 | +0.00207 |
| Grocery | N0 | normal (n=3) | 81.87 ± 0.06 | 75.57 ± 0.17 | 0.7067 ± 0.0208 | +0.000 | +0.000 | +0.00000 |
| Grocery | N0 | traj_global_mean (n=3) | 81.59 ± 0.18 | 75.31 ± 0.32 | 0.7119 ± 0.0212 | -0.283 | -0.259 | +0.00514 |
| Grocery | N0 | traj_node_shuffle (n=15) | 81.49 ± 0.17 | 75.14 ± 0.30 | 0.7126 ± 0.0211 | -0.388 | -0.427 | +0.00584 |
| Grocery | NC | normal (n=3) | 82.16 ± 0.15 | 76.04 ± 0.27 | 0.7105 ± 0.0210 | +0.000 | +0.000 | +0.00000 |
| Grocery | NC | traj_global_mean (n=3) | 81.90 ± 0.21 | 75.88 ± 0.30 | 0.7151 ± 0.0214 | -0.254 | -0.164 | +0.00456 |
| Grocery | NC | traj_node_shuffle (n=15) | 81.89 ± 0.20 | 75.89 ± 0.46 | 0.7159 ± 0.0212 | -0.264 | -0.149 | +0.00537 |
| Grocery | NS | normal (n=3) | 82.18 ± 0.16 | 75.84 ± 0.47 | 0.7156 ± 0.0408 | +0.000 | +0.000 | +0.00000 |
| Grocery | NS | traj_global_mean (n=3) | 81.97 ± 0.33 | 75.64 ± 0.86 | 0.7206 ± 0.0415 | -0.205 | -0.206 | +0.00501 |
| Grocery | NS | traj_node_shuffle (n=15) | 82.02 ± 0.22 | 75.77 ± 0.63 | 0.7215 ± 0.0417 | -0.152 | -0.073 | +0.00593 |
| ele-fashion | N0 | normal (n=3) | 87.38 ± 0.02 | 73.80 ± 0.26 | 0.4144 ± 0.0162 | +0.000 | +0.000 | +0.00000 |
| ele-fashion | N0 | traj_global_mean (n=3) | 87.22 ± 0.08 | 73.34 ± 0.33 | 0.4190 ± 0.0162 | -0.160 | -0.461 | +0.00460 |
| ele-fashion | N0 | traj_node_shuffle (n=15) | 87.19 ± 0.06 | 73.36 ± 0.29 | 0.4200 ± 0.0164 | -0.188 | -0.440 | +0.00560 |
| ele-fashion | NC | normal (n=3) | 87.28 ± 0.07 | 74.37 ± 0.28 | 0.4086 ± 0.0032 | +0.000 | +0.000 | +0.00000 |
| ele-fashion | NC | traj_global_mean (n=3) | 87.09 ± 0.05 | 73.70 ± 0.56 | 0.4130 ± 0.0027 | -0.184 | -0.664 | +0.00441 |
| ele-fashion | NC | traj_node_shuffle (n=15) | 87.09 ± 0.07 | 73.87 ± 0.58 | 0.4140 ± 0.0026 | -0.182 | -0.493 | +0.00541 |
| ele-fashion | NS | normal (n=3) | 87.43 ± 0.17 | 74.46 ± 1.01 | 0.4406 ± 0.0349 | +0.000 | +0.000 | +0.00000 |
| ele-fashion | NS | traj_global_mean (n=3) | 87.16 ± 0.10 | 73.71 ± 0.44 | 0.4461 ± 0.0361 | -0.273 | -0.759 | +0.00556 |
| ele-fashion | NS | traj_node_shuffle (n=15) | 86.98 ± 0.29 | 73.56 ± 0.34 | 0.4530 ± 0.0439 | -0.454 | -0.904 | +0.01247 |

## Selected-checkpoint adapter interventions

For GC and NC checkpoints, `adapter_off` removes the residual correction; `coeff_global_mean` removes node variation in coefficients; `coeff_node_shuffle` permutes coefficient rows in five deterministic repeats. These are checkpoint reliance tests, separate from GC−GS and NC−NS retraining contrasts.

| Dataset | Cell | Intervention | Accuracy (%) | Macro-F1 (%) | CE | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Movies | GC | adapter_off (n=3) | 53.85 ± 0.46 | 44.76 ± 0.82 | 1.4545 ± 0.0471 | -1.550 | -2.543 | +0.04135 |
| Movies | GC | coeff_global_mean (n=3) | 55.14 ± 0.21 | 47.06 ± 0.43 | 1.4126 ± 0.0351 | -0.260 | -0.242 | -0.00054 |
| Movies | GC | coeff_node_shuffle (n=15) | 55.12 ± 0.24 | 47.12 ± 0.55 | 1.4149 ± 0.0362 | -0.278 | -0.179 | +0.00180 |
| Movies | GC | normal (n=3) | 55.40 ± 0.16 | 47.30 ± 0.51 | 1.4131 ± 0.0354 | +0.000 | +0.000 | +0.00000 |
| Movies | NC | adapter_off (n=3) | 54.60 ± 0.29 | 47.81 ± 0.58 | 1.4367 ± 0.0204 | -0.780 | -1.239 | +0.02919 |
| Movies | NC | coeff_global_mean (n=3) | 55.06 ± 0.18 | 48.84 ± 0.54 | 1.4069 ± 0.0246 | -0.320 | -0.214 | -0.00063 |
| Movies | NC | coeff_node_shuffle (n=15) | 55.06 ± 0.28 | 48.63 ± 0.39 | 1.4089 ± 0.0263 | -0.322 | -0.420 | +0.00139 |
| Movies | NC | normal (n=3) | 55.38 ± 0.10 | 49.05 ± 0.62 | 1.4075 ± 0.0272 | +0.000 | +0.000 | +0.00000 |
| Grocery | GC | adapter_off (n=3) | 81.34 ± 0.12 | 74.86 ± 0.20 | 0.7366 ± 0.0157 | -0.722 | -0.828 | +0.02542 |
| Grocery | GC | coeff_global_mean (n=3) | 81.91 ± 0.06 | 75.59 ± 0.27 | 0.7105 ± 0.0161 | -0.146 | -0.092 | -0.00063 |
| Grocery | GC | coeff_node_shuffle (n=15) | 81.79 ± 0.12 | 75.40 ± 0.36 | 0.7119 ± 0.0165 | -0.273 | -0.286 | +0.00074 |
| Grocery | GC | normal (n=3) | 82.06 ± 0.16 | 75.69 ± 0.14 | 0.7112 ± 0.0170 | +0.000 | +0.000 | +0.00000 |
| Grocery | NC | adapter_off (n=3) | 81.56 ± 0.14 | 75.25 ± 0.21 | 0.7314 ± 0.0232 | -0.595 | -0.792 | +0.02089 |
| Grocery | NC | coeff_global_mean (n=3) | 81.93 ± 0.15 | 75.86 ± 0.25 | 0.7101 ± 0.0195 | -0.225 | -0.181 | -0.00047 |
| Grocery | NC | coeff_node_shuffle (n=15) | 81.93 ± 0.10 | 75.92 ± 0.29 | 0.7113 ± 0.0208 | -0.228 | -0.121 | +0.00075 |
| Grocery | NC | normal (n=3) | 82.16 ± 0.15 | 76.04 ± 0.27 | 0.7105 ± 0.0210 | +0.000 | +0.000 | +0.00000 |
| ele-fashion | GC | adapter_off (n=3) | 85.88 ± 0.38 | 70.55 ± 0.21 | 0.4519 ± 0.0237 | -1.367 | -3.404 | +0.02827 |
| ele-fashion | GC | coeff_global_mean (n=3) | 86.87 ± 0.04 | 72.82 ± 0.13 | 0.4226 ± 0.0162 | -0.375 | -1.132 | -0.00105 |
| ele-fashion | GC | coeff_node_shuffle (n=15) | 86.15 ± 0.43 | 72.24 ± 0.58 | 0.4442 ± 0.0295 | -1.092 | -1.713 | +0.02058 |
| ele-fashion | GC | normal (n=3) | 87.24 ± 0.07 | 73.95 ± 0.32 | 0.4236 ± 0.0189 | +0.000 | +0.000 | +0.00000 |
| ele-fashion | NC | adapter_off (n=3) | 86.19 ± 0.57 | 71.79 ± 0.72 | 0.4320 ± 0.0142 | -1.088 | -2.577 | +0.02343 |
| ele-fashion | NC | coeff_global_mean (n=3) | 86.98 ± 0.13 | 73.13 ± 0.25 | 0.4111 ± 0.0042 | -0.300 | -1.232 | +0.00249 |
| ele-fashion | NC | coeff_node_shuffle (n=15) | 86.58 ± 0.41 | 73.09 ± 0.30 | 0.4206 ± 0.0069 | -0.691 | -1.274 | +0.01202 |
| ele-fashion | NC | normal (n=3) | 87.28 ± 0.07 | 74.37 ± 0.28 | 0.4086 ± 0.0032 | +0.000 | +0.000 | +0.00000 |

## Mechanism diagnostics

Each selected checkpoint is reported by modality. Alpha means and node standard deviations, effective hop, entropy, dominant-hop frequency, response RMS, correction/response RMS ratios, coefficient node variation, lambda, gamma and cosine are recorded in the CSV. The CSV also records whether a normal re-evaluation reproduces the saved formal validation metrics. No cross-modality rank disagreement is calculated because the modality adapter bases are not aligned.

| Dataset | Cell | Modality | λ | γ | α k1/k2/k3 means | α node SD k1/k2/k3 | Effective hop | Entropy | Dominant hop frequency | R RMS | Δ/R | γΔ/R | Node coefficient SD | cos(Δ,R) |
|---|---|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| Movies | GC | text | 0.484 | 0.105 | 0.345/0.515/0.140 | 0.000/0.000/0.000 | 1.795 ± 0.000 | 0.969 ± 0.000 | 0.00/1.00/0.00 | 0.4731 | 2.7769 | 0.2911 | 0.29660 | -0.0412 |
| Movies | GC | visual | 0.510 | 0.107 | 0.393/0.479/0.129 | 0.000/0.000/0.000 | 1.736 ± 0.000 | 0.961 ± 0.000 | 0.33/0.67/0.00 | 0.6440 | 1.6384 | 0.1757 | 0.25283 | 0.2106 |
| Movies | N0 | text | 0.483 | 0.100 | 0.425/0.280/0.295 | 0.307/0.172/0.223 | 1.870 ± 0.508 | 0.809 ± 0.286 | 0.53/0.17/0.30 | 0.5084 | 0.0000 | 0.0000 | 0.13883 | 0.0000 |
| Movies | N0 | visual | 0.511 | 0.100 | 0.470/0.192/0.338 | 0.303/0.110/0.253 | 1.868 ± 0.547 | 0.787 ± 0.280 | 0.60/0.04/0.36 | 0.6695 | 0.0000 | 0.0000 | 0.16988 | 0.0000 |
| Movies | NS | text | 0.483 | 0.105 | 0.459/0.261/0.280 | 0.281/0.149/0.202 | 1.821 ± 0.467 | 0.846 ± 0.263 | 0.60/0.13/0.27 | 0.4988 | 2.6906 | 0.2821 | 0.00000 | -0.0252 |
| Movies | NS | visual | 0.511 | 0.108 | 0.435/0.209/0.356 | 0.291/0.108/0.245 | 1.920 ± 0.527 | 0.823 ± 0.262 | 0.56/0.04/0.39 | 0.6733 | 1.7842 | 0.1920 | 0.00000 | 0.2184 |
| Movies | NC | text | 0.487 | 0.104 | 0.428/0.278/0.294 | 0.272/0.143/0.200 | 1.866 ± 0.455 | 0.871 ± 0.247 | 0.57/0.15/0.28 | 0.4839 | 2.5118 | 0.2620 | 0.30690 | -0.0250 |
| Movies | NC | visual | 0.510 | 0.106 | 0.422/0.220/0.358 | 0.285/0.110/0.240 | 1.936 ± 0.516 | 0.838 ± 0.256 | 0.55/0.05/0.39 | 0.6571 | 1.7574 | 0.1869 | 0.21712 | 0.2211 |
| Grocery | GC | text | 0.509 | 0.110 | 0.270/0.531/0.199 | 0.000/0.000/0.000 | 1.929 ± 0.000 | 1.002 ± 0.000 | 0.00/1.00/0.00 | 0.5693 | 2.2589 | 0.2480 | 0.35640 | 0.1191 |
| Grocery | GC | visual | 0.510 | 0.108 | 0.106/0.253/0.640 | 0.000/0.000/0.000 | 2.534 ± 0.000 | 0.856 ± 0.000 | 0.00/0.00/1.00 | 0.6501 | 1.5776 | 0.1712 | 0.27817 | 0.3157 |
| Grocery | N0 | text | 0.509 | 0.100 | 0.368/0.259/0.373 | 0.298/0.164/0.246 | 2.005 ± 0.522 | 0.809 ± 0.288 | 0.42/0.15/0.43 | 0.6035 | 0.0000 | 0.0000 | 0.15907 | 0.0000 |
| Grocery | N0 | visual | 0.510 | 0.100 | 0.255/0.289/0.456 | 0.230/0.126/0.216 | 2.201 ± 0.428 | 0.877 ± 0.221 | 0.26/0.13/0.61 | 0.6581 | 0.0000 | 0.0000 | 0.16388 | 0.0000 |
| Grocery | NS | text | 0.509 | 0.109 | 0.389/0.240/0.371 | 0.299/0.156/0.248 | 1.983 ± 0.527 | 0.801 ± 0.283 | 0.45/0.14/0.41 | 0.5844 | 1.9051 | 0.2075 | 0.00000 | 0.1365 |
| Grocery | NS | visual | 0.510 | 0.108 | 0.255/0.284/0.461 | 0.226/0.119/0.212 | 2.206 ± 0.422 | 0.883 ± 0.217 | 0.26/0.12/0.62 | 0.6450 | 1.5257 | 0.1647 | 0.00000 | 0.3160 |
| Grocery | NC | text | 0.510 | 0.110 | 0.388/0.245/0.367 | 0.293/0.153/0.242 | 1.979 ± 0.514 | 0.817 ± 0.276 | 0.45/0.14/0.41 | 0.5822 | 2.1440 | 0.2363 | 0.29410 | 0.1313 |
| Grocery | NC | visual | 0.510 | 0.108 | 0.256/0.283/0.461 | 0.228/0.120/0.213 | 2.205 ± 0.425 | 0.881 ± 0.217 | 0.26/0.12/0.62 | 0.6452 | 1.4841 | 0.1610 | 0.29133 | 0.3207 |
| ele-fashion | GC | text | 0.499 | 0.113 | 0.081/0.778/0.141 | 0.000/0.000/0.000 | 2.060 ± 0.000 | 0.665 ± 0.000 | 0.00/1.00/0.00 | 0.5086 | 3.9229 | 0.4459 | 0.40652 | 0.0131 |
| ele-fashion | GC | visual | 0.491 | 0.114 | 0.809/0.014/0.177 | 0.000/0.000/0.000 | 1.368 ± 0.000 | 0.521 ± 0.000 | 1.00/0.00/0.00 | 0.5795 | 2.9669 | 0.3384 | 0.26824 | 0.0230 |
| ele-fashion | N0 | text | 0.496 | 0.100 | 0.390/0.413/0.197 | 0.225/0.191/0.112 | 1.806 ± 0.300 | 0.903 ± 0.220 | 0.44/0.51/0.05 | 0.5171 | 0.0000 | 0.0000 | 0.17180 | 0.0000 |
| ele-fashion | N0 | visual | 0.495 | 0.100 | 0.446/0.241/0.313 | 0.343/0.233/0.244 | 1.866 ± 0.547 | 0.701 ± 0.338 | 0.47/0.21/0.32 | 0.5392 | 0.0000 | 0.0000 | 0.14855 | 0.0000 |
| ele-fashion | NS | text | 0.511 | 0.116 | 0.377/0.465/0.158 | 0.231/0.209/0.096 | 1.781 ± 0.287 | 0.847 ± 0.227 | 0.39/0.58/0.03 | 0.5099 | 2.6807 | 0.3140 | 0.00000 | 0.1446 |
| ele-fashion | NS | visual | 0.521 | 0.120 | 0.518/0.183/0.298 | 0.329/0.209/0.241 | 1.780 ± 0.538 | 0.679 ± 0.336 | 0.57/0.10/0.32 | 0.5430 | 2.5338 | 0.3084 | 0.00000 | 0.1446 |
| ele-fashion | NC | text | 0.506 | 0.112 | 0.382/0.380/0.238 | 0.224/0.184/0.130 | 1.856 ± 0.315 | 0.917 ± 0.216 | 0.45/0.45/0.11 | 0.5120 | 3.4960 | 0.3931 | 0.39188 | 0.0660 |
| ele-fashion | NC | visual | 0.508 | 0.113 | 0.477/0.190/0.333 | 0.339/0.208/0.259 | 1.856 ± 0.566 | 0.682 ± 0.335 | 0.51/0.10/0.39 | 0.5251 | 2.8663 | 0.3244 | 0.25901 | 0.0791 |

## Conservative interpretation

**Case 5:** Neither trajectory adaptation nor context-over-static correction shows stable incremental value; the current evidence does not support moving to MoE.

- Node-trajectory value N0−G0: 0/3 datasets meet the primary positive rule; 3/3 are within ±0.5 accuracy points.
- Context beyond static: GC−GS is positive in 0/3 datasets; NC−NS is positive in 0/3.
- Pooled descriptive context interaction is -0.063 accuracy points; pooled adapter interaction is +0.079 points. These are paired descriptive summaries, not hypothesis tests.
- Primary-accuracy consistency rule: mean delta at least ±0.5 points and at least two of three paired seeds in that direction. Current summary: {'case': 'Case 5', 'trajectory_positive_datasets': 0, 'trajectory_approximately_zero_datasets': 3, 'global_context_positive_datasets': 0, 'node_context_positive_datasets': 0, 'global_context_approximately_zero_datasets': 3, 'node_context_approximately_zero_datasets': 3, 'dataset_dependent': False}.
- This screen leaves diffusion, prior retention and global hop utilization shared across cells, so it cannot attribute the strong baseline to any one of them. The result is compatible with those shared components carrying most of the performance, but does not establish that explanation. If a later stage is approved after review, dynamic structural trust or the backbone are reasonable next audit targets; neither is run here.
- **Metric trade-off:** at least one paired contrast has Accuracy pointing one way while Macro-F1 or CE points the other way. Accuracy remains primary; the disagreement is reported rather than treated as support.

## Self-audit

1. **是否严格从 `a44a556517112f15a8550f81c4b916906756d77f` 开始？** 是；fetch 后核验远端指定分支和本地 HEAD 一致，再从该 SHA 建分支。
2. **是否访问过任何其他历史实验分支？** 没有检查、checkout、merge 或使用其他实验分支；只按协议执行了 `git fetch origin`，之后只核验指定 A0.2 分支 SHA 并读取该来源提交/current worktree。
3. **是否运行/读取 test metrics？** 否；正式配置关闭 test，分析器拒绝 `test_*` 键，只评估 validation。
4. **六 variants 参数量是否一致？** 是；测试逐格验证参数数量、state_dict 键和布局完全一致。
5. **同 seed 初始化是否一致？** 是；测试逐张量验证六格同 seed bitwise identical。
6. **N0/NS/NC 是否严格回归到 v0 对应 variants？** 是； strict-load 后 eval 输出逐元素相等。
7. **Global trajectory 是否真的 node-invariant？** 是；selected global checkpoint 的最大 alpha node SD=0（容差 1e-8）。
8. **Node trajectory 是否真的 node-varying？** 是；测试验证 alpha 存在 recipient 间变化，formal alpha node SD 保存在 diagnostics。
9. **N0−G0 是否建立 node-specific trajectory 的 retrained value？** 0/3 数据集满足预设 Accuracy 正向规则；详见 paired contrasts。
10. **trajectory global-mean/shuffle 是否显示 checkpoint correspondence reliance？** 五次 shuffle 后 Accuracy 平均下降的 node trajectory 单元为 9/9；详细 paired checkpoint deltas 在 intervention CSV。
11. **intervention 与 retrained comparison 是否一致？** 独立重算 normal validation 对正式 checkpoint 指标逐项复现；两类结果仍作为不同证据列出，不把 checkpoint reliance 当作架构增益。
12. **GS−G0 / NS−N0 是否显示 generic adapter capacity？** 两个配对对照均按 dataset/seed 汇总；见 B 类 comparison。
13. **GC−GS / NC−NS 是否显示 context beyond static？** 分别按 dataset/seed 汇总并用 Accuracy 优先解释；见 C 类 comparison。
14. **factorial interaction 是正、负还是接近零？** context: pooled mean -0.063 pp; adapter: +0.079 pp。它们是描述性汇总，需结合各数据集行判断。
15. **是否出现 trajectory/adapter redundancy？** Case 5 中的 context×trajectory paired pattern 未明确符合功能冗余模式。
16. **是否存在 dataset-specific regime？** 未按当前描述性规则检出关键 effect 分类随数据集变化。
17. **是否仍有明显 Accuracy vs Macro-F1/CE trade-off？** 有；见报告中标记的对照。
18. **是否有充分证据进入 A3 MoE？** 否；本阶段仅形成供人工审查的 placement 证据，不自动进入 MoE。
19. **LP smoke 是否仍满足 sampled protocol？** 是；只包含 sports-copurchase 的 GC/NC 两个短 sampled smoke，不含性能 campaign。
20. **有没有超出证据的结论？** 结论仅针对三个固定 split、三次 paired model/training seeds 和 validation；不宣称 test 泛化或 LP 质量。

## Stop recommendation

本阶段已完成后停止并等待人工审查。不要自行继续 A3 MoE、dynamic trust、cross-modal router cue、auxiliary supervision、expert diversity loss、OT、正式 LP 或 NC test。
