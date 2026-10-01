# L0 — Relation-to-Function Learnability Audit

## Question and scope

This audit asks whether observable semantic and local relation evidence predicts the P1.3 validation-derived utility vector `[U_S^T, Δ_D^T, Δ_P^T, U_S^V, Δ_D^V, Δ_P^V]` on target-disjoint groups. `Δ_D = U_AbsDiff − U_Smooth` and `Δ_P = U_Product − U_Smooth`. Positive values indicate larger marginal utility under the P1.3 shared joint readout for that tested message; they are not operator labels or ground-truth edge roles.

The P1.3 utility uses validation labels: validation Accuracy selected the joint head and validation CE defines each removal utility. Although the outer CV keeps all edges for a destination in one fold, P1.3 selected its joint head using the full validation set. L0 is therefore an in-universe learnability diagnostic, not an independent generalization estimate. No original NC test labels or test metrics were accessed. The inputs reuse the same transductive graph and frozen P1.3 H0 representations as the source utility audit.

P1.3 Smooth/AbsDiff/Product targets are not mapped to E0 Smooth/Relational/Cross-Modal functions. The direct-utility probe receives stronger supervision than NC task training and does not imply an end-to-end router can learn the same signal.

## Protocol

- Reused all nine full P1.3 edge tables and all nine joint-head checkpoints; no P1.3 regeneration was needed. Utility rows are checked in exact `(src,dst,dst_degree)` order against each frozen P1.3 H0 graph support.
- Cross-seed reliability aligns by `(src,dst)` and reports overlap, Pearson/Spearman, positive-vs-nonpositive sign agreement, exact-zero fraction and target-centered Delta Spearman.
- Three deterministic, edge-count-balanced outer folds split by `dst`; all probes share each assignment. Inner early-stopping groups are also `dst`-disjoint (90/10 by unique outer-training target count).
- Target moments and full 1542D feature moments use outer-training edges only. Feature masking is applied after standardization. The six-target loss is target-balanced with `1/d_dst`, normalized to mean one, and uses standardized Huber loss, AdamW (`1e-3`, `1e-4`), gradient clipping 1, 150 epoch maximum, min epoch 20 and patience 15.
- All five EvidenceMLP inputs use `Linear(1542,128) → GELU → Linear(128,6)` and identical parameter counts/initialization. The shuffled control moves six-output tuples only among edges of the same destination in actual training groups.
- DirectState uses frozen P1.3 H0 and the M0 128→32 relation projection, exact M0 pair evidence, q/r/u encoders and a shared utility head. Frozen E0.1 Q/R/U readouts do not update KeepEdge checkpoints.
- Metrics are first averaged over the three outer folds within dataset×seed, then summarized as three-seed mean ± population SD. Edges are never pooled across seeds.

## 1–2. Target reliability by dataset and modality

| Dataset | Modality | Target | Common edges (pair mean) | Overlap / smaller table | Cross-seed Spearman (mean ± pair SD) | Centered Delta Spearman | Sign agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Grocery | text | Delta_D | 5284.333 | 0.193 | 0.714 ± 0.008 | 0.617 | 0.881 |
| Grocery | text | Delta_P | 5284.333 | 0.193 | 0.454 ± 0.018 | 0.368 | 0.683 |
| Grocery | text | U_S | 5284.333 | 0.193 | 0.769 ± 0.010 | NA | 0.901 |
| Grocery | visual | Delta_D | 5284.333 | 0.193 | 0.586 ± 0.065 | 0.571 | 0.818 |
| Grocery | visual | Delta_P | 5284.333 | 0.193 | 0.420 ± 0.053 | 0.345 | 0.679 |
| Grocery | visual | U_S | 5284.333 | 0.193 | 0.679 ± 0.033 | NA | 0.862 |
| Movies | text | Delta_D | 6403.667 | 0.204 | 0.565 ± 0.019 | 0.350 | 0.763 |
| Movies | text | Delta_P | 6403.667 | 0.204 | 0.234 ± 0.162 | 0.322 | 0.601 |
| Movies | text | U_S | 6403.667 | 0.204 | 0.517 ± 0.013 | NA | 0.727 |
| Movies | visual | Delta_D | 6403.667 | 0.204 | 0.636 ± 0.019 | 0.557 | 0.755 |
| Movies | visual | Delta_P | 6403.667 | 0.204 | 0.473 ± 0.051 | 0.410 | 0.690 |
| Movies | visual | U_S | 6403.667 | 0.204 | 0.676 ± 0.039 | NA | 0.791 |
| ele-fashion | text | Delta_D | 40181.000 | 1.000 | 0.597 ± 0.081 | 0.482 | 0.831 |
| ele-fashion | text | Delta_P | 40181.000 | 1.000 | 0.420 ± 0.093 | 0.314 | 0.718 |
| ele-fashion | text | U_S | 40181.000 | 1.000 | 0.696 ± 0.014 | NA | 0.866 |
| ele-fashion | visual | Delta_D | 40181.000 | 1.000 | 0.737 ± 0.020 | 0.245 | 0.866 |
| ele-fashion | visual | Delta_P | 40181.000 | 1.000 | 0.483 ± 0.075 | 0.223 | 0.741 |
| ele-fashion | visual | U_S | 40181.000 | 1.000 | 0.634 ± 0.098 | NA | 0.775 |

Reliability is target-specific and should bound all probe interpretations. `U_S` is utility, while the two deltas are the primary function-sensitive targets. Exact seed-pair values, overlap counts and zero fractions are in `target_reliability_by_seed_pair.csv`.

## 3–10. EvidenceMLP predictability and edge specificity

| Dataset | Modality | Target | Probe | Spearman mean ± seed SD | Sign AUROC | Within-target residual ρ | Mean per-target rank ρ ± seed SD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Grocery | text | delta_absdiff | ENDPOINT | 0.101 ± 0.014 | 0.533 | 0.079 | 0.078 ± 0.007 |
| Grocery | text | delta_product | ENDPOINT | 0.090 ± 0.011 | 0.577 | 0.029 | 0.029 ± 0.014 |
| Grocery | visual | delta_absdiff | ENDPOINT | 0.069 ± 0.016 | 0.541 | 0.055 | 0.061 ± 0.023 |
| Grocery | visual | delta_product | ENDPOINT | 0.056 ± 0.022 | 0.535 | 0.024 | 0.022 ± 0.014 |
| Grocery | text | delta_absdiff | ENDPOINT_LOCAL | 0.092 ± 0.015 | 0.502 | 0.040 | 0.029 ± 0.022 |
| Grocery | text | delta_product | ENDPOINT_LOCAL | 0.064 ± 0.011 | 0.540 | 0.020 | 0.022 ± 0.013 |
| Grocery | visual | delta_absdiff | ENDPOINT_LOCAL | 0.061 ± 0.027 | 0.514 | 0.036 | 0.042 ± 0.009 |
| Grocery | visual | delta_product | ENDPOINT_LOCAL | 0.065 ± 0.014 | 0.526 | 0.024 | 0.025 ± 0.006 |
| Grocery | text | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.076 ± 0.015 | 0.480 | 0.018 | 0.010 ± 0.019 |
| Grocery | text | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.060 ± 0.009 | 0.537 | 0.009 | 0.014 ± 0.016 |
| Grocery | visual | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.041 ± 0.024 | 0.490 | -0.006 | -0.000 ± 0.010 |
| Grocery | visual | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.061 ± 0.015 | 0.521 | 0.008 | 0.005 ± 0.012 |
| Grocery | text | delta_absdiff | SIM_ONLY | 0.053 ± 0.014 | 0.655 | 0.161 | 0.170 ± 0.004 |
| Grocery | text | delta_product | SIM_ONLY | 0.015 ± 0.021 | 0.535 | 0.044 | 0.045 ± 0.031 |
| Grocery | visual | delta_absdiff | SIM_ONLY | 0.044 ± 0.003 | 0.658 | 0.160 | 0.166 ± 0.011 |
| Grocery | visual | delta_product | SIM_ONLY | 0.052 ± 0.011 | 0.526 | 0.014 | -0.000 ± 0.028 |
| Grocery | text | delta_absdiff | TARGET_ONLY | 0.066 ± 0.007 | 0.504 | NA | NA |
| Grocery | text | delta_product | TARGET_ONLY | 0.065 ± 0.005 | 0.558 | NA | NA |
| Grocery | visual | delta_absdiff | TARGET_ONLY | 0.068 ± 0.012 | 0.505 | NA | NA |
| Grocery | visual | delta_product | TARGET_ONLY | 0.052 ± 0.017 | 0.522 | NA | NA |
| Movies | text | delta_absdiff | ENDPOINT | 0.047 ± 0.026 | 0.543 | 0.001 | 0.006 ± 0.009 |
| Movies | text | delta_product | ENDPOINT | 0.076 ± 0.017 | 0.557 | 0.029 | 0.039 ± 0.015 |
| Movies | visual | delta_absdiff | ENDPOINT | 0.045 ± 0.008 | 0.543 | 0.020 | 0.025 ± 0.004 |
| Movies | visual | delta_product | ENDPOINT | 0.027 ± 0.014 | 0.532 | 0.005 | 0.010 ± 0.007 |
| Movies | text | delta_absdiff | ENDPOINT_LOCAL | 0.047 ± 0.019 | 0.522 | 0.001 | 0.010 ± 0.002 |
| Movies | text | delta_product | ENDPOINT_LOCAL | 0.048 ± 0.016 | 0.521 | 0.038 | 0.046 ± 0.010 |
| Movies | visual | delta_absdiff | ENDPOINT_LOCAL | 0.040 ± 0.009 | 0.524 | 0.019 | 0.030 ± 0.003 |
| Movies | visual | delta_product | ENDPOINT_LOCAL | 0.033 ± 0.011 | 0.529 | 0.007 | 0.005 ± 0.004 |
| Movies | text | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.044 ± 0.020 | 0.522 | -0.001 | 0.002 ± 0.003 |
| Movies | text | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.040 ± 0.014 | 0.517 | 0.026 | 0.031 ± 0.007 |
| Movies | visual | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.036 ± 0.009 | 0.520 | 0.011 | 0.016 ± 0.004 |
| Movies | visual | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.033 ± 0.013 | 0.529 | 0.002 | -0.001 ± 0.006 |
| Movies | text | delta_absdiff | SIM_ONLY | -0.007 ± 0.008 | 0.531 | 0.005 | 0.003 ± 0.011 |
| Movies | text | delta_product | SIM_ONLY | 0.049 ± 0.022 | 0.544 | 0.055 | 0.074 ± 0.032 |
| Movies | visual | delta_absdiff | SIM_ONLY | 0.034 ± 0.011 | 0.569 | 0.028 | 0.053 ± 0.013 |
| Movies | visual | delta_product | SIM_ONLY | 0.007 ± 0.031 | 0.509 | -0.003 | 0.005 ± 0.018 |
| Movies | text | delta_absdiff | TARGET_ONLY | 0.065 ± 0.029 | 0.560 | NA | NA |
| Movies | text | delta_product | TARGET_ONLY | 0.076 ± 0.014 | 0.573 | NA | NA |
| Movies | visual | delta_absdiff | TARGET_ONLY | 0.050 ± 0.006 | 0.552 | NA | NA |
| Movies | visual | delta_product | TARGET_ONLY | 0.026 ± 0.017 | 0.533 | NA | NA |
| ele-fashion | text | delta_absdiff | ENDPOINT | 0.335 ± 0.037 | 0.686 | 0.182 | 0.138 ± 0.035 |
| ele-fashion | text | delta_product | ENDPOINT | 0.357 ± 0.057 | 0.697 | 0.170 | 0.160 ± 0.019 |
| ele-fashion | visual | delta_absdiff | ENDPOINT | 0.330 ± 0.020 | 0.716 | 0.127 | 0.120 ± 0.011 |
| ele-fashion | visual | delta_product | ENDPOINT | 0.296 ± 0.042 | 0.622 | 0.142 | 0.140 ± 0.040 |
| ele-fashion | text | delta_absdiff | ENDPOINT_LOCAL | 0.324 ± 0.007 | 0.653 | 0.150 | 0.111 ± 0.023 |
| ele-fashion | text | delta_product | ENDPOINT_LOCAL | 0.325 ± 0.013 | 0.650 | 0.154 | 0.141 ± 0.026 |
| ele-fashion | visual | delta_absdiff | ENDPOINT_LOCAL | 0.271 ± 0.008 | 0.653 | 0.128 | 0.128 ± 0.024 |
| ele-fashion | visual | delta_product | ENDPOINT_LOCAL | 0.250 ± 0.040 | 0.599 | 0.117 | 0.122 ± 0.038 |
| ele-fashion | text | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.306 ± 0.008 | 0.639 | 0.092 | 0.070 ± 0.018 |
| ele-fashion | text | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.312 ± 0.014 | 0.642 | 0.103 | 0.094 ± 0.014 |
| ele-fashion | visual | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.264 ± 0.012 | 0.649 | 0.084 | 0.091 ± 0.026 |
| ele-fashion | visual | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 0.237 ± 0.044 | 0.591 | 0.070 | 0.078 ± 0.025 |
| ele-fashion | text | delta_absdiff | SIM_ONLY | 0.131 ± 0.013 | 0.667 | 0.173 | 0.096 ± 0.007 |
| ele-fashion | text | delta_product | SIM_ONLY | 0.118 ± 0.024 | 0.606 | 0.105 | 0.130 ± 0.041 |
| ele-fashion | visual | delta_absdiff | SIM_ONLY | 0.036 ± 0.016 | 0.542 | -0.002 | -0.025 ± 0.070 |
| ele-fashion | visual | delta_product | SIM_ONLY | 0.069 ± 0.032 | 0.555 | 0.081 | 0.116 ± 0.023 |
| ele-fashion | text | delta_absdiff | TARGET_ONLY | 0.342 ± 0.022 | 0.656 | NA | NA |
| ele-fashion | text | delta_product | TARGET_ONLY | 0.367 ± 0.038 | 0.693 | NA | NA |
| ele-fashion | visual | delta_absdiff | TARGET_ONLY | 0.344 ± 0.016 | 0.722 | NA | NA |
| ele-fashion | visual | delta_product | TARGET_ONLY | 0.324 ± 0.070 | 0.609 | NA | NA |

The shuffled control preserves each target's six-vector multiset within each training destination, so LOCAL-vs-SHUFFLED contrasts test edge-evidence correspondence. `TARGET_ONLY` measures target/context-level signal. The distinction between total and within-target metrics is essential: a positive total correlation alone does not show that the probe can rank different incoming edges for one recipient.

## 11. High-margin diagnostic

Within each outer test fold and Delta target, the high-margin subset is defined as `|true Delta| >= test-fold q75`; it is used for evaluation only. No rows are excluded from primary fits or metrics.

| Dataset | Modality | Target | Probe | High-margin n | Sign AUROC | Balanced accuracy | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Grocery | text | delta_absdiff | ENDPOINT_LOCAL | 2313.556 | 0.536 | 0.525 | 0.069 |
| Grocery | text | delta_product | ENDPOINT_LOCAL | 2313.556 | 0.518 | 0.516 | 0.026 |
| Grocery | visual | delta_absdiff | ENDPOINT_LOCAL | 2313.556 | 0.538 | 0.528 | 0.061 |
| Grocery | visual | delta_product | ENDPOINT_LOCAL | 2313.556 | 0.522 | 0.514 | 0.046 |
| Grocery | text | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 2313.556 | 0.526 | 0.517 | 0.060 |
| Grocery | text | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 2313.556 | 0.515 | 0.514 | 0.021 |
| Grocery | visual | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 2313.556 | 0.529 | 0.520 | 0.051 |
| Grocery | visual | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 2313.556 | 0.520 | 0.512 | 0.043 |
| Grocery | text | delta_absdiff | Frozen_U_MODAL | 2313.556 | 0.548 | 0.526 | 0.081 |
| Grocery | text | delta_product | Frozen_U_MODAL | 2313.556 | 0.528 | 0.523 | 0.051 |
| Grocery | visual | delta_absdiff | Frozen_U_MODAL | 2313.556 | 0.527 | 0.517 | 0.042 |
| Grocery | visual | delta_product | Frozen_U_MODAL | 2313.556 | 0.522 | 0.507 | 0.041 |
| Grocery | text | delta_absdiff | M0StateDirect | 2313.556 | 0.524 | 0.505 | 0.061 |
| Grocery | text | delta_product | M0StateDirect | 2313.556 | 0.496 | 0.498 | 0.010 |
| Grocery | visual | delta_absdiff | M0StateDirect | 2313.556 | 0.531 | 0.518 | 0.052 |
| Grocery | visual | delta_product | M0StateDirect | 2313.556 | 0.516 | 0.508 | 0.027 |
| Movies | text | delta_absdiff | ENDPOINT_LOCAL | 2664.667 | 0.536 | 0.520 | 0.059 |
| Movies | text | delta_product | ENDPOINT_LOCAL | 2664.667 | 0.532 | 0.520 | 0.079 |
| Movies | visual | delta_absdiff | ENDPOINT_LOCAL | 2664.667 | 0.522 | 0.512 | 0.028 |
| Movies | visual | delta_product | ENDPOINT_LOCAL | 2664.667 | 0.514 | 0.509 | 0.021 |
| Movies | text | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 2664.667 | 0.536 | 0.519 | 0.056 |
| Movies | text | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 2664.667 | 0.529 | 0.516 | 0.076 |
| Movies | visual | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 2664.667 | 0.521 | 0.514 | 0.028 |
| Movies | visual | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 2664.667 | 0.515 | 0.510 | 0.023 |
| Movies | text | delta_absdiff | Frozen_U_MODAL | 2664.667 | 0.527 | 0.512 | 0.031 |
| Movies | text | delta_product | Frozen_U_MODAL | 2664.667 | 0.540 | 0.519 | 0.076 |
| Movies | visual | delta_absdiff | Frozen_U_MODAL | 2664.667 | 0.508 | 0.500 | 0.012 |
| Movies | visual | delta_product | Frozen_U_MODAL | 2664.667 | 0.513 | 0.505 | 0.015 |
| Movies | text | delta_absdiff | M0StateDirect | 2664.667 | 0.524 | 0.507 | 0.024 |
| Movies | text | delta_product | M0StateDirect | 2664.667 | 0.513 | 0.503 | 0.027 |
| Movies | visual | delta_absdiff | M0StateDirect | 2664.667 | 0.511 | 0.504 | 0.010 |
| Movies | visual | delta_product | M0StateDirect | 2664.667 | 0.517 | 0.512 | 0.027 |
| ele-fashion | text | delta_absdiff | ENDPOINT_LOCAL | 3349.111 | 0.697 | 0.646 | 0.277 |
| ele-fashion | text | delta_product | ENDPOINT_LOCAL | 3349.111 | 0.691 | 0.642 | 0.264 |
| ele-fashion | visual | delta_absdiff | ENDPOINT_LOCAL | 3349.000 | 0.620 | 0.581 | 0.188 |
| ele-fashion | visual | delta_product | ENDPOINT_LOCAL | 3349.000 | 0.631 | 0.591 | 0.201 |
| ele-fashion | text | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 3349.111 | 0.691 | 0.639 | 0.271 |
| ele-fashion | text | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 3349.111 | 0.687 | 0.638 | 0.261 |
| ele-fashion | visual | delta_absdiff | ENDPOINT_LOCAL_SHUFFLED_TARGET | 3349.000 | 0.616 | 0.579 | 0.182 |
| ele-fashion | visual | delta_product | ENDPOINT_LOCAL_SHUFFLED_TARGET | 3349.000 | 0.624 | 0.589 | 0.194 |
| ele-fashion | text | delta_absdiff | Frozen_U_MODAL | 3349.111 | 0.693 | 0.646 | 0.247 |
| ele-fashion | text | delta_product | Frozen_U_MODAL | 3349.111 | 0.677 | 0.631 | 0.222 |
| ele-fashion | visual | delta_absdiff | Frozen_U_MODAL | 3349.000 | 0.613 | 0.561 | 0.165 |
| ele-fashion | visual | delta_product | Frozen_U_MODAL | 3349.000 | 0.630 | 0.568 | 0.180 |
| ele-fashion | text | delta_absdiff | M0StateDirect | 3349.111 | 0.700 | 0.647 | 0.272 |
| ele-fashion | text | delta_product | M0StateDirect | 3349.111 | 0.695 | 0.648 | 0.254 |
| ele-fashion | visual | delta_absdiff | M0StateDirect | 3349.000 | 0.625 | 0.572 | 0.180 |
| ele-fashion | visual | delta_product | M0StateDirect | 3349.000 | 0.637 | 0.579 | 0.192 |

## 12–13. M0StateDirect capacity under utility supervision

| Dataset | Modality | Target | Spearman mean ± seed SD | Sign AUROC | Within-target residual ρ | Per-target rank ρ |
| --- | --- | --- | --- | --- | --- | --- |
| Grocery | text | delta_absdiff | 0.091 ± 0.037 | 0.493 | 0.044 | 0.044 |
| Grocery | text | delta_product | 0.049 ± 0.018 | 0.529 | 0.017 | 0.021 |
| Grocery | text | smooth_utility | 0.120 ± 0.035 | 0.500 | 0.034 | 0.026 |
| Grocery | visual | delta_absdiff | 0.081 ± 0.030 | 0.518 | 0.026 | 0.032 |
| Grocery | visual | delta_product | 0.046 ± 0.025 | 0.519 | 0.009 | 0.006 |
| Grocery | visual | smooth_utility | 0.096 ± 0.026 | 0.512 | 0.030 | 0.025 |
| Movies | text | delta_absdiff | 0.040 ± 0.015 | 0.539 | -0.002 | 0.004 |
| Movies | text | delta_product | 0.062 ± 0.033 | 0.563 | 0.007 | 0.019 |
| Movies | text | smooth_utility | 0.055 ± 0.006 | 0.548 | -0.006 | 0.001 |
| Movies | visual | delta_absdiff | 0.047 ± 0.026 | 0.547 | 0.003 | 0.007 |
| Movies | visual | delta_product | 0.027 ± 0.021 | 0.537 | 0.001 | 0.002 |
| Movies | visual | smooth_utility | 0.048 ± 0.022 | 0.538 | 0.005 | 0.013 |
| ele-fashion | text | delta_absdiff | 0.358 ± 0.039 | 0.620 | 0.126 | 0.076 |
| ele-fashion | text | delta_product | 0.365 ± 0.034 | 0.648 | 0.132 | 0.106 |
| ele-fashion | text | smooth_utility | 0.347 ± 0.061 | 0.576 | 0.121 | 0.088 |
| ele-fashion | visual | delta_absdiff | 0.317 ± 0.043 | 0.678 | 0.053 | 0.029 |
| ele-fashion | visual | delta_product | 0.268 ± 0.053 | 0.580 | 0.076 | 0.052 |
| ele-fashion | visual | smooth_utility | 0.282 ± 0.040 | 0.640 | 0.069 | 0.047 |

This is a utility-supervised capacity probe, not an NC architecture or router. Compare it with `ENDPOINT_LOCAL` descriptively; direct utility supervision is stronger than task-derived learning.

## 14–17. Frozen task-trained E0.1 state decodability

| Dataset | Modality | Target | Representation | Spearman mean ± seed SD | Sign AUROC | Within-target residual ρ | Per-target rank ρ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Grocery | text | delta_absdiff | Q_PAIR | 0.174 ± 0.037 | 0.487 | 0.025 | 0.027 |
| Grocery | text | delta_product | Q_PAIR | 0.081 ± 0.014 | 0.535 | 0.006 | 0.021 |
| Grocery | text | smooth_utility | Q_PAIR | 0.202 ± 0.034 | 0.466 | 0.009 | 0.019 |
| Grocery | visual | delta_absdiff | Q_PAIR | 0.100 ± 0.010 | 0.491 | 0.015 | 0.015 |
| Grocery | visual | delta_product | Q_PAIR | 0.095 ± 0.025 | 0.523 | 0.001 | -0.006 |
| Grocery | visual | smooth_utility | Q_PAIR | 0.150 ± 0.008 | 0.475 | 0.004 | -0.004 |
| Grocery | text | delta_absdiff | R_SHARED | 0.097 ± 0.019 | 0.505 | 0.029 | 0.030 |
| Grocery | text | delta_product | R_SHARED | 0.056 ± 0.004 | 0.526 | 0.006 | 0.013 |
| Grocery | text | smooth_utility | R_SHARED | 0.128 ± 0.042 | 0.495 | 0.019 | 0.018 |
| Grocery | visual | delta_absdiff | R_SHARED | 0.092 ± 0.015 | 0.488 | 0.009 | 0.003 |
| Grocery | visual | delta_product | R_SHARED | 0.044 ± 0.025 | 0.499 | 0.000 | -0.004 |
| Grocery | visual | smooth_utility | R_SHARED | 0.097 ± 0.013 | 0.483 | 0.019 | 0.003 |
| Grocery | text | delta_absdiff | U_MODAL | 0.119 ± 0.020 | 0.507 | 0.033 | 0.036 |
| Grocery | text | delta_product | U_MODAL | 0.059 ± 0.018 | 0.527 | 0.017 | 0.017 |
| Grocery | text | smooth_utility | U_MODAL | 0.144 ± 0.011 | 0.503 | 0.035 | 0.029 |
| Grocery | visual | delta_absdiff | U_MODAL | 0.080 ± 0.032 | 0.497 | -0.002 | -0.005 |
| Grocery | visual | delta_product | U_MODAL | 0.063 ± 0.035 | 0.514 | -0.000 | -0.003 |
| Grocery | visual | smooth_utility | U_MODAL | 0.113 ± 0.044 | 0.471 | -0.020 | -0.030 |
| Movies | text | delta_absdiff | Q_PAIR | 0.089 ± 0.020 | 0.524 | 0.004 | 0.007 |
| Movies | text | delta_product | Q_PAIR | 0.077 ± 0.012 | 0.534 | 0.013 | 0.031 |
| Movies | text | smooth_utility | Q_PAIR | 0.087 ± 0.016 | 0.518 | -0.001 | 0.004 |
| Movies | visual | delta_absdiff | Q_PAIR | 0.056 ± 0.011 | 0.534 | 0.016 | 0.020 |
| Movies | visual | delta_product | Q_PAIR | 0.039 ± 0.010 | 0.526 | 0.004 | 0.005 |
| Movies | visual | smooth_utility | Q_PAIR | 0.066 ± 0.002 | 0.518 | 0.000 | 0.003 |
| Movies | text | delta_absdiff | R_SHARED | 0.058 ± 0.020 | 0.540 | -0.001 | 0.001 |
| Movies | text | delta_product | R_SHARED | 0.059 ± 0.023 | 0.533 | 0.001 | 0.002 |
| Movies | text | smooth_utility | R_SHARED | 0.075 ± 0.013 | 0.546 | -0.002 | -0.002 |
| Movies | visual | delta_absdiff | R_SHARED | 0.033 ± 0.030 | 0.534 | 0.008 | 0.014 |
| Movies | visual | delta_product | R_SHARED | 0.039 ± 0.028 | 0.527 | 0.007 | 0.012 |
| Movies | visual | smooth_utility | R_SHARED | 0.056 ± 0.026 | 0.511 | -0.000 | 0.007 |
| Movies | text | delta_absdiff | U_MODAL | 0.056 ± 0.010 | 0.534 | 0.004 | 0.004 |
| Movies | text | delta_product | U_MODAL | 0.073 ± 0.023 | 0.543 | 0.026 | 0.039 |
| Movies | text | smooth_utility | U_MODAL | 0.066 ± 0.014 | 0.525 | 0.010 | 0.013 |
| Movies | visual | delta_absdiff | U_MODAL | 0.050 ± 0.010 | 0.527 | 0.011 | 0.018 |
| Movies | visual | delta_product | U_MODAL | 0.048 ± 0.019 | 0.524 | 0.010 | 0.016 |
| Movies | visual | smooth_utility | U_MODAL | 0.079 ± 0.019 | 0.509 | 0.010 | 0.020 |
| ele-fashion | text | delta_absdiff | Q_PAIR | 0.370 ± 0.040 | 0.607 | 0.091 | 0.049 |
| ele-fashion | text | delta_product | Q_PAIR | 0.292 ± 0.018 | 0.591 | 0.060 | 0.051 |
| ele-fashion | text | smooth_utility | Q_PAIR | 0.385 ± 0.021 | 0.560 | 0.062 | 0.033 |
| ele-fashion | visual | delta_absdiff | Q_PAIR | 0.310 ± 0.025 | 0.648 | 0.068 | 0.060 |
| ele-fashion | visual | delta_product | Q_PAIR | 0.305 ± 0.033 | 0.581 | 0.072 | 0.067 |
| ele-fashion | visual | smooth_utility | Q_PAIR | 0.278 ± 0.020 | 0.620 | 0.070 | 0.065 |
| ele-fashion | text | delta_absdiff | R_SHARED | 0.317 ± 0.014 | 0.614 | 0.074 | 0.035 |
| ele-fashion | text | delta_product | R_SHARED | 0.278 ± 0.015 | 0.603 | 0.054 | 0.052 |
| ele-fashion | text | smooth_utility | R_SHARED | 0.302 ± 0.011 | 0.533 | 0.035 | 0.017 |
| ele-fashion | visual | delta_absdiff | R_SHARED | 0.306 ± 0.035 | 0.677 | 0.024 | 0.018 |
| ele-fashion | visual | delta_product | R_SHARED | 0.283 ± 0.052 | 0.560 | 0.022 | 0.015 |
| ele-fashion | visual | smooth_utility | R_SHARED | 0.258 ± 0.068 | 0.631 | 0.023 | 0.017 |
| ele-fashion | text | delta_absdiff | U_MODAL | 0.310 ± 0.013 | 0.608 | 0.069 | 0.041 |
| ele-fashion | text | delta_product | U_MODAL | 0.285 ± 0.035 | 0.608 | 0.033 | 0.025 |
| ele-fashion | text | smooth_utility | U_MODAL | 0.304 ± 0.035 | 0.533 | 0.028 | 0.013 |
| ele-fashion | visual | delta_absdiff | U_MODAL | 0.313 ± 0.027 | 0.692 | 0.018 | 0.010 |
| ele-fashion | visual | delta_product | U_MODAL | 0.286 ± 0.043 | 0.551 | 0.020 | 0.029 |
| ele-fashion | visual | smooth_utility | U_MODAL | 0.283 ± 0.044 | 0.658 | 0.032 | 0.033 |

`U_MODAL` is the E0.1 router input. `Q_PAIR`, `R_SHARED` and `U_MODAL` are 64D and were extracted from fully frozen KeepEdge checkpoints after exact ordered edge alignment. Differences from DirectState locate where task-trained state may fail to retain a directly supervised signal; they are not matched-supervision performance comparisons.

## 18. Diagnostic judgment

**Final label: `DATASET_DEPENDENT_MIXED`.**

The dataset-specific rationale and all individual question answers are recorded below. The label is based on the full reliability, correspondence-control, within-target, high-margin and representation evidence rather than a single metric.

This classification combines cross-seed target reliability, target-only vs endpoint/local evidence, the correspondence-shuffled control, within-target residual/rank metrics, high-margin results, DirectState, and frozen Q/R/U. It does not use a universal Spearman threshold.

## 19–20. Limits and next step

The target is a P1.3 diagnostic derived from validation labels and a joint head selected on that same validation split. Outer target-group CV prevents same-destination probe train/test leakage, but it cannot undo the upstream P1.3 validation reuse. The three seeds are limited and their shared edge support is not an independent sample. Near-tie utility remains in all primary analyses; high-margin results are secondary only.

The next step should be selected after human review of this attribution. This report does not design or implement M1, change E0/E0.1, or train an end-to-end utility router.

## Correctness, execution and self-audit

- Utility formula, finite-target, artifact-presence, edge-order, master dimension, neighborhood/LOO construction, degree standardization, masks, group splits, training-only normalization, balancing weights, tuple shuffle, exact pair evidence, DirectState output shapes and degenerate AUROC behavior are covered by the L0 correctness tests. Movies/42 smoke additionally verifies P1.3 H0 regression, frozen E0.1 extraction/alignment, finite losses/gradients, all readout shapes, no visible test labels, and GPU memory.
- All EvidenceMLP variants share the same 1542D input shape and equal parameter count; their initialization is bitwise equal within dataset×seed×fold. The direct probe has direct utility labels; this does not imply NC training should learn the same mapping.
- No label, CE, utility, logits, preferred-channel, source ID or destination ID enters any feature tensor. Node IDs are used only for indexing, grouping and alignment.
- Utility targets are never called ground truth. No P1.3 AbsDiff/Product to E0 Relational/Cross mapping is made. High-margin edges do not replace or filter the full-edge analysis.

### Explicit answers to questions 1–20

1. **Cross-seed stability:** `Delta_D` is more repeatable than `Delta_P` in most dataset/modality cells. Mean pairwise Spearman spans 0.565–0.737 for `Delta_D`, versus 0.234–0.483 for `Delta_P`; Movies/Text `Delta_P` is the least stable (0.234 ± 0.162 across seed pairs). This estimate is qualified by low common-edge overlap for Movies and Grocery (about 20% of the smaller table); ele-fashion has complete edge overlap.
2. **Dataset/modality differences:** reliability varies by operator delta and modality. The clearest example is ele-fashion/Visual `Delta_D`: total cross-seed rho is 0.737 while target-centered rho is 0.245, showing that much of its repeatability is target-level. Movies/Text `Delta_P` is weaker than the other cells. No single stability profile covers all six dataset×modality groups.
3. **Smooth versus function delta:** Smooth utility is not consistently easier to predict. `TARGET_ONLY` predicts Smooth at rho 0.05 (Movies), 0.108 (Grocery), and 0.342 (ele-fashion); Delta ranges are 0.026–0.058, 0.052–0.067, and 0.324–0.367 respectively. Smooth is near the delta targets within each dataset, with operator/modality variation.
4. **SIM_ONLY:** endpoint semantic cosine has weak total correlation in Movies/Grocery (typically |rho| below 0.06), while ele-fashion shows limited signal (up to 0.131 total). It sometimes ranks within-target variation, especially Grocery `Delta_D` (residual rho about 0.16), but this does not generalize across target types/modalities.
5. **TARGET_ONLY:** Movies/Grocery total function-delta predictability is small (rho about 0.026–0.067). Ele-fashion has appreciable total predictability (rho 0.324–0.367; sign AUROC about 0.61–0.72), but a target-only prediction is constant within a destination and therefore cannot rank that destination's incoming edges.
6. **ENDPOINT versus TARGET_ONLY:** endpoint evidence does not yield a consistent total-prediction gain. The mean Delta Spearman difference is −0.005 in Movies, +0.016 in Grocery, and −0.015 in ele-fashion. Endpoint probes do show limited within-target ranking for some Delta cells, particularly ele-fashion (residual rho about 0.13–0.18), so target-only results do not exclude all source-specific signal.
7. **LOCAL versus ENDPOINT:** full-neighborhood and LOO context do not add a consistent increment. Mean Delta total Spearman changes by −0.007 (Movies), −0.008 (Grocery), and −0.037 (ele-fashion); within-target residual changes are near zero or negative on average.
8. **LOCAL versus SHUFFLED:** preserving real edge/utility correspondence yields a small positive mean difference: Delta total Spearman +0.004 (Movies), +0.011 (Grocery), +0.013 (ele-fashion), and within-target residual rho +0.007, +0.023, and +0.050 respectively. The shuffled control remains close in total metrics, so correspondence signal is modest and strongest in ele-fashion.
9. **Total versus within-target:** they do not agree in scale. In ele-fashion, total Delta rho is roughly 0.25–0.37, while LOCAL residual rho is 0.12–0.15 and SHUFFLED is 0.07–0.10. Movies/Grocery totals are generally at or below 0.10 and residuals are mostly close to zero. Total association therefore overstates edge-specific predictability.
10. **Incoming-edge ranking:** there is partial rankability in ele-fashion: ENDPOINT residual rho is about 0.13–0.18 and LOCAL per-target mean rank rho about 0.11–0.14, compared with SHUFFLED per-target rank about 0.07–0.09. Movies/Grocery rank correlations are generally weak and inconsistent. This is not a uniform cross-dataset result.
11. **High-margin edges:** the top `|Delta|` quartile is more predictable in ele-fashion (LOCAL sign AUROC about 0.62–0.70; Spearman about 0.19–0.28), but SHUFFLED is close (AUROC about 0.62–0.69). Movies/Grocery remain near chance. Strong-preference edges carry some predictable signal in ele-fashion, but the high-margin gain does not establish real edge-evidence correspondence. All near-tie edges remain in the primary analysis.
12. **DirectState capacity:** direct utility supervision learns little edge-specific signal in Movies/Grocery (Delta total rho mostly below 0.10 and residual rho below about 0.05). Ele-fashion DirectState reaches total rho 0.27–0.37 and residual rho 0.05–0.13.
13. **DirectState versus LOCAL:** their total predictability is comparable in ele-fashion (DirectState rho 0.27–0.37; LOCAL 0.25–0.32), and both are weak in Movies/Grocery. DirectState within-target residuals are usually lower than LOCAL in ele-fashion. This does not support a single, global M0 state-capacity bottleneck; it leaves some edge-level compression concern in that dataset.
14. **Frozen q:** Q_PAIR has total Delta rho 0.29–0.37 in ele-fashion, versus about 0.04–0.17 in Movies/Grocery. Its within-target residual/per-target ranks remain much smaller, so q contains target/context association more clearly than edge preference.
15. **Frozen r:** R_SHARED is close to Q_PAIR in total ele-fashion association (about 0.28–0.32), with some reduction in within-target metrics. The relation-level compression retains much of the total context signal but has weaker evidence for preserving edge-specific ordering.
16. **Frozen u:** U_MODAL retains ele-fashion total association (rho about 0.285–0.313) and modest Grocery/Movies association, but within-target residual rho is only about 0.018–0.069 in ele-fashion and near zero in most Movies/Grocery cells.
17. **Actual router input:** U_MODAL can decode total function utility in ele-fashion, but its within-target ranking is weak, particularly for Visual targets. Across all three datasets it does not provide stable, strong incoming-edge preference prediction. Thus the L0 evidence does not isolate routing/credit assignment as the primary global bottleneck.
18. **Overall label:** `DATASET_DEPENDENT_MIXED`. Movies/Grocery mostly show low edge-level predictability despite moderate Delta reliability, while ele-fashion shows stronger target/context predictability, some within-target signal and high-margin signal, with only a modest advantage over correspondence-shuffled controls.
19. **Reliability limitation:** upstream validation-label reuse remains, and common-edge overlap is only about one fifth of the smaller Movies/Grocery tables. Ele-fashion has full overlap. The target stability ceiling and dataset-specific overlap limit direct comparisons; three seeds do not support edge-level pseudo-significance.
20. **Next step (not implemented):** first build an independent target audit in which a joint head is selected on one validation partition and edge-removal CE utilities are computed on a disjoint validation partition, while preserving target-group CV. Re-evaluate whether the ele-fashion within-target and high-margin signal survives before designing or training any utility-aware routing mechanism.


### A–J self-audit

- **A. Utility interpretation:** validation-derived P1.3 utilities are treated as diagnostic targets, not ground truth.
- **B. Same-target leakage:** every incoming edge for a `dst` stays in one outer fold and one inner split.
- **C. Target-level versus edge-level:** total correlations are interpreted alongside destination-centered residual correlations and per-target ranks.
- **D. Input leakage:** no labels, CE, utility, logits, preferred channel or numeric IDs enter the feature tensor.
- **E. Probe capacity fairness:** every EvidenceMLP uses 1542 inputs and the same parameter count; initialization is bitwise equal within each fold.
- **F. Direct supervision:** DirectState is described only as utility-supervised representational capacity, not evidence that NC training should learn it.
- **G. Function mapping:** P1.3 AbsDiff/Product are not mapped onto E0 Relational/Cross-Modal.
- **H. Target noise:** cross-seed reliability, overlap and centered Delta stability qualify all conclusions.
- **I. Within-target analysis:** residual Spearman and per-target ranking are reported; total Spearman alone is not used to claim edge-specific predictability.
- **J. High-margin selection:** the top `|Delta|` quartile is diagnostic only; near-tie edges remain in every primary analysis.

## Files

Run-level fold metrics, seed-aggregated summaries, reliability records, assignment CSV, high-margin rows, parameter audit, smoke audit and figures are under `data/` and `figures/`. Large predictions/checkpoints and extracted E0.1 state caches are in ignored `outputs/l0_relation_function_learnability_audit/`.
