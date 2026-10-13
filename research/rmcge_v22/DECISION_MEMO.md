# Decision memo

This memo distinguishes (a) measured paired Validation facts, (b) frozen-checkpoint sensitivity, and (c) hypotheses this experiment cannot identify. Three seeds are descriptive; no significance claim is made. All deltas below are left arm minus right arm, with seed order 42/43/44.

## 1. Does M0 reproduce the strong collaborative baseline?

The frozen implementation audit found M0/C0 shared tensors and classifier exact and the real-data initialization forward within the preregistered `5e-5` tolerance (largest observed difference `2.0266e-6`). This supports forward compatibility. The new M0 is the internal baseline for all study claims; historical V2.1 C0 is context only:

| Dataset | New M0 Acc % | Historical V2.1 C0 Acc % | Δ pp | New M0 Macro-F1 % | Historical C0 Macro-F1 % | Δ pp | New M0 CE | Historical C0 CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 55.889 | 55.87 | +0.019 | 47.785 | 47.20 | +0.585 | 1.4333 | 1.4172 |
| Grocery | 83.367 | 83.35 | +0.017 | 75.889 | 75.83 | +0.059 | 0.6595 | 0.6721 |
| ele-fashion | 87.597 | 87.55 | +0.047 | 74.537 | 75.34 | -0.803 | 0.4404 | 0.4363 |

These historical comparisons are descriptive, not exact replay criteria. Differences can include stochastic training trajectory; they do not invalidate the audited V2.1 C0 mathematics or the internally paired comparisons.

## Best Accuracy versus Best F1 checkpoint

The Best F1 checkpoint is selected independently and can trade Accuracy for Macro-F1. The following reports that selection gap for the two pair-content arms:

| Dataset | Arm | Best F1 ckpt minus Best Accuracy: Accuracy pp | Macro-F1 pp | CE |
|---|---|---:|---:|---:|
| Movies | M3 | -0.89 ± 0.79 | 0.72 ± 0.63 | 0.0337 ± 0.0292 |
| Movies | M4 | -0.58 ± 0.25 | 1.17 ± 0.79 | -0.0054 ± 0.0682 |
| Grocery | M3 | -0.54 ± 0.31 | 0.71 ± 0.52 | 0.0448 ± 0.0086 |
| Grocery | M4 | -0.58 ± 0.36 | 1.01 ± 0.23 | 0.0396 ± 0.0233 |
| ele-fashion | M3 | -0.95 ± 0.65 | 1.38 ± 0.36 | 0.0170 ± 0.0114 |
| ele-fashion | M4 | -0.59 ± 0.45 | 1.11 ± 0.49 | 0.0098 ± 0.0060 |

These are checkpoint-selection differences within the same trained run, not separate condition effects. The full M0–M4 Best F1 paired contrasts are in `PILOT_REPORT.md` and `data/paired_comparisons.csv`.

## 2–3. Are sender-only or receiver-only transforms sufficient?

Best-Accuracy checkpoint paired contrasts (percentage-point Acc / Macro-F1, then raw CE):

| Dataset | M1−M0 Acc / F1 / CE | M2−M0 Acc / F1 / CE | M3−M2 Acc / F1 / CE |
|---|---|---|---|
| Movies | Acc +0.110 ± 0.279 pp (seeds 42/43/44: +0.300, -0.210, +0.240; Δ>0 2/3, Δ<0 1/3); F1 +0.259 ± 0.571 pp (seeds 42/43/44: +0.273, -0.320, +0.823; Δ>0 2/3, Δ<0 1/3); CE -0.039 ± 0.031 (seeds 42/43/44: -0.018, -0.075, -0.024; Δ>0 0/3, Δ<0 3/3)<br>Acc +0.070 ± 0.390 pp (seeds 42/43/44: +0.090, -0.330, +0.450; Δ>0 2/3, Δ<0 1/3); F1 -0.571 ± 0.575 pp (seeds 42/43/44: -0.017, -1.165, -0.532; Δ>0 0/3, Δ<0 3/3); CE -0.025 ± 0.074 (seeds 42/43/44: -0.028, -0.098, +0.050; Δ>0 1/3, Δ<0 2/3)<br>Acc -0.080 ± 0.387 pp (seeds 42/43/44: +0.240, +0.030, -0.510; Δ>0 2/3, Δ<0 1/3); F1 +1.777 ± 1.172 pp (seeds 42/43/44: +3.035, +0.717, +1.579; Δ>0 3/3, Δ<0 0/3); CE -0.040 ± 0.050 (seeds 42/43/44: +0.003, -0.027, -0.095; Δ>0 1/3, Δ<0 2/3) |
| Grocery | Acc +0.068 ± 0.135 pp (seeds 42/43/44: -0.088, +0.146, +0.146; Δ>0 2/3, Δ<0 1/3); F1 +0.154 ± 0.742 pp (seeds 42/43/44: -0.630, +0.248, +0.845; Δ>0 2/3, Δ<0 1/3); CE +0.022 ± 0.069 (seeds 42/43/44: -0.016, -0.019, +0.101; Δ>0 1/3, Δ<0 2/3)<br>Acc +0.010 ± 0.161 pp (seeds 42/43/44: +0.000, +0.176, -0.146; Δ>0 1/3, Δ<0 1/3); F1 +0.407 ± 0.418 pp (seeds 42/43/44: -0.056, +0.523, +0.754; Δ>0 2/3, Δ<0 1/3); CE +0.003 ± 0.018 (seeds 42/43/44: +0.016, -0.018, +0.012; Δ>0 2/3, Δ<0 1/3)<br>Acc -0.039 ± 0.212 pp (seeds 42/43/44: -0.176, -0.146, +0.205; Δ>0 1/3, Δ<0 2/3); F1 -0.510 ± 0.405 pp (seeds 42/43/44: -0.471, -0.126, -0.932; Δ>0 0/3, Δ<0 3/3); CE -0.015 ± 0.021 (seeds 42/43/44: -0.037, +0.005, -0.012; Δ>0 1/3, Δ<0 2/3) |
| ele-fashion | Acc -0.259 ± 0.083 pp (seeds 42/43/44: -0.245, -0.348, -0.184; Δ>0 0/3, Δ<0 3/3); F1 -0.460 ± 0.448 pp (seeds 42/43/44: -0.129, -0.969, -0.281; Δ>0 0/3, Δ<0 3/3); CE -0.028 ± 0.020 (seeds 42/43/44: -0.049, -0.025, -0.010; Δ>0 0/3, Δ<0 3/3)<br>Acc -0.194 ± 0.161 pp (seeds 42/43/44: -0.123, -0.378, -0.082; Δ>0 0/3, Δ<0 3/3); F1 -0.879 ± 0.540 pp (seeds 42/43/44: -0.726, -1.479, -0.431; Δ>0 0/3, Δ<0 3/3); CE -0.028 ± 0.023 (seeds 42/43/44: -0.036, -0.046, -0.002; Δ>0 0/3, Δ<0 3/3)<br>Acc +0.061 ± 0.160 pp (seeds 42/43/44: +0.143, +0.164, -0.123; Δ>0 2/3, Δ<0 1/3); F1 +0.975 ± 1.019 pp (seeds 42/43/44: +1.230, +1.842, -0.147; Δ>0 2/3, Δ<0 1/3); CE +0.008 ± 0.010 (seeds 42/43/44: +0.019, +0.002, +0.002; Δ>0 3/3, Δ<0 0/3) |

Observed pattern: M1's paired changes are small on Movies/Grocery and negative on both Accuracy and Macro-F1 for ele-fashion. M2 has modest mixed changes; receiver-only gating alone is not sufficient for a consistent cross-dataset gain. M3−M2 adds Macro-F1 on Movies and ele-fashion on average, but not Grocery. These are measured paired associations, not capacity-matched causal effects.

## 4–6. Pair behavior and M4 interaction

| Dataset | Contrast | Acc Δ pp | Macro-F1 Δ pp | CE Δ |
|---|---|---:|---:|---:|
| Movies | M3−M0 | -0.010 ± 0.318 pp (seeds 42/43/44: +0.330, -0.300, -0.060; Δ>0 1/3, Δ<0 2/3) | +1.205 ± 1.738 pp (seeds 42/43/44: +3.018, -0.448, +1.047; Δ>0 2/3, Δ<0 1/3) | -0.065 ± 0.052 (seeds 42/43/44: -0.026, -0.125, -0.045; Δ>0 0/3, Δ<0 3/3) |
| Movies | M3−M1 | -0.120 ± 0.167 pp (seeds 42/43/44: +0.030, -0.090, -0.300; Δ>0 1/3, Δ<0 2/3) | +0.947 ± 1.567 pp (seeds 42/43/44: +2.745, -0.129, +0.224; Δ>0 2/3, Δ<0 1/3) | -0.026 ± 0.021 (seeds 42/43/44: -0.008, -0.049, -0.021; Δ>0 0/3, Δ<0 3/3) |
| Movies | M3−M2 | -0.080 ± 0.387 pp (seeds 42/43/44: +0.240, +0.030, -0.510; Δ>0 2/3, Δ<0 1/3) | +1.777 ± 1.172 pp (seeds 42/43/44: +3.035, +0.717, +1.579; Δ>0 3/3, Δ<0 0/3) | -0.040 ± 0.050 (seeds 42/43/44: +0.003, -0.027, -0.095; Δ>0 1/3, Δ<0 2/3) |
| Movies | M4−M3 | -0.110 ± 0.439 pp (seeds 42/43/44: -0.510, -0.180, +0.360; Δ>0 1/3, Δ<0 2/3) | -0.496 ± 0.952 pp (seeds 42/43/44: -1.390, +0.504, -0.603; Δ>0 1/3, Δ<0 2/3) | +0.033 ± 0.061 (seeds 42/43/44: +0.104, -0.005, +0.001; Δ>0 2/3, Δ<0 1/3) |
| Movies | M4−M0 | -0.120 ± 0.393 pp (seeds 42/43/44: -0.180, -0.480, +0.300; Δ>0 1/3, Δ<0 2/3) | +0.709 ± 0.819 pp (seeds 42/43/44: +1.627, +0.056, +0.444; Δ>0 3/3, Δ<0 0/3) | -0.032 ± 0.104 (seeds 42/43/44: +0.078, -0.130, -0.044; Δ>0 1/3, Δ<0 2/3) |
| Grocery | M3−M0 | -0.029 ± 0.128 pp (seeds 42/43/44: -0.176, +0.029, +0.059; Δ>0 2/3, Δ<0 1/3) | -0.103 ± 0.467 pp (seeds 42/43/44: -0.527, +0.397, -0.178; Δ>0 1/3, Δ<0 2/3) | -0.011 ± 0.010 (seeds 42/43/44: -0.021, -0.013, -0.001; Δ>0 0/3, Δ<0 3/3) |
| Grocery | M3−M1 | -0.098 ± 0.017 pp (seeds 42/43/44: -0.088, -0.117, -0.088; Δ>0 0/3, Δ<0 3/3) | -0.257 ± 0.664 pp (seeds 42/43/44: +0.103, +0.150, -1.023; Δ>0 2/3, Δ<0 1/3) | -0.033 ± 0.060 (seeds 42/43/44: -0.005, +0.007, -0.102; Δ>0 1/3, Δ<0 2/3) |
| Grocery | M3−M2 | -0.039 ± 0.212 pp (seeds 42/43/44: -0.176, -0.146, +0.205; Δ>0 1/3, Δ<0 2/3) | -0.510 ± 0.405 pp (seeds 42/43/44: -0.471, -0.126, -0.932; Δ>0 0/3, Δ<0 3/3) | -0.015 ± 0.021 (seeds 42/43/44: -0.037, +0.005, -0.012; Δ>0 1/3, Δ<0 2/3) |
| Grocery | M4−M3 | +0.059 ± 0.211 pp (seeds 42/43/44: +0.000, -0.117, +0.293; Δ>0 1/3, Δ<0 1/3) | -0.107 ± 0.587 pp (seeds 42/43/44: -0.123, -0.686, +0.487; Δ>0 1/3, Δ<0 2/3) | -0.001 ± 0.021 (seeds 42/43/44: +0.017, +0.003, -0.024; Δ>0 2/3, Δ<0 1/3) |
| Grocery | M4−M0 | +0.029 ± 0.282 pp (seeds 42/43/44: -0.176, -0.088, +0.351; Δ>0 1/3, Δ<0 2/3) | -0.210 ± 0.485 pp (seeds 42/43/44: -0.650, -0.288, +0.309; Δ>0 1/3, Δ<0 2/3) | -0.013 ± 0.011 (seeds 42/43/44: -0.004, -0.009, -0.025; Δ>0 0/3, Δ<0 3/3) |
| ele-fashion | M3−M0 | -0.133 ± 0.133 pp (seeds 42/43/44: +0.020, -0.215, -0.205; Δ>0 1/3, Δ<0 2/3) | +0.096 ± 0.588 pp (seeds 42/43/44: +0.504, +0.363, -0.578; Δ>0 2/3, Δ<0 1/3) | -0.020 ± 0.022 (seeds 42/43/44: -0.017, -0.044, +0.000; Δ>0 1/3, Δ<0 2/3) |
| ele-fashion | M3−M1 | +0.126 ± 0.143 pp (seeds 42/43/44: +0.266, +0.133, -0.020; Δ>0 2/3, Δ<0 1/3) | +0.556 ± 0.818 pp (seeds 42/43/44: +0.634, +1.332, -0.297; Δ>0 2/3, Δ<0 1/3) | +0.008 ± 0.025 (seeds 42/43/44: +0.032, -0.018, +0.010; Δ>0 2/3, Δ<0 1/3) |
| ele-fashion | M3−M2 | +0.061 ± 0.160 pp (seeds 42/43/44: +0.143, +0.164, -0.123; Δ>0 2/3, Δ<0 1/3) | +0.975 ± 1.019 pp (seeds 42/43/44: +1.230, +1.842, -0.147; Δ>0 2/3, Δ<0 1/3) | +0.008 ± 0.010 (seeds 42/43/44: +0.019, +0.002, +0.002; Δ>0 3/3, Δ<0 0/3) |
| ele-fashion | M4−M3 | -0.194 ± 0.169 pp (seeds 42/43/44: -0.389, -0.113, -0.082; Δ>0 0/3, Δ<0 3/3) | -0.365 ± 0.722 pp (seeds 42/43/44: -0.257, -1.136, +0.297; Δ>0 1/3, Δ<0 2/3) | -0.016 ± 0.020 (seeds 42/43/44: -0.036, +0.003, -0.015; Δ>0 1/3, Δ<0 2/3) |
| ele-fashion | M4−M0 | -0.327 ± 0.041 pp (seeds 42/43/44: -0.368, -0.327, -0.286; Δ>0 0/3, Δ<0 3/3) | -0.269 ± 0.510 pp (seeds 42/43/44: +0.247, -0.773, -0.281; Δ>0 1/3, Δ<0 2/3) | -0.036 ± 0.020 (seeds 42/43/44: -0.053, -0.041, -0.014; Δ>0 0/3, Δ<0 3/3) |

Pair-path evidence must be read at three levels. Gate standard deviations/correlations and fixed receiver-query roll sensitivity indicate whether the pair-dependent pathway responds to its inputs; trained checkpoint metrics indicate utility; frozen interventions indicate sensitivity with all other trained parameters held fixed. The full channel statistics are in `content_gate_diagnostics.csv`.

| Dataset | Arm | Gate channel SD | Saturated fraction | Receiver-query-roll Gate RMS | Query-roll Δ Acc / F1 pp; flip % |
|---|---|---:|---:|---:|---:|
| Movies | M3 | 0.13963 | 0.066796 | 0.17724 | -0.310 / -0.537; 2.25 |
| Movies | M4 | 0.17109 | 0.063044 | 0.21722 | -0.570 / -0.874; 4.66 |
| Grocery | M3 | 0.11464 | 0 | 0.15123 | -0.166 / -0.113; 0.97 |
| Grocery | M4 | 0.1081 | 0 | 0.14086 | -0.107 / -0.229; 0.92 |
| ele-fashion | M3 | 0.37113 | 0.31396 | 0.38542 | -0.348 / -0.814; 1.17 |
| ele-fashion | M4 | 0.34243 | 0.39455 | 0.35576 | -0.133 / -0.247; 0.59 |

Observed conclusion: M3's edge-conditioned Gate responds to receiver-query changes in every dataset, and frozen query rolls change Validation logits/predictions. M3 has favorable paired Macro-F1 deltas over M2 in Movies and ele-fashion but not Grocery; it is not a consistent cross-dataset improvement over M1/M2 or M0. M4−M3 is mixed on Movies/Grocery and negative on ele-fashion Accuracy and Macro-F1; this does not establish robust cooperation with Intra selection. A nonzero receiver-query roll effect shows frozen output sensitivity, not a retrained ablation or proof of class benefit. M3−M1 and M3−M2 are not strict capacity-matched effects because the gate input channels and active capacity differ. M4−M3 is a trained condition-bundle comparison; the M4 `intra_selection_off` intervention is complementary frozen sensitivity, not a factorial interaction estimate.

At Best Accuracy, M3−M0 is Movies: Acc -0.010 ± 0.318 pp (seeds 42/43/44: +0.330, -0.300, -0.060; Δ>0 1/3, Δ<0 2/3), Macro-F1 +1.205 ± 1.738 pp (seeds 42/43/44: +3.018, -0.448, +1.047; Δ>0 2/3, Δ<0 1/3); Grocery: Acc -0.029 ± 0.128 pp (seeds 42/43/44: -0.176, +0.029, +0.059; Δ>0 2/3, Δ<0 1/3), Macro-F1 -0.103 ± 0.467 pp (seeds 42/43/44: -0.527, +0.397, -0.178; Δ>0 1/3, Δ<0 2/3); ele-fashion: Acc -0.133 ± 0.133 pp (seeds 42/43/44: +0.020, -0.215, -0.205; Δ>0 1/3, Δ<0 2/3), Macro-F1 +0.096 ± 0.588 pp (seeds 42/43/44: +0.504, +0.363, -0.578; Δ>0 2/3, Δ<0 1/3). At the independently selected Best F1 checkpoint, M3−M0 Macro-F1 is Movies: Macro-F1 -0.024 ± 0.884 pp (seeds 42/43/44: +0.785, -0.968, +0.110; Δ>0 2/3, Δ<0 1/3); Grocery: Macro-F1 -0.186 ± 0.103 pp (seeds 42/43/44: -0.199, -0.077, -0.283; Δ>0 0/3, Δ<0 3/3); ele-fashion: Macro-F1 -0.355 ± 0.295 pp (seeds 42/43/44: -0.406, -0.039, -0.621; Δ>0 0/3, Δ<0 3/3). The selected-checkpoint result therefore trades off by dataset and selection rule; it does not support a blanket M3 gain.

## 7. Do degree-1 nodes benefit?

| Dataset | Contrast | Degree-1 Acc Δ pp | Degree-1 Macro-F1 Δ pp | Mean Text/Visual Content RMS |
|---|---|---:|---:|---:|
| Movies | M1−M0 | 1.21 ± 0.20 | 2.60 ± 1.22 | 0.3424/0.05553 |
| Movies | M2−M0 | 1.36 ± 0.44 | 0.61 ± 1.59 | 0.472/0.3085 |
| Movies | M3−M0 | 0.64 ± 0.83 | 3.82 ± 1.91 | 0.499/0.2371 |
| Movies | M4−M0 | 1.37 ± 1.10 | 3.71 ± 3.39 | 0.4835/0.338 |
| Grocery | M1−M0 | 0.19 ± 0.18 | 0.89 ± 0.34 | 0.03323/0.1401 |
| Grocery | M2−M0 | 0.15 ± 0.89 | 1.67 ± 1.04 | 0.1315/0.3538 |
| Grocery | M3−M0 | 0.16 ± 0.63 | 0.48 ± 1.02 | 0.08701/0.3653 |
| Grocery | M4−M0 | -0.15 ± 0.18 | -0.26 ± 0.30 | 0.08894/0.3193 |
| ele-fashion | M1−M0 | -0.41 ± 0.20 | -0.49 ± 0.54 | 0.4587/0.5071 |
| ele-fashion | M2−M0 | -0.23 ± 0.18 | -0.70 ± 0.36 | 0.7504/0.669 |
| ele-fashion | M3−M0 | -0.27 ± 0.14 | -0.07 ± 0.99 | 0.7878/0.6278 |
| ele-fashion | M4−M0 | -0.39 ± 0.15 | -0.45 ± 1.11 | 0.5807/0.5964 |

Degree-group prediction changes remain effects of the trained model as a whole: shared parameters, multi-hop propagation, and direct one-hop Content can all contribute. The subgroup table does not isolate a direct causal effect. M4 `intra_selection_off` frozen Validation effect across datasets: Acc Δ -0.09 ± 0.16 pp; Macro-F1 Δ -0.39 ± 0.42 pp.

Degree ≥2 subgroup deltas versus M0 are shown separately below; the bins are retained rather than combined because subgroup Macro-F1 is not additive:

| Dataset | Arm | Degree 2–3 Acc/F1 Δ pp | Degree 4–7 Acc/F1 Δ pp | Degree ≥8 Acc/F1 Δ pp |
|---|---|---:|---:|---:|
| Movies | M1−M0 | Acc 0.49 ± 0.79 / F1 -0.63 ± 1.47 | Acc -0.95 ± 0.21 / F1 0.46 ± 1.02 | Acc -0.12 ± 0.88 / F1 0.24 ± 2.22 |
| Movies | M2−M0 | Acc 0.31 ± 1.29 / F1 -1.09 ± 0.95 | Acc -0.78 ± 1.14 / F1 -1.56 ± 0.41 | Acc -0.28 ± 0.46 / F1 0.53 ± 1.88 |
| Movies | M3−M0 | Acc -0.27 ± 0.85 / F1 0.21 ± 0.84 | Acc -0.79 ± 0.69 / F1 1.11 ± 1.03 | Acc 0.23 ± 1.52 / F1 1.13 ± 3.49 |
| Movies | M4−M0 | Acc -0.70 ± 0.74 / F1 -0.46 ± 1.88 | Acc -0.79 ± 0.96 / F1 0.32 ± 0.95 | Acc -0.22 ± 0.96 / F1 1.25 ± 3.27 |
| Grocery | M1−M0 | Acc -0.32 ± 0.25 / F1 -0.33 ± 0.84 | Acc 0.63 ± 0.82 / F1 -0.65 ± 2.36 | Acc -0.10 ± 0.10 / F1 1.14 ± 0.78 |
| Grocery | M2−M0 | Acc -0.40 ± 1.07 / F1 -0.80 ± 0.35 | Acc 0.05 ± 1.02 / F1 0.06 ± 1.14 | Acc 0.20 ± 0.34 / F1 1.47 ± 2.04 |
| Grocery | M3−M0 | Acc -0.16 ± 0.36 / F1 -0.71 ± 0.61 | Acc 0.07 ± 1.15 / F1 -0.34 ± 2.64 | Acc -0.14 ± 0.54 / F1 0.54 ± 2.17 |
| Grocery | M4−M0 | Acc 0.48 ± 0.21 / F1 -0.13 ± 0.17 | Acc 0.06 ± 0.92 / F1 -1.17 ± 2.40 | Acc -0.19 ± 0.95 / F1 1.78 ± 0.39 |
| ele-fashion | M1−M0 | Acc -0.27 ± 0.33 / F1 -0.47 ± 1.36 | Acc 0.20 ± 0.15 / F1 -1.91 ± 1.06 | Acc -0.14 ± 0.53 / F1 0.00 ± 2.11 |
| ele-fashion | M2−M0 | Acc -0.35 ± 0.42 / F1 -1.39 ± 1.90 | Acc 0.29 ± 0.27 / F1 0.92 ± 1.06 | Acc -0.24 ± 0.16 / F1 -1.98 ± 0.52 |
| ele-fashion | M3−M0 | Acc 0.09 ± 0.41 / F1 0.43 ± 0.50 | Acc 0.05 ± 0.45 / F1 -1.32 ± 2.10 | Acc -0.41 ± 0.10 / F1 -1.24 ± 0.47 |
| ele-fashion | M4−M0 | Acc -0.33 ± 0.37 / F1 -0.55 ± 1.81 | Acc -0.05 ± 0.37 / F1 1.17 ± 2.63 | Acc -0.41 ± 0.55 / F1 -0.44 ± 1.88 |

## 8. Fashion class 11 and class 1

Best Accuracy and Best F1 checkpoint recall deltas versus M0 (percentage points; three-seed mean ± sample SD):

| Checkpoint | Arm | Class 11 Recall Δ pp | Class 1 Recall Δ pp |
|---|---|---:|---:|
| best_accuracy | M1 | -0.48 ± 4.52 | -0.96 ± 3.62 |
| best_accuracy | M2 | -4.05 ± 4.10 | 1.27 ± 4.80 |
| best_accuracy | M3 | -1.16 ± 4.78 | 2.69 ± 2.53 |
| best_accuracy | M4 | -0.19 ± 3.66 | 1.00 ± 1.46 |
| best_f1 | M1 | -7.90 ± 4.92 | -6.30 ± 1.61 |
| best_f1 | M2 | -6.55 ± 9.50 | -1.11 ± 1.84 |
| best_f1 | M3 | -1.73 ± 12.22 | -3.03 ± 0.33 |
| best_f1 | M4 | -1.25 ± 6.31 | -6.91 ± 3.36 |

Observed classwise pattern: at Best Accuracy, M2 retains a mean Class 11 recall loss while M3/M4 are closer to the internal M0 reference; Class 1 improves modestly for M2/M3/M4. At Best F1, all four arms have lower mean Class 11 and Class 1 recall than M0, with large seed spread for Class 11. Thus the prior Fashion minority-class concern is reduced for some Accuracy-selected arms, not resolved across checkpoint criteria. Full precision/recall/F1/support/predicted counts and confusion matrices are retained for each seed and both checkpoints; no Validation subgroup was used to retune the frozen model.

## 9–11. Expert contribution, Cross redundancy, and Router

| Dataset | Arm | Physical Strength mean | Intrinsic / Graph-correction RMS | Content normalized RMS / Cross-hat RMS | Residual–Cross cosine | Eta-weighted Content RMS / Physical Profile RMS | Router entropy (max normalized expert share) | Content-off Δ Acc / F1 pp; flip % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | M0 | 0.4886 | 3.838 | 0/0.8614 | 0 | 0/0.7419 | 1.268/0.484 | +0.000/+0.000; 0.00 |
| Movies | M1 | 0.4415 | 4.687 | 0.2558/0.8064 | 0.1503 | 0.06717/0.7235 | 1.262/0.463 | -0.100/-0.214; 0.92 |
| Movies | M2 | 0.448 | 4.338 | 0.5632/0.8386 | 0.1164 | 0.1563/0.7296 | 1.288/0.455 | -0.220/-0.129; 3.41 |
| Movies | M3 | 0.4259 | 4.519 | 0.515/0.7994 | 0.1132 | 0.1391/0.7056 | 1.282/0.465 | -0.240/-0.098; 2.30 |
| Movies | M4 | 0.4887 | 3.549 | 0.5742/0.7892 | 0.1006 | 0.1603/0.7302 | 1.257/0.435 | -0.440/-0.625; 4.11 |
| Grocery | M0 | 0.5746 | 2.159 | 0/1.039 | 0 | 0/0.6927 | 0.6813/0.476 | +0.000/+0.000; 0.00 |
| Grocery | M1 | 0.5743 | 1.613 | 0.1217/1.051 | 0.09564 | 0.03249/0.6861 | 0.6729/0.444 | -0.088/-0.081; 0.30 |
| Grocery | M2 | 0.5731 | 1.953 | 0.4143/1.064 | 0.06379 | 0.1098/0.6914 | 0.6801/0.478 | -0.107/-0.131; 1.20 |
| Grocery | M3 | 0.5767 | 1.681 | 0.385/1.049 | 0.05839 | 0.1007/0.6873 | 0.6804/0.491 | -0.049/-0.133; 0.96 |
| Grocery | M4 | 0.5842 | 1.952 | 0.3428/1.026 | 0.05751 | 0.08937/0.6843 | 0.678/0.483 | -0.078/-0.424; 0.93 |
| ele-fashion | M0 | 0.6365 | 1.569 | 0/0.9591 | 0 | 0/0.7744 | 1.131/0.334 | +0.000/+0.000; 0.00 |
| ele-fashion | M1 | 0.5212 | 1.813 | 0.8321/0.8241 | 0.1214 | 0.2529/0.7829 | 1.214/0.339 | -0.095/-0.038; 0.79 |
| ele-fashion | M2 | 0.5526 | 1.55 | 1.352/0.9658 | 0.08997 | 0.441/0.7793 | 1.17/0.326 | -0.218/+0.104; 1.35 |
| ele-fashion | M3 | 0.5614 | 1.602 | 1.322/0.8714 | 0.07491 | 0.435/0.7819 | 1.2/0.331 | -0.290/-0.274; 1.63 |
| ele-fashion | M4 | 0.5978 | 1.625 | 1.142/0.8093 | 0.09656 | 0.3635/0.7697 | 1.19/0.356 | -0.170/-0.339; 1.11 |

Observed interpretation: every active arm has a nonzero first-step final ContentGate gradient and a nonzero learned output-layer update; content-off interventions change logits/predictions after the shared expert/router stack, so the pathway is not completely erased downstream. Residual–Cross cosine is comparatively low in these summaries, which argues against simple vector duplication. Cross Base off generally produces larger logit RMS; metric sensitivity is material but varies by dataset and arm. This does not rule out functional overlap or establish the causal utility of either component. Router entropy/share describe allocation (notably low average entropy on Grocery); there is no direct evidence that the Router wholly overrides Content because frozen content-off comparisons remain sensitive, but those interventions do not retrain or isolate Router-mediated effects.

## 12. Compute, memory, and effective capacity

| Dataset | Arm | Parameters | Mean trainable parameters | Mean wall seconds | Peak allocated GPU GiB |
|---|---|---:|---:|---:|---:|
| Movies | M0 | 1,681,043 | 1,681,043 | 30.5 | 2.68 |
| Movies | M1 | 1,681,043 | 1,681,043 | 29.3 | 2.72 |
| Movies | M2 | 1,681,043 | 1,681,043 | 30.3 | 2.72 |
| Movies | M3 | 1,681,043 | 1,681,043 | 28.5 | 2.67 |
| Movies | M4 | 1,681,043 | 1,681,043 | 43.1 | 2.75 |
| Grocery | M0 | 1,681,043 | 1,681,043 | 27.0 | 2.56 |
| Grocery | M1 | 1,681,043 | 1,681,043 | 31.4 | 2.60 |
| Grocery | M2 | 1,681,043 | 1,681,043 | 28.6 | 2.60 |
| Grocery | M3 | 1,681,043 | 1,681,043 | 27.5 | 2.55 |
| Grocery | M4 | 1,681,043 | 1,681,043 | 36.2 | 2.63 |
| ele-fashion | M0 | 1,416,843 | 1,416,843 | 181.2 | 11.12 |
| ele-fashion | M1 | 1,416,843 | 1,416,843 | 154.8 | 11.28 |
| ele-fashion | M2 | 1,416,843 | 1,416,843 | 154.2 | 11.28 |
| ele-fashion | M3 | 1,416,843 | 1,416,843 | 169.5 | 11.03 |
| ele-fashion | M4 | 1,416,843 | 1,416,843 | 211.5 | 11.44 |

The Pair path uses rank-32 chunked edge computation, so no dense N×N adjacency or full E×256 edge tensor is formed. Runtime and activation memory still vary with graph size and sparse scatter. M0 carries the same declared but inactive module layout for matched initialization; equal parameter count is not equal active capacity. M1/M2/M3 gate inputs differ, so their paired comparison is not a strict capacity-matched causal ablation.

## 13–14. Research judgment and next discriminating experiment

Low-rank message content modeling is supported as an implemented, trainable mechanism. A claim of general classification effectiveness requires consistent paired Validation gains across datasets/checkpoints and tolerable resource cost; the contrast tables above are the measured basis. Accuracy and Macro-F1 can trade off: inspect both checkpoint criteria, and do not describe an F1-only gain as an unqualified improvement. Dropout `.2` and Weight Decay `1e-4` were fixed, so neither main effect nor their interaction with Gate saturation, residual scale, or early stopping is identified.

If the paired gains are small or seed-sensitive, the most discriminating next study would compare true receiver Query matching with a label-free permuted/mismatched Query under matched active parameterization, plus an explicit pair-content versus scalar-edge-weight control. That addresses whether alignment—not just added capacity or extra optimization pathways—accounts for any M3 advantage. This is a proposal only; no new condition is run here.

Run-to-run trajectory variation is summarized by the three-seed SDs and retained per-seed deltas. The Fashion/M0 seed-42 analysis failure was corrected by aligning the analyzer Macro-F1 label set to Train+Validation; the untouched failed record/checkpoints and successful rerun are preserved. Small discrepancies among retries/three seeds remain stochastic trajectory evidence, not proof of a specific CUDA cause. Deterministic algorithms were not enabled for the formal environment because the inherited sparse propagation path may not support every deterministic kernel.

## Fixed optimization factors and failure provenance

AdamW, learning rate `1e-3`, dropout `.2`, weight decay `1e-4`, epochs/early stopping, and remaining training settings are fixed across arms. No independent Dropout or Weight Decay effect or factorial interaction is estimated. Historical pilot scores are reference only; new conclusions use paired internal M0–M4 training.

| Dataset / arm / seed | Failure stage | Recorded symptom | Resolution and evidence boundary |
|---|---|---|---|
| ele-fashion/M0_vs_V2.1_C0/seed 43 | real_data_initialization_preflight | Preflight stopped because a provisional 2e-6 max-logit threshold was exceeded; observed max absolute difference was 2.9802322387695312e-6. | The real-data comparison now uses atol=5e-5, matching the existing V2.1 paired-initialization audit tolerance. The observed 2.98e-6 remains recorded; it is not replaced with zero. |
| Movies/M1/seed 42 | pre_run_gpu_guard | The initial 512 MiB total-memory guard observed 583 MiB on physical GPU 1 after Movies/M0 completed and misclassified the current runner CUDA context as another workload. No M1 tra | The per-run guard now identifies the current PID from nvidia-smi compute-app telemetry, rejects foreign compute PIDs, and bounds unattributed GPU memory. Movies/M0 seed 42 was already fully checkpointed and audited before this stop; model, training, and analys |
| ele-fashion/M0/seed 42 | - | RuntimeError('reloaded best_accuracy validation macro_f1 mismatch: 0.6811467966229473') | Preserved raw failure record; no metric was rewritten. |
| ele-fashion/M0/seed 42 | checkpoint_validation_metric_reproduction | Analyzer recomputed validation Macro-F1 as 0.6811467966229473 while the Best Accuracy checkpoint stored 0.7430692326795789. | Use the NC protocol label set resolved from Train+Validation only via _resolve_nc_eval_labels(include_test=False) for checkpoint and frozen-intervention global metrics. Preserve original failed run and rerun ele-fashion/M0/seed_42 into an attempt directory. |

All raw tracebacks, prior artifacts, and successful rerun provenance remain in `data/failures.json`, the run manifest, and the ignored local output tree. The analyzer correction affects only post-training Validation metric reproduction; it does not alter model weights, optimizer updates, formal training, or Test-label access.
