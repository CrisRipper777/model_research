# M0 — Adaptive Propagation Mechanism Screen

## Decision summary

**Most promising mechanism: default propagation plus one adaptive correction (M0-B), with parameter control recommended.** Extent and functional controls are used by trained models, but M0-A does not outperform uniform propagation; B’s same-checkpoint function-off intervention consistently lowers validation accuracy and Macro-F1, while B’s paired advantage over A is small. M0-C does not produce a consistent benefit over B, and replacing edge-conditioned basis weights with uniform weights barely changes mean validation results. This is a mechanism screen, not a leaderboard result or a claim that propagation-function heterogeneity has been newly established.

| Mechanism | Assessment | Main evidence |
|---|---|---|
| Extent adaptation | **MIXED** | A is effectively tied with UNI on held-out validation metrics. Extent-off and edge-control shuffle generally worsen metrics, but within-target gate variation is weak for many Movies/Grocery targets and effects vary by dataset. |
| Single functional correction | **SUPPORTED** | B function-off reduces accuracy and Macro-F1 in 9/9 paired runs and increases CE in 7/9. Edge-control shuffle also has a small, directionally consistent cost. B’s paired gain over A remains modest and its mean CE is slightly worse. |
| Multi-basis functional modulation | **NOT SUPPORTED** | C is not consistently better than B; basis-uniform has almost no average effect. Bases remain distinct and pi varies, but this complexity does not yield reliable incremental task value. |

**Recommendation:** choose **default + single adaptive correction** for the next reviewed stage. Mark **PARAMETER_CONTROL_RECOMMENDED**: B has 16,514 more trainable parameters than A and its B−A performance advantage is small. A parameter-matched A-wide control is a sensible later ablation; it was not added to M0.

## Protocol and provenance

- Source: branch `exp/p13_joint_readout_operator_probe`, SHA `83487eada5ca4909a83140a3de498514f2d6d7d8`; experiment branch `exp/m0_adaptive_propagation_screen`.
- 3 datasets × 3 seeds (42–44) × 5 variants = 45 final unique runs, using `unified_full_graph_nc_v1`. Full graph; train labels only; best validation accuracy selection; no test split fields attached, no test metrics, no link prediction, no dataset-specific tuning.
- One propagation step. Directed non-self physical edges are preserved; messages use the original non-self incoming degree denominator. The leave-one-out local-context scalar is detached. No extra losses were added.
- All five variants use end-to-end learned Text/Visual projections and the same prior-retaining fusion. A/B/C share relation-state construction; A adds extent gates, B adds one low-rank correction, C replaces it with four rank-8 bases.
- Device use: first 40 unique runs on GPU 0, final 5 resumed on GPU 1 after GPU 0 was occupied. GPU 0 shared an unrelated process during part of the campaign.
- One initial Movies/42/SEM attempt stopped in the diagnostic summarizer because the topology-free SEM variant had an empty gate tensor. The empty-control path was fixed and the resumable campaign completed that record. Final status is 45/45; no final run failed.
- Accumulated model training time was 924.3 s; validation interventions took 658.1 s; total recorded compute time was 1,582.4 s (26 min 22 s), excluding process start-up and report generation. Maximum formal-run allocated GPU memory was 8.18 GiB. No OOM or non-finite training run was observed.

### Correctness and fairness checks

- Text-then-Visual slicing, directed physical support, non-self degree, target/source ordering, leave-one-out context, fixed denominator, identity cases, tied controls, neighborhood-shuffle tuples, gradients, and no-test-access behavior are covered by 19 tests: **19 passed** (one upstream PyG deprecation warning).
- CPU chunked/single-chunk equivalence passed at 1e−6; CUDA forward/inference maximum observed absolute difference was 1.43e−6, so the smoke comparison uses a 1e−5 tolerance for CUDA index-add roundoff. There was no deviation from the frozen one-step architecture or its hyperparameters.
- Movies/42 five-variant GPU smoke passed. Adaptive gates, correction controls, and C routing/basis gradients were finite and nonzero. Initial correction/default norm ratios were B mean 0.00428 (median 0.00434) and C mean 0.00116 (median 0.00112).
- A/B/C common initialization was bitwise identical for all 9 dataset×seed combinations. The B and C correction bases each contain exactly 16,384 parameters; C adds only a 520-parameter basis head beyond B. Full audits are in `data/common_initialization_audit.csv` and `data/parameter_summary.csv`.

## Validation performance

Each cell is mean ± population standard deviation over the three seeds. Accuracy and Macro-F1 are percentages; CE is on the native scale. The ALL rows equally average the nine dataset×seed runs and should be read descriptively because the datasets have different scales. Trainable parameter counts (including the NC classifier) are Movies/Grocery: SEM 627,604; UNI 660,372; A 703,046; B 719,560; C 720,080. Ele-fashion: SEM 495,500; UNI 528,268; A 570,942; B 587,456; C 587,976. Per-run parameter, best-epoch, runtime, and memory records are in `data/performance_by_run.csv`; component counts are in `data/parameter_summary.csv`.

### Movies

| Variant | Accuracy (%) | Macro-F1 (%) | CE |
| --- | --- | --- | --- |
| SEM | 52.36 ± 0.29 | 37.98 ± 3.48 | 1.4935 ± 0.0171 |
| UNI | 55.42 ± 0.34 | 45.87 ± 0.80 | 1.4193 ± 0.0273 |
| M0-A | 55.32 ± 0.15 | 45.02 ± 1.66 | 1.4353 ± 0.0610 |
| M0-B | 55.56 ± 0.30 | 45.33 ± 1.82 | 1.4137 ± 0.0302 |
| M0-C | 55.39 ± 0.18 | 44.10 ± 1.64 | 1.3783 ± 0.0097 |

### Grocery

| Variant | Accuracy (%) | Macro-F1 (%) | CE |
| --- | --- | --- | --- |
| SEM | 78.24 ± 0.16 | 69.23 ± 0.41 | 0.8595 ± 0.0208 |
| UNI | 82.25 ± 0.23 | 73.41 ± 1.07 | 0.7085 ± 0.0293 |
| M0-A | 82.06 ± 0.14 | 73.54 ± 1.23 | 0.6996 ± 0.0226 |
| M0-B | 82.25 ± 0.06 | 74.13 ± 0.77 | 0.7173 ± 0.0260 |
| M0-C | 82.24 ± 0.12 | 74.00 ± 0.76 | 0.7263 ± 0.0489 |

### ele-fashion

| Variant | Accuracy (%) | Macro-F1 (%) | CE |
| --- | --- | --- | --- |
| SEM | 86.94 ± 0.11 | 66.73 ± 0.09 | 0.4187 ± 0.0026 |
| UNI | 87.41 ± 0.08 | 68.66 ± 0.80 | 0.4162 ± 0.0032 |
| M0-A | 87.42 ± 0.07 | 68.65 ± 0.31 | 0.4215 ± 0.0033 |
| M0-B | 87.34 ± 0.12 | 68.68 ± 0.83 | 0.4295 ± 0.0310 |
| M0-C | 87.44 ± 0.11 | 68.96 ± 0.28 | 0.4208 ± 0.0038 |

### Equal-weight aggregate across nine runs

| Variant | Accuracy (%) | Macro-F1 (%) | CE |
| --- | --- | --- | --- |
| SEM | 72.51 ± 14.69 | 57.98 ± 14.32 | 0.9239 ± 0.4414 |
| UNI | 75.03 ± 14.03 | 62.64 ± 12.05 | 0.8480 ± 0.4219 |
| M0-A | 74.93 ± 14.04 | 62.40 ± 12.51 | 0.8522 ± 0.4294 |
| M0-B | 75.05 ± 13.94 | 62.71 ± 12.55 | 0.8535 ± 0.4142 |
| M0-C | 75.02 ± 14.05 | 62.35 ± 13.11 | 0.8418 ± 0.4004 |

## Paired same-seed deltas

Positive values are left minus right. Accuracy and Macro-F1 deltas are percentage points; CE stays on its native scale. Values are mean ± population SD over three paired seeds per dataset and nine pairs in ALL.

### Movies

| Paired delta | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
| --- | --- | --- | --- |
| UNI-SEM | +3.06 ± 0.44 | +7.89 ± 2.81 | -0.0742 ± 0.0405 |
| A-UNI | -0.10 ± 0.22 | -0.85 ± 1.25 | +0.0160 ± 0.0800 |
| B-A | +0.24 ± 0.15 | +0.31 ± 1.10 | -0.0216 ± 0.0362 |
| C-B | -0.17 ± 0.12 | -1.23 ± 1.26 | -0.0354 ± 0.0274 |
| C-A | +0.07 ± 0.04 | -0.91 ± 0.17 | -0.0571 ± 0.0537 |

### Grocery

| Paired delta | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
| --- | --- | --- | --- |
| UNI-SEM | +4.01 ± 0.13 | +4.18 ± 1.07 | -0.1510 ± 0.0266 |
| A-UNI | -0.20 ± 0.10 | +0.13 ± 1.13 | -0.0088 ± 0.0067 |
| B-A | +0.19 ± 0.10 | +0.59 ± 0.51 | +0.0177 ± 0.0036 |
| C-B | -0.00 ± 0.06 | -0.13 ± 0.39 | +0.0090 ± 0.0285 |
| C-A | +0.19 ± 0.07 | +0.46 ± 0.54 | +0.0267 ± 0.0302 |

### ele-fashion

| Paired delta | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
| --- | --- | --- | --- |
| UNI-SEM | +0.48 ± 0.16 | +1.93 ± 0.73 | -0.0026 ± 0.0022 |
| A-UNI | +0.01 ± 0.11 | -0.01 ± 0.99 | +0.0053 ± 0.0018 |
| B-A | -0.08 ± 0.09 | +0.03 ± 0.78 | +0.0080 ± 0.0332 |
| C-B | +0.10 ± 0.04 | +0.28 ± 0.76 | -0.0087 ± 0.0347 |
| C-A | +0.02 ± 0.06 | +0.31 ± 0.55 | -0.0007 ± 0.0036 |

### ALL

| Paired delta | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
| --- | --- | --- | --- |
| UNI-SEM | +2.52 ± 1.52 | +4.67 ± 3.04 | -0.0759 ± 0.0668 |
| A-UNI | -0.10 ± 0.17 | -0.24 ± 1.21 | +0.0042 ± 0.0475 |
| B-A | +0.12 ± 0.18 | +0.31 ± 0.86 | +0.0014 ± 0.0330 |
| C-B | -0.02 ± 0.14 | -0.36 ± 1.08 | -0.0117 ± 0.0355 |
| C-A | +0.09 ± 0.09 | -0.05 ± 0.77 | -0.0103 ± 0.0499 |

UNI−SEM is positive for accuracy and Macro-F1 and lower for CE in all 9 paired runs, with positive per-dataset mean differences: +3.06 / +4.01 / +0.48 accuracy points and +7.89 / +4.18 / +1.94 Macro-F1 points for Movies / Grocery / ele-fashion. This is a consistent topology gain under the new end-to-end backbone.

A−UNI does not show a corresponding gain: pooled paired means are −0.10 accuracy points, −0.24 Macro-F1 points, and +0.0042 CE; A beats UNI on accuracy in only 3/9 and on Macro-F1 in 5/9. B−A is small (+0.12 accuracy points and +0.31 Macro-F1 points, with CE +0.00135), and C−B is mixed (−0.03 accuracy points, −0.36 Macro-F1 points, CE −0.0117). No significance testing was done.

## Extent mechanism: M0-A

The following summarizes the three seed-level validation-edge diagnostics. Gate statistics are by modality; within-target statistics summarize per-target edge standard deviations over validation targets with non-self in-degree ≥5. Full per-run mean/std/quantile/fraction fields are in `data/gate_diagnostics.csv` and `data/within_node_control_variation.csv`.

| Dataset | Modality | g mean ± edge SD | q10 / median / q90 | frac <.1 / >.9 | mean |gT−gV| | within-target std: median [q25,q75] |
| --- | --- | --- | --- | --- | --- | --- |
| Grocery | Text | 0.951 ± 0.122 | 0.956 / 0.986 / 0.987 | 0.000 / 0.925 | 0.016 | 0.005 [0.000, 0.115] |
| Grocery | Visual | 0.943 ± 0.127 | 0.872 / 0.985 / 0.985 | 0.000 / 0.905 | 0.016 | 0.017 [0.001, 0.132] |
| Movies | Text | 0.937 ± 0.121 | 0.761 / 0.982 / 0.984 | 0.000 / 0.881 | 0.051 | 0.029 [0.002, 0.092] |
| Movies | Visual | 0.916 ± 0.181 | 0.586 / 0.989 / 0.990 | 0.000 / 0.853 | 0.051 | 0.045 [0.001, 0.151] |
| ele-fashion | Text | 0.800 ± 0.331 | 0.207 / 0.990 / 0.998 | 0.120 / 0.726 | 0.133 | 0.095 [0.005, 0.262] |
| ele-fashion | Visual | 0.909 ± 0.165 | 0.709 / 0.981 / 0.990 | 0.000 / 0.793 | 0.133 | 0.042 [0.005, 0.114] |

M0-A gates are high on average, with most edges above 0.9, especially in Grocery. Within-target edge variation is uneven: median target-level std is 0.005/0.017 for Grocery Text/Visual, 0.029/0.045 for Movies, and 0.095/0.042 for ele-fashion. The upper quartiles reach 0.11–0.26, so a subset of targets uses edge-specific extent strongly, while many targets—particularly in Grocery—remain close to a node-level scalar.

Mean |gT−gV| is 0.016 (Grocery), 0.051 (Movies), and 0.133 (ele-fashion). Thus modality divergence is small in Grocery and clear in ele-fashion. The control is not uniformly modality-specific across datasets.

Validation interventions (delta = intervention minus normal inference; positive CE and negative accuracy/F1 indicate degradation) pooled over 9 model runs:

| Variant | Intervention | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE | Runs / repeat records |
| --- | --- | --- | --- | --- | --- |
| M0-A | extent off | -0.673 ± 0.753 | -0.771 ± 1.544 | +0.02242 ± 0.02320 | 9 / 9 |
| M0-A | edge control shuffle | -0.109 ± 0.091 | -0.062 ± 0.162 | +0.00183 ± 0.00175 | 9 / 45 |
| M0-A | modality tied | -0.087 ± 0.095 | -0.307 ± 0.325 | +0.00227 ± 0.00604 | 9 / 9 |
| M0-B | extent off | -0.580 ± 0.600 | -0.383 ± 0.959 | +0.01017 ± 0.00989 | 9 / 9 |
| M0-B | function off | -0.443 ± 0.340 | -0.856 ± 0.492 | +0.00934 ± 0.01086 | 9 / 9 |
| M0-B | edge control shuffle | -0.100 ± 0.114 | -0.116 ± 0.149 | +0.00147 ± 0.00125 | 9 / 45 |
| M0-B | modality tied | -0.102 ± 0.094 | -0.057 ± 0.170 | -0.00108 ± 0.00464 | 9 / 9 |
| M0-C | extent off | -0.652 ± 0.415 | -0.718 ± 0.766 | +0.01656 ± 0.01100 | 9 / 9 |
| M0-C | function off | -0.184 ± 0.212 | -0.370 ± 0.551 | +0.00264 ± 0.00458 | 9 / 9 |
| M0-C | edge control shuffle | -0.130 ± 0.145 | -0.078 ± 0.204 | +0.00183 ± 0.00138 | 9 / 45 |
| M0-C | modality tied | -0.102 ± 0.128 | -0.207 ± 0.446 | +0.00057 ± 0.00393 | 9 / 9 |
| M0-C | basis uniform | -0.069 ± 0.119 | -0.111 ± 0.190 | -0.00008 ± 0.00303 | 9 / 9 |

For A, forcing g=1 raises CE by 0.02242 on average and lowers accuracy by 0.673 points; CE is worse in 8/9 runs, accuracy in 7/9, and Macro-F1 in 5/9. Edge-control shuffle raises CE by 0.00183, lowers accuracy by 0.109 points, and lowers Macro-F1 by 0.062 points; directions are adverse in 8/9, 8/9, and 7/9 runs. Across five deterministic shuffles per run, mean within-run CE standard deviation is 0.00103. The matching signal exists but its effect is small. Modality tying is dataset-dependent: for A it raises CE by 0.00981 on ele-fashion but lowers it by 0.00311 on Movies, with near-zero effect on Grocery. Extent adaptation is therefore **MIXED**, not a performance win over UNI.

## Single correction: M0-B

| Dataset | Variant | Modality | c mean ± edge SD | c q10 / q50 / q90 | frac c<.1 / >.9 | correction/default ratio median [q25,q75], q90 | mean |cT−cV| |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Movies | M0-B | Text | 0.361 ± 0.138 | 0.099 / 0.428 / 0.466 | 0.103 / 0.000 | 0.097 [0.060, 0.125], q90 0.146 | 0.180 |
| Movies | M0-B | Visual | 0.524 ± 0.208 | 0.111 / 0.629 / 0.662 | 0.111 / 0.000 | 0.148 [0.107, 0.182], q90 0.213 | 0.180 |
| Movies | M0-C | Text | 0.361 ± 0.090 | 0.280 / 0.392 / 0.411 | 0.054 / 0.000 | 0.030 [0.021, 0.038], q90 0.044 | 0.053 |
| Movies | M0-C | Visual | 0.361 ± 0.109 | 0.202 / 0.405 / 0.423 | 0.099 / 0.000 | 0.031 [0.025, 0.036], q90 0.041 | 0.053 |
| Grocery | M0-B | Text | 0.555 ± 0.182 | 0.209 / 0.628 / 0.645 | 0.093 / 0.000 | 0.099 [0.079, 0.119], q90 0.137 | 0.095 |
| Grocery | M0-B | Visual | 0.534 ± 0.191 | 0.153 / 0.623 / 0.632 | 0.114 / 0.000 | 0.109 [0.081, 0.137], q90 0.159 | 0.095 |
| Grocery | M0-C | Text | 0.577 ± 0.203 | 0.211 / 0.667 / 0.685 | 0.091 / 0.201 | 0.044 [0.031, 0.057], q90 0.069 | 0.138 |
| Grocery | M0-C | Visual | 0.443 ± 0.174 | 0.140 / 0.534 / 0.552 | 0.114 / 0.000 | 0.040 [0.019, 0.055], q90 0.068 | 0.138 |
| ele-fashion | M0-B | Text | 0.632 ± 0.334 | 0.100 / 0.698 / 0.926 | 0.224 / 0.308 | 0.285 [0.153, 0.429], q90 0.525 | 0.171 |
| ele-fashion | M0-B | Visual | 0.517 ± 0.353 | 0.009 / 0.524 / 0.912 | 0.266 / 0.318 | 0.177 [0.091, 0.294], q90 0.394 | 0.171 |
| ele-fashion | M0-C | Text | 0.620 ± 0.384 | 0.002 / 0.821 / 0.953 | 0.248 / 0.430 | 0.198 [0.041, 0.327], q90 0.387 | 0.119 |
| ele-fashion | M0-C | Visual | 0.609 ± 0.364 | 0.011 / 0.742 / 0.950 | 0.227 / 0.383 | 0.176 [0.068, 0.289], q90 0.366 | 0.119 |

B’s correction/default message norm ratio has nonzero medians: approximately 0.10–0.15 on Movies/Grocery and 0.18–0.29 on ele-fashion. The correction is not dominating the default transform; it is a moderate deviation, larger on ele-fashion. The per-run c distributions and Text/Visual difference are shown in `data/correction_diagnostics.csv`.

With B’s trained checkpoint fixed, setting c=0 lowers accuracy by 0.443 points and Macro-F1 by 0.856 points on average; accuracy and Macro-F1 fall in 9/9 runs, while CE rises in 7/9. Edge-control shuffle lowers Macro-F1 in 8/9 and raises CE in 9/9, with mean CE increase 0.00147 and mean five-shuffle within-run CE SD 0.00109. Modality tying has mixed near-zero pooled effects (CE −0.00108; accuracy −0.102 points; Macro-F1 −0.057 points) and is not consistently harmful. The same-checkpoint interventions support that the learned correction is used, even though the B−A paired performance improvement is modest and does not improve mean CE.

## Continuous basis mechanism: M0-C

| Dataset | Modality | mean pi over 4 bases | mean edgewise SD(pi_k) | normalized H | effective bases | mean L1(piT,piV) | basis |cos| offdiag mean / max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Grocery | Text | [0.29, 0.13, 0.17, 0.41] | 0.100 | 0.647 | 2.64 | 0.983 | 0.052 / 0.073 |
| Grocery | Visual | [0.23, 0.17, 0.36, 0.23] | 0.097 | 0.772 | 3.03 | 0.983 | 0.064 / 0.095 |
| Movies | Text | [0.11, 0.40, 0.09, 0.40] | 0.079 | 0.691 | 2.66 | 1.074 | 0.044 / 0.071 |
| Movies | Visual | [0.39, 0.15, 0.27, 0.20] | 0.072 | 0.761 | 3.00 | 1.074 | 0.063 / 0.085 |
| ele-fashion | Text | [0.09, 0.31, 0.11, 0.48] | 0.209 | 0.351 | 1.83 | 1.623 | 0.062 / 0.116 |
| ele-fashion | Visual | [0.36, 0.09, 0.27, 0.29] | 0.220 | 0.355 | 1.80 | 1.623 | 0.059 / 0.105 |

The C pi coefficients are not uniformly near-uniform. Effective basis counts average about 2.64–3.03 on Movies/Grocery and 1.80–1.83 on ele-fashion; mean edgewise coefficient SD is 0.07–0.10 on Movies/Grocery and about 0.21–0.22 on ele-fashion. Text/Visual coefficient distributions differ, with aligned-edge mean L1 distance 0.98 (Grocery), 1.07 (Movies), and 1.62 (ele-fashion). Basis usage varies by seed, particularly in ele-fashion, so the mean pi vector is descriptive rather than evidence of a stable named basis preference.

The four learned bases do not collapse: across dataset and modality summaries, mean absolute off-diagonal cosine is 0.044–0.064 and maximum is 0.071–0.116. However, distinct basis matrices and nonuniform routing do not establish useful multi-basis task value. The basis-uniform intervention changes pooled accuracy by −0.069 points, Macro-F1 by −0.111 points, and CE by −0.00008—effectively negligible—with mixed run-level outcomes. C−B is also mixed, and C requires 520 more parameters than B. Function-off in C has a smaller pooled effect than B’s, while edge-control shuffle has a small adverse direction in 9/9 CE, 8/9 accuracy, and 7/9 Macro-F1 runs. Multi-basis modulation is therefore **NOT SUPPORTED** as an incremental mechanism in this screen.

## Answers to the frozen questions

1. **UNI vs SEM:** Yes. All 9 paired runs improve accuracy and Macro-F1 and lower CE; each dataset has a positive mean gain.
2. **M0-A vs UNI:** No consistent performance gain. Gates vary, but extent-off and edge shuffle reveal only modest and dataset-dependent utility.
3. **Edge-specific gate or node/global scalar?** Mixed. Median within-target spread is close to zero in many Movies/Grocery targets; upper quartiles and ele-fashion medians show substantial edge-specific variation for a subset.
4. **Text/Visual g and modality tying:** |gT−gV| ranges by dataset from 0.016 to 0.133. Tying shows the clearest cost for A on ele-fashion, but effects across all variants and datasets are mixed.
5. **Does B add task value beyond extent?** B−A is a small, mixed metric difference, but same-checkpoint function-off harms B consistently on accuracy and Macro-F1. The correction is used; its incremental performance gain over A is not established.
6. **Does function-off hurt B consistently?** Accuracy and Macro-F1 decline in 9/9 runs; CE rises in 7/9.
7. **Does c differentiate, and is correction size reasonable?** Yes. c and its Text/Visual distributions differ by dataset; correction/default median ratios are roughly 0.10–0.29, nonzero and below the default magnitude.
8. **Does C add value over B?** Not consistently. C−B improves accuracy in 5/9, Macro-F1 in 4/9, and CE in 5/9; pooled deltas are small/mixed.
9. **Is pi uniform or edge-dependent?** It is nonuniform and edge-varying, with effective basis counts from about 1.8 to 3.0. Variation is seed- and modality-dependent.
10. **Do the learned bases collapse?** No. Pairwise cosine summaries are low.
11. **Does basis-uniform matter?** Little on average; pooled deltas are near zero and run directions are mixed.
12. **Does edge-control shuffle matter?** Yes, modestly. CE worsens in 8–9/9 runs and accuracy/Macro-F1 usually fall; this supports some edge-control matching value without implying a large effect.
13. **Does modality tying support modality-resolved controls?** Mixed. The clearest effect is on ele-fashion; pooled effects vary by mechanism and metric.
14. **Could B/C gains reflect parameter count?** Yes. B adds 16,514 parameters to A (about 2.35%); the B−A metric difference is small. **PARAMETER_CONTROL_RECOMMENDED** for a future A-wide comparison.
15. **Which mechanism is most promising?** **Default + single adaptive correction.** It best fits active function-off evidence and the lack of incremental C value, with the parameter caveat above.
16. **What next?** After human review, evaluate a parameter-matched A-wide control beside a single-correction model and keep analysis focused on one-step edge-local controls. Do not infer a need for multi-basis or multi-hop propagation from this screen.

## Artifacts

- Report: `research/m0_adaptive_propagation_screen/report.md`
- Run manifest: `research/m0_adaptive_propagation_screen/run_manifest.json`
- CSV tables: `research/m0_adaptive_propagation_screen/data/`
- Figures: `research/m0_adaptive_propagation_screen/figures/`
- Large checkpoints and raw per-run records: ignored `outputs/m0_adaptive_propagation_screen/`
