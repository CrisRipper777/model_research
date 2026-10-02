# R1 — Baseline-Preserving Residual Response Audit

## Scope and protocol

This validation-only screen completed 27/27 planned full-graph node-classification runs on Movies, Grocery, and ele-fashion (seeds 42–44), with 0 failures and 0 retries. The three variants are Smooth, residual Generic, and residual Protected. Test split indices were not attached to model runs; labels outside train/validation were masked before training, and no test metrics or link-prediction task were run. Values are descriptive; the nine matched dataset-seed pairs are not treated as IID significance-test replicates.

The fixed R0 schedule was used: full-graph NC, AdamW (lr 1e−3, weight decay 1e−4), up to 300 epochs, early-stop patience 30 after minimum epoch 30, gradient clip 1.0, best checkpoint by validation Accuracy (minimum improvement 1e−4).

Source: `exp/r0_mature_response_expert_prototype` at `d60db509e15e830a09a691665dcbe72435137477`. Experiment branch: `exp/r1_baseline_preserving_residual_response`. Device: `cuda:1`. Wall time: 460.0 s; summed training time: 432.3 s; maximum allocated GPU memory: 8.79 GiB.

## Performance (mean ± population SD across three seeds)

Accuracy and Macro-F1 are percentages; CE is raw cross-entropy.

| Dataset | Variant | Accuracy (%) | Macro-F1 (%) | CE |
|---|---|---:|---:|---:|
| Movies | smooth_base | 55.209 ± 0.296 | 44.426 ± 0.566 | 1.4253 ± 0.0568 |
| Movies | residual_generic | 54.699 ± 0.422 | 40.766 ± 1.138 | 1.3858 ± 0.0086 |
| Movies | residual_protected | 54.239 ± 0.413 | 39.949 ± 1.661 | 1.4066 ± 0.0112 |
| Grocery | smooth_base | 81.855 ± 0.325 | 72.980 ± 1.305 | 0.6983 ± 0.0092 |
| Grocery | residual_generic | 81.181 ± 0.425 | 71.729 ± 1.239 | 0.7091 ± 0.0249 |
| Grocery | residual_protected | 81.308 ± 0.409 | 71.656 ± 1.566 | 0.7300 ± 0.0209 |
| ele-fashion | smooth_base | 87.372 ± 0.084 | 68.360 ± 0.228 | 0.4087 ± 0.0087 |
| ele-fashion | residual_generic | 87.402 ± 0.104 | 68.913 ± 0.416 | 0.4364 ± 0.0047 |
| ele-fashion | residual_protected | 87.211 ± 0.089 | 67.550 ± 0.732 | 0.4161 ± 0.0098 |

## Paired within-dataset, within-seed changes

Accuracy and Macro-F1 deltas are percentage points. Negative CE change is favorable. The pooled row is descriptive only.

| Contrast | Dataset | Metric | Mean Δ | Population SD | Favorable / n | + / − / tie |
|---|---|---|---:|---:|---:|---:|
| residual_generic-Smooth | Grocery | accuracy | -0.6735 | 0.1042 | 0 / 3 | 0 / 3 / 0 |
| residual_generic-Smooth | Grocery | ce | 0.0107 | 0.0165 | 1 / 3 | 2 / 1 / 0 |
| residual_generic-Smooth | Grocery | macro_f1 | -1.2509 | 0.2244 | 0 / 3 | 0 / 3 / 0 |
| residual_generic-Smooth | Movies | accuracy | -0.5099 | 0.1273 | 0 / 3 | 0 / 3 / 0 |
| residual_generic-Smooth | Movies | ce | -0.0395 | 0.0483 | 2 / 3 | 1 / 2 / 0 |
| residual_generic-Smooth | Movies | macro_f1 | -3.6603 | 1.5469 | 0 / 3 | 0 / 3 / 0 |
| residual_generic-Smooth | ele-fashion | accuracy | 0.0307 | 0.1880 | 2 / 3 | 2 / 1 / 0 |
| residual_generic-Smooth | ele-fashion | ce | 0.0277 | 0.0103 | 0 / 3 | 3 / 0 / 0 |
| residual_generic-Smooth | ele-fashion | macro_f1 | 0.5529 | 0.6402 | 2 / 3 | 2 / 1 / 0 |
| residual_protected-Smooth | Grocery | accuracy | -0.5466 | 0.0966 | 0 / 3 | 0 / 3 / 0 |
| residual_protected-Smooth | Grocery | ce | 0.0317 | 0.0130 | 0 / 3 | 3 / 0 / 0 |
| residual_protected-Smooth | Grocery | macro_f1 | -1.3245 | 0.4025 | 0 / 3 | 0 / 3 / 0 |
| residual_protected-Smooth | Movies | accuracy | -0.9698 | 0.1208 | 0 / 3 | 0 / 3 / 0 |
| residual_protected-Smooth | Movies | ce | -0.0186 | 0.0647 | 1 / 3 | 2 / 1 / 0 |
| residual_protected-Smooth | Movies | macro_f1 | -4.4770 | 2.0985 | 0 / 3 | 0 / 3 / 0 |
| residual_protected-Smooth | ele-fashion | accuracy | -0.1602 | 0.1712 | 1 / 3 | 1 / 2 / 0 |
| residual_protected-Smooth | ele-fashion | ce | 0.0074 | 0.0027 | 0 / 3 | 3 / 0 / 0 |
| residual_protected-Smooth | ele-fashion | macro_f1 | -0.8098 | 0.6916 | 1 / 3 | 1 / 2 / 0 |
| residual_protected-residual_generic | Grocery | accuracy | 0.1269 | 0.0276 | 3 / 3 | 3 / 0 / 0 |
| residual_protected-residual_generic | Grocery | ce | 0.0210 | 0.0169 | 1 / 3 | 2 / 1 / 0 |
| residual_protected-residual_generic | Grocery | macro_f1 | -0.0735 | 0.3333 | 1 / 3 | 1 / 2 / 0 |
| residual_protected-residual_generic | Movies | accuracy | -0.4599 | 0.0141 | 0 / 3 | 0 / 3 / 0 |
| residual_protected-residual_generic | Movies | ce | 0.0208 | 0.0181 | 0 / 3 | 3 / 0 / 0 |
| residual_protected-residual_generic | Movies | macro_f1 | -0.8167 | 2.1417 | 2 / 3 | 2 / 1 / 0 |
| residual_protected-residual_generic | ele-fashion | accuracy | -0.1909 | 0.0193 | 0 / 3 | 0 / 3 / 0 |
| residual_protected-residual_generic | ele-fashion | ce | -0.0203 | 0.0102 | 3 / 3 | 0 / 3 / 0 |
| residual_protected-residual_generic | ele-fashion | macro_f1 | -1.3627 | 0.8731 | 0 / 3 | 0 / 3 / 0 |
| residual_generic-Smooth | ALL_POOLED_DESCRIPTIVE | accuracy | -0.3842 | 0.3337 | 2 / 9 | 2 / 7 / 0 |
| residual_generic-Smooth | ALL_POOLED_DESCRIPTIVE | ce | -0.0003 | 0.0414 | 3 / 9 | 6 / 3 / 0 |
| residual_generic-Smooth | ALL_POOLED_DESCRIPTIVE | macro_f1 | -1.4528 | 1.9824 | 2 / 9 | 2 / 7 / 0 |
| residual_protected-Smooth | ALL_POOLED_DESCRIPTIVE | accuracy | -0.5589 | 0.3565 | 1 / 9 | 1 / 8 / 0 |
| residual_protected-Smooth | ALL_POOLED_DESCRIPTIVE | ce | 0.0068 | 0.0433 | 1 / 9 | 8 / 1 / 0 |
| residual_protected-Smooth | ALL_POOLED_DESCRIPTIVE | macro_f1 | -2.2038 | 2.0759 | 1 / 9 | 1 / 8 / 0 |
| residual_protected-residual_generic | ALL_POOLED_DESCRIPTIVE | accuracy | -0.1746 | 0.2408 | 3 / 9 | 3 / 6 / 0 |
| residual_protected-residual_generic | ALL_POOLED_DESCRIPTIVE | ce | 0.0072 | 0.0248 | 4 / 9 | 5 / 4 / 0 |
| residual_protected-residual_generic | ALL_POOLED_DESCRIPTIVE | macro_f1 | -0.7510 | 1.4489 | 3 / 9 | 3 / 6 / 0 |

## Correctness and mechanism audit

- Smooth compatibility: `passed`. The R1 Smooth path was mapped separately from M0 UNI, N1 SmoothOnly, and R0 SmoothBase; maximum observed error was `2.6226043701171875e-06` against rtol 1e−6 / atol 1e−5.
- R0 response branch regression: `passed`; maximum absolute error `1.2159347534179688e-05` (GPU tolerance 2e−5 to cover measured scatter-order roundoff after LayerNorm; CPU tolerance 1e−6).
- Initial identity: `passed`; maximum error `0.0` across both train/eval modes and both modalities.
- Transform ownership: `{'smooth_vs_low_distinct': True, 'smooth_vs_high_distinct': True, 'low_vs_high_distinct': True, 'all_transforms_have_no_bias': True}`. Smooth W_S, bank W_L, and bank W_H are separate bias-free 128×128 transforms; W_C is bias free and zero initialized.
- Initialization/capacity fairness: all three variants were bitwise equal at each dataset-seed initialization, classifier states were identical, and each had exactly 1,706,004 model parameters plus 2,580 classifier parameters.
- Gradient bootstrap: 2/2 residual variants passed. Task CE produced a finite nonzero W_C gradient while response-branch task gradients remained zero at step 1; after one optimizer update, step-2 task gradients reached the bank, experts, router, and active composer. Orthogonality gradients were isolated and reported separately.
- Best-checkpoint W_C largest modality Frobenius norm: 2.14586. Across modality/run correction-ratio medians, the mean median was 0.40289; largest q90 was 0.967186. These scales describe representation magnitudes, not utility.
- Correction-off checkpoint controls: 18 records; protected tuple shuffles: 45 repeat records (five per protected checkpoint). The shuffle control checks validation-row-only changes to structural states; final-embedding differences within non-validation rows were tolerated at 1e−5 for floating-point kernel variation.
- CrossMoE diagnostics: 36 run/modality records. Protected attention diagnostics: 54 run/modality/token records. They describe routing/attention use, not architectural value.

### Correction scale, direction, and branch use

Correction-ratio entries below are the mean across run-level validation-node distribution quantiles (3 datasets × 3 seeds), split by modality. Per-run RMS(S), RMS(R), RMS(correction), ratio, and cosine quantiles q10/q25/median/q75/q90/q99 are in the corresponding CSV files.

| Variant | Modality | Ratio q10 | q25 | Median | q75 | q90 | q99 | mean best ||W_C|| | cosine(C,S) median | cosine(C,R) median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| residual_generic | text | 0.192 | 0.266 | 0.374 | 0.483 | 0.575 | 0.707 | 1.492 ± 0.430 | -0.085 | -0.002 |
| residual_generic | visual | 0.235 | 0.320 | 0.429 | 0.528 | 0.602 | 0.704 | 1.436 ± 0.330 | -0.167 | 0.052 |
| residual_protected | text | 0.211 | 0.282 | 0.374 | 0.470 | 0.558 | 0.690 | 1.423 ± 0.246 | -0.054 | 0.008 |
| residual_protected | visual | 0.245 | 0.329 | 0.434 | 0.545 | 0.631 | 0.756 | 1.449 ± 0.182 | -0.112 | 0.031 |

The average run-level node-median RMS values were:
Generic L/H/X = 1.002 / 1.035 / 0.649; Protected L/H/X = 1.008 / 1.026 / 0.638.

| Variant | Mean largest Top-1 expert share | Range | Smallest Top-2 inclusion share |
|---|---:|---:|---:|
| residual_generic | 0.511 | 0.354–0.690 | 0.170 |
| residual_protected | 0.555 | 0.378–0.806 | 0.170 |

| Protected modality | Attention L mean (node SD) | H mean (node SD) | X mean (node SD) |
|---|---:|---:|---:|
| text | 0.214 (0.188) | 0.309 (0.212) | 0.477 (0.234) |
| visual | 0.125 (0.170) | 0.359 (0.244) | 0.516 (0.266) |

### Checkpoint interventions

Each row reports the mean and population SD across three selected checkpoints. For Protected tuple shuffle, the five validation-only permutations were first averaged within each checkpoint; mean within-checkpoint repeat SD is shown separately. Negative Acc/F1 and positive CE deltas indicate a checkpoint relied on the learned correction, not retrained architectural gain.

| Dataset | Variant | Intervention | Δ Acc (pp) | Δ Macro-F1 (pp) | Δ CE | mean within-checkpoint shuffle SD (Acc / F1 / CE) |
|---|---|---|---:|---:|---:|---:|
| Movies | residual_generic | correction_off | -1.150 ± 0.110 | -4.009 ± 1.099 | 0.014 ± 0.004 | — |
| Movies | residual_protected | correction_off | -0.790 ± 0.404 | -2.268 ± 0.518 | 0.012 ± 0.011 | — |
| Movies | residual_protected | correction_tuple_shuffle | -1.410 ± 0.350 | -0.626 ± 0.483 | 0.035 ± 0.002 | 0.321 / 0.790 / 0.007 |
| Grocery | residual_generic | correction_off | -0.429 ± 0.183 | -0.274 ± 0.107 | -0.001 ± 0.006 | — |
| Grocery | residual_protected | correction_off | -0.664 ± 0.239 | -0.469 ± 0.341 | 0.001 ± 0.010 | — |
| Grocery | residual_protected | correction_tuple_shuffle | -2.729 ± 0.582 | -2.451 ± 0.795 | 0.069 ± 0.012 | 0.183 / 0.517 / 0.009 |
| ele-fashion | residual_generic | correction_off | -5.557 ± 3.810 | -7.808 ± 1.795 | 0.093 ± 0.078 | — |
| ele-fashion | residual_protected | correction_off | -2.438 ± 0.673 | -4.763 ± 0.687 | 0.041 ± 0.014 | — |
| ele-fashion | residual_protected | correction_tuple_shuffle | -3.149 ± 0.508 | -5.600 ± 0.513 | 0.088 ± 0.019 | 0.201 / 0.447 / 0.003 |

## Decision

**Final label: `RESIDUAL_ACTIVE_NO_INCREMENT`.** The correction branch learned and affects selected checkpoints, but matched retraining does not provide a consistent increment over Smooth.

This screen isolates baseline-preserving residual augmentation from R0's replacement design. Correction-off and correction-shuffle results describe checkpoint reliance only; architectural judgment comes from matched retraining against Smooth.

| Required question | Answer |
|---|---|
| Was the replacement confound corrected? | YES — R1 retains the complete Smooth state and adds the response branch through zero-initialized W_C. |
| Is the response family useful as a residual? | NO — RESIDUAL_ACTIVE_NO_INCREMENT. |
| Does Protected add value over Generic? | NO / NOT ESTABLISHED. |
| Was correction trained and used? | YES; best-checkpoint max W_C norm 2.14586, mean median correction ratio 0.40289. |
| Do checkpoint reliance and retrained gain agree? | NO — checkpoint reliance is present, while retrained gain is not supported. |
| Should lightweight response research stop? | YES under the frozen terminal rule. |
| Is faithful DiP the recommended next paradigm? | YES — after human review; this report does not start DiP implementation. |

## Required R1 questions (1–18)

1. R1 Smooth is compatible with M0 UNI, N1 SmoothOnly, and R0 SmoothBase within the declared tolerance; maximum mapped forward error was `2.62e-6`.
2. Smooth W_S is parameter-decoupled from W_L and W_H; all three are independent, bias-free 128×128 Linear layers.
3. Both residual variants were exactly identical to Smooth before an optimizer update: maximum train/eval output error was 0.
4. Step 1 produced a finite, nonzero task gradient on W_C for both active variants.
5. Step 1 task gradients into the bank, experts, router, and active composer were zero; after one optimizer step, step 2 task gradients reached those modules. Orth-only gradients were audited separately.
6. Residual Generic did not improve Accuracy or Macro-F1 consistently over Smooth; Movies and Grocery declined on all three matched seeds, while ele-fashion showed small Accuracy/F1 gains with worse CE.
7. Residual Protected did not improve over Smooth; Accuracy and Macro-F1 declined on all three Movies/Grocery seeds and generally declined on ele-fashion, with CE worse on all three ele-fashion seeds and all three Grocery seeds.
8. Protected did not show a consistent advantage over Generic: accuracy favored Protected on Grocery, while Movies and ele-fashion favored Generic; Macro-F1 and CE also reversed by dataset.
9. Mean validation-node correction-ratio medians were 0.403; mean q90 was 0.591; maximum run-level q99 was 1.043. The correction was moderate for most validation nodes, with a small q99 tail near/slightly above Smooth RMS.
10. Correction direction cosines are reported in `correction_direction_summary.csv`; mean run-level median cosine was near zero against both Smooth and the branch residual. No semantic meaning is assigned.
11. W_C left zero initialization in all residual runs; the largest best-checkpoint modality Frobenius norm was 2.146. Per-epoch and best-epoch norms are in `correction_weight_norm.csv`.
12. CrossMoE routing remained non-degenerate (mean largest Top-1 share Generic/Protected: 0.511/0.555; all experts received Top-2 mass). Protected attention had node variation across L/H/X, summarized above.
13. Correction-off intervention effects are given by dataset, metric, and variant in the table above; they indicate checkpoint reliance only.
14. Five-seed Protected correction tuple shuffle effects are listed above; each modality was independently permuted among validation nodes as a whole vector, and non-validation structural states remained bitwise unchanged.
15. Checkpoint reliance and retrained architecture value do not agree: controls perturb selected checkpoints, while matched training does not beat Smooth consistently.
16. The replacement confound was corrected by retaining Smooth, but replacement is not a sufficient primary explanation for R0's failure because residual augmentation did not recover consistent gains.
17. Is it worth continuing the lightweight response family? **NO** under the terminal rule; stop this family.
18. Should the next paradigm be faithful DiP? **YES, as the recommended direction after human review.** No DiP code or test/LP evaluation was started.

## Figures

- `r1_performance.png`
- `r1_paired_deltas.png`
- `r1_correction_scale.png`
- `r1_correction_weight_growth.png`
- `r1_interventions.png`

## Scope boundary

No test evaluation, LP, Toys, Reddit-S, R1.1, or DiP implementation was conducted. The result is limited to validation behavior on the three development datasets and the matched seeds.
