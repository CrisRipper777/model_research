# E0 — Structured Relation-Function Executor Screening

## Decision summary

**Pre-frozen outcome: `FUNCTION_BANK_NOT_CONVINCING`.** The structured bank is executable and its functions have distinct learned output directions, but the screen does not show robust task value over the historical UNI reference or incremental value from edge-specific routing beyond TargetMix.

Across the nine paired runs, EdgeMix−TargetMix averages **+0.05 accuracy points, −0.05 Macro-F1 points, and −0.006 CE**. Its signs and sizes vary by dataset. The within-target shuffle and target-mean interventions are close to zero on average. Router diagnostics show substantial within-target variation on ele-fashion, but little on Movies and Grocery; variation by itself is not evidence of task value. StaticMix−UNI is mixed (−0.18 accuracy points, +0.11 Macro-F1 points, −0.0094 CE), and the E0 variants do not show a stable across-dataset gain over UNI.

The model does use some alternatives to Smooth: smooth-only intervention lowers validation accuracy and Macro-F1 by 1.63 and 2.61 points on average, and relational-off has a modest average cost. Those are whole-checkpoint branch-reliance results. They do **not** establish edge-specific correspondence value: pi-shuffle-within-target and pi-target-mean effects remain small. Function output directions do not collapse, while the routing itself is close to Smooth-only on Movies/Grocery and more mixed on ele-fashion.

This screen does not contradict P0–P1.3: the tested executor did not consistently convert propagation-function heterogeneity into an EdgeMix-over-TargetMix task gain. No M1, multi-hop, relation-evidence change, test evaluation, or auxiliary expert loss was introduced.

## Provenance and protocol

- Source branch: `exp/m01_adaptive_correction_attribution`, source/base SHA `9f0446a982963d962b882d9bdbc18d6669c96f52`.
- Experiment branch: `exp/e0_structured_relation_function_executor`, created from that SHA after fetching and verifying the matching origin ref. No merge from `main`.
- Formal screen: 3 datasets × 3 seeds × 3 variants = **27/27 completed**. No failed or retried formal run.
- Datasets/seeds/variants: Movies, Grocery, ele-fashion; 42, 43, 44; StaticMix, TargetMix, EdgeMix.
- Same M0 full-graph node-classification protocol: AdamW, learning rate 1e−3, weight decay 1e−4, maximum 300 epochs, patience 30, minimum epoch 30, gradient clip 1.0, best validation accuracy checkpoint. No per-dataset tuning.
- Only train/validation labels were exposed. `evaluate_test=false`; test indices were removed, test labels masked, link prediction disabled. Historical UNI and SEM metrics were read from the committed M0 `performance_by_run.csv`; they were not retrained.
- Compute: CUDA 0, NVIDIA RTX 3090 24 GiB. Recorded training time 1,023.3 s; EdgeMix intervention time 116.0 s; maximum allocated memory 15.87 GiB. The largest dataset, ele-fashion, accounts for most of the runtime and memory.
- Exact commands, package versions, and environment are in `run_manifest.json`. Formal raw JSON and checkpoints remain under ignored `outputs/e0_structured_relation_function_executor/`.

The external design sanity review was limited to checking whether “different function” means a genuinely different information path. E0 implements target/source-dependent intra-modal and cross-modal paths; no role labels or auxiliary routing method was borrowed. References reviewed: [RoleMAG](https://arxiv.org/abs/2604.12271), [PLANET](https://arxiv.org/abs/2602.04116), [CAMPA](https://arxiv.org/abs/2605.11468), [CoMAG](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6736317), [DiP](https://proceedings.neurips.cc/paper/2021/hash/253614bbac999b38b5b60cae531c4969-Abstract.html), [I²MoE](https://arxiv.org/abs/2505.19190), and [heterophily-informed message passing](https://openreview.net/pdf?id=9fPinz1iH2).

## Implementation and correctness

E0 changes only the `relation-state → message execution` interface. It reuses M0's modality projectors, P/q/r/u relation-state definitions, leave-one-out context, directed non-self physical edges, original incoming-degree denominator, residual normalization, and four-way semantic fusion. Relation-state parameters remain end-to-end trainable. The one-step function bank is Null, same-modality Smooth, target-conditioned intra-modal Relational, and physical-edge Cross-Modal. Relational and Cross-Modal are calibrated to Smooth's per-edge RMS with only the Smooth reference detached.

Correctness checks:

- **Tests:** 45 passed across M0, M0.1, and E0; one upstream PyG deprecation warning.
- **M0 regression:** P, LOO context, q, r, u, H0, non-self support, and incoming degree matched. CUDA reductions can differ in their final few bits; the regression uses `rtol=1e−6, atol=1e−7`. The E0 relation transforms are compared using a shared computed context so reduction order does not masquerade as a definition change.
- **Smooth identity:** forced `[0,1,0,0]` reproduced ordinary M0 uniform propagation; maximum aggregate-message error was 7.15e−7 and final fused-output error was 1.67e−6.
- **Function sources:** perturbing a Visual source changes the Text-target Cross-Modal output and leaves Text-target Relational output unchanged. The unit tests also cover the opposite direction and same-modality source paths.
- **Calibration:** smoke maximum calibrated RMS-ratio error was 2.98e−7; scales and expert outputs were finite. Formal run diagnostics show relational and cross-modal calibrated RMS ratios at 1 to floating-point precision.
- **Gradients:** smoke and unit tests found finite, nonzero gradients for the router, Smooth transform, Relational source and MLP, Cross-Modal source and MLP, and relation-state encoder.
- **Common architecture:** each model has 1,098,996 trainable model parameters on Movies/Grocery and 967,924 on ele-fashion; the three variants have exactly equal counts within a dataset. With the classifier, totals are 1,101,576 and 969,472 respectively. All 9 dataset×seed initialization audits were bitwise equal across variants.
- **Chunking:** small and large chunk outputs were identical in the synthetic smoke graph for all three variants. Static/global and Target/target-mean identities were within 3e−6 on Movies smoke and are unit tested.
- **Numerical/protocol audit:** finite logits, probabilities, loss, gradients, and calibration scales; probability rows sum to one; no test labels/indices or LP.

### Figure quality assurance

The six figures are exported at 182.9 mm width in PNG, editable SVG/PDF, and 600-dpi TIFF. The Python source preflight passed 21 checks with no warnings or failures. All six PDF text audits found no glyph below 5 pt (minimum 5.5 pt). Five multi-panel alignment gates passed at 1.5 pt; the one single-panel figure was marked not applicable. All six collision audits passed with zero failures and zero warnings. Alignment/collision JSON records are under `figures/qa/`.

No architecture or experiment-protocol deviation was made. An early smoke attempt exposed a validation-label slicing error in the harness; it was corrected before the passing smoke and before any formal run. One independent CUDA LOO reduction showed a few 1e−7 of summation-order variation; the context itself is checked within the specified tolerance.

## Validation performance

Cells are mean ± population SD over seeds 42–44. Accuracy and Macro-F1 are percentages; CE is native. SEM and UNI are historical M0 references. `ALL` equally averages the nine dataset×seed runs and is descriptive across datasets with different scales.

### Movies

| Method | Accuracy (%) | Macro-F1 (%) | CE |
| --- | ---: | ---: | ---: |
| SEM | 52.36 ± 0.29 | 37.98 ± 3.48 | 1.4935 ± 0.0171 |
| UNI | 55.42 ± 0.34 | 45.87 ± 0.80 | 1.4193 ± 0.0273 |
| StaticMix | 55.17 ± 0.33 | 46.00 ± 0.82 | 1.4055 ± 0.0133 |
| TargetMix | 55.03 ± 0.24 | 44.92 ± 1.32 | 1.4131 ± 0.0051 |
| EdgeMix | 55.14 ± 0.39 | 45.01 ± 0.41 | 1.3995 ± 0.0252 |

### Grocery

| Method | Accuracy (%) | Macro-F1 (%) | CE |
| --- | ---: | ---: | ---: |
| SEM | 78.24 ± 0.16 | 69.23 ± 0.41 | 0.8595 ± 0.0208 |
| UNI | 82.25 ± 0.23 | 73.41 ± 1.07 | 0.7085 ± 0.0293 |
| StaticMix | 81.84 ± 0.28 | 73.23 ± 0.44 | 0.7019 ± 0.0319 |
| TargetMix | 82.00 ± 0.18 | 73.84 ± 0.28 | 0.7198 ± 0.0419 |
| EdgeMix | 81.94 ± 0.19 | 73.50 ± 0.53 | 0.7033 ± 0.0329 |

### ele-fashion

| Method | Accuracy (%) | Macro-F1 (%) | CE |
| --- | ---: | ---: | ---: |
| SEM | 86.94 ± 0.11 | 66.73 ± 0.09 | 0.4187 ± 0.0026 |
| UNI | 87.41 ± 0.08 | 68.66 ± 0.80 | 0.4162 ± 0.0032 |
| StaticMix | 87.54 ± 0.12 | 69.04 ± 0.45 | 0.4082 ± 0.0032 |
| TargetMix | 87.24 ± 0.14 | 68.40 ± 0.58 | 0.4078 ± 0.0031 |
| EdgeMix | 87.33 ± 0.05 | 68.50 ± 0.15 | 0.4208 ± 0.0081 |

### Equal-weight aggregate across nine runs

| Method | Accuracy (%) | Macro-F1 (%) | CE |
| --- | ---: | ---: | ---: |
| SEM | 72.51 ± 14.69 | 57.98 ± 14.32 | 0.9239 ± 0.4414 |
| UNI | 75.03 ± 14.03 | 62.64 ± 12.05 | 0.8480 ± 0.4219 |
| StaticMix | 74.85 ± 14.11 | 62.76 ± 11.99 | 0.8385 ± 0.4189 |
| TargetMix | 74.76 ± 14.11 | 62.39 ± 12.57 | 0.8469 ± 0.4209 |
| EdgeMix | 74.80 ± 14.08 | 62.33 ± 12.43 | 0.8412 ± 0.4120 |

### Paired same-seed deltas

Positive Accuracy/Macro-F1 is favorable; positive CE is unfavorable. Accuracy and Macro-F1 are percentage points. Entries are paired mean ± population SD.

| Dataset | Comparison | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
| --- | --- | ---: | ---: | ---: |
| Movies | Static−UNI | −0.25 ± 0.53 | +0.13 ± 1.11 | −0.0138 ± 0.0401 |
| Movies | Target−Static | −0.14 ± 0.14 | −1.07 ± 1.97 | +0.014 ± 0.011 |
| Movies | Edge−Target | +0.11 ± 0.15 | +0.09 ± 1.10 | −0.014 ± 0.029 |
| Movies | Edge−Static | −0.03 ± 0.15 | −0.99 ± 0.88 | −0.006 ± 0.029 |
| Movies | Edge−UNI | −0.28 ± 0.65 | −0.86 ± 0.40 | −0.020 ± 0.040 |
| Grocery | Static−UNI | −0.41 ± 0.13 | −0.17 ± 1.42 | −0.0065 ± 0.0231 |
| Grocery | Target−Static | +0.16 ± 0.12 | +0.60 ± 0.66 | +0.018 ± 0.010 |
| Grocery | Edge−Target | −0.06 ± 0.02 | −0.34 ± 0.48 | −0.017 ± 0.017 |
| Grocery | Edge−Static | +0.10 ± 0.12 | +0.26 ± 0.52 | +0.001 ± 0.012 |
| Grocery | Edge−UNI | −0.31 ± 0.17 | +0.09 ± 1.02 | −0.005 ± 0.018 |
| ele-fashion | Static−UNI | +0.12 ± 0.17 | +0.38 ± 0.70 | −0.0080 ± 0.0044 |
| ele-fashion | Target−Static | −0.29 ± 0.13 | −0.64 ± 0.17 | −0.000 ± 0.002 |
| ele-fashion | Edge−Target | +0.09 ± 0.10 | +0.09 ± 0.59 | +0.013 ± 0.008 |
| ele-fashion | Edge−Static | −0.20 ± 0.12 | −0.55 ± 0.50 | +0.013 ± 0.009 |
| ele-fashion | Edge−UNI | −0.08 ± 0.05 | −0.16 ± 0.69 | +0.005 ± 0.009 |
| ALL | Static−UNI | −0.18 ± 0.40 | +0.11 ± 1.14 | −0.0094 ± 0.0271 |
| ALL | Target−Static | −0.09 ± 0.23 | −0.37 ± 1.40 | +0.008 ± 0.012 |
| ALL | Edge−Target | +0.05 ± 0.13 | −0.05 ± 0.80 | −0.006 ± 0.024 |
| ALL | Edge−Static | −0.05 ± 0.18 | −0.42 ± 0.83 | +0.003 ± 0.021 |
| ALL | Edge−UNI | −0.22 ± 0.40 | −0.31 ± 0.85 | −0.007 ± 0.026 |

StaticMix−UNI is a capacity comparison, not a pure routing attribution: pooled paired changes are −0.18 ± 0.40 accuracy points, +0.11 ± 1.14 Macro-F1 points, and −0.0094 ± 0.0271 CE. TargetMix−StaticMix and EdgeMix−TargetMix are the parameter-matched granularity comparisons. No significance tests were run.

## Router use and edge-level evidence

Means below average the 9 run-level summaries equally within each variant×modality. Each cell gives `pi_null / pi_smooth / pi_relational / pi_cross_modal`; normalized entropy is `H/log(4)`, effective function count is `exp(H)`, and margin is the average run-level median of top-1 minus top-2 probability.

| Variant | Modality | Mean pi (Null / Smooth / Relational / Cross) | Normalized entropy | Effective functions | Median routing margin |
| --- | --- | --- | ---: | ---: | ---: |
| StaticMix | Text | 0.001 / 0.680 / 0.220 / 0.099 | 0.160 | 1.342 | 0.785 |
| StaticMix | Visual | 0.001 / 0.719 / 0.120 / 0.160 | 0.240 | 1.555 | 0.737 |
| TargetMix | Text | 0.009 / 0.846 / 0.088 / 0.057 | 0.200 | 1.444 | 0.861 |
| TargetMix | Visual | 0.007 / 0.880 / 0.062 / 0.050 | 0.169 | 1.365 | 0.911 |
| EdgeMix | Text | 0.007 / 0.821 / 0.107 / 0.064 | 0.191 | 1.417 | 0.829 |
| EdgeMix | Visual | 0.005 / 0.884 / 0.067 / 0.044 | 0.145 | 1.300 | 0.937 |

These pooled means hide a strong dataset split. Movies and Grocery routes are overwhelmingly Smooth: for EdgeMix, mean Smooth mass is 0.970–0.995 by modality; relational and cross-modal masses are mostly around 0.001–0.014. On ele-fashion, EdgeMix mean Smooth mass is 0.523 Text and 0.678 Visual, with Relational 0.296/0.192 and Cross-Modal 0.172/0.123. Null remains low across all datasets; null-off has negligible mean effect.

### Within-target variation and modality disagreement

For validation targets with in-degree ≥5, EdgeMix within-target routing variation (mean over the three seed-level medians; q90 is the mean of seed-level q90) is:

| Dataset | Modality | Eligible targets | Within-target L1 median | Within-target L1 q90 |
| --- | --- | ---: | ---: | ---: |
| Movies | Text | 1,706–1,775 | 0.014 | 0.151 |
| Movies | Visual | 1,706–1,775 | 0.001 | 0.006 |
| Grocery | Text | 1,454–1,468 | 0.031 | 0.171 |
| Grocery | Visual | 1,454–1,468 | 0.014 | 0.113 |
| ele-fashion | Text | 1,758 | 0.265 | 0.626 |
| ele-fashion | Visual | 1,758 | 0.208 | 0.604 |

Thus EdgeMix has substantial within-target routing variation on ele-fashion; it is weak for Movies Visual and modest for Movies Text/Grocery. The edge routing variance is not uniform across the benchmark.

Mean Text/Visual disagreement on the same physical edge (averaged over three seeds) is:

| Dataset | Variant | Mean L1 | Mean JS divergence | Argmax disagreement |
| --- | --- | ---: | ---: | ---: |
| Movies | Static / Target / Edge | 0.000 / 0.042 / 0.048 | 0.000 / 0.008 / 0.010 | 0.000 / 0.011 / 0.013 |
| Grocery | Static / Target / Edge | 0.000 / 0.032 / 0.026 | 0.000 / 0.003 / 0.003 | 0.000 / 0.007 / 0.012 |
| ele-fashion | Static / Target / Edge | 0.602 / 0.346 / 0.399 | 0.079 / 0.046 / 0.065 | 0.667 / 0.158 / 0.204 |

The static variants have equal per-modality pi across all edges, but Text and Visual can have different global pi. Target/Edge modality disagreement is small on Movies/Grocery and much larger on ele-fashion. Tying modalities increases CE in 8/9 runs but lowers accuracy in only 4/9 and Macro-F1 in 5/9, a weak/mixed reliance signal rather than a broad modality-control gain.

### EdgeMix routing interventions

Each delta is intervention minus normal EdgeMix inference. Shuffle has 5 repeats for each of 9 runs (45 records); other interventions have one record per run. Values are pooled mean ± population SD. Accuracy/Macro-F1 are percentage points; CE is native.

| Intervention | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE | Run-level direction |
| --- | ---: | ---: | ---: | --- |
| pi-shuffle-within-target | −0.037 ± 0.059 | −0.041 ± 0.132 | +0.00063 ± 0.0013 | 6/9 acc harmed; 4/9 F1 harmed; CE worse in 7/9 |
| pi-target-mean | −0.032 ± 0.033 | −0.034 ± 0.099 | −0.00005 ± 0.0010 | 6/9 acc harmed; 4/9 F1 harmed; CE worse in 6/9 |
| pi-global-mean | −0.552 ± 0.547 | −1.116 ± 1.233 | +0.0102 ± 0.0101 | 6/9 acc/F1 harmed; CE worse in 9/9 |
| modality-tied pi | −0.109 ± 0.180 | −0.161 ± 0.229 | +0.0013 ± 0.0017 | 4/9 acc harmed; 5/9 F1 harmed; CE worse in 8/9 |
| smooth-only | −1.629 ± 2.109 | −2.608 ± 3.489 | +0.0398 ± 0.0530 | 7/9 acc/F1 harmed; CE worse in 6/9 |
| null-off | −0.040 ± 0.050 | −0.008 ± 0.050 | +0.0002 ± 0.0003 | Small; mixed |
| relational-off | −0.507 ± 0.677 | −0.703 ± 1.187 | +0.0093 ± 0.0142 | 7/9 acc/F1 harmed; CE worse in 4/9 |
| cross-off | −0.236 ± 0.385 | −0.469 ± 0.806 | +0.0037 ± 0.0086 | 5/9 acc/F1 harmed; CE worse in 3/9 |

Global-mean has a clearer average cost than target-mean, but it removes both target-level and within-target routing structure. It cannot identify edge-specific correspondence on its own. Shuffle and target-mean directly test correspondence/within-target variation and are weak on average. Expert-off and smooth-only test branch reliance, not the value of edge-conditioned routing.

## Function-bank distinctness and calibration

Across EdgeMix validation edges and modalities, the average of run-level median cosines is:

| Pair | Mean of run medians | Range of run medians |
| --- | ---: | ---: |
| Smooth–Relational | 0.091 | −0.039 to 0.253 |
| Smooth–Cross-Modal | 0.056 | −0.177 to 0.233 |
| Relational–Cross-Modal | 0.246 | 0.075 to 0.518 |

The functions do not collapse into parallel copies of one direction. Calibration also works: formal median per-edge RMS ratios `rms(F_R)/rms(F_S)` and `rms(F_X)/rms(F_S)` are 1.000 to floating-point precision. Across EdgeMix run/modality summaries, typical raw RMS medians are 1.087 for Smooth, 0.702 for raw Relational, and 0.611 for raw Cross-Modal. Calibration scale medians are approximately 2.26 and 2.41; the largest run-level q99 scales are 12.99 and 11.67. Scales were finite. The elevated q99 values are recorded without clamping: calibrated outputs stayed finite and matched the reference RMS, so no numerical correction was indicated.

Function directions are distinct, but routing concentration is dataset-dependent: near-Smooth-only on Movies/Grocery, mixed on ele-fashion. This is a routing-use pattern, not evidence that the structured bank consistently improves task performance.

## Frozen questions and attribution

1. **Are the three variants parameter-identical?** Yes. Exact model counts match within every dataset; total trainable parameters including the classifier also match across variants.
2. **Are same-seed initializations identical?** Yes. All nine dataset×seed audits are bitwise equal, including identical hashes across all variants.
3. **Does E0 preserve M0 relation-state?** Yes, within the stated floating-point tolerance for P, LOO context, q, r, u, H0, edge support, and degree.
4. **Does Smooth-only reduce to ordinary Smooth?** Yes. Aggregate and final-output errors are below 2e−6 in CUDA smoke; synthetic unit checks pass.
5. **How is StaticMix vs UNI?** Mixed: pooled accuracy −0.18 pp, Macro-F1 +0.12 pp, CE −0.0095. StaticMix is not a stable broad performance gain, and its comparison also includes added function capacity.
6. **Does TargetMix improve over StaticMix?** No consistent gain. Pooled paired delta is −0.09 accuracy pp, −0.37 Macro-F1 pp, +0.008 CE; Grocery is the exception on accuracy/F1.
7. **Does EdgeMix improve over TargetMix?** No consistent incremental gain. Pooled paired delta is +0.05 accuracy pp, −0.05 Macro-F1 pp, −0.006 CE with mixed per-dataset directions.
8. **Is the structured function bank used?** Some alternatives are used, especially on ele-fashion; smooth-only degrades the fixed checkpoint. This branch reliance does not establish edge-specific task value.
9. **How is Null used?** Very little: below 1% mean in most summaries, with negligible null-off impact.
10. **Is Smooth still the dominant default?** Yes on Movies/Grocery and in pooled Target/Edge summaries. StaticMix and EdgeMix on ele-fashion are more mixed, especially Text.
11. **Does Relational have stable nonzero usage?** It is nontrivial on ele-fashion but near zero on Movies/Grocery; no cross-dataset stable usage.
12. **Does Cross-Modal have stable nonzero usage?** Same pattern: meaningful mass on ele-fashion, low mass on Movies/Grocery.
13. **Do function outputs collapse?** No directional collapse: pairwise median cosine is low to moderate. The router itself is concentrated on Smooth for two datasets.
14. **Is there within-node routing variation?** Yes on ele-fashion; weak or modest on Movies/Grocery. It is not a benchmark-wide phenomenon.
15. **Do Text and Visual routing distributions differ?** Strongly on ele-fashion, minimally on Movies/Grocery. The average tying cost is mostly visible in CE and is weak for accuracy/F1.
16. **Does within-target shuffle hurt?** Only slightly on average. Across per-run five-repeat means, accuracy decreases in 6/9, Macro-F1 in 4/9, and CE increases in 7/9; mean magnitudes are tiny.
17. **Does target-mean hurt?** Weak/mixed: accuracy decreases in 6/9, F1 in 4/9, CE worsens in 6/9, with near-zero average changes.
18. **Does global-mean hurt?** Yes more clearly, but it removes target and edge routing together; it is not an edge-only attribution.
19. **Does modality tying hurt?** Weakly for CE (8/9 runs) but not consistently for accuracy/F1.
20. **What does smooth-only show?** The selected EdgeMix model relies on more than Smooth in aggregate; this intervention is a branch-reliance test, not a retrained UNI comparison.
21. **What do null-/relational-/cross-off show?** Null freedom is barely used. Relational-off has a modest average adverse effect; Cross-off is smaller and mixed. Neither proves a branch is necessary or that edge correspondence has value.
22. **Does edge-specific composition have task value beyond target-level composition?** **NO, not established by this screen.** EdgeMix−TargetMix is near zero and the correspondence-specific interventions are weak.
23. **What is the likely bottleneck?** Not determined. The current function bank does not show robust task value; these results do not identify relation-state as the cause. A future reviewed step should first inspect function definitions and the relation-state’s ability to separate useful cases.
24. **Frozen category:** `FUNCTION_BANK_NOT_CONVINCING`.
25. **Next stage:** after human review only, reconsider the tested function definitions and evidence-to-function fit. This report does not implement that work or enter M1.

## Required self-audit

- **Branch reliance vs edge-routing value:** separated. Smooth-only and expert-off effects are not used as evidence for EdgeMix-over-TargetMix value.
- **Function-bank capacity control:** StaticMix is included. StaticMix−UNI is described as a capacity comparison, not a pure routing gain.
- **Target-level control:** TargetMix has identical parameters and initialization and changes only pooling granularity relative to StaticMix.
- **Edge-level attribution:** EdgeMix−TargetMix changes only routing granularity; within-target pi interventions are reported separately.
- **Output magnitude:** Relational/Cross-Modal RMS is calibrated to Smooth; scale quantiles are saved, including the elevated q99 values, with no clamp because outputs remained finite and correctly calibrated.
- **Expert collapse:** Output directions remain distinct; the router concentrates on Smooth on Movies/Grocery. Both facts are reported without conflating them.
- **P0–P1.3 premise:** The screen evaluates one executor bank and does not revise the independently validated propagation utility/function heterogeneity premise.
- **Router variance:** Ele-fashion variation is not described as task value without paired and intervention support.

## Artifacts

Run-level and aggregate tables are under `data/`; the six figures are under `figures/`. The manifest records source provenance, code/environment, commands, correctness checks, and compute. Formal raw records/checkpoints are ignored under `outputs/e0_structured_relation_function_executor/`.
