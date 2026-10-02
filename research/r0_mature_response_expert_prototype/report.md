# R0 — Mature Structural-Response Expert Prototype

## Study scope and decision boundary

This report covers 36/36 validation-only, full-graph node-classification runs across Movies, Grocery, and ele-fashion (seeds 42–44). The fixed comparison is Smooth, mature low/high response bank, Bank + CrossMoE, and Bank + CrossMoE + protected intrinsic-semantic query. No test metrics or LP tasks were run. All pooled summaries are descriptive and are not IID significance tests.

Source: `exp/n1_recipient_function_strength_mixer` at `2fbe1f14cda0d33d72f8c8468ef9068af28f6217`; experiment branch `exp/r0_mature_response_expert_prototype`. Runtime reported by campaign: 345.4 s; summed training plus checkpoint-intervention time: 314.9 s. Device: cuda:1.

Correctness audit: 21 unit tests passed (one upstream PyG deprecation warning); the Movies/42 four-variant one-epoch GPU smoke passed with finite active-path gradients and exact initialization fairness. Across formal runs, the largest observed Lraw + Hraw − I absolute error was 2.38e−7; test indices and test labels were absent from every run.

## Main validation performance

Values are mean ± population SD across the three seeds. Accuracy and macro-F1 are percentages; CE is cross-entropy.

| Dataset | Variant | Accuracy (%) | Macro-F1 (%) | CE |
|---|---|---:|---:|---:|
| Movies | Smooth | 55.579 ± 0.289 | 44.592 ± 1.795 | 1.402 ± 0.027 |
| Movies | Bank | 54.109 ± 0.218 | 40.594 ± 0.755 | 1.441 ± 0.041 |
| Movies | Bank + CrossMoE | 54.449 ± 0.368 | 42.299 ± 0.144 | 1.512 ± 0.058 |
| Movies | Bank + CrossMoE + Protected | 53.939 ± 0.269 | 42.076 ± 0.468 | 1.435 ± 0.008 |
| Grocery | Smooth | 82.030 ± 0.183 | 74.116 ± 0.954 | 0.724 ± 0.036 |
| Grocery | Bank | 81.571 ± 0.307 | 72.397 ± 1.595 | 0.732 ± 0.033 |
| Grocery | Bank + CrossMoE | 81.103 ± 0.325 | 71.526 ± 1.022 | 0.725 ± 0.033 |
| Grocery | Bank + CrossMoE + Protected | 80.966 ± 0.337 | 71.307 ± 1.185 | 0.752 ± 0.030 |
| ele-fashion | Smooth | 87.402 ± 0.100 | 68.370 ± 0.244 | 0.413 ± 0.007 |
| ele-fashion | Bank | 87.191 ± 0.110 | 68.413 ± 0.368 | 0.416 ± 0.014 |
| ele-fashion | Bank + CrossMoE | 87.140 ± 0.076 | 67.432 ± 0.546 | 0.414 ± 0.001 |
| ele-fashion | Bank + CrossMoE + Protected | 87.300 ± 0.118 | 68.253 ± 0.239 | 0.422 ± 0.010 |

## Paired within-dataset, within-seed changes

Accuracy and macro-F1 deltas are percentage points; CE is raw CE change. Negative CE is favorable. The pooled descriptive row combines nine matched dataset-seed pairs and is not treated as nine IID replicates.

| Contrast | Dataset | Metric | Mean delta | Population SD | n |
|---|---|---|---:|---:|---:|
| Bank-Smooth | Movies | accuracy_delta_pp | -1.4697 | 0.2449 | 3 |
| Bank-Smooth | Movies | macro_f1_delta_pp | -3.9978 | 1.4018 | 3 |
| Bank-Smooth | Movies | ce_delta | 0.0384 | 0.0144 | 3 |
| Bank-Smooth | Grocery | accuracy_delta_pp | -0.4588 | 0.1317 | 3 |
| Bank-Smooth | Grocery | macro_f1_delta_pp | -1.7185 | 0.6542 | 3 |
| Bank-Smooth | Grocery | ce_delta | 0.0080 | 0.0177 | 3 |
| Bank-Smooth | ele-fashion | accuracy_delta_pp | -0.2114 | 0.1543 | 3 |
| Bank-Smooth | ele-fashion | macro_f1_delta_pp | 0.0429 | 0.2104 | 3 |
| Bank-Smooth | ele-fashion | ce_delta | 0.0038 | 0.0078 | 3 |
| Bank-Smooth | ALL_POOLED_DESCRIPTIVE | accuracy_delta_pp | -0.7133 | 0.5745 | 9 |
| Bank-Smooth | ALL_POOLED_DESCRIPTIVE | macro_f1_delta_pp | -1.8911 | 1.8838 | 9 |
| Bank-Smooth | ALL_POOLED_DESCRIPTIVE | ce_delta | 0.0167 | 0.0208 | 9 |
| CrossMoE-Bank | Movies | accuracy_delta_pp | 0.3399 | 0.3676 | 3 |
| CrossMoE-Bank | Movies | macro_f1_delta_pp | 1.7046 | 0.6986 | 3 |
| CrossMoE-Bank | Movies | ce_delta | 0.0713 | 0.0308 | 3 |
| CrossMoE-Bank | Grocery | accuracy_delta_pp | -0.4685 | 0.0414 | 3 |
| CrossMoE-Bank | Grocery | macro_f1_delta_pp | -0.8714 | 0.7859 | 3 |
| CrossMoE-Bank | Grocery | ce_delta | -0.0067 | 0.0264 | 3 |
| CrossMoE-Bank | ele-fashion | accuracy_delta_pp | -0.0511 | 0.1533 | 3 |
| CrossMoE-Bank | ele-fashion | macro_f1_delta_pp | -0.9808 | 0.8591 | 3 |
| CrossMoE-Bank | ele-fashion | ce_delta | -0.0021 | 0.0148 | 3 |
| CrossMoE-Bank | ALL_POOLED_DESCRIPTIVE | accuracy_delta_pp | -0.0599 | 0.4030 | 9 |
| CrossMoE-Bank | ALL_POOLED_DESCRIPTIVE | macro_f1_delta_pp | -0.0492 | 1.4678 | 9 |
| CrossMoE-Bank | ALL_POOLED_DESCRIPTIVE | ce_delta | 0.0208 | 0.0436 | 9 |
| Protected-CrossMoE | Movies | accuracy_delta_pp | -0.5099 | 0.4880 | 3 |
| Protected-CrossMoE | Movies | macro_f1_delta_pp | -0.2233 | 0.3280 | 3 |
| Protected-CrossMoE | Movies | ce_delta | -0.0766 | 0.0640 | 3 |
| Protected-CrossMoE | Grocery | accuracy_delta_pp | -0.1367 | 0.1441 | 3 |
| Protected-CrossMoE | Grocery | macro_f1_delta_pp | -0.2193 | 0.1745 | 3 |
| Protected-CrossMoE | Grocery | ce_delta | 0.0267 | 0.0419 | 3 |
| Protected-CrossMoE | ele-fashion | accuracy_delta_pp | 0.1602 | 0.1934 | 3 |
| Protected-CrossMoE | ele-fashion | macro_f1_delta_pp | 0.8205 | 0.5291 | 3 |
| Protected-CrossMoE | ele-fashion | ce_delta | 0.0075 | 0.0106 | 3 |
| Protected-CrossMoE | ALL_POOLED_DESCRIPTIVE | accuracy_delta_pp | -0.1621 | 0.4171 | 9 |
| Protected-CrossMoE | ALL_POOLED_DESCRIPTIVE | macro_f1_delta_pp | 0.1260 | 0.6169 | 9 |
| Protected-CrossMoE | ALL_POOLED_DESCRIPTIVE | ce_delta | -0.0141 | 0.0632 | 9 |
| Protected-Smooth | Movies | accuracy_delta_pp | -1.6397 | 0.1435 | 3 |
| Protected-Smooth | Movies | macro_f1_delta_pp | -2.5164 | 2.0709 | 3 |
| Protected-Smooth | Movies | ce_delta | 0.0331 | 0.0314 | 3 |
| Protected-Smooth | Grocery | accuracy_delta_pp | -1.0639 | 0.2359 | 3 |
| Protected-Smooth | Grocery | macro_f1_delta_pp | -2.8092 | 0.4620 | 3 |
| Protected-Smooth | Grocery | ce_delta | 0.0279 | 0.0357 | 3 |
| Protected-Smooth | ele-fashion | accuracy_delta_pp | -0.1023 | 0.0684 | 3 |
| Protected-Smooth | ele-fashion | macro_f1_delta_pp | -0.1174 | 0.4833 | 3 |
| Protected-Smooth | ele-fashion | ce_delta | 0.0091 | 0.0105 | 3 |
| Protected-Smooth | ALL_POOLED_DESCRIPTIVE | accuracy_delta_pp | -0.9353 | 0.6551 | 9 |
| Protected-Smooth | ALL_POOLED_DESCRIPTIVE | macro_f1_delta_pp | -1.8143 | 1.7415 | 9 |
| Protected-Smooth | ALL_POOLED_DESCRIPTIVE | ce_delta | 0.0234 | 0.0299 | 9 |

## Requested research questions

1. **Smooth compatibility:** passed; Movies/42 smoke max absolute error 2.6226043701171875e-06. The dedicated smoke regression compared mapped M0 UNI, N1 SmoothOnly, and R0 SmoothBase intermediate and final paths.
2. **Low/high numerical health:** each training run asserted Lraw + Hraw = I at max error ≤2e−6; isolated nodes use Lraw=I and Hraw=0. Chunked aggregation equivalence and brute-force formula checks passed in the correctness suite.
3. **L/H response non-collapse:** for P3 the mean of run-level node-median RMS values is L raw node RMS median: 0.902, H raw node RMS median: 0.356, L transformed node RMS median: 1.011, H transformed node RMS median: 1.021. The corresponding transformed-response cosine medians are near zero (I–L cosine median: 0.001; I–H cosine median: -0.000; L–H cosine median: -0.004; Lraw–Hraw cosine median: 0.011); the full per-run distributions remain in the response CSVs. These diagnostics establish nonzero, geometrically distinct responses, not independent information or task utility.
4. **Bank-only P1:** Bank-Smooth validation deltas are unfavorable overall: Accuracy and CE worsen on all three datasets, while Macro-F1 improves only marginally on ele-fashion and declines on Movies and Grocery. See the matched table for effect sizes and seed spread.
5. **CrossMoE routing:** 36 modality-level records show non-degenerate routing. For P3, largest top-1 expert share averages 0.610 (range 0.450–0.880); all experts receive top-2 mass, although the smallest observed inclusion is 0.049, so use is not uniform and some routes are concentrated.
6. **Prompt specialization:** the mean final pairwise prompt cosine is -0.500; orthogonality loss moves from mean 1.069 to 0.607. Prompts are separated by the regularizer; this alone does not establish expert utility.
7. **Expert output separation:** on identical other-modality validation inputs, P3 expert-pair cosine medians average 0.110, normalized L2 averages 1.355, and expert output RMS is nonzero. Outputs do not collapse to the same vector, but this is not evidence that the experts encode known semantic relations.
8. **P2 vs P1:** CrossMoE-Bank is dataset and metric dependent: Movies gains Accuracy/Macro-F1 but worsens CE; Grocery loses Accuracy/Macro-F1 with a small CE decrease; ele-fashion shows a slight Macro-F1/CE improvement but no Accuracy gain. This does not form a consistent retrained increment.
9. **Protected-attention node variation:** 18 run/modality summaries show node variation in every token. Across P3 runs, mean attention is L/H/X = 0.231/0.309/0.460 for text and 0.177/0.381/0.442 for visual. Node-wise attention SD and quantiles are in `protected_attention_summary.csv`.
10. **Text vs Visual utilization:** the mean node-level attention L1 distance is 0.609; per-token Text–Visual correlations are low and dataset-dependent. The two target modalities therefore use different response mixtures, without evidence that the difference improves accuracy.
11. **P3 vs P2:** Protected-CrossMoE deltas vary by dataset and metric: P3 lowers Accuracy/Macro-F1 on Movies while improving CE, is worse on all three metrics on Grocery, and improves Accuracy/Macro-F1 but worsens CE on ele-fashion. Full model parameter counts match exactly; active composers differ by 206 parameters across both modalities (Generic 265166, Protected 264960).
12. **P3 vs Smooth:** Accuracy, Macro-F1, and CE means all move against P3 on each of the three datasets. This rejects the strong-prototype hypothesis for this validation screen; no significance test or population claim is made.
13. **Dataset or metric dependence:** P2-versus-P1 and P3-versus-P2 show reversals. P3-versus-Smooth is directionally unfavorable on all metrics and datasets, with the largest drops on Movies and Grocery; the pooled row is descriptive only.
14. **Interventions:** masking L, H, X, or shuffling router tuples lowers validation scores on the selected P3 checkpoints. The five router tuple shuffles are first averaged within checkpoint. These show checkpoint co-adaptation; they do not establish retrained architectural value, and the effects do not reverse the P3-versus-Smooth comparison.

| Dataset | Checkpoint intervention | Acc Δ (pp) | Macro-F1 Δ (pp) | CE Δ |
|---|---|---:|---:|---:|
| Movies | L_off | -1.290 | -3.478 | 0.0270 |
| Movies | H_off | -1.550 | -2.459 | 0.0526 |
| Movies | X_off | -0.550 | -1.961 | 0.0043 |
| Movies | router_tuple_shuffle | -0.400 | -1.063 | 0.0031 |
| Grocery | L_off | -1.835 | -1.591 | 0.0353 |
| Grocery | H_off | -2.567 | -3.122 | 0.0904 |
| Grocery | X_off | -0.810 | -0.721 | 0.0017 |
| Grocery | router_tuple_shuffle | -0.687 | -0.874 | 0.0115 |
| ele-fashion | L_off | -0.815 | -1.184 | 0.0030 |
| ele-fashion | H_off | -1.094 | -2.942 | 0.0276 |
| ele-fashion | X_off | -0.883 | -2.353 | 0.0093 |
| ele-fashion | router_tuple_shuffle | -1.368 | -2.700 | 0.0335 |

15. **Orthogonality loss:** 36 best-epoch records are available. Across the 18 CrossMoE runs, the mean orth/task ratio at the selected epoch is 0.0032; the raw loss is retained with fixed coefficient 1e−3 and per-epoch ratios are in `orth_loss_summary.csv`.
16. **Parameter/capacity confound:** all 36 initialization-audit rows report bitwise-equal full-model and classifier states across variants and exact total parameter equality. Model count is 1640468 and classifier count is 2580; active composer counts differ by only 206 parameters, which is disclosed above.
17. **Final scientific label:** `ACTIVE_NO_INCREMENT`.
18–20. **Expansion and route decision:** do not expand to Toys or Reddit-S in this phase. Stop this lightweight response-family iteration and prioritize a DiP-style mediator as the next research direction, subject to the requested human review.

## Interpretation and next-step decision

**Final label: `ACTIVE_NO_INCREMENT`.** Cross-modal experts, prompts, and protected attention exhibit non-degenerate routing and node-dependent use, but validation retraining provides no stable performance increment over Smooth. The P3-Smooth mean deltas are unfavorable for Accuracy, Macro-F1, and CE on every dataset. Checkpoint intervention sensitivity is evidence of co-adaptation, not a substitute for the matched retrained comparison.

The mature L/H bank was not compared head-to-head with Smooth/AbsDiff/Product in this R0 screen. Prior N1 results did not support AbsDiff/Product as stable function-bank gains, and the present Bank-Smooth retraining also declines overall; current evidence gives no reason to prioritize this response-family line, without claiming a direct cross-stage ranking.

**Expansion decision: NO.** Do not extend this prototype to Toys or Reddit-S. Stop the lightweight response-family branch and make a DiP-style mediator the recommended next design direction; begin that work only after human review.

## Capacity, orthogonality, and execution notes

The observed two-modality Generic and protected first stages differ by 206 parameters; the shared post-composer is reported separately. Campaign manifest declares 36/36 completed runs and 0 failure records. Per-run epoch count, runtime and peak allocated GPU memory are in `performance_by_run.csv`.

No edge router, edge role, utility supervision, multiband spectral mechanism, test evaluation, or LP objective is present in this experiment. L/H are called structural response modes; CrossMoE components are node-level cross-modal response experts; attention variation is response-utilization variation.

## Figure contract

Core question: does the fixed A+B+C prototype add competitive validation value over Smooth across three development datasets, and do its diagnostics support non-degenerate response use? Figure archetype: quantitative evidence grid. The performance figure carries the primary comparison; paired deltas show matched evidence; response, MoE, attention, and intervention figures bound mechanism claims. All uncertainty bars in performance panels are population SD across three seeds; paired plots show matched per-seed deltas. Source data are the CSVs in `data/`; each point is traceable to dataset, seed, and variant. No p-values or inferential tests are reported.
Final figure QA: six figures passed strict panel alignment, PDF collision audits, and the 5 pt text floor; the minimum PDF glyph size was 5.7 pt. The plotting-source preflight passed 21 checks with zero warnings or failures, and all panels were visually inspected after export. Details are in `figure_qa.md`.

## Artifacts

Machine-readable run provenance is in `run_manifest.json`. Large checkpoints and per-node router tables are kept under the ignored outputs directory; the report bundle contains compact per-run and summary CSVs plus vector PDFs and PNG previews.
