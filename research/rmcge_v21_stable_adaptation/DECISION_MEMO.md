# RM-CGE V2.1 Stable Adaptation: Decision Memo

## Decision

The experiment validates the implementation and shows that V2.1 selectors can learn on top of a frozen, task-trained physical/shared-expert backbone. It does **not** show a stable, cross-dataset classification benefit from staged selector adaptation. On ele-fashion, all Best-Accuracy checkpoints remain at the shared C0 `t0`; W1/W2 therefore avoid the historical cold-start regression in this selection view by retaining C0, not by demonstrating a gain from adaptation. The three-seed differences are descriptive and do not establish statistical significance.

A separate, tightly controlled study of receiver-conditioned message-content transformation is worth considering as a hypothesis. It should be evaluated from matched C0 starts, with the current selector protocol as a comparator and without changing V2.1 in this study. This memo does not implement that model.

## Answers to the study questions

1. **Which ele-fashion classes drove the old cold-start decline?**
   - At historical Best Accuracy, C1−C0 F1 losses are largest for class 11 (−10.1 pp) and class 1 (−4.5 pp), followed by classes 7, 3, and 4. C2−C0 is also led by classes 11 (−10.6 pp) and 1 (−4.1 pp).
   - At historical Best F1, the largest losses are classes 1 (−4.7 pp), 7 (−3.8 pp), and 11 (−2.9 pp). Details are in `CLASSWISE_DIAGNOSTICS.md` and `data/classwise_metrics.csv`.

2. **Was there a Precision–Recall shift?**
   Yes in the historical cold-start Best-Accuracy comparison for classes 11 and 1: precision rises by about +18.1/+8.0 pp while recall falls by about −8.7/−6.1 pp, alongside fewer predictions for those classes. That is consistent with a changed class prediction preference. Warm-start classwise changes are much smaller and mixed; their per-seed values are preserved in `data/warmstart_classwise_metrics.csv`.

3. **Was the old decline concentrated on degree-1 nodes?**
   No. Historical ele-fashion Best-Accuracy F1 changes were about −1.41 pp for degree 1 versus −2.44 pp across degree ≥2, with degree 2–3 the worst subgroup (about −3.74 pp). Degree-1 nodes cannot perform relative choice among multiple physical neighbors; their output changes are indirect through shared parameters or multi-hop propagation. The warm Best-F1 comparison is small: W1−W0 degree-1 F1 is −0.13 pp versus +0.21 pp for degree ≥2; W2−W0 is −0.04 pp versus +0.03 pp. These are descriptive subgroup differences, not interventions.

4. **What does the cold-start Train Macro-F1 decline say, and what does it not say?**
   At the historical selected Accuracy checkpoints, Fashion C1/C2 Train Macro-F1 was about 0.803/0.802 versus 0.871 for C0, with earlier selected Accuracy epochs (124/118 versus 176) and lower Validation Macro-F1. This shows that the trained C1/C2 states achieved lower measured training-set class balance as well as lower Validation F1 in that protocol. It is compatible with an optimization/co-adaptation problem. It does not prove that neighbor selection itself is harmful, identify a unique cause, or isolate initialization, training budget, optimizer state, or selection effects.

5. **Did W1/W2 selectors train on a stable physical backbone?**
   Yes. All 18 W1/W2 Stage 1 runs preserved public-model and classifier state hashes through 12 epochs; all 18 updated the scorer. Each of the nine W2 runs also had a nonzero cross-delta gradient and update. W1/W2 initialization was matched per Dataset × Seed and t0 logits were within `1.79e-6` maximum absolute difference from W0. Stage 1 used Eval mode with autograd enabled.

6. **Did staged adaptation improve ele-fashion Macro-F1?**
   It did not yield a clear internal improvement. Best Accuracy stayed at t0 for W0/W1/W2 for all seeds. At Best F1, W1−W0 is +0.029 pp Macro-F1 and W2−W0 is −0.016 pp; W2−W1 is −0.045 pp. These tiny three-seed differences do not support a robust gain. Against historical cold C1/C2, warm W1/W2 Best-Accuracy Macro-F1 is higher by about 1.80/1.73 pp, but warm models select the inherited C0 checkpoint in all those rows. That is not evidence that selector adaptation caused recovery.

7. **Were Movies/Grocery selective-neighborhood gains preserved?**
   Not consistently across checkpoint criteria. On Movies Best F1, W1−W0 is +0.136 pp Macro-F1 and W2−W0 is +0.123 pp, while Best Accuracy W2−W0 is −0.250 pp Macro-F1 and −0.090 pp Accuracy. On Grocery Best Accuracy, W2−W0 is +0.181 pp Macro-F1 and +0.020 pp Accuracy, while W1−W0 is −0.059 pp Macro-F1 with essentially zero Accuracy change. Grocery Best-F1 comparisons are slightly negative versus W0. No cross-dataset, cross-checkpoint preservation claim is warranted.

8. **Did W2 add useful gains over W1?**
   There are isolated small changes, not consistent incremental value. Grocery Best Accuracy W2−W1 is +0.241 pp Macro-F1; Movies Best Accuracy is −0.299 pp; Fashion Best Accuracy is identical because all select t0. At Best F1, W2−W1 ranges from −0.045 pp on Fashion to −0.014 pp on Movies and −0.019 pp on Grocery. Cross-delta gradients and updates verify that the module is active, but activity did not translate into a repeatable metric gain here.

9. **How much did W0 continuation contribute?**
   Relative to t0 at Best Accuracy, mean W0 changes were Movies +0.23 pp Accuracy/+1.23 pp Macro-F1, Grocery +0.05/+0.18 pp, and ele-fashion 0/0 pp. Thus the continued-training/restarted-AdamW control improves the selected Movies and Grocery metrics modestly on average, while Fashion keeps t0. The separate Best-F1 states can trade Accuracy for Macro-F1 even in W0.

10. **Were score saturation, router concentration, or weak cross-modal residuals present?**
    - At Best-Accuracy checkpoints selector saturation was rare: mean/max boundary fractions were 0.000607/0.01092 for W1 and 0.000095/0.00342 for W2. Stage 1 had no score at `|score| ≥ 0.99`.
    - Grocery Visual routing remains extremely concentrated (maximum expert probability about 0.9997 and entropy about 0.0023), but this was already present in inherited C0 and remains in W0/W1/W2.
    - W2 cross-delta residual contribution is small relative to the inherited cross-modal base (about 0.008 raw RMS and 0.0033 weighted RMS versus about 0.55 and 0.28 for the base). This is a scale observation, not proof that the residual is ineffective.

11. **Does the evidence favor co-optimization difficulty or insufficient scalar-edge expressiveness?**
    The historical cold-start class/Train-F1 pattern and the stable-backbone Stage 1 results make optimization interaction a plausible concern. However, this study did not isolate its cause: Stage 1 learned only modest redistributions, and Best-Accuracy often retained t0. Nor did it directly test the representational ceiling of scalar edge reweighting. The evidence is insufficient to choose one explanation as established.

12. **Is receiver-conditioned message-content transformation worth studying?**
    Yes, as a controlled next hypothesis, because V2.1 mainly redistributes scalar edge influence and its learned residuals remain small next to inherited physical/cross-modal contributions. A follow-up should compare the content transform against matched W0 and W1/W2 from identical C0 starts, log per-class/degree outcomes, and keep the transform’s capacity and optimizer effects controlled. It should not be described in advance as a fix, and this experiment did not implement it.

## Evidence and limits

The 27-run matrix completed with finite Train/Validation metrics, 54 selected checkpoint save/reload checks, and Test evaluation disabled. Three implementation/analysis/export failures are retained in `data/failures.json`; each was corrected and documented without rewriting historical results. The full suite passed 512 tests with 1 skipped. See `IMPLEMENTATION_AUDIT.md`, `TRAINING_DYNAMICS.md`, and `ADAPTATION_REPORT.md` for protocol, paired tables, and traceability. No Test evaluation, additional HPO, V2.2, or extra dataset was run.
