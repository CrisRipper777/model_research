# Decision Memo

## Decision basis

This memo uses only the internal A0/A1/A2 Validation comparisons paired by dataset and seed. Historical V7A/RM-CGE results are design context, not an exact score target. Three seeds per dataset support descriptive replication only; no significance claim is made.

## Measured paired results

- `A1_separate_intra-A0_legacy`: Accuracy mean Δ 0.058 pp (6/9 positive pairs); Macro-F1 mean Δ 0.419 pp (5/9 positive pairs).
- `A2_cross_source-A1_separate_intra`: Accuracy mean Δ 0.043 pp (6/9 positive pairs); Macro-F1 mean Δ -0.044 pp (4/9 positive pairs).
- `A2_cross_source-A0_legacy`: Accuracy mean Δ 0.101 pp (5/9 positive pairs); Macro-F1 mean Δ 0.375 pp (6/9 positive pairs).

## Interpretation

A1−A0 is the complete condition comparison for separating source-specific KNN context into its own semantic view. A2−A1 adds the cross-source support view. These contrasts do not isolate gamma, a particular expert, or an independent retraining intervention. Interpret any aggregate improvement alongside dataset-specific pairs, best-F1 checkpoints, Train/Validation gaps, classwise and degree-stratified results, view contribution scales, and GPU cost.
The view/router diagnostics establish that the implementation executed with measured nonzero propagation and expert responses where supports exist; they do not by themselves prove classification value. Frozen view closure is checkpoint sensitivity. It is not a retrained causal ablation.

## Scope limits

No Test metrics were produced. The study does not estimate a universal effect beyond these datasets/splits/seeds, and does not separately identify optimization, stochastic, or class-prior causes of run variation. No new model version is prescribed by this memo.

## Evidence categories

- **Code facts:** A0 uses the legacy V2.2 M0 forward; A1/A2 separate cached source-specific KNN context into shared-expert semantic views, with A2 adding opposite-source KNN support. All views call the same four experts and share one S/C router; Test labels are masked and the analyzer rejects Test metrics.
- **Measured validation facts:** all 27 runs completed with matched initial model/head hashes per Dataset × Seed, official split/cache fingerprints, finite metrics and 54 hashed checkpoints. Paired estimates and dataset/seed results in `PILOT_REPORT.md` are validation evidence only.
- **Unproven explanation:** router entropy, contribution scales, expert corrections and frozen-checkpoint flips show execution and sensitivity; they do not establish per-view retraining causality or a robust performance mechanism. Three seeds do not support broad population claims.
- **Execution history:** two Movies/A0/seed42 failures were retained before a successful attempt; a GPU-idle guard paused seed44 while preserving an independent GPU job. All events are itemized in `data/failures.json`; failed attempts do not count as completed scores.
