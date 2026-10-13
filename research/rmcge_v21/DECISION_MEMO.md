# RM-CGE V2.1 Decision Memo

## Facts

- Movies: C1−C0 Validation Accuracy mean paired change +0.21 pp (sample SD 0.05; seed signs 3/3); C2−C1 -0.27 pp; C2−C0 -0.06 pp.
- Grocery: C1−C0 Validation Accuracy mean paired change +0.25 pp (sample SD 0.06; seed signs 3/3); C2−C1 -0.22 pp; C2−C0 +0.03 pp.
- ele-fashion: C1−C0 Validation Accuracy mean paired change -0.12 pp (sample SD 0.27; seed signs 1/3); C2−C1 -0.11 pp; C2−C0 -0.23 pp.
- Learned path selection has mean |p_select/p−1| 0.3775 and mean receiver JS(P_select,P) 0.0290 over measured paths and seeds; row-mass errors are separately audited per run.
- Replacing each receiver query with its modality's active-node mean changes the trained selector distribution by mean receiver JS 0.0151; this is a frozen mechanism probe, not a retrained ablation.
- Receiver-wise conservation held to a maximum absolute row-mass error of 1.20e-04; enabled selection changed relative coefficients by mean |ratio−1| 0.378 and mean receiver JS 0.029.
- The run completed 27/27 Dataset × Condition × Seed combinations, with 0 failed attempts. No Test metrics or Test-indexed label fingerprints were created.
- C0 delegates to the V2/V1 forward. C1/C2 common model parameters, added selector state and classifier weights passed the paired initialization fingerprint audit. Synthetic receiver-feature interaction reversed the ranking of identical candidates for two targets.

## Answers to the frozen research questions

- **Movies: paired model-level results.** Validation Accuracy means are +0.21 pp and -0.27 pp; Macro-F1 means are +0.64 pp and +1.31 pp.
- **Grocery: paired model-level results.** Validation Accuracy means are +0.25 pp and -0.22 pp; Macro-F1 means are +0.62 pp and +0.01 pp.
- **ele-fashion: paired model-level results.** Validation Accuracy means are -0.12 pp and -0.11 pp; Macro-F1 means are -1.80 pp and +0.07 pp.
1. **Does C1 improve C0 at model level?** On Movies and Grocery, mean paired Accuracy improves by +0.21 pp and +0.25 pp, with Macro-F1 +0.64 pp and +0.62 pp. On ele-fashion, Accuracy is −0.12 pp and Macro-F1 −1.80 pp. This is a dataset-dependent signal, not a general improvement.
2. **Does C2 add benefit over C1?** C2−C1 Accuracy is −0.27 pp on Movies, −0.22 pp on Grocery and −0.11 pp on ele-fashion. Macro-F1 changes are +1.31 pp, +0.01 pp and +0.07 pp, respectively. The result depends on metric and dataset.
3. **Does the model truly change relative neighbor choice?** Yes. Across enabled paths, mean |p_select/p−1| is 0.378 and mean receiver JS from physical P is 0.029, while total physical coefficient mass is conserved to 1.20e-04 absolute error. Selection changes within-row allocation rather than global message strength.
4. **Does the receiver representation affect ranking?** The synthetic test proves capacity for target-dependent rank reversal. On trained checkpoints, replacing receiver queries with their modality-wide active-node mean gives mean receiver JS 0.0151 (per data/condition in `MECHANISM_REPORT.md`). This is measurable but modest sensitivity; it does not prove retrained causal benefit.
5. **Do Intra and Cross choices differ by modality and task?** Yes: measured path-pair receiver JS ranges from 0.0177 to 0.1055; matched edge-probability differences and correlations vary by dataset and pair. The full comparison is in `MECHANISM_REPORT.md`; no single cross-dataset pattern dominates.
6. **Do selection residuals reach shared experts?** Intra and Cross residuals are nonzero in their enabled conditions. Coefficient-weighted per-expert RMS is reported by dataset and condition in `MECHANISM_REPORT.md`. C2 Cross Delta also feeds the existing Node Router input. Gradient/update records cover the selector, Cross Projection, experts, Router and Fusion.
7. **What is their fitted-checkpoint prediction effect?** Frozen interventions yield C1 Intra forced-physical Accuracy changes: Movies -0.56 pp Acc; Grocery -0.03 pp Acc; ele-fashion -0.03 pp Acc. C2 forced-physical Intra/Cross/both changes are Movies: Intra -0.45 pp, Cross -0.16 pp, both -0.48 pp Accuracy; Grocery: Intra -0.11 pp, Cross -0.06 pp, both -0.13 pp Accuracy; ele-fashion: Intra -0.04 pp, Cross -0.03 pp, both -0.06 pp Accuracy. The full table reports Macro-F1, logit RMS and prediction flips. These are checkpoint sensitivities, not retrained ablations.
8. **Does the unchanged Node Router show concentration or compensation?** Yes, a clear concentration exists for Grocery Visual in every condition: the largest mean dense probability is about 0.9996–0.9998 and route entropy is near zero. It is already present in C0, so this study does not attribute it to V2.1 selection. Movies routes are more distributed; ele-fashion is intermediate. Per-record Top-2 flip fractions range 0.000–0.999, showing substantial route changes in some groups. This is a real inherited pathology to investigate separately; the frozen experiment did not alter Router or balance loss.
9. **Which datasets appear suitable?** Movies and Grocery are the strongest candidates for Intra selection in this matrix because C1 shows small positive Accuracy and Macro-F1 paired means. ele-fashion is a poor fit under this configuration: C1 and C2 Macro-F1 are below C0 in all three paired seeds.
10. **Should both purposes be kept?** The evidence does not support keeping both as a universal default. C2 consistently increases Movies Macro-F1 over C1, while its Accuracy is lower across all datasets and Grocery/ele-fashion F1 is nearly flat. Keep C2 as a research condition for targeted follow-up rather than treating it as the preferred model.
11. **Should message transforms, propagation order, hops or routing change now?** No. Residuals reach shared experts and Cross selection follows the inherited sender projection. This study does not isolate a defect that justifies changing these components. Router calibration can be a separately justified future question.
12. **Does this justify a broader model or additional modules?** It supports the bounded V2.1 mechanism claim, but not a broad performance claim or another stacked model. The paired effects are mixed, and frozen interventions are not retrained ablations.

## Next evidence gap

A follow-up should focus on retrained purpose-specific ablation with adequate seeds where paired performance and frozen sensitivity agree, and explain the ele-fashion Macro-F1 decline alongside Router changes and the frequent near-bound scores on Fashion Visual paths. No Test evaluation is part of this evidence.

## Assumptions and evidence limits

- The official full-graph transductive protocol allows unlabeled graph and feature propagation; label supervision is restricted to Train indices. Validation selects checkpoints.
- Three seeds provide paired descriptive changes, not statistical significance. Frozen interventions are sensitivity probes and cannot identify causal contributions under retraining.
- Score bias gradients can be zero because a common per-receiver score offset cancels in the mass-preserving normalization. Sparse CUDA scatter may have backend-dependent accumulation order; this does not establish CUDA as the sole source of training variation.
- The report-template repair occurred after all training and changes only analysis/report generation; raw training records, checkpoints, initialization hashes and measured metrics are unchanged.

## Decision

Retain the implementation and complete evidence as a bounded mechanism study. It supports the claim that V2.1 performs receiver-conditioned, purpose-specific relative selection while conserving physical row mass; it does not support a generalization claim that the model improves every dataset or that C2 should replace C1. Keep later work targeted to the observed dataset/purpose effects and the stated evidence gaps.
