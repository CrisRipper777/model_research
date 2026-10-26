# Decision handoff

Recommendation: **C**.

Movies has a nonzero higher-order residual in every seed and modality (nu2 >= 0.30; nu3 conditional on X2 >= 0.12), while X2 remains strongly aligned with the actual Late SameNode candidate (cosine >= 0.84). This supports Cross-Aligned Propagation as the next controlled question: align or gate the novel residual against the existing candidate before combining orders. The cutoffs are exploratory evidence summaries, not guarantees.

The full checkpoint-level data and missing-cell records are preserved for independent review. This recommendation uses no accuracy, Test metric, label, or cross-model hidden coordinate comparison. High linear novelty does not establish classification usefulness; low novelty does not rule out nonlinear expert benefits. Hybrid structural differences are not isolated causal effects.

Stage B was not executed. No training, checkpoint creation, or model architecture work was started. The next action is independent review of this branch's raw CSV/JSON evidence before any further experiment.
