# V7B.1 corrective replication design

## Objective

Correct one parameter-permission error in V7B and rerun the intended six-arm, validation-only full-graph node-classification study. This is a correction to the frozen V7B experiment, not a new model family.

The baseline is `75157f052a50c60b791566b2c91207e372956d7b`. The working branch is `exp/v7b1_corrective_replication`.

## Corrective scope

`b0_v7a_original` inherits the complete V7A A2 relation path. Its two modality projections and shared Sigmoid scorer remain trainable. Only B1 and B2 freeze those modules because they use fixed edge weights. B3 and B4 retain their frozen V7B relative scoring implementations.

No physical propagation, semantic candidates, edge weights, randomization algorithm, relation dimensions/formula, experts, routers, gates, fusion, task split, optimizer, or stopping rule changes are included.

## Experimental arms

| Arm | Model | Semantic topology/weight |
|---|---|---|
| R0 | V7A `a0_raw` | no semantic graph |
| B0 | corrected V7B `b0_v7a_original` | V7A A2 learned shared Sigmoid relation |
| B1 | V7B `b1_fixed_knn` | KNN union, fixed weight 0.8 |
| B2 | V7B `b2_random_fixed` | reused degree-preserving random topology, fixed weight 0.8 |
| B3 | V7B `b3_relative_shared` | source-aware relative weight shared across modalities |
| B4 | V7B `b4_relative_modality` | source-aware modality-specific relative weights |

The full campaign uses Movies, Grocery, and ele-fashion; training seeds 42, 43, and 44; six arms per dataset/seed. Checkpoints are selected by validation accuracy. Macro-F1 is diagnostic. Test evaluation is disabled.

## Reproducibility

The campaign launcher checks the baseline ancestry, freeze SHA, source and candidate fingerprints, task configuration, checkpoints, and the absence of Test metric keys. B2 reuses the V7B topology caches with the original seed mapping 42→2026, 43→2027, and 44→2028. R0 follows the V7A raw path and does not load a semantic cache.

See `CORRECTIVE_AUDIT.md`, `FEASIBILITY_REVIEW.md`, and `data/` for the recorded checks. Formal campaign status is recorded in `data/campaign_manifest.json`.
