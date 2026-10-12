# RM-CGE V1 Frozen Mechanism Diagnostics

## Evidence boundary

These are Validation-only interventions on the original Seed-42 Best-Accuracy checkpoints. They do not retrain the model. Setting a cross-modal context to zero changes the input observed by the Router; all same-modal supports, relation validity rules, expert parameters, Router parameters, and classifier weights remain loaded from the checkpoint. The measured route is the model's actual information flow, not an intervention that holds every downstream activation fixed.

No Test labels were indexed. Node counts below are descriptive units within a frozen checkpoint, not independent experimental replicates. Correlations describe associations with the Router's Semantic relation mass and do not imply labels or reliability.

## Frozen-checkpoint replay (Stage B)

- **Movies:** replay passed; best epoch 70; Validation Accuracy 0.540492; Macro-F1 0.429959; repeated full-graph embedding/logit max difference 1.1e-06/2.4e-06; archived all-cross-off replay passed.
- **Grocery:** replay passed; best epoch 91; Validation Accuracy 0.823133; Macro-F1 0.752066; repeated full-graph embedding/logit max difference 1.3e-06/2.9e-06; archived all-cross-off replay passed.
- **ele-fashion:** replay passed; best epoch 115; Validation Accuracy 0.879309; Macro-F1 0.741882; repeated full-graph embedding/logit max difference 1.7e-06/3.8e-06; archived all-cross-off replay passed.

This confirms the archived Seed-42 checkpoints and Stage B inference path. It does **not** mean the separate Stage C Movies/O0 training replay passed: that gate remains failed, so the 36-run matrix was not started. See `IMPLEMENTATION_AUDIT.md` and `GENERALIZATION_REPORT.md` for the training replay evidence.

## Relation-specific cross-modal interventions

| Dataset | Condition | Val CE | Δ CE | Val Acc | Δ Acc (pp) | Macro-F1 | Δ F1 (pp) | Prediction flips | Relative logit RMS Δ | Relative embedding RMS Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | original | 1.4463 | +0.0000 | 0.5405 | +0.00 | 0.4300 | +0.00 | 0.0000 | 0.0000 | 0.0000 |
| Movies | all_cross_modal_off | 1.4738 | +0.0274 | 0.5258 | -1.47 | 0.3984 | -3.16 | 0.0930 | 0.1264 | 0.1521 |
| Movies | physical_cross_modal_off | 1.4707 | +0.0244 | 0.5306 | -0.99 | 0.4122 | -1.78 | 0.0726 | 0.1008 | 0.1163 |
| Movies | semantic_cross_modal_off | 1.4459 | -0.0005 | 0.5387 | -0.18 | 0.4268 | -0.32 | 0.0378 | 0.0465 | 0.0583 |
| Grocery | original | 0.6871 | +0.0000 | 0.8231 | +0.00 | 0.7521 | +0.00 | 0.0000 | 0.0000 | 0.0000 |
| Grocery | all_cross_modal_off | 0.7750 | +0.0879 | 0.7880 | -3.51 | 0.7225 | -2.96 | 0.0770 | 0.2143 | 0.2466 |
| Grocery | physical_cross_modal_off | 0.7469 | +0.0598 | 0.7959 | -2.72 | 0.7328 | -1.93 | 0.0615 | 0.1599 | 0.1822 |
| Grocery | semantic_cross_modal_off | 0.6933 | +0.0062 | 0.8158 | -0.73 | 0.7470 | -0.51 | 0.0220 | 0.0685 | 0.0813 |
| ele-fashion | original | 0.4021 | +0.0000 | 0.8793 | +0.00 | 0.7419 | +0.00 | 0.0000 | 0.0000 | 0.0000 |
| ele-fashion | all_cross_modal_off | 0.4036 | +0.0016 | 0.8731 | -0.62 | 0.7328 | -0.91 | 0.0248 | 0.1066 | 0.1446 |
| ele-fashion | physical_cross_modal_off | 0.4033 | +0.0012 | 0.8780 | -0.13 | 0.7412 | -0.07 | 0.0126 | 0.0721 | 0.0964 |
| ele-fashion | semantic_cross_modal_off | 0.4001 | -0.0020 | 0.8776 | -0.17 | 0.7459 | +0.40 | 0.0113 | 0.0527 | 0.0705 |

The physical and semantic conditions zero only the corresponding relation's cross-modal context. They leave its same-modal context active. The all-cross-off baseline matches the previously archived frozen-checkpoint intervention within 2×10⁻⁶.

Observed validation sensitivity differs by dataset. On Movies and Grocery, removing Physical cross-modal context produces a larger Accuracy decline than removing Semantic cross-modal context. On ele-fashion, both single-relation changes are small and mixed across CE, Accuracy, and Macro-F1. These are fitted-path sensitivities; they do not estimate the benefit of retraining without a path.

## Node-conditioned routing

The table reports node-level Semantic relation mass by layer and modality (mean, population SD, and 5th–95th percentiles), together with mean joint-router entropy. Values are masked to nodes with the measured relation support.

| Dataset | Layer | View | Semantic mass mean ± SD | q05–q95 | Joint entropy | Top-1 expert proportions |
|---|---:|---|---:|---:|---:|---|
| Movies | 1 | text | 0.357 ± 0.054 | 0.241–0.420 | 1.690 | E0 0.74, E1 0.24, E2 0.01 |
| Movies | 1 | visual | 0.364 ± 0.024 | 0.321–0.402 | 1.680 | E0 0.73, E1 0.16, E2 0.11 |
| Movies | 2 | text | 0.344 ± 0.071 | 0.226–0.452 | 1.560 | E0 0.67, E1 0.28, E2 0.05 |
| Movies | 2 | visual | 0.358 ± 0.057 | 0.260–0.452 | 1.504 | E0 0.36, E1 0.36, E2 0.28 |
| Grocery | 1 | text | 0.321 ± 0.118 | 0.114–0.519 | 1.235 | E0 0.52, E1 0.16, E2 0.31 |
| Grocery | 1 | visual | 0.352 ± 0.055 | 0.255–0.443 | 1.594 | E0 0.20, E1 0.33, E2 0.47 |
| Grocery | 2 | text | 0.328 ± 0.111 | 0.156–0.523 | 1.245 | E0 0.31, E1 0.12, E2 0.57 |
| Grocery | 2 | visual | 0.329 ± 0.094 | 0.165–0.484 | 1.282 | E0 0.26, E1 0.52, E2 0.23 |
| ele-fashion | 1 | text | 0.403 ± 0.118 | 0.220–0.619 | 1.068 | E0 0.16, E1 0.35, E2 0.49 |
| ele-fashion | 1 | visual | 0.385 ± 0.130 | 0.188–0.635 | 1.079 | E0 0.51, E1 0.47, E2 0.02 |
| ele-fashion | 2 | text | 0.435 ± 0.167 | 0.212–0.792 | 1.006 | E0 0.46, E1 0.14, E2 0.40 |
| ele-fashion | 2 | visual | 0.376 ± 0.098 | 0.217–0.541 | 1.339 | E0 0.11, E1 0.69, E2 0.19 |

Relation mass varies across nodes: observed Semantic-mass SD spans about 0.02–0.17 across these checkpoint/layer/view groups, with especially broad ranges in ele-fashion. This rules out describing the Router as emitting only a fixed 0.6/0.4 ratio on each node. The entropy and Top-1 expert distributions show that node variation coexists with different degrees of expert concentration. They do not establish that the Router identifies a reliable source: mass-to-neighborhood correlations are often small, with larger associations in some ele-fashion views, and remain observational within one frozen graph.

The machine-readable table contains all quantiles, validity-masked relation masses, top-1 expert proportions, paired Text/Visual mass comparisons, and descriptive Pearson/Spearman coefficients with Semantic mass. These coefficients are reported without p-values and do not imply labels or relation reliability.

## Context scales

Raw and post-block-LayerNorm context RMS, together with low-variance node fractions below 1e-6, are in `data/cross_modal_scales.csv`. Raw cross-modal RMS is lower than same-modal RMS, while both independently normalized blocks are near unit RMS. Therefore, the raw magnitude difference is removed before the Expert input LayerNorm; raw scale alone would understate the normalized cross-modal block's scale.

The check files contain frozen checkpoint hashes, cache fingerprints, and recomputed embedding/logit SHA-256 values. Original Pilot embedding/logit arrays were not persisted, so hashes establish deterministic recomputation in this study rather than a bytewise comparison against an archived tensor artifact.
