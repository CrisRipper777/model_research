# MR-CGE V1.3 Physical Basis Refinement — Design Freeze

## Scope

This study begins from `4864db09356fb3a426c03dc1e2c097af5342aebb` on `exp/mrcge_v13_view_cross_screen`, on branch `exp/mrcge_v13_physical_basis_screen`. It authorizes exactly these seven new training cells:

| Model | Condition | Dataset | Training seed | Split |
|---|---|---|---:|---|
| Hybrid-CGE paper V1 | H2_physical_3hop | Movies | 43 | frozen seed42 |
| Hybrid-CGE paper V1 | H2_physical_3hop | Movies | 44 | frozen seed42 |
| MR-CGE V1.3-A | joint_view | Movies | 44 | frozen seed42 |
| MR-CGE basis screen | joint_view_physical_no_p0 | Movies | 42 | frozen seed42 |
| MR-CGE basis screen | joint_view_physical_no_p0 | Movies | 43 | frozen seed42 |
| MR-CGE basis screen | joint_view_physical_no_p0 | Grocery | 42 | frozen seed42 |
| MR-CGE basis screen | joint_view_physical_no_p0 | Grocery | 43 | frozen seed42 |

The Movies split fingerprint is `2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d`. Grocery uses the existing seed42 split recorded in the study configuration and checked against the frozen V1.3-A history. The split path is resolved from the project MAGB configuration through the existing loader path resolver. Automatic split generation is disabled.

## NoP0 definition

The new model reuses the V1.3-A `M_repaired` / `joint_view` model and the same learned `theta[4,4]`, four experts, Router, Joint View Contribution, Cross, Late, Gate, Fusion and losses. For Physical filtering only, order zero is masked before signed L1 normalization:

`alpha_P[k,0] = 0`, and `alpha_P[k,1:4] = theta[k,1:4] / (sum(abs(theta[k,1:4])) + eps)`.

The propagated basis remains `[H, PH, P²H, P³H]`; order zero remains available to generate higher propagation orders and to Semantic filtering. Semantic continues to use orders 0–1. `theta[:,0]` remains a shared, trainable parameter through Semantic order zero. The identical Physical mask applies to Private and Collaborative Physical expert calls, including the low-memory checkpointed training path and evaluation/diagnostic expert-stack path.

No other parameterization, coefficient, cross path, data, loss, optimizer or hyperparameter changes are part of this screen. The compatibility switch `joint_view_original` exists only for equivalence checks; it is not a training cell.

## Training and stopping rules

All cells use the existing full-graph node-classification protocol, AdamW (`lr=0.001`, `weight_decay=0.0001`), 300 maximum epochs, patience 30, minimum epoch 30, early-stop delta `0.0001`, gradient clip 1.0, dropout 0.2 and late dropout 0.15. Best Validation Accuracy is the primary checkpoint; Best Macro-F1 is retained separately. Test labels are set to `-100` before Trainer entry and Test evaluation remains off.

For Hybrid-CGE, the corrected H2 3-hop wrapper is used with the complete early Cross, SameNode and Semantic Late completion, MoE, Router, private auxiliary CE and original settings. A positive SameNode support/candidate/source-weight audit is required in the real-data GPU preflight and final checkpoint diagnostics.

## Boundaries

Only physical GPU 1 is authorized, under a fresh launch-time ownership/resource guard and runtime monitoring. Each cell is isolated in its own process. New foreign GPU compute PIDs, insufficient free memory, high utilization or unsafe temperature stop only this task's child and are recorded. A model-specific single-step preflight must establish peak reserved memory; training launch requires that measured peak plus at least 2 GiB free, with an 8 GiB minimum.

No Test, link prediction, Static-Strength, additional physical-order search, cross redesign, ele-fashion NoP0 or later model version is authorized. Results are descriptive; fixed split and seed improve comparability, but different model parameterizations do not provide paired parameter initialization or a strict causal ablation. No final model winner will be selected here.
