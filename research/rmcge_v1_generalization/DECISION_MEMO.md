# RM-CGE V1 Generalization Study: Decision Memo

## Decision

Proceed under the researcher-approved revised admission standard. Both historical Movies/O0/seed-42 replays remain recorded as `historical_full_training_replay_not_exact`; their score, stopping epoch, and curve differences do not block the experiment. The revised preflight passed. Commit the Implementation Freeze, run the 12 seed-42 comparisons, inspect run integrity and matched initialization, then continue seeds 43 and 44 if that review passes. O0 conclusions will use the new internal O0 runs only.

## Evidence by question

| Question | Current evidence | Decision |
|---|---|---|
| Can regularization improve V1 generalization? | The 36-run internal matrix is pending. | Unknown until paired O0–O3 runs complete. |
| Is the Router node-conditioned? | Frozen checkpoints show node-level Semantic route-mass SD around 0.02–0.17 and different expert mixtures. | There is node-conditional variation; reliability or usefulness is not established. |
| Which cross-modal path matters? | Frozen single-path interventions hurt Movies/Grocery more when Physical cross-modal context is removed; ele-fashion changes are small and mixed. | Treat this as fitted-path sensitivity for three checkpoints, not a retrained causal ablation. |
| Are raw cross-modal contexts suppressed? | Raw RMS is lower, but the independently LayerNormed blocks are both near unit RMS. | Raw scale alone does not establish weaker Expert input. |
| Which architecture should change next? | No seed-level regularization comparison, and only one frozen checkpoint per dataset. | No architecture choice is supported yet. Reliability-aware routing, zero-update handling, semantic relation processing, and expert/fusion changes remain untested hypotheses. |

## Reproducibility recommendation

The operational standard is implementation correctness, protocol identity, paired internal fairness, finite training/validation values, RNG-safe diagnostics, Test isolation, checkpoint reload consistency, and provenance. The two-update parity result (`5.05e-7` maximum parameter error) supports short-run numerical agreement but cannot prove identical full training trajectories. Python/NumPy/PyTorch/CUDA/cuDNN/driver/GPU, TF32 and deterministic settings are recorded; the formal runtime is not changed to force replay. Any unexplained seed-level differences or optimization issues will be reported with their evidence boundaries.

No Test evaluation, new hyperparameter search, formal retrained model ablation, or V1.1 implementation is authorized in this study.
