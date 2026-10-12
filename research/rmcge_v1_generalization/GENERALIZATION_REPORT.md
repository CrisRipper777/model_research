# RM-CGE V1 Generalization & Optimization Study: Status Report

## Study status

The researcher revised the admission standard on 2026-10-13. Exact historical Pilot trajectory, best Epoch, stopping horizon, and validation scores are no longer hard gates. The prior replay attempts remain preserved below and in machine-readable records. The study is reopened for implementation audit, Implementation Freeze, and the internal O0–O3 matrix.

The code, configurations, tests, and study-specific outputs are on branch `exp/rmcge_v1_generalization_study`. The revised preflight passed on 2026-10-13: official splits, feature/graph/cache fingerprints, O0 control configuration, initialization pairing, short-step parity, Test isolation, and regression tests were verified. The Implementation Freeze commit is the next gate; the 12-run seed-42 batch has not yet begun. Formal treatment conclusions are pending.

## O0 reproduction evidence

The archived reference run used the frozen RM-CGE V1 source and completed 100 epochs. It selected epoch 70 with Validation Accuracy `0.540492` and Macro-F1 `0.429959`. Model and classifier parameter counts were 892,242 and 2,580.

| Run | Completed epochs | Best epoch | Val Accuracy | Val Macro-F1 | Train CE RMSE | Val Accuracy RMSE | Val Macro-F1 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Archived Pilot | 100 | 70 | 0.540492 | 0.429959 | — | — | — |
| O0 attempt 1 | 100 | 70 | 0.537493 | 0.436371 | 0.03011 | 0.00390 | 0.00714 |
| O0 attempt 2 | 97 | 67 | 0.540792 | 0.435800 | 0.01082 | 0.00443 | 0.00721 |

The attempt-1 curve correlations with the archived trajectory were 0.99955 for Train CE, 0.99714 for Validation Accuracy, and 0.99842 for Macro-F1. Attempt 2 correlations were 0.99982, 0.99636, and 0.99841. Despite these high correlations, epochwise discrepancies exceed the former exact-replay tolerance, selected metrics differ, and attempt 2 has a different stopping horizon. These runs are classified as `historical_full_training_replay_not_exact`; they are not the internal O0 baseline and do not block the revised study. Paired factorial conclusions will use only O0–O3 runs trained under the same committed code and protocol.

The maximum absolute epoch-trajectory differences from the archived run were, for attempt 1, `0.0806655` Train CE, `0.0188962` Validation Accuracy, and `0.0290976` Macro-F1; for attempt 2, they were `0.0383927`, `0.0209958`, and `0.0263047`, respectively. These values come from the retained common-epoch comparisons. They describe replay discrepancies and do not define acceptance thresholds for the new matrix.

The two-update frozen-source parity check found identical initialization and loss values, a maximum parameter difference of `5.05e-7` after each update, and no CPU/CUDA RNG-state change from the additional CPU-only validation metrics. This is evidence against a material first-step instrumentation change, but it does not establish full-run reproducibility. Repeated CUDA sparse operations vary around 10⁻⁶; long-horizon amplification is plausible but not isolated. No further full historical replay was run, and no formal runtime determinism or TF32 setting was changed.

## Revised correctness and reproducibility standard

Admission now depends on matching official data, features, semantic cache and protocol; matching O0 model/configuration/initialization, Forward and optimizer semantics; reasonable short-run Forward/Backward/AdamW parity; RNG-invariant train/validation diagnostics; Test evaluation disabled without Test-label indexing; finite histories; successful checkpoint reloads; and identical O0–O3 initial model/head states within each Dataset × Seed. Preflight matched all frozen O0 controls and dataset configurations. The documented additions are CPU-side train/validation metrics, Best-F1 bookkeeping, provenance, and isolated output paths. The former exact best-Epoch, 100-epoch, score, and full-curve checks are historical evidence only.

The recorded preflight environment was Python 3.12.13, NumPy 2.4.3, PyTorch 2.4.0+cu121 (CUDA build 12.1), cuDNN 9.1, NVIDIA driver 535.309.01, and NVIDIA GeForce RTX 3090 on physical GPU 1. CUDA matmul TF32 was off, cuDNN TF32 was on, float32 matmul precision was `highest`; deterministic algorithms and cuDNN deterministic mode were off, cuDNN benchmark was off, and `CUBLAS_WORKSPACE_CONFIG` was unset. The run uses standard Dropout with per-run Python/NumPy/Torch/CUDA reseeding and activation checkpointing with RNG preservation. Added Train/Validation evaluation reuses cached eval embeddings on CPU and consumes no extra model forward or RNG draw. Sparse CUDA propagation remains a possible source of small numerical variation, not an established sole cause.

## Frozen-checkpoint mechanism results

Stage B completed independently using the original three Seed-42 Best-Accuracy checkpoints. Their archived Validation metrics and all-cross-modal-off intervention were reproduced. No Test labels were indexed.

- Removing Physical cross-modal context reduced Validation Accuracy more than removing Semantic cross-modal context on Movies and Grocery. The single-path effects on ele-fashion were small and mixed.
- Semantic route mass varied across nodes, with groupwise SD from roughly 0.02 to 0.17. This shows node-conditional variation, while not establishing that the selected relation is reliable.
- Raw cross-modal context RMS was lower than same-modal RMS, while both blocks were near unit RMS after their independent LayerNorms.

These are frozen-checkpoint sensitivities and descriptive node statistics, not retrained ablations or independent seed-level evidence. Detailed tables and their limits are in `MECHANISM_REPORT.md`.

## What remains unanswered

Because the matrix did not run, the study cannot determine whether higher Dropout or Weight Decay improves validation performance, whether they interact, whether any setting is more stable over three seeds, or whether training adaptation closes the V1 performance gap. Best-F1 selection was implemented and covered by tests, but no complete matrix-level checkpoint trade-off is available.

The internal O0–O3 matrix will determine descriptive regularization contrasts. This study does not authorize V1.1 or changes to the Router, experts, relations, or fusion.
