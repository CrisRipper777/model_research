# RM-CGE V1 Generalization Study: Implementation Audit

## Scope and revised status

**Researcher-approved revision (2026-10-13):** Exact historical Pilot trajectory replay is no longer an admission gate. The two prior O0 attempts remain immutable records classified as `historical_full_training_replay_not_exact`; they are references, not internal baselines. The study is reopened for implementation audit, an Implementation Freeze commit, and the paired internal matrix.

The study branch is `exp/rmcge_v1_generalization_study`, based on Pilot Report commit `7e4b3dedb9de419ecec4193a1c2e50efd80dc3ab`. The revised preflight passed; the run-level Checkpoint reload audit is part of each formal run. The Implementation Freeze commit is next.

The model change is limited to allowing dropout 0.20 or 0.35 and adding opt-in node diagnostics and relation-specific cross-modal interventions. The default V1 forward path at dropout 0.20 retains the same core operators, initialization, relation propagation, router, expert bank, and fusion. Training changes are behind `record_extended_validation`; they add train/validation diagnostics and a separate Best-F1 checkpoint while preserving Accuracy-driven stopping. Test evaluation is disabled for the study.

## Validation performed

- Required RM-CGE, PACE-MAG, V7A/V7B regression suites and the new study suite: **106 passed, 1 skipped** (latest run 5.70 s; 15 existing dependency/deprecation warnings).
- Python compilation and `git diff --check` passed for the study scripts and sources.
- Preflight verified official split-file, feature, physical-edge, semantic-cache and train/validation-label fingerprints for all three datasets. All four treatments had identical model and classifier initialization for each of the nine Dataset × Seed blocks. Frozen O0 controls and dataset configurations match the Pilot; only CPU-side train/validation diagnostics, Best-F1 bookkeeping, provenance and isolated output paths are added. Model/classifier parameter counts match the archived Movies O0 reference: 892,242 / 2,580.
- Stage B replay reproduced the archived Validation metrics for all three Seed-42 checkpoints; repeated full-graph GPU inference varied by only about 1–4 × 10⁻⁶ in maximum embedding/logit values. The per-relation interventions, valid-mask-aware router summaries, and context-scale audit were generated without Test labels.
- A two-update parity check loaded the frozen model class directly from commit `8b94d16c3b2a66b658419c7f79e0de688b51a23a`. Initialization and classifier state were identical. Both versions had equal first- and second-step loss; maximum parameter difference after each update was `5.0478e-7`. The additional CPU-only train/validation head diagnostics preserved CPU and CUDA RNG states.

## Historical full-training replay: not exact, nonblocking

The reference Movies/O0/seed-42 run completed 100 epochs, selected epoch 70, and reported Validation Accuracy `0.540492` and Macro-F1 `0.429959`.

| Study attempt | Epochs | Best epoch | Val Accuracy | Val Macro-F1 | Train CE RMSE vs reference | Val Acc RMSE | Val F1 RMSE | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `seed_42` | 100 | 70 | 0.537493 | 0.436371 | 0.03011 | 0.00390 | 0.00714 | Failed |
| `seed_42_attempt_02` | 97 | 67 | 0.540792 | 0.435800 | 0.01082 | 0.00443 | 0.00721 | Failed |

Both trajectories remain highly correlated with the reference, but validation metrics diverged from about epochs 31–33 onward, and the second attempt stopped three epochs earlier. These are observed historical differences, not corrected or rewritten results. Their records remain in `data/o0_attempt_02_comparison.json`, `data/o0_replay_attempts.json`, and `data/failures.json`.

The parity result shows identical initial states and step losses, a two-step parameter difference of `5.0478e-7`, and no RNG-state change from the CPU-only extra metrics. Repeated CUDA sparse aggregation differed around 10⁻⁶ in the earlier probe. These facts make floating-point amplification plausible, but they do not establish CUDA as the sole cause. Python/NumPy/Torch/CUDA/cuDNN/driver/GPU, TF32 and deterministic settings are now captured in preflight; the formal runtime settings are not changed to pursue exact historical replay.

## Engineering exceptions and next gate

An initial parity invocation landed on a GPU occupied by the VLLM engine and ran out of memory. That diagnostic failure is retained; the successful parity retry explicitly used physical GPU 1. One retry attempt also encountered the existing `seed_42` output directory and launched no training; its record has been corrected to show that no directory was written by that invocation. Both actual O0 runs are preserved in their independent output directories.

After the 12-run seed-42 review passed, the first seed-43/44 launch stopped before training because the shared Git helper stripped the leading status-column whitespace and corrupted one modified path. The worktree parser now reads raw porcelain output, with a regression test; the complete suite passes **107 passed, 1 skipped**. This follow-up changes only the launch guard. The model, NC training, configuration, optimizer, and metric code remain at Implementation Freeze `e002af6a9c57de21986129d2b1c6e8478bcc74e3`; seed-43/44 provenance will record the follow-up runner commit.

The revised hard checks are configuration and optimizer parity, official split/data/cache fingerprints, per-seed initialization hashes, short-run numerical parity, RNG preservation, finite histories, no Test-label indexing, and successful Validation-only Checkpoint reloads. The Implementation Freeze precedes training. The first batch contains all 12 seed-42 runs; seeds 43 and 44 follow only after that batch passes integrity review.

## Evidence files

- `data/implementation_checks.json`
- `data/o0_replay_attempts.json`
- `data/o0_step_parity.json`
- `data/o0_step_parity_attempts.json`
- `data/o0_attempt_02_comparison.json`
- `data/failures.json`
- `data/frozen_checkpoint_replay_checks.json`
- `data/frozen_cross_relation_interventions.csv`
- `data/router_node_statistics.csv`
- `data/cross_modal_scales.csv`
