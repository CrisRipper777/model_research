# RM-CGE V1 Generalization Study Implementation Audit

## Frozen boundary

The RM-CGE V1 model math, initializers, projectors, relation operators, relation-specific same/cross support, shared experts, joint masked Softmax, residual scale, fusion, and CE objective remain unchanged. The model source changes only the legal dropout set (`0.20`, `0.35`), read-only diagnostic output, and relation-specific cross-context zeroing interventions. Default dropout remains `0.20`; regression testing checks the original default initialization and evaluation forward exactly.

## Training instrumentation

Extended validation is opt-in through `task.record_extended_validation`. Full-graph embeddings from the existing eval pass are reused for train/validation CE and Train Accuracy; index batching is ordered and does not construct a random DataLoader. Test evaluation must be false. Validation CE is unweighted Cross-Entropy. The Accuracy early-stopping comparator, min-delta, patience horizon, optimizer sequence, and ordinary training loss remain the V1 logic.

Best-Accuracy and Best-Macro-F1 snapshots are cloned independently during the same Accuracy-stopped trajectory. Best-F1 is selected only from epochs observed before Accuracy-driven stopping. Checkpoints include both state dictionaries, selection epoch/rule, validation scorecard, full resolved training configuration, configuration hash, source SHA, and cache fingerprints.

## Protocol and optimizer audit

The 2×2 study changes only dropout and the existing AdamW weight decay. Existing `build_optimizer` resolves to a single parameter group for RM-CGE V1, containing every model and classifier parameter. The study records per-group parameter names/counts, LR/decay, initial and final parameter norms, actual optimizer updates, first-epoch gradient diagnostics, and CUDA peak memory for each run. No no-decay parameter exclusion is added.

Initial model state and classifier state hashes were compared for every Dataset × Seed × Treatment before optimization. Details are in `implementation_checks.json`.

## Completed matrix integrity

All 36 planned Dataset × Treatment × Seed runs completed: 12 per seed and 9 matched Dataset × Seed blocks. The initial seed-42 batch passed its integrity review. Two later seed-43/44 launcher attempts stopped before training because the worktree status parser dropped a status-column space and the runner conflated current HEAD with the freeze identity; both issues were fixed before the successful runs. These post-freeze changes affect only launch validation, its tests, and audit prose; exact source SHAs are recorded per run.

The final integrity audit reports finite histories and metrics, 72 verified checkpoint files and Validation reloads, no Test evaluation or Test-label indexing, identical data/cache fingerprints within each Dataset × Seed block, and matched initial model and classifier hashes across O0–O3. It retains four historical failure entries and records zero failures in the completed matrix.

## Regression and revised replay evidence

The isolated CPU regression suite passed 106 tests across 7 RM-CGE/PACE/V7 files. Full captured output is stored in `implementation_checks.json`.

The post-freeze launcher-guard regression run passed 108 tests with 1 skipped (15 dependency/deprecation warnings; 8.58 s). The original preflight result of 106 passed / 1 skipped remains unchanged. Exact command and result metadata are in `data/runner_guard_regression_test.json`; the two guarded runner fixes are commits `0876797` and `0896373`.

The original Movies/O0/seed-42 replay was recorded as failed under the former full-trajectory gate: Pilot Best Epoch 70 vs replay 70; epochs 100 vs 100; maximum Train CE trajectory difference 0.080665; Validation Accuracy difference 0.018896; Macro-F1 difference 0.029098. Replay checkpoint embedding/logit differences were 1.091299/1.963385.

This is now classified `historical_full_training_replay_not_exact` and is nonblocking. Original Pilot embedding/logit arrays were not archived; the checks reconstruct them from its checkpoint. Repeated CUDA sparse aggregation varied by 1.19e-06 in embeddings and 1.91e-06 in logits, but this is evidence of possible numerical variation, not proof that CUDA caused the long-run mismatch.

The controlled two-update Forward/Backward/AdamW check matched initial weights and both step losses; maximum parameter error was 5.05e-07. RNG-preservation checks for added metrics passed. This bounds a short numerical comparison and does not establish full-trajectory identity. No further full historical O0 replay was run.

Revised correctness criteria require matching data/splits/features/semantic-cache/protocol; V1 model math, O0 dropout-0.20 initialization, optimizer and update rules; reasonable short-run numerical parity; RNG-safe Train/Validation metrics; disabled Test metrics with no Test labels indexed; finite metrics; verified checkpoint save/reload; and matched Dataset × Seed model and classifier initialization across O0–O3. Exact historical best epoch, epoch count, scores, and per-epoch curve equality are not acceptance gates.

The Test split was disabled in all training configs and diagnostic checkpoint scorecards; per-class and aggregate analyses index only Validation labels, while Macro-F1's historical class set is built from Train and Validation labels. No architecture variants, Test evaluation, extra seeds, or hyperparameter searches are included.

Parent Pilot Report SHA: `7e4b3dedb9de419ecec4193a1c2e50efd80dc3ab`. Source SHA and worktree fingerprints for the pre-freeze replay and post-freeze matrix are stored in the manifest and run CSV.
