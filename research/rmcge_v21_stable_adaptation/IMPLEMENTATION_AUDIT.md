# Implementation Audit and Freeze Review

## Source and artifact verification

- Expected source branch/commit: `exp/rmcge_v21_message_purpose_selection` / `218d6d230e5afc16e259c09777f6363de5e0c6cf`.
- The working tree was clean before the new branch was created. The study branch is `exp/rmcge_v21_stable_adaptation`.
- Historical V2.1 model code, runs, checkpoints, and reports remain unchanged. Diagnosis outputs are additional files under this study directory.
- Phase A hashed all 54 historical V2.1 checkpoints (Best Accuracy and Best Macro-F1 for C0/C1/C2 × three datasets × three seeds). The source manifest now also records each old resolved-config hash and run-metrics hash.
- Stage B selects only the matching old C0 Best Accuracy checkpoint for each dataset × seed. The runner cross-checks manifest hash, resolved-config hash, run-metrics hash, checkpoint identity/selection/provenance, data dimensions, split and cache fingerprints, and finite model/head state. It rejects unexpected state-dictionary keys and rejects any optimizer/scheduler state in the source payload.
- Project cache verification passed for Movies, Grocery, and ele-fashion. Their cached feature, candidate, and V7B fingerprints match the old V2.1 manifest; caches contain no labels.

## Runtime and device

The authorized project environment is `yhf_env`: Python 3.12.13, NumPy 2.4.3, PyTorch 2.4.0+cu121, CUDA build 12.1, cuDNN 90100. The host driver is 535.309.01. Both devices are RTX 3090. At preflight, GPU 0 was occupied by an unrelated inference service and GPU 1 was idle; this study is pinned to physical GPU 1 via `CUDA_VISIBLE_DEVICES=1` (inside the process this is `cuda:0`). Matmul TF32 is disabled, cuDNN TF32 is enabled, and PyTorch deterministic algorithms remain disabled. The study does not change host-wide determinism or TF32 settings.

## Initialization correctness

Seed 42 has passed the real-data t0 triplet audit for all three datasets before any new training. Exact per-dataset fingerprints and logit deltas are in `data/initialization_audit.json`.

| Dataset | Public C0 state | Classifier | W1/W2 selector init | Max Validation-logit difference from W0 |
|---|---|---|---|---:|
| Movies | exact | exact | exact | 1.32e-6 |
| Grocery | exact | exact | exact | 1.67e-6 |
| ele-fashion | exact | exact | exact | 1.79e-6 |

For W1/W2, the added state-dictionary keys are exactly the declared selector additions; no unexpected missing or extra keys occur. All initialized intra and cross scores are zero, scorer output weights and bias are zero, and t0 logits pass `atol=rtol=5e-5`. The audited differences are below `1.8e-6`. T0 evaluation leaves CPU and CUDA RNG byte states unchanged. It uses eval mode and does not consume Test labels.

## Training implementation controls

- W0 uses the loaded C0 model/head. W1 and W2 load the same C0 public model/head state with only the expected selector keys missing.
- Stage 1 uses model/classifier eval mode with autograd enabled. Inherited public parameters and classifier are frozen; W1 receives selector parameters, and W2 additionally receives `cross_delta_coefficients`. `delta_coefficients`, `cross_coefficients`, and `knn_coefficients` remain inherited and frozen.
- Stage 1 checks public-model and classifier state hashes before and after all 12 epochs. It requires a nonzero scorer gradient at the first update and a nonzero scorer parameter update. W2 also checks an actual `cross_delta_coefficients` gradient and parameter update.
- Stage 2 unfreezes all parameters and constructs a fresh AdamW optimizer with explicit common/classifier/selector groups and an empty initial state. It does not restore historical AdamW momentum or scheduler state.
- Training uses the existing Full-Graph NC forward and `cross_entropy + model-returned balance loss` (`balance_weight=0.01`), clips gradients at 1.0, evaluates only Train/Validation, and records selector, physical/semantic expert, router, gradient, update, and resource diagnostics.
- Main checkpoint selection compares only t0 and Stage 2. Stage 1 remains exploratory. Best Accuracy drives stopping; Best F1 is independent. A no-improvement run retains and marks t0.
- Checkpoints are written through a temporary file, loaded back, state-hash checked, and re-evaluated on Validation before a run is marked verified.

## Tests before freeze

Before the first freeze, the new focused suite passed **25 tests** and the complete repository suite passed **509 tests with 1 skipped** (18 existing deprecation/future warnings). The first formal launch then exposed a finalization guard defect: after Movies/W0/seed 42 completed 49 joint updates and wrote both selected checkpoint files, `_zero_test_attestation` incorrectly treated the intentional boolean audit fields `test_evaluation` and `test_label_values_indexed` as Test metrics. The run was therefore not marked complete. The original failure record and partial outputs are preserved; no Test evaluation occurred.

The guard now permits only those two explicit false-valued attestations and rejects any unexpected `test_*` result field. A regression test covers both cases. The focused suite now passes **26 tests**, and the full repository suite passes **510 tests with 1 skipped** (18 existing deprecation/future warnings). The runner accepts a validated `--attempt-tag`, so a retry writes to a separate directory and preserves the failed attempt. The failure is a harness bookkeeping defect; it did not alter the frozen model, data, split, cache, optimizer, or training configuration.

After all 27 runs passed, the first analysis invocation found a second bookkeeping issue: the history reader constructed canonical run paths instead of using each run's registered `result_path`, so it could not read tagged retry directories. The invocation reloaded Validation checkpoints for Movies/seed 42 and then stopped before writing any analysis tables; Test remained disabled. The failure is preserved. The analyzer now resolves history through `runs.csv`, and a regression test exercises attempt-directory paths. This correction changes only analysis artifact lookup, not the frozen training implementation or any run/checkpoint.

The focused suite now passes **27 tests**, and the full repository suite passes **511 tests with 1 skipped** (18 existing deprecation/future warnings). No historical V2.1 regression was found. Tests run in CPU-only mode; real GPU/state/cache equivalence is separately evidenced by the seed-42 t0 audit above.

## Freeze boundary

The corrected implementation freeze includes the diagnosis and training/analyzer entry points, study configuration, utility functions, focused tests, initialization audit, failure-safe attempt handling, and this document. Seed 42 remains the first nine-run gate: its preserved first attempt is not counted as a verified run, and seeds 43/44 remain gated on a clean nine-run batch with per-run numerical, state-freeze, selector-update, checkpoint, and Test-disabled checks.
