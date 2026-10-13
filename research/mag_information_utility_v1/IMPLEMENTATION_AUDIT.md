# Implementation and execution audit

This document records implementation-level checks before the 45-run study.
The Implementation Freeze is preserved as a separate commit. Runtime guard
corrections and completed batch checks are recorded below. The final analyzer
validated and summarized all 45 runs without evaluating Test labels.

## Frozen implementation

- Parent baseline: `4cd655a411a6c0ab12e067285cd2202208b0ab18`.
- New branch: `exp/mag_information_utility_v1`.
- Model: `src/models/mag_information_probe_v1.py`.
- Five treatments differ only by a fixed six-slot mask; all use the same
  module tree and parameter shapes.
- Sparse physical first/second order and separate row-normalized Text/Visual
  semantic operators are used. No dense adjacency powers are formed.
- Context processing is bias-free before the residual LayerNorm; all-zero
  slots therefore yield a zero context update.
- Training and Validation share one evaluation-mode forward per epoch. The
  runner checks Python, NumPy, Torch CPU, and selected-device CUDA RNG equality
  around this diagnostic pass.
- Test evaluation is hard-disabled in the study YAML and runner.
- Formal loading revalidates random Text/Visual cache metadata fingerprints,
  source and randomized degree hashes, edge hashes, edge counts, and simple
  canonical undirected edge constraints.
- The analyzer refuses a partial 45-run matrix and validates checkpoint hashes,
  reload markers, Validation prediction hashes/metrics, epoch histories,
  paired initializations, and fixed random-topology hashes before writing
  tracked aggregates.

## Checks still to record

| Audit | Status |
|---|---|
| Full repository test suite | pass after replay-bound and source-audit changes: 452 passed, 16 upstream warnings |
| Model-mask, shape, initialization, provenance, and replay-tolerance tests | pass: 11 study-specific tests |
| Analysis pairing, report, and history-validation tests | pass: 3 study-specific tests |
| V7B cache feature and physical fingerprints | pass: all three datasets |
| Separate Text/Visual degree-preserving null graph audit | pass: all three datasets; degree/edge constraints exact |
| Training-path random-cache provenance revalidation | pass: all three datasets |
| Pre-label node-group construction | pass: all three datasets; group arrays hash recorded |
| 15-run Seed 42 gate | pass: 15/15 complete; all finite, no Test evaluation, checkpoints reloaded, paired initializations match |
| 15-run Seed 43 gate | pass: 15/15 complete with the same integrity checks |
| Seed 44 retry | pass: reran all 15 runs at source `946f224`; all finite, no Test evaluation, checkpoints reloaded, paired initializations match |
| 45-run checkpoint and finite-metric audit | pass: 45/45 complete; prediction and checkpoint hashes, reload metrics, epoch histories, paired starts, Test boundary, and frozen semantic-topology hashes validate |
| Source/model freeze audit | pass: runs use `9435ee0` and `946f224`, both descend from Freeze; changes since Freeze are limited to runner/replay/audit files and the model/config/protocol SHA-256 is identical (`2d92672127fabfe6040381925a89b709c25f309471da0aab487eaf1ce52f5945`) |

## Runtime observed during the pre-freeze GPU gate

- Python 3.12.13; NumPy 2.4.3; PyTorch 2.4.0+cu121; PyTorch CUDA build 12.1;
  cuDNN 90100.
- NVIDIA driver 535.309.01; two NVIDIA GeForce RTX 3090 cards.
- PyTorch sees both devices. GPU 0 is busy (about 95% utilization, ~1.4 GiB
  free at the check); GPU 1 is idle (0% utilization, ~23.7 GiB free). The
  runner's selector was checked and selected `cuda:1`.
- TF32: CUDA matmul disabled, cuDNN TF32 enabled, float32 matmul precision
  `highest`. Deterministic algorithms disabled; cuDNN deterministic and
  benchmark both disabled. No determinism or workspace environment override
  was applied.
- Dropout is `torch.nn.Dropout(p=0.20)` in training and disabled in evaluation.
  Activation checkpointing is off. Every post-update Train/Validation pass
  checks Python, NumPy, Torch CPU, and selected CUDA RNG state.

An earlier process check briefly saw no CUDA device nodes, but the subsequent
pre-batch environment check passed and the selector mapped the idle GPU 1 to
PyTorch `cuda:1`. The formal batch can proceed under the unchanged environment;
per-run hardware and determinism details are written to the run provenance.

## Pre-epoch runner defect and correction

The first Seed 42 invocation reached `cuda:1` and prepared Movies, then stopped
before epoch 1 because `_run_provenance` looked for `candidate_cache` while the
graph bundle exposes its V7B cache as `cache`. No optimizer step, checkpoint,
epoch history, or validation prediction was produced. The runner's failure row
is preserved in ignored `outputs/mag_information_utility_v1/failures.json` and
has been copied into the tracked failure record. The field lookup was
corrected in a separate follow-up commit and covered by a regression test.
The final analyzer requires each run source to descend from the Implementation
Freeze, limits later source changes to the documented runner/replay and audit
files, and verifies that every model/configuration/protocol core hash matches
the Freeze. Seeds 42/43 retain their source `9435ee0`; all seed44 runs use
`946f224`, which only adjusts the per-node CE diagnostic bound based on the
recorded Movies and ele-fashion replay envelope and updates source validation.
The full model, configurations, data/cache fingerprints, and experiment
protocol hash remain frozen across both run-source commits.

The final matrix was executed on the idle `cuda:1` NVIDIA GeForce RTX 3090.
Each 15-run seed gate reported finite metrics, Test evaluation disabled,
successful checkpoint reloads, matching treatment initialization hashes, and
no errors. The analyzer generated the paired endpoint, epoch, checkpoint,
node-group, conditional-error, R0-reference, and manifest evidence. It
validated 45 runs and confirmed the validation-prediction arrays reproduce
the stored metrics. No Test labels were read or evaluated.

Three early failed attempts remain in `data/failures.json`: the pre-epoch
cache provenance key defect, a Movies/I3 checkpoint replay rejection, and an
ele-fashion/I3 seed44 per-node CE difference of `3.81e-6` above the prior
`3e-6` check. Their original failure records are unchanged. Failed checkpoints
and partial run artifacts retain their SHA manifests in ignored local
`outputs/mag_information_utility_v1/failed_attempts/` directories. The updated
`5e-6` per-node CE bound is based on observed repeat evaluations; prediction
identity and aggregate checkpoint metrics remain checked separately.

Final analysis artifacts and their SHA-256 values are recorded in
`data/manifest.json`. Full repository tests pass (452 passed; 16 upstream
warnings), including 11 model/runner and 3 analysis-specific study tests.

The corrected-source attempt later stopped after Movies/I3 because a 2e-6
maximum per-node CE replay check was narrower than observed forward-to-forward
floating-point variation. Five repeated evaluations of the same Movies
checkpoint had identical predictions and Accuracy/Macro-F1; scalar Validation
CE differed by at most `1.2e-7`, while maximum per-node CE difference was
`2.15e-6`. On the failed ele-fashion/I3 seed44 checkpoint, five repeated
evaluations again had zero prediction flips and identical Accuracy/Macro-F1;
scalar CE varied by at most `3.0e-8`, while maximum per-node CE difference was
`3.34e-6`. The original training-pass to reload maximum was `3.81e-6`.

A separate diagnostic process enabled deterministic algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; on Movies it still showed zero prediction
flips and identical scalar CE with maximum per-node CE difference `1.91e-6`.
This confirms small forward-pass numeric variation but does not identify one
kernel as its sole cause. The formal environment was not changed. The per-node
CE diagnostic bound is now `5e-6`, narrowly above the largest repeated and
cross-pass values observed on both datasets. Node IDs, labels, and predictions
must still match exactly, and aggregate checkpoint metrics retain the
existing `2e-6` check. The failed checkpoints and earlier seed42/44 outputs
were copied with SHA-256 manifests under ignored
`outputs/mag_information_utility_v1/failed_attempts/` before retrying affected
runs. Runtime source commits differ only by explicit provenance/replay guard
fixes; model and study protocol source/configuration remain frozen and are
hash-checked by the final analyzer.
