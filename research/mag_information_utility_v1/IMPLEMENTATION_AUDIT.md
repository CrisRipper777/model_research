# Implementation audit (pre-run freeze record)

This document records implementation-level checks before the 45-run study.
Run-specific GPU, checkpoint, and finite-metric checks are added after the
first 15-run batch and the final matrix. No training pilot has been completed.

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
| Full repository test suite | pass after the provenance fix: 451 passed, 16 upstream warnings |
| Model-mask, shape, initialization, and provenance tests | pass: 10 study-specific tests |
| Analysis pairing, report, and history-validation tests | pass: 3 study-specific tests |
| V7B cache feature and physical fingerprints | pass: all three datasets |
| Separate Text/Visual degree-preserving null graph audit | pass: all three datasets; degree/edge constraints exact |
| Training-path random-cache provenance revalidation | pass: all three datasets |
| Pre-label node-group construction | pass: all three datasets; group arrays hash recorded |
| 15-run Seed 42 gate | pending; first invocation aborted before epoch 1 due to a provenance key mismatch, and the failure is retained |
| 45-run checkpoint and finite-metric audit | pending |

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
will be copied into the final tracked failure record. The field lookup is
corrected in a separate follow-up commit and covered by a regression test; all
45 completed runs must share that corrected source commit.
