# Implementation audit

## Source and protocol

Source is RM-CGE V2.1 stable-adaptation commit `151f1c08a872d55cdf3e7de37d950c7485fb4014`, on branch `exp/rmcge_v22_cross_message_content`. Old V7B.1, V2, V2.1 and stable-adaptation files remain untouched. Dataset and V7B.1 cache verification ran in `yhf_env`; all three caches match their registered feature/candidate/V7B fingerprints and contain no labels. The train driver loads the existing seed-specific official NC split and verifies feature/topology identity across seeds.

## Runtime and GPU

Authorized env: Python 3.12.13, NumPy 2.4.3, PyTorch 2.4.0+cu121, CUDA build 12.1, cuDNN 90100, NVIDIA driver 535.309.01. Both physical devices are RTX 3090 (24,576 MiB): GPU 0 UUID `GPU-28bf6c52-0671-4e90-e88b-4e47457b7cce` was occupied by an unrelated process (22,858 MiB, 88% utilization at snapshot); it is excluded. GPU 1 UUID `GPU-84aac143-953d-a98f-a8ff-8ebf88d076b6` is selected with `CUDA_VISIBLE_DEVICES=1` (logical `cuda:0`). It was idle before the CUDA smoke test and showed 351 MiB / 0% afterward, including CUDA context. The runner checks GPU 1 remains below 512 MiB before each new run and refuses to share it if that changes. Preflight observed matmul TF32 disabled, cuDNN TF32 enabled, deterministic algorithms disabled, and no CUBLAS workspace override. The implementation does not change those settings. GPU scatter/index-add may retain backend-level floating-order variability; this is recorded as a reproducibility boundary, not attributed to CUDA without evidence.

## Mathematics and reuse

The class subclasses V2.1 and uses the inherited V2/V1 public modules, validated semantic-cache loader, Cross Projection, KNN, physical profiles, expert bank, router, strength, fusion and balance loss. M0 calls the inherited C1 implementation with Intra explicitly masked; zero-effect and V2.1 C0 equivalence are tested. M1–M4 add only the specified expert-input residual. M4 uses the original C1 selector; no C2 Cross selector is active. Balance loss is multiplied by `.01` in the model and added once by NC with auxiliary weight 1.

Pair mode computes only chunk-sized low-rank gates and a low-rank scatter. A custom backward recomputes per-edge chunks; finite-difference-to-reference unit tests compare its gradients against a dense small-graph formulation. M2 is checked against aggregate-then-gate; M1's sender transform is checked against direct edge aggregation. Zero-degree output is masked after a bias-free projection.

## Initialization and synthetic evidence

The V2.1 C1 public tree and the V2.2 content module are created under RNG-isolated initialization scopes. Every V2.2 arm declares the same content shapes and starts from identical state per seed; the NC classifier is initialized after model creation and is checked through recorded state hashes. Real-data initialization checks cover all 3 datasets × 3 seeds: common tensors and classifier weights are exact, M0/C0 eval logits differ by at most `2.0266e-6` (the accepted comparison tolerance is `5e-5`), and both forwards leave CPU/CUDA RNG states unchanged. The zero final Gate layer makes initial content exactly zero; W_output remains nonzero, and eta starts at `.25`. Synthetic tests distinguish sender-only, receiver-only and pair-conditioned behavior and construct a pair output outside the scalar-rescaling direction.

Pre-freeze verification: full repository suite **524 passed, 2 skipped**; focused CUDA V2.2 suite **10 passed** on physical GPU 1; Python compile and `git diff --check` pass. Checkpoint reload auditing verifies Best Accuracy and Best F1 validation Accuracy, Macro-F1, and CE against saved values at `1e-6`. Reports retain stochastic pre-update Train CE separately from post-update eval-mode Train CE and Validation CE, and include their comparable Train−Validation CE gap. Formal runs also check that the selected GPU remains available before each run.

## First-run guard correction

The initial Freeze (`49c20e5b9c7adca2d07d9806a2cf24088bb65fff`) completed Movies/M0/seed 42, saved and reloaded both checkpoints, and recorded Validation Accuracy `0.557289`, Macro-F1 `0.465377`, and CE `1.401347`. Before Movies/M1 began, the original GPU guard saw 583 MiB total use (the current Python process's own CUDA context plus baseline) and stopped the batch. This false positive is preserved in `data/failures.json`; no M1 training began. The guard is corrected to identify the current PID via `nvidia-smi` compute-app telemetry, reject other compute PIDs, and bound unattributed device memory. The correction is outside the model/training path; the completed M0 run remains valid, while the corrected implementation is frozen in a follow-up commit before resuming. The post-correction full suite passes **525 passed, 2 skipped**, including a mocked PID-aware guard regression test; a live CUDA guard smoke is recorded before the follow-up freeze.

## Checkpoint metric-set correction

Fashion/M0/seed 42 training reached epoch 178 and saved both checkpoints, but checkpoint analysis correctly stopped when its Validation Macro-F1 (`0.681147`) did not match the saved metric (`0.743069`). The cause was an analyzer-only mismatch: global validation Macro-F1 used all `num_classes`, while the NC trainer freezes its metric labels to classes observed in Train+Validation when Test evaluation is disabled. The analyzer now calls the same `_resolve_nc_eval_labels(..., include_test=False)` and uses that label set for checkpoint and frozen-intervention metrics. This reads no Test labels. The failed attempt and checkpoints remain untouched in `outputs/rmcge_v22/runs/ele-fashion/M0/seed_42`; a direct GPU recheck of both retained checkpoints reproduces Accuracy, Macro-F1, and CE, and all three M0 frozen interventions complete with finite metrics. The formal condition will be rerun into a new attempt directory. After this correction the full repository suite passes **526 passed, 2 skipped**, including a regression test for the train/validation-only metric label set.

## Test-label boundary

Before NC training, the runner writes sentinel `-100` into a copied label tensor at Test indices. The shared trainer receives only that safe copy, Test evaluation is false, and Train/Validation audits reject any sentinel values. Cache generation and verification report no label access. No Test metric is written.

## Known limits

M1/M2/M3 differ in active input channels and are not strict capacity-matched causal ablations. M0–M3 carry an inactive C1 selector tensor layout so initialized state stays paired; its path is disabled and its gradient remains absent. The process does not enable deterministic algorithms because inherited graph scatter operations may not have deterministic kernels; this is not used as a reason to discard the study. Full training-trajectory bitwise equality is not a success criterion.

## Final 45-run audit and report-only follow-up

The frozen matrix completed with 45/45 unique Dataset × Condition × Seed keys (15 per seed), all status `complete`, and all Test-evaluation flags false. The offline analyzer validated 5,842 contiguous finite epoch rows; 90 checkpoint records against local byte counts and SHA-256; 45 checkpoint/RNG/initialization audits; 45 gradient audits; paired initial model/head hashes; manifest-matched feature, V7B.1 cache, and seed-specific split fingerprints; and finite, unique-key diagnostic schemas. It emitted 378 per-seed paired contrasts and 126 paired summaries. Test metric fields were absent, the manifest records Test labels were not indexed, and the run protocol had already masked Test labels before training.

After Freeze `5bfee4de9d90fb4047e93b67428310bf8898c9a3`, the evidence analyzer was made more explicit and strict: it now checks provenance and checkpoint file integrity, validates diagnostic coverage, and writes measured answers into the decision memo and cross-modal scale report. This is an analysis/report-only change; model, trainer, configuration, checkpoints, and training results were not modified. The analyzer was syntax-checked and its completed 45-run integration audit passed.

Immediately after the training process exited successfully, a device probe found `nvidia-smi` unable to communicate with NVML and no `/dev/nvidia*` device nodes, although the NVIDIA 535.309.01 kernel module remained loaded and the preflight recorded physical GPU 1. PyTorch then reported CUDA unavailable. This was observed only after all 45 runs/checkpoints had completed; the cause is unresolved and is not attributed to the experiments. No GPU reset, extra training, or Test evaluation was attempted. The final analysis and checkpoint-hash verification ran on CPU against the saved artifacts.
