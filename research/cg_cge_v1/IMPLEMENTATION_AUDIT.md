# CG-CGE V1 Implementation and Evidence Audit

## Final status and provenance

- Branch: `exp/cg_cge_v1_complementary_experts`.
- Parent at branch creation: `ffdb8c2b53ea4c67793d26c420832aa9a118cb58` (`exp/mag_view_collaborative_experts_v1`).
- Implementation Freeze Commit: `aac21ef7d4f8d4e25b2dd61cf1ed5fd6b8a67556`.
- Formal training source: `c3d32ae3b32e331cb4390b05823dbd36a50d8162`. All 18 accepted run records name this source and the same freeze commit.
- The frozen matrix is complete: Movies, Grocery and ele-fashion × B1/B2 × seeds 42/43/44. The report and machine-readable tables are in this directory and `data/`.
- The Test split was disabled; Test labels were masked at trainer entry. No Test metrics were computed.

## Implementation and protocol review

The model implementation is isolated in `src/models/cg_cge_v1.py`; the NC trainer integration is opt-in. The implementation review and focused tests cover the following:

1. Each modality's unimodal representation is computed from its projector, physical propagation and same-source KNN before cross-modal completion. The old physical-cross input in the inherited trunk is zeroed for this stage; physical cross propagation is introduced only by the new completion branch.
2. Semantic attention is receiver-conditioned and edge-only. Text completion uses the visual semantic support and visual completion uses the text semantic support. The custom sparse backward accumulates receiver-softmax and Q/K/V contributions in bounded chunks. Global IDs are validated by the cache loader and sampled/local IDs are rejected by the model.
3. B1 and B2 share the same expert bank and matched initialization. The source selector masks unsupported candidates; the empty-support branch becomes No-completion. Candidate capping is one-way and measured on valid support. Acceptance is applied once to the final residual.
4. B1 bypasses and freezes the zero-initialized Role-FiLM parameters. B2 applies the identity-initialized affine residual to Semantic and Complementary calls. Across all nine B2 runs, Role-FiLM parameters have finite, nonzero first-step gradients and nonzero checkpoint updates.
5. Auxiliary heads are model and optimizer/checkpoint parameters. The trainer computes their cross entropy on `train_idx` only; validation and checkpoint selection use the fused classifier.

The CPU preflight (`data/preflight_cpu.json`) confirms dataset, official split, feature and semantic-cache fingerprints. All nine Dataset × Seed blocks have matching initial model and external-classifier hashes between B1/B2. The corresponding A2 baseline fingerprints match within those blocks. `data/gradient_audit.json` records 18 first-step audits; all required cross projections, Q/K, source selector, channel gate, complementary router, cross scale and physical view-router gradients were present, finite and nonzero. B2 Role-FiLM gradients also passed for all nine cells. B1's FiLM parameters are intentionally unused.

## Verification and execution

- Focused CG-CGE unit tests passed (8 tests after the final telemetry fix).
- Full repository suite after the final implementation fixes: **550 passed, 2 skipped, 22 warnings**. CUDA was hidden for this CPU test run; warnings were upstream PyG/checkpoint deprecations.
- The real-data GPU preflight passed for all three datasets and both conditions on physical GPU0. It checked finite losses/gradients, nonzero physical-router gradient, validation RNG preservation and memory use. The maximum preflight peak was 16.70 GiB allocated / 18.29 GiB reserved.
- All 18 formal run rows are complete. Histories and metrics are finite; all 36 Best-Accuracy/Best-F1 checkpoints passed SHA256 and byte-count verification. The maximum accepted-run peak was 16.72 GiB allocated / 18.31 GiB reserved on a 24 GiB RTX 3090. No accepted run recorded OOM.
- B1/B2 initialization, split, feature, graph-cache, candidate and V7B.1 fingerprints were checked per Dataset × Seed. The strict A2 baseline pairing passed on all three datasets.
- Run metadata records `test_evaluation=false` for all 18 cells and has no Test metric fields. The runner's NC path receives masked Test labels; all reported selection and analysis metrics are Validation-only.

The normal workspace command sandbox did not expose `/dev/nvidia*` or `/dev/dri`. Therefore its first GPU probe failed before data loading; this was an access boundary, not evidence of an occupied GPU. After the user reported GPU0 idle, a read-only host probe found GPU0 at 99/24,576 MiB and 0% utilization, with PyTorch CUDA available. Training used physical GPU0 through the approved host execution path with `CUDA_VISIBLE_DEVICES=0`; GPU1 was not used. The manifest records Python 3.12.13, PyTorch 2.4.0+cu121, CUDA build 12.1, cuDNN 90100, NumPy 2.4.3, NVIDIA driver 535.309.01, and RTX 3090. `matmul_tf32=false`, `cudnn_tf32=true`, deterministic algorithms disabled and `CUBLAS_WORKSPACE_CONFIG` unset.

## Preserved failures and fixes

Nothing in the failure history was deleted or rewritten. `data/preflight_attempts.json` retains two preflight outcomes, and `data/failures.json` retains three runner failures; local attempt folders retain the related logs and checkpoints.

1. The October 14 preflight was refused because GPUs were actively used by other processes; a transient NVML/driver communication error was also recorded. No process was stopped and no driver restart was attempted.
2. The first ordinary-sandbox GPU attempt on October 17 had no device nodes and failed before data loading. The later host-side probe and full preflight passed as described above.
3. The first two Movies/B1/seed42 launches failed before training because the runner reset peak-memory statistics before a lazy CUDA context existed. A minimal probe reproduced the failure. The runner now calls `torch.cuda.init()` before resetting stats, inside its error-recording path.
4. A third Movies/B1/seed42 attempt trained 144 epochs and saved both checkpoints, but the runner rejected it because the model callback wrote `_last_gradient_diagnostics` while trainer telemetry read `_gradient_diagnostics`. The model now mirrors the diagnostics into the consumed attribute; the physical view-router parameter is audited explicitly, and the GPU preflight checks its nonzero gradient. This attempt remains marked failed; the cell was repeated as attempt 04.

The failed attempt 03 and accepted attempt 04 used the same dataset/split/cache and identical initial model and classifier hashes, but their Validation trajectories differed: attempt 03 best Accuracy was 56.5087% with Macro-F1 49.6891%; attempt 04 was 56.5387% and 49.4394%. This proves that identical saved initial state and data fingerprints did not produce an identical full trajectory in this pair. The source is unresolved. Dropout RNG execution and sparse/index CUDA operations are candidates, but the evidence does not isolate either and does not justify assigning the discrepancy solely to CUDA.

## Evidence limits

The three-seed comparisons are descriptive, not population-level significance tests. The measurements do not isolate dropout or weight-decay effects because both conditions use dropout 0.2 and weight decay 1e-4. Router activity, attention entropy, expert selection and nonzero cross-path scale show that mechanisms execute; they do not establish causal task utility. Validation checkpoint choice also changes the Accuracy/F1 tradeoff. Exact full-training trajectory replay was not demonstrated. No Test evaluation, additional seed, hyperparameter search, or successor model was performed.
