# CG-CGE V1 Implementation Audit

## Provenance and current execution status

- New branch: `exp/cg_cge_v1_complementary_experts`.
- Frozen parent at branch creation: `ffdb8c2b53ea4c67793d26c420832aa9a118cb58` (`exp/mag_view_collaborative_experts_v1`).
- Implementation Freeze Commit: `aac21ef7d4f8d4e25b2dd61cf1ed5fd6b8a67556`. It freezes code, tests and CPU pairing evidence; GPU memory/compute acceptance remains an explicit outstanding preflight gate.
- The parent tree was clean before work began. Existing View-CGE reports, run CSVs, failed attempts and checkpoints have not been edited.
- The V1 code is isolated in `src/models/cg_cge_v1.py`; the NC change is opt-in and defaults to the original `main CE + aux_weight * aux_loss` arithmetic.
- Dataset/cache/init CPU preflight passes across three datasets and all three seeds. All nine B0 cells match on split, feature, physical graph, semantic cache SHA, candidate and V7B.1 fingerprints, and protocol. B1/B2 model and external-head initialization hashes match in all nine cells.
- Focused CG-CGE tests: 10 passed. They include custom sparse attention reference forward/backward parity, Q/K/V finite nonzero gradients, source direction and support masks, empty/single-source behavior, Role-FiLM identity/update, matched initialization, model checkpoint reload, and NC auxiliary-head optimizer/checkpoint integration.
- Full repository pytest: **550 passed, 2 skipped, 22 warnings in 56.77 s** with CUDA hidden. Warnings are upstream PyG/checkpoint deprecations.
- As of the 2026-10-14 checks, full real-data GPU checks and formal training had not begun. At 14:47 UTC, the environment snapshot saw RTX 3090 GPU0 at 23,159/24,576 MiB and 91% utilization, and GPU1 at 13,787/24,576 MiB and 98% utilization. An earlier process list showed VLLM on GPU0 and Python PID 98051 on GPU1. The safe GPU-preflight attempt at 14:49 UTC found CUDA unavailable / failed isolation (`Can't initialize NVML`); a subsequent `nvidia-smi` call then could not communicate with the NVIDIA driver. At 14:57 UTC the driver responded again, but GPU0 remained at 22,858/24,576 MiB and 96% utilization (VLLM PID 3138154), and GPU1 at 13,787/24,576 MiB and 92% utilization (Python PID 98051). The follow-up resource decision was to refuse GPU use. No process was interrupted and no driver restart was attempted. The transient driver/NVML failure is not attributed to a single cause without evidence.
- On 2026-10-17 02:03 UTC (10:03 China Standard Time), the user reported GPU0 idle. The ordinary workspace command sandbox had no `/dev/nvidia*` or `/dev/dri` nodes, although `/proc/driver/nvidia` exposed the loaded 535.309.01 kernel module and both RTX 3090 device records. Accordingly, its CUDA probe and first physical-GPU0 preflight failed before data loading; that failure is preserved in `data/preflight_attempts.json` and is evidence of lack of device access from that command context, not evidence that GPU0 was occupied. A read-only `require_escalated` host probe then showed GPU0 at 99/24,576 MiB, 0% utilization, with PyTorch CUDA available and exactly one visible device; GPU1 was left untouched. The full real-data GPU preflight completed successfully on physical GPU0 at 02:04:58 UTC for all three datasets and both conditions. All required first-step gradients were finite, validation evaluation preserved RNG state, and there was no OOM; the maximum peak was 16.70 GiB allocated / 18.29 GiB reserved on ele-fashion B2, with per-step time 0.15–0.71 s. The full fingerprints, losses, gradients and memory records are in `data/preflight.json`. No formal training has begun yet.
- CPU snapshot: Python 3.12.13, PyTorch 2.4.0+cu121, CUDA build 12.1, cuDNN 90100, NumPy 2.4.3, driver 535.309.01, RTX 3090 devices. `matmul_tf32=false`, `cudnn_tf32=true`, deterministic algorithms disabled, no `CUBLAS_WORKSPACE_CONFIG`. CUDA device count was zero because this preflight deliberately hid CUDA; the physical snapshot still reported both GPUs.

## Code review facts

1. `Model` subclasses the V2 V1 trunk for its established projectors, directional cross projections, physical router, normalized graph propagation, four experts and fusion. Old V2 edge scorers/coefficient trees and V7A semantic scorer modules are removed because the new forward does not call them. The old Physical Cross candidate is never calculated in the independent modality stage; zero tensors are passed to the retained node-router correction interface.
2. Same-source supports are set by V7B.1 source provenance. Text completion reads `semantic_visual_edge_index_v2`; Visual completion reads `semantic_text_edge_index_v2`.
3. `receiver_sparse_attention` computes only the listed edges. Its custom backward analytically propagates through receiver softmax and accumulates Q, K and V gradients by bounded chunks. The cache loader remains responsible for global ID validation; sampled/local IDs are rejected by `forward`.
4. B1 and B2 instantiate the same FiLM tensors at zero. B1 disables gradients and bypasses FiLM arithmetic; B2 applies the identity-initialized affine residual only for Semantic and Complementary expert calls. All three functions call the same `self.experts` objects.
5. The source selector masks unsupported candidates; the both-empty branch becomes exactly No-completion. The conditional mixture divides by acceptance once; acceptance multiplies the final residual once. Candidate cap is one-way and measured on valid support.
6. Auxiliary heads are model parameters and are included in optimizer/checkpoint state. The NC trainer computes their CE only from `train_idx` and only when the opt-in weight is positive and the model advertises the interface. Validation and checkpoint choice use the fused head.

## Current verification record

`data/preflight_cpu.json` records cache, split, feature, B0 and initial-state checks. The focused test command is:

```text
conda run -n yhf_env python -m pytest tests/test_cg_cge_v1.py tests/test_cg_cge_v1_analysis.py -q
```

The Implementation Freeze Commit contains the passing software tests and CPU pairing evidence. GPU memory/compute acceptance is explicitly pending a later non-shared GPU preflight; the current driver/NVML failure prevents it. No formal run started, so a formal-run source SHA and final evidence commit do not yet exist. The runner resolves the source SHA from `HEAD` at launch and locates the freeze commit from its message.

## Evidence limits / risks under review

- Unit tests establish operator and autograd behavior on small CPU graphs; they do not establish full-size GPU memory, kernel support or throughput.
- CUDA sparse/index accumulation may use atomics and may be nondeterministic. Formal paired initial states and protocol are exact; bitwise identity of separate GPU training trajectories is not promised.
- Three-seed results are descriptive. Router or attention activity does not establish causal utility, and checkpoint sensitivity is not a retrained ablation.
- The current study has not produced Validation metrics. No claim about improvement is supported yet.
