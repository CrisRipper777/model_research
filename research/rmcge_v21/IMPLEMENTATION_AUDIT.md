# RM-CGE V2.1 Implementation Audit

## Freeze preflight

The implementation descends from the requested RM-CGE V2 evidence commit `ba1fbbd100a0b218ec66853e29ea0cfee8cfa5b4`. Preflight was run on `exp/rmcge_v21_message_purpose_selection`. The complete machine-readable evidence is in `data/implementation_checks.json`.

The final repository suite at freeze preflight reported **484 passed, 1 skipped**. The targeted V2.1 model and analysis suites reported **18 passed**.

## C0 delegation and paired initialization

C0 constructs the original RM-CGE V2 `V1` parameter tree and calls its Forward unchanged. The V2.1 model test compares C0 state and returned tensors against a direct V2/V1 reference. For all nine Dataset × Seed pairs, the initialization audit found exact equality for the common C0/C1/C2 model parameters, complete C1/C2 parameters, and classifier state. Initial model and classifier fingerprints are saved in the machine-readable audit and each formal run record.

The C1/C2 selector modules initialize under `torch.random.fork_rng`; their construction does not advance the random stream used to create the classifier. Dropout and the optimizer/protocol settings are inherited from the frozen V2 configuration.

## Graph and selector checks

Tests cover the original physical coefficients, per-receiver mass conservation, zero/equal scores, single-neighbor rows, isolates, separately scored reverse edges, duplicate directed edges, chunked score equivalence, finite values, Cross Projection sender-before-aggregation order, C0 equivalence, C1/C2 zero-effect behavior, Cross selection shutdown, and checkpoint round trip. A controlled two-target graph verifies that the receiver interaction can reverse neighbor ordering over an identical candidate set.

Gradient tests show task gradients and AdamW updates for the selector output, shared experts, projections, router and fusion. After an update, query/key projections and the hidden scorer layer receive nonzero gradients. C2's Cross Delta coefficient also receives a later-step task gradient. A common per-receiver scorer bias can have zero task gradient due to normalization shift invariance; this is expected.

## GPU resource audit

The preflight ele-fashion full forward/backward/optimizer smoke ran C1 and C2 on the available RTX 3090 physical GPU 1. Peak allocated memory was 11.62 GiB for C1 and 12.22 GiB for C2; both completed two finite optimization steps. GPU 0 was left untouched because an unrelated VLLM process occupied it. Runtime metadata records Python 3.12.13, NumPy 2.4.3, PyTorch 2.4.0+cu121, CUDA build 12.1, cuDNN 90100, driver 535.309.01, RTX 3090, TF32 matmul disabled, cuDNN TF32 enabled and deterministic algorithms disabled.

The model uses bounded edge-score chunks, differentiable FP32 receiver scatter and chunked SpMM. It does not form a dense adjacency or an edge-by-hidden message tensor. CUDA scatter accumulation may have backend-dependent floating-point order; the audit does not attribute all variation to CUDA.

## Validation-only boundary

The runner uses Train indices for supervision and Validation for checkpoint selection. Extended Train/Validation scorecards reuse deterministic cached embeddings and do not add a stochastic model forward. The runner sets `evaluate_test=false`; initialization fingerprints include Train and Validation labels only. Test metrics and Test-indexed label values must remain absent from checkpoints and reports.
