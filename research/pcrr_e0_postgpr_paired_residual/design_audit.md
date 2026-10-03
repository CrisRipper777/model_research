# PCRR-E0 design audit

## Provenance

- Required parent branch: `exp/orci_d0_interaction_alignment_synergy`
- Required parent SHA: `3ef56df39a2d554aaac9a97ae9f6d4943d345f13`
- PCRR-E0 branch: `exp/pcrr_e0_postgpr_paired_residual`
- Implementation is being made directly on the required parent; no other experiment branch is merged or cherry-picked.
- Dataset splits are fixed by the existing repository paths. NC evaluation is validation-only (`development_no_test=true`, `evaluate_test=false`).

## A. Retained C1/D0-B backbone

For modality `m ∈ {text, visual}`, the independent C1 projector produces

`P^m = projector_m(X^m)`, `H_0^m=P^m`, and `H_k^m=Â H_{k-1}^m` for `k=1,2,3`,

where `Â` is the existing normalized physical graph operator (including the configured self-loop policy). Shared raw GPR coefficients are `c = c_prior + delta_c_raw`, and

`G^m = LayerNorm(sum_{k=0..3} c_k H_k^m)`.

The existing C1/D0-B late fusion is applied after the two modality representations. PCRR keeps the projectors, coefficient parameterization, graph operator, GPR composition, modality LayerNorm, and late-fusion modules and their initialization order unchanged.

## B–C. Why move interaction after GPR

D0 found that injecting order-level cross-attention into `H1:H3` harmed the strong RGD baseline, while alignment partly rescued that interaction without making the joint model exceed B. The trained joint checkpoint was sensitive to source-node shuffling, which is evidence that it used correct modality correspondence. PCRR therefore retains the RGD representation first and tests one paired residual after GPR composition; it does not interpret correspondence sensitivity alone as an architecture gain.

## D. Excluded mechanisms

The PCRR residual has no cross-attention, alignment loss, optimal transport, mixture of experts, node/edge router, or auxiliary loss. `aux_loss` is an exactly zero scalar.

## E. Zero-function initialization

For each direction, the feature is `[target, source, abs(target-source), target*source]` (1024 values at hidden width 256). Both directions share `LayerNorm(1024) → Linear(1024,64) → GELU → Linear(64,256)`. The final linear weight and bias are initialized to exactly zero. There is no dropout in the pair block and no normalization after the residual. Thus `G_tilde^m=G^m` at initialization, and late fusion receives the same ordered per-node modality pair as D0-B.

## Variant contract

- B constructs the same residual parameters but skips the pair response.
- P pairs text and visual representations at the same node.
- S applies a deterministic, run-fixed derangement only to the pair response source. Text and visual backbones, propagation, GPR, and the final fusion pairing remain aligned by node.
- Runtime `residual_off` and `source_node_shuffle` affect only P's residual branch. Shuffle repeats are supplied explicitly by the analyzer.
- B/P/S have identical parameter/state layouts and same-seed initialization. S's permutation is generated with a private CPU generator seeded by `model_seed + 73000` and its fixed-point count/rate are recorded.

## Large-graph execution

The implementation retains D0's chunked projector, chunked normalized propagation, and streaming raw-GPR path. Pair features are materialized only for node chunks (8192 nodes by default); optional checkpointing recomputes chunk response activations on large training graphs. `src/tasks/nc.py` and `src/tasks/inference.py` are not changed unless later correctness checks demonstrate a need.

## Frozen experiment scope

The campaign is limited to Movies, Grocery, and ele-fashion, variants B/P/S, and training seeds 42/43/44 on the existing fixed NC splits. The only auxiliary smoke is B/P on sports-copurchase LP (2 epochs, at most 2 train batches, fanout `[5,5,5]`, test disabled). There is no NC test, formal LP, split modification, rank sweep, or architecture change based on interim results. The primary comparisons are paired validation P−B and P−S; S−B estimates the capacity-matched control. No pseudo-IID p-values or arbitrary minimum-gain threshold will be used.

## Runtime validation status

Pending implementation tests, smoke runs, and the 27-run validation campaign. Final evidence and deviations will be appended to `report.md`.
