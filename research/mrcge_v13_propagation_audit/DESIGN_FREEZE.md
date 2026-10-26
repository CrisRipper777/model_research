# MR-CGE V1.3 Cross-Trajectory & Physical-Basis Audit — Design Freeze

## Scope and provenance

This is a label-free, read-only analysis of frozen Best-Validation-Accuracy checkpoints at base `788c35392f049fea32f42f5081656924888f7082`: seven `mrcge_v13/joint_view` cells (Movies 42/43/44, Grocery 42/43, ele-fashion 42/43) and three `hybrid_cge_paper_v1/H2_physical_3hop` Movies cells (42/43/44). The checkpoint, source-commit, fixed-split, V7B.1 K=8 cache, feature and graph fingerprints are verified before each worker runs. The fixed split fingerprints are Movies `2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d`, Grocery `7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056`, and ele-fashion `04d43160594823ab35d62b1bde6547225dc816405e55eb1f9c44790eabfae0c5`.

No optimizer, backward pass, training, checkpoint write, Test evaluation, link prediction, or label-based computation is permitted. Data are loaded with the existing fixed split and `auto_generate_splits=false`; `mask_test_labels` is applied before model invocation. Test labels are checked as `-100`. Full-graph features and edges are used only under the established transductive protocol. Only physical GPU 1 may be used for full-cell work, via the existing ownership guard, in an isolated child process for each checkpoint and with at least 2 GiB free VRAM.

## MR-CGE V1.3 operator and basis

The implementation is the frozen `src/models/mrcge_v13.py`, chiefly `Model._physical_operator`, `Model._basis`, and `chunked_weighted_spmm`:

1. Self loops are removed with `torch_geometric.utils.remove_self_loops`.
2. For each directed `src -> dst` edge, degree is the unweighted incoming edge count at `dst`. The edge weight is `d(src)^(-1/2) d(dst)^(-1/2)`, with the implementation's `clamp_min(1)` degree handling. This is the actual frozen operator; it is not replaced by a random-walk or self-loop operator.
3. The support `S` is `degree > 0`, equivalently the nodes appearing as edge destinations after self-loop removal.
4. `B0 = layer_norm(H_m, (hidden_dim,))`. For `k=1..3`, `Bk = S[:,None] * P(B{k-1})`, with the same weighted sparse multiplication and `physical_chunk_size` as the model. There is no additional normalization of `B1..B3`.

The primary energy and similarity scope is the same support `S` for every order. RMS is `sqrt(mean_{n in S,d} T[n,d]^2)`. We additionally report all-node RMS to expose dilution by unsupported zero rows, support count, all-node count, and per-node zero/near-zero count. Empty support and all-zero operands are explicit invalid/degenerate statuses, not silently reported as ordinary zeros.

Global Frobenius cosine is `<A,B>/(||A||F ||B||F)`, accumulated in FP64 in row blocks. Node cosine is calculated only where both node vectors are above the declared near-zero threshold (`1e-12` in FP32 norm); valid-node count and omitted zero-vector fraction accompany P10/median/P90. The 4-by-4 normalized Gram uses the same common support, and is separately identified from pairwise cosine distributions.

For each checkpoint and modality, the trained `filter_bank.theta` is converted by the frozen `filter_bank.coefficients(3, dtype)` implementation (signed row-wise L1 normalization with the frozen epsilon and order mask). `G_private[k] = sum_l alpha[k,l] B_l` is computed from those exact coefficients. Expert-input RMS, pairwise global cosine, normalized Gram and rank diagnostics do not substitute initialization coefficients. Existing filter/router detail from an eval forward is reused for activation and route diagnostics.

## V1.3 Early Cross and diagnostic trajectories

For receiver modality `m`, sender is `other(m)`. The actual source projection is `cross_projections[visual_to_text or text_to_visual]`. The raw path is `C_raw = P Phi(H_other)`, masked by the frozen physical support. `_early_cross` calibrates it to receiver `B1` on that same support:

`X1 = C_raw * stopgrad(RMS_S(B1)/(RMS_S(C_raw)+eps))`, except that raw RMS at or below `eps` produces an exact zero tensor and zero scale. This reproduces `Model._early_cross` exactly; each reconstructed tensor is checked against `model._last_state_audit["early_cross"]` from a fresh `eval()`/`no_grad()` forward with `_capture_state_audit=True`. Any failed tolerance check invalidates that checkpoint's trajectories.

The diagnostic-only continuation is `X2=S*P(X1)`, `X3=S*P(X2)`. It adds no calibration, parameter, injection, or model-forward path. `beta_m` is recorded independently; the model's injected term is `beta_m X1`, not X1. Cross energy, adjacent RMS ratios, pairwise cosines, node distributions, and the 3-by-4 cross/basis similarities use the same support and explicit zero/empty statuses.

For same-node context, `Phi(H_other)` is measured directly. The true frozen V1.3 Late SameNode candidate is separately reconstructed as `Phi(U_private_other)` using the directional CrossProjection and the audited `private` tensor in `_last_state_audit`. They are never conflated. The V1.3 late semantic sender value `Phi(0.7 H_other + 0.3 U_private_other)` is not the SameNode candidate. We record RMS/cosine for X2 and X3 against each candidate; a cosine is an alignment clue only, not a fraction of backtracking mass.

## Linear novelty and numerical policy

The flattened basis columns are interpreted as vectors over the same support-node × feature coordinates. `S_base=[B0,B1,B2,B3,X1]`. For target `X`, novelty is `||X - Proj(span(V),X)||F/(||X||F+eps)`. We also compute X3 conditional novelty with `V=[S_base,X2]`.

No large projector or `[N,N]` matrix is formed. In FP64, row-block accumulation computes the small column Gram `G=VᵀV`, target cross-products `q=VᵀX`, and target norm. Each V column is first scaled to unit Frobenius norm; zero columns are separately marked. Symmetric eigendecomposition of the symmetrized small Gram identifies numerical rank at the singular-value threshold `tol=sqrt(max(m,k)*eps64)*sigma_max` (the corresponding Gram eigenvalue threshold is `max(m,k)*eps64*lambda_max`); negative eigenvalues within roundoff are clamped to zero and materially negative values invalidate the calculation. The retained eigenspace gives the minimum-norm pseudoinverse coefficients. A second block pass computes the residual directly, avoiding cancellation in `||X||²-qᵀG⁺q`. Report column scales, singular spectrum, effective rank (entropy rank of normalized singular values), retained-spectrum condition number, tolerance, residual reconstruction error, pseudoinverse/deficient-rank status, and finite/stability status. A zero target has novelty 0 with status `zero_target_undefined_ratio`; it is not treated as evidence of no novel signal.

The numerical contract is: FP64; novelty/reference agreement within `1e-9` on CPU synthetic cases; direct reconstructed residual finite; block-size variation within `1e-10` relative or `1e-12` absolute on CPU; production novelty reconstruction must be finite with relative reconstruction error within `1e-6`; independent X1 reconstruction must agree with the model capture within `1e-6` relative Frobenius error and `1e-4` absolute maximum error. No threshold is selected using labels or validation outcomes.

## Hybrid-CGE H2 Physical 3-hop (independent coordinate system)

The Hybrid comparison uses only `hybrid_cge_paper_v1/H2_physical_3hop`; the wrapper sets `physical_hops=3` and retains the first three columns of its frozen Hadamard `alpha_raw` initialization, with the inherited row L2 normalization. Its physical operator and projectors come from `hybrid_cge_v11` / `cg_cge_v1`, not V1.3. `_physical_states` applies the Hybrid `_propagate` three times with the Hybrid active mask and stores raw orders 1–3. Its `_active_rms_norm` then normalizes each state per channel using active nodes; alpha mixes these normalized states into expert profiles. The report keeps raw Physical States, normalized states, and expert profiles distinct.

Hybrid early physical Cross is its own `Phi(prior_other) -> chunked_weighted_spmm -> active mask`, then divides by the receiving modality's first-hop per-channel scale from `_active_channel_scale`; it is not V1.3's scalar RMS calibration. H2 late SameNode is `Phi(U_private_other)`. Hybrid internal expert inputs, route shares/entropy, response RMS and cross/profile ratios are reported only within Hybrid. Cross-model comparisons are limited to dimensionless, semantically aligned quantities (within-model order ratios, Gram/rank summaries, route entropy/share and relative input/response scales). No cross-model coordinate cosine is computed.

## Source, data, runtime and interpretation safeguards

- Model construction and data loading use the historical runner logic solely for config, cache and fixed-split resolution. The label-masked object is the only label-bearing object passed to the forward call; forward receives features and physical edges only.
- Before full runs, source and checkpoint bytes, model source-at-recorded-commit hashes, cache SHA/fingerprints, split path/SHA/fingerprint, Test-mask status, and protected historical file hashes are audited. Each worker repeats the GPU ownership guard and exits on unknown GPU users or less than 2 GiB headroom.
- Full `[N,256]` tensors may exist transiently for a checkpoint, but are not written. Similarity/Gram reductions are blockwise. One isolated process handles one checkpoint and is terminated before the next.
- This audit describes propagation scales, directions, linear-span redundancy, and trained router/expert usage. Cosine, rank, novelty, and energy are distinct: none alone establishes classification information, generalization, causal module benefit, or accuracy. High novelty is not class utility; low novelty does not rule out nonlinear expert benefit. No model winner, SOTA, Test, or LP claim is allowed.

## Frozen implementation audit checklist

The independent analyzer and its tests will be committed before any checkpoint worker starts. The resulting commit SHA, analysis file SHA256 values and parameters will be embedded in every output record. Stage B and all new model development remain out of scope.
