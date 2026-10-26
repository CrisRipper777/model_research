# MR-CGE V1.3 Two-Hop Cross Controlled Screen — Design Freeze

## Objective and frozen comparison

This is a validation screen of exactly two additions to the frozen V1.3-A `M_repaired` / `joint_view` path:

- `raw`: the calibrated second physical propagation of existing Early Cross, `W_m = X2_m`.
- `complementary`: the same `X2_m` after removing its global Frobenius projection onto the actual Late SameNode reference, `S_m = Phi_m(U_private_other)`.

Both arms share one implementation, the same zero-initialized two-element parameter `t`, optimizer membership, candidate calibration, model initialization, classifier initialization, data protocol, and training configuration. The only mechanism switch is the definition of `W_m`. There is no third arm and no tuning after validation results are observed.

## Frozen source call order and insertion point

In `src/models/mrcge_v13.py`, `Model.forward` computes projectors `h`, physical and semantic bases, and then both modalities' private responses and `U_private` values before entering its collaborative-modality loop. Within that loop it calls `_early_cross` to form the existing calibrated `X1_m`; constructs `B_collab[1] = B1 + beta_m * X1_m`; then calls `_filter` for the collaborative Physical route. `SharedExpertBank.coefficients()` supplies the live signed-L1 alpha values. The filter forms `G_old[m,k] = sum_l alpha[k,l] B_collab[l]`, then applies each expert bottleneck and LayerNorm. The Router receives `current`, the first collaborative basis column, and `X1_m`; it does not consume the post-filter expert input.

The new model is a subclass in `src/models/mrcge_v13_two_hop_cross.py`. It preserves the parent's private, semantic, late, fusion, and initialization paths. Its finite `forward` override follows the frozen V1.3-A sequence and changes only the collaborative Physical filter call: the calibrated candidate is supplied as an explicit tensor argument and added to each `G_old[m,k]` immediately before the existing expert bottleneck. It does not edit `B_collab`, so the Router still sees only the original `B1 + beta_m X1_m` and `X1_m`. The added term is `beta_m * alpha[k,1] * tanh(t[m]) * Z_m`.

## Candidate construction

The physical operator is exactly `Model._physical_operator` from the frozen parent: remove self-loops, preserve the directed edge orientation and existing symmetric degree weights, and use its destination support mask. `X2_m` is one call to the repository's `chunked_weighted_spmm(X1_m, edge_index, edge_weight, physical_chunk_size)`, followed by the same support mask. No third hop, new edge weights, or changes to `B0..B3` are made.

For `raw`, `W_m = X2_m`. For `complementary`, use the cross projection already used by `_late` to build `S_m = Phi_m(U_private_other)`, detach it, and compute the global supported-node Frobenius coefficient

`a_m = <X2_m[M], S_m[M]> / (||S_m[M]||_F^2 + 1e-8)`.

The coefficient is accumulated in FP64 node blocks and detached. Then `W_m = M * (X2_m - a_m * S_m.detach())`. The dynamic reference is stop-gradient only along this new projection path; the existing gradients through Private, Late, Cross Projection, and `X2` remain as in the frozen model. This is a **dynamic, stop-gradient SameNode reference direction used for analytic projection**, not a fully end-to-end differentiable orthogonality constraint. Raw records its alignment cosine to the same reference for diagnosis, without applying a projection.

## Shared candidate calibration and numerical rules

On physical support `M`, `reference_rms = RMS(X1_m[M])` and `candidate_rms = RMS(W_m[M])`. If `M` is empty, `reference_rms <= 1e-8`, or `candidate_rms <= 1e-3 * reference_rms`, return zero. Otherwise use detached `scale = min(reference_rms / (candidate_rms + 1e-8), 4.0)` and `Z_m = M * W_m * scale`. The thresholds, epsilon, and cap are frozen; no threshold search is allowed. Record degeneration and cap events and the actual RMS. Projection inner products and cosine diagnostics use blockwise FP64 scalar accumulation; tensors remain in their model dtype. No `N x N` matrix is constructed.

The new trainable parameter is `t = nn.Parameter(zeros(2))`; `lambda_m = tanh(t[m])`. Zero initialization consumes no random numbers. The new item is added to `G_filtered` only, before Bottleneck/LayerNorm, with the live `alpha[:,1]`. It is passed explicitly into the filter/checkpoint function so backward recomputation sees the same tensor and coefficients. No mutable per-forward context is read by checkpoint recomputation. There is no added loss or parameter-specific optimizer setting.

## Initialization and execution-path invariants

At `t=0`, both arms must exactly preserve every shared initialized parameter, external NC classifier initialization, evaluation embedding/logit/auxiliary loss, and the ordinary training output and common-parameter gradient under paired RNG. Their state differs from V1.3-A only by `two_hop_raw`; Raw and Complementary share the same full state. The new parameters must receive finite, nonzero gradients on a valid synthetic graph.

The model tests cover the stacked/evaluation path and low-memory non-reentrant checkpoint path, including output and gradient parity, strict reload, RNG preservation, zero-cross/empty-support/isolated-node cases, Router invariance to `t`, and stop-gradient projection behavior. In the checkpoint path, `Z_m` and the expert addition coefficients are explicit checkpoint inputs, never model-global scratch state.

## Frozen training and data protocol

- Base: `988419283ef8f49c392de954aa739afc95c4c8cc`; source branch `exp/mrcge_v13_propagation_mechanism_audit`.
- New branch: `exp/mrcge_v13_two_hop_cross_screen`.
- V1.3-A `joint_view`, `M_repaired`, `sender_private_ratio=0`; hidden 256; 4 shared experts; Top-2; Physical orders 0–3; Semantic orders 0–1; SameNode and Joint View enabled.
- Full-graph NC, AdamW, LR `1e-3`, weight decay `1e-4`, up to 300 epochs, patience 30 after minimum epoch 30, min delta `1e-4`, gradient clipping 1.0, dropout 0.2, Late dropout 0.15, private auxiliary CE 0.3, balance loss 0.01, K=8.
- Existing fixed split seed 42, `auto_generate_splits=false`, V7B.1 cache. Verify feature, graph, split and split-file, cache, candidate and V7B fingerprints for each cell. Mask Test labels to `-100` before Trainer; `evaluate_test=false`; no Test metric access and no LP.
- Train and Validation metrics, Best-Validation-Accuracy and Best-Validation-Macro-F1 checkpoints, strict reload and complete epoch history are retained.

## Pre-registered experiment matrix and decision

Stage 1 is six Movies runs: Raw and Complementary, each at training seeds 42, 43, 44. Pair each seed with its historical V1.3-A Joint View checkpoint. Report the frozen Hybrid-CGE H2 fixed-split runs as context only.

An arm advances only if all conditions hold: mean paired Delta Accuracy at least +0.30 percentage points; positive Delta Accuracy in at least two of three seeds; mean Delta Macro-F1 at least zero; mean Delta CE no greater than +0.020; and no NaN, reload, or protocol failure. Report means and population SD. These are exploratory screening thresholds, not significance tests. If both pass, choose by mean Delta Accuracy, then (if within 0.10 pp) mean Delta Macro-F1, then Delta CE, then Raw. If neither passes, Stage 2 is zero runs and training stops.

Only the one selected arm may run Grocery seeds 42/43 and ele-fashion seeds 42/43. These four runs are a fixed cross-dataset confirmation, not a second opportunity to choose different variants by dataset. No other mechanism, tuning, or model development is authorized.

## GPU safety and stop conditions

Only physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`) is authorized. Reuse `run_mrcge_v13_screen.guard_gpu`; require at least 8 GiB free and at least measured peak Reserved plus 2 GiB before a formal run; one isolated subprocess and one GPU job at a time. Run a real single-step forward/backward preflight for both arms on each dataset before formal training. A newly appearing foreign PID, insufficient free memory, invalid fingerprint, or failed preflight stops the affected work without killing processes, switching GPUs, reducing the frozen configuration, or starting Training. Training is never started until the source freeze commit and tests pass.

## Interpretation boundary

Use Validation only for the registered screen. A gain is exploratory and does not establish statistical significance, causal contribution, or test generalization. Report both arms regardless of outcome, all failures and blocks, Train–Validation gaps, lambda trajectories, injection magnitudes, and the pre-registered decision. After commit and push, stop for independent review; do not upgrade the default architecture or begin another experiment automatically.
