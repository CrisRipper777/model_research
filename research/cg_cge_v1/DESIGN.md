# CG-CGE V1: Complementary Graph Experts

## Status and scope

This design is frozen for the study on branch `exp/cg_cge_v1_complementary_experts`, parented at `ffdb8c2b53ea4c67793d26c420832aa9a118cb58`. The study compares B1 and B2 as whole models. It does not start another architecture iteration and does not evaluate Test.

- **B0:** frozen View-CGE V1 `A2_cross_source` results. It is an external reference only when all dataset/seed split, feature, semantic-cache, candidate, V7B.1 and NC protocol fingerprints match.
- **B1:** independent Text/Visual graph encoders, physical and same-source KNN context, one-pass complementary completion, source selection with a no-completion option, channel gate, shared experts and train-only unimodal CE.
- **B2:** B1 with zero-initialized Role-FiLM on Semantic and Complementary calls to the same four experts. B1 and B2 instantiate the same complete parameter tree and public values; B1 freezes and bypasses FiLM. Physical expert calls never use FiLM.

## Frozen computation

For modality `m`, the inherited V7A projector produces `H_m`. Four normalized physical-hop profiles are mixed by the learned physical layer coefficients. The inherited physical router and node correction receive only the modality's own prior, first physical hop, and explicit zero delta/cross tensors. Their four expert responses form `R_P^m`; no other modality is read at this stage.

The same-source V7B.1 support produces `Q_S^m = S_m H_m`. Its response uses the same expert objects, a small view router and `gamma_S` initialized to 0.08. Thus `U_m = H_m + rho_P R_P^m + gamma_S R_S^m` is final before any cross-modal operation.

For `o -> m`, the sender value is `Phi_(o->m)(0.7 H_o + 0.3 U_o)`, using the inherited directional 256→64→256 projection. `P V` gives the physical candidate. The semantic candidate uses the opposite source's cached KNN support, receiver query `W_Q^m LN(U_m)`, sender key `W_K^o LN(U_o)`, and a receiver-wise sparse softmax over listed incoming edges only. The custom autograd operation stores edge scalars and recomputes `E_chunk × 256` temporaries; it does not form dense `N × N` attention or retain an `E × 256` activation.

Each candidate is scaled by `min(1, detached physical-reference RMS / detached candidate RMS)`, measured only on its valid support. A floor stabilizes the scale and cannot enlarge a weak candidate. The source selector masks unavailable candidates and selects P, S or None. Empty receivers are explicitly `[0, 0, 1]`. The accepted mixture is normalized by `p = alpha_P + alpha_S`, and `p` is multiplied exactly once in the residual. A zero-initialized final selector layer starts at P/S/None = 0.45/0.35/0.20 when both sources exist.

A dimension-wise sigmoid gate starts at 0.73. An interaction MLP and Top-2 router condition the shared experts for complementary messages. The final residual is `lambda_m * p * g * (Cbar + 0.25 E_C)`, with `lambda_m = 0.5 sigmoid(raw_m)` initialized to 0.25. The two directions are computed once and fused by the inherited late fusion block.

## Objective and protocol

The NC trainer adds an opt-in `task.loss.unimodal_weight`. At 0.3, it adds the mean of Text and Visual CE on `train_idx` only. Existing `aux_loss` already contains `balance_weight` and enters with the established `aux_weight=1.0`. Validation selects checkpoints from the main fused representation only. Test indices are set to label `-100` before the trainer call; Test evaluation is disabled.

The formal matrix is Movies/Grocery/ele-fashion × B1/B2 × seeds 42/43/44, under `unified_full_graph_nc_v1`, AdamW, lr `1e-3`, weight decay `1e-4`, dropout `0.2`, at most 300 epochs, patience 30, minimum epoch 30 and gradient clipping 1.0. The primary checkpoint is best Validation Accuracy; a separate best Validation Macro-F1 checkpoint is retained.

The two conditions are paired by dataset and seed. Their model state and external classifier head initial hashes must match before training; split and cache fingerprints must match. B0 metrics are included in paired analyses only if exact protocol fingerprints pass. Three seeds are descriptive evidence, not a precise population uncertainty estimate.

## Deliberate exclusions

This version has no learned physical edge attention, repeated cross-modal feedback, contrastive/OT objectives, topology changes, pair-content gates, extra expert banks, or auxiliary validation selection. Link prediction and sampled local node IDs are rejected because semantic supports use global cache IDs.
