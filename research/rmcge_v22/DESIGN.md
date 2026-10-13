# RM-CGE V2.2: Receiver-Conditioned Cross-modal Message Experts

## Research question

V2.1 changes how much each incoming physical edge contributes through scalar neighbor reweighting. V2.2 tests whether a target node can also choose *which low-rank feature content* to receive from each cross-modal sender, while retaining the four-hop physical expert profiles, Cross Base, KNN context, shared expert bank, Top-2 Node Router, Physical Strength, and Fusion.

## Shared base and five conditions

| Arm | Cross-modal content path | V2.1 Intra selection |
|---|---|---|
| M0 | off | off; forward is checked against V2.1 C0 |
| M1 | sender-only gate | off |
| M2 | receiver-only gate | off |
| M3 | receiver–sender pair gate | off |
| M4 | same pair gate as M3 | on; exact C1 path |

The implementation constructs a common V2.1 C1 initialization tree in every arm and masks its Intra path in M0–M3. This keeps public and selector tensors paired without changing the active M0 function. Content modules are declared in all five arms and initialized inside an RNG fork. Thus all arms share identical initial model state and subsequently constructed classifier state. Dormant parameters do not receive gradients; equal parameter totals do not imply equal active capacity.

## Content transform

For target modality `m`, projected node state is `H_m`; the existing opposite-modality projection before neighborhood aggregation is `Xcross_m = Phi_other_to_m(H_other)`. Shared maps produce:

`q_i = W_query(H_m[i])`, `k_j = W_key(Xcross_m[j])`, `v_j = W_value(Xcross_m[j])`,

with dimensions 32, 32, and 32. No extra normalization is applied. The gate input is `[q, k, q*k, abs(q-k), modality_embedding]` (136 features), followed by `Linear(136,64)`, GELU, `Linear(64,32)`, and tanh.

- M1 sets q to zero. Its sender transform is computed before sparse propagation.
- M2 sets k to zero. It computes `gate_i * (P @ v)_i`, equivalent to gating every incoming neighbor before aggregation because the gate is shared across all senders of receiver i.
- M3/M4 evaluate the gate separately for every directed `j -> i` edge.

For Pair mode, `lowrank_i = sum_j P_ij * gate_ij * v_j` and `D_content_i = W_output(lowrank_i)`. Chunked custom autograd recomputes edge gates in backward and keeps only node state, edge indices, and one `E_chunk x 32` working set. It does not build dense adjacency or `E x 256` messages. Original symmetric degree-normalized P is unchanged; there is no new softmax or denominator.

The final Gate layer starts at zero, W_output uses ordinary nonzero initialization, and four shared expert scales start at `eta_k=0.25`. `D_content` is divided by the detached per-channel first physical-hop RMS already used by V2/V2.1. Zero-degree output is exactly zero. The content term enters only the expert input:

`Q_m,k = PhysicalProfile_m,k + b_k CrossBase_hat_m + c_k KNN_hat_m + a_k IntraDelta_hat_m + eta_k Dcontent_hat_m`.

It does not enter Node Router features or the auxiliary loss. M4 uses original C1 IntraDelta and original P for the new content path; it does not use C2 Cross Scalar Selection.

## Interpretation boundary

M3−M2 measures the observed increment associated with adding sender-specific conditioning; M3−M1 measures the increment associated with receiver state; M4−M3 measures the package effect of enabling C1 alongside Pair content. These are not strict capacity-matched causal ablations: M1/M2/M3 expose different input channels and M4 activates extra selector capacity. Three seeds are descriptive paired evidence. Gate changes, gradients, and frozen interventions establish pathway activity or checkpoint sensitivity, not standalone causal utility.

The protocol is `unified_full_graph_nc_v1`, full graph transductive features/topology, official seed-specific Train/Validation/Test split, AdamW, lr `1e-3`, weight decay `1e-4`, dropout `0.2`, max 300 epochs, patience 30, minimum epoch 30, gradient clip 1.0, Validation Accuracy stopping/checkpoint selection, separate Best Macro-F1 checkpoint, and the inherited `CE + 0.01 * balance` objective. Test evaluation is disabled and Test labels are replaced by a sentinel before calling the shared NC trainer.
