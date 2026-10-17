# Hybrid-CGE V1.1 Frozen Design

## Scope

This study adds exactly two conditions: `H1_hybrid` and `H2_hybrid_plus`. It is a validation-only, full-graph node-classification study on Movies, Grocery and ele-fashion with seeds 42/43/44. Historical View-CGE A2 and CG-CGE B1 are read-only references. No Test metric, Role-FiLM, pair-content gate, optimal transport, contrastive loss, graph rewiring or extra expert bank is included.

The model is implemented as a focused extension of CG-CGE V1 and reuses its V7A/V2 trunk, V7B.1 semantic-cache validation, directional 256→64→256 projections, four shared residual experts, Top-2 routing, channel gate, fusion head, and custom receiver-sparse attention. `src/tasks/nc.py` remains unchanged; its existing opt-in full-graph unimodal-loss interface consumes logits computed from the Private states.

## Operators and state flow

For modality `m` and opposite modality `o`, `H_m = Projector_m(X_m)`. The physical operator P is the existing normalized operator after self-loop removal. Same-source KNN operators and opposite-source KNN supports come from the frozen V7B.1 cache (K=8); no retrieval, labels or cache normalization changes are introduced.

Four physical propagation states are computed once per modality:

`G_m^l = P^l H_m`, `l=1..4`.

Each hop is channel-wise RMS-normalized over physical-active nodes using the inherited operator, and shared `alpha[k,l]` coefficients form `B[m,k]`. The existing directional Phi maps the opposite modality into the receiver space.

### Private state

`R_priv[m] = sum_k route_priv[m,k] * (B[m,k] + 0.1 F_k(B[m,k]))`.

The Private physical route uses the inherited physical router with zero cross correction. The Same-source branch uses `S_m H_m`, the existing same-source Top-2 expert response and `gamma_S = softplus(gamma_S_raw)` initialized to 0.08:

`U_priv[m] = H_m + rho[m] R_priv[m] + gamma_S[m] R_S[m]`.

The Private calculation accepts only `H_m`, its own same-source operator, P and model parameters. It does not accept or read `H_o`.

### Early Physical Cross collaboration

Compute `C_P[m] = P Phi[o→m](H_o)`. Normalize by the detached, per-channel RMS of the receiver's first physical hop, using the inherited scale floor; mask physical-isolated receiver nodes to zero. The four expert coefficients `b[k]` are shared across modalities, independently trainable, and initialized to 0.25:

`B_collab[m,k] = B[m,k] + b[k] C_hat_P[m]`.

The collaborative route calls the same physical router and the same `_node_correction` parameters as the Private route, but passes the actual `C_hat_P` in its Cross input. The four expert objects are also the same objects used by Private, Same-source and Late Complementary calls. Physical strength `rho[m]` is reused. The state is formed by a single physical delta so Same-source response is counted once:

`U_collab[m] = U_priv[m] + rho[m] * (R_collab[m] - R_priv[m])`.

This is the only physical-cross message path. Late Completion never constructs a Physical Cross candidate. Isolated physical receivers have zero early-cross message and use the inherited active-node routing mask.

### Late Semantic and Same-node completion

Semantic Cross uses only the opposite-source KNN edges. For Visual→Text:

`q_i = W_Q LN(U_collab_text[i])`,
`k_j = W_K LN(U_priv_visual[j])`,
`v_j = Phi[visual→text](0.7 H_visual[j] + 0.3 U_priv_visual[j])`.

The existing custom sparse operator computes receiver-wise edge softmax over `q_i·k_j/sqrt(32)` and aggregates only listed edges. Dense all-pairs attention is forbidden. Text receivers use Visual-KNN; Visual receivers use Text-KNN.

H2 additionally computes `C_self[m,i] = Phi[o→m](U_priv[o,i])`; it does not use a graph edge. The MAG data interface provides dense Text and Visual feature matrices for every node but no reliable modality-presence mask. Therefore Same-node validity is all nodes for these datasets; zero-valued vectors are not treated as missing. H1 skips the Same-node projection, supplies a zero placeholder, and permanently masks that selector logit.

Both conditions use a common three-output selector with source order `[Semantic, SameNode, None]`, input dimension `3*256+5`, zero-initialized final weights and final bias `log([0.4,0.4,0.2])`. Its shared inputs are `U_collab`, both candidate tensors, candidate cosine consistency, each candidate's RMS relative to `U_collab`, and the two validity flags. H1 masks SameNode, giving initial Semantic/None probabilities 2/3 and 1/3. Unsupported candidates are masked before softmax. With no valid candidate, None is exactly one and the selected candidate and final residual are exactly zero.

`p_accept = alpha_sem + alpha_self`,
`C_bar = (alpha_sem C_sem + alpha_self C_self) / max(p_accept, 1e-6)`.

The acceptance probability appears once in the residual scale and is never counted as a fraction of positive softmax values.

### Complementary Experts and relative control

`F_inter = [U_collab, C_bar, U_collab*C_bar, abs(U_collab-C_bar)]`.

The shared channel gate is sigmoid-initialized to mean 0.73. A shared interaction MLP produces J; a modality-conditioned Top-2 router mixes the same four experts:

`D = lambda[m] p_accept g (C_bar + 0.25 E_comp)`.

`lambda[m] = 0.5 sigmoid(lambda_raw[m])`, initialized to 0.25. Relative control is applied to the whole residual without amplification:

`scale_i = min(1, tau[m] ||U_collab[i]||_2 / (||D[i]||_2 + eps))`,
`Delta_i = scale_i D_i`,
`tau[m] = 0.1 + 0.7 sigmoid(tau_raw[m])`, initialized to 0.25.

The cap is measured before H2 dropout. H2 then applies non-inverted, per-node dropout with probability 0.15 to the entire final `Delta` during training only; evaluation and inference disable it. H1 has dropout probability zero. Neither condition drops Early Physical Cross or fusion.

Finally, `U_final[m] = U_collab[m] + Delta[m]`; the inherited late fusion maps the concatenated two states to the 256-dimensional graph representation. The model performs one bidirectional Late Completion pass after both Collaborative states are ready, avoiding cycles.

## Objective and auxiliary balance

The existing trainer composes:

`L = CE(Head(Z[train_idx]), y_train) + 0.3 * 0.5*(CE(Head_T(U_priv_T),y_train)+CE(Head_V(U_priv_V),y_train)) + ExistingBalanceAux`.

Private heads consume only Private states. ExistingBalanceAux already includes the model's balance weight and is passed once with `task.loss.aux_weight=1.0`; it is not multiplied by 0.01 again. Balance is the mean across the four physical Private/Collaborative calls, two Same-source calls and two Complementary calls. This averages the duplicated physical route role rather than doubling its balance contribution.

## Initialization and comparability

H1/H2 use the same module declaration, tensor initialization and state-dict keys. H2's dropout probability changes only training-time stochastic behavior. New scalar/vector parameters (`b`, bounded `tau`) are deterministically initialized; added module initialization is kept inside a forked RNG so the external classifier initialization is not shifted. For every Dataset×Seed, H1/H2 must have matching initial model and classifier hashes, official split, features, physical graph, semantic cache, candidate and V7B.1 fingerprints.

## Deliberate evidence boundaries

Source weights, routing, early-cross scale, residual/control statistics and auxiliary loss are execution diagnostics. They do not by themselves establish causal utility. The formal conclusion is based on paired Validation results and the strict historical fingerprint gate. Three seeds are descriptive; no Test access is allowed.
