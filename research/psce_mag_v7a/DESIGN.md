# PSCE-MAG V7A design

## Question

Do label-free, feature-neighbor relations add useful structural context beyond the raw MAG, and does node-conditional use of a shared physical/semantic expert bank improve over a modality-static semantic view?

## Candidate relation cache

For each modality independently, subtract the all-node feature mean and L2-normalize each nonzero vector. Retrieve eight non-self cosine neighbors using an indexed FAISS search. Form the union of Text and Visual proposals, canonicalize node pairs, remove self-links/duplicates, and symmetrize. Candidate degrees are fixed by this topology. The cache stores only candidate indices and degrees plus dataset/rule/version/feature-hash/index provenance and diagnostics; it never stores learned edge weights or labels. Training and inference recompute `a_ij` from the current relation scorer.

The candidate graph is external to NC splits and invariant across seeds. Exact retrieval is used where its indexed search is practical; the large graph uses a recorded approximate FAISS index and search settings. A label-free held-out feature-query recall sample measures retrieval quality without changing K or consulting validation performance.

## Model

The common trunk follows V4A `R0_raw`: independent modality projectors; symmetric, self-loop-free physical propagation for four hops; active-node RMS; four Hadamard coefficient rows; four shared bias-free bottleneck experts; modality-static physical Top-2 routes and structural strengths; and the existing late fusion MLP/LayerNorm. A0 is this path with semantic strength fixed to zero.

For candidate pair `(i,j)`, learned 32-dimensional modality projections `uᵀ,uⱽ` form symmetric relation features `[u_i ⊙ u_j, |u_i-u_j|]` per modality. A shared MLP `128→64→1` produces `a_ij=sigmoid(s_ij)`. Candidate propagation uses `a_ij / sqrt(d_i^cand d_j^cand)` with the cached degrees, so changing the scorer's global scale changes semantic output rather than being normalized away.

Semantic states are `S1=Psem H0` and `S2=Pphy S1`. Semantic states use a one-sided, detached RMS cap against their corresponding Raw state on a defined active-node set; a weak semantic state is never expanded to unit RMS. Four learnable semantic coefficient rows combine the two semantic hops. The same expert functions used for physical profiles transform semantic profiles; there is no second expert bank. The final modality output stays `H0 + τ Cphy + g Csem` and uses the original fusion head.

A1 and A2 use one router implementation with identical parameter shapes and initialization. A1 pools a modality-global semantic context and broadcasts its expert probabilities and semantic strength. A2 routes each node from its intrinsic state, first physical state, first semantic state, modality embedding, and semantic amplitude. A0 does not execute semantic scoring, propagation, semantic expert transforms, or semantic routing; its physical expert bank remains active.

## Efficient differentiable execution

Relation scoring is chunked and backward-recomputed, retaining only node-level projections, candidate indices, and edge weights. Weighted sparse propagation uses a custom chunked autograd operation: forward accumulates edge-weighted source states by destination; backward recomputes source gathers and returns gradients for node states and edge weights. Shared expert FFNs are activation-checkpointed during training. This avoids persistent `E × hidden_dim` message tensors while preserving gradients to the relation scorer. No `N × N` or `E × experts × hidden_dim` tensor is formed.

## Interfaces and checkpoints

`src/tasks/nc.py` attaches and validates the semantic cache before moving the model to the selected device. Cache edge buffers are non-persistent; the model state stores persistent candidate-graph and input-feature fingerprints. Loading a checkpoint compares both fingerprints with the already attached cache and fails on mismatch. Full-graph inference uses the same model and cache. Any LP call fails with an explicit global-ID/local-sampled-ID incompatibility error.

## Variants and primary comparisons

- A0 `a0_raw`: physical Raw branch only.
- A1 `a1_static_dual`: A0 plus learned semantic graph, two semantic hops, shared experts, static semantic use.
- A2 `a2_conditional_dual`: same parameter count and initial tensors as A1; only semantic routing/use is node-conditional.

Primary comparisons are A1−A0 (semantic context), A2−A1 (conditional view use), and A2−A0 (full model). They are paired by dataset and seed, reported descriptively without significance claims. Gate/edge values indicate learned branch use, not causal task utility.

## Frozen initial configuration

`hidden_dim=256`, physical hops 4, semantic hops 2, four experts, Top-2, bottleneck 64, expert residual scale 0.1, `text_knn_k=8`, `visual_knn_k=8`, relation dimensions 32/64, initial semantic edge weight 0.8, initial semantic strength 0.2, dropout 0.2, and balance weight 0.01. The same values apply to every dataset.
