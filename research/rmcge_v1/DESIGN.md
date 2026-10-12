# RM-CGE V1 design review

**Status:** frozen before implementation and before any RM-CGE training result was observed.
**Baseline:** `e32eaeb676b3851cffc0067c549f42a748569e12`
**Task:** seed-42 Full-Graph Node Classification on Movies, Grocery, and ele-fashion; Validation-only model selection.

## 1. Scope and design decision

RM-CGE V1 builds two target-modality views over the same node set. Both views use the same physical graph. The Text target uses the Text-source KNN support; the Visual target uses the Visual-source KNN support. Within each graph layer, the two target views exchange messages over the target view's relation support, then use one shared expert bank and one shared router. The two updated modality states remain separate until the final intrinsic-preserving fusion.

The frozen computation is:

```text
Text / Visual features
  → independent two-layer residual projectors
  → two graph layers, each with Physical and target-modality Semantic contexts
  → same-support cross-modal context conversion
  → layer-shared three-expert bank and joint six-way router
  → 0.5 residual update
  → feature-interaction fusion plus initial-state skip
  → existing linear NC classifier
```

The model is full-graph NC only. It requires global node IDs and a validated V7B.1 K=8 candidate cache. It does not claim sampled Link Prediction support.

## 2. Sparse relation operators

Let (E_{phy}) be the physical graph after removing self-loops, canonicalizing each undirected edge, and removing duplicates. For an undirected adjacency (A_{phy}), let (d_i=sum_j A_{ij}). Then

\[
P_{phy}[i,j] = A_{phy}[i,j]/d_i \quad (d_i>0),
\]

with an all-zero row for (d_i=0). The row is therefore a uniform neighbor mean. The operator is not the symmetric (D^{-1/2}AD^{-1/2}) operator used by PACE-MAG V1.

The V7B.1 cache stores one canonical undirected candidate edge and two raw proposal-count columns. For each cached edge (e), define source presence by (b_T(e)=1[flags[e,0]>0]) and (b_V(e)=1[flags[e,1]>0]). Counts 1 and 2 both mean “present.” Build the Text support from all edges with (b_T=1), and the Visual support from all edges with (b_V=1). If both bits are present, that edge is included at full unit weight in both supports. Each support is symmetrized, independently deduplicated, and independently row-normalized:

\[
P_T=D_T^{-1}A_T, \qquad P_V=D_V^{-1}A_V,
\]

where empty rows remain exactly zero. There are no learned or similarity-derived edge weights, and (P_T+P_V=P_U) is not an invariant or implementation check. The cache is only read; FAISS is not run and K is not changed.

For target modality (m\in\{T,V\}), the relation supports are

\[
P_{T,phy}=P_{phy},\quad P_{T,sem}=P_T,\qquad
P_{V,phy}=P_{phy},\quad P_{V,sem}=P_V.
\]

One layer needs six sparse-dense multiplications: each of the three supports times each of the two layer-normalized modality states. Those results are reused for both same-modality and cross-modal contexts. Adjacencies remain sparse COO tensors; no dense (N\times N) matrix is created.

## 3. Projectors and layer contexts

Separate residual MLPs map (X_T\in\mathbb{R}^{N\times d_T}) and (X_V\in\mathbb{R}^{N\times d_V}) to (H_T^0,H_V^0\in\mathbb{R}^{N\times128}). Each projector has a learned linear shortcut and a two-linear-layer GELU residual branch, with dropout 0.2 and output LayerNorm. Projector parameters are independent across modalities.

At layer (l\in\{0,1\}), (H_m^l) is the current state. A learned LayerNorm is applied independently to each modality state before propagation. For each target view and relation:

\[
U_{m,r}^l=P_{m,r}\,LN_{l,m}(H_m^l),
\]

\[
C_{m,r}^l= M_{m,r}\odot\Phi_{l,\bar m\to m}
\left(P_{m,r}\,LN_{l,\bar m}(H_{\bar m}^l)\right),
\]

where (M_{m,r}[i]=1[d_{m,r}[i]>0]). (Phi) is a bottleneck MLP (128\to64\to128) with GELU and dropout. It runs **after** sparse aggregation. Within one layer, the same (\Phi_{V\to T}) is used by Physical and Text Semantic contexts, and the same (\Phi_{T\to V}) is used by Physical and Visual Semantic contexts. The two directions and the two layers have independent parameters. The mask is applied after the MLP so its biases cannot create an invalid-neighbor message. Text Semantic Cross is therefore (\Phi_{V\to T}(P_T LN(H_V))), and Visual Semantic Cross is (\Phi_{T\to V}(P_V LN(H_T))).

### Block scale calibration

The four prescribed expert-input blocks can have different distributions: the node state, a row mean, their signed difference, and a learned cross-modal transform. To make the scale rule explicit and identical for every relation and modality, each block is passed independently through the same per-layer, non-affine feature LayerNorm before concatenation:

\[
B_{m,r}^l=\operatorname{concat}\bigl(
LN_b(H_m^l),\ LN_b(U_{m,r}^l),\ LN_b(H_m^l-U_{m,r}^l),\ LN_b(C_{m,r}^l)\bigr).
\]

The same (LN_b) object is reused for all four blocks, both relations, and both target modalities in that layer. It has no affine parameters and does not change the propagation operator. An empty context and its cross-modal block stay exactly zero under this normalization.

## 4. Shared experts and joint routing

Each layer owns three separate expert modules. Every expert maps (4d\to64\to d), uses GELU and dropout 0.2, and is called for all four relation-target combinations. Thus an expert object is shared across Physical/Semantic and Text/Visual within a layer. The two layers have independent expert banks. Experts have no assigned semantic roles and no auxiliary classifiers.

For every node and target modality, one layer router scores the six relation-expert candidates. Its shared input for a relation is the normalized current node state, normalized same-modality context, normalized cross-modal context, a learned 8-dimensional relation embedding, a learned 8-dimensional target-modality embedding, and (\log(1+d_{m,r})). Its two-layer MLP has hidden width 64 and returns three expert corrections. It is the same router object for both modalities and both relation contexts in that layer.

The learnable relation log-prior is initialized to

\[
b_{phy,k}=\log(0.6/3),\qquad b_{sem,k}=\log(0.4/3),\quad k=1,2,3.
\]

The final router correction layer uses a small, nonzero weight initialization (standard deviation 0.01), and the prior itself is trainable. These choices make the initial Physical/Semantic marginal approximately 0.6/0.4 while allowing it to move immediately under gradients. A single masked softmax is applied over all six candidates. Invalid relation logits are masked before normalization. If only one relation is valid, its three expert probabilities sum to one. If neither is valid, a safe zero-logit fallback is softmaxed and then fully masked to exact zeros, avoiding `softmax([-inf,…])` and NaNs.

\[
\pi_{m,r,k}^l=\operatorname{MaskedSoftmax}_{r,k}(s_{m,r,k}^l),\qquad
\Delta_m^l=\sum_{r\in\{phy,sem\}}\sum_{k=1}^3
\pi_{m,r,k}^l E_{l,k}(B_{m,r}^l).
\]

There is no per-relation softmax followed by a fixed relation mixer. There is no load-balance objective, Top-K selection, or role-specific head. Expert collapse is monitored from the actual routed mass, entropy, and expert outputs rather than prevented with an extra loss.

## 5. Two residual graph layers and output

Each layer applies an outer dropout to the routed update and a fixed residual scale:

\[
H_m^{l+1}=H_m^l+0.5\,Dropout(\Delta_m^l).
\]

The second layer uses (H_T^1,H_V^1) from the first update and recomputes all six sparse propagation results. The supports themselves remain fixed. The model keeps (H_T^0,H_V^0) for the skip path.

The final graph and intrinsic terms are

\[
G=W_f[H_T^2,H_V^2,|H_T^2-H_V^2|,H_T^2\odot H_V^2],
\]

\[
S=W_s[H_T^0,H_V^0],\qquad Z=LN_f(G+S)\in\mathbb{R}^{N\times128}.
\]

The model returns the repository's five-item API `(z, None, None, aux_loss, info)`, with an exact zero scalar `aux_loss`. NC optimization remains Cross-Entropy only; the existing runner supplies the linear classifier and selects by Validation Accuracy. Pilot runs explicitly set `evaluate_test=false`.

## 6. Engineering review and failure checks

| Risk | Frozen choice | Evidence to collect |
|---|---|---|
| Relation support accidentally shared or split by raw count | Build from per-source `flags > 0`; Both edges remain full in both graphs | Synthetic source-flag tests, support/degree counts, cache fingerprints |
| Wrong cross-modal semantic support | Aggregate the other modality on the target support before direction-specific (\Phi) | Direct operator equality tests for Text-on-(P_T H_V) and Visual-on-(P_V H_T) |
| Invalid-row MLP bias | Mask after (\Phi), mask all three routed responses, and use an all-invalid-safe softmax | Isolated-node and no-neighbor forward tests; exact zero updates |
| Independent relation normalization or router | One row-stochastic support per relation; one joint 6-way masked softmax per layer | Row-sum, route-sum, single-valid, and all-invalid tests |
| Experts/router accidentally copied per view | One module bank and router stored per layer and passed to all four contexts | Object identity, parameter identity, and gradient/update audit |
| Layer 2 reuses stale inputs | Recompute six SpMM results from the outputs of layer 1 | Instrumented state-flow test and full backward preflight |
| Expert activation memory | Stream relation contexts and expert calls; optional non-reentrant layer activation checkpointing plus configured chunk size | Largest-graph GPU preflight, peak allocated/reserved bytes |
| Expert starvation | Do not add an unrequested balancing loss; retain a trainable small-logit router and measure use | Layer/modality/relation marginal mass, expert mass, entropy, output RMS/cosine |
| Skip masks a weak graph branch | Keep the requested skip and record pre-LN graph-term RMS, skip RMS, and frozen branch-off Validation changes | Best-checkpoint diagnostics; interpret as fitted-model dependence only |
| Cross-modal scale too weak or unstable | No multiplicative gate or tiny output initialization; normalize the four expert blocks consistently | Cross-context RMS relative to same-modality context, finite gradients, one real AdamW update |
| PACE/V7 behavior changes | Add RM-CGE files and a small opt-in telemetry hook only if required for exact epoch history | Targeted PACE/V7 regression tests and clean diff against their source/artifacts |

The trainer currently loads the complete label tensor but the RM-CGE run uses only `train_idx` for loss and `val_idx` for selection/metrics. With `evaluate_test=false`, the NC runner builds its F1 label set from Train and Validation only and does not index `test_idx` for an evaluation. The run metadata and report will record this boundary explicitly.

## 7. Relationship to prior methods and overlap risk

MvCGE provides the conceptual starting point: shared graph experts across graph views with a graph-aware router. RM-CGE V1 carries that parameter-sharing/collaborative-routing idea into two MAG target-modality views, while changing the graph experts into feature-context MLPs over fixed Physical and source-specific KNN row means. It does not reproduce MvCGE's spectral expert construction or its load-equilibrium and graph-discrepancy losses. See [MvCGE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7315c16422558ae81d65a812723d2cec-Abstract-Conference.html).

The method also uses fixed-support cross-modal context conversion. This is materially narrower than ReCoG's jointly learned multimodal graph refinement and coupled propagation: RM-CGE neither edits edges nor co-evolves topology with embeddings. It should not be described as ReCoG propagation or dynamic graph collaboration. See [ReCoG, arXiv:2608.22786](https://arxiv.org/abs/2608.22786).

Relative to PACE-MAG V1, RM-CGE changes source allocation, operator normalization, the location and scope of cross-modal exchange, relation/expert normalization, and depth. It has no PACE V1 source half-splitting for Both edges, symmetric physical normalization, row-quality weighted source allocation, bounded (\rho/\eta) gates, or post-update physical propagation. The prior PACE mechanism audit is background evidence only; its Validation numbers are not a paired comparison to this pilot.

Shared MoE banks, graph-conditioned routing, relation-specific propagation, modality projectors, and residual fusion are established ingredients. The combined RM-CGE recipe therefore has clear overlap risk with multi-view GNN/MoE and multimodal graph message-passing work. This implementation and three single-seed Pilot scores alone do not support a novelty or superiority claim. A claim-level literature review and controlled multi-seed study would be needed before positioning it as a research contribution.

## 8. Freeze boundary

After the implementation tests and largest-graph one-step preflight pass, the implementation and fixed configuration are committed as the Implementation Freeze. The only subsequent work in this task is the three prescribed seed-42 full RM-CGE trainings, frozen-checkpoint diagnostics, report/data generation, and Pilot Report commit. Pilot scores do not authorize architecture edits, additional seeds, Test evaluation, or automatic tuning.
