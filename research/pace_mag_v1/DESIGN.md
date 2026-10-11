# PACE-MAG V1 design

## Research question and responsibilities

PACE-MAG asks whether physical edges and Text/Visual KNN edges carry useful, distinct neighborhood context for multimodal attributed graph node classification, and whether those contexts benefit from a shared graph expert bank. The V1 implementation is a supervised **Full-Graph Node Classification** model. It does not support Link Prediction or sampled subgraphs.

Each block has one job:

1. Two independent projectors map frozen Text and Visual features to the same hidden width.
2. A fixed physical graph and the frozen V7B.1 Text/Visual candidate union provide three sparse propagation views.
3. Small nonlinear cross-modal bottlenecks modify the messages sent along the two semantic views. Physical messages stay within their modality.
4. The same three graph experts process every relation-view/modality pair. A graph-aware router combines their outputs.
5. Bounded global branch strengths integrate physical and semantic responses; one fixed physical propagation follows; a final multimodal fusion produces the node representation.

The NC runner supplies the linear classifier. The only objective is Cross-Entropy; `aux_loss` is an exact zero scalar.

## Operators and source allocation

Let the simple physical graph omit self-loops and duplicate undirected edges. Its symmetric normalized operator is

\[
P_0=D_0^{-1/2}A_0D_0^{-1/2}.
\]

The cached semantic union uses the V7B.1 fixed-KNN edge weight. For a cached undirected edge \(\{i,j\}\), with semantic union degrees \(d_i,d_j\),

\[
w_{ij}^{U}=0.8(d_i d_j)^{-1/2}.
\]

The two raw source-count columns are not binary indicators. Their presence masks are `flags[:,0] > 0` and `flags[:,1] > 0`. A Text-only edge gives all \(w_{ij}^{U}\) to \(P_T\); a Visual-only edge gives all of it to \(P_V\); a Both edge gives half to each. Reverse directions receive the same values. Thus, before message aggregation,

\[
P_T+P_V=P_U
\]

up to floating-point tolerance. The split operators are not independently degree-normalized. E0 instead sets \(P_T=P_V=0.5P_U\).

The cache loader checks the V7B schema, feature/candidate/source fingerprints, node count, sorted unique canonical edges, source-flag shape and values, and exact alignment of the symmetric directed edge layout. The model checks its actual input modality arrays against the cache feature hashes on first use. Physical self-loops and repeated edges are removed while building \(P_0\).

## Modalities, messages, and row-quality aggregation

For modality \(m\in\{T,V\}\), an independent residual MLP projector gives

\[
H_T=\operatorname{Projector}_T(X_T),\qquad H_V=\operatorname{Projector}_V(X_V).
\]

Two shared-direction nonlinear bottlenecks produce \(\Phi_{V\to T}(H_V)\) and \(\Phi_{T\to V}(H_T)\). For semantic source \(r\in\{T,V\}\),

\[
A_r^m=H_m+\rho_r^m\Phi_{\bar m\to m}(H_{\bar m}),\qquad A_0^m=H_m.
\]

Four bounded learnable \(\rho_r^m\) values start at 0.1 and lie in \((0,0.25)\). The cross-modal output layer starts small but remains trainable. When \(\rho=0\), the message is exactly \(H_m\). E1 skips both bottlenecks and the \(\rho\) path entirely.

For each view, the receiver row quality and normalized neighborhood context are

\[
q_r=P_r\mathbf 1,\quad M_r=\mathbb{1}[q_r>0],\quad
U_r^m=M_r\odot\frac{P_rA_r^m}{\max(q_r,\epsilon)}.
\]

This row-quality normalization is intentional. It is not the complete numerical behavior of the V7B.1 symmetric normalized operator. Empty rows yield exactly zero context; expert outputs on invalid rows are masked back to zero so biases cannot invent a message.

## Shared graph experts and routing

After LayerNorm,

\[
B_r^m=[H_{norm}^m,U_{norm,r}^m,|H_{norm}^m-U_{norm,r}^m|,
H_{norm}^m\odot U_{norm,r}^m]
\]

In code the expert input has the specified four blocks: `H_norm`, `U_norm`, `abs(H_norm-U_norm)`, and `H_norm*U_norm`, for width \(4d\). Three bottleneck MLP experts map \(4d\to64\to d\). The same module objects are called for all three relations and both target modalities.

The shared router receives the node representation, row-normalized context, signed difference, relation and modality embeddings, the valid-node pooled context for that view, and \(\log(1+q_r)\). A softmax over the three experts gives \(\pi_{i,r,m,k}\), and

\[
V_r^m=\mathbb{1}[q_r>0]\sum_{k=1}^3\pi_{i,r,m,k}\operatorname{Expert}_k(B_r^m).
\]

There is no Top-K selection, balance penalty, contrastive loss, edge scorer, label-conditioned routing, or expert role assignment. E4 keeps the same router parameters but supplies pooled view/modality context only; each valid node in that view therefore receives the same expert mixture. For node counts above 32,768, expert input evaluation is chunked and non-reentrant activation checkpointing recomputes chunk MLP activations during backward. This is an execution-memory choice and leaves the operator/model equations unchanged.

## Relation collaboration and output

The physical response is \(V_{phy}^m=V_0^m\). Semantic responses use the receiving-node row qualities:

\[
V_{sem}^m=\frac{q_TV_T^m+q_VV_V^m}{\max(q_T+q_V,\epsilon)}.
\]

Two bounded global scalar strengths start at \(\beta_{phy}=0.5\) and \(\beta_{sem}=0.3\), with range \((0,1)\):

\[
H_{mix}^m=\operatorname{LN}(H_m+\beta_{phy}V_{phy}^m+\beta_{sem}V_{sem}^m),
\]

\[
Q_m=\operatorname{LN}(H_{mix}^m+0.2(P_0H_{mix}^m)).
\]

The final width-\(d\) embedding is

\[
Z=\operatorname{LN}\left(W_f[Q_T,Q_V,|Q_T-Q_V|,Q_T\odot Q_V]
+W_s[H_T,H_V]\right).
\]

There are three distinct mathematical facts: (1) the source-partitioned sparse operators reconstruct the fixed union operator; (2) each \(U_r\) applies row-quality normalization; and (3) the expert/router/nonlinear integration means the complete model is not linearly reconstructible from the two source branches.

## Variant definitions

| Variant | Change from E2 Full | What the comparison can probe |
| --- | --- | --- |
| E0 `e0_source_agnostic` | Replaces the real split with `PT=PV=0.5*PU`; retains both semantic slots, cross-modal messages, experts, and router. | Whether real source assignment adds value beyond two identical half-union supports. This does **not** represent every source-agnostic architecture. |
| E1 `e1_no_cross_modal` | Uses real PT/PV but sets cross-modal message construction to the exact same-modality path and does not call its parameters. | Whether learned cross-modal messages help in this model. Cross-modal and rho parameters must have no task gradients or AdamW updates. |
| E2 `e2_full` | Real source split, nonlinear cross-modal messages, three shared experts, node-conditioned router. | The full V1 design. |
| E3 `e3_single_expert_control` | Retains graph/message/integration blocks; replaces the bank and router with one ordinary MLP whose bottleneck width is chosen to match the combined expert+router+embedding parameter budget. | Shared expert/router processing versus one wider learned function. Actual count mismatch is recorded; this is not an exact compute match. |
| E4 `e4_static_routing` | Keeps the same bank and router parameters as E2 but routes from pooled relation-view/modality context, shared over valid nodes in that view. | Node-conditioned routing versus relation/modality-view static routing. |

All variants share one implementation. The dimensions and task protocol are frozen by the YAML config, apart from the explicitly defined E3 budget match.

## Relationship to V7B.1 and MvCGE

PACE-MAG reuses the V7B.1 K=8 candidate cache, its feature/candidate/source fingerprints, raw source proposal counts, and its fixed 0.8 degree-scaled semantic edge-weight reference. It replaces V7B.1's learned relative edge scoring, multihop PSCE trunk, calibration, auxiliary routing objective, and deeper expert pathways with a single row-quality neighborhood context and an ordinary Cross-Entropy objective. The physical operator is an unweighted simple-graph symmetric normalization.

The conceptual connection to MvCGE is the shared expert dictionary processing multiple structural contexts. PACE-MAG is a MAG-specific adaptation, not a reproduction of the original MvCGE implementation. It adds source-separated Text/Visual neighborhoods and makes cross-modal content modification happen before semantic aggregation.

## Novelty and overlap risk

The design combines established ingredients: multimodal feature projection, relation-specific message passing, shared mixture-of-experts processing, graph-conditioned routing, and residual late fusion. Similar combinations may already exist in heterogeneous GNN, multimodal graph learning, and graph-MoE work. The specific implementation question is the value of provenance-count allocation across two fixed KNN supports together with target-modality cross-modal messages and one shared router/expert bank. This is a research hypothesis, not a novelty claim. A dedicated literature and claim-level comparison is required before positioning it as a method contribution.

## Explicit non-goals and unsupported use

- No Link Prediction, neighbor-sampled training, or global-ID cache remapping to subgraphs.
- No learned KNN edge weights, graph rewiring, edge scoring, Top-K routing, balance/MMD/orthogonality/contrastive losses, or GW geometric bases.
- No CAME-style dimension-wise router or global pretraining.
- No automatic tuning from the one-seed Pilot and no Test evaluation.
- Engineering correctness and successful optimization are not evidence that the proposed mechanism improves generalization.
