# V6A MACE-MAG design review

## Research question

Text and Visual use the same physical graph, while their useful structural responses may differ. V6A tests whether four shared structural experts, selected independently by each modality, can exchange a small amount of node-local evidence through the experts selected by the other modality. The target modality keeps its own intrinsic representation and decides how to weight any received correction.

The experiment is limited to expert-level collaboration. It does not add a semantic graph, learned topology, private expert pools, cross-modal contrastive alignment, optimal transport, or new spectral filters.

## Computation graph

For each modality \(m\in\{T,V\}\), an independent V2-series intrinsic projector produces \(H_0^m=\phi_m(X^m)\). The modalities are never averaged. The shared physical edge list is stripped of self-loops and normalized with the V2.2b destination-degree symmetric rule. Four raw propagation states are formed as \(S_0^m=H_0^m\), \(S_k^m=P S_{k-1}^m\). Each propagated state is zeroed outside the existing active mask and RMS-normalized over active nodes, independently per feature.

Four learned Hadamard-initialized coefficient rows produce modality-specific profiles

\[
B_{i,r}^m=\sum_{k=1}^{4}\alpha_{r,k}\widehat S_{k,i}^m,
\qquad E_{i,r}^m=B_{i,r}^m+0.1F_r(B_{i,r}^m),
\]

where each bias-free bottleneck transform \(F_r\) is shared between Text and Visual. A modality-specific router sees its projected intrinsic state, one-hop router state, active-node global context, a zero evidence block for checkpoint-compatible V2.2b input shape, and a modality embedding. A0 uses the static input `[0, 0, global, 0, modality]`; A1/A2/A3 use the node input `[ego, local, global, 0, modality]`. Its dense probabilities feed balancing; its per-node Top-2 logits produce two unique indices and normalized weights. Structural strengths are independent modality scalars, not outputs of the router.

For each target modality, selected expert responses are combined into its ordinary structural mixture \(C_i^m\). A2/A3 also gather only the target modality's two selected expert queries and the selected source modality's two expert tokens. The attention uses a 64-dimensional content projection, learned target/source expert-ID bias, log source route-weight bias, and a learned Null logit. A strict zero Null Value is appended; Value and output projections have no bias. The correction for an inactive target node is explicitly zero. The two selected target experts' corrections are combined using the target router's own Top-2 weights:

\[
\Delta_i^m=\sum_{r\in\mathrm{Top2}(\pi_i^m)}\pi_{i,r}^mJ_{i,r}^{m\leftarrow\bar m}.
\]

The modality update is \(Y_i^m=H_{0,i}^m+\tau_m C_i^m+\lambda_m\Delta_i^m\) in A2/A3 and \(Y_i^m=H_{0,i}^m+\tau_m C_i^m\) in A0/A1. The independent initial strengths are approximately 0.25 and 0.15. The two modality outputs use the existing V2.2b concatenation residual MLP and LayerNorm.

## Variants

| Variant | Router | Attention source | Cross-modal correction |
|---|---|---|---|
| A0 `a0_static` | Modality-static Top-2 | None | Off |
| A1 `a1_node` | Node-conditional Top-2 | None | Off |
| A2 `a2_cross` | Same as A1 | Other modality's selected Top-2 | On; primary model |
| A3 `a3_intra` | Same as A1 | Same modality's selected Top-2 | On; capacity control |

A2 and A3 use the same attention module shapes, construction order, parameter initialization, and training budget. They differ only in which modality supplies the two source tokens. A0/A1 have the same attention modules constructed under an isolated RNG stream but frozen, so their attention parameters do not count as trainable and do not perturb the immediately initialized NC classifier. Every variant constructs the shared trunk identically.

An explicit zero collaboration scale bypasses the collaboration contribution, allowing A2/A3 to use the exact A1 non-collaboration output path under matched trunk weights. A0/A1 and A0/A1's router math are designed to permit a same-weight small-graph regression against V2.2b D0/D1. V2.2b's inactive structural-evidence modules are not needed by D0/D1; a regression test will map the common weights and report any remaining arithmetic difference.

## Relation to prior work in this repository

- **MvCGE-MAG V2/V2.2b:** reuse the independent intrinsic projectors, four Raw propagation orders, active-mask RMS normalization, shared Hadamard-initialized experts, Top-2 routing, per-modality load-balancing surrogate, decoupled modality strength, and late fusion. A0/A1 map to V2.2b D0/D1. V6A does not carry V2.2b's structural reliability/evidence router variants into the frozen screen.
- **MvCGE-MAG V4A:** retain its principle of leaving the Raw trajectory intact and adding a separately attributed structural correction. V6A's new correction is selected-expert collaboration, not its role-partition residual.
- **ORCI-D0:** ORCI attends between per-hop structural-state tokens for every node, with its interaction/alignment arms and order-wise correction. V6A attends only among the two already selected experts per modality at each node, with no interaction between different nodes or propagation orders, and no alignment loss. The target's own route weights gate the correction; a Null token can reject it. This makes the hypothesis expert-level, conditional collaboration rather than order-level cross-view interaction.

## Objective and protocol

Use the existing `unified_full_graph_nc_v1` trainer, fixed dataset splits, train-node labels only for cross-entropy, Validation Accuracy for checkpoint selection, and Validation Macro-F1 as a reported metric. The campaign contains Movies, Grocery, and ele-fashion; seeds 42, 43, and 44; all four variants; 36 independent runs. Every command sets `task.evaluate_test=false`. No formal LP, tuning, sampling, split changes, or Test metrics are in scope.

## Risks and resource plan

Full-graph feature and expert activations may dominate memory, especially on the largest graph. Initial resource inspection found GPU0 occupied by an unrelated VLLM process (about 23.1 GiB) and GPU1 free; no process will be stopped. Use GPU1 unless its availability changes. First try normal full-graph execution. If memory pressure requires optimization, use bounded edge-message chunks and activation checkpointing that preserve the exact graph operator and dropout RNG; compare outputs and gradients against an unchunked small-graph path. CPU feature staging is not enabled by a model-only flag: it will only be introduced if needed with explicit trainer/inference routing and equivalence tests. Do not reduce the frozen 256-dimensional width, order, Top-2, batch/full-graph protocol, epochs, or variant-specific budgets to work around OOM.

Detailed diagnostics are opt-in (`return_details=True`) so ordinary training does not retain full attention maps or all intermediate tensors. The campaign runner must bind runs to the implementation freeze SHA, check existing metrics/checkpoints on resume, and reject any Test metric key.
