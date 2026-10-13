# RM-CGE V2.1 Design

## Frozen research question

Does receiver-conditioned relative neighbor choice improve the shared RM-CGE V1 graph experts when each receiver keeps the same total incoming physical propagation mass? Does giving Intra and Cross messages independent neighbor distributions add value?

## Conditions

- **C0** strictly delegates to the existing RM-CGE V2 `V1` forward and parameter tree.
- **C1** adds a shared selector for same-modality incoming messages and the residual `Delta_intra_m = (Pselect_intra_m - P) @ H_m`.
- **C2** adds a separately normalized Cross-purpose selector and `Delta_cross_m = (Pselect_cross_m - P) @ Phi_other_to_m(H_other)`. It retains the ordinary `C_base_m = P @ Phi_other_to_m(H_other)` context.

C1/C2 use the same added parameter structure and initialization. Both retain V2/V1 projectors, four-hop physical profiles, shared four-expert bank, Cross Projection, KNN context, node correction, Top-2, fusion and balance loss. C2's Cross Delta enters the shared experts and the existing Node Router input. No new expert bank, Router head, role labels or auxiliary objective is introduced.

## Shared edge scorer

For each directed physical message `j → i`, the shared scorer receives a 32-dimensional query from the receiver, a 32-dimensional key from the sent message, their product, absolute difference, receiver/sender cosine, log physical degrees, an 8-dimensional target-modality embedding and an 8-dimensional purpose embedding. The input width is 147; the MLP is 147→64→1. Its final layer starts at zero and `tanh` bounds scores to [-1,1]. The receiver/sender interaction allows the same candidate set to be ranked differently for different target nodes.

For physical coefficient `p_e`, receiver mass `R_i = Σ p_e`, and denominator `D_i = Σ p_e exp(score_e)`, the selector uses `p_select_e = p_e exp(score_e) R_i / D_i`. R and D use FP32 receiver scatter sums. This preserves each receiver's physical row mass. Equal scores and one-incoming-edge rows recover the original physical coefficients. Isolates remain zero. The original symmetric degree-normalized operator and edge multiplicities are untouched; selection is a residual against P.

## Implementation boundaries

Cross Projection runs on sender nodes before physical aggregation. Edge features are scored in chunks, with differentiable scatter normalization and chunked SpMM. The implementation never forms a dense adjacency or an E×hidden tensor. Zero-initialized scores make the new residuals start at zero. New module initialization is isolated with `torch.random.fork_rng`, preserving the paired classifier state.

## Relation to prior designs

GATv2 motivates testing whether the receiver query can reverse neighbor rankings; the synthetic test directly checks this expressivity. GNN-FiLM motivates using the target representation to condition incoming messages, while V2.1 uses the target in a shared score function and keeps the payload transform unchanged. RoleMAG motivates distinguishing neighbor functions in multimodal propagation; V2.1 tests only the frozen Intra/Cross purpose split and adds no role labels or role-specific experts. MvCGE motivates shared expert parameters with graph-aware routing, which remain inherited from V1. CAME motivates modality-aware experts and cross-modal alignment; V2.1 retains existing Cross Projection and shared expert processing without adding a contrastive loss. These are design inspirations, not claims that V2.1 reproduces those methods.

References: [GATv2 (ICLR 2022)](https://openreview.net/forum?id=F72ximsx7C1); [GNN-FiLM](https://arxiv.org/abs/1906.12192); [RoleMAG](https://arxiv.org/abs/2604.12271); [MvCGE (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7315c16422558ae81d65a812723d2cec-Abstract-Conference.html); [CAME (ICML 2026)](https://proceedings.mlr.press/v306/he26at.html).
