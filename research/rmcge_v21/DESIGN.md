# RM-CGE V2.1 Design

## Frozen question

Does receiver-conditioned relative neighbor choice improve the shared RM-CGE V1 graph experts when total incoming physical propagation mass is held fixed, and does giving Intra and Cross messages independent choice distributions help?

## Conditions

- **C0** delegates to the existing RM-CGE V2 V1 forward and parameter tree.
- **C1** adds one shared bounded scorer over target-conditioned Intra messages and adds `(P_select_intra − P) H` to the existing expert path.
- **C2** adds an independently scored Cross-purpose distribution and adds `(P_select_cross − P) Phi_other_to_target(H_other)` while retaining the ordinary Cross Context.

The scorer is shared across target modalities and purposes. It uses receiver query/key interaction features, cosine similarity, endpoint degrees, an 8-dimensional target-modality embedding and an 8-dimensional message-purpose embedding. A zero initialized output layer makes the selector start at the physical operator. Scores are bounded with tanh to [-1,1].

For incoming edges of node i, the implementation computes `p_select[e] = p[e] exp(score[e]) R_i / D_i` with FP32 receiver scatter sums. The physical graph, symmetric degree-normalized coefficients, edge multiplicities and four-hop shared physical expert profiles remain intact. Selection is a residual against the original physical operator.

C1 and C2 have identical added module shapes and initialization. The V2/V1 Cross Projection, shared four-expert bank, original Top-2 node route, fusion and balance loss are reused. C2's selected Cross message enters both expert profiles and the existing Node Router input; no Router head is added.

## Boundaries

Training is full-graph transductive with supervision on Train indices only. Validation selects checkpoints. Test evaluation is disabled. Three seeds are descriptive paired evidence, not a significance test.

## Relation to prior designs

GATv2 motivates explicitly testing whether the receiver query can reverse neighbor rankings; the synthetic test directly exercises this property. GNN-FiLM motivates using the target representation to condition incoming messages, while V2.1 uses the target only in a shared score function and preserves the original message transform. RoleMAG motivates distinguishing neighbor functions in multimodal propagation; V2.1 operationalizes only the frozen Intra/Cross message-purpose split and adds no role labels or role-specific expert banks. MvCGE motivates shared expert parameters with graph-aware routing, which remain inherited from V1. CAME motivates modality-aware expert fusion and cross-modal alignment; the existing V1 Cross Projection and shared expert path are retained without adding a contrastive objective. These are design inspirations, not claims that V2.1 reproduces those methods.

References: [GATv2 (ICLR 2022)](https://openreview.net/forum?id=F72ximsx7C1); [GNN-FiLM](https://arxiv.org/abs/1906.12192); [RoleMAG](https://arxiv.org/abs/2604.12271); [MvCGE (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7315c16422558ae81d65a812723d2cec-Abstract-Conference.html); [CAME (ICML 2026)](https://proceedings.mlr.press/v306/he26at.html).
