# MAG Information Utility V1: experiment design

## Question and evidence boundary

This is a diagnostic study, not a new paper model. It asks how much validation
signal a small, fixed-capacity encoder can extract from intrinsic Text/Visual
features, physical-neighborhood propagation, explicit cross-modal physical
contexts, and modality-specific KNN supports. The five treatments are trained
end to end on the same official node-classification splits. Test evaluation is
disabled in the study configuration and runner.

The primary unit for comparisons is one paired Dataset × Seed block. Three
training seeds are repeated initializations on the same graph, not three
independent datasets; all seed summaries are descriptive and no significance
claim is made.

## Fixed probe

Text and Visual have separate residual projectors with the same module
definition and 128-dimensional output. Each branch receives six fixed context
slots in a fixed order: same-modality physical first and second orders,
cross-modality physical first and second orders, then same- and cross-modality
semantic first order. The Text and Visual processors are separate but have
identical shapes and capacity. Each processor is bias-free before its residual
normalization, so an all-zero context tensor produces an exactly zero context
update. The final fusion receives both intrinsic projector outputs and both
processed branch outputs; a linear classifier is trained separately.

Physical propagation uses a loop-free, deduplicated, symmetrized,
row-mean-normalized sparse operator `P_phy`. It computes `P_phy @ H` and then
`P_phy @ (P_phy @ H)` with sparse matrix multiplication; no dense adjacency or
materialized `P_phy^2` is used. Text and Visual semantic operators are selected
independently from V7B K=8 provenance flags and are independently row
normalized. An edge proposed by both modality searches belongs to both source
supports.

| ID | Enabled context slots |
|---|---|
| I0 | none; intrinsic Text/Visual states only |
| I1 | each branch's same-modality physical orders 1 and 2 |
| I2 | I1 plus the other modality's physical orders 1 and 2 in each branch |
| I3 | I2 plus the branch's own semantic source's same- and cross-modal messages |
| I4 | same as I3, with separately randomized Text and Visual semantic supports |

The `I2−I1` contrast measures the inductive-bias increment from explicit
cross-modal processing inside propagation; it is not newly observed raw
information. `I3−I2` also opens effective input channels. `I3−I4` is the most
direct semantic-topology-content contrast in this probe, with one fixed random
topology sample per dataset and source view.

All treatments instantiate identical parameters, tensor shapes, projectors,
fusion, and classifier. Treatment masks set inactive context slots to zero.
Thus total parameter counts are equal, but effective input-conditioned
capacity differs as slots are opened; this is not a perfectly isolated causal
test of information utility. Disabled input slices can have zero
input-dependent gradients even though their weights remain in the model.

## Data and graph controls

Movies and Grocery use their existing MAGB NC split files for seeds 42, 43,
and 44. ele-fashion uses its supplied MM-Graph split. Existing V7B semantic
caches must validate against current feature arrays and the physical graph
fingerprint before use. KNN construction and random rewiring read features and
physical topology only, never labels or split outcomes.

The random semantic nulls are generated independently from the Text and Visual
source supports by valid undirected double-edge swaps. Each source-specific
null must keep its node set, undirected edge count, and every node degree
exactly, with no loops or duplicate pairs. Topology seeds depend on dataset
and modality only, not training seed. The older V7B null randomizes the union
support; it is not reused as a substitute for these two separate null graphs.

Train–Train homophily and class-transition tables are a separate descriptive
analysis using only labels indexed by each seed's training indices. They do not
alter graph construction, weights, masks, groups, or model input. The random
agreement reference is the sum of squared Train class proportions. Structural
homophily is not treated as evidence of predictive gain.

## Training and model selection

Protocol: `unified_full_graph_nc_v1`, full graph, AdamW, learning rate
`1e-3`, weight decay `1e-4`, dropout `0.20`, 300 epoch maximum, patience 30,
minimum epoch 30, minimum accuracy delta `1e-4`, gradient clipping `1.0`,
validation every epoch, no scheduler, and `evaluate_test=false`. The primary
checkpoint is the best Validation Accuracy checkpoint. A separate Best
Macro-F1 checkpoint records the highest Validation Macro-F1 reached through
the Accuracy-driven stopping horizon. Macro-F1 always uses all configured
classes, not a class set inferred from Test labels.

Within each Dataset × Seed block, every condition starts from the same
byte-identical model and classifier initialization. The runner verifies their
initial-state hashes before optimization. Train and Validation scorecards use
the same post-update, evaluation-mode embeddings; those extra measurements run
without dropout and assert that CPU/CUDA RNG states are unchanged.

## Analysis plan

Primary comparisons use Best-Accuracy checkpoints and paired seed-level
differences for Validation Accuracy, Macro-F1, cross-entropy, Train–Validation
accuracy gap, wall time, and peak CUDA allocation. Dataset means and sample SD
(`ddof=1`) accompany the three individual paired differences. Best-F1 results
and Best-F1 minus Best-Accuracy trade-offs are reported separately.

Node groups use pre-label graph/feature attributes: physical/Text-KNN/Visual-KNN
degree quantiles and self-to-neighbor-mean cosine consistency. Boundaries are
computed without labels. Validation labels are only joined at final metric
calculation. Per-run validation predictions and labels stay in ignored local
outputs; only grouped aggregates are committed.

Historical V7B.1 R0 Raw is an external capability reference, not a
capacity-matched ablation. Differences between this simple 128-dimensional
probe and R0 cannot establish whether graph experts or semantic KNN are
unnecessary in stronger encoders.

## Optional frozen-representation probe

The optional linear probe over frozen I0 representations is not included.
This study prioritizes the specified five-treatment end-to-end matrix and
does not add a second classifier-training protocol. Its absence limits the
evidence to end-to-end utility under the fixed probe.

## Frozen literature context

- SIGN motivates fixed multi-order propagation channels that can be computed
  without dense powers of the adjacency; this probe uses only orders 0–2 and
  does not implement SIGN's full architecture ([SIGN](https://arxiv.org/abs/2004.11198)).
- RoleMAG motivates asking whether neighbors should be processed differently
  before propagation and across modalities; this study tests only pooled,
  fixed cross-modal context slots, not edge roles or routing
  ([RoleMAG](https://arxiv.org/abs/2604.12271)).
- Node-MoE motivates node-dependent graph-filter selection; no experts or
  node-wise filter routing are implemented here
  ([Node-MoE](https://arxiv.org/abs/2406.03464)).
- MvCGE motivates collaborative experts across views; this probe instead
  establishes a simple relation-utility reference without MoE
  ([MvCGE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7315c16422558ae81d65a812723d2cec-Abstract-Conference.html)).
- LATTICE learns modality-aware latent item graphs from multimodal features;
  FREEDOM argues a frozen item-item graph can be useful while separately
  denoising interactions. Both are multimodal-recommendation contexts, not
  direct evidence for KNN utility in this node-classification setting
  ([LATTICE](https://arxiv.org/abs/2104.09036),
  [FREEDOM](https://arxiv.org/abs/2211.06924)).
