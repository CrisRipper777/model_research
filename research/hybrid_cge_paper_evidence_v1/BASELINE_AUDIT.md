# Baseline implementation audit

The nine requested model families are run through the repository's supervised, transductive full-graph NC interface. All use the same loaded feature rows, global node IDs, physical edge index, Train/Validation/Test indices and metric code. The trainer masks Test labels before `run_nc`; all scores in the comparison are produced on Validation only. The task runner uses AdamW, LR `1e-3`, weight decay `1e-4`, 300-epoch cap, patience 30, minimum epoch 30, minimum accuracy improvement `1e-4`, and gradient clipping at 1.0. Models retain their architecture-specific dimensions and internal regularization.

The existing SAGE, MMGCN, and RoleMAG 45 cells were admitted only after checking source commit, Test-off provenance, shared protocol ID, current feature/physical graph/split fingerprints, initial model and classifier hashes, checkpoint SHA, strict checkpoint loading, and the CSV/JSON Validation metrics. The audit is stored in `data/baseline_reuse_audit.json`; source records remain in the parent branch's CSV and checkpoint paths are unchanged. The split fingerprint is recomputed from the current loader and checked against the inherited run CSV. The inherited metrics metadata did not repeat that split hash, so this part of the audit uses the original CSV row as its source.

## Executable adaptations and limits

- **GCN:** PyTorch Geometric GCNConv stack; uses full-graph normalized adjacency with self-loops and the configured BatchNorm/ReLU/Dropout between layers. This is a transductive full-graph supervised adaptation.
- **SAGE:** PyTorch Geometric SAGEConv stack on the full physical edge index. It is not sampled inductive GraphSAGE training.
- **MMGCN:** separate Text and Visual branches in `[text, visual]` input order, then mean fusion. Its `id_embedding` is a plain Tensor with `requires_grad=True`, not a registered Parameter; it receives no AdamW update and is absent from `state_dict`. The runner records its initialization hash and reconstructs it from the seed for evaluation. Results are for this repository adaptation, not an exact official reproduction.
- **MGAT:** separate Visual and Text graph-attention branches with the repository input split `[text, visual]`; the wrapper routes each view to its own branch and averages the outputs. Its random ID embedding is a registered buffer, fixed during optimization and included in checkpoints. This is the repository's supervised node-classification wrapper.
- **DiP:** full-graph dynamic multimodal subsystem with learned pseudo-nodes (512 Visual and 64 Text under the supplied config), path-integral fusion, and three message-passing steps. The wrapper reorders the repository's `[text, visual]` tensor to the model's `[visual, text]` internal order. It uses plain supervised CE with no paper-specific auxiliary objective in this runner.
- **DGF supervised filtering-core adaptation:** retains per-view normalized projections, symmetric-softmax feature filtering, and the truncated node-domain propagation. The OpenMAG cross-modal NCE, graph contrastive, and k-means/community objectives are not part of this plain-CE experiment.
- **DMGC supervised adaptation:** retains the low/high frequency graph encoders, per-view projections and view fusion. OpenMAG dual-frequency and cross-modal InfoNCE objectives are omitted; the repository runner returns zero model auxiliary loss and trains with the shared supervised CE protocol.
- **LGMRec NC adaptation:** uses the feature encoder, LightGCN stack, mean layer pooling and hypergraph layer. The modality InfoNCE reconstruction losses and corresponding vision/text decoder heads are omitted; no auxiliary loss is included.
- **RoleMAG NC wrapper:** converts `[text, visual]` into the core's Visual/Text argument order, derives finite/nonzero modality masks, combines the graph backbone with the role core, and uses the configured warmup/ramp schedules. Optional routing regularizers are zero under the supplied defaults. It is a project adaptation, not an official full reproduction.

No MLP or MvCGE-MAG result is included. Published scores under different data, feature, split, or training protocols are contextual literature only and must not be mixed into the same-protocol table.

## Results

The final comparison is in `BASELINE_REPORT.md`; it includes all 90 new baseline cells and the 45 inherited rows only while `data/baseline_reuse_audit.json` remains `pass`. Baseline models do not use the Hybrid-CGE paper wrapper, so the later SameNode condition-dispatch correction does not invalidate these rows.
