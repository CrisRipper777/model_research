# ORCI-D0 design audit

## Frozen implementation facts

- This phase freezes the PIGPR-C1 raw direct GPR (RGD) backbone. It does not reopen the GPR attribution audit.
- C1's `H0` is the intrinsic projected semantic state and was strongly supported by the preceding evidence. ORCI-D0 preserves it as the untouched order-zero state.
- Physical-graph propagation remains `H0=P`, `Hk=A_hat H(k-1)` for `k=1..3`, with the same self-loop handling and symmetric degree normalization as C1. There is no anchoring, relation calibration, semantic edge weighting, or learned topology.
- For graphs above 200,000 directed edges, the same weighted `index_add` aggregation is executed in 32,768-edge chunks to bound temporary memory. This changes only the summation execution path, not the graph operator or model/training hyperparameters; a large-edge forward/gradient equivalence test covers it.
- For training graphs with at least 50,000 nodes, activation checkpointing recomputes the existing projector, attention, and fusion activations during backward while preserving dropout RNG state. It changes the compute/memory tradeoff only; forward values, loss, optimizer, and hyperparameters are unchanged.
- For ORCI full-graph NC inputs with at least 50,000 nodes, raw features remain on CPU and pass through the unchanged modality projectors in 8,192-node chunks. The projected priors and graph operator remain on the model device. This storage/transfer path does not change feature values, graph sampling, objective, or optimizer; CUDA checkpoint calls receive a device anchor so projector dropout RNG is preserved during recomputation.
- The large-graph B/A GPR mixture is streamed through its exact analytic backward. Large-graph I/IA recomputes per-order propagation and interaction inside an activation checkpoint and retains only the algebraically equivalent aggregate correction. Equivalence checks compare outputs, auxiliary loss, dropout behavior, and gradients to the explicit formulation.
- The monomial CoSI prior is `[0.15, 0.1275, 0.7225, 0]`; one direct coefficient vector `c=c_prior+delta_c_raw` is shared by text and visual modalities. `delta_c_raw` starts at zero.
- C1 composition is `G=sum_k c_k H_k`, followed by per-modality LayerNorm and the same residual MLP late fusion. ORCI-D0 changes only orders 1 through 3 when interaction is active; order zero is never cross-modal updated.
- NC computes cross entropy plus `task.loss.aux_weight * model.aux_loss`; the repository default is `1.0`. LP uses the same task-level combination for the model's returned auxiliary loss. No task loss protocol is changed.

## Experimental interpretation frozen before implementation

- The 2x2 screen treats B (RGD), I (interaction), A (selective alignment), and IA (joint) as a factorial design. Interaction and alignment may be complementary; I and A are attribution arms and need not each beat B.
- The primary evidence is retrained IA versus B. Paired I-B, A-B, IA-I, IA-A and the descriptive 2x2 contrast help attribution; checkpoint interventions do not replace retrained comparisons.
- Alignment acts only in the shared 64-dimensional content space used by interaction queries and keys. It does not align H0, full Hk, or final embeddings.
- The only permitted calibration is IA on Movies/Grocery seed 42 with weights `{0.02, 0.05, 0.10}`, plus one B reference run per dataset. Selection uses mean validation accuracy delta against B, the predeclared `<0.10` percentage-point smaller-weight rule, then Macro-F1 as tie-break. No test split is evaluated or inspected.
- After calibration the selected alignment weight is frozen for the 36 formal NC runs (3 datasets x 4 variants x 3 seeds). No test evaluation, split changes, Toys/Reddit-S, or formal LP is in scope.

## Task protocol audit

`src/tasks/nc.py` adds the returned `aux_loss` to training cross entropy using the configured `task.loss.aux_weight`. The frozen `configs/task/nc.yaml` sets that value to `1.0`. NC validation selects the checkpoint by validation accuracy, and `development_no_test=true` requires `evaluate_test=false` and limits observed labels to train/validation.

`src/tasks/lp.py` likewise adds `task.loss.aux_weight * aux_loss` to the sampled-batch link loss. Its frozen sampler defaults to `[5,5,5]`; the ORCI model advertises `requires_full_lp_sampler_depth=True`, preserving the three-hop LinkNeighborLoader protocol. The LP smoke explicitly disables test evaluation and does not constitute formal LP.
