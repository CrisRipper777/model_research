# MR-CGE V1.1 implementation specification

## Frozen-lineage review

- The working tree was clean at `6f5eb322f142c332e6ed63df86224ed9115ee47d`, and the historical branch `exp/hybrid_cge_paper_evidence_v1` and its `origin` ref both point to that SHA.
- The requested branch name was absent from local refs and from `origin` when checked. This branch is created directly from the frozen Evidence SHA.
- The V1.1 and Paper V1 model implementations, older research directories, checkpoints, and final-test lock are treated as frozen inputs. This study is implemented in `src/models/mrcge_v11.py` and `research/mrcge_v11/`.
- The V7B.1 cache loader checks feature, candidate and V7B fingerprints. `build_modality_knn_operators()` uses `source_flags` to make separate Text and Visual symmetric KNN operators. The physical operator follows the frozen V7A behavior: remove self loops, compute incoming degree, and use symmetric inverse-square-root degree weights.
- Candidate-graph and source-feature fingerprints are persistent model buffers. Strict checkpoint reload checks them against the attached V7B.1 cache before loading any parameter state.
- NC full-graph forward returns `(z, None, None, aux_loss, info)`. The Trainer composes main CE, Train-only Text/Visual private CE, and `aux_weight * aux_loss`. It already masks Test labels before entering the training loop when the run configuration disables Test evaluation.
- `receiver_sparse_attention()` computes receiver-grouped softmax over source-to-receiver edges and has a custom chunked backward. The existing validation audit patterns preserve RNG state when collecting secondary validation diagnostics.
- The Paper Evidence protocol records the earlier Late Complementary Balance Loss accounting defect. It remains outside this branch's scope; the new model removes the late expert/router path entirely.

## Explicit implementation choices

- The model has three `condition` values (`S`, `F`, `M`) under one class and one common module tree. All common modules are constructed in the same order before condition-specific filter parameters. Reinitializing each condition from the same seed therefore permits a state-dict hash check over the actual shared named tensors.
- Each modality gets its own 256-dimensional projector. Each forward builds physical `B0..B3` and semantic `B0..B1` once per modality. The two propagation bases are reused for every expert; no trainable projector output is cached across forwards.
- A support mask is based only on real incoming non-self edges. It masks propagated orders and every complete graph response. The raw projected modality state is retained as the private residual for unsupported nodes.
- M's filter bank is a single `ModuleList` of four experts, referenced by every modality/relation call. Its shared signed `theta[4,4]` uses relation-specific order truncation and L1 normalization. S uses fixed equal coefficients over valid orders; F uses one shared learnable signed coefficient vector. S and F each use one shared bottleneck transform over all views.
- The single M router sees `[H, B1, B1-H, modality embedding, relation embedding, calibrated early context]`. Private and semantic calls receive zero early context. Top-2 applies a second softmax over selected logits; balance is `K * sum(mean_dense_probability * selected_fraction)` on active nodes. Balance is averaged across the six actual M router calls: two modalities × (private physical, semantic, collaborative physical). S/F have no router and return zero balance.
- Early Cross uses directional bias-free `Phi` projections, the physical normalized operator, and a detached RMS ratio targeting the receiver's private physical first-order response. The learned bounded beta starts at 0.25. Recomputed collaborative outputs are explicitly replaced by private outputs at nodes with exactly zero calibrated context, preventing training Dropout from fabricating a Cross increment.
- Late completion uses V7B.1 source-modality edges, receiver sparse attention, the three-way Semantic/Same-node/None selector, conditional candidate normalization, a channel gate, bounded lambda, relative residual cap, and node-wise non-inverted Dropout 0.15. Same-node support is controlled by `same_node_enabled`; it is not inferred from feature values.
- `aux_loss` contains `0.01 * balance` exactly once; config `aux_weight` is 1.0 and `unimodal_weight` is 0.3. S/F have exactly zero auxiliary loss.
- CUDA 0 was observed occupied before this task. The user explicitly authorized sharing CUDA 0. Any GPU execution must log the baseline process list and recheck free memory/utilization and the process set at each isolated run launch; no foreign process is stopped or modified. CUDA 1 is not selected.

## Training contract

Only Movies, Grocery and ele-fashion, conditions S/F/M, and seed 42 are in Stage A. Each cell runs in its own process. The run configuration fixes AdamW, LR `1e-3`, WD `1e-4`, Dropout `0.2`, up to 300 epochs, patience 30, minimum epoch 30, gradient clip 1.0, every-epoch validation, and Best Validation Accuracy primary selection with a separate Best Validation Macro-F1 checkpoint. Test evaluation is false and Test labels are masked to `-100` before Trainer entry. No seed 43/44 or Test operation is allowed in this phase.
