# SPGPR B0–B1 design audit

## Provenance

This experiment starts from the reviewed A0.3 commit `e8470fdd529f4e71100c32f25d75a400095fc6de` on `exp/care_a03_adaptation_placement_audit`. That commit contains the validated `development_no_test` NC path, the current full-graph NC protocol, and sampled LP positive-supervision-edge removal. The new branch is `exp/spgpr_b01_filter_decomposition_screen`, created directly at that SHA. The initial worktree was clean. A live `git fetch origin` was attempted before branching but could not resolve `github.com`; the local HEAD and cached `origin/exp/care_a03_adaptation_placement_audit` both equal the required SHA. No other experiment branch is used as a source.

## Current task interfaces and data split

The project factory constructs `src.models.<cfg.model.name>.Model(cfg, data_info)`. NC and LP call `forward(x, edge_index)` and consume the standard five-item result `(z, None, None, aux_loss, aux_info)`. The NC path trains one full-graph forward per epoch and selects checkpoints by validation Accuracy. Validation inference uses the complete fixed graph. The inference helper supports either full forward or a model-provided `inference(x, edge_index, device, batch_size)` method.

LP graph encoders train with `LinkNeighborLoader`. Each sampled batch carries its own `x`, `edge_index`, global node IDs, and edge IDs. With the frozen `global_eid` positive-mask backend, positive supervision edges are removed from that batch's message graph before the model forward; validation embedding inference uses the complete training message graph. The model therefore consumes the current batch `edge_index`, without caching or rebuilding another graph. `requires_full_lp_sampler_depth=True` makes the project resolve sampler depth from `model.num_layers`; with this model's fixed `K=3` and the LP config `[5,5,5]`, the resulting fanouts remain `[5,5,5]`.

The dataset's existing seed-42 NC split file is held fixed across every process by passing its explicit configured path as `dataset.nc_split_path`; the top-level `seed=42,43,44` then controls only model/training randomness. Thus the three values are paired model/training seeds on one split, not three independent splits. The existing loader reads the split indices so the no-test protocol can mask test labels, but training, checkpoint selection, diagnostics, and analysis do not evaluate or consume NC test labels/metrics. `development_no_test=true` and `evaluate_test=false` are set on every NC invocation; no LP test split or formal LP evaluation is part of this campaign.

## Model boundary

`spgpr_mag_v0` is an independently implemented, prior-retaining GPR-style polynomial filter. It reuses only the established project interface and the audited projector/fusion pattern. It does not import or call `care_mag_v1`. Text and visual inputs are split in the declared `[text, visual]` order and projected separately before any fusion. Both modalities propagate over the same fixed physical graph with the CARE-compatible GCN symmetric normalization and self-loop completion. Propagation has no learned message transform. The model uses exactly three residual hops, `R_k=H_k-P`, and global modality hop filters.

The model does not include node/edge-conditioned routing, relation calibration, rewiring, roles, OT, MoE, cross-modal alignment, or auxiliary objectives. Its response is `Z=LayerNorm(P+lambda*sum_k gamma_k R_k)`, so its spectral expression is `1+lambda*sum_k gamma_k(xi^k-1)`. This is an explicitly prior-retaining GPR-style polynomial, not a claim of line-by-line reproduction of the original GPR-GNN.

## Matched variants and initialization

All five filter modes construct all six filter parameter tensors in the same order: positive shared logits, signed shared raw vector, independent text/visual raw vectors, and shared/difference raw vectors. Thus a fixed seed gives identical state-dict keys, tensor shapes, parameter count, and bitwise-matched same-name initial values across modes. Forward selection alone chooses which filter tensors participate. At initialization U/P/S/I/SP all produce `[1/3,1/3,1/3]` for both modalities. The two lambda parameters are global scalars initialized to sigmoid value 0.5. `aux_loss` is exactly zero.

For SP, reported effective filters are computed from normalized modality filters: `gamma_shared_effective=(gamma_text+gamma_visual)/2` and `delta_effective=(gamma_text-gamma_visual)/2`. Raw shared/difference parameters are not interpreted as the final effective decomposition. I and SP each expose six effective raw degrees of freedom; SP is a parameterization/inductive-bias comparison, not a larger-capacity variant. No success criterion requires SP to exceed I.

## Frozen execution plan

The sequence is model/tests and repository tests, six independent-process S repeatability runs (Movies and Grocery, seed 42), NC U/P/S/I/SP smoke runs (Movies, seed 42, two epochs), LP S/SP execution smoke (sports-copurchase, one run, two epochs, two train batches), then the 45-run validation-only NC campaign on Movies/Grocery/ele-fashion × U/P/S/I/SP × seeds 42/43/44. Optimizer and project NC hyperparameters remain unchanged. No midpoint result may change the model or settings. The LP run checks sampler/masking compatibility only and is not interpreted as an LP result.

Comparisons are paired within dataset and model seed: P−U, S−P, I−S, SP−I, and SP−S. Accuracy is primary; Macro-F1 and CE are secondary signals. Paired means, population SDs, sign counts, and repeatability-scale context are reported without pseudo-IID p-values. Validation-selected filters, descriptive polynomial responses at the requested five xi values, and mean/swap checkpoint interventions are reported separately from retrained architecture comparisons. The interventions measure reliance within selected checkpoints and are not substitutes for retrained gains.

The only claimed graph-frequency information will be the polynomial response at sampled xi values. No eigendecomposition or empirical graph-spectrum energy weighting is performed, so those responses are descriptive and will not be described as measured spectral mass or causal mechanism.

## Explicit stop boundary

This work ends after the B0–B1 analysis and self-audit. It does not proceed to B2 selective alignment, OT, Sinkhorn, MMD, MoE, semantic fusion, node/edge routers, Toys, Reddit-S, NC test, or formal LP. Results are pushed only to the named experiment branch after the run artifacts are complete.
