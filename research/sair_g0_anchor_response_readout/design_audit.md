# SAIR-G0 design audit

## Provenance and evidence entering this screen

This experiment starts from the F0 final commit `997b89b8b0e59ca2da9654d07cc184cc1ce14169` on `exp/imosi_f0_interaction_mode_mixture`, with no other experiment branch merged or cherry-picked. The implementation is a readout-only screen over the existing raw direct GPR/RGD backbone.

Prior screens motivate a narrow comparison rather than another adaptive mechanism. C1 supports retaining intrinsic order zero (`H0 = P`) in the RGD backbone; its order-zero ablation reduced accuracy across the evaluated RGD/AGD datasets, while the direct GPR RGD result was positive or tied against RFD. C1 did not establish a stable additional AGD basis gain. E0.1 found paired residual effects mixed across datasets, and the target-only control was not consistently positive. F0 found the Preserve/Self/Paired mixture gains small and inconsistent, with Macro-F1 relative to G negative across the reported datasets. These results favor preserving RGD and asking whether exposing its already-computed components at readout adds value.

## Frozen backbone and decomposition

For each modality, the RGD core remains unchanged:

\[
P = \mathrm{projector}(X),\qquad H_0=P,\qquad H_k=\hat A^kP,
\]
\[
Q=\sum_{k=0}^{3}c_kH_k=c_0P+R,\qquad G=\mathrm{LayerNorm}(Q),
\]
where
\[
R=Q-c_0P=\sum_{k=1}^{3}c_kH_k.
\]

Projectors, `delta_c_raw`, fusion layers and normalization, `c_prior`, raw normalized propagation, and `StreamingRawGPR` retain the PCRR-v1 module creation order and behavior. The strong main readout `G` remains present for B, C, and D; only an optional zero-initialized residual adapter changes the final modality representation. The response will be computed by subtraction from the existing proposal `Q`; SAIR will not retain an `H1:H3` state bank merely to form `R`.

## Interpretation of P and R

`P` is the intrinsic modality semantic anchor/reference supplied by that modality's projector. It is not ground-truth semantics. `R` is the accumulated structure-induced response after removing the order-zero contribution from the GPR proposal. It is not assumed to be pure low-frequency content or smoothing. Depending on graph, coefficients, and features, it may include supportive structural evidence, multi-hop context, contrastive effects, and high-frequency information. This screen does not separate those possible components.

The hypothesis is that preserving the provenance of `P` and `R` until the task readout may let a downstream readout use structure–semantics relationships that were compressed in `G = LN(c0 P + R)`. This is a hypothesis about readout utility, not a claim that the response has a specific frequency or homophily interpretation.

## Matched variants

- **B / base:** bypasses the pair adapter and returns `G`.
- **C / generic:** forms `[G, G, 0, G*G]` and applies the shared PCRR-layout pair adapter. It sees only the already-mixed representation.
- **D / decoupled:** forms `[P, R, abs(P-R), P*R]` and applies the same adapter. It uses only the current modality's `P` and `R`; it cannot read the other modality.

All variants keep the same module layout and parameter count. `pair_norm`, `pair_down` (`4*256 -> 64`), and `pair_up` (`64 -> 256`) are shared across Text and Visual. `pair_up` is exactly zero-initialized, so all three variants start at the RGD function. `R` remains raw; there is no separate normalization before `pair_norm`. Readout is chunked at 8192 nodes, with the existing non-reentrant checkpoint path used for large-graph training where applicable.

## Excluded mechanisms

This screen does not add topology reconstruction, an explicit low/high-frequency filter, a router, MoE, cross-modal interaction, alignment, optimal transport, or an auxiliary objective. It also does not add modality-specific adapter heads, gates, node trust, attention, residual scales, separate P/R heads, PC-Conv, or another normalization after the adapter residual. There are no checkpoint interventions; B/C/D are retrained controls.

## Decision discipline

Primary paired comparisons are D−B and D−C over dataset-by-seed pairs; C−B is the generic-adapter reference. Results will include paired Accuracy deltas in percentage points with population SD and positive/negative/tie counts, plus Macro-F1 and cross-entropy deltas. With three training seeds, no pseudo-IID p-values or fixed success threshold will be used. Geometry and adapter diagnostics describe representations and corrections but do not establish performance value. Any conclusion about provenance will require retrained performance evidence, and no result from this screen alone will be described as evidence that `R` contains heterophilous information.

The campaign is validation-only: Movies, Grocery, and ele-fashion; variants B/C/D; seeds 42/43/44; fixed existing splits; `evaluate_test=false` and `development_no_test=true`. NC test, Toys, Reddit-S, formal LP, and further model development are outside this phase. The only LP run is the requested protocol smoke check.
