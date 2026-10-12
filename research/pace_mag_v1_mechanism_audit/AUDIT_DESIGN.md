# Audit design and measurement definitions

## Frozen source and reproduction gate

For each dataset, use `outputs/pace_mag_v1_pilot/runs/{dataset}/seed_42/e2_full/best.pt` and its exact `resolved_config.yaml`. Before inference, verify checkpoint SHA-256, saved config SHA-256 against the original Pilot row, E2/seed/protocol/selection metadata, shape metadata, validation metrics, and V7B feature/candidate/source fingerprints. Load the same physical graph, features, split, and semantic cache.

Run the original V1 `Model.forward` and the audit-only forward in `eval()` under `torch.inference_mode()`. The audit forward reuses the same projector, cross-modal, expert, router, normalization, and fusion modules and the same sparse operators. Require embedding and logit maximum absolute differences and Validation Accuracy/Macro-F1 differences within `1e-5`. If parity fails, no interpretive intervention is accepted.

`src/tasks/nc.py` builds the validation Macro-F1 class set from train and validation labels when `evaluate_test=False`. The audit reproduces that rule. `load_mag_data` materializes all labels in host memory, so this limitation is disclosed; only train/validation label indices are indexed for metric computation. Test indices are not queried by the audit, and Test label values and metrics are not read.

## Source operators and context

With the cached V7B.1 union, `PT + PV = PU` and `qT + qV = qU`. For each target-modality projection `H`, measure the two linear identities before any expert or router:

\[
q_T U_T + q_V U_V = P_U H,
\qquad
\frac{q_T U_T+q_V U_V}{q_T+q_V} = \frac{P_UH}{q_U}, \quad q_U>0.
\]

Node coverage is `q > 0`. Raw row-quality distributions include zero rows. Distinct neighbor count counts only positive sparse weights, excluding the explicit zero entries retained by source-split COO assembly. Weighted effective neighbor count is the participation ratio `q_i² / sum_j P_ij²`, with zero for empty rows. Source categories use presence (`flags[:,s] > 0`): Text-only, Visual-only, and Both-present. Raw proposal count 2 is a per-modality proposal multiplicity; it does not itself mean that both modalities proposed the edge.

Support overlap uses directed receiver-neighbor pairs (both directions of each undirected edge). Global overlap is set Jaccard. Per-node overlap is receiver-wise Jaccard, excluding nodes with no neighbor in either support and assigning zero to one-sided support. This is descriptive topology overlap, not a causal estimate.

For target modality `m`, compare `U_text^m` with `U_visual^m` both with trained cross-modal injection on and with the same E2 checkpoint's injection locally off. Pair eligibility requires `qT > 0` and `qV > 0`. Cosine is further masked if either context vector has zero norm. Normalized node L2 is `||a-b||₂ / max(||a||₂, ||b||₂, eps)`. “Large” means greater than 0.1; “small” means at most 0.01. RMS and distribution quantiles are also recorded.

## Cross-modal strength

For each semantic source (`text`, `visual`) and target modality (`text`, `visual`), record the trained scalar `rho`, its bounded-parameter derivative, and RMS of target `H`, nonlinear `Phi(H_other)`, injection `rho*Phi`, and sent message. The global injection ratio is `RMS(rho*Phi)/RMS(H)`. Nodewise change is `||A-H||₂ / max(||H||₂, eps)`, summarized on nodes valid for that semantic source.

Run cross-off by passing `H_m` instead of `H_m + rho*Phi` within the local audit forward. No parameters are changed. For each source/target pair, compare the on/off aggregated contexts; report absolute RMS delta, relative RMS delta against the off-context RMS, nodewise relative-change quantiles, and the ratio of aggregate-relative-change to send-relative-change. A ratio below 0.5 is labeled as at least 50% attenuation between send and context. This ratio is a scale comparison, not an energy-conservation law.

## Router and expert measurements

For each of three relation views and two target modalities, summarize the original nodewise expert probabilities: mean, population standard deviation per expert, entropy in nats, and top-1 node share. Same-node JS divergence and L1 differences compare Text-KNN with Visual-KNN routes where both views are valid, compare physical with each semantic view, and compare target modalities within each view. View-average similarity alone is not used to dismiss node-conditioned behavior.

For each view/modality cell, record each expert's output RMS, pairwise cosine, and the actual router-weighted response RMS relative to sent-message RMS. Cosine and RMS diagnose output geometry and scale only; they do not establish task-functional complementarity.

Frozen route interventions are:

- **R0 Original:** trained node-specific router output.
- **R1 Uniform:** `[1/3, 1/3, 1/3]` at each valid node.
- **R2 View-mean:** each view/modality cell's original mean route vector repeated at its valid nodes.
- **R3 Node-permuted:** original route vectors permuted among valid nodes within the same view and target modality. No invalid rows or other views/modalities are mixed. The fixed seed is `20261012`.

## Branch interventions

Each intervention changes one addition only, with all trained weights fixed:

- Semantic response off: set `beta_sem * V_sem` to zero in `H + beta_phy*V_phy + beta_sem*V_sem`.
- Physical response off: set `beta_phy * V_phy` to zero in that same sum.
- Intrinsic skip off: set the final `W_skip[H_T,H_V]` addition to zero before the final LayerNorm.

For every route and branch intervention, record embedding/logit RMS deltas, validation CE, Accuracy, Macro-F1, and validation prediction-flip fraction against R0. Branch input and scaled/projected RMS are recorded separately because LayerNorm means branch amplitude alone does not predict logit impact.

After each intervention, immediately rerun the original V1 forward and classifier. Require restoration within `1e-5`. Hash persistent model/classifier state before and after the entire audit, and verify both modules remain in eval mode.

## Interpretation limits

The 0.8 fixed semantic weight cancels from a single view's row mean when it scales both numerator and row quality. It still changes absolute `q`, which enters the router through `log1p(q)`. The common 0.8 factor cancels from the `qT/qV` proportions in semantic response fusion, while source-specific `q` values still determine those proportions. Thus the constant can affect the complete model through routing and is not globally irrelevant. `PT+PV=PU` is an operator-level identity; row-quality division is a separate normalization; modality-specific messages and nonlinear experts/router prevent a general end-to-end linear equivalence. Frozen interventions describe checkpoint dependence and are not retrained ablations, and no single-seed result is a significance claim.
