# ORCI-D0 — Order-Resolved Cross-modal Interaction + Selective Alignment

## Status and protocol

- Base commit: `cd9d440aa43454215e1a512b5426a5372e47a41a`; branch: `exp/orci_d0_interaction_alignment_synergy`.
- Formal campaign: 36 validation-only NC runs (Movies/Grocery/ele-fashion × B/I/A/IA × seeds 42/43/44), one fixed split per dataset.
- Calibration: 8 validation-only runs (six IA candidates plus two B references), with selected alignment weight `0.02`. Rule: selected highest mean validation accuracy delta.
- `task.evaluate_test=false`; NC used `task.development_no_test=true`. No test metrics appear in the run files. Calibration is excluded from formal tables.
- Parameterization, fixed CoSI prior, physical graph operator, C1 RGD baseline, and auxiliary-loss task weight are audited in [design_audit.md](design_audit.md).
- Full repository pytest: **133 passed, 2 skipped**; four warnings (upstream PyG deprecation and PyTorch checkpoint CPU AMP deprecation). The two CUDA staging tests also passed separately on GPU1.
- Two early ele-fashion cells hit CUDA OOM while unrelated workloads occupied the available GPUs. They completed on physical GPU1 after enabling CPU-resident feature staging and memory-bounded equivalent propagation/interaction paths; model, split, loss, optimizer and early-stopping protocol were unchanged. No external process was stopped.

## Alignment-weight calibration

| Weight | Mean IA−B Val Acc (pp) | Mean IA−B Macro-F1 (pp) | Decision |
| --- | --- | --- | --- |
| 0.02 | -0.222 | -0.216 | selected |
| 0.05 | -0.533 | -1.004 |  |
| 0.10 | -0.637 | -1.169 |  |

The comparison averages the paired validation-accuracy deltas from Movies and Grocery seed 42. No calibration test data were evaluated or read. Full per-dataset values are in `data/calibration_by_run.csv`.

## Formal validation performance

Values are mean ± population SD over the three model seeds; Accuracy and Macro-F1 are percent, CE is unscaled.

| Dataset | Variant | Val Accuracy (%) | Val Macro-F1 (%) | Val CE |
| --- | --- | --- | --- | --- |
| Movies | B | 56.52 ± 0.06 | 48.67 ± 1.09 | 1.3944 ± 0.0058 |
| Movies | I | 54.70 ± 0.71 | 47.58 ± 1.21 | 1.4268 ± 0.0102 |
| Movies | A | 56.67 ± 0.13 | 49.14 ± 1.09 | 1.3815 ± 0.0135 |
| Movies | IA | 55.85 ± 0.38 | 48.68 ± 0.72 | 1.3510 ± 0.0193 |
| Grocery | B | 82.89 ± 0.29 | 76.35 ± 0.87 | 0.6646 ± 0.0091 |
| Grocery | I | 81.98 ± 0.19 | 74.25 ± 0.44 | 0.6773 ± 0.0053 |
| Grocery | A | 82.69 ± 0.21 | 75.78 ± 0.94 | 0.6636 ± 0.0236 |
| Grocery | IA | 82.63 ± 0.16 | 75.05 ± 0.30 | 0.6496 ± 0.0068 |
| ele-fashion | B | 87.36 ± 0.13 | 74.28 ± 0.77 | 0.4112 ± 0.0028 |
| ele-fashion | I | 86.70 ± 0.07 | 72.15 ± 0.20 | 0.4210 ± 0.0026 |
| ele-fashion | A | 87.43 ± 0.24 | 73.91 ± 0.63 | 0.4213 ± 0.0018 |
| ele-fashion | IA | 87.15 ± 0.07 | 73.91 ± 0.16 | 0.4067 ± 0.0015 |

## Paired comparisons

Accuracy and Macro-F1 differences are percentage points. Seed direction is positive/negative/tie.

| Dataset | Comparison | Accuracy Δ (pp), mean ± SD | +/−/= seeds | Macro-F1 Δ (pp) | CE Δ |
| --- | --- | --- | --- | --- | --- |
| Movies | I-B | -1.820 ± 0.674 | 0/3/0 | -1.084 | +0.0324 |
| Movies | A-B | +0.150 ± 0.196 | 2/1/0 | +0.473 | -0.0129 |
| Movies | IA-B | -0.670 ± 0.320 | 0/3/0 | +0.015 | -0.0434 |
| Movies | IA-I | +1.150 ± 0.673 | 3/0/0 | +1.099 | -0.0758 |
| Movies | IA-A | -0.820 ± 0.510 | 0/3/0 | -0.458 | -0.0305 |
| Grocery | I-B | -0.908 ± 0.470 | 0/3/0 | -2.093 | +0.0127 |
| Grocery | A-B | -0.195 ± 0.097 | 0/3/0 | -0.564 | -0.0011 |
| Grocery | IA-B | -0.264 ± 0.250 | 0/3/0 | -1.297 | -0.0151 |
| Grocery | IA-I | +0.644 ± 0.276 | 3/0/0 | +0.796 | -0.0277 |
| Grocery | IA-A | -0.068 ± 0.227 | 1/2/0 | -0.734 | -0.0140 |
| ele-fashion | I-B | -0.655 ± 0.085 | 0/3/0 | -2.125 | +0.0098 |
| ele-fashion | A-B | +0.068 ± 0.170 | 1/2/0 | -0.368 | +0.0101 |
| ele-fashion | IA-B | -0.208 ± 0.119 | 0/3/0 | -0.365 | -0.0045 |
| ele-fashion | IA-I | +0.447 ± 0.035 | 3/0/0 | +1.760 | -0.0143 |
| ele-fashion | IA-A | -0.276 ± 0.176 | 0/3/0 | +0.002 | -0.0145 |

## Descriptive 2×2 interaction

The factorial contrast is `IA − I − A + B`, paired within dataset and seed. It is descriptive and does not replace IA−B.

| Dataset | Accuracy synergy (pp), mean ± SD | +/−/= seeds | Macro-F1 synergy (pp) | CE factorial contrast |
| --- | --- | --- | --- | --- |
| Movies | +1.000 ± 0.690 | 3/0/0 | +0.626 | -0.0629 |
| Grocery | +0.839 ± 0.371 | 3/0/0 | +1.359 | -0.0267 |
| ele-fashion | +0.378 ± 0.160 | 3/0/0 | +2.127 | -0.0244 |

## Interaction and alignment mechanisms

I and IA attention aggregates cover both directions, all nodes/seeds, three query orders and four source orders. `H0 mass`, same-order/cross-order mass, and entropy describe the learned attention matrix; `rhoJ/Hk` compares the injected residual RMS with the raw structural state.

| Variant | Entropy (nats) | Source H0 mass | Same-order mass | Cross-order mass | RMS(ρJ)/RMS(Hk) | cos(J,Hk) |
| --- | --- | --- | --- | --- | --- | --- |
| I | 1.029 | 0.335 | 0.274 | 0.726 | 0.2159 | 0.011 |
| IA | 0.726 | 0.124 | 0.361 | 0.639 | 0.1703 | 0.003 |

GPR coefficient means (`c0:c3`) by variant: `{"B": [0.23796, 0.05597, 0.62204, -0.09637], "I": [0.18773, 0.12592, 0.68505, -0.03663], "A": [0.23848, 0.05582, 0.62077, -0.09738], "IA": [0.20177, 0.1148, 0.66851, -0.05305]}`. These coefficient changes are descriptive and are not treated as performance evidence.

Alignment is evaluated only on the normalized shared content projections of structural orders 1–3. `INIT` is the same-seed pre-training geometry; B/I values are counterfactual diagnostics because alignment was inactive during those runs.

| Checkpoint | Alignment trained? | Raw loss | Mean same-order cosine | Order 1 cosine | Order 2 cosine | Order 3 cosine |
| --- | --- | --- | --- | --- | --- | --- |
| INIT | no | 1.0040 | -0.0040 | -0.0066 | -0.0016 | -0.0036 |
| B | no | 0.9974 | 0.0026 | 0.0028 | 0.0024 | 0.0024 |
| I | no | 0.9927 | 0.0073 | 0.0062 | 0.0080 | 0.0079 |
| A | yes | 0.1583 | 0.8417 | 0.8547 | 0.8263 | 0.8442 |
| IA | yes | 0.2774 | 0.7226 | 0.7481 | 0.6953 | 0.7245 |

## IA checkpoint interventions

The source shuffle applies five deterministic node permutations to cross-modal source tokens after graph propagation; it leaves backbone states and late fusion inputs in their original node order. These are checkpoint reliance diagnostics only.

| Intervention | Val Acc (%) | Macro-F1 (%) | CE | Accuracy Δ vs normal | Macro-F1 Δ vs normal | CE Δ vs normal |
| --- | --- | --- | --- | --- | --- | --- |
| normal | 75.21 | 65.88 | 0.8024 | — | — | — |
| interaction_off | 74.54 | 64.98 | 0.8163 | -0.667 pp | -0.900 pp | +0.0138 |
| source_node_shuffle | 73.39 | 63.90 | 0.8479 | -1.817 pp | -1.982 pp | +0.0455 |

Across IA checkpoints, interaction-off accuracy change averaged -0.667 pp and source-node shuffle averaged -1.817 pp. Neither intervention establishes retrained architectural value; the primary evidence remains IA−B.

## Interpretation

IA-B is not directionally positive on at least two datasets; the present validation evidence does not establish a broad full-block gain. Retain the result and stop at D0 for review.

- Paired IA−B mean accuracy is positive on 0/3 datasets. Direction, magnitude, seed spread, Macro-F1 and CE are all shown above; no fixed 0.5 pp success threshold was applied.
- I−B: Movies -1.820 pp; Grocery -0.908 pp; ele-fashion -0.655 pp.
- A−B: Movies +0.150 pp; Grocery -0.195 pp; ele-fashion +0.068 pp.
- IA−B: Movies -0.670 pp; Grocery -0.264 pp; ele-fashion -0.208 pp.
- IA−I: Movies +1.150 pp; Grocery +0.644 pp; ele-fashion +0.447 pp; IA−A: Movies -0.820 pp; Grocery -0.068 pp; ele-fashion -0.276 pp.
- Mean descriptive accuracy synergy is positive for 3/3 datasets; this does not supersede the paired IA−B result.
- Mean attention entropy is 0.878 nats, same-order mass 0.317, and source-H0 mass 0.230. Mean injected RMS ratio is 0.1931. These values indicate whether attention and residual injection were numerically active, not whether the model is useful.
- Same-order projected cosine means by checkpoint group: `{"INIT": -0.00395, "B": 0.00256, "I": 0.00735, "A": 0.84175, "IA": 0.72262}`. The A/IA-trained geometry should be compared with INIT, B and I; this observational comparison does not isolate alignment from all other training effects.
- Intervention means: interaction-off -0.667 pp; five-shuffle average -1.817 pp. Agreement with retrained IA−B is supportive mechanistic context only; disagreement does not override the architecture comparison.
- Accuracy/F1/CE directional conflicts for IA−B occur in: Movies, Grocery, ele-fashion.
- Dataset pattern: 0/3 positive IA−B means. The observed split consistency should inform any later review; the single fixed split per dataset limits claims about split-level generalization.

## Stop and recommendation

The D0 screen is complete. No Toys, Reddit-S, NC test, formal LP, or follow-on architecture experiments were run. Do not add MoE/OT/MMD/prototypes or more attention in this phase. For a later phase, retain IA only if its retrained IA−B evidence and secondary metrics are sufficiently consistent under human review; retain I or A alone only if their paired comparisons support the simpler arm. Otherwise keep the results as a negative/uncertain screen and stop. This report does not claim causal mechanism proof or generalization beyond the fixed validation splits and three model seeds.

## Final self-audit

1. Strictly started from `cd9d440aa43454215e1a512b5426a5372e47a41a`: **yes**; the branch was created after verifying local and origin C1 HEAD and a clean worktree.
2. Used another experiment branch: **no merge or cherry-pick**; this branch is based directly on C1.
3. Ran or inspected NC test: **no**; test evaluation was disabled, no test metrics were written, and validation analysis uses train/validation labels only.
4. B strict C1 RGD regression: **yes**, exact eval output at zero tolerance in the regression test.
5. Alignment weight: **0.02**, selected by `selected highest mean validation accuracy delta` from the predeclared Movies/Grocery seed-42 calibration.
6. Calibration test-free: **yes**, `evaluate_test=false`, `development_no_test=true`, no test metrics.
7. Variant parameter counts/state layouts/seed initialization matched: **yes**, covered by tests.
8. H0 remains pure intrinsic order: **yes**, no interaction residual is added at order zero; tested exactly.
9. Standalone I−B signal: **Movies -1.820 pp; Grocery -0.908 pp; ele-fashion -0.655 pp**.
10. Standalone A−B signal: **Movies +0.150 pp; Grocery -0.195 pp; ele-fashion +0.068 pp**.
11. Full IA−B increment: **Movies -0.670 pp; Grocery -0.264 pp; ele-fashion -0.208 pp**.
12. Pattern I≈B, A≈B, IA>B: **inspect paired values above**; no single-arm requirement was used to stop IA.
13. Descriptive 2×2 synergy positive: **3/3 dataset means**, see seed-level CSV.
14. Attention non-trivial: mean entropy 0.878, same-order mass 0.317, H0 mass 0.230; all source orders remain unmasked.
15. Residual interaction magnitude: mean RMS(ρJ)/RMS(Hk) **0.1931**.
16. Shared geometry: trained A/IA projected cosine and raw loss are compared with same-seed initialization and inactive-arm checkpoints; interpretation remains observational.
17. Source-node shuffle effect: mean accuracy delta **-1.817 pp**; per-seed and per-repeat values are in `data/intervention_by_run.csv`.
18. Intervention/retrained evidence: intervention reliance is reported separately; primary retrained evidence remains IA−B.
19. GPR coefficient changes: B `[0.23796, 0.05597, 0.62204, -0.09637]`, I `[0.18773, 0.12592, 0.68505, -0.03663]`, A `[0.23848, 0.05582, 0.62077, -0.09738]`, IA `[0.20177, 0.1148, 0.66851, -0.05305]`; coefficients alone are not performance evidence.
20. Dataset-specific regime: IA−B mean is positive on **0/3** datasets.
21. Accuracy/F1/CE conflict: Movies, Grocery, ele-fashion.
22. Worth entering a next phase: **wait for human review of the paired IA−B table and secondary metrics**; D0 itself stops here.
23. Mechanisms to retain: **conditional**—IA if full-block evidence supports it; otherwise consider only the simpler supported I or A arm; no unsupported mechanism is promoted.
24. LP smoke protocol: **passed** for B and IA, 2 epochs, 2 batches, LinkNeighborLoader, `[5,5,5]`, positive-edge removal, sampled-batch aux loss, validation and checkpoint; no test evaluation.
25. Claims beyond evidence: **none intended**; fixed-split validation and three seeds do not establish broad generalization or causal mechanism.

## Artifacts

- `data/performance_by_run.csv`, `data/performance_summary.csv`
- `data/paired_delta_by_run.csv`, `data/paired_delta_summary.csv`, `data/factorial_synergy.csv`
- `data/interaction_diagnostics.csv`, `data/alignment_diagnostics.csv`
- `data/intervention_by_run.csv`, `data/intervention_summary.csv`
- `data/calibration_by_run.csv`, `data/calibration_summary.csv`
- `run_manifest.json`, `smoke_status.json`, `alignment_weight_selection.json`
