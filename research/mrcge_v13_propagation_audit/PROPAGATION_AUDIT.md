# MR-CGE V1.3 Cross-Trajectory & Physical-Basis Audit

Analysis source freeze: `63b8f94a4d44a47099c56a544ad0a488f2fd7403`. Base: `788c35392f049fea32f42f5081656924888f7082`. Test/LP/training: **off**.

## Q1. Physical basis by dataset and modality

| Dataset | Modality | B2/B1 RMS | B3/B1 RMS | B3/B2 RMS | Effective rank B0–B3 | Mean off-diagonal cosine |
|---|---|---:|---:|---:|---:|---:|
| Movies | text | 0.956 ± 0.001 | 0.937 ± 0.002 | 0.980 ± 0.001 | 2.604 ± 0.015 | 0.853 ± 0.104 |
| Movies | visual | 0.925 ± 0.002 | 0.890 ± 0.003 | 0.962 ± 0.001 | 2.838 ± 0.019 | 0.793 ± 0.133 |
| Grocery | text | 0.895 ± 0.000 | 0.838 ± 0.000 | 0.936 ± 0.000 | 2.877 ± 0.005 | 0.780 ± 0.140 |
| Grocery | visual | 0.884 ± 0.001 | 0.824 ± 0.002 | 0.932 ± 0.001 | 2.997 ± 0.003 | 0.736 ± 0.161 |
| ele-fashion | text | 0.955 ± 0.000 | 0.925 ± 0.000 | 0.969 ± 0.000 | 2.824 ± 0.003 | 0.723 ± 0.165 |
| ele-fashion | visual | 0.958 ± 0.000 | 0.932 ± 0.000 | 0.973 ± 0.000 | 2.852 ± 0.002 | 0.706 ± 0.174 |

Across the V1.3 cells, the four-column effective rank is about 2.60–3.00 and mean off-diagonal cosine about 0.71–0.85. B3/B1 RMS stays about 0.82–0.94 (B2/B1 about 0.88–0.96), so the data show basis correlation and moderate attenuation, without a collapse to near-zero propagation. This is numerical span redundancy, not evidence of redundant classification information. The CSVs retain supported/all-node RMS, zero-norm counts, Gram entries, singular spectra, condition numbers, and node cosine P10/median/P90.

## Q2. Higher-order Cross novelty

| Dataset | Modality | nu2 | nu3 | nu3 conditional on X2 | X2/X1 RMS | X3/X2 RMS |
|---|---|---:|---:|---:|---:|---:|
| Movies | text | 0.346 ± 0.041 | 0.198 ± 0.019 | 0.149 ± 0.020 | 0.963 ± 0.008 | 0.983 ± 0.004 |
| Movies | visual | 0.449 ± 0.044 | 0.240 ± 0.019 | 0.190 ± 0.019 | 0.948 ± 0.008 | 0.977 ± 0.004 |
| Grocery | text | 0.527 ± 0.023 | 0.268 ± 0.013 | 0.218 ± 0.013 | 0.917 ± 0.007 | 0.952 ± 0.004 |
| Grocery | visual | 0.492 ± 0.009 | 0.257 ± 0.005 | 0.207 ± 0.006 | 0.918 ± 0.002 | 0.950 ± 0.001 |
| ele-fashion | text | 0.644 ± 0.003 | 0.162 ± 0.001 | 0.141 ± 0.001 | 0.966 ± 0.001 | 0.979 ± 0.000 |
| ele-fashion | visual | 0.722 ± 0.009 | 0.154 ± 0.002 | 0.138 ± 0.002 | 0.962 ± 0.001 | 0.974 ± 0.001 |

Movies nu2 remains 0.304–0.483 and nu3 conditional on X2 0.128–0.205 across the six seed-by-modality cells. X2/X1 RMS is 0.94–0.97 and X3/X2 0.96–0.98, so the diagnostic trajectory does not show severe energy decay. Grocery and ele-fashion show higher nu2 but lower conditional nu3 for Fashion. `nu2` and `nu3` project on `span(B0..B3,X1)`; `nu3_conditional_on_X2` projects on that span plus X2. Per-checkpoint singular spectra, scales, tolerance, rank, pseudoinverse and reconstruction diagnostics are in `data/cross_novelty.csv`; Movies seed consistency is in `data/aggregate.json`. A residual is structural novelty, not evidence of class utility.

## Q3. SameNode redundancy clues

| Movies receiver | Cross order | Candidate | Mean global cosine across seeds |
|---|---:|---|---:|
| text | 2 | Phi(H_other) | 0.862 ± 0.024 |
| text | 2 | Phi(U_private_other) | 0.915 ± 0.016 |
| text | 3 | Phi(H_other) | 0.792 ± 0.038 |
| text | 3 | Phi(U_private_other) | 0.870 ± 0.027 |
| visual | 2 | Phi(H_other) | 0.800 ± 0.023 |
| visual | 2 | Phi(U_private_other) | 0.867 ± 0.017 |
| visual | 3 | Phi(H_other) | 0.693 ± 0.041 |
| visual | 3 | Phi(U_private_other) | 0.790 ± 0.027 |

`same_node_overlap.csv` separately compares X2/X3 with `Phi(H_other)` and the actual V1.3 `Phi(U_private_other)`. The latter uses the read-only `_capture_state_audit` state. These cosine values indicate alignment only, not a fraction of two-step return paths.

## Q4. Hybrid and MR-CGE structural differences

| Hybrid receiver | Raw hop2/hop1 RMS | Raw hop3/hop2 RMS | Normalized state effective rank | Cross/private profile RMS | Collaborative/private response RMS | Private route entropy | Collaborative route entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 0.962 ± 0.003 | 0.983 ± 0.001 | 2.020 ± 0.015 | 0.344 ± 0.016 | 0.553 ± 0.072 | 0.953 ± 0.053 | 1.005 ± 0.041 |
| visual | 0.940 ± 0.004 | 0.971 ± 0.002 | 2.101 ± 0.024 | 0.272 ± 0.020 | 1.015 ± 0.025 | 0.291 ± 0.095 | 0.220 ± 0.162 |

| Movies modality | MR-CGE B3/B1 | Hybrid raw hop3/hop1 | MR basis rank (of 4) | Hybrid normalized-state rank (of 3) | MR X1/B1 | Hybrid Cross/private-profile |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.937 ± 0.002 | 0.945 ± 0.004 | 2.604 ± 0.015 | 2.020 ± 0.015 | 1.000 ± 0.000 | 0.344 ± 0.016 |
| visual | 0.890 ± 0.003 | 0.913 ± 0.006 | 2.838 ± 0.019 | 2.101 ± 0.024 | 1.000 ± 0.000 | 0.272 ± 0.020 |

Hybrid normalization brings each physical order to active-channel RMS near one before alpha mixing; its effective rank is around 2.0–2.1 for three normalized states, versus around 2.6–2.8 for four MR basis columns. The raw hop ratios remain similar in scale to MR. Hybrid collaborative/private response RMS is 0.553 ± 0.072 for text and 1.015 ± 0.025 for visual, while Cross/private profile RMS is 0.344 ± 0.016 and 0.272 ± 0.020, respectively. This modality difference motivates checking the response path separately from its input scale.

The four MR expert-filter inputs have effective rank 3.06–3.29; their mean off-diagonal cosine is about 0.37–0.53. Hybrid's four alpha-mixed expert profiles have effective rank 2.38 ± 0.03 (text) and 2.46 ± 0.03 (visual). These are model-specific input constructions and should not be treated as coordinate-aligned representations. MR's per-expert input RMS spans about 0.185–0.515 across the Movies seeds; Hybrid private/collaborative profiles are each normalized to RMS near one. Full expert Gram matrices, RMS values, per-seed route entropies, and Top-2 share vectors are in the CSV/JSON evidence.

### Router entropy and Top-2 selection (Movies, three seeds)

| Modality | Model | Private entropy (nats) | Private max expert Top-2 share | Collaborative entropy (nats) | Collaborative max expert Top-2 share |
|---|---|---:|---:|---:|---:|
| text | MR-CGE V1.3 | 1.045 ± 0.084 | 0.366 ± 0.040 | 1.049 ± 0.073 | 0.396 ± 0.026 |
| text | Hybrid-CGE | 0.953 ± 0.053 | 0.487 ± 0.023 | 1.005 ± 0.041 | 0.499 ± 0.002 |
| visual | MR-CGE V1.3 | 1.006 ± 0.058 | 0.349 ± 0.015 | 0.865 ± 0.049 | 0.383 ± 0.042 |
| visual | Hybrid-CGE | 0.291 ± 0.095 | 0.487 ± 0.012 | 0.220 ± 0.162 | 0.493 ± 0.007 |

Entropy is the mean per-node entropy of dense router probabilities. “Max expert Top-2 share” is the largest of four experts’ mean selected shares (selection count divided by K=2), averaged across seeds; full expert-wise shares are preserved in `expert_input_diagnostics.csv` and the Hybrid JSON details. The measurements use the models’ own router outputs and are descriptive. The lower Hybrid visual entropy and larger Top-2 concentration suggest a modality-specific routing difference to investigate, but do not identify a cause or an accuracy benefit.

## Q5. Recommended next route

**C — Movies has a nonzero higher-order residual in every seed and modality (nu2 >= 0.30; nu3 conditional on X2 >= 0.12), while X2 remains strongly aligned with the actual Late SameNode candidate (cosine >= 0.84). This supports Cross-Aligned Propagation as the next controlled question: align or gate the novel residual against the existing candidate before combining orders. The cutoffs are exploratory evidence summaries, not guarantees.**

This selection uses only the declared diagnostic summaries, not validation or test accuracy. The numerical screening thresholds are exploratory decision aids. Even a stable high novelty does not show predictive benefit; a low novelty does not rule out nonlinear expert use. Hybrid accuracy cannot identify a causal component. The strong SameNode cosine is an alignment clue, not a measured backtracking fraction.

## Audit status

- Verified checkpoint cells: 10/10; incomplete: 0.
- CPU preflight datasets: 3; frozen source files match baseline: True.
- Failures/resource blocks: 0.
- No Test metrics or labels were serialized; Test evaluation and LP were disabled.
- Each full checkpoint analysis ran in an isolated child on physical GPU 1 under the existing ownership guard.

The exact evidence rows are preserved in CSV/JSON, including unsuccessful cells. No Stage B or model development was started.
