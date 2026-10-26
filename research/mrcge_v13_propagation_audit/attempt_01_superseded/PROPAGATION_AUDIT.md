# MR-CGE V1.3 Cross-Trajectory & Physical-Basis Audit

Analysis source freeze: `fe0e390e2d2282f8dcb761ab327b381f942f5933`. Base: `788c35392f049fea32f42f5081656924888f7082`. Test/LP/training: **off**.

## Q1. Physical basis by dataset and modality

| Dataset | Modality | B2/B1 RMS | B3/B1 RMS | B3/B2 RMS | Effective rank B0–B3 | Mean off-diagonal cosine |
|---|---|---:|---:|---:|---:|---:|
| Movies | text | 0.956 ± 0.001 | 0.937 ± 0.002 | 0.980 ± 0.001 | 2.604 ± 0.015 | 0.853 ± 0.104 |
| Movies | visual | 0.925 ± 0.002 | 0.890 ± 0.003 | 0.962 ± 0.001 | 2.838 ± 0.019 | 0.793 ± 0.133 |
| Grocery | text | 0.895 ± 0.000 | 0.838 ± 0.000 | 0.936 ± 0.000 | 2.877 ± 0.005 | 0.780 ± 0.140 |
| Grocery | visual | 0.884 ± 0.001 | 0.824 ± 0.002 | 0.932 ± 0.001 | 2.997 ± 0.003 | 0.736 ± 0.161 |
| ele-fashion | text | — | — | — | — | — |
| ele-fashion | visual | — | — | — | — | — |

The CSVs also retain supported and all-node RMS, zero-norm counts, full normalized Gram entries, singular spectra, condition numbers, and node cosine P10/median/P90. A high cosine, low RMS, and rank loss are separate observations; none proves redundant classification content.

## Q2. Higher-order Cross novelty

| Dataset | Modality | nu2 | nu3 | nu3 conditional on X2 | X2/X1 RMS | X3/X2 RMS |
|---|---|---:|---:|---:|---:|---:|
| Movies | text | 0.346 ± 0.041 | 0.198 ± 0.019 | 0.149 ± 0.020 | 0.963 ± 0.008 | 0.983 ± 0.004 |
| Movies | visual | 0.449 ± 0.044 | 0.240 ± 0.019 | 0.190 ± 0.019 | 0.948 ± 0.008 | 0.977 ± 0.004 |
| Grocery | text | 0.527 ± 0.023 | 0.268 ± 0.013 | 0.218 ± 0.013 | 0.917 ± 0.007 | 0.952 ± 0.004 |
| Grocery | visual | 0.492 ± 0.009 | 0.257 ± 0.005 | 0.207 ± 0.006 | 0.918 ± 0.002 | 0.950 ± 0.001 |
| ele-fashion | text | — | — | — | — | — |
| ele-fashion | visual | — | — | — | — | — |

`nu2` and `nu3` project on `span(B0..B3,X1)`; `nu3_conditional_on_X2` projects on that span plus X2. Per-checkpoint singular spectra, scales, tolerance, rank, pseudoinverse and reconstruction diagnostics are in `data/cross_novelty.csv`; Movies seed consistency is in `data/aggregate.json`. A high residual is structural novelty, not evidence of class utility.

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

| Hybrid receiver | Raw hop2/hop1 RMS | Raw hop3/hop2 RMS | Normalized state effective rank | Cross/private profile RMS | Private route entropy | Collaborative route entropy |
|---|---:|---:|---:|---:|---:|---:|
| text | — | — | — | — | — | — |
| visual | — | — | — | — | — | — |

`hybrid_structural_comparison.csv` keeps Hybrid raw Physical States 1–3, per-channel normalized states, expert profiles, route entropy/Top-2 shares, cross/profile scales, and response scales within the Hybrid coordinate system. No cross-model coordinate cosine is computed. These are structural contrasts, not causal module effects.

## Q5. Recommended next route

**D — No route is selected from accuracy; conservative mechanism-only screening leaves the evidence insufficient unless a stable structural pattern is directly visible in all required cells.

This selection uses only the declared diagnostic summaries, not validation or test accuracy. The numerical screening thresholds are exploratory decision aids. Even a stable high novelty does not show predictive benefit; a low novelty does not rule out nonlinear expert use. Hybrid accuracy cannot identify a causal component.

## Audit status

- Verified checkpoint cells: 5/10; incomplete: 5.
- CPU preflight datasets: 3; frozen source files match baseline: True.
- Failures/resource blocks: 1.
- No Test metrics or labels were serialized; Test evaluation and LP were disabled.
- Each full checkpoint analysis ran in an isolated child on physical GPU 1 under the existing ownership guard.

The exact evidence rows are preserved in CSV/JSON, including unsuccessful cells. No Stage B or model development was started.
