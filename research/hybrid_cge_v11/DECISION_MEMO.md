# Hybrid-CGE V1.1 Decision Memo

The 18-run Validation matrix is complete. Paired conclusions use same-seed comparisons and strict historical fingerprint gates. Three seeds are descriptive, not population-level significance tests.

## Paired Best-Accuracy differences

| Dataset | Contrast | Δ Accuracy pp | Δ Macro-F1 pp | Δ CE |
|---|---|---:|---:|---:|
| Movies | H1_hybrid − CG-CGE B1 | -0.180 ± 0.030 | +0.259 ± 1.041 | -0.002 ± 0.041 |
| Movies | H1_hybrid − View-CGE A2 | +0.380 ± 0.135 | +1.916 ± 0.569 | -0.019 ± 0.110 |
| Movies | H2_hybrid_plus − CG-CGE B1 | +0.260 ± 0.075 | +0.169 ± 0.323 | -0.026 ± 0.030 |
| Movies | H2_hybrid_plus − View-CGE A2 | +0.820 ± 0.200 | +1.826 ± 1.539 | -0.043 ± 0.071 |
| Movies | H2_hybrid_plus − within-study H1 | +0.440 ± 0.069 | -0.090 ± 1.186 | -0.024 ± 0.044 |
| Grocery | H1_hybrid − CG-CGE B1 | -0.029 ± 0.051 | +0.009 ± 0.447 | +0.004 ± 0.021 |
| Grocery | H1_hybrid − View-CGE A2 | +0.088 ± 0.232 | +0.149 ± 0.766 | -0.015 ± 0.029 |
| Grocery | H2_hybrid_plus − CG-CGE B1 | +0.225 ± 0.235 | -0.444 ± 1.118 | -0.008 ± 0.011 |
| Grocery | H2_hybrid_plus − View-CGE A2 | +0.342 ± 0.089 | -0.304 ± 0.216 | -0.027 ± 0.015 |
| Grocery | H2_hybrid_plus − within-study H1 | +0.254 ± 0.278 | -0.453 ± 0.926 | -0.012 ± 0.014 |
| ele-fashion | H1_hybrid − CG-CGE B1 | -0.048 ± 0.198 | -0.292 ± 0.619 | -0.006 ± 0.004 |
| ele-fashion | H1_hybrid − View-CGE A2 | -0.136 ± 0.189 | -1.100 ± 1.278 | -0.022 ± 0.019 |
| ele-fashion | H2_hybrid_plus − CG-CGE B1 | +0.143 ± 0.205 | +0.337 ± 0.396 | -0.002 ± 0.008 |
| ele-fashion | H2_hybrid_plus − View-CGE A2 | +0.055 ± 0.119 | -0.471 ± 1.142 | -0.019 ± 0.025 |
| ele-fashion | H2_hybrid_plus − within-study H1 | +0.191 ± 0.073 | +0.629 ± 0.458 | +0.003 ± 0.009 |

Best-F1 results, seed-level signs and Train/Validation gaps are in PILOT_REPORT.md and paired_comparisons.csv. Interpret small gains against seed spread, checkpoint sensitivity and the ele-fashion lower-support class results.

No optional training-strategy screen was run. H2's paired Best-Accuracy differences are small and mixed by metric: Accuracy is positive across datasets, Macro-F1 is slightly lower on Movies and Grocery but higher on ele-fashion, and CE improves on Movies/Grocery but is slightly higher on ele-fashion. Since H2 combines SameNode with late dropout, this pattern does not provide a clean, isolated hypothesis for changing dropout, weight decay or learning rate in a small screen.

## Interpretation boundary

H2 is a combined Same-node plus training-only late-dropout package; this matrix does not separate those effects. Physical/semantic router activity and residual scales are diagnostics, not causal ablations. The mixed endpoint pattern did not justify an optional training-strategy screen. No Test evaluation or successor method was included.
