# Physical propagation depth: three-hop versus four-hop

All 15 independent 3-hop runs completed. The 3-hop implementation retains four experts and Top-2, and constructs only P·H, P²·H and P³·H. Existing 4-hop checkpoint compatibility was tested; historical checkpoints and source records were not changed.

| Dataset | 3-hop Val Acc | 3-hop Val F1 | Historical 4-hop Val Acc | Historical 4-hop Val F1 | Paired Δ Acc | Paired Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 56.68% ± 0.59% | 49.79% ± 0.99% | 56.78% ± 0.44% | 49.78% ± 1.17% | -0.10% ± 0.15% | 0.01% ± 0.96% |
| Grocery | 83.84% ± 0.51% | 76.62% ± 0.99% | 83.86% ± 0.39% | 75.97% ± 1.11% | -0.02% ± 0.13% | 0.64% ± 1.13% |
| ele-fashion | 87.55% ± 0.14% | 75.22% ± 0.71% | 87.72% ± 0.06% | 74.65% ± 0.35% | -0.17% ± 0.16% | 0.57% ± 1.06% |
| Toys | 80.33% ± 0.13% | 77.36% ± 0.85% | 80.33% ± 0.31% | 77.47% ± 0.13% | 0.01% ± 0.28% | -0.12% ± 0.73% |
| Reddit-S | 96.60% ± 0.11% | 92.89% ± 0.61% | 96.60% ± 0.25% | 93.02% ± 0.50% | 0.00% ± 0.14% | -0.13% ± 0.17% |

Seed-aligned paired differences and both checkpoint hashes are in `data/hop_depth_paired_differences.csv`. The 4-hop rows are from earlier campaigns (Movies/Grocery/ele-fashion: `H2_hybrid_plus`; Toys/Reddit-S: `H2-Frozen`), so this is a cross-campaign reference comparison, not a fresh matched control. Three seeds provide descriptive evidence only. A global depth choice is made in `MODEL_SELECTION.md` using the declared Accuracy-primary, Macro-F1-secondary rule; Test was not evaluated.
