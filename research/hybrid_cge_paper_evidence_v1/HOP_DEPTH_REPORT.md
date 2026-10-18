# Physical propagation depth: three-hop versus four-hop

All 15 corrected 3-hop runs completed. The 3-hop implementation retains four experts and Top-2, constructs only P·H, P²·H and P³·H, and preserves the SameNode path. First attempts from the initial freeze are preserved and superseded in `data/superseded_runs.csv`. Existing 4-hop checkpoint compatibility was tested; historical checkpoints and source records were not changed.

| Dataset | 3-hop Val Acc | 3-hop Val F1 | Historical 4-hop Val Acc | Historical 4-hop Val F1 | Paired Δ Acc | Paired Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 56.81% ± 0.57% | 50.42% ± 1.39% | 56.78% ± 0.44% | 49.78% ± 1.17% | 0.03% ± 0.14% | 0.64% ± 0.42% |
| Grocery | 83.89% ± 0.54% | 76.97% ± 0.59% | 83.86% ± 0.39% | 75.97% ± 1.11% | 0.04% ± 0.14% | 1.00% ± 0.66% |
| ele-fashion | 87.80% ± 0.17% | 75.03% ± 0.91% | 87.72% ± 0.06% | 74.65% ± 0.35% | 0.08% ± 0.18% | 0.38% ± 0.63% |
| Toys | 80.27% ± 0.29% | 77.05% ± 0.61% | 80.33% ± 0.31% | 77.47% ± 0.13% | -0.06% ± 0.07% | -0.42% ± 0.49% |
| Reddit-S | 96.61% ± 0.13% | 93.07% ± 0.42% | 96.60% ± 0.25% | 93.02% ± 0.50% | 0.01% ± 0.12% | 0.05% ± 0.10% |

Seed-aligned paired differences and both checkpoint hashes are in `data/hop_depth_paired_differences.csv`. The 4-hop rows are from earlier campaigns (Movies/Grocery/ele-fashion: `H2_hybrid_plus`; Toys/Reddit-S: `H2-Frozen`), so this is a cross-campaign reference comparison, not a fresh matched control. Three seeds provide descriptive evidence only. A global depth choice is made in `MODEL_SELECTION.md` using the declared Accuracy-primary, Macro-F1-secondary rule; Test was not evaluated.
