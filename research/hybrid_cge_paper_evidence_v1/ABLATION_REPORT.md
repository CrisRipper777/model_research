# Ablation report

## Implementation checks

The historical `src/models/hybrid_cge_v11.py` and checkpoints remain unchanged. A post-freeze source audit found that the paper wrapper replaced the parent's `H2_hybrid_plus` condition with paper condition IDs, while the inherited late path used that exact string to enable SameNode. This silently disabled SameNode in A1, 3-hop, and Fashion F1/F2/F4; the original 39 attempts are retained in `data/runs.csv` and listed in `data/superseded_runs.csv`. The wrapper now explicitly enables the existing SameNode helper for every late-completion arm except A4, and the corrected 39 attempts were appended without replacing their predecessors. Corrected implementation/training source is `5e052dbc26237ddb534e0f779d91951f26079748`; the first freeze `0db724e5eec704a1f3db92329bfbbac056df7202` is retained as the source for unaffected runs. Ten focused Hybrid-CGE paper tests and all 581 repository tests passed after the correction. All selected 60 ablation cells pass their current stage gate; initial classifier and shared-model hashes match within Dataset × Seed, and Train/Validation metrics, checkpoint hashes/reloads and Test-off flags pass. Raw attempts and selected rows are traceable in `data/runs.csv` and `data/superseded_runs.csv`.

The comparison reference is the seed-aligned historical 4-hop H2 row. It is from a prior campaign and commit, so the paired deltas are descriptive, not same-batch or matched-initialization effects. The historical Pilot remains unchanged.

## Validation metrics and paired delta versus historical 4-hop H2

| Dataset | Ablation | n | Val Acc | Val Macro-F1 | Δ Acc | Δ Macro-F1 |
|---|---|---|---|---|---|---|
| Movies | A1_no_early_physical_cross | 3/3 | 56.39% ± 0.76% | 48.58% ± 1.57% | -0.39% ± 0.45% | -1.20% ± 1.50% |
| Movies | A2_no_late_completion | 3/3 | 56.35% ± 0.38% | 49.12% ± 1.73% | -0.43% ± 0.06% | -0.66% ± 1.44% |
| Movies | A3_no_moe | 3/3 | 57.19% ± 0.12% | 50.54% ± 0.97% | 0.41% ± 0.53% | 0.76% ± 1.68% |
| Movies | A4_no_same_node | 3/3 | 56.36% ± 0.67% | 49.85% ± 1.46% | -0.42% ± 0.30% | 0.07% ± 0.75% |
| Grocery | A1_no_early_physical_cross | 3/3 | 83.60% ± 0.56% | 75.72% ± 1.91% | -0.25% ± 0.26% | -0.25% ± 1.26% |
| Grocery | A2_no_late_completion | 3/3 | 83.75% ± 0.56% | 76.72% ± 0.71% | -0.11% ± 0.18% | 0.75% ± 0.73% |
| Grocery | A3_no_moe | 3/3 | 83.50% ± 0.67% | 75.86% ± 1.43% | -0.35% ± 0.35% | -0.12% ± 1.43% |
| Grocery | A4_no_same_node | 3/3 | 83.78% ± 0.43% | 76.61% ± 1.11% | -0.08% ± 0.11% | 0.64% ± 1.17% |
| ele-fashion | A1_no_early_physical_cross | 3/3 | 87.60% ± 0.15% | 74.68% ± 0.34% | -0.12% ± 0.14% | 0.03% ± 0.44% |
| ele-fashion | A2_no_late_completion | 3/3 | 87.53% ± 0.11% | 74.65% ± 0.68% | -0.19% ± 0.12% | -0.01% ± 0.50% |
| ele-fashion | A3_no_moe | 3/3 | 87.64% ± 0.05% | 75.04% ± 0.42% | -0.08% ± 0.10% | 0.39% ± 0.40% |
| ele-fashion | A4_no_same_node | 3/3 | 87.68% ± 0.12% | 74.65% ± 0.09% | -0.04% ± 0.12% | -0.00% ± 0.40% |
| Toys | A1_no_early_physical_cross | 3/3 | 80.33% ± 0.19% | 77.41% ± 0.06% | 0.01% ± 0.13% | -0.06% ± 0.12% |
| Toys | A2_no_late_completion | 3/3 | 80.21% ± 0.13% | 77.36% ± 0.34% | -0.11% ± 0.21% | -0.11% ± 0.22% |
| Toys | A3_no_moe | 3/3 | 80.05% ± 0.33% | 77.25% ± 0.36% | -0.27% ± 0.04% | -0.22% ± 0.44% |
| Toys | A4_no_same_node | 3/3 | 80.23% ± 0.26% | 77.58% ± 0.23% | -0.10% ± 0.08% | 0.11% ± 0.14% |
| Reddit-S | A1_no_early_physical_cross | 3/3 | 96.58% ± 0.19% | 92.90% ± 0.82% | -0.02% ± 0.13% | -0.12% ± 0.35% |
| Reddit-S | A2_no_late_completion | 3/3 | 96.58% ± 0.20% | 92.74% ± 0.39% | -0.02% ± 0.08% | -0.27% ± 0.36% |
| Reddit-S | A3_no_moe | 3/3 | 96.72% ± 0.22% | 92.88% ± 0.30% | 0.12% ± 0.12% | -0.14% ± 0.33% |
| Reddit-S | A4_no_same_node | 3/3 | 96.59% ± 0.20% | 92.79% ± 0.62% | -0.01% ± 0.05% | -0.22% ± 0.37% |

The four mechanisms are true framework-level path removals (see `ABLATION_DESIGN.md`): corrected A1 bypasses early physical cross computation while retaining SameNode; A2 skips late completion; A3 removes expert/router parameter trees and their balance loss while retaining SameNode; A4 masks and skips the same-node path while retaining 0.15 late-cross dropout. Negative or near-zero changes are retained as observed, not interpreted causally from only three seeds.

Training was Validation-only. Test labels were masked before the Trainer and Test metrics were not evaluated.
