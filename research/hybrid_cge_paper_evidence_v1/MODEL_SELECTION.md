# Validation-only model selection

## Predeclared rule

Validation Accuracy is primary. Macro-F1, cross-entropy (CE), per-class metrics, Train–Validation gaps and stability are secondary. Comparisons use the five datasets equally and align seeds where recorded. Fashion candidates remain explicitly dataset-specific. Historical Test results and Test labels never select a model.

## Global physical depth

| Dataset | 3-hop Val Accuracy | 3-hop Val Macro-F1 | Historical 4-hop Val Accuracy | Historical 4-hop Val Macro-F1 | Paired Δ Accuracy | Paired Δ Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 56.81% ± 0.57% | 50.42% ± 1.39% | 56.78% ± 0.44% | 49.78% ± 1.17% | +0.03 ± 0.14 pp | +0.64 ± 0.42 pp |
| Grocery | 83.89% ± 0.54% | 76.97% ± 0.59% | 83.86% ± 0.39% | 75.97% ± 1.11% | +0.04 ± 0.14 pp | +1.00 ± 0.66 pp |
| ele-fashion | 87.80% ± 0.17% | 75.03% ± 0.91% | 87.72% ± 0.06% | 74.65% ± 0.35% | +0.08 ± 0.18 pp | +0.38 ± 0.63 pp |
| Toys | 80.27% ± 0.29% | 77.05% ± 0.61% | 80.33% ± 0.31% | 77.47% ± 0.13% | −0.06 ± 0.07 pp | −0.42 ± 0.49 pp |
| Reddit-S | 96.61% ± 0.13% | 93.07% ± 0.42% | 96.60% ± 0.25% | 93.02% ± 0.50% | +0.01 ± 0.12 pp | +0.05 ± 0.10 pp |

The equal-dataset mean paired change is +0.02 pp Accuracy and +0.33 pp Macro-F1; 3-hop is higher on both measures in four of five datasets, while Toys favors 4-hop. These effects are small and the 4-hop values come from earlier campaigns rather than a fresh matched-init control. Treat this as a Validation-based working choice, not a statistically established gain or a causal estimate.

The CPU constructor audit recreated all 15 Dataset × Seed pairs with labels masked. The classifier and every common non-`alpha_raw` tensor were bitwise identical between H2 4-hop and 3-hop. The only expected state difference was `alpha_raw` changing shape from [4,4] to [4,3], initialized from the first three columns. See `data/initialization_matching_audit.json`.

## Working choice

Use `H2_physical_3hop` as the provisional global paper model configuration under the Accuracy-primary rule, with H2 4-hop retained as the historical reference and Toy-specific counterexample. For the pending final-model review record, seed 42 is the first preregistered seed, not the best Validation seed; this avoids selecting a seed by its observed Validation score. Proposed per-dataset checkpoints and all artifact hashes are recorded in `FINAL_MODEL_LOCK.json`. The lock remains `pending_review`, and Test remains disabled.

## Fashion

F3 (the same H2 3-hop model) has the highest mean Validation Accuracy: 87.80% ± 0.17%, +0.08 pp vs historical F0, and Macro-F1 75.03% ± 0.91%, +0.38 pp. F4 has the highest mean Macro-F1 (75.18% ± 0.89%, +0.52 pp vs F0) and Best-F1-checkpoint Macro-F1 (76.81%), but Accuracy is 87.37% ± 0.18%, −0.35 pp vs F0. Under Accuracy-primary selection, retain F3 rather than introducing a Fashion-specific class-weighted configuration. If Macro-F1 becomes the primary objective, F4 is a candidate for separate review, not a locked choice.

F1 jointly changes Dropout (0.20→0.30) and private auxiliary CE (0.3→0.1), so the individual effects cannot be separated. Weight decay is fixed at 1e−4 throughout; neither its main effect nor any interaction with Dropout or class balancing is estimated. F2 isolates class-balanced main CE at 4-hop; F4 combines it with 3-hop. Their Accuracy/F1 trade-off is reported in `FASHION_TUNING_REPORT.md`. A Fashion class with zero Train support receives weight zero; no Train target belongs to that class.

## Baseline position

Against same-protocol supervised NC adaptations, 3-hop Hybrid-CGE has strong Validation results but is not uniformly best. DiP leads Grocery Accuracy and leads ele-fashion on both Accuracy and Macro-F1. Hybrid-CGE is competitive on Movies and Reddit-S and close on Toys. Baseline adaptation limits and complete means are in `BASELINE_AUDIT.md` and `BASELINE_REPORT.md`; these results do not establish faithful reproduction of every source paper.

## Checkpoint criterion and limits

Best-Accuracy is the primary checkpoint; Best-F1 is reported separately. Across the 15-run Hybrid H2 ablation/depth cells, selecting Best-F1 instead changes Accuracy by −0.56 ± 0.60 pp and increases Macro-F1 by +0.68 ± 0.68 pp on average. CE and Train–Validation gaps also vary; see `CHECKPOINT_SELECTION_REPORT.md` and its CSV for all model families. Only three seeds are available, the depth comparison is cross-campaign, and no Test result is included.
