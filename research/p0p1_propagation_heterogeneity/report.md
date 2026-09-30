# P0/P1 report: physical-relation propagation utility heterogeneity

## Scope and interpretation

This is a descriptive NC probe study, not a benchmark comparison. It covers Movies, Grocery, and ele-fashion with seeds 42–44 (9 dataset/seed runs). The probe uses separate 128-dimensional `Linear → LayerNorm` modality projections, `[h_text || h_visual || c_text || c_visual]`, uniform incoming one-hop means over the configured undirected physical graph, no self-loops, and a linear classifier. Semantic-only uses `[h_text || h_visual]`. AdamW (`lr=1e-3`, `weight_decay=1e-4`), at most 300 epochs, validation-accuracy checkpoint selection, patience 30, and minimum epoch 30 were shared across datasets.

Only train labels enter the loss. Validation labels select the checkpoint and are the target labels for utility and intervention analysis. No test metric was computed and the probe's data object has no `test_idx`; labels outside train/validation are masked from the object. The graph and frozen node features are used transductively for one-hop context as in NC. Split tensor storages are memory-mapped, and the accessor requests only train/validation fields; it raises if code requests a test-index key.

The requested CE fraction and its stated sign convention conflict: `CE_removed / CE_full` is positive for both helpful and harmful messages. We therefore use the signed relative CE change `(CE_removed - CE_full) / max(CE_full, 1e-12)`: positive means deletion raises CE (helpful message); negative means deletion lowers CE (harmful message). This resolves the displayed formula in favor of its explicit sign interpretation. Margin utility is a secondary check. No threshold is imposed to label a message “neutral”; exact zero CE utilities were absent in the observed edge records.

## 1. Uniform one-hop context vs semantic-only

Validation metrics improved for all 9 paired runs. Table values are mean ± population SD across seeds; deltas are uniform-one-hop minus semantic-only.

| Dataset | Semantic Acc | Uniform 1-hop Acc | Δ Acc | Semantic Macro-F1 | Uniform 1-hop Macro-F1 | Δ Macro-F1 | Δ CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 51.83 ± 0.45% | 54.58 ± 0.04% | +2.75 pp | 0.389 ± 0.009 | 0.459 ± 0.010 | +0.070 | −0.078 |
| Grocery | 77.21 ± 0.33% | 81.33 ± 0.15% | +4.12 pp | 0.692 ± 0.013 | 0.732 ± 0.010 | +0.041 | −0.136 |
| ele-fashion | 86.40 ± 0.08% | 86.96 ± 0.08% | +0.56 pp | 0.652 ± 0.007 | 0.671 ± 0.005 | +0.019 | −0.018 |

Across all 9 runs, accuracy rose by 2.48 pp on average, Macro-F1 by 4.32 pp, and validation CE fell by 0.077. This is a sanity check that uniform context carries useful signal in these splits. It does not show that any learned weighting or multiple operator design is better.

## Margin sanity metric

The secondary margin utility agreed in direction with the primary CE utility on 87.0–96.6% of nonzero edge/modality pairs across runs; per-run CE/margin Spearman correlations ranged from 0.878 to 0.969. Agreement was lowest in Movies and highest for ele-fashion Text. This supports the broad CE direction while leaving CE as the primary analysis, as specified.

## 2. H1 — Edge-level utility heterogeneity

**SUPPORTED.** Every dataset/seed/modality combination had both positive and negative edge utilities overall, and every run had a positive median within-node IQR and q90–q10 spread. Within-node mixed-sign neighborhoods occurred in every dataset and seed, though their prevalence differed materially.

| Dataset | Modality | Helpful edge fraction | Harmful edge fraction | Median node IQR | Median node q90–q10 | Mixed-sign target neighborhoods |
|---|---|---:|---:|---:|---:|---:|
| Movies | Text | 62.0% | 38.0% | 0.0198 | 0.0374 | 38.9% |
| Movies | Visual | 63.0% | 37.0% | 0.0356 | 0.0667 | 54.9% |
| Grocery | Text | 78.1% | 21.9% | 0.0629 | 0.1197 | 39.6% |
| Grocery | Visual | 76.5% | 23.5% | 0.0776 | 0.1471 | 42.9% |
| ele-fashion | Text | 77.7% | 22.3% | 0.0085 | 0.0166 | 15.6% |
| ele-fashion | Visual | 26.7% | 73.3% | 0.0077 | 0.0149 | 16.5% |

Mixed-sign ratios were stable across the three seeds within each dataset: Movies ranged 32–47% for Text and 53–58% for Visual; Grocery 39–41% and 42–44%; ele-fashion 15–16% and 13–19%. Thus the within-node phenomenon is present across datasets, but its prevalence and magnitude are lower in ele-fashion. These results support heterogeneous neighbor utility without implying that every target has both signs.

## 3. H2 — Same edge, different modality utility

**SUPPORTED.** The two modalities' utilities were not interchangeable on the same directed edge. Across seeds, mean Text/Visual Spearman correlation was 0.576 in Grocery, 0.168 in Movies, and −0.303 in ele-fashion. Sign disagreement was 22.3%, 41.4%, and 67.2%, respectively. Among degree≥5 validation targets, mean top-20% set Jaccard was 0.229, 0.187, and 0.199; bottom-20% Jaccard was 0.266, 0.195, and 0.201.

The median absolute Text–Visual utility difference was 0.0417 in Grocery (mean seed IQR 0.1022), 0.0247 in Movies (IQR 0.0530), and 0.1237 in ele-fashion (IQR 0.3617). Within-dataset seed ranges were narrow: Grocery sign disagreement 21.5–23.1%, Movies 40.1–43.6%, and ele-fashion 65.0–68.7%. The consistent dataset-specific pattern is notable: utilities align most in Grocery, are weakly aligned in Movies, and are inversely associated with substantial sign disagreement in ele-fashion. This supports modality-dependent utility while preserving Grocery as an important case where the modalities share some ordering.

## 4. H3 — Raw semantic similarity is not propagation utility

**SUPPORTED.** Raw frozen-feature cosine had only weak association with CE utility overall. Mean per-run Spearman correlations were 0.056 for Text (range 0.010–0.102) and 0.068 for Visual (range −0.070–0.171). Similarity-to-positive-utility AUROC averaged 0.526 and 0.534, near 0.5. On degree≥5 targets, high-similarity vs high-utility top-set Jaccard averaged 0.181 for Text and 0.175 for Visual; low-similarity vs low-utility bottom-set Jaccard averaged 0.224 and 0.226.

The pattern varies by dataset. Grocery shows a weak positive trend (mean Spearman 0.094 Text, 0.162 Visual); Movies is weaker (0.048, 0.070); ele-fashion is near zero for Text (0.026) and slightly negative for Visual (−0.029). All 180 dataset × seed × modality decile bins contained both positive and negative utility messages. Similarity therefore provides weak statistical evidence in some settings, but does not determine an edge's utility sign or ranking.

## 5. Group intervention sanity check

The interventions were restricted to validation targets with degree≥5. Each intervention changed one modality's neighbor context only, preserved intrinsic semantics and the original degree denominator, and did not retrain the probe. Random controls used 10 deterministic repeats with the same per-target mask count.

| Modality | Removed messages | Δ validation CE | Δ Accuracy | Δ Macro-F1 |
|---|---|---:|---:|---:|
| Text | Bottom utility 20% | −0.0433 | +1.19 pp | +2.47 pp |
| Text | Random matched count | +0.0064 | −0.54 pp | +0.02 pp |
| Text | Top utility 20% | +0.0587 | −2.27 pp | −2.08 pp |
| Visual | Bottom utility 20% | −0.0576 | +1.92 pp | +2.40 pp |
| Visual | Random matched count | +0.0126 | −0.63 pp | −0.82 pp |
| Visual | Top utility 20% | +0.0868 | −3.11 pp | −3.82 pp |

Across the 18 paired dataset × seed × modality comparisons, bottom-utility removal was better than random on CE, accuracy, and Macro-F1 in 18/18; the mean paired CE difference was −0.0599, accuracy +2.14 pp, and Macro-F1 +2.83 pp. Top-utility removal was worse than random in the expected direction in 18/18; corresponding differences were +0.0633 CE, −2.10 pp accuracy, and −2.55 pp Macro-F1. This supports the practical ranking signal in the single-edge utility probe. It remains a validation-set intervention sanity check, not a held-out test result.

## 6. Cross-dataset and cross-seed consistency

- P0 context gains were positive on all 9 runs, with a smaller gain in ele-fashion.
- H1 mixed-sign neighborhoods appeared for all datasets and seeds, but were substantially less frequent in ele-fashion than in Movies and Grocery.
- H2's dataset ordering was stable across seeds: strongest Text/Visual alignment in Grocery, weaker in Movies, and negative with the most sign disagreement in ele-fashion.
- H3 remained weak in all three datasets. Grocery had the clearest positive similarity trend; ele-fashion Visual similarity was uninformative or slightly negative.
- Group interventions had the expected direction against random in all 18 comparisons.

No edge-level p-values are reported; directed edges share target nodes and are not treated as independent observations.

## 7. Recommendation for a later Minimal V0

The results justify asking whether utility-aware message handling improves on uniform aggregation. A small Minimal V0 is reasonable, with uniform mean and scalar gating as essential references and operator routing as a candidate comparison. P0/P1 does **not** establish that multiple operators beat scalar weighting.

A shared-vs-modality-specific comparison is also worth a controlled test because same-edge utilities diverged, especially in ele-fashion. The moderate Grocery correlation cautions against presuming that modality-specific routing will help every dataset. These are questions for the next review; no V0 model, router, or operator bank is implemented here.

## Limitations

1. The checkpoint is selected using validation accuracy and edge utility/interventions are then measured on those same validation targets. This is the requested protocol but can make effects optimistic; a later confirmatory design should reserve independent targets or another validation split.
2. This is transductive full-graph context over fixed features. It does not test inductive behavior or external graph generalization.
3. CE utility uses the signed relative-change interpretation needed for the stated sign semantics; the attachment's displayed quotient omitted subtraction. The chosen formula and denominator are recorded in code and manifest.
4. The investigation is descriptive and makes no causal claim beyond these probe counterfactuals. No test evaluation, LP, role feature, learned edge weight, router, or model-design conclusion is included.
