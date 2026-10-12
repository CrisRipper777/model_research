# PACE-MAG V1 mechanism audit

This report describes the three frozen seed-42 E2 Full validation checkpoints. Every intervention holds model and classifier weights fixed. Validation deltas are observations on these checkpoints, not retrained ablation scores or significance tests.

## A. Semantic source views

All physical, Text-KNN, Visual-KNN, and semantic-union operators covered every node in these cached graphs. Text and Visual supports were mostly distinct. “Both-present” means that each source proposed the undirected edge at least once; a raw proposal count of two for one source is not evidence of joint provenance.

| Dataset | Unique semantic edges: Text only / Visual only / Both | Text–Visual global support Jaccard | Physical–Text / Physical–Visual / Physical–union Jaccard | Mean text share of semantic row quality |
|---|---:|---:|---:|---:|
| Movies | 50.4% / 46.2% / 3.4% | 3.4% | 3.7% / 8.8% / 6.7% | 0.510 |
| Grocery | 49.1% / 46.9% / 4.0% | 4.0% | 3.9% / 5.3% / 4.8% | 0.507 |
| ele-fashion | 44.8% / 47.1% / 8.1% | 8.1% | 4.5% / 2.4% / 3.4% | 0.487 |

Per-node quality, text-share, distinct degree, and participation-ratio effective-neighbor distributions are in `data/source_context_stats.csv`. Median effective semantic-union neighbors were 21.7, 21.6, and 20.5 respectively; median effective neighbors per individual semantic source were about 9.7–10.7. The median Text share was near 0.5 on every dataset, but the node-level distributions are broad and are retained in the CSV.

With each target modality fixed, Text-source and Visual-source aggregated contexts differed for nearly every node, including when cross-modal injection was turned off. The table gives mean cosine, median and 90th-percentile normalized L2, and the fraction with relative L2 above 0.1. The denominator for normalized L2 is the larger of the two nodewise context norms. All eligible nodes had both sources; zero-vector rows are masked for cosine.

| Dataset | Target | Cross-on cosine mean | Cross-on relative L2 median / q90 | Nodes with relative L2 > 0.1 | Cross-off cosine mean |
|---|---|---:|---:|---:|---:|
| Movies | Text | 0.633 | 0.737 / 1.062 | 100.0% | 0.634 |
| Movies | Visual | 0.686 | 0.704 / 0.963 | 100.0% | 0.685 |
| Grocery | Text | 0.732 | 0.664 / 0.914 | 100.0% | 0.731 |
| Grocery | Visual | 0.685 | 0.708 / 0.952 | 99.8% | 0.687 |
| ele-fashion | Text | 0.873 | 0.400 / 0.702 | 96.8% | 0.875 |
| ele-fashion | Visual | 0.867 | 0.420 / 0.737 | 98.5% | 0.869 |

These are distinct contexts, although the ele-fashion contexts are more aligned by cosine than Movies and Grocery. They establish that the two supports carry different messages, not that a learned source selector would improve labels.

### Operator identity and normalization boundary

For fixed same-modality messages, the audit directly verified `qT*UT + qV*UV = PU@H` and equality of the corresponding row means on valid rows. The largest absolute residual over the three datasets and two modalities was `1.0e-6` for the weighted identity and `7.2e-7` for the row-mean identity. Rebuilding a single source view with edge weight `0.9` instead of `0.8` changed its row mean by at most `7.2e-6`, within floating-point tolerance, while changing mean `log1p(q_union)` by about `0.053`.

Thus the source split has an exact operator-level reconstruction under fixed `H`, and row-quality normalization changes local scale. The fixed `0.8` factor cancels from a single view's row mean but changes absolute `q` supplied to the Router. Modality-specific cross-modal messages, nonlinear experts, and routing mean this operator identity is not a general end-to-end model equivalence.

## B. Cross-modal path

The four source-to-target cells are fully recorded in `data/cross_modal_strength.csv`. Across cells, checkpoint `rho` ranged from `0.0997` to `0.1079`; `RMS(rho*Phi(H_other))/RMS(H_target)` ranged from `1.8%` to `10.4%`. The source-sent message changed by the same RMS ratio because the message adds that injection directly. After relation aggregation, the relative context change ranged from `1.7%` to `12.0%`. The aggregate-to-send relative-change ratio ranged from `0.94` to `1.54`; none of the four paths showed 50% or greater attenuation. Aggregation sometimes increased the relative change.

Turning off only the cross-modal message in the frozen checkpoint changed the final Validation outputs very little:

| Dataset | Embedding relative delta RMS | Logit relative delta RMS | Validation CE delta | Accuracy delta (pp) | Macro-F1 delta (pp) | Prediction flips |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0.0043 | 0.0037 | +0.0004 | +0.03 | -0.05 | 0.27% |
| Grocery | 0.0081 | 0.0072 | -0.0002 | +0.03 | -0.06 | 0.20% |
| ele-fashion | 0.0198 | 0.0153 | +0.0005 | +0.02 | approximately 0.00 | 0.23% |

The message is small relative to `H`, but is not erased by aggregation. The best-supported description is that it reaches the aggregate contexts and changes embeddings slightly, while the final trained classifier's predictions are highly insensitive to removing it on these checkpoints. This does not establish that a differently trained cross-modal branch would be useless.

## C. Router and expert use

Router probabilities vary across nodes and across relation/target-modality cells. Per-expert node standard deviations range from approximately `0.075` to `0.389`; mean entropy ranges from `0.359` to `0.891` nats (the three-expert maximum is `log(3)`). Same-node Text-versus-Visual route JS divergence ranges from `0.020` to `0.121` nats, while target-modality comparisons within a view range from `0.209` to `0.248` nats. For example, Movies routes text targets mostly to expert 2 (top-1 share 88–95%), while visual targets favor expert 0 (48–58%). Ele-fashion routes visual targets mostly to expert 1 (92.5–98.4%). Grocery also shows different mean preferences for text and visual targets. Full per-view means, node standard deviations, entropy, top-1 shares, same-node JS and L1 comparisons are in `data/router_diagnostics.csv`.

The frozen route interventions show that node-to-route assignment matters to these fitted checkpoints. Changes below are relative to R0 Original:

| Dataset | Uniform: ΔAcc / ΔF1 (pp), flips | View-mean: ΔAcc / ΔF1 (pp), flips | Node-permuted: ΔAcc / ΔF1 (pp), flips |
|---|---:|---:|---:|
| Movies | -0.78 / -1.87, 11.91% | -0.87 / -0.44, 11.64% | -1.74 / -2.32, 13.29% |
| Grocery | -2.02 / -0.14, 7.53% | -1.70 / -0.08, 6.44% | -2.78 / -1.11, 8.14% |
| ele-fashion | -3.33 / -2.39, 7.16% | -0.88 / -0.90, 3.43% | -1.69 / -1.61, 5.07% |

Node-permuted routing is consistently worse than original routing on Validation, and uniform routing is also worse. The view-mean result is smaller but not uniformly zero. This is evidence that the fitted model uses node-conditioned route assignments; it is not a controlled estimate of the benefit of training E2 instead of E3 or E4.

Expert output RMS varies by relation and target modality. Pairwise expert cosine spans approximately `-0.128` to `0.168`, and the actual weighted expert response RMS is `0.33–0.75` of the sent-message RMS. The outputs therefore have different directions and scales, and changing routes affects predictions. Directional differences alone do not prove task-functional complementarity; that would require retrained and replicated comparisons.

## D. Physical, semantic, and skip branches

Each row below removes one addition from the original frozen forward. Accuracy and Macro-F1 changes are in percentage points; flips are the fraction of Validation predictions that changed.

| Dataset | Semantic response off: ΔAcc / ΔF1, flips | Physical response off: ΔAcc / ΔF1, flips | Intrinsic skip off: ΔAcc / ΔF1, flips |
|---|---:|---:|---:|
| Movies | -0.72 / -2.08, 6.18% | -2.31 / -6.06, 15.72% | -5.73 / -4.88, 22.62% |
| Grocery | -0.59 / -0.34, 2.81% | -4.22 / -2.18, 9.37% | -1.76 / -3.34, 7.38% |
| ele-fashion | -0.38 / -0.73, 1.77% | -0.71 / -2.00, 2.97% | -5.15 / -7.07, 10.24% |

Removing the semantic response has relatively small Validation effects, despite source contexts being distinct. Physical response removal has clearer cost on Movies and Grocery and a smaller but nonzero Macro-F1 cost on ele-fashion. Removing the intrinsic skip changes embeddings substantially (relative delta RMS `0.52–0.60`) and affects Movies and ele-fashion strongly; effects are smaller on Grocery. This shows that the direct node path is used, but does not by itself show excessive dependence: the intervention also changes the representation before final normalization and a classifier trained with the skip present.

Movies has a sizable Accuracy/Macro-F1 gap. In the original checkpoint, class 16 has F1 `0.038` with support 38 and 14 Validation predictions; class 9 has F1 `0.208` with support 37 and 11 predictions. These errors help explain why the aggregate Macro-F1 is lower than Accuracy, but classwise single-split counts are descriptive. The complete per-class results for R0 and all interventions are in `data/validation_class_metrics.csv`.

## Limits

All intervention metrics use the official Validation indices and frozen E2 parameters. They do not measure Test performance, retrained ablations, statistical significance, or whether a proposed V1.1 component will improve generalization. See `OPTIMIZATION_REVIEW.md` for the evidence available about training and initialization.
