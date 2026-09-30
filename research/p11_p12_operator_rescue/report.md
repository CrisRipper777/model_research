# P1.2 fixed-operator rescue results

Three deterministic propagation primitives were evaluated on the same graph, validation-directed edge set, and frozen semantic substrate H0 per dataset×seed. Only a new linear classifier was trained for each primitive. `U_raw = CE_removed − CE_full`; positive means deletion raises validation CE and is therefore useful under that fixed operator. The original degree denominator was retained after message deletion. No test or link-prediction evaluation was run.

## Dataset × modality rescue summary

Values are means across seeds 42–44; parentheses show population SD across the three seed-level estimates. Rescue is conditioned on `U_smooth < 0`. The unrescued column means both tested alternatives have utility ≤0 for a smooth-harmful message.

| Dataset | Modality | Smooth harmful | AbsDiff rescue | Product rescue | Any rescue | Unrescued by tested primitives | AbsDiff enrichment | Product enrichment | Smooth T/V disagreement | AbsDiff T/V disagreement | Product T/V disagreement |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | Text | 0.399 (0.047) | 0.539 (0.039) | 0.329 (0.007) | 0.739 (0.028) | 0.261 (0.028) | 1.352 (0.067) | 0.529 (0.010) | 0.422 | 0.660 | 0.442 |
| Movies | Visual | 0.356 (0.010) | 0.591 (0.017) | 0.387 (0.011) | 0.763 (0.016) | 0.237 (0.016) | 0.971 (0.024) | 0.618 (0.026) | 0.422 | 0.660 | 0.442 |
| Grocery | Text | 0.206 (0.002) | 0.509 (0.014) | 0.375 (0.032) | 0.704 (0.029) | 0.296 (0.029) | 0.925 (0.012) | 0.466 (0.034) | 0.213 | 0.689 | 0.235 |
| Grocery | Visual | 0.233 (0.005) | 0.474 (0.030) | 0.411 (0.033) | 0.680 (0.033) | 0.320 (0.033) | 0.931 (0.086) | 0.530 (0.034) | 0.213 | 0.689 | 0.235 |
| ele-fashion | Text | 0.191 (0.005) | 0.547 (0.005) | 0.338 (0.019) | 0.717 (0.009) | 0.283 (0.009) | 1.438 (0.040) | 0.433 (0.020) | 0.685 | 0.451 | 0.339 |
| ele-fashion | Visual | 0.714 (0.039) | 0.453 (0.058) | 0.567 (0.031) | 0.776 (0.034) | 0.224 (0.034) | 1.012 (0.017) | 0.872 (0.036) | 0.685 | 0.451 | 0.339 |

The output CSV also gives seed-level counts, joint prevalences `P(U_smooth<0 and U_alt>0)`, and overall alternative-positive fractions. Rescue enrichment is descriptive: values above/below one indicate enrichment/depletion among smooth-harmful messages relative to all edges.

## Sign-transition matrices

The four cells are shares of all aligned physical messages: S+/A+, S+/A−, S−/A+, and S−/A−. The S−/A+ cell is the rescue quadrant. `operator_transition_matrix.csv` contains every dataset×seed×modality comparison and the dataset-level three-seed mean and sample SD.

### Movies

| Modality | Alternative | S+/A+ | S+/A− | S−/A+ | S−/A− |
|---|---|---:|---:|---:|---:|
| Text | absdiff | 0.182 | 0.419 | 0.217 | 0.182 |
| Text | product | 0.490 | 0.111 | 0.132 | 0.268 |
| Visual | absdiff | 0.398 | 0.246 | 0.210 | 0.146 |
| Visual | product | 0.490 | 0.155 | 0.138 | 0.218 |

### Grocery

| Modality | Alternative | S+/A+ | S+/A− | S−/A+ | S−/A− |
|---|---|---:|---:|---:|---:|
| Text | absdiff | 0.445 | 0.348 | 0.105 | 0.101 |
| Text | product | 0.727 | 0.067 | 0.077 | 0.129 |
| Visual | absdiff | 0.401 | 0.366 | 0.111 | 0.122 |
| Visual | product | 0.678 | 0.089 | 0.096 | 0.137 |

### ele-fashion

| Modality | Alternative | S+/A+ | S+/A− | S−/A+ | S−/A− |
|---|---|---:|---:|---:|---:|
| Text | absdiff | 0.276 | 0.533 | 0.104 | 0.086 |
| Text | product | 0.717 | 0.093 | 0.064 | 0.126 |
| Visual | absdiff | 0.123 | 0.163 | 0.325 | 0.389 |
| Visual | product | 0.244 | 0.042 | 0.405 | 0.309 |

## Probe validation sanity

Validation metrics select each linear head and are shown only as a sanity check, not as the primary rescue evidence. Values below are seed means.

| Dataset | Operator | Validation accuracy | Macro-F1 | CE |
|---|---|---:|---:|---:|
| Movies | smooth | 0.543 | 0.448 | 1.376 |
| Movies | absdiff | 0.519 | 0.407 | 1.453 |
| Movies | product | 0.535 | 0.447 | 1.395 |
| Grocery | smooth | 0.809 | 0.731 | 0.683 |
| Grocery | absdiff | 0.775 | 0.694 | 0.811 |
| Grocery | product | 0.795 | 0.719 | 0.749 |
| ele-fashion | smooth | 0.865 | 0.663 | 0.415 |
| ele-fashion | absdiff | 0.865 | 0.665 | 0.420 |
| ele-fashion | product | 0.863 | 0.665 | 0.420 |

Smooth has the strongest mean validation probe in Movies and Grocery; the three operators are close in ele-fashion. This sanity comparison does not show that a multi-operator model or router would outperform scalar message handling.

## Cross-seed stability and operator specificity

Use the seed rows in `operator_rescue_summary.csv` to assess whether rescue appears in all three seeds or is driven by a subset. Compare conditional rescue with overall positivity and enrichment: enrichment near one is consistent with an alternative being broadly positive rather than especially useful on smooth-harmful messages; enrichment above one is descriptive evidence of edge-specific complementarity. These two fixed alternatives are a small probe set, not a final operator bank.

## Assessment and next step



**P1.2 status: SUPPORTED.** In every dataset×modality group, `R_any` is 0.68–0.78 on average across seeds, with the corresponding seed-level values consistently nontrivial. Each alternative separately rescues a sizeable smooth-harmful subset in all three seeds. This supports the bounded claim that message usefulness depends on the propagation function tested; it does not establish that these three primitives form a final operator bank or that routing improves a model.

For the research question, advance from `Propagation Utility Heterogeneity` to the narrower `Propagation Function Heterogeneity` claim: the same physical message can change CE-utility sign under fixed relation transforms. Keep the claim explicitly limited to smooth, absdiff, and product in validation probes.

**Next step: scalar gating/blocking first; do not start operator-routing V0 yet.** Rescue enrichment is mixed: absdiff is enriched in Movies Text and ele-fashion Text, near one in several other groups, and below one in Grocery Text/Visual and Movies Visual; product is below one in five of six groups. The alternatives often help broadly rather than selectively recovering smooth-harmful edges, and smooth has the strongest mean validation probe performance in Movies and Grocery. The result justifies keeping operator routing as a later controlled comparison, while a scalar utility-handling baseline is the more interpretable next step. No model is implemented in this stage.

## Limitations

1. Validation accuracy selects each new linear head and validation labels are also the targets for utility counterfactuals; this is a controlled probe, not independent confirmation.
2. Smooth, absolute-difference, and elementwise-product operations are fixed examples. Results do not establish a complete or optimal operator bank.
3. Directed edges share destination nodes. Seed summaries are descriptive; no p-values or independent-edge claims are made.
4. All experiments are transductive NC over the supplied graph and fixed features. Test labels/indices and LP were not accessed.
