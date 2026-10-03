# IMoSI-F0 — Interaction-Mode Mixture Prototype

## Execution and provenance

- Required parent SHA: `188b940a88b1852608c9e2e25e6d8bd7013ed8b5`; branch: `exp/imosi_f0_interaction_mode_mixture`.
- Source was committed before smoke/formal execution: `0187b3c610792166a0393ddb2f031165b96b9278`. Formal runs record this source SHA.
- Validation-only NC campaign: 27/27 complete (B/G/M × three datasets × seeds 42/43/44). The three seeds are repeated model/training seeds on each fixed split, not separate dataset splits.
- NC uses `evaluate_test=false`, `development_no_test=true`; no NC test metrics, test evaluation, split changes, or formal LP runs. Only the requested B/M two-batch LP smoke was run.
- Means and population SDs (`ddof=0`) describe three paired seeds; no pseudo-IID p-values or hard minimum-gain threshold are used.

## Validation performance

Accuracy and Macro-F1 are percent; CE is in task units. Cells are mean ± population SD over seeds.

| Dataset | B Acc | G Acc | M Acc | B F1 | G F1 | M F1 | B CE | G CE | M CE |
|---|---|---|---|---|---|---|---|---|---|
| Movies | 56.68 ± 0.05 | 56.57 ± 0.07 | 56.69 ± 0.21 | 48.92 ± 1.39 | 49.52 ± 0.31 | 48.70 ± 0.96 | 1.3642 ± 0.0330 | 1.3726 ± 0.0140 | 1.3234 ± 0.0067 |
| Grocery | 82.96 ± 0.19 | 82.87 ± 0.14 | 83.01 ± 0.07 | 76.44 ± 0.27 | 75.65 ± 0.56 | 75.58 ± 0.46 | 0.6493 ± 0.0077 | 0.6495 ± 0.0177 | 0.6464 ± 0.0081 |
| ele-fashion | 87.31 ± 0.14 | 87.40 ± 0.08 | 87.50 ± 0.15 | 73.90 ± 0.65 | 74.94 ± 0.45 | 74.51 ± 0.43 | 0.4095 ± 0.0026 | 0.4219 ± 0.0039 | 0.4157 ± 0.0085 |

Run-level metrics: [performance_by_run.csv](data/performance_by_run.csv); seed summary: [performance_summary.csv](data/performance_summary.csv).

## Paired retrained comparisons

Accuracy and Macro-F1 deltas are percentage points. Accuracy directions are positive/negative/tied seeds. M−B is primary; G−B and M−G are secondary.

| Dataset | Comparison | Accuracy Δ mean ± SD | Acc +/−/= | Macro-F1 Δ mean ± SD | CE Δ mean ± SD |
|---|---|---|---|---|---|
| Movies | M-B | 0.010 ± 0.163 | 2/1/0 | -0.219 ± 0.427 | -0.0407 ± 0.0264 |
| Movies | G-B | -0.110 ± 0.028 | 0/3/0 | 0.600 ± 1.132 | 0.0085 ± 0.0395 |
| Movies | M-G | 0.120 ± 0.136 | 2/1/0 | -0.819 ± 0.707 | -0.0492 ± 0.0179 |
| Grocery | M-B | 0.049 ± 0.241 | 2/1/0 | -0.856 ± 0.727 | -0.0029 ± 0.0083 |
| Grocery | G-B | -0.088 ± 0.124 | 1/2/0 | -0.782 ± 0.822 | 0.0002 ± 0.0194 |
| Grocery | M-G | 0.137 ± 0.159 | 2/1/0 | -0.074 ± 0.113 | -0.0031 ± 0.0251 |
| ele-fashion | M-B | 0.194 ± 0.225 | 2/1/0 | 0.606 ± 1.015 | 0.0062 ± 0.0099 |
| ele-fashion | G-B | 0.092 ± 0.217 | 2/1/0 | 1.037 ± 0.687 | 0.0125 ± 0.0052 |
| ele-fashion | M-G | 0.102 ± 0.177 | 2/1/0 | -0.431 ± 0.830 | -0.0062 ± 0.0047 |

Full paired rows: [paired_delta_by_run.csv](data/paired_delta_by_run.csv). M−G is the retrained evidence for node-specific routing beyond modality-global mode preference.

## Routing diagnostics

Selected G/M checkpoints; values are averaged over three seeds for each dataset/modality. Routes are architecture-defined Preserve/Self/Paired mixtures, not latent ground-truth roles.

| Variant | Dataset | Modality | Mean π preserve | Mean π self | Mean π paired | Entropy | Node-logit RMS |
|---|---|---|---|---|---|---|---|
| G | Movies | text | 0.5960 | 0.2009 | 0.2031 | 0.9546 | 0.0000 |
| G | Movies | visual | 0.5818 | 0.2172 | 0.2010 | 0.9692 | 0.0000 |
| G | Grocery | text | 0.5953 | 0.2005 | 0.2041 | 0.9553 | 0.0000 |
| G | Grocery | visual | 0.5855 | 0.2121 | 0.2024 | 0.9655 | 0.0000 |
| G | ele-fashion | text | 0.5769 | 0.2224 | 0.2006 | 0.9738 | 0.0000 |
| G | ele-fashion | visual | 0.5875 | 0.1989 | 0.2137 | 0.9634 | 0.0000 |
| M | Movies | text | 0.2506 | 0.5416 | 0.2078 | 0.9529 | 0.8565 |
| M | Movies | visual | 0.0524 | 0.8289 | 0.1187 | 0.4557 | 1.9581 |
| M | Grocery | text | 0.2501 | 0.5544 | 0.1954 | 0.9332 | 0.8938 |
| M | Grocery | visual | 0.0328 | 0.9268 | 0.0404 | 0.2543 | 2.3816 |
| M | ele-fashion | text | 0.3341 | 0.5383 | 0.1276 | 0.7086 | 1.5514 |
| M | ele-fashion | visual | 0.3547 | 0.3181 | 0.3273 | 0.8597 | 1.0028 |

Per-seed mean/std, p10/p50/p90, entropy, argmax fractions, and M-vs-global route difference are in [routing_diagnostics.csv](data/routing_diagnostics.csv). Nonuniform routes alone do not imply task value.

## Effective expert contribution

The table averages direction/checkpoint summaries; full per dataset/seed/modality values are in [expert_contribution.csv](data/expert_contribution.csv). Raw deltas and probability-weighted contributions are reported separately.

| Variant | RMS δ self | RMS δ pair | RMS πself·δself | RMS πpair·δpair | Self/G | Pair/G | cos(self_eff,G) | cos(pair_eff,G) |
|---|---|---|---|---|---|---|---|---|
| G | 0.4144 | 0.3687 | 0.0872 | 0.0755 | 0.0872 | 0.0755 | -0.3305 | -0.2604 |
| M | 0.3882 | 0.3631 | 0.2203 | 0.0832 | 0.2203 | 0.0832 | -0.3762 | -0.2605 |

## GPR coefficients

Mean selected-checkpoint coefficients over the nine dataset-seed cells per variant. These only describe possible backbone co-adaptation.

| Variant | c0 | c1 | c2 | c3 |
|---|---|---|---|---|
| B | 0.23161 | 0.06291 | 0.63115 | -0.08787 |
| G | 0.23298 | 0.06005 | 0.62797 | -0.09065 |
| M | 0.21799 | 0.09553 | 0.64933 | -0.07036 |

Per-run values: [gpr_diagnostics.csv](data/gpr_diagnostics.csv).

## M checkpoint globalize-router intervention

This intervention sets node-specific router deltas to zero while retaining the selected M backbone, experts, and modality-global logits. It measures checkpoint reliance only; retrained M−G remains the adaptive-routing test.

| Dataset | Accuracy Δ vs M | Acc +/−/= | Macro-F1 Δ | CE Δ |
|---|---|---|---|---|
| Movies | -2.170 ± 0.837 | 0/3/0 | -7.298 ± 4.574 | 0.0541 ± 0.0210 |
| Grocery | -0.547 ± 0.256 | 0/3/0 | -0.625 ± 0.234 | 0.0094 ± 0.0063 |
| ele-fashion | -4.337 ± 2.644 | 0/3/0 | -7.519 ± 3.990 | 0.1595 ± 0.0671 |

Rows: [intervention_by_run.csv](data/intervention_by_run.csv); seed summary: [intervention_summary.csv](data/intervention_summary.csv).

## Interpretation

Mixed, low-magnitude, dataset-dependent evidence (closest to Case C); the results do not establish a general adaptive-routing advantage. M's mean Accuracy deltas are positive versus B in all datasets (+0.010, +0.049, +0.194 pp) and versus G (+0.120, +0.137, +0.102 pp), but each comparison is positive for only two of three paired seeds, and the mean deltas are no larger than their paired-seed population SDs. Macro-F1 does not follow Accuracy consistently: M−G is negative in all three datasets, while M−B is negative on Movies and Grocery and positive on ele-fashion. CE favors M over G in all three datasets, and favors M over B on Movies/Grocery but not ele-fashion. This is a tradeoff, not a robust win.

Do not infer efficacy from route entropy, expert frequency, route variance, or globalize-router score alone. The paired retrained M−B and M−G results take precedence.

## Final self-audit (30 items)

1. **Started from `188b940…`?** Yes; required base is `188b940a88b1852608c9e2e25e6d8bd7013ed8b5`.
2. **Used another experiment branch?** No merge/cherry-pick from other experiment branches.
3. **Was source committed before formal runs?** Yes; source commit is `0187b3c610792166a0393ddb2f031165b96b9278`.
4. **Which source SHA did formal runs use?** `0187b3c610792166a0393ddb2f031165b96b9278` is recorded in every formal run.
5. **Ran/read NC test?** No test evaluation or `test_*` metric. The dev-only task masks test labels and computes validation metrics from train/validation splits; full-graph features follow the existing transductive protocol.
6. **Changed splits?** No; the three fixed paths above were used.
7. **Does B regress to PCRR-v1/RGD?** Yes, at `atol=1e-6` after matching shared state; test passed.
8. **Do F0 modules preserve downstream RNG state?** Yes; global CPU RNG and next-classifier tensors match PCRR-v1 under identical starting RNG.
9. **Are B/G/M initial functions identical?** Yes, elementwise in eval and after train-mode RNG reset.
10. **Are Self/Paired heads zero-initialized?** Yes; exact zero weights and biases.
11. **Does M start with G routing?** Yes; zero `router_out` gives zero node logits and routes equal G.
12. **Are B/G/M parameter count and initialization matched?** Yes; identical count, state layout, and same-seed tensors.
13. **Did LP smoke preserve its protocol?** See [smoke_status.json](smoke_status.json): LinkNeighborLoader, `[5,5,5]`, global-eid positive-edge removal, sampled forward/backward, validation, checkpoint; test disabled.
14. **How does M−B perform?** Dataset means, SDs, and seed directions are in the paired table above; interpret by dataset.
15. **How does G−B perform?** Shown separately in the paired table; it tests global mode composition.
16. **How does M−G perform?** Shown separately; it is the retrained node-routing increment.
17. **Is global mixture sufficient?** Answer depends on G−B and M−G paired signs; do not infer from router diagnostics.
18. **Does adaptive routing vary by node?** Mean/std/quantiles and M node-logit RMS are reported per dataset, seed, and modality.
19. **Does node variation coincide with retrained M−G gain?** Compare the direct M−G table; route variation itself is not evidence of utility.
20. **What are mean Preserve/Self/Paired routes?** Reported by G/M, dataset, and modality in the routing table/CSV.
21. **Do Text and Visual routes differ?** Both modality-specific distributions are shown; they are not treated as semantic role labels.
22. **How large are effective expert contributions?** Probability-weighted RMS and relative-to-G values are in the contribution table/CSV.
23. **Do route probabilities match functional contribution?** Both are reported independently; high route probability alone need not mean a large correction.
24. **Does globalize-router alter the selected M checkpoint?** Its validation metric deltas are in the intervention table/CSV.
25. **Does the intervention agree with retrained M−G?** It is checkpoint reliance evidence and is not substituted for M−G.
26. **Are dataset regimes different?** Compare each dataset's paired signs and routing; no universal regime is presumed.
27. **Did GPR coefficients co-adapt?** Coefficients are shown descriptively; no GPR attribution is reopened.
28. **Which variant is currently most defensible?** See recommendation below, based on retrained paired metrics.
29. **Should this direction continue?** See conservative recommendation; no autonomous follow-on stage is run.
30. **Any claim beyond evidence?** No; validation-only, fixed-split, three-seed descriptive evidence.

## Recommendation

**Recommendation: B (Preserve-only) as the conservative default.** M has small positive mean Accuracy deltas, but its seed directions are 2/1 in every dataset, the gains are comparable to the paired-seed SDs, and Macro-F1 is lower than G in every dataset. The router clearly varies by node and the selected checkpoints rely on that variation under `globalize_router`, but those diagnostics do not override the small retrained M−G gains. If a mixture variant must be carried forward for a separately approved study, prefer the simpler G over M because their Accuracy results are close and M does not improve Macro-F1 over G. G itself has no consistent advantage over B, so this F0 does not justify continuing the interaction-mode mixture without human review.
