# SAIR-G0 — Semantic-Anchor / Structural-Response Decoupled Readout Screen

Generated: 2026-10-03T21:18:53+00:00

## Experiment summary

- Base SHA: `997b89b8b0e59ca2da9654d07cc184cc1ce14169`; source SHA: `6acbc20808ad0534ae7c0b54a026f1b50c7aabb6`; branch: `exp/sair_g0_anchor_response_readout`.
- Validation-only campaign: 27/27 completed; seeds 42/43/44 are model/training seeds on fixed splits.
- No NC test metrics were requested or accepted by the runner. No split changes, Toys, Reddit-S, or formal LP runs were made.
- Readout variants: B = RGD base; C = generic adapter on G; D = shared adapter on raw P/R response provenance.

## Smoke

| NC variant | status | Val Acc % | Macro-F1 % | CE | selected pair_up norm |
|---|---|---|---|---|---|
| B | complete | 32.93 | 2.48 | 3.7617 | 0.0 |
| C | complete | 32.93 | 2.48 | 3.7761 | 0.12793375551700592 |
| D | complete | 32.93 | 2.48 | 3.7759 | 0.12794077396392822 |

Movies seed 42 used two epochs for each NC smoke run. B retained exact zero `pair_up`; C and D updated the initially zero adapter. Initial B/C/D eval equality and RNG-reset train equality are covered by the committed tests.

| LP variant | status | LinkNeighborLoader | [5,5,5] | removed positive edges/epoch | forward/backward + validation | checkpoint |
|---|---|---|---|---|---|---|
| B | complete | True | True | 2306, 2214 | True | True |
| D | complete | True | True | 2306, 2214 | True | True |

LP smoke was limited to sports-copurchase B/D, two epochs, two train batches, fanouts [5,5,5], with test evaluation disabled. It is a protocol check only.

## Validation performance

Values are mean ± population SD over three training seeds. Accuracy and Macro-F1 are percentages; CE is cross-entropy.

| Dataset | B Acc | C Acc | D Acc | B Macro-F1 | C Macro-F1 | D Macro-F1 | B CE | C CE | D CE |
|---|---|---|---|---|---|---|---|---|---|
| Movies | 56.64 ± 0.07 | 56.68 ± 0.06 | 56.25 ± 0.23 | 48.58 ± 0.69 | 48.59 ± 0.91 | 47.97 ± 0.54 | 1.3497 ± 0.0050 | 1.3702 ± 0.0163 | 1.3361 ± 0.0212 |
| Grocery | 82.96 ± 0.15 | 82.83 ± 0.07 | 82.74 ± 0.16 | 76.10 ± 0.14 | 75.85 ± 0.95 | 75.54 ± 0.42 | 0.6597 ± 0.0319 | 0.6557 ± 0.0092 | 0.6707 ± 0.0192 |
| ele-fashion | 87.35 ± 0.11 | 87.49 ± 0.05 | 87.37 ± 0.05 | 74.51 ± 0.28 | 75.06 ± 0.69 | 73.89 ± 0.19 | 0.4088 ± 0.0075 | 0.4264 ± 0.0112 | 0.4100 ± 0.0080 |

## Paired comparisons

Accuracy deltas are percentage points. Seed counts are positive/negative/tie. Comparisons are paired by dataset and seed. No pseudo-IID p-values or fixed performance threshold are used.

| Dataset | Comparison | Pairs | Accuracy Δ pp mean ± SD | +/−/= | Macro-F1 Δ pp mean ± SD | CE Δ mean ± SD |
|---|---|---|---|---|---|---|
| Movies | D-B | 3 | -0.390 ± 0.301 | 1/2/0 | -0.609 ± 0.744 | -0.0136 ± 0.0242 |
| Movies | D-C | 3 | -0.430 ± 0.287 | 0/3/0 | -0.622 ± 1.411 | -0.0341 ± 0.0320 |
| Movies | C-B | 3 | 0.040 ± 0.014 | 3/0/0 | 0.013 ± 1.034 | 0.0205 ± 0.0113 |
| Grocery | D-B | 3 | -0.215 ± 0.173 | 1/2/0 | -0.565 ± 0.369 | 0.0110 ± 0.0152 |
| Grocery | D-C | 3 | -0.088 ± 0.157 | 1/2/0 | -0.311 ± 1.334 | 0.0150 ± 0.0152 |
| Grocery | C-B | 3 | -0.127 ± 0.077 | 0/3/0 | -0.254 ± 0.964 | -0.0040 ± 0.0246 |
| ele-fashion | D-B | 3 | 0.017 ± 0.076 | 2/1/0 | -0.624 ± 0.115 | 0.0012 ± 0.0020 |
| ele-fashion | D-C | 3 | -0.123 ± 0.030 | 0/3/0 | -1.176 ± 0.498 | -0.0164 ± 0.0084 |
| ele-fashion | C-B | 3 | 0.140 ± 0.063 | 3/0/0 | 0.553 ± 0.431 | 0.0176 ± 0.0102 |
| ALL | D-B | 9 | -0.196 ± 0.264 | 4/5/0 | -0.599 ± 0.485 | -0.0005 ± 0.0194 |
| ALL | D-C | 9 | -0.213 ± 0.244 | 1/8/0 | -0.703 ± 1.211 | -0.0118 ± 0.0292 |
| ALL | C-B | 9 | 0.018 ± 0.124 | 6/3/0 | 0.104 ± 0.917 | 0.0114 ± 0.0199 |

## Anchor-response geometry

Cosines are computed over flattened checkpoint tensors, summarized across dataset × seed observations by modality/variant. They describe geometry and do not imply beneficial or harmful information.

| Variant | Modality | RMS(R)/RMS(P) | cos(P,R) | cos(P,G) | cos(R,G) | RMS(R)/RMS(c0P) |
|---|---|---|---|---|---|---|
| B | text | 0.458 ± 0.067 | 0.858 ± 0.030 | 0.957 ± 0.019 | 0.940 ± 0.006 | 2.044 ± 0.516 |
| B | visual | 0.403 ± 0.037 | 0.806 ± 0.056 | 0.938 ± 0.027 | 0.928 ± 0.011 | 1.794 ± 0.398 |
| C | text | 0.469 ± 0.069 | 0.861 ± 0.028 | 0.958 ± 0.019 | 0.940 ± 0.007 | 2.136 ± 0.567 |
| C | visual | 0.408 ± 0.036 | 0.805 ± 0.059 | 0.937 ± 0.028 | 0.927 ± 0.012 | 1.842 ± 0.406 |
| D | text | 0.593 ± 0.023 | 0.876 ± 0.026 | 0.960 ± 0.013 | 0.941 ± 0.009 | 2.951 ± 0.175 |
| D | visual | 0.526 ± 0.064 | 0.818 ± 0.049 | 0.933 ± 0.026 | 0.933 ± 0.003 | 2.623 ± 0.352 |

The per-run values, including RMS(P), RMS(R), and RMS(c0P), are in `data/anchor_response_diagnostics.csv`.

Across selected D checkpoints, mean RMS(P) is 1.002 for Text and 0.999 for Visual; mean RMS(R) is 0.594 and 0.526, respectively. Thus mean RMS(R)/RMS(P) is 0.593 for Text and 0.526 for Visual. RMS(R)/RMS(c0P) is larger than one in both modalities (2.951 Text, 2.623 Visual). Geometry varies by modality and dataset: D cos(P,R) ranges from 0.848 to 0.910 across Text dataset means and from 0.781 to 0.888 across Visual dataset means. These values describe scale and overlap only.

## Adapter and GPR diagnostics

| Variant | Modality | RMS(δ)/RMS(G) | cos(δ,G) | cos(δ,P) | cos(δ,R) | pair hidden RMS | pair_up norm | pair_down norm |
|---|---|---|---|---|---|---|
| C | text | 0.368 ± 0.119 | -0.487 ± 0.188 | — | — | 1.528 ± 0.429 | 2.017 ± 0.444 | 6.101 ± 0.370 |
| C | visual | 0.243 ± 0.060 | -0.309 ± 0.026 | — | — | 1.226 ± 0.300 | 2.017 ± 0.444 | 6.101 ± 0.370 |
| D | text | 0.467 ± 0.115 | -0.404 ± 0.345 | -0.352 ± 0.327 | -0.399 ± 0.326 | 1.909 ± 0.397 | 1.935 ± 0.344 | 6.749 ± 0.638 |
| D | visual | 0.274 ± 0.057 | -0.266 ± 0.108 | -0.225 ± 0.096 | -0.270 ± 0.100 | 1.360 ± 0.209 | 1.935 ± 0.344 | 6.749 ± 0.638 |

For D, per-run `cos(δ,P)` and `cos(δ,R)` are in `data/adapter_diagnostics.csv`; corresponding values for C are intentionally blank. Adapter activity is descriptive, not performance evidence.

On average the D correction is least anti-aligned with P, but its mean cosines with P, R, and G are negative in both modalities. Its direction varies by dataset: Text cos(δ,G) means are -0.718 on Movies, -0.568 on Grocery, and 0.074 on ele-fashion. This describes correction geometry and does not establish utility.

| Dataset | Variant | c0 | c1 | c2 | c3 |
|---|---|---|---|---|---|
| Movies | B | 0.23385 ± 0.00063 | 0.07293 ± 0.00188 | 0.63348 ± 0.00166 | -0.08679 ± 0.00149 |
| Movies | C | 0.23583 ± 0.00381 | 0.06993 ± 0.00372 | 0.62963 ± 0.00559 | -0.09038 ± 0.00558 |
| Movies | D | 0.21671 ± 0.00424 | 0.10252 ± 0.00279 | 0.66904 ± 0.00423 | -0.05704 ± 0.00417 |
| Grocery | B | 0.19532 ± 0.01332 | 0.08327 ± 0.01556 | 0.67670 ± 0.01667 | -0.04368 ± 0.01595 |
| Grocery | C | 0.18722 ± 0.00779 | 0.08801 ± 0.00923 | 0.68723 ± 0.00857 | -0.03306 ± 0.00810 |
| Grocery | D | 0.18420 ± 0.00526 | 0.10517 ± 0.00453 | 0.69377 ± 0.00330 | -0.02702 ± 0.00378 |
| ele-fashion | B | 0.26405 ± 0.00771 | 0.03115 ± 0.01409 | 0.58524 ± 0.01305 | -0.13108 ± 0.01165 |
| ele-fashion | C | 0.26076 ± 0.00902 | 0.04132 ± 0.01598 | 0.58953 ± 0.01489 | -0.12754 ± 0.01355 |
| ele-fashion | D | 0.20359 ± 0.00214 | 0.13916 ± 0.00594 | 0.66943 ± 0.00305 | -0.05269 ± 0.00242 |

Relative to B, C makes comparatively small coefficient shifts. D consistently lowers mean c0 and raises c1/c2 across the three datasets, while c3 becomes less negative; this is compatible with backbone co-adaptation under D and remains descriptive.

## Conservative diagnosis and recommendation

The pooled D−B Accuracy delta is -0.196 ± 0.264 pp (4 positive, 5 negative paired seeds), while D−C is -0.213 ± 0.244 pp (1 positive, 8 negative). D−B Macro-F1 is negative in all three dataset means; D−C Accuracy and Macro-F1 are negative in all three. D lowers CE on some comparisons, including a pooled D−B CE delta of -0.0005 ± 0.0194, so CE does not mirror Macro-F1. C−B Accuracy is nearly neutral at +0.018 ± 0.124 pp pooled, with 6 positive and 3 negative seeds; its Macro-F1 is slightly positive on average but variable, and its CE is higher by 0.0114 ± 0.0199.

The dataset pattern is not a provenance win: D−B mean Accuracy is negative on Movies and Grocery and near neutral on ele-fashion, while D−C is negative in every dataset. On ele-fashion specifically, D−B Accuracy is +0.017 pp but Macro-F1 is -0.624 pp. This supports a conservative no-value conclusion for the provenance-aware architecture in this screen, while recognizing the three-seed uncertainty.

B, the unchanged RGD readout, is the conservative recommendation unless a clear dataset-specific use case is identified.

No. Stop new core-module development pending human review; these results do not motivate G1 frequency decomposition.

`R = Q - c0P` is reported only as a structure-induced response. The experiment does not show that it is pure smoothing, low-frequency, or heterophilous information. Accuracy/F1/CE conflicts and per-seed variation should be reviewed directly in the CSVs.

## Self-audit (30 items)

1. Started strictly from `997b89b8b0e59ca2da9654d07cc184cc1ce14169`: **yes**.
2. Used another experiment branch: **no**; branch ancestry is the required F0 SHA.
3. Source committed before smoke/formal: **yes**, source SHA `6acbc20808ad0534ae7c0b54a026f1b50c7aabb6` was fixed in the manifest before running.
4. Formal source SHA: `6acbc20808ad0534ae7c0b54a026f1b50c7aabb6`.
5. NC test read/run: **no**; runner requires validation-only flags and rejects test metrics.
6. Dataset split modified: **no**.
7. B regression to PCRR-v1/RGD: **yes**, output and proposal/embedding tests pass at atol 1e-6.
8. Module construction preserves PCRR base RNG compatibility: **yes**, CPU RNG state and shared state tensors are bitwise matched by tests.
9. B/C/D parameter counts, state layouts, and same-seed initialization matched: **yes**, tested.
10. Initial B/C/D functions identical: **yes**, eval and RNG-reset train tests.
11. R equals Q−c0P and explicit k≥1 sum: **yes**, small and StreamingRawGPR synthetic tests pass at atol 1e-6.
12. D uses raw R without separate normalization: **yes**.
13. C adapter input is only G: **yes**, its exact pair layout is [G,G,0,G*G].
14. D is modality-local: **yes**, text adapter correction is invariant to changes in visual input.
15. D−B Accuracy/F1/CE: see `D-B` rows in the paired table; pooled Accuracy `-0.196 ± 0.264` pp, Macro-F1 `-0.599 ± 0.485` pp, CE `-0.0005 ± 0.0194`.
16. D−C: pooled Accuracy `-0.213 ± 0.244` pp, Macro-F1 `-0.703 ± 1.211` pp, CE `-0.0118 ± 0.0292`; dataset rows above.
17. C−B: pooled Accuracy `0.018 ± 0.124` pp, Macro-F1 `0.104 ± 0.917` pp, CE `0.0114 ± 0.0199`.
18. Evidence provenance adds value beyond generic adapter: not established; D was not consistently better than both controls.
19. RMS(R)/RMS(P): for D, mean is 0.593 Text and 0.526 Visual; full per-dataset values are in the geometry table/CSV.
20. Is cos(P,R) dataset/modality dependent: **yes**; for D the dataset means span 0.848–0.910 for Text and 0.781–0.888 for Visual.
21. D correction magnitude: mean RMS(δ)/RMS(G) is 0.467 Text and 0.274 Visual.
22. Is δ closer to P, R, or G: pooled D cosines are negative to all three; P is least anti-aligned (Text: P -0.352, R -0.399, G -0.404; Visual: P -0.225, G -0.266, R -0.270). This does not establish utility.
23. GPR co-adaptation: vs B, D lowers mean c0 and raises c1/c2 in all datasets, while c3 becomes less negative; C shifts are smaller.
24. Dataset-specific regime: D−B Accuracy is negative for Movies/Grocery and near zero or slightly positive for ele-fashion; D−C is negative in each dataset.
25. Accuracy/F1/CE conflict: **yes**; ele-fashion D−B Accuracy is +0.017 pp while Macro-F1 is -0.624 pp, and pooled D−B CE is slightly lower despite negative Macro-F1.
26. Current recommendation: B, the unchanged RGD readout, is the conservative recommendation unless a clear dataset-specific use case is identified.
27. Worth entering G1 frequency decomposition: No. Stop new core-module development pending human review; these results do not motivate G1 frequency decomposition.
28. If not, stop new core modules: yes, await human review before further core-module work.
29. LP smoke protocol correctness: **yes**; each B/D run used LinkNeighborLoader and [5,5,5], removed 2306 and 2214 positive message edges over two epochs, ran validation MRR/backpropagation, and saved a checkpoint; test was disabled.
30. Claims beyond direct evidence: none intended; R is not labeled as heterophilous or frequency-pure, and diagnostics are not treated as performance evidence.
