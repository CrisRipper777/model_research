# PSCE-MAG V7B validation-only results

Branch: `exp/v7b_provenance_relative_semantic_edges`
Baseline: `bcf0d8328642dc6747f8f94f42e56b1717555d64`
Freeze: `9aa0b85bee7b5f6a5a0016bea970543d8378984d`
Runs: 45/45, protocol `unified_full_graph_nc_v1`, `task.evaluate_test=false`. No Test metrics were generated, loaded, saved, or reported.

## Candidate graph and provenance audit

All candidate tensors and legacy graph fingerprints were compared directly with the frozen V7A cache. Candidate construction used only feature arrays and the physical edge list; it did not access labels or splits.

| Dataset | Nodes | Undirected edges | Exact V7A edge/degree match | Text only | Visual only | Both |
|---|---:|---:|---|---:|---:|---:|
| Movies | 16,672 | 205,881 | True | 103,720 | 95,167 | 6,994 |
| Grocery | 17,074 | 205,557 | True | 100,991 | 96,359 | 8,207 |
| ele-fashion | 97,766 | 1,102,200 | True | 493,351 | 519,477 | 89,372 |

Degree-preserving null graphs used three successful double-edge swaps per input edge. Exact degrees and edge counts were audited for all nine dataset/seed topologies; graph seeds were 2026/2027/2028 for training seeds 42/43/44. See `data/random_graph_summary.json` for acceptance rates and changed-edge counts.

## Validation results

Cells show mean ± population SD over seeds 42/43/44. These are three repeated seeds per dataset, not nine independent datasets.

| Dataset | Variant | Val Accuracy (%) | Val Macro-F1 (%) | Best epoch | Seconds | Trainable params | Peak allocated / reserved (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | B0 `b0_v7a_original` | 56.079 ± 0.615 | 48.138 ± 1.542 | 71.3 ± 22.1 | 27.1 | 1,551,682 + 5,140 | 3.32 / 3.68 |
| Movies | B1 `b1_fixed_knn` | 56.389 ± 0.394 | 48.595 ± 0.728 | 89.7 ± 10.2 | 30.2 | 1,551,682 + 5,140 | 3.32 / 3.68 |
| Movies | B2 `b2_random_fixed` | 56.039 ± 0.593 | 48.604 ± 0.876 | 81.7 ± 7.6 | 28.2 | 1,551,682 + 5,140 | 3.32 / 3.68 |
| Movies | B3 `b3_relative_shared` | 56.279 ± 0.376 | 48.561 ± 1.301 | 89.0 ± 8.3 | 31.7 | 1,576,578 + 5,140 | 3.32 / 3.69 |
| Movies | B4 `b4_relative_modality` | 56.239 ± 0.513 | 49.559 ± 0.608 | 79.7 ± 15.2 | 29.2 | 1,576,578 + 5,140 | 3.32 / 3.69 |
| Grocery | B0 `b0_v7a_original` | 83.338 ± 0.447 | 75.824 ± 1.705 | 78.0 ± 12.7 | 29.0 | 1,551,682 + 5,140 | 3.16 / 3.58 |
| Grocery | B1 `b1_fixed_knn` | 83.397 ± 0.394 | 75.110 ± 0.869 | 69.7 ± 7.4 | 25.4 | 1,551,682 + 5,140 | 3.16 / 3.58 |
| Grocery | B2 `b2_random_fixed` | 83.202 ± 0.366 | 75.279 ± 2.047 | 78.0 ± 14.3 | 27.7 | 1,551,682 + 5,140 | 3.16 / 3.58 |
| Grocery | B3 `b3_relative_shared` | 83.446 ± 0.449 | 75.712 ± 1.824 | 73.7 ± 15.8 | 28.0 | 1,576,578 + 5,140 | 3.17 / 3.58 |
| Grocery | B4 `b4_relative_modality` | 83.416 ± 0.332 | 75.814 ± 0.966 | 73.3 ± 11.7 | 27.9 | 1,576,578 + 5,140 | 3.17 / 3.58 |
| ele-fashion | B0 `b0_v7a_original` | 87.508 ± 0.067 | 73.707 ± 0.497 | 144.7 ± 26.5 | 175.7 | 1,289,538 + 3,084 | 13.80 / 15.13 |
| ele-fashion | B1 `b1_fixed_knn` | 87.532 ± 0.107 | 74.359 ± 0.271 | 146.3 ± 19.3 | 161.3 | 1,289,538 + 3,084 | 13.79 / 15.14 |
| ele-fashion | B2 `b2_random_fixed` | 87.563 ± 0.098 | 74.721 ± 1.078 | 170.0 ± 13.7 | 183.8 | 1,289,538 + 3,084 | 13.79 / 15.14 |
| ele-fashion | B3 `b3_relative_shared` | 87.634 ± 0.071 | 74.726 ± 0.609 | 174.0 ± 3.7 | 207.5 | 1,314,434 + 3,084 | 13.82 / 15.15 |
| ele-fashion | B4 `b4_relative_modality` | 87.491 ± 0.082 | 74.117 ± 0.247 | 139.3 ± 9.8 | 173.3 | 1,314,434 + 3,084 | 13.83 / 15.14 |

## Paired comparisons

Differences are left minus right, paired within dataset and seed. Values below summarize the three seed-paired deltas per dataset; no significance test or causal claim is made.

### B1 minus B2

| Dataset | Accuracy Δ (pp), mean ± SD | Macro-F1 Δ (pp), mean ± SD | Positive seed pairs (Acc / F1) |
|---|---:|---:|---:|
| Movies | +0.350 ± 0.205 | -0.009 ± 0.394 | 3/3 / 1/3 |
| Grocery | +0.195 ± 0.028 | -0.169 ± 1.484 | 3/3 / 1/3 |
| ele-fashion | -0.031 ± 0.200 | -0.362 ± 1.115 | 1/3 / 1/3 |

### B0 minus B1

| Dataset | Accuracy Δ (pp), mean ± SD | Macro-F1 Δ (pp), mean ± SD | Positive seed pairs (Acc / F1) |
|---|---:|---:|---:|
| Movies | -0.310 ± 0.238 | -0.457 ± 0.875 | 0/3 / 1/3 |
| Grocery | -0.059 ± 0.266 | +0.715 ± 1.137 | 1/3 / 2/3 |
| ele-fashion | -0.024 ± 0.047 | -0.652 ± 0.244 | 1/3 / 0/3 |

### B3 minus B0

| Dataset | Accuracy Δ (pp), mean ± SD | Macro-F1 Δ (pp), mean ± SD | Positive seed pairs (Acc / F1) |
|---|---:|---:|---:|
| Movies | +0.200 ± 0.347 | +0.423 ± 1.025 | 1/3 / 2/3 |
| Grocery | +0.107 ± 0.138 | -0.112 ± 0.162 | 2/3 / 1/3 |
| ele-fashion | +0.126 ± 0.093 | +1.019 ± 0.847 | 3/3 / 2/3 |

### B4 minus B3

| Dataset | Accuracy Δ (pp), mean ± SD | Macro-F1 Δ (pp), mean ± SD | Positive seed pairs (Acc / F1) |
|---|---:|---:|---:|
| Movies | -0.040 ± 0.380 | +0.997 ± 0.873 | 1/3 / 3/3 |
| Grocery | -0.029 ± 0.120 | +0.102 ± 0.861 | 1/3 / 1/3 |
| ele-fashion | -0.143 ± 0.030 | -0.609 ± 0.856 | 0/3 / 1/3 |

### B4 minus B0

| Dataset | Accuracy Δ (pp), mean ± SD | Macro-F1 Δ (pp), mean ± SD | Positive seed pairs (Acc / F1) |
|---|---:|---:|---:|
| Movies | +0.160 ± 0.196 | +1.420 ± 0.940 | 2/3 / 3/3 |
| Grocery | +0.078 ± 0.183 | -0.011 ± 0.739 | 2/3 / 1/3 |
| ele-fashion | -0.017 ± 0.119 | +0.411 ± 0.508 | 1/3 / 2/3 |

## Relationship diagnostics

Per selected checkpoint, `data/semantic_diagnostics.csv` records each modality's edge-weight mean/SD/quantiles, text-visual weight correlation and absolute difference, near-bound saturation fractions, source-group weight distributions, semantic gates, semantic-to-physical and semantic-to-intrinsic RMS ratios, semantic/Raw cosine, RMS calibration factors, scorer/shared-expert gradients, and validation-only closed-path effects. These describe model behavior; they do not establish per-edge causal utility.

B3 averages two output scores before relative centering. Its common score direction is trained, while the output-head difference is unidentifiable in B3. B4 can assign modality-specific weights over the same edge index. The initialization audit verified B3/B4 scorer and common public tensors exactly; the two final output rows begin identical.

## Baseline checks, compute, and historical context

The full-graph Movies same-weight regression passed: max embedding error 1.19e-06, auxiliary-loss error 0, semantic edge summaries equal=True. CPU tests: True (exit 0); GPU preflight: True; five-variant smoke: True.

GPU preflight and formal resource maxima are in `data/preflight_summary.json` and `data/resource_profile.json`. The experiment used GPU 1 (RTX 3090); unrelated GPU 0 processes were left running.

For descriptive context only, historical V4A R0 validation Acc/F1 were Movies 56.029/47.997%, Grocery 83.094/75.766%, Fashion 87.413/74.778%; historical PIGPR-C1 RGD values were 56.389/49.698%, 83.094/77.342%, 87.324/74.536%. Protocols/configurations differ, so these are not paired comparisons.

## Recommendation and limitations

Read the five paired comparisons together with the stability and scale diagnostics. Retain the simplest variant whose validation gains repeat across seeds and datasets; treat fixed-vs-learned and real-vs-random gaps as empirical questions. If gains are small or inconsistent, prefer the simpler fixed/shared relation path. This campaign does not support causal interpretations of learned weights, and three datasets × three seeds do not justify broad statistical claims.

The protocol is full-graph node classification only. Link prediction remains unsupported because the candidate cache stores global node IDs that are incompatible with sampled LP-local IDs. No Test evaluation or hyperparameter search was performed.

## Artifacts

- Raw runs: `data/run_rows.json`
- Aggregate results: `data/summary.csv`
- Paired seed differences: `data/paired_comparisons.csv`
- Mechanism diagnostics: `data/semantic_diagnostics.csv`
- GPU resources: `data/resource_profile.json` and `.csv`
- Candidate provenance and random topology: `data/candidate_graph_summary.json`, `data/random_graph_summary.json`
