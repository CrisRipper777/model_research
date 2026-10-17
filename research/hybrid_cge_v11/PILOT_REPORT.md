# Hybrid-CGE V1.1 Pilot Report

Validation-only: Test labels were masked before trainer entry and no Test metrics were computed.
Implementation Freeze Commit: affc45b94f4298eb4c524e6fd7d9f55591481c8a; formal training source: affc45b94f4298eb4c524e6fd7d9f55591481c8a.

## Protocol and historical comparability

All 18 runs use Movies, Grocery and ele-fashion; H1/H2; seeds 42/43/44; the official full-graph NC protocol; identical H1/H2 initial model/head hashes per Dataset×Seed; and the same split, feature, physical graph, V7B.1 cache and candidate fingerprints.
Historical pairing eligibility: CG-CGE B1: 3/3 datasets paired; View-CGE A2: 3/3 datasets paired. Only fully matched three-seed blocks appear in paired summaries.

## Best-Accuracy Validation metrics

| Dataset | Model | Accuracy % | Macro-F1 % | CE |
|---|---|---:|---:|---:|
| Movies | H1_hybrid | 56.339 ± 0.375 | 49.869 ± 0.722 | 1.3732 ± 0.0666 |
| Movies | H2_hybrid_plus | 56.779 ± 0.436 | 49.780 ± 1.173 | 1.3497 ± 0.0226 |
| Movies | CG-CGE B1 | 56.519 ± 0.360 | 49.611 ± 1.226 | 1.3753 ± 0.0446 |
| Movies | View-CGE A2 | 55.959 ± 0.241 | 47.953 ± 0.534 | 1.3924 ± 0.0529 |
| Grocery | H1_hybrid | 83.602 ± 0.439 | 76.425 ± 1.204 | 0.6513 ± 0.0350 |
| Grocery | H2_hybrid_plus | 83.856 ± 0.393 | 75.972 ± 1.113 | 0.6393 ± 0.0433 |
| Grocery | CG-CGE B1 | 83.631 ± 0.442 | 76.416 ± 0.923 | 0.6477 ± 0.0540 |
| Grocery | View-CGE A2 | 83.514 ± 0.456 | 76.276 ± 0.942 | 0.6659 ± 0.0547 |
| ele-fashion | H1_hybrid | 87.529 ± 0.127 | 74.024 ± 0.352 | 0.4080 ± 0.0067 |
| ele-fashion | H2_hybrid_plus | 87.719 ± 0.062 | 74.653 ± 0.349 | 0.4113 ± 0.0035 |
| ele-fashion | CG-CGE B1 | 87.576 ± 0.189 | 74.316 ± 0.277 | 0.4136 ± 0.0048 |
| ele-fashion | View-CGE A2 | 87.665 ± 0.099 | 75.124 ± 0.939 | 0.4303 ± 0.0241 |

## Best-F1 Validation metrics

| Dataset | Model | Accuracy % | Macro-F1 % | CE |
|---|---|---:|---:|---:|
| Movies | H1_hybrid | 55.229 ± 0.890 | 50.756 ± 1.352 | 1.4096 ± 0.0381 |
| Movies | H2_hybrid_plus | 55.439 ± 0.330 | 50.854 ± 0.621 | 1.3988 ± 0.0487 |
| Movies | CG-CGE B1 | 55.659 ± 0.862 | 50.913 ± 0.563 | 1.3936 ± 0.0697 |
| Movies | View-CGE A2 | 54.759 ± 0.618 | 49.571 ± 0.742 | 1.4402 ± 0.0374 |
| Grocery | H1_hybrid | 83.319 ± 0.646 | 76.926 ± 1.399 | 0.6669 ± 0.0583 |
| Grocery | H2_hybrid_plus | 83.436 ± 0.470 | 77.042 ± 1.094 | 0.6874 ± 0.0544 |
| Grocery | CG-CGE B1 | 83.153 ± 0.597 | 76.825 ± 0.845 | 0.6691 ± 0.0644 |
| Grocery | View-CGE A2 | 83.289 ± 0.476 | 77.168 ± 1.306 | 0.6955 ± 0.0717 |
| ele-fashion | H1_hybrid | 87.000 ± 0.232 | 75.893 ± 0.703 | 0.4118 ± 0.0078 |
| ele-fashion | H2_hybrid_plus | 86.877 ± 0.413 | 76.164 ± 0.220 | 0.4232 ± 0.0204 |
| ele-fashion | CG-CGE B1 | 86.530 ± 0.610 | 75.910 ± 0.365 | 0.4216 ± 0.0169 |
| ele-fashion | View-CGE A2 | 86.918 ± 0.384 | 76.481 ± 0.250 | 0.4352 ± 0.0160 |

## Per-seed results and Train/Validation gaps

| Dataset | Condition | Seed | Val Acc % | Val Macro-F1 % | Val CE | Best Acc Epoch | Best F1 Epoch | Train–Val Acc Gap pp | Train–Val F1 Gap pp | Train–Val CE Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | H1_hybrid | 42 | 56.329 | 50.517 | 1.4489 | 114 | 97 | 27.046 | 33.897 | -0.9225 |
| Movies | H1_hybrid | 43 | 55.969 | 49.091 | 1.3469 | 54 | 72 | 10.901 | 14.976 | -0.3934 |
| Movies | H1_hybrid | 44 | 56.719 | 50.000 | 1.3237 | 59 | 81 | 11.251 | 13.670 | -0.3977 |
| Movies | H2_hybrid_plus | 42 | 56.809 | 49.268 | 1.3753 | 93 | 108 | 18.209 | 25.535 | -0.6743 |
| Movies | H2_hybrid_plus | 43 | 56.329 | 48.950 | 1.3409 | 55 | 85 | 9.432 | 13.553 | -0.3554 |
| Movies | H2_hybrid_plus | 44 | 57.199 | 51.121 | 1.3327 | 71 | 72 | 13.120 | 16.476 | -0.4780 |
| Grocery | H1_hybrid | 42 | 83.602 | 76.857 | 0.6508 | 83 | 80 | 10.112 | 14.422 | -0.4295 |
| Grocery | H1_hybrid | 43 | 84.041 | 77.354 | 0.6165 | 80 | 92 | 8.755 | 13.006 | -0.3722 |
| Grocery | H1_hybrid | 44 | 83.163 | 75.064 | 0.6866 | 103 | 127 | 13.060 | 20.216 | -0.5377 |
| Grocery | H2_hybrid_plus | 42 | 83.572 | 75.353 | 0.6249 | 62 | 92 | 7.203 | 10.945 | -0.3141 |
| Grocery | H2_hybrid_plus | 43 | 84.305 | 77.257 | 0.6049 | 74 | 102 | 7.964 | 12.297 | -0.3500 |
| Grocery | H2_hybrid_plus | 44 | 83.690 | 75.307 | 0.6879 | 103 | 133 | 11.986 | 19.289 | -0.5269 |
| ele-fashion | H1_hybrid | 42 | 87.460 | 73.985 | 0.4056 | 121 | 145 | 1.943 | 6.614 | -0.1027 |
| ele-fashion | H1_hybrid | 43 | 87.675 | 74.394 | 0.4155 | 150 | 163 | 2.539 | 8.055 | -0.1392 |
| ele-fashion | H1_hybrid | 44 | 87.450 | 73.693 | 0.4027 | 121 | 150 | 1.854 | 6.942 | -0.0967 |
| ele-fashion | H2_hybrid_plus | 42 | 87.665 | 75.044 | 0.4093 | 157 | 183 | 2.713 | 8.770 | -0.1386 |
| ele-fashion | H2_hybrid_plus | 43 | 87.788 | 74.541 | 0.4092 | 155 | 178 | 2.758 | 8.651 | -0.1462 |
| ele-fashion | H2_hybrid_plus | 44 | 87.706 | 74.373 | 0.4154 | 158 | 188 | 3.010 | 9.586 | -0.1561 |

Train–Validation Accuracy/F1/CE gaps are Train minus Validation, with Train evaluated in eval mode at that checkpoint. The table above uses the Best-Accuracy checkpoint; the next table uses each run's Best-F1 checkpoint.

### Per-seed Best-F1 checkpoint metrics and gaps

| Dataset | Condition | Seed | Val Acc % | Val Macro-F1 % | Val CE | Train–Val Acc Gap pp | Train–Val F1 Gap pp | Train–Val CE Gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | H1_hybrid | 42 | 54.889 | 50.599 | 1.4478 | 22.988 | 29.578 | -0.7915 |
| Movies | H1_hybrid | 43 | 54.559 | 49.488 | 1.4093 | 16.530 | 21.353 | -0.5872 |
| Movies | H1_hybrid | 44 | 56.239 | 52.179 | 1.3717 | 18.169 | 21.254 | -0.6205 |
| Movies | H2_hybrid_plus | 42 | 55.099 | 51.138 | 1.4308 | 25.177 | 30.797 | -0.8277 |
| Movies | H2_hybrid_plus | 43 | 55.459 | 50.143 | 1.4229 | 19.229 | 24.613 | -0.6888 |
| Movies | H2_hybrid_plus | 44 | 55.759 | 51.283 | 1.3427 | 14.620 | 16.998 | -0.4884 |
| Grocery | H1_hybrid | 42 | 83.485 | 77.585 | 0.6403 | 10.209 | 13.898 | -0.4150 |
| Grocery | H1_hybrid | 43 | 83.865 | 77.874 | 0.6267 | 10.463 | 14.716 | -0.4292 |
| Grocery | H1_hybrid | 44 | 82.606 | 75.319 | 0.7337 | 15.373 | 22.240 | -0.6423 |
| Grocery | H2_hybrid_plus | 42 | 83.397 | 77.705 | 0.6689 | 11.508 | 15.542 | -0.4847 |
| Grocery | H2_hybrid_plus | 43 | 83.924 | 77.642 | 0.6447 | 11.068 | 15.608 | -0.4698 |
| Grocery | H2_hybrid_plus | 44 | 82.987 | 75.779 | 0.7486 | 14.817 | 21.361 | -0.6599 |
| ele-fashion | H1_hybrid | 42 | 87.102 | 75.344 | 0.4034 | 3.559 | 9.798 | -0.1424 |
| ele-fashion | H1_hybrid | 43 | 86.734 | 76.685 | 0.4129 | 4.906 | 10.928 | -0.1763 |
| ele-fashion | H1_hybrid | 44 | 87.164 | 75.651 | 0.4190 | 3.973 | 10.702 | -0.1679 |
| ele-fashion | H2_hybrid_plus | 42 | 87.215 | 75.924 | 0.4140 | 5.178 | 12.870 | -0.2023 |
| ele-fashion | H2_hybrid_plus | 43 | 87.000 | 76.357 | 0.4090 | 5.037 | 12.118 | -0.1865 |
| ele-fashion | H2_hybrid_plus | 44 | 86.417 | 76.211 | 0.4465 | 6.437 | 14.118 | -0.2452 |

## Paired Validation differences

Candidate minus reference; Accuracy and Macro-F1 are percentage points. Means and sample SDs are descriptive for three seeds.

| Dataset | Contrast | Checkpoint | Metric | Mean ± SD | +/−/0 Seeds |
|---|---|---|---|---:|---:|
| Movies | H1_hybrid − CG-CGE B1 | Best-Accuracy | Accuracy | -0.180 ± 0.030 | 0/3/0 |
| Movies | H1_hybrid − View-CGE A2 | Best-Accuracy | Accuracy | +0.380 ± 0.135 | 3/0/0 |
| Movies | H1_hybrid − CG-CGE B1 | Best-Accuracy | Macro-F1 | +0.259 ± 1.041 | 2/1/0 |
| Movies | H1_hybrid − View-CGE A2 | Best-Accuracy | Macro-F1 | +1.916 ± 0.569 | 3/0/0 |
| Movies | H1_hybrid − CG-CGE B1 | Best-Accuracy | CE | -0.002 ± 0.041 | 1/2/0 |
| Movies | H1_hybrid − View-CGE A2 | Best-Accuracy | CE | -0.019 ± 0.110 | 1/2/0 |
| Movies | H1_hybrid − CG-CGE B1 | Best-F1 | Accuracy | -0.430 ± 0.590 | 0/3/0 |
| Movies | H1_hybrid − View-CGE A2 | Best-F1 | Accuracy | +0.470 ± 0.272 | 3/0/0 |
| Movies | H1_hybrid − CG-CGE B1 | Best-F1 | Macro-F1 | -0.157 ± 1.319 | 2/1/0 |
| Movies | H1_hybrid − View-CGE A2 | Best-F1 | Macro-F1 | +1.185 ± 1.274 | 3/0/0 |
| Movies | H1_hybrid − CG-CGE B1 | Best-F1 | CE | +0.016 ± 0.068 | 2/1/0 |
| Movies | H1_hybrid − View-CGE A2 | Best-F1 | CE | -0.031 ± 0.075 | 1/2/0 |
| Movies | H2_hybrid_plus − CG-CGE B1 | Best-Accuracy | Accuracy | +0.260 ± 0.075 | 3/0/0 |
| Movies | H2_hybrid_plus − View-CGE A2 | Best-Accuracy | Accuracy | +0.820 ± 0.200 | 3/0/0 |
| Movies | H2_hybrid_plus − CG-CGE B1 | Best-Accuracy | Macro-F1 | +0.169 ± 0.323 | 2/1/0 |
| Movies | H2_hybrid_plus − View-CGE A2 | Best-Accuracy | Macro-F1 | +1.826 ± 1.539 | 3/0/0 |
| Movies | H2_hybrid_plus − CG-CGE B1 | Best-Accuracy | CE | -0.026 ± 0.030 | 1/2/0 |
| Movies | H2_hybrid_plus − View-CGE A2 | Best-Accuracy | CE | -0.043 ± 0.071 | 1/2/0 |
| Movies | H2_hybrid_plus − CG-CGE B1 | Best-F1 | Accuracy | -0.220 ± 0.884 | 1/2/0 |
| Movies | H2_hybrid_plus − View-CGE A2 | Best-F1 | Accuracy | +0.680 ± 0.445 | 3/0/0 |
| Movies | H2_hybrid_plus − CG-CGE B1 | Best-F1 | Macro-F1 | -0.059 ± 0.931 | 1/2/0 |
| Movies | H2_hybrid_plus − View-CGE A2 | Best-F1 | Macro-F1 | +1.283 ± 0.467 | 3/0/0 |
| Movies | H2_hybrid_plus − CG-CGE B1 | Best-F1 | CE | +0.005 ± 0.050 | 2/1/0 |
| Movies | H2_hybrid_plus − View-CGE A2 | Best-F1 | CE | -0.041 ± 0.085 | 1/2/0 |
| Movies | H2_hybrid_plus − within-study H1 | Best-Accuracy | Accuracy | +0.440 ± 0.069 | 3/0/0 |
| Movies | H2_hybrid_plus − within-study H1 | Best-Accuracy | Macro-F1 | -0.090 ± 1.186 | 1/2/0 |
| Movies | H2_hybrid_plus − within-study H1 | Best-Accuracy | CE | -0.024 ± 0.044 | 1/2/0 |
| Movies | H2_hybrid_plus − within-study H1 | Best-F1 | Accuracy | +0.210 ± 0.690 | 2/1/0 |
| Movies | H2_hybrid_plus − within-study H1 | Best-F1 | Macro-F1 | +0.099 ± 0.864 | 2/1/0 |
| Movies | H2_hybrid_plus − within-study H1 | Best-F1 | CE | -0.011 ± 0.022 | 1/2/0 |
| Grocery | H1_hybrid − CG-CGE B1 | Best-Accuracy | Accuracy | -0.029 ± 0.051 | 1/2/0 |
| Grocery | H1_hybrid − View-CGE A2 | Best-Accuracy | Accuracy | +0.088 ± 0.232 | 1/1/1 |
| Grocery | H1_hybrid − CG-CGE B1 | Best-Accuracy | Macro-F1 | +0.009 ± 0.447 | 1/2/0 |
| Grocery | H1_hybrid − View-CGE A2 | Best-Accuracy | Macro-F1 | +0.149 ± 0.766 | 2/1/0 |
| Grocery | H1_hybrid − CG-CGE B1 | Best-Accuracy | CE | +0.004 ± 0.021 | 2/1/0 |
| Grocery | H1_hybrid − View-CGE A2 | Best-Accuracy | CE | -0.015 ± 0.029 | 1/2/0 |
| Grocery | H1_hybrid − CG-CGE B1 | Best-F1 | Accuracy | +0.166 ± 0.266 | 2/1/0 |
| Grocery | H1_hybrid − View-CGE A2 | Best-F1 | Accuracy | +0.029 ± 0.279 | 2/1/0 |
| Grocery | H1_hybrid − CG-CGE B1 | Best-F1 | Macro-F1 | +0.101 ± 0.608 | 2/1/0 |
| Grocery | H1_hybrid − View-CGE A2 | Best-F1 | Macro-F1 | -0.242 ± 0.639 | 1/2/0 |
| Grocery | H1_hybrid − CG-CGE B1 | Best-F1 | CE | -0.002 ± 0.023 | 1/2/0 |
| Grocery | H1_hybrid − View-CGE A2 | Best-F1 | CE | -0.029 ± 0.018 | 0/3/0 |
| Grocery | H2_hybrid_plus − CG-CGE B1 | Best-Accuracy | Accuracy | +0.225 ± 0.235 | 2/0/1 |
| Grocery | H2_hybrid_plus − View-CGE A2 | Best-Accuracy | Accuracy | +0.342 ± 0.089 | 3/0/0 |
| Grocery | H2_hybrid_plus − CG-CGE B1 | Best-Accuracy | Macro-F1 | -0.444 ± 1.118 | 1/2/0 |
| Grocery | H2_hybrid_plus − View-CGE A2 | Best-Accuracy | Macro-F1 | -0.304 ± 0.216 | 0/3/0 |
| Grocery | H2_hybrid_plus − CG-CGE B1 | Best-Accuracy | CE | -0.008 ± 0.011 | 1/2/0 |
| Grocery | H2_hybrid_plus − View-CGE A2 | Best-Accuracy | CE | -0.027 ± 0.015 | 0/3/0 |
| Grocery | H2_hybrid_plus − CG-CGE B1 | Best-F1 | Accuracy | +0.283 ± 0.144 | 3/0/0 |
| Grocery | H2_hybrid_plus − View-CGE A2 | Best-F1 | Accuracy | +0.146 ± 0.051 | 3/0/0 |
| Grocery | H2_hybrid_plus − CG-CGE B1 | Best-F1 | Macro-F1 | +0.217 ± 0.266 | 2/1/0 |
| Grocery | H2_hybrid_plus − View-CGE A2 | Best-F1 | Macro-F1 | -0.126 ± 0.456 | 2/1/0 |
| Grocery | H2_hybrid_plus − CG-CGE B1 | Best-F1 | CE | +0.018 ± 0.019 | 3/0/0 |
| Grocery | H2_hybrid_plus − View-CGE A2 | Best-F1 | CE | -0.008 ± 0.018 | 1/2/0 |
| Grocery | H2_hybrid_plus − within-study H1 | Best-Accuracy | Accuracy | +0.254 ± 0.278 | 2/1/0 |
| Grocery | H2_hybrid_plus − within-study H1 | Best-Accuracy | Macro-F1 | -0.453 ± 0.926 | 1/2/0 |
| Grocery | H2_hybrid_plus − within-study H1 | Best-Accuracy | CE | -0.012 ± 0.014 | 1/2/0 |
| Grocery | H2_hybrid_plus − within-study H1 | Best-F1 | Accuracy | +0.117 ± 0.240 | 2/1/0 |
| Grocery | H2_hybrid_plus − within-study H1 | Best-F1 | Macro-F1 | +0.116 ± 0.347 | 2/1/0 |
| Grocery | H2_hybrid_plus − within-study H1 | Best-F1 | CE | +0.020 ± 0.007 | 3/0/0 |
| ele-fashion | H1_hybrid − CG-CGE B1 | Best-Accuracy | Accuracy | -0.048 ± 0.198 | 2/1/0 |
| ele-fashion | H1_hybrid − View-CGE A2 | Best-Accuracy | Accuracy | -0.136 ± 0.189 | 1/2/0 |
| ele-fashion | H1_hybrid − CG-CGE B1 | Best-Accuracy | Macro-F1 | -0.292 ± 0.619 | 1/2/0 |
| ele-fashion | H1_hybrid − View-CGE A2 | Best-Accuracy | Macro-F1 | -1.100 ± 1.278 | 1/2/0 |
| ele-fashion | H1_hybrid − CG-CGE B1 | Best-Accuracy | CE | -0.006 ± 0.004 | 0/3/0 |
| ele-fashion | H1_hybrid − View-CGE A2 | Best-Accuracy | CE | -0.022 ± 0.019 | 0/3/0 |
| ele-fashion | H1_hybrid − CG-CGE B1 | Best-F1 | Accuracy | +0.470 ± 0.396 | 3/0/0 |
| ele-fashion | H1_hybrid − View-CGE A2 | Best-F1 | Accuracy | +0.082 ± 0.276 | 2/1/0 |
| ele-fashion | H1_hybrid − CG-CGE B1 | Best-F1 | Macro-F1 | -0.017 ± 1.064 | 1/2/0 |
| ele-fashion | H1_hybrid − View-CGE A2 | Best-F1 | Macro-F1 | -0.588 ± 0.594 | 1/2/0 |
| ele-fashion | H1_hybrid − CG-CGE B1 | Best-F1 | CE | -0.010 ± 0.018 | 1/2/0 |
| ele-fashion | H1_hybrid − View-CGE A2 | Best-F1 | CE | -0.023 ± 0.012 | 0/3/0 |
| ele-fashion | H2_hybrid_plus − CG-CGE B1 | Best-Accuracy | Accuracy | +0.143 ± 0.205 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − View-CGE A2 | Best-Accuracy | Accuracy | +0.055 ± 0.119 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − CG-CGE B1 | Best-Accuracy | Macro-F1 | +0.337 ± 0.396 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − View-CGE A2 | Best-Accuracy | Macro-F1 | -0.471 ± 1.142 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − CG-CGE B1 | Best-Accuracy | CE | -0.002 ± 0.008 | 1/2/0 |
| ele-fashion | H2_hybrid_plus − View-CGE A2 | Best-Accuracy | CE | -0.019 ± 0.025 | 1/2/0 |
| ele-fashion | H2_hybrid_plus − CG-CGE B1 | Best-F1 | Accuracy | +0.348 ± 0.934 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − View-CGE A2 | Best-F1 | Accuracy | -0.041 ± 0.782 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − CG-CGE B1 | Best-F1 | Macro-F1 | +0.254 ± 0.579 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − View-CGE A2 | Best-F1 | Macro-F1 | -0.317 ± 0.113 | 0/3/0 |
| ele-fashion | H2_hybrid_plus − CG-CGE B1 | Best-F1 | CE | +0.002 ± 0.034 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − View-CGE A2 | Best-F1 | CE | -0.012 ± 0.004 | 0/3/0 |
| ele-fashion | H2_hybrid_plus − within-study H1 | Best-Accuracy | Accuracy | +0.191 ± 0.073 | 3/0/0 |
| ele-fashion | H2_hybrid_plus − within-study H1 | Best-Accuracy | Macro-F1 | +0.629 ± 0.458 | 3/0/0 |
| ele-fashion | H2_hybrid_plus − within-study H1 | Best-Accuracy | CE | +0.003 ± 0.009 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − within-study H1 | Best-F1 | Accuracy | -0.123 ± 0.546 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − within-study H1 | Best-F1 | Macro-F1 | +0.271 ± 0.518 | 2/1/0 |
| ele-fashion | H2_hybrid_plus − within-study H1 | Best-F1 | CE | +0.011 ± 0.016 | 2/1/0 |

## Dropout, weight decay and interaction

The shared backbone dropout is 0.2 and AdamW weight decay is 1e-4 in every formal run. H1 has no late-residual dropout and masks SameNode; H2 adds a valid SameNode candidate and applies p=0.15 non-inverted dropout to the final Late residual during training. The H2−H1 estimate therefore describes this combined package. This design cannot separate SameNode from late dropout, estimate a SameNode×dropout interaction, or estimate a weight-decay main effect or interaction. The three-seed paired contrasts below are descriptive, not a factorial causal estimate.

| Dataset | H2−H1 Δ Accuracy pp | H2−H1 Δ Macro-F1 pp | H2−H1 Δ CE |
|---|---:|---:|---:|
| Movies | +0.440 ± 0.069 | -0.090 ± 1.186 | -0.0235 ± 0.0440 |
| Grocery | +0.254 ± 0.278 | -0.453 ± 0.926 | -0.0121 ± 0.0136 |
| ele-fashion | +0.191 ± 0.073 | +0.629 ± 0.458 | +0.0033 ± 0.0095 |

The train-mode GPU preflight recorded realized node-mask drop fractions of 0 for H1 and approximately 0.148–0.150 for H2 across datasets and both target modalities (one seed-42 optimizer-step check). Best-checkpoint mechanism diagnostics run in evaluation mode, so their realized dropout fraction is 0 by design; the configured training probability remains 0.15 for H2.


## Training curves and auxiliary loss

Full epoch-by-epoch Train/Validation Accuracy, Macro-F1, CE, Private auxiliary CE, balance auxiliary and timing are retained in data/epoch_history.csv. The figure [training curves](figures/training_curves/hybrid_cge_v11_training_curves.pdf) shows seed means with sample SD; the data behind it are the full run histories.

| Dataset | Condition | First-epoch Private CE | Best-Accuracy-epoch Private CE | Best-F1 − Best-Accuracy Δ Acc pp | Δ Macro-F1 pp | Δ CE |
|---|---|---:|---:|---:|---:|---:|
| Movies | H1_hybrid | 3.2951 ± 0.2091 | 1.2618 ± 0.1076 | -1.110 ± 0.546 | 0.886 ± 1.131 | 0.0364 ± 0.0333 |
| Movies | H2_hybrid_plus | 3.2951 ± 0.2091 | 1.2703 ± 0.0700 | -1.340 ± 0.429 | 1.075 ± 0.860 | 0.0492 ± 0.0364 |
| Grocery | H1_hybrid | 3.2797 ± 0.0125 | 0.5370 ± 0.0331 | -0.283 ± 0.238 | 0.501 ± 0.237 | 0.0156 ± 0.0292 |
| Grocery | H2_hybrid_plus | 3.2797 ± 0.0125 | 0.5698 ± 0.0624 | -0.420 ± 0.266 | 1.070 ± 1.111 | 0.0481 ± 0.0111 |
| ele-fashion | H1_hybrid | 2.6942 ± 0.1114 | 0.4488 ± 0.0018 | -0.528 ± 0.359 | 1.869 ± 0.472 | 0.0038 ± 0.0108 |
| ele-fashion | H2_hybrid_plus | 2.6942 ± 0.1114 | 0.4357 ± 0.0028 | -0.842 ± 0.422 | 1.511 ± 0.547 | 0.0119 ± 0.0169 |

## Figure and QA

The linked 3×3 training-curve figure plots mean ± sample SD across three matched seeds at every displayed epoch, truncated to each dataset's shared complete horizon. Strict source, panel-alignment, PDF text-size and rendered-collision checks all passed; see [figure QA](figures/QA.md) and its machine-readable records.


## Resource profile

| Dataset | Condition | Wall seconds | Mean epoch seconds | Peak allocated GiB | Peak reserved GiB | Model parameters | Optimizer parameters |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | H1_hybrid | 108.0 ± 28.8 | 0.913 | 4.20 | 4.67 | 2268134 | 2273274 |
| Movies | H2_hybrid_plus | 97.5 ± 17.8 | 0.845 | 4.21 | 4.69 | 2268134 | 2273274 |
| Grocery | H1_hybrid | 115.6 ± 21.0 | 0.830 | 4.12 | 4.43 | 2268134 | 2273274 |
| Grocery | H2_hybrid_plus | 122.2 ± 42.4 | 0.945 | 4.13 | 4.45 | 2268134 | 2273274 |
| ele-fashion | H1_hybrid | 360.5 ± 70.8 | 2.003 | 19.97 | 20.82 | 2001878 | 2004962 |
| ele-fashion | H2_hybrid_plus | 419.4 ± 91.9 | 2.024 | 20.07 | 20.90 | 2001878 | 2004962 |

Formal training used physical GPU0: NVIDIA GeForce RTX 3090, driver 535.309.01; the runner restricted visibility to this device.
Environment: Python 3.12.13, PyTorch 2.4.0+cu121, CUDA build 12.1, cuDNN 90100, NumPy 2.4.3; matmul TF32=False, cuDNN TF32=True, deterministic algorithms=False, CUBLAS_WORKSPACE_CONFIG=None.
All 36 selected checkpoints reloaded and reproduced their Validation metrics, and passed SHA256 and byte-count verification. No accepted run recorded an OOM.

## Mechanism and architecture diagnostics

Values below summarize Best-Accuracy checkpoints. They verify executed paths, not causal usefulness.

| Dataset | Condition | Path | Diagnostic | Mean | SD | n |
|---|---|---|---|---:|---:|---:|
| Movies | H1_hybrid | early_physical_cross | raw_rms | 0.31942 | 0.11193 | 6 |
| Movies | H1_hybrid | early_physical_cross | calibrated_rms | 0.56418 | 0.20029 | 6 |
| Movies | H1_hybrid | early_physical_cross | coefficient_mean | 0.26361 | 0.015532 | 6 |
| Movies | H1_hybrid | late_mean | semantic_source_weight | 0.50153 | 0.37506 | 6 |
| Movies | H1_hybrid | late_mean | same_node_source_weight | 0 | 0 | 6 |
| Movies | H1_hybrid | late_mean | none_source_weight | 0.49847 | 0.37506 | 6 |
| Movies | H1_hybrid | late_mean | p_accept_mean | 0.50153 | 0.37506 | 6 |
| Movies | H1_hybrid | late_mean | candidate_rms | 0.49006 | 0.2144 | 6 |
| Movies | H1_hybrid | late_mean | uncontrolled_residual_rms | 0.064077 | 0.056155 | 6 |
| Movies | H1_hybrid | late_mean | controlled_residual_rms | 0.063994 | 0.055963 | 6 |
| Movies | H1_hybrid | late_mean | residual_to_collaborative_rms | 0.045922 | 0.044496 | 6 |
| Movies | H1_hybrid | late_mean | relative_control_trigger_fraction | 0.0014595 | 0.0035751 | 6 |
| Movies | H1_hybrid | late_mean | relative_control_trigger_fraction_supported | 0.0014595 | 0.0035751 | 6 |
| Movies | H1_hybrid | late_mean | relative_control_scale_mean | 0.99988 | 0.00029366 | 6 |
| Movies | H1_hybrid | late_mean | tau | 0.25088 | 0.0021061 | 6 |
| Movies | H1_hybrid | late_mean | channel_gate_mean | 0.69841 | 0.023919 | 6 |
| Movies | H1_hybrid | late_mean | channel_gate_saturation_fraction | 0.21917 | 0.25935 | 6 |
| Movies | H1_hybrid | late_mean | late_cross_dropout_probability | 0 | 0 | 6 |
| Movies | H1_hybrid | late_mean | complementary_expert_rms | 0.3122 | 0.082338 | 6 |
| Movies | H1_hybrid | late_mean | semantic_attention_entropy_multi_neighbor | 0.96288 | 0.55218 | 6 |
| Movies | H1_hybrid | eval_mode | late_cross_dropout_realized_fraction | 0 | 0 | 6 |
| Movies | H1_hybrid | per_modality_mean | physical_cross_input_router_correction_rms | 1.1723 | 0.76807 | 6 |
| Movies | H1_hybrid | per_modality_mean | physical_route_top2_flip_fraction | 0.35646 | 0.2919 | 6 |
| Movies | H1_hybrid | per_modality_mean | physical_private_response_rms | 0.79276 | 0.29231 | 6 |
| Movies | H1_hybrid | per_modality_mean | physical_collab_response_rms | 0.59577 | 0.32067 | 6 |
| Movies | H1_hybrid | per_modality_mean | same_semantic_response_rms | 1.1979 | 0.047202 | 6 |
| Movies | H1_hybrid | per_modality_mean | same_semantic_weighted_rms | 0.09563 | 0.00365 | 6 |
| Movies | H2_hybrid_plus | early_physical_cross | raw_rms | 0.31888 | 0.078388 | 6 |
| Movies | H2_hybrid_plus | early_physical_cross | calibrated_rms | 0.56839 | 0.16372 | 6 |
| Movies | H2_hybrid_plus | early_physical_cross | coefficient_mean | 0.26597 | 0.0092081 | 6 |
| Movies | H2_hybrid_plus | late_mean | semantic_source_weight | 0.13579 | 0.20077 | 6 |
| Movies | H2_hybrid_plus | late_mean | same_node_source_weight | 0.19395 | 0.20973 | 6 |
| Movies | H2_hybrid_plus | late_mean | none_source_weight | 0.67026 | 0.30919 | 6 |
| Movies | H2_hybrid_plus | late_mean | p_accept_mean | 0.32974 | 0.30919 | 6 |
| Movies | H2_hybrid_plus | late_mean | candidate_rms | 0.61767 | 0.15289 | 6 |
| Movies | H2_hybrid_plus | late_mean | uncontrolled_residual_rms | 0.051273 | 0.039044 | 6 |
| Movies | H2_hybrid_plus | late_mean | controlled_residual_rms | 0.051273 | 0.039044 | 6 |
| Movies | H2_hybrid_plus | late_mean | residual_to_collaborative_rms | 0.037096 | 0.032496 | 6 |
| Movies | H2_hybrid_plus | late_mean | relative_control_trigger_fraction | 0 | 0 | 6 |
| Movies | H2_hybrid_plus | late_mean | relative_control_trigger_fraction_supported | 0 | 0 | 6 |
| Movies | H2_hybrid_plus | late_mean | relative_control_scale_mean | 1 | 0 | 6 |
| Movies | H2_hybrid_plus | late_mean | tau | 0.25 | 2.404e-07 | 6 |
| Movies | H2_hybrid_plus | late_mean | channel_gate_mean | 0.68835 | 0.04832 | 6 |
| Movies | H2_hybrid_plus | late_mean | channel_gate_saturation_fraction | 0.13332 | 0.14962 | 6 |
| Movies | H2_hybrid_plus | late_mean | late_cross_dropout_probability | 0.15 | 0 | 6 |
| Movies | H2_hybrid_plus | late_mean | complementary_expert_rms | 0.26359 | 0.040383 | 6 |
| Movies | H2_hybrid_plus | late_mean | semantic_attention_entropy_multi_neighbor | 1.244 | 0.53553 | 6 |
| Movies | H2_hybrid_plus | eval_mode | late_cross_dropout_realized_fraction | 0 | 0 | 6 |
| Movies | H2_hybrid_plus | per_modality_mean | physical_cross_input_router_correction_rms | 1.0461 | 0.54958 | 6 |
| Movies | H2_hybrid_plus | per_modality_mean | physical_route_top2_flip_fraction | 0.36268 | 0.35367 | 6 |
| Movies | H2_hybrid_plus | per_modality_mean | physical_private_response_rms | 0.80884 | 0.25783 | 6 |
| Movies | H2_hybrid_plus | per_modality_mean | physical_collab_response_rms | 0.63082 | 0.35593 | 6 |
| Movies | H2_hybrid_plus | per_modality_mean | same_semantic_response_rms | 1.1949 | 0.042719 | 6 |
| Movies | H2_hybrid_plus | per_modality_mean | same_semantic_weighted_rms | 0.09535 | 0.0032385 | 6 |
| Grocery | H1_hybrid | early_physical_cross | raw_rms | 0.46024 | 0.085275 | 6 |
| Grocery | H1_hybrid | early_physical_cross | calibrated_rms | 0.79194 | 0.1857 | 6 |
| Grocery | H1_hybrid | early_physical_cross | coefficient_mean | 0.26246 | 0.00090621 | 6 |
| Grocery | H1_hybrid | late_mean | semantic_source_weight | 0.93428 | 0.055694 | 6 |
| Grocery | H1_hybrid | late_mean | same_node_source_weight | 0 | 0 | 6 |
| Grocery | H1_hybrid | late_mean | none_source_weight | 0.065716 | 0.055694 | 6 |
| Grocery | H1_hybrid | late_mean | p_accept_mean | 0.93428 | 0.055694 | 6 |
| Grocery | H1_hybrid | late_mean | candidate_rms | 0.86866 | 0.1502 | 6 |
| Grocery | H1_hybrid | late_mean | uncontrolled_residual_rms | 0.17904 | 0.037828 | 6 |
| Grocery | H1_hybrid | late_mean | controlled_residual_rms | 0.17877 | 0.037348 | 6 |
| Grocery | H1_hybrid | late_mean | residual_to_collaborative_rms | 0.076237 | 0.019757 | 6 |
| Grocery | H1_hybrid | late_mean | relative_control_trigger_fraction | 0.0030944 | 0.0056291 | 6 |
| Grocery | H1_hybrid | late_mean | relative_control_trigger_fraction_supported | 0.0030944 | 0.0056291 | 6 |
| Grocery | H1_hybrid | late_mean | relative_control_scale_mean | 0.99969 | 0.00059365 | 6 |
| Grocery | H1_hybrid | late_mean | tau | 0.25019 | 0.0034555 | 6 |
| Grocery | H1_hybrid | late_mean | channel_gate_mean | 0.74464 | 0.04927 | 6 |
| Grocery | H1_hybrid | late_mean | channel_gate_saturation_fraction | 0.54948 | 0.11712 | 6 |
| Grocery | H1_hybrid | late_mean | late_cross_dropout_probability | 0 | 0 | 6 |
| Grocery | H1_hybrid | late_mean | complementary_expert_rms | 0.31278 | 0.025512 | 6 |
| Grocery | H1_hybrid | late_mean | semantic_attention_entropy_multi_neighbor | 0.86884 | 0.14576 | 6 |
| Grocery | H1_hybrid | eval_mode | late_cross_dropout_realized_fraction | 0 | 0 | 6 |
| Grocery | H1_hybrid | per_modality_mean | physical_cross_input_router_correction_rms | 1.694 | 1.0549 | 6 |
| Grocery | H1_hybrid | per_modality_mean | physical_route_top2_flip_fraction | 0.15548 | 0.22355 | 6 |
| Grocery | H1_hybrid | per_modality_mean | physical_private_response_rms | 1.4901 | 0.21835 | 6 |
| Grocery | H1_hybrid | per_modality_mean | physical_collab_response_rms | 1.4893 | 0.238 | 6 |
| Grocery | H1_hybrid | per_modality_mean | same_semantic_response_rms | 1.2796 | 0.11491 | 6 |
| Grocery | H1_hybrid | per_modality_mean | same_semantic_weighted_rms | 0.10361 | 0.010439 | 6 |
| Grocery | H2_hybrid_plus | early_physical_cross | raw_rms | 0.32169 | 0.083618 | 6 |
| Grocery | H2_hybrid_plus | early_physical_cross | calibrated_rms | 0.57132 | 0.19519 | 6 |
| Grocery | H2_hybrid_plus | early_physical_cross | coefficient_mean | 0.26322 | 0.0036563 | 6 |
| Grocery | H2_hybrid_plus | late_mean | semantic_source_weight | 0.434 | 0.23572 | 6 |
| Grocery | H2_hybrid_plus | late_mean | same_node_source_weight | 0.46243 | 0.22727 | 6 |
| Grocery | H2_hybrid_plus | late_mean | none_source_weight | 0.10357 | 0.14119 | 6 |
| Grocery | H2_hybrid_plus | late_mean | p_accept_mean | 0.89643 | 0.14119 | 6 |
| Grocery | H2_hybrid_plus | late_mean | candidate_rms | 0.9024 | 0.25871 | 6 |
| Grocery | H2_hybrid_plus | late_mean | uncontrolled_residual_rms | 0.15872 | 0.040619 | 6 |
| Grocery | H2_hybrid_plus | late_mean | controlled_residual_rms | 0.15645 | 0.038463 | 6 |
| Grocery | H2_hybrid_plus | late_mean | residual_to_collaborative_rms | 0.069438 | 0.021338 | 6 |
| Grocery | H2_hybrid_plus | late_mean | relative_control_trigger_fraction | 0.0067451 | 0.015899 | 6 |
| Grocery | H2_hybrid_plus | late_mean | relative_control_trigger_fraction_supported | 0.0067451 | 0.015899 | 6 |
| Grocery | H2_hybrid_plus | late_mean | relative_control_scale_mean | 0.99835 | 0.0038797 | 6 |
| Grocery | H2_hybrid_plus | late_mean | tau | 0.25129 | 0.0036005 | 6 |
| Grocery | H2_hybrid_plus | late_mean | channel_gate_mean | 0.70569 | 0.042943 | 6 |
| Grocery | H2_hybrid_plus | late_mean | channel_gate_saturation_fraction | 0.2227 | 0.077089 | 6 |
| Grocery | H2_hybrid_plus | late_mean | late_cross_dropout_probability | 0.15 | 0 | 6 |
| Grocery | H2_hybrid_plus | late_mean | complementary_expert_rms | 0.28246 | 0.048704 | 6 |
| Grocery | H2_hybrid_plus | late_mean | semantic_attention_entropy_multi_neighbor | 1.0216 | 0.31225 | 6 |
| Grocery | H2_hybrid_plus | eval_mode | late_cross_dropout_realized_fraction | 0 | 0 | 6 |
| Grocery | H2_hybrid_plus | per_modality_mean | physical_cross_input_router_correction_rms | 1.6683 | 0.81446 | 6 |
| Grocery | H2_hybrid_plus | per_modality_mean | physical_route_top2_flip_fraction | 0.1414 | 0.14385 | 6 |
| Grocery | H2_hybrid_plus | per_modality_mean | physical_private_response_rms | 1.4606 | 0.25197 | 6 |
| Grocery | H2_hybrid_plus | per_modality_mean | physical_collab_response_rms | 1.4129 | 0.34064 | 6 |
| Grocery | H2_hybrid_plus | per_modality_mean | same_semantic_response_rms | 1.286 | 0.12078 | 6 |
| Grocery | H2_hybrid_plus | per_modality_mean | same_semantic_weighted_rms | 0.10413 | 0.010888 | 6 |
| ele-fashion | H1_hybrid | early_physical_cross | raw_rms | 0.74994 | 0.075598 | 6 |
| ele-fashion | H1_hybrid | early_physical_cross | calibrated_rms | 1.139 | 0.25645 | 6 |
| ele-fashion | H1_hybrid | early_physical_cross | coefficient_mean | 0.25258 | 0.0077384 | 6 |
| ele-fashion | H1_hybrid | late_mean | semantic_source_weight | 0.88097 | 0.11072 | 6 |
| ele-fashion | H1_hybrid | late_mean | same_node_source_weight | 0 | 0 | 6 |
| ele-fashion | H1_hybrid | late_mean | none_source_weight | 0.11903 | 0.11072 | 6 |
| ele-fashion | H1_hybrid | late_mean | p_accept_mean | 0.88097 | 0.11072 | 6 |
| ele-fashion | H1_hybrid | late_mean | candidate_rms | 1.0045 | 0.088078 | 6 |
| ele-fashion | H1_hybrid | late_mean | uncontrolled_residual_rms | 0.22352 | 0.048395 | 6 |
| ele-fashion | H1_hybrid | late_mean | controlled_residual_rms | 0.21706 | 0.043539 | 6 |
| ele-fashion | H1_hybrid | late_mean | residual_to_collaborative_rms | 0.15103 | 0.026948 | 6 |
| ele-fashion | H1_hybrid | late_mean | relative_control_trigger_fraction | 0.098432 | 0.078126 | 6 |
| ele-fashion | H1_hybrid | late_mean | relative_control_trigger_fraction_supported | 0.098432 | 0.078126 | 6 |
| ele-fashion | H1_hybrid | late_mean | relative_control_scale_mean | 0.9885 | 0.011986 | 6 |
| ele-fashion | H1_hybrid | late_mean | tau | 0.25839 | 0.0031893 | 6 |
| ele-fashion | H1_hybrid | late_mean | channel_gate_mean | 0.81885 | 0.033809 | 6 |
| ele-fashion | H1_hybrid | late_mean | channel_gate_saturation_fraction | 0.61244 | 0.058395 | 6 |
| ele-fashion | H1_hybrid | late_mean | late_cross_dropout_probability | 0 | 0 | 6 |
| ele-fashion | H1_hybrid | late_mean | complementary_expert_rms | 1.464 | 0.28661 | 6 |
| ele-fashion | H1_hybrid | late_mean | semantic_attention_entropy_multi_neighbor | 1.1957 | 0.20969 | 6 |
| ele-fashion | H1_hybrid | eval_mode | late_cross_dropout_realized_fraction | 0 | 0 | 6 |
| ele-fashion | H1_hybrid | per_modality_mean | physical_cross_input_router_correction_rms | 1.852 | 0.82644 | 6 |
| ele-fashion | H1_hybrid | per_modality_mean | physical_route_top2_flip_fraction | 0.32612 | 0.13693 | 6 |
| ele-fashion | H1_hybrid | per_modality_mean | physical_private_response_rms | 0.49875 | 0.10851 | 6 |
| ele-fashion | H1_hybrid | per_modality_mean | physical_collab_response_rms | 0.48375 | 0.10155 | 6 |
| ele-fashion | H1_hybrid | per_modality_mean | same_semantic_response_rms | 1.8464 | 0.35233 | 6 |
| ele-fashion | H1_hybrid | per_modality_mean | same_semantic_weighted_rms | 0.15909 | 0.031914 | 6 |
| ele-fashion | H2_hybrid_plus | early_physical_cross | raw_rms | 0.63083 | 0.079416 | 6 |
| ele-fashion | H2_hybrid_plus | early_physical_cross | calibrated_rms | 0.95966 | 0.23978 | 6 |
| ele-fashion | H2_hybrid_plus | early_physical_cross | coefficient_mean | 0.27548 | 0.0050182 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | semantic_source_weight | 0.77205 | 0.20571 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | same_node_source_weight | 0.092099 | 0.13467 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | none_source_weight | 0.13586 | 0.15685 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | p_accept_mean | 0.86414 | 0.15685 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | candidate_rms | 0.9161 | 0.049731 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | uncontrolled_residual_rms | 0.20514 | 0.039458 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | controlled_residual_rms | 0.20405 | 0.03908 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | residual_to_collaborative_rms | 0.13687 | 0.026964 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | relative_control_trigger_fraction | 0.019917 | 0.016366 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | relative_control_trigger_fraction_supported | 0.019917 | 0.016366 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | relative_control_scale_mean | 0.99829 | 0.001482 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | tau | 0.26098 | 0.0098098 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | channel_gate_mean | 0.80042 | 0.020809 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | channel_gate_saturation_fraction | 0.56998 | 0.066512 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | late_cross_dropout_probability | 0.15 | 0 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | complementary_expert_rms | 1.4121 | 0.55063 | 6 |
| ele-fashion | H2_hybrid_plus | late_mean | semantic_attention_entropy_multi_neighbor | 0.96219 | 0.1361 | 6 |
| ele-fashion | H2_hybrid_plus | eval_mode | late_cross_dropout_realized_fraction | 0 | 0 | 6 |
| ele-fashion | H2_hybrid_plus | per_modality_mean | physical_cross_input_router_correction_rms | 1.6344 | 0.63189 | 6 |
| ele-fashion | H2_hybrid_plus | per_modality_mean | physical_route_top2_flip_fraction | 0.28394 | 0.12157 | 6 |
| ele-fashion | H2_hybrid_plus | per_modality_mean | physical_private_response_rms | 0.55493 | 0.11366 | 6 |
| ele-fashion | H2_hybrid_plus | per_modality_mean | physical_collab_response_rms | 0.55026 | 0.098649 | 6 |
| ele-fashion | H2_hybrid_plus | per_modality_mean | same_semantic_response_rms | 2.0125 | 0.40849 | 6 |
| ele-fashion | H2_hybrid_plus | per_modality_mean | same_semantic_weighted_rms | 0.17713 | 0.037551 | 6 |

## Fashion lower-support Validation classes

The loader does not provide a canonical class-name map in run metadata; IDs and per-seed support are used.

| Class ID | Support | Condition | F1 mean ± SD % | n |
|---:|---:|---|---:|---:|
| 5 | 0 | H1_hybrid | 0.000 ± 0.000 | 3 |
| 5 | 0 | H2_hybrid_plus | 0.000 ± 0.000 | 3 |
| 5 | 0 | B1_complementary | 0.000 ± 0.000 | 3 |
| 5 | 0 | A2_cross_source | 0.000 ± 0.000 | 3 |
| 3 | 45 | H1_hybrid | 63.177 ± 5.627 | 3 |
| 3 | 45 | H2_hybrid_plus | 62.541 ± 3.106 | 3 |
| 3 | 45 | B1_complementary | 62.104 ± 0.857 | 3 |
| 3 | 45 | A2_cross_source | 62.279 ± 4.584 | 3 |
| 7 | 95 | H1_hybrid | 60.153 ± 1.929 | 3 |
| 7 | 95 | H2_hybrid_plus | 61.382 ± 2.584 | 3 |
| 7 | 95 | B1_complementary | 59.380 ± 1.746 | 3 |
| 7 | 95 | A2_cross_source | 62.288 ± 1.643 | 3 |
| 9 | 192 | H1_hybrid | 89.596 ± 0.807 | 3 |
| 9 | 192 | H2_hybrid_plus | 89.593 ± 0.311 | 3 |
| 9 | 192 | B1_complementary | 89.783 ± 0.784 | 3 |
| 9 | 192 | A2_cross_source | 89.622 ± 0.399 | 3 |
| 4 | 290 | H1_hybrid | 76.104 ± 0.803 | 3 |
| 4 | 290 | H2_hybrid_plus | 77.082 ± 0.483 | 3 |
| 4 | 290 | B1_complementary | 76.378 ± 1.402 | 3 |
| 4 | 290 | A2_cross_source | 75.986 ± 0.839 | 3 |
| 11 | 346 | H1_hybrid | 19.584 ± 0.808 | 3 |
| 11 | 346 | H2_hybrid_plus | 21.894 ± 2.437 | 3 |
| 11 | 346 | B1_complementary | 23.871 ± 1.758 | 3 |
| 11 | 346 | A2_cross_source | 27.585 ± 5.371 | 3 |
| 10 | 424 | H1_hybrid | 91.968 ± 0.223 | 3 |
| 10 | 424 | H2_hybrid_plus | 91.676 ± 0.070 | 3 |
| 10 | 424 | B1_complementary | 91.817 ± 0.328 | 3 |
| 10 | 424 | A2_cross_source | 91.683 ± 0.405 | 3 |
| 8 | 560 | H1_hybrid | 89.230 ± 0.123 | 3 |
| 8 | 560 | H2_hybrid_plus | 89.813 ± 0.525 | 3 |
| 8 | 560 | B1_complementary | 89.729 ± 0.573 | 3 |
| 8 | 560 | A2_cross_source | 90.278 ± 0.453 | 3 |
| 1 | 868 | H1_hybrid | 45.244 ± 0.844 | 3 |
| 1 | 868 | H2_hybrid_plus | 47.811 ± 1.152 | 3 |
| 1 | 868 | B1_complementary | 45.162 ± 0.936 | 3 |
| 1 | 868 | A2_cross_source | 47.375 ± 3.741 | 3 |
| 6 | 2028 | H1_hybrid | 93.560 ± 0.078 | 3 |
| 6 | 2028 | H2_hybrid_plus | 93.662 ± 0.118 | 3 |
| 6 | 2028 | B1_complementary | 93.575 ± 0.052 | 3 |
| 6 | 2028 | A2_cross_source | 93.477 ± 0.049 | 3 |

## Independent implementation audit

- Private state is computed from each receiver modality only. The early cross message enters every Collaborative physical expert profile and the inherited physical-router Cross input. The Late branch has no Physical Cross candidate.
- H1/H2 share the same four experts, same source selector architecture and matched parameter initialization. H1 masks SameNode. H2 enables SameNode and applies non-inverted node dropout only to the final Late residual during training.
- Relative control caps large residuals and never amplifies weak ones. A zero tau gradient is allowed when no node reaches the cap.
- Private auxiliary CE is produced from U_priv and the existing NC objective indexes training labels only. Both held-out splits remain unavailable to objective or checkpoint selection.
- Three seeds do not establish significance. Best-F1 selection can trade Accuracy for Macro-F1; inspect the paired tables and per-class Fashion metrics.

## Preserved failures and unresolved issues

Preserved records: 0 formal-run failures; 0 preflight attempts/refusals; 1 runner launch-access records; 1 foreign-GPU guard stops.
- Runner access 2026-10-17T14:00:52+00:00: refused_before_training; training_started=False — Sandbox-launched nvidia-smi child process could not communicate with the NVIDIA driver (exit status 9) during the runner idle guard. Reran through the previously authorized host GPU access after independently confirming physical GPU 0 idle; the idle guard then passed.
- Runner 2026-10-17T08:56:41+00:00: stopped before the next cell on physical GPU 0 because foreign PID 1353598 held 5350 MiB; no process was interrupted.
Unresolved findings:
- H2's Best-Accuracy Accuracy gain is small but positive on all three datasets; its Macro-F1 difference is negative on Movies/Grocery and positive on ele-fashion. The dataset-dependent tradeoff is not explained by this study, and SameNode cannot be separated from late dropout.
- Train/Validation gaps are larger on Movies and Grocery than ele-fashion. This study records the pattern but does not identify its data or optimization cause.
- Seed-to-seed variation is descriptive with three seeds. Formal runs retain the recorded default nondeterministic-algorithm setting; no deterministic full-run replay was attempted, so the contribution of individual stochastic sources remains unmeasured.

No Test evaluation was performed. Mechanism magnitudes do not by themselves establish causal contribution.

## Evidence files

runs.csv, summary.csv, paired_comparisons.csv, epoch_history.csv, classwise_metrics.csv, elefashion_minority_class_metrics.csv, mechanism_diagnostics.csv, mechanism_summary.csv, resource_profile.csv, gradient_audit.json, artifact_integrity.json, preflight_cpu.json, preflight.json, runner_interference_events.json, runner_access_attempts.json and manifest.json are retained under data/. Figure exports and [figure QA](figures/QA.md) are retained alongside the report.
