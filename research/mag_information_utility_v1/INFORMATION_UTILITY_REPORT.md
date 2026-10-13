# MAG Information Utility V1 — Results

All reported endpoint results use the Validation Accuracy selected checkpoint. Values are mean ± sample SD across the three paired training seeds (42/43/44). The seed is the training replication unit; Validation nodes are not treated as independent experimental replicates. No Test evaluation was performed.

## Validation performance

| Dataset | Treatment | Accuracy (%) | Macro-F1 (%) | Validation CE | Parameters |
|---|---:|---:|---:|---:|---:|
| Movies | I0 | 52.38 ± 0.73 | 40.93 ± 0.97 | 1.4837 ± 0.0529 | 742164 |
| Movies | I1 | 53.96 ± 0.19 | 44.04 ± 0.55 | 1.4475 ± 0.0481 | 742164 |
| Movies | I2 | 54.26 ± 0.40 | 44.19 ± 0.33 | 1.4654 ± 0.0654 | 742164 |
| Movies | I3 | 54.21 ± 0.38 | 44.44 ± 0.43 | 1.4729 ± 0.0513 | 742164 |
| Movies | I4 | 53.94 ± 0.32 | 43.83 ± 0.55 | 1.4716 ± 0.0691 | 742164 |
| Grocery | I0 | 78.01 ± 0.56 | 69.37 ± 1.19 | 0.8047 ± 0.0077 | 742164 |
| Grocery | I1 | 81.34 ± 0.25 | 72.63 ± 1.46 | 0.7242 ± 0.0015 | 742164 |
| Grocery | I2 | 81.87 ± 0.36 | 73.38 ± 0.81 | 0.7105 ± 0.0251 | 742164 |
| Grocery | I3 | 81.82 ± 0.38 | 73.44 ± 1.15 | 0.7113 ± 0.0153 | 742164 |
| Grocery | I4 | 81.74 ± 0.15 | 73.20 ± 1.17 | 0.7058 ± 0.0096 | 742164 |
| ele-fashion | I0 | 87.17 ± 0.03 | 67.63 ± 0.19 | 0.4270 ± 0.0095 | 610060 |
| ele-fashion | I1 | 87.39 ± 0.17 | 67.89 ± 0.41 | 0.4133 ± 0.0096 | 610060 |
| ele-fashion | I2 | 87.48 ± 0.04 | 68.50 ± 0.40 | 0.4125 ± 0.0036 | 610060 |
| ele-fashion | I3 | 87.64 ± 0.10 | 68.21 ± 0.50 | 0.4145 ± 0.0220 | 610060 |
| ele-fashion | I4 | 87.42 ± 0.03 | 68.32 ± 0.44 | 0.4095 ± 0.0123 | 610060 |

## Paired information contrasts

Positive Accuracy/Macro-F1 deltas favor the candidate; negative CE deltas favor the candidate. `positive_seed_count` counts positive numerical differences regardless of whether that direction improves the metric. Each dataset has three paired seed differences; summaries are descriptive and do not establish statistical significance.

| Dataset | Contrast | Metric | Mean paired delta | Sample SD | Positive seeds |
|---|---|---|---:|---:|---:|
| Movies | I1-I0 | validation_accuracy | 1.58 ± 0.59 | 3/3 |
| Movies | I1-I0 | validation_macro_f1 | 3.12 ± 1.17 | 3/3 |
| Movies | I1-I0 | validation_cross_entropy | -0.0363 ± 0.0055 | 0/3 |
| Movies | I1-I0 | train_validation_accuracy_gap | 0.0226 ± 0.0428 | 2/3 |
| Movies | I1-I0 | training_seconds | -0.2519 ± 0.5266 | 1/3 |
| Movies | I1-I0 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| Movies | I2-I1 | validation_accuracy | 0.30 ± 0.24 | 3/3 |
| Movies | I2-I1 | validation_macro_f1 | 0.15 ± 0.68 | 2/3 |
| Movies | I2-I1 | validation_cross_entropy | 0.0179 ± 0.0207 | 2/3 |
| Movies | I2-I1 | train_validation_accuracy_gap | 0.0355 ± 0.0506 | 2/3 |
| Movies | I2-I1 | training_seconds | 0.3917 ± 0.5245 | 2/3 |
| Movies | I2-I1 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| Movies | I3-I2 | validation_accuracy | -0.05 ± 0.31 | 1/3 |
| Movies | I3-I2 | validation_macro_f1 | 0.25 ± 0.74 | 2/3 |
| Movies | I3-I2 | validation_cross_entropy | 0.0076 ± 0.0350 | 1/3 |
| Movies | I3-I2 | train_validation_accuracy_gap | 0.0166 ± 0.0368 | 2/3 |
| Movies | I3-I2 | training_seconds | 0.1831 ± 0.3964 | 2/3 |
| Movies | I3-I2 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| Movies | I3-I4 | validation_accuracy | 0.27 ± 0.06 | 3/3 |
| Movies | I3-I4 | validation_macro_f1 | 0.61 ± 0.80 | 2/3 |
| Movies | I3-I4 | validation_cross_entropy | 0.0013 ± 0.0320 | 2/3 |
| Movies | I3-I4 | train_validation_accuracy_gap | 0.0170 ± 0.0470 | 2/3 |
| Movies | I3-I4 | training_seconds | 0.1753 ± 0.7102 | 2/3 |
| Movies | I3-I4 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| Grocery | I1-I0 | validation_accuracy | 3.33 ± 0.32 | 3/3 |
| Grocery | I1-I0 | validation_macro_f1 | 3.26 ± 0.28 | 3/3 |
| Grocery | I1-I0 | validation_cross_entropy | -0.0805 ± 0.0084 | 0/3 |
| Grocery | I1-I0 | train_validation_accuracy_gap | 0.0263 ± 0.0248 | 3/3 |
| Grocery | I1-I0 | training_seconds | 0.9997 ± 0.5353 | 3/3 |
| Grocery | I1-I0 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| Grocery | I2-I1 | validation_accuracy | 0.54 ± 0.17 | 3/3 |
| Grocery | I2-I1 | validation_macro_f1 | 0.75 ± 0.68 | 3/3 |
| Grocery | I2-I1 | validation_cross_entropy | -0.0137 ± 0.0255 | 1/3 |
| Grocery | I2-I1 | train_validation_accuracy_gap | -0.0078 ± 0.0269 | 1/3 |
| Grocery | I2-I1 | training_seconds | 0.2034 ± 0.7558 | 1/3 |
| Grocery | I2-I1 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| Grocery | I3-I2 | validation_accuracy | -0.06 ± 0.11 | 1/3 |
| Grocery | I3-I2 | validation_macro_f1 | 0.06 ± 0.67 | 2/3 |
| Grocery | I3-I2 | validation_cross_entropy | 0.0007 ± 0.0239 | 1/3 |
| Grocery | I3-I2 | train_validation_accuracy_gap | 0.0030 ± 0.0149 | 2/3 |
| Grocery | I3-I2 | training_seconds | 0.0262 ± 0.3115 | 1/3 |
| Grocery | I3-I2 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| Grocery | I3-I4 | validation_accuracy | 0.08 ± 0.28 | 2/3 |
| Grocery | I3-I4 | validation_macro_f1 | 0.24 ± 0.80 | 2/3 |
| Grocery | I3-I4 | validation_cross_entropy | 0.0055 ± 0.0243 | 2/3 |
| Grocery | I3-I4 | train_validation_accuracy_gap | 0.0072 ± 0.0225 | 2/3 |
| Grocery | I3-I4 | training_seconds | 0.1799 ± 0.7455 | 2/3 |
| Grocery | I3-I4 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| ele-fashion | I1-I0 | validation_accuracy | 0.22 ± 0.15 | 3/3 |
| ele-fashion | I1-I0 | validation_macro_f1 | 0.26 ± 0.58 | 2/3 |
| ele-fashion | I1-I0 | validation_cross_entropy | -0.0137 ± 0.0128 | 0/3 |
| ele-fashion | I1-I0 | train_validation_accuracy_gap | -0.0066 ± 0.0191 | 1/3 |
| ele-fashion | I1-I0 | training_seconds | -2.4706 ± 5.0753 | 2/3 |
| ele-fashion | I1-I0 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| ele-fashion | I2-I1 | validation_accuracy | 0.09 ± 0.15 | 2/3 |
| ele-fashion | I2-I1 | validation_macro_f1 | 0.60 ± 0.41 | 3/3 |
| ele-fashion | I2-I1 | validation_cross_entropy | -0.0008 ± 0.0125 | 2/3 |
| ele-fashion | I2-I1 | train_validation_accuracy_gap | 0.0081 ± 0.0188 | 2/3 |
| ele-fashion | I2-I1 | training_seconds | 2.2089 ± 5.1404 | 2/3 |
| ele-fashion | I2-I1 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| ele-fashion | I3-I2 | validation_accuracy | 0.17 ± 0.09 | 3/3 |
| ele-fashion | I3-I2 | validation_macro_f1 | -0.29 ± 0.87 | 1/3 |
| ele-fashion | I3-I2 | validation_cross_entropy | 0.0020 ± 0.0193 | 1/3 |
| ele-fashion | I3-I2 | train_validation_accuracy_gap | -0.0136 ± 0.0072 | 0/3 |
| ele-fashion | I3-I2 | training_seconds | -1.0850 ± 3.6780 | 2/3 |
| ele-fashion | I3-I2 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |
| ele-fashion | I3-I4 | validation_accuracy | 0.23 ± 0.11 | 3/3 |
| ele-fashion | I3-I4 | validation_macro_f1 | -0.12 ± 0.87 | 1/3 |
| ele-fashion | I3-I4 | validation_cross_entropy | 0.0049 ± 0.0173 | 2/3 |
| ele-fashion | I3-I4 | train_validation_accuracy_gap | -0.0085 ± 0.0153 | 1/3 |
| ele-fashion | I3-I4 | training_seconds | -1.4502 ± 5.5721 | 1/3 |
| ele-fashion | I3-I4 | cuda_peak_allocated_bytes | 0.0000 ± 0.0000 | 0/3 |

## Interpretation boundaries

- I1−I0 estimates the value of adding physical neighborhood channels within this fixed probe.
- I2−I1 changes where and how the same physical information is processed across modalities. It is an inductive-bias comparison, not additional raw information.
- I3−I2 opens real Text and Visual semantic channels, so it changes the active input set and is not a pure topology-only contrast.
- I3−I4 is the topology-content comparison: the model, slots, initialization, optimizer, and degrees are paired while the semantic graph topology changes. It uses one frozen random topology per view and dataset, so the null is one topology sample.
- Equal total parameter counts do not imply equal effective capacity: closed input slots can receive no input-related gradient.
- Historical V7B.1 R0 Raw is shown only as a capability reference. It is not capacity-matched to this probe.

## Train/Validation fit and checkpoint selection

The primary endpoint uses the Best-Accuracy checkpoint. Train metrics below are evaluated at that same checkpoint; the separate Best-Macro-F1 checkpoint is not mixed into the primary tables.

| Dataset | Treatment | Train Accuracy (%) | Validation Accuracy (%) | Train−Validation Accuracy gap (pp) |
|---|---|---:|---:|---:|
| Movies | I0 | 65.56 ± 6.96 | 52.38 ± 0.73 | 13.18 ± 7.51 |
| Movies | I1 | 69.40 ± 5.87 | 53.96 ± 0.19 | 15.44 ± 6.01 |
| Movies | I2 | 73.24 ± 8.14 | 54.26 ± 0.40 | 18.98 ± 8.37 |
| Movies | I3 | 74.85 ± 6.15 | 54.21 ± 0.38 | 20.64 ± 5.94 |
| Movies | I4 | 72.88 ± 8.99 | 53.94 ± 0.32 | 18.94 ± 8.95 |
| Grocery | I0 | 89.38 ± 1.81 | 78.01 ± 0.56 | 11.37 ± 1.27 |
| Grocery | I1 | 95.34 ± 1.21 | 81.34 ± 0.25 | 14.00 ± 1.34 |
| Grocery | I2 | 95.09 ± 1.66 | 81.87 ± 0.36 | 13.22 ± 1.35 |
| Grocery | I3 | 95.33 ± 0.90 | 81.82 ± 0.38 | 13.52 ± 0.70 |
| Grocery | I4 | 94.53 ± 1.59 | 81.74 ± 0.15 | 12.80 ± 1.62 |
| ele-fashion | I0 | 91.33 ± 0.98 | 87.17 ± 0.03 | 4.17 ± 1.00 |
| ele-fashion | I1 | 90.90 ± 1.97 | 87.39 ± 0.17 | 3.51 ± 2.00 |
| ele-fashion | I2 | 91.80 ± 0.29 | 87.48 ± 0.04 | 4.33 ± 0.25 |
| ele-fashion | I3 | 90.61 ± 0.69 | 87.64 ± 0.10 | 2.96 ± 0.62 |
| ele-fashion | I4 | 91.23 ± 0.97 | 87.42 ± 0.03 | 3.81 ± 0.96 |

| Dataset | Treatment | Best-F1 minus Best-Accuracy: Accuracy (pp) | Macro-F1 (pp) | CE |
|---|---|---:|---:|---:|
| Movies | I0 | -0.79 ± 0.18 | 1.49 ± 0.87 | 0.0388 ± 0.0356 |
| Movies | I1 | -1.53 ± 0.32 | 1.13 ± 0.36 | 0.0537 ± 0.0271 |
| Movies | I2 | -1.27 ± 0.46 | 1.70 ± 0.14 | 0.0217 ± 0.0419 |
| Movies | I3 | -1.18 ± 0.48 | 1.56 ± 0.49 | 0.0129 ± 0.0097 |
| Movies | I4 | -1.57 ± 0.62 | 1.75 ± 0.43 | 0.0699 ± 0.0208 |
| Grocery | I0 | -0.16 ± 0.11 | 0.98 ± 0.50 | 0.0246 ± 0.0259 |
| Grocery | I1 | -0.45 ± 0.26 | 0.61 ± 0.27 | 0.0372 ± 0.0025 |
| Grocery | I2 | -0.29 ± 0.22 | 0.37 ± 0.10 | 0.0125 ± 0.0150 |
| Grocery | I3 | -0.40 ± 0.21 | 0.41 ± 0.06 | 0.0296 ± 0.0265 |
| Grocery | I4 | -0.43 ± 0.21 | 0.40 ± 0.45 | 0.0386 ± 0.0095 |
| ele-fashion | I0 | -0.68 ± 0.33 | 1.15 ± 0.34 | 0.0089 ± 0.0033 |
| ele-fashion | I1 | -0.20 ± 0.06 | 1.04 ± 0.15 | 0.0042 ± 0.0153 |
| ele-fashion | I2 | -0.65 ± 0.39 | 0.76 ± 0.40 | 0.0083 ± 0.0108 |
| ele-fashion | I3 | -0.66 ± 0.22 | 1.35 ± 0.55 | -0.0128 ± 0.0217 |
| ele-fashion | I4 | -0.30 ± 0.21 | 0.92 ± 0.08 | 0.0064 ± 0.0067 |

## Physical and semantic neighborhood diagnostics

See `GRAPH_SIGNAL_REPORT.md` and `data/graph_structure_stats.csv` for feature geometry, degree, connectivity, and physical-overlap diagnostics. Graph homophily is reported only for Train–Train edges and is not a predictive utility estimate. Degree matching does not match component structure or physical overlap; see `data/random_graph_audit.json`.

## Train/Validation fit and checkpoint criteria

`data/runs.csv` and `data/epoch_history.csv` retain Train/Validation CE, Accuracy, Macro-F1, and the Train–Validation Accuracy gap. `data/checkpoint_tradeoffs.csv` compares independently selected Best-Accuracy and Best-Macro-F1 checkpoints; endpoint claims above use only Best-Accuracy checkpoints.

## Historical R0 Raw reference

| Dataset | Treatment | Probe−R0 Accuracy (pp) | Probe−R0 Macro-F1 (pp) | Probe parameters | R0 parameters |
|---|---|---:|---:|---:|---:|
| Movies | I0 | -3.44 ± 1.04 | -6.91 ± 2.94 | 742164 | 1454025 |
| Movies | I1 | -1.86 ± 0.72 | -3.79 ± 1.85 | 742164 | 1454025 |
| Movies | I2 | -1.56 ± 0.83 | -3.64 ± 2.32 | 742164 | 1454025 |
| Movies | I3 | -1.61 ± 0.93 | -3.39 ± 1.58 | 742164 | 1454025 |
| Movies | I4 | -1.88 ± 0.87 | -4.01 ± 2.13 | 742164 | 1454025 |
| Grocery | I0 | -5.10 ± 0.83 | -6.44 ± 0.51 | 742164 | 1454025 |
| Grocery | I1 | -1.77 ± 0.57 | -3.18 ± 0.28 | 742164 | 1454025 |
| Grocery | I2 | -1.23 ± 0.73 | -2.43 ± 0.95 | 742164 | 1454025 |
| Grocery | I3 | -1.29 ± 0.78 | -2.38 ± 0.73 | 742164 | 1454025 |
| Grocery | I4 | -1.37 ± 0.55 | -2.62 ± 0.74 | 742164 | 1454025 |
| ele-fashion | I0 | -0.28 ± 0.18 | -6.94 ± 1.61 | 610060 | 1189825 |
| ele-fashion | I1 | -0.06 ± 0.13 | -6.68 ± 2.01 | 610060 | 1189825 |
| ele-fashion | I2 | 0.03 ± 0.21 | -6.08 ± 2.03 | 610060 | 1189825 |
| ele-fashion | I3 | 0.20 ± 0.13 | -6.37 ± 1.45 | 610060 | 1189825 |
| ele-fashion | I4 | -0.03 ± 0.21 | -6.25 ± 2.09 | 610060 | 1189825 |

R0 Raw uses a different architecture and capacity; these gaps calibrate probe strength and do not isolate graph utility. Per-seed reference values are retained in `data/r0_reference_comparison.csv`.
