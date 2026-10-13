# RM-CGE V2 Pilot Report

## Protocol and integrity

The frozen protocol completed 27 Dataset × Variant × Seed runs. Test evaluation is disabled. Implementation Freeze: `08d7a16b9f2f9201a7ec12c9ce5dab979c7a2f02`.
Before Freeze, the full regression command recorded 461 passed / 1 skipped. The ele-fashion RTX smoke peaked at 12100887552 allocated bytes.
After the post-Freeze reporting and artifact-schema fixes, the full regression command recorded 467 passed / 0 skipped.
The seed-batch gates verified 27 runs, finite histories, and Test disabled: True.
The primary checkpoint is selected by Validation Accuracy. Best-F1 checkpoints and per-epoch Train/Validation CE, accuracy, and Macro-F1 are retained for diagnosis.

## Validation results

Values are mean ± sample SD over seeds 42, 43, and 44. They are descriptive; no significance test is performed.

| Dataset | Variant | Val Accuracy | Val Macro-F1 | Val CE | Train CE (train mode) | Train CE (eval mode) | Train Macro-F1 | Train−Val Acc gap | Best Epoch | Wall time (s) | Params | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | V0 | 0.5603 ± 0.0038 | 0.4825 ± 0.0155 | 1.4087 ± 0.0401 | 0.7718 ± 0.1194 | 0.7462 ± 0.1213 | 0.7390 ± 0.0685 | 0.1802 ± 0.0516 | 83.3 ± 18.1 | 20.1 ± 3.4 | 1,454,025 | 2353 |
| Movies | V1 | 0.5598 ± 0.0036 | 0.4862 ± 0.0102 | 1.4167 ± 0.0549 | 0.7073 ± 0.1026 | 0.6690 ± 0.0967 | 0.7813 ± 0.0496 | 0.2113 ± 0.0352 | 94.0 ± 12.5 | 27.0 ± 2.6 | 1,611,450 | 2703 |
| Movies | V2 | 0.5598 ± 0.0052 | 0.4795 ± 0.0141 | 1.4143 ± 0.0394 | 0.7746 ± 0.2172 | 0.7446 ± 0.2250 | 0.7349 ± 0.1051 | 0.1812 ± 0.0753 | 85.7 ± 29.4 | 35.0 ± 8.9 | 1,611,450 | 2777 |
| Grocery | V0 | 0.8316 ± 0.0039 | 0.7606 ± 0.0204 | 0.6682 ± 0.0296 | 0.2403 ± 0.0496 | 0.2247 ± 0.0512 | 0.9157 ± 0.0221 | 0.1045 ± 0.0163 | 86.3 ± 15.6 | 19.9 ± 2.8 | 1,454,025 | 2218 |
| Grocery | V1 | 0.8347 ± 0.0032 | 0.7653 ± 0.0112 | 0.6619 ± 0.0411 | 0.2417 ± 0.0246 | 0.2173 ± 0.0289 | 0.9163 ± 0.0131 | 0.1041 ± 0.0125 | 84.0 ± 6.1 | 25.2 ± 1.2 | 1,611,450 | 2578 |
| Grocery | V2 | 0.8347 ± 0.0047 | 0.7640 ± 0.0204 | 0.6521 ± 0.0166 | 0.2648 ± 0.1005 | 0.2495 ± 0.0962 | 0.9048 ± 0.0469 | 0.0947 ± 0.0257 | 81.7 ± 30.2 | 33.3 ± 8.8 | 1,611,450 | 2652 |
| ele-fashion | V0 | 0.8735 ± 0.0014 | 0.7426 ± 0.0154 | 0.4271 ± 0.0245 | 0.2569 ± 0.0641 | 0.2472 ± 0.0609 | 0.8522 ± 0.0463 | 0.0358 ± 0.0164 | 181.3 ± 45.8 | 126.5 ± 27.3 | 1,189,825 | 9082 |
| ele-fashion | V1 | 0.8742 ± 0.0002 | 0.7476 ± 0.0017 | 0.4076 ± 0.0035 | 0.2873 ± 0.0222 | 0.2731 ± 0.0275 | 0.8310 ± 0.0160 | 0.0276 ± 0.0078 | 141.0 ± 21.3 | 140.4 ± 17.2 | 1,347,250 | 11120 |
| ele-fashion | V2 | 0.8734 ± 0.0008 | 0.7416 ± 0.0095 | 0.4225 ± 0.0331 | 0.2679 ± 0.0768 | 0.2644 ± 0.0829 | 0.8340 ± 0.0611 | 0.0323 ± 0.0240 | 145.0 ± 53.8 | 204.4 ± 66.9 | 1,347,250 | 11539 |

## Paired differences

Each row uses matched Dataset × Seed pairs. Positive means the left-hand variant scored higher. These three-seed summaries are not significance claims.

| Dataset | Metric | Contrast | Mean difference ± sample SD | Seed differences |
|---|---|---|---:|---|
| Movies | val_accuracy | V1-V0 | -0.0005 ± 0.0014 | -0.0018, +0.0009, -0.0006 |
| Movies | val_accuracy | V2-V1 | +0.0000 ± 0.0057 | +0.0066, -0.0039, -0.0027 |
| Movies | val_accuracy | V2-V0 | -0.0005 ± 0.0046 | +0.0048, -0.0030, -0.0033 |
| Movies | val_macro_f1 | V1-V0 | +0.0037 ± 0.0085 | -0.0004, +0.0135, -0.0019 |
| Movies | val_macro_f1 | V2-V1 | -0.0068 ± 0.0126 | +0.0061, -0.0191, -0.0072 |
| Movies | val_macro_f1 | V2-V0 | -0.0030 ± 0.0077 | +0.0057, -0.0057, -0.0091 |
| Movies | val_ce | V1-V0 | +0.0080 ± 0.0740 | -0.0745, +0.0685, +0.0300 |
| Movies | val_ce | V2-V1 | -0.0024 ± 0.0942 | +0.0701, -0.1090, +0.0315 |
| Movies | val_ce | V2-V0 | +0.0056 ± 0.0517 | -0.0044, -0.0405, +0.0615 |
| Grocery | val_accuracy | V1-V0 | +0.0030 ± 0.0041 | -0.0012, +0.0032, +0.0070 |
| Grocery | val_accuracy | V2-V1 | +0.0000 ± 0.0024 | +0.0006, +0.0020, -0.0026 |
| Grocery | val_accuracy | V2-V0 | +0.0030 ± 0.0032 | -0.0006, +0.0053, +0.0044 |
| Grocery | val_macro_f1 | V1-V0 | +0.0047 ± 0.0105 | +0.0017, -0.0040, +0.0164 |
| Grocery | val_macro_f1 | V2-V1 | -0.0013 ± 0.0143 | -0.0068, +0.0149, -0.0119 |
| Grocery | val_macro_f1 | V2-V0 | +0.0034 ± 0.0081 | -0.0051, +0.0109, +0.0045 |
| Grocery | val_ce | V1-V0 | -0.0064 ± 0.0298 | +0.0167, -0.0401, +0.0043 |
| Grocery | val_ce | V2-V1 | -0.0098 ± 0.0542 | -0.0188, +0.0484, -0.0589 |
| Grocery | val_ce | V2-V0 | -0.0161 ± 0.0337 | -0.0021, +0.0083, -0.0546 |
| ele-fashion | val_accuracy | V1-V0 | +0.0007 ± 0.0015 | -0.0009, +0.0020, +0.0009 |
| ele-fashion | val_accuracy | V2-V1 | -0.0008 ± 0.0010 | -0.0014, -0.0013, +0.0003 |
| ele-fashion | val_accuracy | V2-V0 | -0.0001 ± 0.0019 | -0.0024, +0.0007, +0.0012 |
| ele-fashion | val_macro_f1 | V1-V0 | +0.0050 ± 0.0171 | -0.0118, +0.0223, +0.0044 |
| ele-fashion | val_macro_f1 | V2-V1 | -0.0060 ± 0.0095 | -0.0106, -0.0123, +0.0050 |
| ele-fashion | val_macro_f1 | V2-V0 | -0.0010 ± 0.0186 | -0.0225, +0.0099, +0.0094 |
| ele-fashion | val_ce | V1-V0 | -0.0195 ± 0.0212 | -0.0422, -0.0001, -0.0162 |
| ele-fashion | val_ce | V2-V1 | +0.0148 ± 0.0352 | -0.0086, -0.0021, +0.0553 |
| ele-fashion | val_ce | V2-V0 | -0.0047 ± 0.0450 | -0.0508, -0.0022, +0.0391 |

## Best-Accuracy versus Best-F1 checkpoint

Differences below are Best-F1 minus Best-Accuracy checkpoint metrics, averaged over three seeds; they describe a checkpoint tradeoff, not a second model-selection protocol.

| Dataset | Val Accuracy difference | Val Macro-F1 difference | Val CE difference | Mean epoch difference |
|---|---:|---:|---:|---:|
| Movies | -0.0146 ± 0.0092 | +0.0173 ± 0.0110 | +0.0453 ± 0.0801 | +7.3 ± 23.3 |
| Grocery | -0.0027 ± 0.0029 | +0.0034 ± 0.0048 | +0.0177 ± 0.0224 | +8.7 ± 14.7 |
| ele-fashion | -0.0080 ± 0.0038 | +0.0146 ± 0.0057 | +0.0087 ± 0.0128 | +20.7 ± 11.2 |

`Train CE (train mode)` is measured before the optimizer update with Dropout active; `Train CE (eval mode)`, Train Accuracy/F1, and Validation metrics reuse the post-update evaluation embeddings. All epoch-level histories are in `data/epoch_history.csv`.

## Implementation and evidence boundary

V0 calls the frozen V7A `a0_raw` forward directly. V1/V2 share parameter names, shapes, initialization, and classifier initialization; forcing V2's edge weights to one matches V1. The unit-edge test and disabled-path test are recorded in the implementation test suite.

V1−V0 measures the combined fixed cross-modal, KNN, expert-coefficient, and node-router additions. V2−V1 measures enabling the edge-conditioned physical path; the scorer is inactive in V1. Neither contrast is a pure capacity-matched causal effect.

No Test labels or Test metrics are used. Historical V7A/V7B/RM-CGE V1 metrics are background only and do not set a score threshold.
