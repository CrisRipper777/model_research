# MAG View-Collaborative Experts V1: Pilot Report

## Protocol and provenance

- Branch: `exp/mag_view_collaborative_experts_v1`; Implementation Freeze commit: `f981a5f0164b1ed73c379724139e75c84abb0c98`; formal-run source commit: `608dbe406e698ce5dea7facd1cdcdb927b6943a7`.
- Runs: 27/27; epoch-history rows: 3715; retained failures: 3.
- Protocol: `unified_full_graph_nc_v1`; Test evaluation disabled and Test labels masked before NC training/evaluation.
- Device: NVIDIA GeForce RTX 3090; physical GPU 1; PyTorch `2.4.0+cu121`, CUDA build `12.1`.
- Results are Validation-only and paired by Dataset × Seed. The historical reports are context, not the acceptance target.

## Retained failed attempts and batch interruption

The three entries in `data/failures.json` are retained execution history, not completed metric rows. Movies/A0/seed42 attempt 1 stopped before training because checkpoint metadata was absent. Attempt 2 trained 103 epochs and wrote both checkpoints, then failed in post-training diagnostics when a node mask was applied to a tensor with an expert-leading axis; the bug was fixed and formal attempt 3 completed the cell. After all nine seed43 cells completed, the GPU-idle guard stopped the batch before seed44 because an independent LLH worker (PID 27585) used 816 MiB on physical GPU 1. It was not interrupted. After it exited, the runner skipped verified seed43 results and completed seed44. No completed checkpoint was overwritten.
Full tracebacks, paths, timestamps and resolution evidence remain in `data/failures.json`; preflight-only engineering corrections are detailed in `data/preflight_attempts.json`.

## Best Accuracy checkpoint

### Validation Accuracy

| Dataset | Condition | Validation Accuracy | Best epochs |
|---|---|---|---|
| Movies | A0_legacy | 55.979 ± 0.420% | 73, 108, 98 |
| Movies | A1_separate_intra | 56.029 ± 0.520% | 107, 43, 81 |
| Movies | A2_cross_source | 55.959 ± 0.241% | 68, 60, 114 |
| Grocery | A0_legacy | 83.367 ± 0.146% | 78, 76, 85 |
| Grocery | A1_separate_intra | 83.455 ± 0.356% | 74, 81, 88 |
| Grocery | A2_cross_source | 83.514 ± 0.456% | 65, 96, 112 |
| ele-fashion | A0_legacy | 87.488 ± 0.129% | 151, 161, 141 |
| ele-fashion | A1_separate_intra | 87.525 ± 0.089% | 120, 129, 211 |
| ele-fashion | A2_cross_source | 87.665 ± 0.099% | 152, 152, 181 |

### Validation Macro-F1

| Dataset | Condition | Validation Macro-F1 | Best epochs |
|---|---|---|---|
| Movies | A0_legacy | 47.447 ± 1.220% | 73, 108, 98 |
| Movies | A1_separate_intra | 47.845 ± 2.226% | 107, 43, 81 |
| Movies | A2_cross_source | 47.953 ± 0.534% | 68, 60, 114 |
| Grocery | A0_legacy | 76.151 ± 0.786% | 78, 76, 85 |
| Grocery | A1_separate_intra | 76.704 ± 1.419% | 74, 81, 88 |
| Grocery | A2_cross_source | 76.276 ± 0.942% | 65, 96, 112 |
| ele-fashion | A0_legacy | 74.630 ± 0.798% | 151, 161, 141 |
| ele-fashion | A1_separate_intra | 74.936 ± 1.507% | 120, 129, 211 |
| ele-fashion | A2_cross_source | 75.124 ± 0.939% | 152, 152, 181 |

### Validation CE

| Dataset | Condition | Validation CE | Best epochs |
|---|---|---|---|
| Movies | A0_legacy | 1.41944 ± 0.06262 | 73, 108, 98 |
| Movies | A1_separate_intra | 1.38201 ± 0.05317 | 107, 43, 81 |
| Movies | A2_cross_source | 1.39239 ± 0.05291 | 68, 60, 114 |
| Grocery | A0_legacy | 0.65664 ± 0.03191 | 78, 76, 85 |
| Grocery | A1_separate_intra | 0.65182 ± 0.02806 | 74, 81, 88 |
| Grocery | A2_cross_source | 0.66589 ± 0.05474 | 65, 96, 112 |
| ele-fashion | A0_legacy | 0.41361 ± 0.00545 | 151, 161, 141 |
| ele-fashion | A1_separate_intra | 0.43064 ± 0.05251 | 120, 129, 211 |
| ele-fashion | A2_cross_source | 0.43025 ± 0.02414 | 152, 152, 181 |

## Best Macro-F1 checkpoint

| Dataset | Condition | Macro-F1 | Accuracy | Validation CE | Best epochs |
|---|---|---|---|---|---|
| Movies | A0_legacy | 50.075 ± 0.078% | 54.559 ± 0.584% | 1.45346 ± 0.03831 | 95, 103, 107 |
| Movies | A1_separate_intra | 49.566 ± 0.923% | 55.209 ± 0.449% | 1.39331 ± 0.05012 | 100, 70, 70 |
| Movies | A2_cross_source | 49.571 ± 0.742% | 54.759 ± 0.618% | 1.44018 ± 0.03743 | 81, 88, 116 |
| Grocery | A0_legacy | 76.586 ± 1.085% | 82.850 ± 0.321% | 0.69854 ± 0.03142 | 99, 105, 109 |
| Grocery | A1_separate_intra | 76.881 ± 1.279% | 83.221 ± 0.500% | 0.67036 ± 0.04216 | 74, 89, 111 |
| Grocery | A2_cross_source | 77.168 ± 1.306% | 83.289 ± 0.476% | 0.69545 ± 0.07167 | 94, 95, 132 |
| ele-fashion | A0_legacy | 76.096 ± 0.008% | 86.826 ± 0.348% | 0.42050 ± 0.00633 | 170, 181, 166 |
| ele-fashion | A1_separate_intra | 76.138 ± 0.420% | 87.184 ± 0.381% | 0.43616 ± 0.04770 | 150, 156, 211 |
| ele-fashion | A2_cross_source | 76.481 ± 0.250% | 86.918 ± 0.384% | 0.43515 ± 0.01596 | 179, 182, 209 |

## Paired contrasts

Values are left minus right; per-seed rows are in `data/paired_comparisons.csv`. These are descriptive three-seed comparisons, not significance tests.

| Dataset | Contrast | Accuracy Δ | Macro-F1 Δ | Acc positive | F1 positive |
|---|---|---|---|---|---|
| Movies | A1_separate_intra-A0_legacy | 0.050 ± 0.135 pp | 0.399 ± 2.528 pp | 2/3 | 2/3 |
| Movies | A2_cross_source-A1_separate_intra | -0.070 ± 0.332 pp | 0.108 ± 2.177 pp | 1/3 | 1/3 |
| Movies | A2_cross_source-A0_legacy | -0.020 ± 0.283 pp | 0.506 ± 1.745 pp | 1/3 | 2/3 |
| Grocery | A1_separate_intra-A0_legacy | 0.088 ± 0.250 pp | 0.553 ± 0.634 pp | 2/3 | 2/3 |
| Grocery | A2_cross_source-A1_separate_intra | 0.059 ± 0.106 pp | -0.428 ± 1.319 pp | 2/3 | 2/3 |
| Grocery | A2_cross_source-A0_legacy | 0.146 ± 0.338 pp | 0.125 ± 0.916 pp | 2/3 | 2/3 |
| ele-fashion | A1_separate_intra-A0_legacy | 0.038 ± 0.183 pp | 0.306 ± 2.071 pp | 2/3 | 1/3 |
| ele-fashion | A2_cross_source-A1_separate_intra | 0.140 ± 0.043 pp | 0.188 ± 0.893 pp | 3/3 | 1/3 |
| ele-fashion | A2_cross_source-A0_legacy | 0.177 ± 0.159 pp | 0.494 ± 1.708 pp | 2/3 | 2/3 |

## Train/Validation, capacity, and resource profile

| Dataset | Condition | Best-Acc Train−Val Acc/F1/CE gap means | Best-F1 Train−Val Acc/F1/CE gap means | Params | Trainable Params | Wall s | Peak allocated GiB |
|---|---|---|---|---|---|---|---|
| Movies | A0_legacy | 0.1922 / 0.2822 / -0.7159 | 0.2430 / 0.3091 / -0.8147 | 1715691 | 1715691 | 30.4 | 2.68 |
| Movies | A1_separate_intra | 0.1535 / 0.2150 / -0.5488 | 0.1661 / 0.2148 / -0.5764 | 1715691 | 1715691 | 26.4 | 2.48 |
| Movies | A2_cross_source | 0.1711 / 0.2401 / -0.6056 | 0.2190 / 0.2842 / -0.7523 | 1715691 | 1715691 | 31.2 | 2.62 |
| Grocery | A0_legacy | 0.0997 / 0.1470 / -0.4240 | 0.1331 / 0.1841 / -0.5495 | 1715691 | 1715691 | 27.6 | 2.56 |
| Grocery | A1_separate_intra | 0.0961 / 0.1402 / -0.4110 | 0.1098 / 0.1537 / -0.4615 | 1715691 | 1715691 | 27.8 | 2.35 |
| Grocery | A2_cross_source | 0.1041 / 0.1530 / -0.4510 | 0.1223 / 0.1710 / -0.5318 | 1715691 | 1715691 | 34.0 | 2.51 |
| ele-fashion | A0_legacy | 0.0318 / 0.0943 / -0.1548 | 0.0534 / 0.1240 / -0.2014 | 1451491 | 1451491 | 184.5 | 11.12 |
| ele-fashion | A1_separate_intra | 0.0347 / 0.0972 / -0.1819 | 0.0494 / 0.1205 / -0.2200 | 1451491 | 1451491 | 191.8 | 9.92 |
| ele-fashion | A2_cross_source | 0.0315 / 0.0953 / -0.1742 | 0.0600 / 0.1340 / -0.2397 | 1451491 | 1451491 | 227.6 | 10.79 |

## View and Router diagnostics

Scale and usage values describe what the frozen checkpoint computes, not causal task utility. Details are retained in `data/view_diagnostics.csv` and `data/router_diagnostics.csv`.

| Dataset | Condition | View | Input RMS | Calibrated RMS | Expert correction RMS | Weighted contribution RMS | Gamma/strength |
|---|---|---|---|---|---|---|---|
| Movies | A0_legacy | physical | 0.75469 | 0.84327 | 0.43075 | 0.43075 | — |
| Movies | A0_legacy | same | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| Movies | A0_legacy | cross | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| Movies | A1_separate_intra | physical | 0.78208 | 0.81931 | 0.02644 | 0.42315 | 0.46112 |
| Movies | A1_separate_intra | same | 0.87700 | 1.15909 | 0.44531 | 0.09457 | 0.07909 |
| Movies | A1_separate_intra | cross | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| Movies | A2_cross_source | physical | 0.77319 | 0.47925 | 0.02593 | 0.41636 | 0.46115 |
| Movies | A2_cross_source | same | 0.87030 | 1.15906 | 0.47025 | 0.09610 | 0.07923 |
| Movies | A2_cross_source | cross | 0.31502 | 0.54672 | 0.10173 | 0.02712 | 0.04897 |
| Grocery | A0_legacy | physical | 0.78489 | 1.04063 | 0.88686 | 0.88686 | — |
| Grocery | A0_legacy | same | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| Grocery | A0_legacy | cross | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| Grocery | A1_separate_intra | physical | 0.78642 | 1.02556 | 0.04148 | 0.89749 | 0.59861 |
| Grocery | A1_separate_intra | same | 0.89715 | 1.16850 | 0.48896 | 0.09807 | 0.07978 |
| Grocery | A1_separate_intra | cross | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| Grocery | A2_cross_source | physical | 0.78175 | 0.71812 | 0.04236 | 0.90947 | 0.60183 |
| Grocery | A2_cross_source | same | 0.89311 | 1.16932 | 0.56062 | 0.10035 | 0.07964 |
| Grocery | A2_cross_source | cross | 0.40419 | 0.80664 | 0.18776 | 0.04084 | 0.04929 |
| ele-fashion | A0_legacy | physical | 0.79528 | 0.96048 | 0.63161 | 0.63161 | — |
| ele-fashion | A0_legacy | same | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| ele-fashion | A0_legacy | cross | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| ele-fashion | A1_separate_intra | physical | 0.79773 | 1.03855 | 0.07385 | 0.51402 | 0.55693 |
| ele-fashion | A1_separate_intra | same | 0.91760 | 1.17070 | 0.98907 | 0.14142 | 0.08476 |
| ele-fashion | A1_separate_intra | cross | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| ele-fashion | A2_cross_source | physical | 0.79062 | 0.77698 | 0.07444 | 0.43214 | 0.47660 |
| ele-fashion | A2_cross_source | same | 0.91258 | 1.17313 | 1.08252 | 0.14438 | 0.08502 |
| ele-fashion | A2_cross_source | cross | 0.59007 | 0.87856 | 0.42330 | 0.05171 | 0.05082 |

| Dataset | Condition | View | Router entropy | Mean Top-2 selection share | Same/Cross exact-route match |
|---|---|---|---|---|---|
| Movies | A0_legacy | physical | — | [] | — |
| Movies | A0_legacy | same | — | [] | — |
| Movies | A0_legacy | cross | — | [] | — |
| Movies | A1_separate_intra | physical | 1.2644 | [0.25489842270811397, 0.33203374563405913, 0.21471128328691216, 0.19835651852190495] | — |
| Movies | A1_separate_intra | same | 0.9891 | [0.46925982894996804, 0.3590550993879636, 0.5035088692481319, 0.6681761679550012] | — |
| Movies | A1_separate_intra | cross | 0.0000 | [0.0, 0.0, 0.0, 0.0] | — |
| Movies | A2_cross_source | physical | 1.2602 | [0.2447366751730442, 0.3044525732596715, 0.0865273102632879, 0.3642834213678725] | — |
| Movies | A2_cross_source | same | 1.0076 | [0.3940738833819826, 0.35886515583842993, 0.7029450337092081, 0.5441158668448528] | 0.7817 |
| Movies | A2_cross_source | cross | 1.2092 | [0.4165466896568735, 0.3398712317769726, 0.6334472798431913, 0.6101347412914038] | 0.7817 |
| Grocery | A0_legacy | physical | — | [] | — |
| Grocery | A0_legacy | same | — | [] | — |
| Grocery | A0_legacy | cross | — | [] | — |
| Grocery | A1_separate_intra | physical | 0.6819 | [0.36208562801281613, 0.23426457455692193, 0.11996310452620189, 0.28368669748306274] | — |
| Grocery | A1_separate_intra | same | 0.7783 | [0.0972921804447348, 0.5378743590166172, 0.6873316292961439, 0.6775018672148386] | — |
| Grocery | A1_separate_intra | cross | 0.0000 | [0.0, 0.0, 0.0, 0.0] | — |
| Grocery | A2_cross_source | physical | 0.6680 | [0.3926437869668007, 0.3495861167709033, 0.2110762990390261, 0.04669380539174502] | — |
| Grocery | A2_cross_source | same | 0.7309 | [0.18271446165939173, 0.3762055422800283, 0.7330346405506134, 0.7080453832944235] | 0.6871 |
| Grocery | A2_cross_source | cross | 1.1697 | [0.1902112408230702, 0.4261059861940642, 0.7081722766160965, 0.6755105182528496] | 0.6871 |
| ele-fashion | A0_legacy | physical | — | [] | — |
| ele-fashion | A0_legacy | same | — | [] | — |
| ele-fashion | A0_legacy | cross | — | [] | — |
| ele-fashion | A1_separate_intra | physical | 1.0842 | [0.1866060420870781, 0.2921734154224396, 0.2200457528233528, 0.30117477973302204] | — |
| ele-fashion | A1_separate_intra | same | 0.6621 | [0.3537818143765132, 0.5901165430744489, 0.5208030740420023, 0.5352985759576162] | — |
| ele-fashion | A1_separate_intra | cross | 0.0000 | [0.0, 0.0, 0.0, 0.0] | — |
| ele-fashion | A2_cross_source | physical | 1.0528 | [0.18929707258939743, 0.2837781111399333, 0.2319307898481687, 0.2949940115213394] | — |
| ele-fashion | A2_cross_source | same | 0.6065 | [0.43351301550865173, 0.45231813192367554, 0.6495134631792704, 0.46465539932250977] | 0.5411 |
| ele-fashion | A2_cross_source | cross | 1.0362 | [0.34170536945263547, 0.49812476336956024, 0.679220457871755, 0.48094942172368366] | 0.5411 |

## Frozen view shutoff sensitivity

These are checkpoint interventions with no retraining.

| Dataset | Condition | View shutoff | Accuracy Δ | Macro-F1 Δ | Prediction flip fraction |
|---|---|---|---|---|---|
| Movies | A1_separate_intra | same_view_off | -0.510 pp | -0.737 pp | 0.0292 |
| Movies | A2_cross_source | all_semantic_views_off | -0.670 pp | -1.280 pp | 0.0369 |
| Movies | A2_cross_source | cross_source_view_off | -0.170 pp | -0.653 pp | 0.0141 |
| Movies | A2_cross_source | same_view_off | -0.500 pp | -1.053 pp | 0.0333 |
| Grocery | A1_separate_intra | same_view_off | -0.010 pp | -0.143 pp | 0.0132 |
| Grocery | A2_cross_source | all_semantic_views_off | -0.156 pp | -0.214 pp | 0.0160 |
| Grocery | A2_cross_source | cross_source_view_off | 0.029 pp | -0.059 pp | 0.0088 |
| Grocery | A2_cross_source | same_view_off | -0.205 pp | -0.308 pp | 0.0137 |
| ele-fashion | A1_separate_intra | same_view_off | -0.811 pp | -1.165 pp | 0.0195 |
| ele-fashion | A2_cross_source | all_semantic_views_off | -0.975 pp | -1.253 pp | 0.0253 |
| ele-fashion | A2_cross_source | cross_source_view_off | -0.164 pp | -0.038 pp | 0.0064 |
| ele-fashion | A2_cross_source | same_view_off | -0.648 pp | -0.877 pp | 0.0196 |

## ele-fashion minority classes

Precision, recall, F1, support and predicted count are per run in `data/classwise_metrics.csv`; the selected rows below show per-seed class results.

| Checkpoint | Condition | Seed | Class | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|---|---|---|
| best_accuracy | A0_legacy | 42 | 1 | 0.6515 | 0.4136 | 0.5060 | 868 | 551 |
| best_accuracy | A0_legacy | 42 | 11 | 0.8704 | 0.1358 | 0.2350 | 346 | 54 |
| best_f1 | A0_legacy | 42 | 1 | 0.6667 | 0.3940 | 0.4953 | 868 | 513 |
| best_f1 | A0_legacy | 42 | 11 | 0.4850 | 0.3266 | 0.3903 | 346 | 233 |
| best_accuracy | A1_separate_intra | 42 | 1 | 0.7612 | 0.3341 | 0.4644 | 868 | 381 |
| best_accuracy | A1_separate_intra | 42 | 11 | 0.8889 | 0.1156 | 0.2046 | 346 | 45 |
| best_f1 | A1_separate_intra | 42 | 1 | 0.6357 | 0.4021 | 0.4926 | 868 | 549 |
| best_f1 | A1_separate_intra | 42 | 11 | 0.4747 | 0.2977 | 0.3659 | 346 | 217 |
| best_accuracy | A2_cross_source | 42 | 1 | 0.6822 | 0.3882 | 0.4949 | 868 | 494 |
| best_accuracy | A2_cross_source | 42 | 11 | 0.6782 | 0.1705 | 0.2725 | 346 | 87 |
| best_f1 | A2_cross_source | 42 | 1 | 0.6495 | 0.3629 | 0.4656 | 868 | 485 |
| best_f1 | A2_cross_source | 42 | 11 | 0.4604 | 0.3699 | 0.4103 | 346 | 278 |
| best_accuracy | A0_legacy | 43 | 1 | 0.7481 | 0.3491 | 0.4760 | 868 | 405 |
| best_accuracy | A0_legacy | 43 | 11 | 0.6471 | 0.2225 | 0.3312 | 346 | 119 |
| best_f1 | A0_legacy | 43 | 1 | 0.6052 | 0.4274 | 0.5010 | 868 | 613 |
| best_f1 | A0_legacy | 43 | 11 | 0.4682 | 0.3613 | 0.4078 | 346 | 267 |
| best_accuracy | A1_separate_intra | 43 | 1 | 0.7224 | 0.3537 | 0.4749 | 868 | 425 |
| best_accuracy | A1_separate_intra | 43 | 11 | 0.6042 | 0.1676 | 0.2624 | 346 | 96 |
| best_f1 | A1_separate_intra | 43 | 1 | 0.6620 | 0.3790 | 0.4821 | 868 | 497 |
| best_f1 | A1_separate_intra | 43 | 11 | 0.4609 | 0.3237 | 0.3803 | 346 | 243 |
| best_accuracy | A2_cross_source | 43 | 1 | 0.8732 | 0.2857 | 0.4306 | 868 | 284 |
| best_accuracy | A2_cross_source | 43 | 11 | 0.8036 | 0.1301 | 0.2239 | 346 | 56 |
| best_f1 | A2_cross_source | 43 | 1 | 0.6065 | 0.4297 | 0.5030 | 868 | 615 |
| best_f1 | A2_cross_source | 43 | 11 | 0.4607 | 0.3555 | 0.4013 | 346 | 267 |
| best_accuracy | A0_legacy | 44 | 1 | 0.7181 | 0.3376 | 0.4592 | 868 | 408 |
| best_accuracy | A0_legacy | 44 | 11 | 0.7500 | 0.1301 | 0.2217 | 346 | 60 |
| best_f1 | A0_legacy | 44 | 1 | 0.6234 | 0.3986 | 0.4863 | 868 | 555 |
| best_f1 | A0_legacy | 44 | 11 | 0.4717 | 0.2890 | 0.3584 | 346 | 212 |
| best_accuracy | A1_separate_intra | 44 | 1 | 0.7195 | 0.3606 | 0.4804 | 868 | 435 |
| best_accuracy | A1_separate_intra | 44 | 11 | 0.6250 | 0.3035 | 0.4086 | 346 | 168 |
| best_f1 | A1_separate_intra | 44 | 1 | 0.7195 | 0.3606 | 0.4804 | 868 | 435 |
| best_f1 | A1_separate_intra | 44 | 11 | 0.6250 | 0.3035 | 0.4086 | 346 | 168 |
| best_accuracy | A2_cross_source | 44 | 1 | 0.7251 | 0.3767 | 0.4958 | 868 | 451 |
| best_accuracy | A2_cross_source | 44 | 11 | 0.6726 | 0.2197 | 0.3312 | 346 | 113 |
| best_f1 | A2_cross_source | 44 | 1 | 0.6879 | 0.3859 | 0.4945 | 868 | 487 |
| best_f1 | A2_cross_source | 44 | 11 | 0.5251 | 0.3324 | 0.4071 | 346 | 219 |

## Interpretation and limits

A1−A0 measures the complete separated same-source semantic-view condition; A2−A1 measures the additional cross-source view bundle. Neither contrast isolates a single learned scalar or a retrained component ablation. Frozen view shutoff is checkpoint sensitivity only.
The primary readout is the paired Validation Accuracy and Macro-F1 pattern over these three fixed seeds. Cross-dataset consistency, minority-class behavior, CE, Train/Validation gaps, router use, contribution scales, and compute cost must be considered together. This study does not use Test metrics and makes no Test-set claim.

### Outstanding uncertainties

Three seeds give limited uncertainty estimates. View routes share the existing experts and are co-trained with the classifier, so expert-use diagnostics do not establish independent mechanism utility. Any remaining seed variability, optimizer sensitivity, and feature/graph interaction are not identified by this experiment matrix.
