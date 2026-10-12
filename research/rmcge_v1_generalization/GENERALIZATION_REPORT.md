# RM-CGE V1 Generalization & Optimization Results

## Protocol and interpretation

The complete 2×2 Dropout × AdamW Weight Decay matrix contains 36 Full-Graph NC runs: Movies, Grocery, and ele-fashion; O0–O3; seeds 42, 43, and 44. The seed-level training run is the experimental unit (n=3 per dataset/treatment). Reported SDs are sample SDs (ddof=1). Treatment comparisons are paired by seed and descriptive; there are no p-values or statistical-significance claims. Test evaluation remained disabled and Test labels were not indexed.

All 72 saved checkpoints were replayed for Validation scorecards on `cpu` at report time. CUDA was unavailable then, so the replay used CPU. The run-time checkpoint reload audit is separately recorded in `data/final_integrity_audit.json`.

Implementation Freeze: `e002af6a9c57de21986129d2b1c6e8478bcc74e3`. Seed 42 used the frozen source; seeds 43/44 record the launcher-guard follow-up source SHA per run. The post-freeze commits changed only runner validation, regression tests, and audit prose; RM-CGE model, optimizer, training, configuration, and metric logic remained frozen.

## Historical O0 replay: non-exact and nonblocking

The archived Pilot is retained as a reference only. The original accuracy checkpoint used 100 epochs, Best Epoch 70, Validation Accuracy 0.540492, and Macro-F1 0.429959. The old exact-replay requirement is retired; the two replay artifacts below remain unchanged and are classified as `historical_full_training_replay_not_exact`.

| Run | Epochs | Best Epoch | Val Accuracy | Val Macro-F1 | Max Train CE trajectory Δ | Max Val Accuracy Δ | Max Val Macro-F1 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical Pilot | 100 | 70 | 0.540492 | 0.429959 | — | — | — |
| O0 replay 1 | 100 | 70 | 0.537493 | 0.436371 | 0.080665 | 0.018896 | 0.029098 |
| O0 replay 2 | 97 | 67 | 0.540792 | 0.435800 | 0.038393 | 0.020996 | 0.026305 |

Both runs had 892,242 model and 2,580 classifier parameters. Their curves remained highly correlated, but the maximum common-epoch differences and Best Epoch/score differences are factual mismatches, not overwritten results. The original Pilot did not save its per-node embedding/logit arrays, so later replay checks reconstructed them from the checkpoint; repeated inference itself varied by about 1.2e-6 in embeddings and 1.9e-6 in logits.

The O0 config audit matched the original dataset config and frozen model/training controls: dropout 0.20, AdamW, LR 1e-3, weight decay 1e-4, protocol, model dimensions, initialization, optimizer grouping and early-stopping rules. The study-only additions are ordered CPU train/validation head metrics, Best-F1 bookkeeping, provenance, and isolated output paths. The additional metrics reuse eval embeddings and preserve CPU/CUDA RNG; standard Dropout is reseeded per run and activation checkpointing preserves RNG state.

A frozen-source two-update Forward/Backward/AdamW audit had identical initial weights and step losses, with maximum parameter error 5.05e-07; extra CPU metrics preserved RNG states. This supports short-run parity but cannot establish identical full training paths. No further full historical replay was performed.

Recorded environment: Python 3.12.13, NumPy 2.4.3, PyTorch 2.4.0+cu121 (CUDA build 12.1), cuDNN 90100, driver 535.309.01, NVIDIA GeForce RTX 3090 physical GPU 1. CUDA matmul TF32=False, cuDNN TF32=True, float32 matmul precision `highest`; deterministic algorithms=False, cuDNN deterministic=False, benchmark=False, CUBLAS_WORKSPACE_CONFIG=None.

Small sparse CUDA propagation variation is plausible but was not isolated as the sole cause. The formal runtime was not changed to force deterministic algorithms or alter TF32. The remaining unresolved issue is why tiny numerical differences amplify over long training; this does not block the internally paired matrix.

## Internal O0 baselines

These O0 results come from the same frozen internal matrix as O1–O3; no historical Pilot score is used as an acceptance target.

| Dataset | O0 seed 42 Acc / F1 | O0 seed 43 Acc / F1 | O0 seed 44 Acc / F1 | Mean Acc ± sample SD | Mean F1 ± sample SD |
|---|---:|---:|---:|---:|---:|
| Movies | 0.541392 / 0.435623 | 0.540792 / 0.433809 | 0.541092 / 0.437820 | 0.541092 ± 0.000300 | 0.435751 ± 0.002009 |
| Grocery | 0.822548 / 0.756259 | 0.818741 / 0.740056 | 0.822255 / 0.755728 | 0.821181 ± 0.002118 | 0.750681 ± 0.009206 |
| ele-fashion | 0.879615 / 0.761872 | 0.879820 / 0.748671 | 0.876956 / 0.749559 | 0.878797 ± 0.001598 | 0.753367 ± 0.007379 |

## Best-Accuracy results (mean ± sample SD)

Train CE is the stochastic training-mode loss before that epoch's optimizer update; Train Accuracy is eval-mode after the update. `Train CE−Val CE` pairs a pre-update Train CE with post-update Validation CE, so it is a trajectory proxy rather than a same-state gap. Validation CE is unweighted.

| Dataset | Treatment | Val Acc | Val Macro-F1 | Val CE | Best epoch | Train CE | Eval Train Acc | Train CE−Val CE | Train Acc−Val Acc | Train time (s) | Peak alloc. GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | O0 | 0.5411 ± 0.0003 | 0.4358 ± 0.0020 | 1.4402 ± 0.0189 | 64.7 ± 9.7 | 0.7979 | 0.7664 | -0.6423 | 0.2253 | 20.9 | 1.00 |
| Movies | O1 | 0.5430 ± 0.0039 | 0.4378 ± 0.0065 | 1.4457 ± 0.0146 | 71.3 ± 8.3 | 0.7423 | 0.7973 | -0.7034 | 0.2543 | 21.9 | 1.00 |
| Movies | O2 | 0.5424 ± 0.0003 | 0.4325 ± 0.0171 | 1.4525 ± 0.0223 | 70.7 ± 5.5 | 0.7279 | 0.7972 | -0.7247 | 0.2548 | 21.6 | 1.00 |
| Movies | O3 | 0.5448 ± 0.0035 | 0.4318 ± 0.0123 | 1.4139 ± 0.0255 | 65.3 ± 9.3 | 0.8684 | 0.7420 | -0.5455 | 0.1972 | 20.3 | 1.00 |
| Grocery | O0 | 0.8212 ± 0.0021 | 0.7507 ± 0.0092 | 0.6896 ± 0.0196 | 96.7 ± 13.4 | 0.1962 | 0.9640 | -0.4934 | 0.1428 | 26.5 | 1.03 |
| Grocery | O1 | 0.8250 ± 0.0030 | 0.7533 ± 0.0081 | 0.7114 ± 0.0285 | 106.7 ± 15.6 | 0.1689 | 0.9724 | -0.5425 | 0.1474 | 29.2 | 1.03 |
| Grocery | O2 | 0.8224 ± 0.0027 | 0.7441 ± 0.0130 | 0.6794 ± 0.0083 | 84.3 ± 6.4 | 0.2522 | 0.9482 | -0.4272 | 0.1257 | 24.0 | 1.03 |
| Grocery | O3 | 0.8245 ± 0.0019 | 0.7526 ± 0.0072 | 0.7164 ± 0.0439 | 112.0 ± 22.3 | 0.1483 | 0.9775 | -0.5682 | 0.1530 | 30.5 | 1.03 |
| ele-fashion | O0 | 0.8788 ± 0.0016 | 0.7534 ± 0.0074 | 0.4278 ± 0.0150 | 145.7 ± 4.2 | 0.2217 | 0.9269 | -0.2061 | 0.0481 | 170.4 | 5.46 |
| ele-fashion | O1 | 0.8790 ± 0.0015 | 0.7512 ± 0.0055 | 0.4125 ± 0.0232 | 143.0 ± 24.6 | 0.2571 | 0.9140 | -0.1554 | 0.0350 | 166.1 | 5.46 |
| ele-fashion | O2 | 0.8796 ± 0.0012 | 0.7545 ± 0.0055 | 0.4345 ± 0.0293 | 148.3 ± 16.4 | 0.2070 | 0.9314 | -0.2275 | 0.0518 | 155.7 | 5.46 |
| ele-fashion | O3 | 0.8792 ± 0.0012 | 0.7526 ± 0.0104 | 0.4284 ± 0.0275 | 160.3 ± 21.5 | 0.2257 | 0.9259 | -0.2027 | 0.0466 | 155.8 | 5.46 |

## Best-Macro-F1 results (mean ± sample SD)

| Dataset | Treatment | Val Acc | Val Macro-F1 | Val CE | Best-F1 epoch |
|---|---|---:|---:|---:|---:|
| Movies | O0 | 0.5256 ± 0.0128 | 0.4459 ± 0.0028 | 1.5229 ± 0.0994 | 74.3 ± 6.7 |
| Movies | O1 | 0.5346 ± 0.0049 | 0.4546 ± 0.0072 | 1.5132 ± 0.0489 | 79.7 ± 14.3 |
| Movies | O2 | 0.5275 ± 0.0084 | 0.4483 ± 0.0011 | 1.5451 ± 0.0850 | 81.3 ± 11.5 |
| Movies | O3 | 0.5329 ± 0.0012 | 0.4490 ± 0.0034 | 1.4908 ± 0.0804 | 78.0 ± 17.3 |
| Grocery | O0 | 0.8199 ± 0.0026 | 0.7545 ± 0.0076 | 0.7192 ± 0.0128 | 113.7 ± 2.1 |
| Grocery | O1 | 0.8224 ± 0.0037 | 0.7567 ± 0.0071 | 0.7408 ± 0.0559 | 121.0 ± 25.9 |
| Grocery | O2 | 0.8206 ± 0.0028 | 0.7525 ± 0.0088 | 0.7035 ± 0.0096 | 100.0 ± 5.3 |
| Grocery | O3 | 0.8210 ± 0.0034 | 0.7565 ± 0.0083 | 0.7428 ± 0.0442 | 120.7 ± 20.0 |
| ele-fashion | O0 | 0.8708 ± 0.0006 | 0.7644 ± 0.0050 | 0.4437 ± 0.0206 | 165.3 ± 5.7 |
| ele-fashion | O1 | 0.8730 ± 0.0033 | 0.7637 ± 0.0055 | 0.4173 ± 0.0292 | 159.0 ± 27.6 |
| ele-fashion | O2 | 0.8694 ± 0.0011 | 0.7662 ± 0.0040 | 0.4639 ± 0.0482 | 173.0 ± 21.0 |
| ele-fashion | O3 | 0.8759 ± 0.0039 | 0.7665 ± 0.0043 | 0.4334 ± 0.0242 | 177.0 ± 12.5 |

## Three-seed paired regularization effects

Acc/F1 positive differences favor the first treatment; Validation CE negative differences favor the first treatment. O1−O0 is Dropout at low decay; O2−O0 is Weight Decay at low dropout; O3−O1 is Weight Decay at high dropout; O3−O2 is Dropout at high decay; Interaction is `(O3−O2)−(O1−O0)`. All are paired by dataset and seed.

| Dataset | Contrast | Metric | seed 42 | seed 43 | seed 44 | Mean Δ | Sample SD |
|---|---|---|---:|---:|---:|---:|---:|
| Grocery | Interaction=(O3-O2)-(O1-O0) | val_accuracy | -0.001757 | +0.000000 | -0.003514 | -0.001757 | 0.001757 |
| Grocery | Interaction=(O3-O2)-(O1-O0) | val_macro_f1 | +0.009799 | +0.008945 | -0.000995 | +0.005916 | 0.006001 |
| Grocery | Interaction=(O3-O2)-(O1-O0) | validation_ce | +0.103596 | -0.037849 | -0.020087 | +0.015220 | 0.077049 |
| Grocery | O1-O0 | val_accuracy | +0.003221 | +0.002928 | +0.005271 | +0.003807 | 0.001276 |
| Grocery | O1-O0 | val_macro_f1 | -0.004866 | +0.006291 | +0.006412 | +0.002612 | 0.006477 |
| Grocery | O1-O0 | validation_ce | -0.025775 | +0.068678 | +0.022608 | +0.021837 | 0.047231 |
| Grocery | O2-O0 | val_accuracy | +0.001757 | +0.000586 | +0.001464 | +0.001269 | 0.000610 |
| Grocery | O2-O0 | val_macro_f1 | -0.004169 | -0.010962 | -0.004624 | -0.006585 | 0.003797 |
| Grocery | O2-O0 | validation_ce | -0.018265 | +0.003387 | -0.015645 | -0.010174 | 0.011817 |
| Grocery | O3-O1 | val_accuracy | +0.000000 | +0.000586 | -0.002050 | -0.000488 | 0.001384 |
| Grocery | O3-O1 | val_macro_f1 | +0.005630 | -0.002017 | -0.005619 | -0.000669 | 0.005745 |
| Grocery | O3-O1 | validation_ce | +0.085331 | -0.034462 | -0.035731 | +0.005046 | 0.069532 |
| Grocery | O3-O2 | val_accuracy | +0.001464 | +0.002928 | +0.001757 | +0.002050 | 0.000775 |
| Grocery | O3-O2 | val_macro_f1 | +0.004933 | +0.015236 | +0.005416 | +0.008529 | 0.005814 |
| Grocery | O3-O2 | validation_ce | +0.077821 | +0.030829 | +0.002521 | +0.037057 | 0.038034 |
| Movies | Interaction=(O3-O2)-(O1-O0) | val_accuracy | +0.001800 | -0.000300 | +0.000000 | +0.000500 | 0.001136 |
| Movies | Interaction=(O3-O2)-(O1-O0) | val_macro_f1 | -0.025981 | +0.031980 | -0.014049 | -0.002683 | 0.030607 |
| Movies | Interaction=(O3-O2)-(O1-O0) | validation_ce | -0.061843 | +0.015874 | -0.086286 | -0.044085 | 0.053345 |
| Movies | O1-O0 | val_accuracy | -0.002100 | +0.006299 | +0.001500 | +0.001900 | 0.004213 |
| Movies | O1-O0 | val_macro_f1 | +0.008085 | -0.002998 | +0.001060 | +0.002049 | 0.005607 |
| Movies | O1-O0 | validation_ce | +0.014959 | -0.012893 | +0.014363 | +0.005477 | 0.015911 |
| Movies | O2-O0 | val_accuracy | +0.000600 | +0.001800 | +0.001500 | +0.001300 | 0.000624 |
| Movies | O2-O0 | val_macro_f1 | +0.000832 | -0.020047 | +0.009367 | -0.003282 | 0.015133 |
| Movies | O2-O0 | validation_ce | -0.006878 | -0.015514 | +0.059385 | +0.012331 | 0.040978 |
| Movies | O3-O1 | val_accuracy | +0.002400 | +0.001500 | +0.001500 | +0.001800 | 0.000519 |
| Movies | O3-O1 | val_macro_f1 | -0.025149 | +0.011934 | -0.004681 | -0.005966 | 0.018574 |
| Movies | O3-O1 | validation_ce | -0.068722 | +0.000360 | -0.026901 | -0.031754 | 0.034796 |
| Movies | O3-O2 | val_accuracy | -0.000300 | +0.005999 | +0.001500 | +0.002400 | 0.003244 |
| Movies | O3-O2 | val_macro_f1 | -0.017896 | +0.028983 | -0.012989 | -0.000634 | 0.025766 |
| Movies | O3-O2 | validation_ce | -0.046884 | +0.002981 | -0.071923 | -0.038608 | 0.038132 |
| ele-fashion | Interaction=(O3-O2)-(O1-O0) | val_accuracy | -0.001637 | +0.001432 | -0.001432 | -0.000546 | 0.001716 |
| ele-fashion | Interaction=(O3-O2)-(O1-O0) | val_macro_f1 | +0.008568 | -0.003682 | -0.004235 | +0.000217 | 0.007238 |
| ele-fashion | Interaction=(O3-O2)-(O1-O0) | validation_ce | -0.031497 | +0.026007 | +0.032997 | +0.009169 | 0.035391 |
| ele-fashion | O1-O0 | val_accuracy | +0.001125 | -0.001534 | +0.001023 | +0.000205 | 0.001507 |
| ele-fashion | O1-O0 | val_macro_f1 | -0.004398 | -0.001518 | -0.000545 | -0.002154 | 0.002003 |
| ele-fashion | O1-O0 | validation_ce | +0.020961 | -0.026493 | -0.040484 | -0.015339 | 0.032205 |
| ele-fashion | O2-O0 | val_accuracy | +0.001227 | -0.000307 | +0.001432 | +0.000784 | 0.000950 |
| ele-fashion | O2-O0 | val_macro_f1 | -0.001976 | +0.000284 | +0.005180 | +0.001163 | 0.003658 |
| ele-fashion | O2-O0 | validation_ce | +0.048488 | +0.008565 | -0.036928 | +0.006708 | 0.042739 |
| ele-fashion | O3-O1 | val_accuracy | -0.000409 | +0.001125 | +0.000000 | +0.000239 | 0.000794 |
| ele-fashion | O3-O1 | val_macro_f1 | +0.006592 | -0.003398 | +0.000945 | +0.001380 | 0.005009 |
| ele-fashion | O3-O1 | validation_ce | +0.016992 | +0.034572 | -0.003931 | +0.015878 | 0.019276 |
| ele-fashion | O3-O2 | val_accuracy | -0.000511 | -0.000102 | -0.000409 | -0.000341 | 0.000213 |
| ele-fashion | O3-O2 | val_macro_f1 | +0.004170 | -0.005200 | -0.004780 | -0.001937 | 0.005293 |
| ele-fashion | O3-O2 | validation_ce | -0.010536 | -0.000486 | -0.007487 | -0.006170 | 0.005153 |

## Checkpoint selection trade-offs

The table reports Best-F1 minus Best-Accuracy for the same run, summarized over the three seeds as mean ± sample SD. Differences may be positive or negative because the primary Accuracy rule and F1 rule select different epochs.

| Dataset | Treatment | Δ Val Acc | Δ Macro-F1 | Δ Val CE | Δ selected epoch |
|---|---|---:|---:|---:|---:|
| Movies | O0 | -0.0155 ± 0.0129 | +0.0102 ± 0.0024 | +0.0827 ± 0.1175 | +9.7 ± 16.3 |
| Movies | O1 | -0.0084 ± 0.0018 | +0.0168 ± 0.0137 | +0.0675 ± 0.0401 | +8.3 ± 8.5 |
| Movies | O2 | -0.0149 ± 0.0082 | +0.0159 ± 0.0161 | +0.0925 ± 0.1018 | +10.7 ± 15.3 |
| Movies | O3 | -0.0119 ± 0.0047 | +0.0172 ± 0.0144 | +0.0768 ± 0.0577 | +12.7 ± 8.1 |
| Grocery | O0 | -0.0013 ± 0.0012 | +0.0038 ± 0.0033 | +0.0297 ± 0.0267 | +17.0 ± 14.7 |
| Grocery | O1 | -0.0025 ± 0.0010 | +0.0034 ± 0.0034 | +0.0294 ± 0.0504 | +14.3 ± 22.0 |
| Grocery | O2 | -0.0019 ± 0.0019 | +0.0084 ± 0.0053 | +0.0242 ± 0.0115 | +15.7 ± 6.1 |
| Grocery | O3 | -0.0035 ± 0.0028 | +0.0038 ± 0.0027 | +0.0264 ± 0.0263 | +8.7 ± 15.5 |
| ele-fashion | O0 | -0.0080 ± 0.0014 | +0.0110 ± 0.0036 | +0.0158 ± 0.0341 | +19.7 ± 9.5 |
| ele-fashion | O1 | -0.0060 ± 0.0020 | +0.0125 ± 0.0044 | +0.0049 ± 0.0068 | +16.0 ± 4.4 |
| ele-fashion | O2 | -0.0102 ± 0.0019 | +0.0117 ± 0.0025 | +0.0294 ± 0.0214 | +24.7 ± 4.6 |
| ele-fashion | O3 | -0.0033 ± 0.0032 | +0.0139 ± 0.0072 | +0.0050 ± 0.0103 | +16.7 ± 10.2 |

## Frozen Physical/Semantic cross-modal diagnostics

Stage B used the original three Seed-42 Best-Accuracy checkpoints. These are frozen-path sensitivities, not retrained ablations; removing a path also changes Router inputs. Physical cross-modal removal had larger single-path sensitivity than Semantic removal on Movies and Grocery; ele-fashion effects were smaller and mixed. Full Validation CE/Accuracy/F1, flip rates, and embedding/logit changes remain in `data/frozen_cross_relation_interventions.csv` and `MECHANISM_REPORT.md`.

The frozen Semantic route-mass SD spans the nodewise range recorded in `data/router_node_statistics.csv` (about 0.02–0.17 across dataset/layer/modality summaries), showing node-conditional variation without establishing relation reliability. Raw cross-modal context RMS is lower than same-modal RMS; after independent block LayerNorm both are near unit RMS. See `data/cross_modal_scales.csv` for per-relation values.

## Train/validation trajectories and generalization flags

See `figures/training_curves.*` for Train CE / Validation CE and eval Train Accuracy / Validation Accuracy. Thin traces show individual seeds; bold traces show the runs available at each epoch. `figures/validation_acc_macro_f1.*` and `figures/best_epoch_distribution.*` show checkpoint-rule outcomes. Accuracy and Macro-F1, as well as their per-class Validation scorecards, are computed under separate checkpoint-selection rules.
- **Movies/O0:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **Movies/O1:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **Movies/O2:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **Movies/O3:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **Grocery/O0:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **Grocery/O1:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **Grocery/O2:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **Grocery/O3:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **ele-fashion/O0:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **ele-fashion/O1:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **ele-fashion/O2:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.
- **ele-fashion/O3:** 0/3 seed trajectories show lower mean Train CE in their last ten epochs and higher mean Validation CE than in their first ten. This endpoint comparison is descriptive and sensitive to early stopping.

## Remaining uncertainty

The historical full-run divergence remains unexplained. Short-run parity, config/data/cache checks, RNG-safe diagnostics, and checkpoint reloads passed, but small sparse CUDA differences are only a plausible contributor and were not isolated. Three seeds on fixed official splits are a small optimization sample; rank changes and early-stopping epoch variation should be read descriptively. No V1.1, Router/expert change, or Test evaluation was performed.

## Reproducibility files

`data/manifest.json` records the parent SHA, Implementation Freeze SHA, per-run source SHAs (`089637306507c9d648bbec4e429c0886a1fbd5bd`, `e002af6a9c57de21986129d2b1c6e8478bcc74e3`), config/data/cache fingerprints, hardware, and run status. `data/runs.csv`, `data/epoch_history.csv`, `data/paired_comparisons.csv`, and `data/factorial_effects.csv` retain run, trajectory, paired, and aggregate evidence. `data/final_integrity_audit.json` records the final audit of all 36 runs and 72 checkpoints. Checkpoints remain under ignored `outputs/` and are referenced by relative paths.

Figure source validation, measured panel alignment, PDF font-floor checks, and rendered collision audits are summarized in `data/figure_qa.json` and `data/figure_qa_external_audits.json`; source figures are exported as SVG, PDF, and 600 dpi PNG.
