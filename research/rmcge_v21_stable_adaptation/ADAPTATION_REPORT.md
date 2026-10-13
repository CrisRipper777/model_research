# RM-CGE V2.1 Stable Adaptation: Results Report

## Protocol and interpretation

This report summarizes the frozen W0/W1/W2 study on Movies, Grocery, and ele-fashion with seeds 42–44. Each Dataset × Seed triplet starts from the same audited historical C0 Best Accuracy checkpoint. W1 and W2 use matched selector initialization. The primary checkpoint criterion is Validation Accuracy selected from shared `t0` plus Stage 2; Best Macro-F1 is selected independently over that same horizon. Stage 1 is exploratory and not eligible for either primary checkpoint. All metrics below are Validation metrics; Test was disabled.

Three seeds are paired descriptive evidence. Means and sample standard deviations are reported without significance claims. Accuracy and Macro-F1 in result tables are percentages; paired differences are percentage points (pp); CE is in nats.

## Warm-start checkpoint results

Mean ± sample SD across seeds. The Best F1 checkpoint is separately selected within the Accuracy-based stopping horizon.

| Dataset | Checkpoint | Condition | Val Accuracy (%) | Macro-F1 (%) | CE |
|---|---|---:|---:|---:|---:|
| Movies | Best Accuracy | W0 | 56.099 ± 0.300 | 48.435 ± 0.762 | 1.4066 ± 0.0487 |
| Movies | Best Accuracy | W1 | 56.009 ± 0.283 | 48.485 ± 0.722 | 1.4170 ± 0.0320 |
| Movies | Best Accuracy | W2 | 56.009 ± 0.262 | 48.185 ± 0.556 | 1.4242 ± 0.0338 |
| Movies | Best F1 | W0 | 55.509 ± 0.303 | 49.592 ± 0.228 | 1.4297 ± 0.0461 |
| Movies | Best F1 | W1 | 55.009 ± 0.935 | 49.729 ± 0.215 | 1.4310 ± 0.0482 |
| Movies | Best F1 | W2 | 55.089 ± 1.057 | 49.715 ± 0.199 | 1.4310 ± 0.0486 |
| Grocery | Best Accuracy | W0 | 83.397 ± 0.415 | 76.002 ± 1.779 | 0.6686 ± 0.0591 |
| Grocery | Best Accuracy | W1 | 83.397 ± 0.460 | 75.943 ± 1.731 | 0.6701 ± 0.0586 |
| Grocery | Best Accuracy | W2 | 83.416 ± 0.471 | 76.184 ± 1.948 | 0.6705 ± 0.0586 |
| Grocery | Best F1 | W0 | 83.260 ± 0.454 | 76.862 ± 1.187 | 0.6696 ± 0.0561 |
| Grocery | Best F1 | W1 | 83.280 ± 0.565 | 76.760 ± 1.210 | 0.6725 ± 0.0579 |
| Grocery | Best F1 | W2 | 83.280 ± 0.585 | 76.741 ± 1.190 | 0.6705 ± 0.0550 |
| ele-fashion | Best Accuracy | W0 | 87.552 ± 0.248 | 75.337 ± 0.440 | 0.4363 ± 0.0329 |
| ele-fashion | Best Accuracy | W1 | 87.552 ± 0.248 | 75.337 ± 0.440 | 0.4363 ± 0.0329 |
| ele-fashion | Best Accuracy | W2 | 87.552 ± 0.248 | 75.337 ± 0.440 | 0.4363 ± 0.0329 |
| ele-fashion | Best F1 | W0 | 86.901 ± 0.067 | 76.038 ± 0.165 | 0.4235 ± 0.0119 |
| ele-fashion | Best F1 | W1 | 86.789 ± 0.263 | 76.067 ± 0.180 | 0.4267 ± 0.0171 |
| ele-fashion | Best F1 | W2 | 86.871 ± 0.087 | 76.022 ± 0.164 | 0.4236 ± 0.0119 |

Fashion Best Accuracy stays at `t0` for all three conditions and seeds (9/9 selections), so those three accuracy-selected rows are identical apart from round-off in CE. All 27 Best F1 checkpoints are from a Stage 2 joint epoch. For Fashion, the Best F1 checkpoint trades about 0.65–0.76 pp Accuracy for about 0.69–0.73 pp Macro-F1 over t0; W0 shows the same pattern, so it is not a selector-specific effect.

## Paired differences

The following are mean ± sample SD of seedwise paired deltas. Best Accuracy and Best F1 comparisons are separate. These results come from `data/paired_comparisons.csv` and `data/paired_comparison_summary.csv`.

### Movies

| Checkpoint | Contrast | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---|---:|---:|---:|
| Best Accuracy | W1-W0 | -0.090 ± 0.108 | +0.049 ± 0.207 | +0.01041 ± 0.01750 |
| Best Accuracy | W2-W1 | +0.000 ± 0.030 | -0.299 ± 0.592 | +0.00726 ± 0.01052 |
| Best Accuracy | W2-W0 | -0.090 ± 0.090 | -0.250 ± 0.766 | +0.01767 ± 0.01665 |
| Best F1 | W1-W0 | -0.500 ± 0.632 | +0.136 ± 0.085 | +0.00130 ± 0.00235 |
| Best F1 | W2-W1 | +0.080 ± 0.125 | -0.014 ± 0.106 | -0.00003 ± 0.00049 |
| Best F1 | W2-W0 | -0.420 ± 0.755 | +0.123 ± 0.041 | +0.00126 ± 0.00263 |

### Grocery

| Checkpoint | Contrast | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---|---:|---:|---:|
| Best Accuracy | W1-W0 | -0.000 ± 0.059 | -0.059 ± 0.151 | +0.00151 ± 0.00243 |
| Best Accuracy | W2-W1 | +0.020 ± 0.017 | +0.241 ± 0.394 | +0.00037 ± 0.00074 |
| Best Accuracy | W2-W0 | +0.020 ± 0.061 | +0.181 ± 0.248 | +0.00189 ± 0.00316 |
| Best F1 | W1-W0 | +0.020 ± 0.111 | -0.102 ± 0.037 | +0.00289 ± 0.00261 |
| Best F1 | W2-W1 | +0.000 ± 0.051 | -0.019 ± 0.050 | -0.00202 ± 0.00326 |
| Best F1 | W2-W0 | +0.020 ± 0.138 | -0.121 ± 0.017 | +0.00087 ± 0.00121 |

### ele-fashion

| Checkpoint | Contrast | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---|---:|---:|---:|
| Best Accuracy | W1-W0 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.00000 ± 0.00000 |
| Best Accuracy | W2-W1 | +0.000 ± 0.000 | +0.000 ± 0.000 | -0.00000 ± 0.00000 |
| Best Accuracy | W2-W0 | +0.000 ± 0.000 | +0.000 ± 0.000 | -0.00000 ± 0.00000 |
| Best F1 | W1-W0 | -0.113 ± 0.204 | +0.029 ± 0.015 | +0.00322 ± 0.00529 |
| Best F1 | W2-W1 | +0.082 ± 0.186 | -0.045 ± 0.037 | -0.00320 ± 0.00523 |
| Best F1 | W2-W0 | -0.031 ± 0.020 | -0.016 ± 0.035 | +0.00002 ± 0.00016 |

## Change from the shared t0 checkpoint

This separates continued optimization from selector comparisons. On Best Accuracy, the mean Validation Macro-F1 changes from t0 were Movies W0/W1/W2 +1.23/+1.28/+0.98 pp; Grocery +0.18/+0.12/+0.36 pp; and ele-fashion 0/0/0 pp. The mean Accuracy changes were Movies +0.23/+0.14/+0.14 pp; Grocery +0.05/+0.05/+0.07 pp; and Fashion 0/0/0 pp. W0 is therefore a meaningful optimization-restart/continuation baseline on Movies and Grocery; Fashion’s Accuracy-selected checkpoints did not displace t0.

On Fashion Best F1, Macro-F1 rose over t0 by +0.70 pp (W0), +0.73 pp (W1), and +0.69 pp (W2), while Accuracy fell by −0.65, −0.76, and −0.68 pp, respectively. This is a checkpoint-objective tradeoff, not evidence of improved classification across objectives.

### Best-F1 versus Best-Accuracy within-run tradeoff

Paired differences are Best F1 minus Best Accuracy within each run, averaged over the three seeds (percentage points for Accuracy/Macro-F1; nats for CE). Train and Validation metrics move together in fit but not always in the same direction on held-out class balance.

| Dataset | Condition | Train Δ Acc (pp) | Train Δ F1 (pp) | Train Δ CE | Val Δ Acc (pp) | Val Δ F1 (pp) | Val Δ CE |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | W0 | +2.033 | +2.652 | -0.0374 | -0.590 | +1.157 | +0.0232 |
| Movies | W1 | +0.400 | +0.761 | +0.0107 | -1.000 | +1.244 | +0.0141 |
| Movies | W2 | +1.120 | +1.411 | -0.0047 | -0.920 | +1.530 | +0.0068 |
| Grocery | W0 | +0.745 | +1.452 | -0.0181 | -0.137 | +0.860 | +0.0010 |
| Grocery | W1 | +0.979 | +1.945 | -0.0256 | -0.117 | +0.817 | +0.0023 |
| Grocery | W2 | +0.648 | +1.325 | -0.0153 | -0.137 | +0.557 | -0.0000 |
| ele-fashion | W0 | +1.329 | +2.925 | -0.0270 | -0.651 | +0.701 | -0.0127 |
| ele-fashion | W1 | +1.470 | +3.175 | -0.0291 | -0.764 | +0.730 | -0.0095 |
| ele-fashion | W2 | +1.345 | +2.959 | -0.0272 | -0.682 | +0.685 | -0.0127 |

## Cold-start comparison

Historical C0/C1/C2 are a descriptive reference only. Warm-start and cold-start jointly change initialization, training budget, learning rate, optimizer state, and checkpoint selection, so their difference does not isolate a causal initialization effect.

| Dataset | Historical condition → warm condition | Best-Accuracy Δ Val Accuracy (pp) | Δ Val Macro-F1 (pp) |
|---|---|---:|---:|
| Movies | C0 → W0 | +0.230 | +1.234 |
| Movies | C1 → W1 | -0.070 | +0.645 |
| Movies | C2 → W2 | +0.200 | -0.965 |
| Grocery | C0 → W0 | +0.049 | +0.176 |
| Grocery | C1 → W1 | -0.205 | -0.503 |
| Grocery | C2 → W2 | +0.039 | -0.272 |
| ele-fashion | C0 → W0 | +0.000 | +0.000 |
| ele-fashion | C1 → W1 | +0.123 | +1.797 |
| ele-fashion | C2 → W2 | +0.232 | +1.731 |

For Fashion, warm W1/W2 show +1.80/+1.73 pp Best-Accuracy Macro-F1 relative to historical cold C1/C2 (paired mean across seeds), but the warm model is selected at the inherited C0 t0 checkpoint. This demonstrates a strong common-start advantage for this specific comparison; it does not show that selector adaptation fixed the cold-start failure. Across Movies/Grocery, the old selective gains are not uniformly retained under the warm protocol.

## Train/Validation and runtime tradeoffs

Per-checkpoint Train/Validation Accuracy and Macro-F1 gaps, epoch, and checkpoint hash are in `data/checkpoint_tradeoffs.csv`. Per-run duration, Stage 1/Stage 2 durations, joint updates, and peak GPU memory are in `data/resource_profile.csv`. W1/W2 pay extra computation for selector adaptation; all conditions used a freshly initialized Stage 2 AdamW state, not restored historical optimizer momentum. See `TRAINING_DYNAMICS.md` for trajectory and process details.

## Source integrity and provenance

Nine historical C0 Best Accuracy files were resolved through the V2.1 source manifest and checkpoint metadata, then byte-hash verified locally. Model and classifier hashes, source epochs, byte counts, and verification flags are in `data/warmstart_source_checkpoints.csv`; detailed initialization/data/cache/split fingerprints are in `data/initialization_audit.json`. All nine checks passed. No source checkpoint was silently regenerated.

## Scope limits

No Test labels or metrics were accessed. The results establish matched starts, valid selector gradients and updates, finite training, and traceable comparisons. They do not establish statistically significant gains, a unique explanation for the historical Fashion regression, or that scalar neighbor reweighting is universally inadequate.
