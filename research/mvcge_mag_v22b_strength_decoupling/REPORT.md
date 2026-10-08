# MvCGE-MAG V2.2b: Strength-Decoupled Routing Repair

## Protocol and provenance

- Branch: `exp/mvcge_mag_v22b_strength_decoupling`; parent: `d1d7bbd4b795abd014cec6fdc194dc056f6d058c`; freeze commit: `0f8001a51180544a43e4ca3e8c7e16239669e6c0`.
- Validation-only `unified_full_graph_nc_v1`; Movies, Grocery, `ele-fashion`; seeds 42–44; D0–D3; 36/36 completed.
- Each training command set `task.evaluate_test=false`; Validation Accuracy selected checkpoints. Run metrics have no Test keys. Checkpoint audits used only features, graph edges, weights and selected metadata; no labels were read.
- No HPO, significance test, LP run or post-freeze model/config edit. Device `cuda:0`; unresolved training failures: 0.
- Freeze SHA: `0f8001a51180544a43e4ca3e8c7e16239669e6c0`; formal campaign HEAD matched this SHA and started with a clean worktree.

## Validation results

Run-level means and population standard deviations; paired deltas are descriptive percentage points across matched dataset-seed runs.

| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Best epoch mean | Trainable params (model + head) |
|---|---|---:|---:|---:|---:|
| Movies | D0_static | 54.92% ± 0.32% | 45.18% ± 0.27% | 77.3 | 1,448,758 + 5,140 |
| Movies | D1_free_node | 54.95% ± 0.15% | 46.47% ± 0.50% | 56.0 | 1,448,758 + 5,140 |
| Movies | D2_struct_free | 55.07% ± 0.32% | 46.14% ± 0.80% | 67.0 | 1,449,654 + 5,140 |
| Movies | D3_struct_residual | 55.25% ± 0.36% | 46.69% ± 1.72% | 74.3 | 1,449,656 + 5,140 |
| Grocery | D0_static | 82.56% ± 0.32% | 75.01% ± 1.80% | 79.7 | 1,448,758 + 5,140 |
| Grocery | D1_free_node | 82.81% ± 0.35% | 75.09% ± 1.23% | 71.3 | 1,448,758 + 5,140 |
| Grocery | D2_struct_free | 82.73% ± 0.31% | 75.34% ± 1.33% | 84.0 | 1,449,654 + 5,140 |
| Grocery | D3_struct_residual | 82.60% ± 0.35% | 74.97% ± 1.07% | 83.3 | 1,449,656 + 5,140 |
| ele-fashion | D0_static | 87.22% ± 0.44% | 73.96% ± 1.58% | 168.7 | 1,186,614 + 3,084 |
| ele-fashion | D1_free_node | 87.28% ± 0.14% | 74.13% ± 0.48% | 133.3 | 1,186,614 + 3,084 |
| ele-fashion | D2_struct_free | 87.33% ± 0.06% | 74.03% ± 0.28% | 128.3 | 1,187,510 + 3,084 |
| ele-fashion | D3_struct_residual | 87.42% ± 0.02% | 74.88% ± 0.50% | 171.0 | 1,187,512 + 3,084 |

Equally weighted run means:

```json
{
  "D0_static": {
    "val_accuracy": 0.7489935755729675,
    "val_macro_f1": 0.647178384050132
  },
  "D1_free_node": {
    "val_accuracy": 0.7501440644264221,
    "val_macro_f1": 0.6523356617423302
  },
  "D2_struct_free": {
    "val_accuracy": 0.7504314316643609,
    "val_macro_f1": 0.6516779902359482
  },
  "D3_struct_residual": {
    "val_accuracy": 0.7508712808291117,
    "val_macro_f1": 0.6551177650168494
  }
}
```

## Required paired comparisons

| Comparison | Δ Accuracy | Δ Macro-F1 | Positive pairs (Acc / F1) |
|---|---:|---:|---:|
| D1 − D0: free node selection after strength decoupling | +0.12 pp | +0.52 pp | 6/9 / 7/9 |
| D2 − D1: structural evidence added to free routing | +0.03 pp | -0.07 pp | 5/9 / 6/9 |
| D3 − D2: static-centered residualization | +0.04 pp | +0.34 pp | 6/9 / 6/9 |
| D2 − D0: structure-grounded free routing vs static selection | +0.14 pp | +0.45 pp | 5/9 / 5/9 |
| D3 − D0: structure-grounded residual routing vs static selection | +0.19 pp | +0.79 pp | 6/9 / 5/9 |

Per-dataset mean deltas and positive seed counts:

| Comparison | Dataset | Δ Accuracy mean (positive seeds) | Δ Macro-F1 mean (positive seeds) |
|---|---|---:|---:|
| D1 − D0: free node selection after strength decoupling | Movies | +0.03 pp (1/3) | +1.29 pp (3/3) |
| D1 − D0: free node selection after strength decoupling | Grocery | +0.25 pp (3/3) | +0.08 pp (2/3) |
| D1 − D0: free node selection after strength decoupling | ele-fashion | +0.06 pp (2/3) | +0.17 pp (2/3) |
| D2 − D1: structural evidence added to free routing | Movies | +0.12 pp (2/3) | -0.34 pp (2/3) |
| D2 − D1: structural evidence added to free routing | Grocery | -0.08 pp (1/3) | +0.25 pp (3/3) |
| D2 − D1: structural evidence added to free routing | ele-fashion | +0.04 pp (2/3) | -0.11 pp (1/3) |
| D3 − D2: static-centered residualization | Movies | +0.18 pp (2/3) | +0.55 pp (2/3) |
| D3 − D2: static-centered residualization | Grocery | -0.14 pp (1/3) | -0.37 pp (1/3) |
| D3 − D2: static-centered residualization | ele-fashion | +0.09 pp (3/3) | +0.85 pp (3/3) |
| D2 − D0: structure-grounded free routing vs static selection | Movies | +0.15 pp (1/3) | +0.95 pp (2/3) |
| D2 − D0: structure-grounded free routing vs static selection | Grocery | +0.18 pp (3/3) | +0.33 pp (2/3) |
| D2 − D0: structure-grounded free routing vs static selection | ele-fashion | +0.11 pp (1/3) | +0.07 pp (1/3) |
| D3 − D0: structure-grounded residual routing vs static selection | Movies | +0.33 pp (3/3) | +1.51 pp (2/3) |
| D3 − D0: structure-grounded residual routing vs static selection | Grocery | +0.04 pp (2/3) | -0.04 pp (1/3) |
| D3 − D0: structure-grounded residual routing vs static selection | ele-fashion | +0.19 pp (1/3) | +0.92 pp (2/3) |

No significance testing was performed.

## Strength decoupling and router diagnostics

| Variant | Direct strength Text | Direct strength Visual | Max node-strength std |
|---|---:|---:|---:|
| D0_static | 0.244865 | 0.242992 | 0.0e+00 |
| D1_free_node | 0.245399 | 0.250386 | 0.0e+00 |
| D2_struct_free | 0.246290 | 0.249917 | 0.0e+00 |
| D3_struct_residual | 0.246363 | 0.246597 | 0.0e+00 |


- For each checkpoint/modality, `node_strength_std` is computed as the maximum absolute deviation from the broadcast scalar and is exactly zero. The effective strength depends only on `direct_strength_raw[m]`; the legacy `strength_head` and compatibility modules were frozen.
- D3 mean eta Text/Visual is `0.103454` / `0.103239`. D3 free→final hard-Top-2 pair-change fractions are `0.6068` / `0.3588`; static→final are `0.4708` / `0.2882`.
- Reliability mean/sigma, degree variation, and hop-cosine/displacement variation for D2/D3 are `0.9137`, `0.0223`, `0.1452`, `0.1196`, `0.1145`. Per-order quantiles are in `data/evidence_diagnostics.csv`.
- Alpha profile mean pairwise cosine is `0.0523`; expert-output mean-node cosine is `0.3737`; flattened cosine is `0.4308`.
- Routing entropy, route-to-modality-mean JS and route-to-static JS are summarized below. Dead slots count experts with zero selection in both modalities; affected run-checkpoints are counted separately.

| Variant | Normalized Top-2 pair entropy | Route-to-mean JS | Route-to-static JS | Mean experts in Text ∪ Visual | Dead slots | Runs with ≥1 dead slot |
|---|---:|---:|---:|---:|---:|---:|
| D0_static | 0.0000 | 0.000000 | 0.000000 | 2.556/4 | 13/9 | 8/9 |
| D1_free_node | 0.5638 | 0.130285 | 0.209326 | 3.667/4 | 3/9 | 2/9 |
| D2_struct_free | 0.5912 | 0.134592 | 0.203007 | 3.556/4 | 4/9 | 3/9 |
| D3_struct_residual | 0.3902 | 0.085308 | 0.137648 | 3.778/4 | 2/9 | 2/9 |


## Historical V2.2 strength-coupling comparison

The table uses only committed V2.2 R0/R1 run metrics and their per-modality strength diagnostics, paired with new D0/D1 results. The contrast change is a descriptive arithmetic comparison, not a causal attribution.

| Dataset | Seed | Old R1−R0 Acc / F1 | New D1−D0 Acc / F1 | Contrast change Acc / F1 |
|---|---:|---:|---:|---:|
| Movies | 42 | -0.030 / +0.320 pp | +0.450 / +1.325 pp | +0.480 / +1.005 pp |
| Movies | 43 | -0.150 / -3.136 pp | -0.030 / +0.987 pp | +0.120 / +4.123 pp |
| Movies | 44 | -0.030 / +0.216 pp | -0.330 / +1.556 pp | -0.300 / +1.340 pp |
| Grocery | 42 | -0.088 / -0.685 pp | +0.176 / -0.784 pp | +0.264 / -0.099 pp |
| Grocery | 43 | +0.059 / +0.312 pp | +0.322 / +0.364 pp | +0.264 / +0.052 pp |
| Grocery | 44 | +0.117 / -0.705 pp | +0.264 / +0.671 pp | +0.146 / +1.376 pp |
| ele-fashion | 42 | +0.297 / +0.082 pp | +0.061 / +0.085 pp | -0.235 / +0.003 pp |
| ele-fashion | 43 | +0.358 / +2.826 pp | +0.593 / +2.070 pp | +0.235 / -0.756 pp |
| ele-fashion | 44 | +0.164 / +2.282 pp | -0.470 / -1.632 pp | -0.634 / -3.914 pp |


Across nine matched dataset-seed runs, old R1−R0 means were Accuracy `+0.077` pp / Macro-F1 `+0.168` pp; new D1−D0 means were `+0.115` / `+0.516` pp. The arithmetic contrast change was `+0.038` / `+0.348` pp. Mean old/new strengths by modality are R0 `0.2504/0.4621`, R1 `0.4182/0.6810`, D0 `0.2449/0.2430`, D1 `0.2454/0.2504` (Text/Visual).

## Interpretation map

Interpretation thresholds were fixed in `README.md`: approximate equality means `|Δ Accuracy| <= 0.15 pp` and `|Δ Macro-F1| <= 0.50 pp`; stable-positive means both overall deltas positive, at least 6/9 positive Accuracy pairs, and positive Accuracy means on at least 2/3 datasets. These are descriptive rules, not inferential tests.

- **A. Observed.** D1−D0 is +0.12 pp Accuracy / +0.52 pp Macro-F1, with 6/9 positive Accuracy pairs and positive Accuracy means on 3/3 datasets. This meets the frozen stable-positive rule after strength is decoupled; node-specific expert selection retains a stable positive validation signal.
- **B. Not observed.** D1−D0 is +0.12 pp Accuracy / +0.52 pp Macro-F1, with 6/9 positive Accuracy pairs and positive Accuracy means on 3/3 datasets. This is outside the approximate-equality band and both deltas are positive, so B's unstable-or-absent interpretation is not supported.
- **C. Not observed.** D2−D1 is +0.03 pp Accuracy / -0.07 pp Macro-F1. At least one delta is non-positive, so a two-metric improvement from structural evidence is not established by this screen.
- **D. Observed.** D2−D1 is +0.03 pp Accuracy / -0.07 pp Macro-F1. This falls within the frozen approximate-equality band; current structural observation provides no distinguishable task value in this screen.
- **E. Not observed.** D3−D2 is +0.04 pp Accuracy / +0.34 pp Macro-F1. At least one metric is not lower, so this screen does not show a two-metric loss from static-centered residualization relative to D2.
- **F. Observed.** D3−D2 is +0.04 pp Accuracy / +0.34 pp Macro-F1. Both deltas are positive; interpret this alongside mean eta Text/Visual 0.1035/0.1032 and pair-change fractions (free→final 0.6068/0.3588; static→final 0.4708/0.2882).
- **G. Not observed.** D1 meets the stable-positive rule versus D0. Positive Accuracy pairs and positive dataset means: D1 6/9 and 3/3; D2 5/9 and 3/3. Retain node routing only as a future candidate after validating the effective structural action space; this screen ends here and does not start V3.

## Post-screen decision gate

This screen ends here regardless of result; it does not automatically launch V3. If neither D1 nor D2 is stable-positive versus D0, the next research question is: **Can modality-conditioned effective structural contexts create a better shared expert action space for MAG?** If either passes, retain node routing as a future candidate only after validating that effective structural action space.

Outcome for this campaign: D1 passes the stable-positive rule. Stop after this screen; do not start V3.

- No compatibility routing, expert keys, extra experts, changed Top-K, topology reweighting, cross-modal input, HPO, LP, Test evaluation, or significance testing were introduced.
- Paired outcomes and diagnostics are descriptive; they do not establish causal mechanisms.

## Reproducibility artifacts

- `data/environment.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`
- `data/summary.csv`, `data/paired_comparisons.csv`, `data/routing_diagnostics.csv`, `data/pair_diagnostics.csv`
- `data/evidence_diagnostics.csv`, `data/strength_diagnostics.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/repair_comparison.csv`
- Model checkpoints, training/Hydra logs, raw node tensors, embeddings and caches remain under ignored server-local `outputs/`.
