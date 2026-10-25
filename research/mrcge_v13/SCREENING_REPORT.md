# MR-CGE V1.3 View Collaboration and Cross Screening Report

## Scope and comparison protocol

This is the completed Validation-only screen from base commit
`a7930442e85ff9450fded54d5869915b434c7431`, on branch
`exp/mrcge_v13_view_cross_screen`. It contains Movies, Grocery, and
ele-fashion; the independent V1.3-A `joint_view` and V1.3-B `sender_private`
conditions; and seeds 42/43 (12 unique formal cells). Both variants use the
V1.2 `M_repaired` backbone. The historical V1.2 rows are same-dataset,
same-seed references from a separate campaign, not same-job retrains. Split,
feature, physical graph, semantic cache, training protocol, and model source
fingerprints were checked for alignment.

The primary values below are from each variant's **Best Validation Accuracy**
checkpoint. Accuracy and Macro-F1 are shown as percentages; CE is the raw
Validation cross entropy. Deltas are V1.3 minus the corresponding historical
V1.2 `M_repaired` row at the same dataset and seed. Thus these are paired by
seed and protocol, but still cross-campaign comparisons. The two-seed means
are descriptive only; no significance or causal claim is made.

## Per-seed Validation results

### Movies

| Seed | Condition | Acc % | Macro-F1 % | CE | Δ Acc pp | Δ F1 pp | Δ CE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | V1.2 `M_repaired` | 55.309 | 46.514 | 1.37182 | — | — | — |
| 42 | V1.3-A `joint_view` | 55.969 | 48.198 | 1.37037 | +0.660 | +1.684 | −0.00144 |
| 42 | V1.3-B `sender_private` | 55.249 | 48.104 | 1.42886 | −0.060 | +1.590 | +0.05705 |
| 43 | V1.2 `M_repaired` | 54.769 | 47.669 | 1.42958 | — | — | — |
| 43 | V1.3-A `joint_view` | 55.219 | 49.171 | 1.37982 | +0.450 | +1.503 | −0.04977 |
| 43 | V1.3-B `sender_private` | 54.679 | 47.529 | 1.40283 | −0.090 | −0.139 | −0.02675 |

### Grocery

| Seed | Condition | Acc % | Macro-F1 % | CE | Δ Acc pp | Δ F1 pp | Δ CE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | V1.2 `M_repaired` | 82.723 | 76.259 | 0.69163 | — | — | — |
| 42 | V1.3-A `joint_view` | 83.514 | 76.621 | 0.63923 | +0.791 | +0.362 | −0.05239 |
| 42 | V1.3-B `sender_private` | 82.782 | 75.965 | 0.69177 | +0.059 | −0.294 | +0.00014 |
| 43 | V1.2 `M_repaired` | 82.840 | 76.439 | 0.70409 | — | — | — |
| 43 | V1.3-A `joint_view` | 83.690 | 76.936 | 0.64558 | +0.849 | +0.497 | −0.05851 |
| 43 | V1.3-B `sender_private` | 82.899 | 76.280 | 0.70219 | +0.059 | −0.159 | −0.00190 |

### ele-fashion

| Seed | Condition | Acc % | Macro-F1 % | CE | Δ Acc pp | Δ F1 pp | Δ CE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | V1.2 `M_repaired` | 87.368 | 73.498 | 0.41303 | — | — | — |
| 42 | V1.3-A `joint_view` | 87.102 | 73.284 | 0.40106 | −0.266 | −0.214 | −0.01197 |
| 42 | V1.3-B `sender_private` | 87.450 | 73.616 | 0.41521 | +0.082 | +0.118 | +0.00218 |
| 43 | V1.2 `M_repaired` | 87.440 | 74.196 | 0.41301 | — | — | — |
| 43 | V1.3-A `joint_view` | 87.501 | 73.791 | 0.40386 | +0.061 | −0.405 | −0.00915 |
| 43 | V1.3-B `sender_private` | 87.430 | 74.649 | 0.43212 | −0.010 | +0.452 | +0.01911 |

## Descriptive two-seed means

| Dataset | Condition | Acc % | Macro-F1 % | CE | Δ Acc pp | Δ F1 pp | Δ CE |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | V1.2 `M_repaired` | 55.039 | 47.091 | 1.40070 | — | — | — |
| Movies | V1.3-A `joint_view` | 55.594 | 48.684 | 1.37510 | +0.555 | +1.593 | −0.02560 |
| Movies | V1.3-B `sender_private` | 54.964 | 47.817 | 1.41585 | −0.075 | +0.725 | +0.01515 |
| Grocery | V1.2 `M_repaired` | 82.782 | 76.349 | 0.69786 | — | — | — |
| Grocery | V1.3-A `joint_view` | 83.602 | 76.778 | 0.64241 | +0.820 | +0.430 | −0.05545 |
| Grocery | V1.3-B `sender_private` | 82.840 | 76.123 | 0.69698 | +0.059 | −0.226 | −0.00088 |
| ele-fashion | V1.2 `M_repaired` | 87.404 | 73.847 | 0.41302 | — | — | — |
| ele-fashion | V1.3-A `joint_view` | 87.302 | 73.538 | 0.40246 | −0.102 | −0.310 | −0.01056 |
| ele-fashion | V1.3-B `sender_private` | 87.440 | 74.132 | 0.42366 | +0.036 | +0.285 | +0.01064 |

For A, both Movies and Grocery seeds improve in Accuracy and Macro-F1, while
ele-fashion is mixed in Accuracy and below the reference on Macro-F1 for both
seeds. For B, Movies Macro-F1 improves at seed 42 but not seed 43; Grocery
Accuracy moves by about +0.06 percentage points at both seeds while Macro-F1
decreases at both; ele-fashion is mixed, with a small Accuracy change and
opposite-sign F1 deltas. The two-seed summaries are too small to establish
stability or a final model choice.

## Best-F1 checkpoint results (kept separate)

These means use the separately selected Best-F1 checkpoint metrics. They are
not mixed into the Best-Accuracy tables above.

| Dataset | Condition | Acc % | Macro-F1 % | CE |
|---|---|---:|---:|---:|
| Movies | V1.2 `M_repaired` | 54.724 | 47.965 | 1.41699 |
| Movies | V1.3-A `joint_view` | 54.919 | 49.837 | 1.41359 |
| Movies | V1.3-B `sender_private` | 54.379 | 48.505 | 1.45616 |
| Grocery | V1.2 `M_repaired` | 82.416 | 76.501 | 0.71930 |
| Grocery | V1.3-A `joint_view` | 83.397 | 77.305 | 0.65513 |
| Grocery | V1.3-B `sender_private` | 82.826 | 76.288 | 0.68703 |
| ele-fashion | V1.2 `M_repaired` | 87.036 | 75.466 | 0.43427 |
| ele-fashion | V1.3-A `joint_view` | 86.207 | 75.170 | 0.42458 |
| ele-fashion | V1.3-B `sender_private` | 86.560 | 75.133 | 0.41796 |

## Training cost and fit summaries

| Dataset | Variant | Total parameters | Mean train seconds | Mean formal peak reserved MiB | Mean Train Acc | Mean Train CE | Mean Train−selected Val Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | `joint_view` | 2,067,935 | 45.70 | 2,694 | 73.763% | 0.76405 | 18.169 pp |
| Movies | `sender_private` | 2,026,877 | 41.47 | 2,388 | 72.898% | 0.77845 | 17.934 pp |
| Grocery | `joint_view` | 2,067,935 | 46.02 | 2,650 | 94.221% | 0.21696 | 10.619 pp |
| Grocery | `sender_private` | 2,026,877 | 49.31 | 2,342 | 96.110% | 0.15474 | 13.270 pp |
| ele-fashion | `joint_view` | 1,799,623 | 260.47 | 13,620 | 89.566% | 0.29822 | 2.264 pp |
| ele-fashion | `sender_private` | 1,758,565 | 294.39 | 11,884 | 90.445% | 0.26965 | 3.005 pp |

The V1.2 model plus external classifier has 2,026,877 parameters on Movies and
Grocery and 1,758,565 on ele-fashion. A adds 41,058 parameters; B adds none.
Run-level best epochs, Train metrics, hashes, and resource snapshots are in
`data/runs.csv` and `data/resource_summary.csv`.

## Mechanism diagnostics

Diagnostics below were recomputed on Validation only from the Best-Accuracy
checkpoints after formal training. The analyzer strictly reloaded each
checkpoint on CPU, reproduced Acc/F1/CE, and verified that evaluation left RNG
state unchanged. These describe module behavior; they do not explain the
observed task-score changes causally.

For A, mean selected private scales across six cells were `sP=1.612` (Text)
and `1.779` (Visual), with `sS=1.496` (Text) and `1.532` (Visual). Across-cell
ranges were Text `sP=1.308–1.848`, `sS=0.350–1.955`, Visual `sP=1.612–1.946`,
and `sS=0.192–1.961`. Thus strengths were not universally neutral; Grocery
seed 43 had especially low Semantic scales. This alone does not show that a
view was reliably or causally identified.

| Variant | Weighted private Physical RMS Text/Visual | Weighted private Semantic RMS Text/Visual | Weighted collaborative Physical RMS Text/Visual | Early Cross raw RMS Text/Visual | Late residual ratio Text/Visual |
|---|---:|---:|---:|---:|---:|
| `joint_view` | 0.386 / 0.431 | 0.100 / 0.101 | 0.380 / 0.428 | 0.602 / 0.432 | 0.092 / 0.088 |
| `sender_private` | 0.247 / 0.252 | 0.067 / 0.071 | 0.246 / 0.252 | 0.669 / 0.462 | 0.109 / 0.098 |

Early Cross calibrated-to-neighbor RMS is 1.0 by the existing RMS calibration
design, so that ratio is not independent evidence of a learned gain. Across
these Validation records, B's raw Cross RMS is somewhat higher on average and
its Late residual ratios are also somewhat higher; this does not distinguish
useful context from repeated smoothing. Full per-cell contribution strengths,
RMS values, calibration, and residual measures are in
`data/weighted_contribution_diagnostics.csv`.

## Attempt and integrity accounting

The formal ledger has 12 successful unique cells, 2 failed formal Attempts,
and 7 `blocked_resource` Attempts; all failed/blocked cells were later
completed with a new Attempt. The two formal aborts were caused by the shared
GPU guard detecting new external compute PIDs 87602 and 89011. No external
process was touched. Six initial GPU-preflight gradient-audit failures are
also preserved separately; after the harness fix, corrected-source preflight
passed 12/12 cells. See `data/failures.json` for the full records and
`PREFLIGHT_REPORT.md` for the guard timeline.

The analyzer verified one successful row per authorized cell, 24 checkpoint
hashes and selection metadata, 12 posthoc strict Best-Accuracy reloads, epoch
history alignment, frozen V1.2 file hashes, matching V1.2 same-seed data/cache/
split references, and the recorded Test-off status. The data manifest records
zero incomplete cells, `test_evaluation=false`, and `link_prediction=false`.

## Limitations and interpretation

- There are only two seeds per dataset in this screen, and V1.2 is historical
  rather than same-job re-trained. Means and paired differences are
  descriptive, not significance tests.
- Best-Accuracy and Best-F1 are different checkpoint-selection targets; their
  metrics are presented separately.
- The mechanism summaries use selected Validation checkpoints and do not
  provide node-level causal attribution or establish that learned strengths
  represent true graph/view reliability.
- No Test evaluation, LP, A+B combination, seed 44, Toys, additional search,
  or follow-on model was run. No A/B winner is selected here.

## Machine-readable evidence

- Attempt ledger: `data/runs.csv`
- Epoch history: `data/epoch_history.csv`
- Same-seed per-run comparison: `data/paired_differences.csv`
- Descriptive means and cost summary: `data/dataset_summary.csv`
- GPU resource snapshots: `data/resource_summary.csv` and
  `data/resource_by_cell.csv`
- View diagnostics: `data/view_diagnostics.csv` and
  `data/weighted_contribution_diagnostics.csv`
- Preflight, failures, checkpoint audit, and manifest:
  `data/preflight.json`, `data/failures.json`, `data/checkpoint_audit.json`,
  and `data/manifest.json`
