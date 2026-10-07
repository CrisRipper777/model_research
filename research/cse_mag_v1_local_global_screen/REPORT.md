# CSE-MAG V1 Local/Global Screening Report

## 1. Provenance and protocol

- Branch: `exp/cse_mag_v1_local_global_screen`
- Frozen commit SHA: `dc0e50c621f60ce802e3e27493ff6b8f45be911d`
- Environment: Python 3.12.13, PyTorch 2.4.0+cu121, CUDA runtime 12.1
- GPU: NVIDIA GeForce RTX 3090 (`cuda:1`)
- Datasets: Movies, Grocery, ele-fashion; seeds: 42, 43, 44
- Protocol: `unified_full_graph_nc_v1`; checkpoint selected by best Validation Accuracy.
- Test evaluation: **false** for the smoke and all campaign runs.
- Completed runs: 36 / 36; failures: 0.

## 2. Code and smoke audit

- Targeted tests: `4 passed`; full `tests/`: `4 passed` (one upstream PyG deprecation warning).
- Test invocation note: the `pytest` shell entry point initially resolved to the host Python 3.8 install; both requested test runs passed via `conda run -n yhf_env python -m pytest`.
- Smoke: Movies, seed 42, four variants, one epoch each; all four completed, saved checkpoints and validation-only `run_metrics.json`, with no OOM/NaN/Inf.
- Smoke ran with the launcher's `--allow-dirty` guard for a test-only fairness assertion; model and configuration files were unchanged. That test change was committed as the frozen checkpoint before the formal campaign.
- Initialization fairness: same-seed Text/Visual projectors, all fusion weights, and downstream classifier are tensor-equal across variants; tested.
- V1 prior initialization: Local/Global mixture `(0.5, 0.5)` and structural gate `sigmoid(-2) = 0.1192029`; tested.
- Isolated-node prior preservation: Local and Global displacements are zero and V1 output equals its intrinsic prior; tested.
- B2 Top-K audit on the post-campaign Movies/seed 42 best checkpoint: selected 2 of 4 experts per modality/node; all experts selected = `True`, all experts had nonzero gradient = `True`.
- Graph operator removes self-loops; dataset configuration also has `add_self_loops: false`. No implementation or scientific-model changes were needed; the only code change was the initialization fairness test extension.
- B2 is an MvCGE-style direct-transfer control, not an exact MvCGE reproduction.

- Campaign failures: none.

## 3. Main validation results

| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Model trainable params | Classifier params |
|---|---|---:|---:|---:|---:|
| Movies | B0_independent | 52.64% ± 0.26% | 40.12% ± 1.02% | 1,577,734 | 5,140 |
| Movies | B1_shared_static | 52.95% ± 0.62% | 42.01% ± 0.65% | 1,314,563 | 5,140 |
| Movies | B2_mvcge_style | 51.59% ± 0.49% | 40.49% ± 1.90% | 1,481,124 | 5,140 |
| Movies | B3_v1 | 52.22% ± 0.71% | 39.87% ± 2.06% | 1,415,459 | 5,140 |
| Grocery | B0_independent | 79.92% ± 0.59% | 71.40% ± 1.25% | 1,577,734 | 5,140 |
| Grocery | B1_shared_static | 79.67% ± 0.16% | 70.90% ± 0.90% | 1,314,563 | 5,140 |
| Grocery | B2_mvcge_style | 75.43% ± 0.53% | 65.27% ± 0.54% | 1,481,124 | 5,140 |
| Grocery | B3_v1 | 80.45% ± 0.44% | 72.24% ± 1.33% | 1,415,459 | 5,140 |
| ele-fashion | B0_independent | 87.00% ± 0.07% | 73.97% ± 0.64% | 1,315,590 | 3,084 |
| ele-fashion | B1_shared_static | 87.23% ± 0.11% | 73.94% ± 0.47% | 1,052,419 | 3,084 |
| ele-fashion | B2_mvcge_style | 86.07% ± 0.12% | 70.96% ± 0.69% | 1,218,980 | 3,084 |
| ele-fashion | B3_v1 | 87.08% ± 0.06% | 74.22% ± 0.43% | 1,153,315 | 3,084 |

## 4. Paired V1 comparisons

| Comparison | Overall Δ Val Accuracy | Overall Δ Val Macro-F1 | Positive accuracy pairs | Positive F1 pairs |
|---|---:|---:|---:|---:|
| B3_v1 − B0_independent | +0.00063 | +0.00279 | 7/9 | 7/9 |
| B3_v1 − B1_shared_static | -0.00032 | -0.00171 | 4/9 | 5/9 |
| B3_v1 − B2_mvcge_style | +0.02220 | +0.03204 | 8/9 | 6/9 |

Dataset-level paired means and positive seeds:

| Comparison | Dataset | Δ Val Accuracy mean | Δ Val Macro-F1 mean | Positive accuracy seeds | Positive F1 seeds |
|---|---|---:|---:|---:|---:|
| B3_v1 − B0_independent | Movies | -0.00420 | -0.00248 | 1/3 | 2/3 |
| B3_v1 − B0_independent | Grocery | +0.00527 | +0.00840 | 3/3 | 3/3 |
| B3_v1 − B0_independent | ele-fashion | +0.00082 | +0.00246 | 3/3 | 2/3 |
| B3_v1 − B1_shared_static | Movies | -0.00730 | -0.02134 | 0/3 | 0/3 |
| B3_v1 − B1_shared_static | Grocery | +0.00781 | +0.01338 | 3/3 | 3/3 |
| B3_v1 − B1_shared_static | ele-fashion | -0.00147 | +0.00282 | 1/3 | 2/3 |
| B3_v1 − B2_mvcge_style | Movies | +0.00630 | -0.00617 | 2/3 | 0/3 |
| B3_v1 − B2_mvcge_style | Grocery | +0.05017 | +0.06974 | 3/3 | 3/3 |
| B3_v1 − B2_mvcge_style | ele-fashion | +0.01013 | +0.03253 | 3/3 | 3/3 |

## 5. Observations

- Highest mean Validation Accuracy on Movies: B1_shared_static.
- Highest mean Validation Accuracy on Grocery: B3_v1.
- Highest mean Validation Accuracy on ele-fashion: B1_shared_static.
- Across all paired runs, B3_v1 overall mean Validation Accuracy versus B0_independent: +0.00063; it is higher = `True`.
- Across all paired runs, B3_v1 overall mean Validation Accuracy versus B1_shared_static: -0.00032; it is higher = `False`.
- Across all paired runs, B3_v1 overall mean Validation Accuracy versus B2_mvcge_style: +0.02220; it is higher = `True`.
- B3_v1 overall mean Validation Macro-F1 delta versus B0_independent: +0.00279.
- B3_v1 overall mean Validation Macro-F1 delta versus B1_shared_static: -0.00171.
- B3_v1 overall mean Validation Macro-F1 delta versus B2_mvcge_style: +0.03204.
- On Movies, B3_v1's mean Validation Accuracy delta versus B2 is +0.00630, while its mean Macro-F1 delta is -0.00617.
- Model trainable parameter counts differ by variant: B0_independent 1,577,734, B1_shared_static 1,314,563, B2_mvcge_style 1,481,124, B3_v1 1,415,459. Classifier size is 5,140 parameters for every variant.
- These are descriptive validation results from a coarse screen; no significance tests were run.

## 6. Decision map

Conditions A–D use the overall mean Validation Accuracy across the nine paired runs. For E, ‘clearly best’ means uniquely highest dataset-level mean Validation Accuracy on all three datasets.

- **C.** Node-specific routing has not shown value; do not further complicate the router yet.
