# MACE-MAG V6A: shared structural expert collaboration

## Protocol and reproducibility

- Branch: `exp/mace_mag_v6a_expert_collaboration`; parent: `0c66c6d65ed5da9e5850b7cfeff2454074e52a3b`; freeze SHA: `515a04d7111108ba2d6c1829db1c612cfa3407c9`.
- Campaign: 36/36 validation-only full-graph NC runs; protocol `unified_full_graph_nc_v1`; datasets Movies/Grocery/ele-fashion; seeds 42/43/44; variants A0–A3.
- Every formal command explicitly set `task.evaluate_test=false`. Checkpoint selection used Validation Accuracy. Macro-F1 is Validation-only. No Test metric was computed, read, or saved.
- No formal LP, hyperparameter search, sampling, split change, or post-freeze architecture change was performed.
- Environment: Python 3.12.13, PyTorch 2.4.0+cu121, CUDA toolkit 12.1, PyG 2.7.0; device NVIDIA GeForce RTX 3090.
- Dataset preflight and A2 full-graph forward/backward passed on all three datasets; smoke passed 4/4. Preflight used no labels/splits and only synthetic class targets for backward validation.
- Full repository pytest: 329 passed, 2 warnings, 0 failed.

## Formal validation performance

Means and population standard deviations are over the three model seeds; accuracy and Macro-F1 are shown in percent. Runtime and GPU memory refer to each selected formal training process.

| Dataset | Variant | Val Accuracy (%) | Val Macro-F1 (%) | Best epoch mean | Trainable params (model + head) | Mean training time (s) | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | a0_static | 54.78 ± 0.17 | 46.44 ± 0.35 | 66.3 | 1,448,758 + 5,140 | 16.8 | 2.60 | 2.71 |
| Movies | a1_node | 54.88 ± 0.21 | 46.20 ± 0.41 | 73.3 | 1,448,758 + 5,140 | 18.6 | 2.64 | 2.96 |
| Movies | a2_cross | 55.13 ± 0.09 | 47.17 ± 0.72 | 69.0 | 1,579,978 + 5,140 | 20.8 | 2.92 | 3.34 |
| Movies | a3_intra | 55.26 ± 0.16 | 46.95 ± 0.13 | 64.3 | 1,579,978 + 5,140 | 20.1 | 2.92 | 3.34 |
| Grocery | a0_static | 82.35 ± 0.25 | 74.77 ± 0.50 | 69.3 | 1,448,758 + 5,140 | 17.4 | 2.46 | 2.59 |
| Grocery | a1_node | 82.65 ± 0.39 | 75.04 ± 1.11 | 84.0 | 1,448,758 + 5,140 | 21.2 | 2.50 | 2.83 |
| Grocery | a2_cross | 82.71 ± 0.39 | 74.77 ± 0.95 | 79.0 | 1,579,978 + 5,140 | 23.0 | 2.79 | 3.21 |
| Grocery | a3_intra | 82.82 ± 0.36 | 75.21 ± 0.88 | 67.0 | 1,579,978 + 5,140 | 20.7 | 2.79 | 3.21 |
| ele-fashion | a0_static | 87.39 ± 0.16 | 74.42 ± 1.32 | 175.3 | 1,186,614 + 3,084 | 120.8 | 10.39 | 11.66 |
| ele-fashion | a1_node | 87.18 ± 0.20 | 73.31 ± 0.87 | 107.0 | 1,186,614 + 3,084 | 82.7 | 10.60 | 11.86 |
| ele-fashion | a2_cross | 87.18 ± 0.10 | 74.05 ± 0.42 | 117.3 | 1,317,834 + 3,084 | 103.1 | 12.25 | 13.88 |
| ele-fashion | a3_intra | 87.23 ± 0.04 | 73.96 ± 0.33 | 131.7 | 1,317,834 + 3,084 | 113.7 | 12.25 | 13.88 |

A0/A1 train the same shared trunk and classifier parameter count. A2/A3 each add 131,218 trainable attention parameters plus two modality collaboration-strength scalars (131,220 extra model parameters total); A2 and A3 have identical trainable parameter counts and initial attention tensors.

## Paired variant comparisons

Deltas are percentage points paired by dataset and seed. Positive/negative counts are descriptive only; no significance test was performed.

| Comparison | Dataset | Accuracy Δ (pp) | Accuracy +/− pairs | Macro-F1 Δ (pp) | Macro-F1 +/− pairs |
|---|---|---:|---:|---:|---:|
| a1_node-a0_static | Movies | +0.100 | 1/3 / 2/3 | -0.243 | 0/3 / 3/3 |
| a1_node-a0_static | Grocery | +0.293 | 3/3 / 0/3 | +0.265 | 2/3 / 1/3 |
| a1_node-a0_static | ele-fashion | -0.215 | 1/3 / 2/3 | -1.108 | 0/3 / 3/3 |
| a1_node-a0_static | ALL | +0.059 | 5/9 / 4/9 | -0.362 | 2/9 / 7/9 |
| a2_cross-a1_node | Movies | +0.250 | 3/3 / 0/3 | +0.967 | 3/3 / 0/3 |
| a2_cross-a1_node | Grocery | +0.068 | 2/3 / 0/3 | -0.270 | 1/3 / 2/3 |
| a2_cross-a1_node | ele-fashion | +0.007 | 2/3 / 1/3 | +0.742 | 3/3 / 0/3 |
| a2_cross-a1_node | ALL | +0.108 | 7/9 / 1/9 | +0.479 | 7/9 / 2/9 |
| a2_cross-a3_intra | Movies | -0.130 | 0/3 / 3/3 | +0.214 | 1/3 / 2/3 |
| a2_cross-a3_intra | Grocery | -0.107 | 1/3 / 2/3 | -0.441 | 0/3 / 3/3 |
| a2_cross-a3_intra | ele-fashion | -0.044 | 1/3 / 2/3 | +0.088 | 2/3 / 1/3 |
| a2_cross-a3_intra | ALL | -0.094 | 2/9 / 7/9 | -0.046 | 3/9 / 6/9 |
| a3_intra-a1_node | Movies | +0.380 | 3/3 / 0/3 | +0.753 | 3/3 / 0/3 |
| a3_intra-a1_node | Grocery | +0.176 | 2/3 / 1/3 | +0.171 | 2/3 / 1/3 |
| a3_intra-a1_node | ele-fashion | +0.051 | 1/3 / 2/3 | +0.653 | 2/3 / 1/3 |
| a3_intra-a1_node | ALL | +0.202 | 6/9 / 3/9 | +0.526 | 7/9 / 2/9 |

Seed-level paired values are preserved in `data/paired_comparisons.csv`.

## Collaboration diagnostics

Selected-checkpoint diagnostics describe use patterns, not causal task utility. Expert use is the mean proportion of selected Top-2 slots; combination count is the mean number of distinct unordered Top-2 pairs within a run.

| Variant | Target modality | Expert-use distribution [0..3] | Mean distinct Top-2 pairs | Source attention | Null attention | Same-ID attention | Scaled correction / structure RMS | Mean τ | Mean λ | Nonzero key-grad runs |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a0_static | text | [0.389, 0.222, 0.056, 0.333] | 1.00 | n/a | n/a | n/a | 0.0000 | 0.254 | 0.000 | 0/n/a |
| a0_static | visual | [0.389, 0.222, 0.111, 0.278] | 1.00 | n/a | n/a | n/a | 0.0000 | 0.253 | 0.000 | 0/n/a |
| a1_node | text | [0.295, 0.308, 0.185, 0.212] | 4.56 | n/a | n/a | n/a | 0.0000 | 0.255 | 0.000 | 0/n/a |
| a1_node | visual | [0.384, 0.183, 0.203, 0.231] | 4.56 | n/a | n/a | n/a | 0.0000 | 0.259 | 0.000 | 0/n/a |
| a2_cross | text | [0.283, 0.269, 0.229, 0.218] | 5.00 | 0.658 | 0.342 | 0.244 | 0.7466 | 0.254 | 0.155 | 9/9 |
| a2_cross | visual | [0.384, 0.247, 0.145, 0.223] | 4.44 | 0.578 | 0.422 | 0.209 | 0.4338 | 0.257 | 0.154 | 9/9 |
| a3_intra | text | [0.293, 0.268, 0.194, 0.245] | 4.67 | 0.575 | 0.425 | 0.332 | 0.4540 | 0.254 | 0.153 | 9/9 |
| a3_intra | visual | [0.390, 0.253, 0.135, 0.222] | 4.44 | 0.632 | 0.368 | 0.366 | 0.5305 | 0.258 | 0.155 | 9/9 |

A0/A1 have no active attention module; their attention measures are `n/a`. V6A smoke/preflight and first-epoch training telemetry record nonzero key/value gradients for the collaboration arms. Attention weights are not interpreted as effect utility.

## Historical performance context

These are historical validation results from different branch/configuration runs, not strict paired comparisons or retrained baselines.

| Dataset | Historical model | Val Accuracy (%) | Val Macro-F1 (%) | Provenance note |
|---|---|---:|---:|---|
| Movies | V4A R0_raw | 56.029 ± 0.385 | 47.997 ± 1.637 | V4A branch `exp/mvcge_mag_v4a_raw_anchored_residual_screen`; R0 Raw trajectory |
| Movies | PIGPR-C1 RGD | 56.389 ± 0.170 | 49.698 ± 0.281 | PIGPR-C1 commit `cd9d440`; RGD direct GPR |
| Grocery | V4A R0_raw | 83.094 ± 0.325 | 75.766 ± 1.387 | V4A branch `exp/mvcge_mag_v4a_raw_anchored_residual_screen`; R0 Raw trajectory |
| Grocery | PIGPR-C1 RGD | 83.094 ± 0.193 | 77.342 ± 0.228 | PIGPR-C1 commit `cd9d440`; RGD direct GPR |
| ele-fashion | V4A R0_raw | 87.413 ± 0.128 | 74.778 ± 1.560 | V4A branch `exp/mvcge_mag_v4a_raw_anchored_residual_screen`; R0 Raw trajectory |
| ele-fashion | PIGPR-C1 RGD | 87.324 ± 0.042 | 74.536 ± 0.252 | PIGPR-C1 commit `cd9d440`; RGD direct GPR |

The historical protocols/configurations differ from this frozen V6A campaign, so this table gives context only. V6A does not rerun Raw GPR or PIGPR-C1.

## Resources and failures

- GPU: NVIDIA GeForce RTX 3090 on `cuda:1`, 23.7 GiB total. At launch GPU0 had an unrelated VLLM process using about 23.1 GiB; GPU1 was selected and no other process was terminated.
- Preflight peak: Movies/Grocery/ele-fashion allocated about 2.90 GiB/2.79 GiB/12.24 GiB. Formal peak per run is in `data/summary.csv` and `data/resource_profile.json`.
- Full-graph execution used the standard unchunked propagation and no activation checkpointing or CPU feature staging. OOMs: none. No variant-specific budget or dimension changes were made.
- Outputs and selected checkpoints remain under ignored `outputs/mace_mag_v6a_expert_collaboration/`; raw datasets and large model files are not committed.

## Conclusion

- A2−A1 averaged +0.108 pp Accuracy and +0.479 pp Macro-F1 over nine paired runs; dataset Accuracy directions were 3/3 positive and dataset Macro-F1 directions 2/3 positive. This is the direct validation signal for adding cross-modal collaboration.
- A2−A3 averaged -0.094 pp Accuracy and -0.046 pp Macro-F1; dataset Accuracy directions were 0/3 positive. This compares cross-modal sourcing with a matched-capacity intra-modal attention control.
- A2 has the best Validation Accuracy on 0/3 datasets among the four V6A variants. Against historical PIGPR-C1 RGD, A2 Accuracy mean differences are -1.260 pp, -0.380 pp, -0.140 pp (context only, not strict paired evidence).
- Decision: A2 improves on A1 on average but does not outperform its matched-capacity A3 control on average; the evidence does not isolate cross-modal sourcing as the source of the gain. Treat the route as unresolved before any follow-on.

These conclusions use Validation metrics only, three model seeds on fixed splits, and descriptive paired deltas. They are not significance tests or claims about Test generalization.

## Artifacts

- `DESIGN.md`, `README.md`, `data/environment.json`, `data/test_summary.json`, `data/preflight_summary.json`, `data/smoke_summary.json`
- `data/campaign_manifest.json`, `data/run_rows.json`, `data/summary.csv`, `data/paired_comparisons.csv`
- `data/resource_profile.json`, `data/collaboration_diagnostics.csv`
- Formal logs/checkpoints: `outputs/mace_mag_v6a_expert_collaboration/formal/`
