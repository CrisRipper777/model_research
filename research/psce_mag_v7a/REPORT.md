# PSCE-MAG V7A: Physical–Semantic Collaborative Experts

## Protocol and provenance

- Baseline: `af848e6c241994d34827fab8d37611159d167965`.
- Frozen implementation: `9c8f1046ba9d90401acd3486752ba085d95a27f7`.
- Branch: `exp/v7a_physical_semantic_collaborative_experts`.
- Protocol: `unified_full_graph_nc_v1`; Validation Accuracy selected checkpoints; Macro-F1 diagnostic.
- Formal runs: 27/27; Test evaluation disabled and no Test metric keys were present.
- Formal failures/retries: 0/27 failed; no formal cell was retried.
- Full repository pytest: passed=True, exit=0.
- Runs use full-graph NC with AdamW (`lr=0.001`, `weight_decay=0.0001`), a 300-epoch cap, patience 30, Validation Accuracy checkpoint selection, fixed K=8 per modality, hidden size 256, four shared experts, and seeds 42–44.

## Validation results

Accuracy and Macro-F1 values are shown as percent mean ± population SD over three seeds.

| Dataset | Variant | Validation Accuracy | Validation Macro-F1 | Best epoch mean |
|---|---|---:|---:|---:|
| Movies | `a0_raw` | 56.009 ± 0.381% | 48.433 ± 1.169% | 76.7 |
| Movies | `a1_static_dual` | 56.079 ± 0.466% | 47.875 ± 0.772% | 81.0 |
| Movies | `a2_conditional_dual` | 56.129 ± 0.332% | 48.888 ± 0.874% | 73.7 |
| Grocery | `a0_raw` | 83.104 ± 0.335% | 75.816 ± 1.382% | 67.3 |
| Grocery | `a1_static_dual` | 83.582 ± 0.253% | 76.145 ± 1.022% | 85.3 |
| Grocery | `a2_conditional_dual` | 83.485 ± 0.524% | 76.721 ± 0.816% | 88.3 |
| ele-fashion | `a0_raw` | 87.361 ± 0.090% | 74.638 ± 1.421% | 185.0 |
| ele-fashion | `a1_static_dual` | 87.600 ± 0.150% | 74.301 ± 0.569% | 155.0 |
| ele-fashion | `a2_conditional_dual` | 87.597 ± 0.084% | 74.632 ± 0.516% | 168.7 |

### Paired comparisons

Deltas are paired by dataset and seed; positive counts are descriptive, not significance tests.

| Comparison | Accuracy Δ (pp) | Macro-F1 Δ (pp) | Positive Accuracy pairs | Positive F1 pairs |
|---|---:|---:|---:|---:|
| a1 − a0 | +0.262 ± 0.220 | -0.189 ± 1.293 | 6/9 | 3/9 |
| a2 − a1 | -0.017 ± 0.234 | +0.640 ± 0.770 | 4/9 | 7/9 |
| a2 − a0 | +0.245 ± 0.209 | +0.451 ± 0.924 | 9/9 | 7/9 |

Per-dataset and per-seed deltas are in `data/paired_comparisons.csv`.

## Candidate graph and retrieval

| Dataset | Nodes | Undirected semantic edges | Nonphysical fraction | Text/Visual Jaccard | Covered nodes | IVF recall@8 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 16672 | 205881 | 0.912 | 0.034 | 16672 | exact |
| Grocery | 17074 | 205557 | 0.938 | 0.040 | 17074 | exact |
| ele-fashion | 97766 | 1102200 | 0.962 | 0.081 | 97766 | 0.988/0.988 |

FAISS provenance, degree/similarity quantiles, cache paths, and fingerprints are in `data/candidate_graph_summary.json`. The nonphysical-edge fraction describes added topology only; it is not evidence of task value.

## GPU preflight and resources

GPU preflight passed: **True** on `NVIDIA GeForce RTX 3090`.

| Dataset | Training step | Validation inference | Peak allocated | Peak reserved | Train step (s) | Relation score (s) | Semantic propagation (s) |
|---|---|---|---:|---:|---:|---:|---:|
| Movies | passed | 0.3293 acc | 3.30 GiB | 3.51 GiB | 0.98 | 0.02 | 0.05 |
| Grocery | passed | 0.1414 acc | 3.15 GiB | 3.51 GiB | 0.20 | 0.01 | 0.02 |
| ele-fashion | passed | 0.2604 acc | 13.79 GiB | 14.71 GiB | 0.70 | 0.02 | 0.08 |

Formal per-run epoch time and peak GPU memory are in `data/resource_profile.json`. No GPU process was stopped.

## Checkpoint semantic-branch diagnostics

Learned edge weights, route strengths, and branch closure report model usage; they are not causal edge-utility estimates. `semantic_diagnostics.csv` records per-run edge-weight quantiles, semantic/Raw state RMS and cosine, shared-expert gradients, semantic contribution ratios, and Validation metrics after closing the semantic path at the selected checkpoint.

Across the three run seeds per variant/dataset (averaging the Text and Visual rows), the mean semantic-to-physical contribution RMS ratios were:

| Dataset | A1 static | A2 conditional |
|---|---:|---:|
| Movies | 0.157 | 1.034 |
| Grocery | 0.087 | 0.188 |
| ele-fashion | 3.811 | 6.501 |

Relation scorer and shared-expert gradients were finite/nonzero in the preflight and selected-checkpoint audits. The much larger Fashion contribution ratio shows that the semantic branch can dominate the physical contribution there; the paired NC results remain the evidence for task value.

## Historical context

| Dataset | Historical V4A R0 Accuracy / Macro-F1 | Historical PIGPR-C1 RGD Accuracy / Macro-F1 |
|---|---:|---:|
| Movies | 56.029 ± 0.385% / 47.997 ± 1.637% | 56.389 ± 0.170% / 49.698 ± 0.281% |
| Grocery | 83.094 ± 0.325% / 75.766 ± 1.387% | 83.094 ± 0.193% / 77.342 ± 0.228% |
| ele-fashion | 87.413 ± 0.128% / 74.778 ± 1.560% | 87.324 ± 0.042% / 74.536 ± 0.252% |

These earlier reports/configurations are descriptive context only; they are not strict paired baselines for the V7A campaign.

## Objective readout

- Semantic context (A1−A0): +0.262 pp Accuracy and -0.189 pp Macro-F1 on average; positive on 6/9 Accuracy and 3/9 F1 seed pairs. Accuracy gains were +0.070 pp Movies, +0.478 pp Grocery, and +0.239 pp ele-fashion; Macro-F1 decreased on Movies and ele-fashion.
- Conditional use (A2−A1): -0.017 pp Accuracy and +0.640 pp Macro-F1; positive on 4/9 Accuracy and 7/9 F1 seed pairs. The Macro-F1 gain appeared in all three dataset means (+1.013 pp Movies, +0.576 pp Grocery, +0.331 pp ele-fashion), while Accuracy was essentially unchanged.
- Full model (A2−A0): +0.245 pp Accuracy and +0.451 pp Macro-F1; Accuracy was higher in all 9 paired seed comparisons, and Macro-F1 was higher in 7/9. By dataset, Accuracy deltas were +0.120 pp Movies, +0.381 pp Grocery, and +0.235 pp ele-fashion; Macro-F1 deltas were +0.455 pp, +0.905 pp, and -0.006 pp, respectively.

These are descriptive results over three fixed seeds and three datasets, not a significance test. The semantic graph gives a small positive Accuracy difference in this experiment matrix, but A1's Macro-F1 is mixed. A2 adds little/no Accuracy over A1 while improving Macro-F1 consistently across dataset means. The full A2 model improves both aggregate metrics over A0, with small effect sizes; broader reliability is not established by three seeds.

**Recommendation:** stop architecture expansion and hyperparameter tuning for this V7A cycle. Preserve the result as a modest, validation-only gain: A2 is the better choice when Macro-F1 is a priority, while there is no meaningful Accuracy case for conditional routing over A1. Any further claim should come from a separately planned replication, not from adding mechanisms to this completed experiment.

## Limitations and artifacts

- Full-graph validation-only NC was evaluated. Link prediction is explicitly unsupported because its sampler uses local IDs; no LP validation is claimed.
- The approximate FAISS retrieval recall sample is feature-only and does not use labels or validation outcomes.
- Historical comparisons differ in run/config provenance and are not paired.
- Failed/retried cells are recorded in `data/failures.json` and the campaign manifest.
- Candidate caches, checkpoints, and training logs are stored under ignored `outputs/psce_mag_v7a/`.

Report generated: 2026-10-10T11:52:20+00:00.
