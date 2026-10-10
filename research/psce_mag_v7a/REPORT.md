# PSCE-MAG V7A implementation and pre-campaign report

> Progress report: implementation and all pre-campaign correctness, GPU fit, and smoke gates passed. The model is not yet frozen, the formal campaign has not started (0/27 cells), and no task-gain conclusion is available.

## Provenance and protocol

- Branch: `exp/v7a_physical_semantic_collaborative_experts`
- Baseline: `af848e6c241994d34827fab8d37611159d167965`
- Freeze SHA: pending the implementation commit
- Formal campaign: **0/27 runs**
- Protocol: `unified_full_graph_nc_v1`; full-graph node classification; Validation Accuracy checkpoint selection; Test evaluation disabled
- Variants: A0 `a0_raw`, A1 `a1_static_dual`, A2 `a2_conditional_dual`
- Datasets/seeds: Movies, Grocery, ele-fashion × 42, 43, 44

## Candidate graphs

Candidates were built from modality features and physical edges only. The builder did not read labels or split files. K=8 is fixed for each modality.

| Dataset | Nodes | Undirected candidate edges | Nonphysical fraction | Text/Visual Jaccard | Retrieval audit |
|---|---:|---:|---:|---:|---|
| Movies | 16,672 | 205,881 | 0.9121 | 0.0340 | exact FAISS |
| Grocery | 17,074 | 205,557 | 0.9378 | 0.0399 | exact FAISS |
| ele-fashion | 97,766 | 1,102,200 | 0.9615 | 0.0811 | IVF; sampled Recall@8 0.9883 for Text and Visual |

The approximate retrieval audit sampled 256 feature queries with seed 2026. These topology statistics do not establish task value. FAISS settings, cache fingerprints, and integrity metadata are recorded in [candidate_graph_summary.json](data/candidate_graph_summary.json).

## Implementation and pre-campaign correctness

- A0 matches the historical Movies V4A `R0_raw` seed-42 path after mapping 48 common weights: across all 16,672 nodes, max absolute and relative embedding errors were 0 and auxiliary-loss error was 0 (`atol=2e-6`, `rtol=2e-5`). See [raw_regression_summary.json](data/raw_regression_summary.json).
- Focused V7A tests passed **27/27**; the full repository suite passed **356/356**, with six deprecation warnings and no failures. See [test_summary.json](data/test_summary.json).
- Three one-epoch CPU integration runs and three one-epoch GPU Movies smokes exercised the full-graph training/checkpoint path. GPU smoke audits passed for all variants: validation metrics were finite, Test metrics were absent, and A1/A2 selected-checkpoint semantic diagnostics were present. These short runs are software-path checks, not comparative evidence.
- V7A link prediction is rejected before sampled-subgraph processing; this implementation supports node classification only.
- The implementation excludes V6A Attention, edge-deletion effect estimation, OT/InfoNCE, separate relation graph banks, private experts, and NC sampling.

## GPU preflight

A2 completed one training-label loss/backward/optimizer step followed by complete full-graph validation inference on GPU 1 (NVIDIA RTX 3090) for all datasets. All audited router, expert, relation, and semantic gradients were finite and nonzero; no Test metrics were generated. Resource values below are from the recorded preflight. GiB uses 1024³ bytes.

| Dataset | Nodes | Model parameters | Peak allocated | Peak reserved | Train step (s) | Validation inference (s) | Relation scoring (s) | Semantic propagation (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 16,672 | 1,576,387 | 3.295 GiB | 3.514 GiB | 1.048 | 0.090 | 0.018 | 0.061 |
| Grocery | 17,074 | 1,576,387 | 3.152 GiB | 3.510 GiB | 0.197 | 0.085 | 0.006 | 0.021 |
| ele-fashion | 97,766 | 1,314,243 | 13.786 GiB | 14.707 GiB | 0.688 | 0.285 | 0.019 | 0.075 |

No OOM occurred. The sandbox did not expose NVIDIA device nodes, so preflight/smokes ran through the approved host GPU execution path. GPU 1 was shared with other activity; no process was stopped. Full timings, memory snapshots, and per-parameter gradient diagnostics are in [preflight_summary.json](data/preflight_summary.json), and smoke audits are in [smoke_summary.json](data/smoke_summary.json).

## Research conclusion and next step

Whether semantic candidates improve validation performance and whether node-conditional A2 improves over static A1 remain **undetermined**. The one-epoch smokes are not formal results. After committing the frozen implementation, run all 27 validation-only NC cells, analyze paired seed-level deltas and selected-checkpoint semantic diagnostics, then replace this progress report with the campaign report. No Test metrics should be collected or reported.
