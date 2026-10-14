# CG-CGE V1 Pilot Report

## Status

Implementation and CPU-only data/init checks are in progress. No formal CG-CGE training result is available yet. The accepted run matrix is 18 runs: three datasets × B1/B2 × seeds 42, 43 and 44. The first execution batch is six seed-42 runs.

No Test evaluation is planned or enabled. All task-performance numbers in the final report will be from Train/Validation only.

## Frozen B0 reference

The frozen View-CGE V1 `A2_cross_source` source is `research/mag_view_cge_v1/data/runs.csv`; its reports and records remain untouched. The new CPU preflight reports exact B0 Dataset × Seed split, feature, physical graph, semantic cache, candidate-cache, V7B.1 and `unified_full_graph_nc_v1` matches for Movies, Grocery and ele-fashion at seeds 42/43/44. Thus the stored A2 Validation metrics are eligible for paired descriptive comparisons, subject to preserving these fingerprints in each formal run.

## Formal results

The final version of this report will be populated from `data/runs.csv`, `data/summary.csv`, `data/paired_comparisons.csv`, `data/epoch_history.csv` and `data/mechanism_diagnostics.csv` after all 18 runs and checkpoint audits complete. It will include Best-Accuracy and Best-F1 Validation Accuracy/Macro-F1/CE, train-validation gaps, B1−B0/B2−B0/B2−B1 paired differences, resources, auxiliary-loss trajectories, cross-source diagnostics and failed attempts.

Current results: **pending GPU preflight and formal runs**.
