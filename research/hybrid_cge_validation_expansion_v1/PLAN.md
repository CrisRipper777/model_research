# Hybrid-CGE V1.1 Validation Expansion Plan

## Objective

Assess the frozen `H2_hybrid_plus` model through a bounded training-configuration study, first-use generalization on Toys and Reddit-S, and same-protocol comparisons with MLP, GraphSAGE, MMGCN, and the repository's RoleMAG interface adaptation. This campaign adds no model component and does not select with Test metrics.

## Provenance and frozen scope

- Parent branch: `exp/hybrid_cge_v11`
- Parent commit: `b726d4ef1bebd9a94a0b3a21e2d52ee7f2b79f2b`
- Frozen Hybrid-CGE implementation: `affc45b94f4298eb4c524e6fd7d9f55591481c8a`
- Working branch: `exp/hybrid_cge_validation_expansion_v1`
- Hybrid-CGE forward math, expert count, dimensions, propagation paths, semantic KNN K=8, same-node candidates, late cross-dropout 0.15, gate/router/fusion, loss definition and the NC task implementation remain frozen.

## Work packages

### A. Fixed-structure optimization

Use historical H2-Frozen results without rerunning them. Train H2-R1 (`dropout=.35`, `weight_decay=1e-4`, private auxiliary CE `.30`) and H2-R2 (`dropout=.30`, `weight_decay=1e-3`, private auxiliary CE `.10`) on Movies, Grocery and ele-fashion with seeds 42, 43 and 44. Keep all other H2 settings fixed. Run seed 42 first (six cells), audit them for integrity and finite metrics, then run seeds 43/44 (twelve cells). Compare each candidate globally across the three datasets; do not choose per-dataset configurations. Accuracy is primary, with same-checkpoint Macro-F1, best-F1 checkpoint metrics, Validation CE and Train–Validation gaps used to judge trade-offs.

### B. First-use generalization

Select one global condition from A, retaining H2-Frozen if neither candidate is better overall. Apply that same condition to Toys and Reddit-S for seeds 42–44. Keep the configured RoBERTa text and CLIP image arrays, existing project-stratified 60/20/20 split files when present, full physical graph, and V7B.1 semantic candidates built from unlabeled features at K=8. Do not tune this condition on these two datasets.

### C. Same-protocol baselines

Preflight and, when compatible, run MLP, GraphSAGE, MMGCN and RoleMAG on Movies, Grocery, ele-fashion, Toys and Reddit-S for seeds 42–44 (up to 60 cells). Keep node/features/physical graph/split/class definitions and validation protocol fixed. Record method-specific dimensional and architectural differences. MLP is feature-only; MMGCN and RoleMAG are project adaptations, not asserted to be exact reproduction of every original benchmark implementation.

## Execution order and gates

1. Inspect the frozen source and all five datasets; hash features, physical graph, split files and semantic caches. Verify actual Hybrid-CGE dropout module probabilities for each candidate configuration.
2. Run unit/integration tests and CPU data/model construction preflights. Freeze implementation and protocol in a commit before formal training.
3. Verify GPU visibility and run the existing physical-GPU guard before each launch and each new job. Never terminate, signal, or otherwise alter another user's process. If the selected GPU is occupied, stop safely and retain progress.
4. Complete six optimization seed-42 runs. Require finite train/validation histories, compatible data/cache/splits, matched initial model and classifier hashes within each Dataset × Seed, valid checkpoints, and Test-off evidence. There is no historical-score threshold.
5. Complete twelve optimization seed-43/44 runs and select one global configuration from validation-only paired results.
6. Audit generalization caches and run a small full-graph forward/backward/checkpoint/index-alignment preflight, then complete six Toys/Reddit-S runs.
7. Preflight all baseline architectures on all five datasets, then run every compatible Dataset × Baseline × Seed cell. Record and stop on concrete incompatibility or resource-guard failure; never mark a missing cell successful.
8. Analyze only Train/Validation results, write the four reports, preserve failure/retry records, commit and push the evidence.

## Success and interpretation

Success means the frozen model implementation and protocol are auditable; candidate configurations are compared fairly; new-dataset and baseline status is explicit; all completed cells are traceable; and no Test labels or metrics enter training or selection. Historical single-run Pilot scores are context only. The study does not establish state-of-the-art status.
