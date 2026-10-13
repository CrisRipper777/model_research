# RM-CGE V2 Experiment Plan

## Frozen matrix

The formal matrix is Movies, Grocery, and ele-fashion × V0/V1/V2 × seeds
42/43/44, for 27 runs. All use the official full-graph NC split, unchanged
input features and physical graph, the verified V7B.1 K=8 cache, hidden size
256, four physical hops, four shared experts, Top-2 routing, bottleneck 64,
dropout 0.2, and AdamW (`lr=1e-3`, `weight_decay=1e-4`). The task uses the
existing 0.01-weighted R0 balance term, Validation Accuracy selection,
300-epoch maximum, patience 30, minimum epoch 30, and per-epoch evaluation.
Test evaluation is disabled.

V0 is the parent V7A `a0_raw` model. V1 adds fixed unit-weight cross-modal
physical messages, modality-specific one-hop KNN context, shared expert
coefficients, and a zero-initialized node router correction. V2 has the same
parameter shapes and initial values as V1 and enables receiver-modality
conditioned physical weights. V2−V1 is interpreted as an internal paired
comparison of the learned edge path, with its effective gradient activity
made explicit.

## Execution gates

1. Run the full repository test suite, V0/R0 equivalence, V1/V2 unit-edge
   equivalence, direction/flag/isolate/SpMM checks, gradient and AdamW update
   audits, checkpoint reload, and an ele-fashion GPU update.
2. Save the immutable implementation and audit evidence in the Implementation
   Freeze commit.
3. Run the nine seed-42 jobs. Check finite histories, checkpoints, parameter
   updates, and the absence of Test metrics or label evaluation.
4. If that batch passes, run the remaining 18 jobs for seeds 43 and 44.
5. Analyze Validation metrics, paired differences, edge/context/router
   diagnostics, frozen interventions, and resource use. Commit the evidence
   and push the branch without force.

No history-dependent accuracy threshold, hyperparameter search, Test
evaluation, new loss, extra expert bank, or additional model arm is allowed.

## Outputs

Large checkpoints and logs live below ignored `outputs/rmcge_v2/`. Compact
machine-readable evidence lives in `research/rmcge_v2/data/`: manifest, run
rows, epoch histories, paired comparisons, edge/context/router diagnostics,
frozen interventions, gradient audit, resource profile, implementation
checks, and preserved failures.
