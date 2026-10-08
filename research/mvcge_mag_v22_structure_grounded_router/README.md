# MvCGE-MAG V2.2: Structure-Grounded Expert Routing

This screen asks whether the V2.1 node router lacked node-modality structural
evidence aligned with the shared expert bank. It changes only router evidence
and expert matching. Intrinsic projectors, the four RawPoly structural experts,
Top-2 routing, modality-static strength, active-only load balancing, and late
fusion remain fixed.

## Frozen protocol

- Branch: `exp/mvcge_mag_v22_structure_grounded_router`
- Parent: `c57b57fcaf742f76416e9ed72df47d1f372a0f5b`
- Datasets: Movies, Grocery, `ele-fashion`
- Seeds: 42, 43, 44
- Variants: R0 modality-static; R1 free node; R2 structure-grounded; R3 expert compatibility
- Task: full-graph node classification, `task.evaluate_test=false`
- Checkpoint selection: Validation Accuracy
- Smoke: Movies, seed 42, one epoch per variant

Run the targeted tests and full suite in `yhf_env`, then smoke the four variants.
After smoke passes, freeze implementation with the specified freeze commit. The
formal 36-run campaign must start with a clean worktree at that freeze commit.
Do not change model or configuration files after the freeze commit.

The analyzer audits validation-selected checkpoints using features, graph edges,
weights, and validation-selected metadata only. It does not load or inspect
labels. No Test metrics are expected or permitted.

## Descriptive interpretation rules

All paired deltas are descriptive; no significance tests are run. For report
wording, “approximately equal” means absolute overall deltas no larger than
0.10 percentage points for both Accuracy and Macro-F1. A result is a stable
positive screen signal when overall Accuracy and Macro-F1 deltas are positive,
Accuracy is positive in at least 6/9 matched pairs, and Accuracy means are
positive on at least two of the three datasets. “Near-zero” learned residuals
are defined as mean eta/kappa at most 0.05; “no material route change” means
mean route-to-static JS at most 1e-6 nats. These are transparent reading rules,
not inferential thresholds.

Interpretation A requires R2 to exceed both R1 and R0 on both overall metrics.
B requires a positive R2−R1 pattern while R2 and R0 are approximately equal.
C describes positive R3−R2 deltas and notes whether learned kappa is above the
near-zero reading threshold. D applies the stable-positive rule to R2 or R3
versus R0. E requires R2/R3 approximately equal to R0, near-zero eta/kappa,
and no material route change. F requires approximate metric equality but
non-near-zero residuals and material route changes. G applies when R2 or R3
is below R0 on both overall metrics. Patterns may overlap; all remain
descriptive and do not establish causality.

## Artifacts

The final screen records compact JSON and CSV summaries under `data/`, the full
results and scientific boundaries in `REPORT.md`, and the V2.1 correction in
`V21_ERRATA.md`. Checkpoints, logs, raw embeddings, and node-level tensors stay
in the ignored `outputs/` tree and are not committed.
