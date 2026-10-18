# Validation-only model selection

## Predeclared rule

Validation Accuracy is primary. Macro-F1, CE, per-class metrics, Train–Validation gaps and stability are secondary. Comparisons are paired by Dataset × Seed. Fashion candidates remain explicitly dataset-specific. Historical Test results or Test labels do not select a model.

## Selection

Pending complete ablation, hop-depth, baseline and Fashion matrices. The proposed lock will remain `pending_review`; no Test run is allowed from this branch until an independent review changes the lock to approved and the user separately authorizes `--final-test`.
