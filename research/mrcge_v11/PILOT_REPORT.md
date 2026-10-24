# MR-CGE V1.1 Stage-A Pilot Report

## Status

Planned: 9 Dataset × Condition × Seed cells (Movies/Grocery/ele-fashion × S/F/M × seed 42).
Completed: 0/9; formal cells with a recorded attempt: 0.
The guarded GPU single-step gate did not pass; no threshold was bypassed.

## Validation metrics

There are no S/F/M Accuracy, Macro-F1, or CE Pilot values. `condition_summary.csv` and `condition_differences.csv` retain explicit `not_run` / `not_available` rows; no synthetic or historical score is substituted.

No conclusions about expert specialization, routing collapse, Early Cross activation, Late completion activation, or model performance can be drawn from an unrun Pilot. CPU unit tests establish code-path behavior only.

## Scope guard

No Test metric or model selection used Test labels. Test labels were masked to `-100` before Trainer entry. Seeds 43/44, the remaining 18 formal runs, full 27-run training, tuning and ablations were not started.

Repeat the guarded GPU preflight only after resolving the recorded failure; preserve all prior attempts.
