# MR-CGE V1.1 Stage-A Pilot Report

## Status

Planned: 9 Dataset × Condition × Seed cells (Movies/Grocery/ele-fashion × S/F/M × seed 42).
Completed: 0/9. **No formal training cell started.**
The CUDA 0 ownership/resource guard refused the GPU single-step preflight before model execution. Per protocol, formal Pilot training remained gated and no thresholds were bypassed.

## Validation metrics

There are no S/F/M Accuracy, Macro-F1, or CE Pilot values. `condition_summary.csv` and `condition_differences.csv` retain explicit `not_run` / `not_available` rows; no synthetic or historical score is substituted.

No conclusions about expert specialization, routing collapse, Early Cross activation, Late completion activation, or model performance can be drawn from the unrun Pilot. CPU unit tests establish code-path behavior only.

## Scope guard

No Test metric or model selection used Test labels. Test labels were masked to `-100` for CPU data checks. Seeds 43/44, the remaining 18 formal runs, full 27-run training, tuning and ablations were not started.

The next executable step is to repeat the guarded nine-cell GPU single-step preflight only when CUDA 0 has at least the registered free-memory headroom, utilization and temperature conditions. Independent review should inspect the source, the preflight records and the empty formal-run tables before authorizing any later phase.
