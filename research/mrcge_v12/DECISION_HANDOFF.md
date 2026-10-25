# MR-CGE V1.2 decision handoff

## Current checkpoint

- Branch: `exp/mrcge_v12_anchor_performance_screen`
- Frozen implementation SHA: `9cfe6e491d2ea2a58de01101af5032199c37cf14`
- Final local screening state: Stage 3 halted after the GPU 1 ownership guard detected new external PID `4030361` during Movies/M_repaired.
- Completed: Movies/S_repaired and Movies/F_sinit, seed 42.
- Failed guarded attempt: Movies/M_repaired attempt 1. No epoch or checkpoint was produced.
- Blocked before start: the other nine core cells and both ele-fashion bonus cells. Their status rows and reasons are in `data/runs.csv` and `data/failures.json`.

## Decision boundary

The partial results are insufficient to choose S, F, M_repaired, or M_anchor for follow-up. F_sinit's small descriptive Movies difference is not a selection result. No formal M comparison, multi-dataset comparison, three-seed validation, Test result, or H2/DiP comparison was produced.

The attached V1.2 task says to stop the affected stage when the GPU ownership guard fails. Do not restart Stage 3 or launch the Fashion bonus cells until a fresh user instruction authorizes resumption and a fresh GPU 1 baseline/ownership/memory guard passes. Do not kill or signal the observed process. On a permitted resume, retain the same frozen source SHA; the completed Movies S/F cells can be reused after checkpoint hash verification, and new attempts must use new numbered attempt directories.

## Audit files

- Design and frozen protocol: `DESIGN_FREEZE.md`
- Code/test/checkpoint audit: `IMPLEMENTATION_AUDIT.md`
- CPU/GPU and memory preflight: `PREFLIGHT_REPORT.md`
- Partial scores and stage stop: `SCREENING_REPORT.md`
- Full machine-readable history: `data/runs.csv`, `data/epoch_history.csv`, `data/filter_diagnostics.csv`, `data/route_diagnostics.csv`, `data/response_diagnostics.csv`, `data/resources.csv`, `data/preflight.json`, and `data/failures.json`

No Test evaluation was performed; V1.1 history was left intact; no other GPU process was affected.
