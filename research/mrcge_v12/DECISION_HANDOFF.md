# MR-CGE V1.2 decision handoff

## Completed state

- Branch: `exp/mrcge_v12_anchor_performance_screen`
- Training implementation freeze: `9cfe6e491d2ea2a58de01101af5032199c37cf14`
- All 12 core cells and both eligible ele-fashion M bonus cells completed at seed 42.
- One Movies/M_repaired guard-aborted attempt was preserved; user-authorized resumption after a fresh passing guard completed that cell in attempt 2.
- Every completed checkpoint pair passed SHA verification. All 14 Validation RNG audits passed; Test evaluation remained false.
- No seeds 43/44, Test, Response Strength, or extra tuning were run.

## Descriptive next-candidate signal

M_anchor is the strongest candidate to prioritize in a later three-seed confirmation: its Best-Accuracy-checkpoint Macro-F1 exceeds M_repaired on all four datasets, and its Best Accuracy is higher on three of four. Toys Accuracy favors M_repaired, and H2/DiP historical values remain higher in Accuracy overall. Treat this as a seed-42 screen, not a final selection. Keep M_repaired as the paired comparator if the user chooses a multi-seed follow-up.

F_sinit versus S_repaired is mixed and small: accuracy rises only on Movies, while Macro-F1 rises only on Grocery. The current results do not support a clear F-versus-S selection.

## Source provenance

`data/manifest.json` records the code freeze SHA and preflight source-file hashes. The first two completed runs used the freeze commit directly. After the GPU guard stop, an audit-only commit (`b024ae8dbe94259611fadb0165df55a67db631b6`) was created before the user authorized resumption; the remaining 12 run rows record that actual HEAD. That commit modified only MR-CGE V1.2 audit data and reports. Model, trainer, launcher, and configuration file hashes remained identical, and no training math/configuration/data processing changed.

## Audit files

- Design and protocol: `DESIGN_FREEZE.md`
- Implementation, unit tests, and V1.1 checkpoint regression: `IMPLEMENTATION_AUDIT.md`
- CPU/GPU/resource preflight: `PREFLIGHT_REPORT.md`
- Full Validation results and historical context: `SCREENING_REPORT.md`
- Machine-readable raw data: `data/runs.csv`, `data/epoch_history.csv`, `data/filter_diagnostics.csv`, `data/route_diagnostics.csv`, `data/response_diagnostics.csv`, `data/resources.csv`, `data/preflight.json`, and `data/failures.json`

The requested V1.2 screen is complete. Stop here and leave any seed-43/44 study or later mechanism work for a separate user decision.
