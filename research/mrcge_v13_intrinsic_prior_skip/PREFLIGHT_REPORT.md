# Preflight Report — Intrinsic-Prior Skip

Branch `exp/mrcge_v13_intrinsic_prior_skip_screen`; frozen source commit `c7f9209c60f18059d330ac23d7b3ab1b13af2e6e`; base `184deaaf12b21c9ee01d3328aef15d86bcc50346`.

## CPU checks

- Intrinsic-Prior Skip tests: 5 passed.
- Existing MR-CGE V1.1, V1.2, V1.3, and Two-Hop regressions: 73 passed.
- CPU data/cache/history preflight: `pass` for all five Stage 1 cells and both conditional Fashion cells.
- Initialization equivalence audited for 7 dataset/seed pairs.
- Test-off masking, historical references, and protected-file hashes are recorded in `data/preflight.json`.

## GPU 1 checks and formal execution

Only physical GPU 1 was used: RTX 3090, UUID `GPU-84aac143-953d-a98f-a8ff-8ebf88d076b6`. Earlier preflight attempts were blocked by status-query/resource guards; the exact frozen source campaign (#12, source commit `c7f9209`) passed all three dataset-specific preflights. Each pass used a fresh UUID/PID/memory/utilization/temperature snapshot, an actual dataset forward/backward, finite nonzero scalar gradient, optimizer membership, and the measured-peak plus 2 GiB start margin.

| Dataset | GPU preflight | Peak allocated MiB | Peak reserved MiB | Required free MiB |
|---|---|---:|---:|---:|
| Movies | pass | 2058.6 | 2390.0 | 8192 |
| Grocery | pass | 2111.9 | 2394.0 | 8192 |
| ele-fashion | pass | 11777.8 | 12244.0 | 14292 |

All five registered Stage 1 trainings completed serially on GPU 1 with the strict runtime guard active and no retries. Each run record confirms checkpoint reload, validation RNG preservation, nonzero prior-scalar gradient, optimizer management, matching frozen data/cache fingerprints, and `test_evaluation=False`. The preregistered Movies metric gate failed; Grocery passed. Stage 2 therefore was not authorized, and ele-fashion was not formally trained. No Test evaluation or Link Prediction was run. Per-run GPU resource snapshots and peaks are in `data/resource_summary.csv` and `data/runs.csv`.

## Development incident

An initial automated test invocation had `CUDA_VISIBLE_DEVICES` unset and attempted to place models on the default CUDA ordinal before failing on a missing import. No model forward or training worker ran, and no UUID was captured for that invocation. This is recorded as a development protocol incident; subsequent unit/regression tests explicitly hid CUDA, and all later GPU actions used the strict physical-GPU-1 guard.
