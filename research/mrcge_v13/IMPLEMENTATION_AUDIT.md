# MR-CGE V1.3 Implementation Audit

## Frozen starting point

- Base commit: `a7930442e85ff9450fded54d5869915b434c7431`
- Source branch: `exp/mrcge_v12_multiseed_confirmation`
- Work branch: `exp/mrcge_v13_view_cross_screen`
- Frozen V1.2 hashes:
  - `src/models/mrcge_v12.py`: `d9a7632f725a7c742dc083fe7737bc5c7384e9fa3bc91ffa64ead01e4a5d4ca6`
  - `configs/model/mrcge_v12.yaml`: `c0e5782fee74bda70a6cc42c3cf50b57e169ccf00981e53f595bd68b04750eed`
  - `src/tasks/nc.py`: `362350f07bb2d076b0f4f791872fbf0105ba695681bc15b3f2a09353bc6aacfd`
  - `scripts/run_mrcge_v12_confirmation.py`: `cbf6b6eff5f772f4e3a5d024454fa76294f27774cd667dccce5e8bccab59739e`
  - V1.2 report: `a6b1353c4a1d75bad116dbfb05fbbfc4e44cd422908778176aa01ba3365f3b4b`
  - V1.2 handoff: `2cd18afbcfcab4a15476c180b9057eab7049bdd1230814423f4d8c97d56eec8e`

These files remain unmodified on the V1.3 branch.

## Implementation surface

- `src/models/mrcge_v13.py` is a V1.2 model copy with the V1.3 additions. The
  only forward-path changes are the optional shared joint-view contribution,
  the optional sender-private mixture at Early Cross input, and scalar-only
  diagnostics. The original V1.2 model source is untouched.
- `configs/model/mrcge_v13.yaml` fixes `M_repaired`, explicit variant control,
  and the compatibility ratio default.
- `configs/study/mrcge_v13_screen.yaml` declares only the 12 authorized
  Dataset × Variant × Seed cells and the frozen training protocol.
- `scripts/run_mrcge_v13_screen.py` reuses read-only V1.2 data, masking,
  checkpoint, and audit utilities. It has a per-cell child process, a physical
  GPU-1 ownership guard, current free-memory checks, runtime PID monitoring,
  and separate resource/failure records.
- `tests/test_mrcge_v13.py` covers initialization and output compatibility,
  joint-view initial scales and both output gradients, paired-dropout zero-Cross
  behavior, sender gradient flow, support masks, strict checkpoint reload,
  exact study matrix, and validation diagnostic RNG behavior.

The new scalar diagnostics are retained as detached device scalars during a
forward and copied to CPU when epoch diagnostics are collected. This avoids
adding per-statistic synchronization to every training forward and does not
change model outputs or RNG consumption.

## Checks completed before source freeze

- `python -m py_compile` passed for the V1.3 model, runner, and test file.
- `conda run -n yhf_env python -m pytest -q tests/test_mrcge_v13.py tests/test_mrcge_v12.py tests/test_mrcge_v12_confirmation.py`
  passed: 34 tests.
- The runner CLI help works, including the guarded GPU-preflight worker entry.
- CPU/data preflight passed for all six dataset/seed groups. Initial parameter
  audits passed for all six groups; seed 42 and 43 share one frozen split per
  dataset; V1.2 same-seed cache/data/split fingerprints matched; Test labels
  were masked and Test evaluation was off.
- The CPU/data preflight above was run before the final diagnostics-only source
  adjustment. It will be rerun after the implementation freeze commit so the
  committed source hashes are authoritative.
- First GPU-preflight attempt found a V1.3 audit-harness bug in
  `capture_gradient_diagnostics`: the `sender_private` M_repaired path looked
  for S/F-only `single_filter` parameters. The forward/backward reached that
  audit call; no formal training began. The audit branch was corrected, and
  tests now exercise gradient auditing for all three explicit variants.
- After correction, the model/V1.2 regression suite passed: 37 tests. The
  failed first-attempt cells remain in `data/preflight.json` and
  `data/failures.json`; all GPU preflights will be repeated against the
  corrected source SHA.

## Pending audits

Real GPU-1 single-step forward/backward preflight against the corrected source,
measured peak memory plus the 2 GiB margin, the 12 formal Validation runs,
dual-checkpoint reload audits, same-seed result analysis, and final push are
pending. No formal training has started. GPU use remains conditional on a
fresh ownership guard pass.
