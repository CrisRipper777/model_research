# MR-CGE V1.3 Implementation Audit

## Frozen starting point and protected files

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

The four model/task/config/runner source files above match the frozen V1.2
hashes. The reports and result records in the V1.2 campaign were read-only.

## Implementation surface

- `src/models/mrcge_v13.py` is based on the frozen V1.2 model. Its forward-path
  additions are the optional shared Joint View Contribution module, the
  optional `0.3` source-private mixture at Early Cross input, and detached
  scalar diagnostics. `supports_link_prediction=False` remains explicit.
- `configs/model/mrcge_v13.yaml` uses `M_repaired`, explicit variant control,
  and ratio zero as the compatibility setting.
- `configs/study/mrcge_v13_screen.yaml` contains only the authorized 12 formal
  cells: three datasets × two independent variants × seeds 42/43.
- `scripts/run_mrcge_v13_screen.py` reuses read-only V1.2 data, masking,
  checkpoint, and audit utilities. It gives each cell an independent child
  process and Attempt directory, records per-attempt resource snapshots, and
  uses a physical GPU-1 ownership guard with a 2 GiB launch margin and runtime
  foreign-PID/free-memory checks.
- `scripts/analyze_mrcge_v13_screen.py` checks the unique-cell matrix, attempt
  history, frozen source hashes, source/config/data fingerprints, checkpoint
  hashes and metadata, dual-checkpoint metrics, and historical same-seed
  alignment. It also strictly reloads all 12 Best-Accuracy checkpoints on CPU
  and regenerates their Validation metrics and weighted response diagnostics.
- `tests/test_mrcge_v13.py` covers base/A/B-closed compatibility, A's neutral
  initialization, both A output gradients, paired-dropout zero-Cross
  behavior, B's source-private gradient, support masks and isolated nodes,
  strict checkpoint reload, exact study matrix, and Validation diagnostic RNG.

The Joint View first linear layer is evaluated as five sliced linear products
and summed; this is algebraically the specified concatenated projection and
does not materialize `[N, 5*256]`. The new A module is initialized within an
isolated CPU RNG context. The detached diagnostic scalars are copied to CPU at
epoch audit time, not individually synchronized on each training forward.

## Correctness and source audit

The final CPU regression command was:

```text
conda run -n yhf_env python -m pytest -q tests/test_mrcge_v13.py tests/test_mrcge_v12.py tests/test_mrcge_v12_confirmation.py
```

Result: **40 passed** (13 upstream/library warnings). `py_compile` passed for
the model, runner, analyzer, and test file. The corrected-source CPU/data
preflight verified all six dataset/seed groups, frozen split reuse, V7B.1
cache identity, same-seed historical references, Test-label masking, and
initialization audits.

The initial GPU-preflight audit exposed a harness bug: `sender_private` on the
`M_repaired` backbone attempted to inspect S/F-only `single_filter` parameters.
The first affected preflight attempts failed before formal training. The
gradient audit branch was corrected, tests were expanded to cover all three
explicit variants, and the corrected source passed all 12 GPU preflight
cells. The initial failures remain in `data/preflight.json` and
`data/failures.json`; they are not counted as successful preflights.

The corrected GPU-preflight model source is commit
`169cddd72aaaba00a5ce3fd25657981dd40c5e27`, model SHA256
`a70e2fd4345efe1f658075eb3cb034d85c8ad67454951cea7af02faf96775a37`. The
formal runs recorded source commit
`78284c82308b117e98f6c2ef59bf5acbe13369bf`; the V1.3 model SHA, V1.2 trainer
SHA, and run-specific config SHA are preserved in every successful `runs.csv`
row. After training, commit `1216376dd234679313c896c61ef3480848b584c8` added
diagnostics-only weighted Physical/Semantic contribution RMS calculations and
strict CPU reload auditing. The posthoc model SHA is
`be9272ad25296927fcb95720776a7fcdaf3f54d00936c96387262159b3770c90`; it does
not alter forward outputs or training behavior. Per-run provenance continues
to point at the source actually used by formal training.

## Completed execution audit

- All 12 unique authorized cells completed; there are no incomplete cells.
- Both Best-Accuracy and Best-F1 checkpoints were reloaded and compared with
  their selected Validation metrics by the formal runner. Their file hashes,
  epoch/selection metadata, protocol, provenance, and metric records passed
  the final analyzer audit (24 checkpoints total).
- A separate CPU posthoc pass strictly loaded all 12 Best-Accuracy
  checkpoints. Accuracy, Macro-F1, and CE matched the run rows within the
  configured tolerance, and each Validation pass preserved CPU RNG state.
- Train and Validation epoch history is retained for every successful run;
  every history row was checked for finite losses and metrics.
- Test labels were masked before trainer entry, `evaluate_test=False`, and no
  Test metrics were computed. No link-prediction run was performed.
- The audit found no changed frozen V1.2 source hash, mismatched cache/split
  fingerprint, duplicate successful cell, orphan epoch-history row, or
  checkpoint hash mismatch.

The formal training ledger contains 12 completed rows, 2 failed formal
Attempts, and 7 `blocked_resource` Attempts. The two failures were caused by
new external GPU compute PIDs (87602 and 89011); seven subsequent cells were
not launched after those guard trips. Only this task's own child was stopped.
No external process was signaled, paused, or changed. Each blocked/failed
Attempt is retained, and the corresponding cells later completed under fresh
passing GPU snapshots. Six initial preflight harness failures are separately
retained and excluded from the corrected 12/12 pass.

All values in this audit are Validation-only. Two-seed means are descriptive;
they do not establish significance, a strict causal module contribution,
SOTA, or that learned strengths correspond to true graph reliability. This
screen does not choose A or B as a final model.
