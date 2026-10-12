# PACE-MAG V1 mechanism and optimization audit

## Scope

This branch audits the three selected E2 validation checkpoints from PACE-MAG V1. It reconstructs their original prediction path, measures source and message flow, and applies inference-only interventions to the frozen checkpoints. It does not edit the V1 model or task runner, run an optimizer, retrain any variant, tune hyperparameters, or calculate Test metrics.

The audit source branch is `exp/pace_mag_v1_mechanism_audit`, based directly on Pilot Report commit `d4094d3f003826d71a496495e27d8998f8f594ec`. The model definition is frozen at `739f340bb6858b7280dede1dfaaf37740c2de3fa`. The input checkpoint/config/cache paths and SHA-256 values are recorded in `data/audit_manifest.json` and `data/forward_parity.json` after the evidence run.

## Files

- `scripts/audit_pace_mag_v1.py` loads the existing checkpoint and matching saved config, verifies the V7B cache fingerprints, runs the strict forward parity check, and writes compact JSON/CSV evidence.
- `src/analysis/pace_mag_v1_audit.py` implements a separate audit forward by calling the existing V1 modules and sparse operators. Its gates are local arguments; it does not replace module methods, hooks, parameters, normalization buffers, or graph weights.
- `tests/test_pace_mag_v1_audit.py` covers parity, frozen checkpoint reload, cross-modal close/restore, route controls, independent branch gates, invalid rows, validation-label boundaries, cache mismatch, and the absence of optimizer calls.
- `AUDIT_DESIGN.md` defines the measurements and interpretation limits.
- Evidence and interpretation are added after the checks in `IMPLEMENTATION_CHECKS.md`, `MECHANISM_REPORT.md`, `OPTIMIZATION_REVIEW.md`, and `DECISION_MEMO.md`.

## Run

In the `yhf_env` environment, with at least 10 GiB free on the selected CUDA device:

```bash
python -m pytest -q tests/test_pace_mag_v1_audit.py
python scripts/audit_pace_mag_v1.py --device cuda:1
```

The CLI refuses to overwrite nonempty audit evidence and refuses to run from a branch/commit with the wrong source parent. It checks free memory before each dataset and does not stop other GPU processes. It stops immediately on checkpoint/cache mismatch or forward/metric parity failure, before interpreting interventions. Random node-route permutation uses the fixed seed `20261012`.

## Data and labels

Full-Graph transductive features and topology are used, matching the V1 prediction path. The repository NC loader materializes the complete label tensor and split indices; this is recorded explicitly. The audit metric code indexes train and validation labels only, mirroring the Pilot's validation Macro-F1 label set. It never indexes Test labels or computes a Test metric. Full node embeddings and logits are used in memory for comparison but are not written to the evidence directory.

## Evidence limits

Interventions answer how these selected frozen E2 checkpoints depend on a branch or route under the current trained weights. They do not predict the outcome of retraining without that branch. A single seed and node-level summaries do not support population-level significance. Optimization review is limited by the original logs and checkpoint cadence; see `OPTIMIZATION_REVIEW.md`.
