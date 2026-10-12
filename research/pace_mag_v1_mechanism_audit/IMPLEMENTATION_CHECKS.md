# Implementation and reproduction checks

## Source and checkpoint identity

The audit branch is `exp/pace_mag_v1_mechanism_audit`. Instrumentation commit `0a0762aac2c4cf25795ebc0ebf8015057f970ddb` is based directly on Pilot Report commit `d4094d3f003826d71a496495e27d8998f8f594ec`. The audited model source is frozen at `739f340bb6858b7280dede1dfaaf37740c2de3fa`. No V1 model, task runner, pilot CSV, configuration, or checkpoint was changed.

| Dataset | Nodes × classes | Best checkpoint SHA-256 | Resolved config SHA-256 | Pilot Validation Accuracy / Macro-F1 |
|---|---:|---|---|---:|
| Movies | 16,672 × 20 | `39374c272be126d2093243d5fb8bf8e7c19fe12fccbcfd4781aa03dedd0f0e3c` | `392a7a97b195b812146fde26488315bb0f2f556040126822d991e050ff36120c` | 0.54229152 / 0.45272549 |
| Grocery | 17,074 × 20 | `2d302ebb45984809a9e612d865292b9eb268fcbe8b2f7e372f40d65f429e1f1e` | `c4d14a1ddfc5a6e956d74478869b61917d2d37165ea548702c29b3fecd89ffbc` | 0.81786239 / 0.73513365 |
| ele-fashion | 97,766 × 12 | `bb406d0e574b3d8f3ff177f49aeeaac4fd8e6327dcda4a57aa175a3541505a74` | `fc593c03c863bb9275e931aba2cc1b668e1eb07309c7dc17912efbfbe7e5c574` | 0.87644470 / 0.74724352 |

Checkpoint metadata, dataset shape, saved configuration, disabled-Test setting, and all semantic-cache fingerprints matched the frozen Pilot records. Full paths and hashes are in `data/forward_parity.json`; cache fingerprints are in `data/audit_manifest.json`.

## Forward parity and state preservation

The audit loaded each best E2 model and classifier in evaluation mode, used `torch.inference_mode()`, and compared an audit-only forward against the original V1 forward before running interventions.

| Dataset | Max absolute embedding difference | Max absolute logit difference | Validation Acc/F1 difference from Pilot | Passed |
|---|---:|---:|---:|---|
| Movies | 1.19e-6 | 1.91e-6 | 0 / 0 | yes |
| Grocery | 1.36e-6 | 2.86e-6 | 0 / 0 | yes |
| ele-fashion | 1.43e-6 | 2.86e-6 | 0 / 0 | yes |

All differences were below the recorded `1e-5` tolerance. The model and classifier state hashes matched before and after each dataset audit; both modules remained in evaluation mode. After every intervention, an original forward restored embeddings and logits with a maximum absolute difference no greater than `3.82e-6`.

The frozen route and branch interventions never created or called an optimizer. No training was performed. `data/failures.json` is empty. The final manifest records the audit commit, checkpoint hashes, environment, device, fixed route-permutation seed (`20261012`), and the explicit `training_performed`, `optimizer_created_or_called`, and `test_metrics_computed` flags as false.

## Labels and saved artifacts

The repository NC loader materializes the full label tensor and split indices in host memory. The audit records this loader behavior. Metric code indexes only training and validation indices, following the Pilot's train-plus-validation class set; it does not read `test_idx`, index Test label values, or calculate Test metrics. All node features and graph topology are used as in the transductive V1 forward.

The audit writes compact summaries and Validation per-class metrics. It does not save node embeddings, logits, raw labels, checkpoints, or raw input data. The selected CUDA device was GPU 1, an NVIDIA GeForce RTX 3090; the runtime environment and memory snapshots are in `data/audit_manifest.json`.

## Tests

The PACE-MAG V1 and V7 regression batch passed: **90 passed, 14 upstream/deprecation warnings**. This included the original V1 tests, the audit suite, and V7a/V7b/V7b.1 corrective suites. The added audit tests cover forward parity, checkpoint reload, sparse source reconstruction, context-row construction, intervention metric deltas, bounded-parameter and Router summaries, frozen cross-modal close/restore, uniform/view-mean/node-permuted routing, independent physical/semantic/skip gates, invalid-source rows, train/validation-only class-set construction, cache mismatch rejection, and a forbidden optimizer step.

## Remaining reliability boundaries

This is a single seed and a frozen-checkpoint analysis. Node-level distributions are descriptive, not independent experimental replicates. Frozen interventions show dependence of the trained E2 checkpoint under the specified forward change; they do not estimate the result of retraining without that component.
