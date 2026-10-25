# MR-CGE V1.3 Preflight and Resource Report

## CPU/data and initialization preflight

CPU/data preflight passed for Movies, Grocery, and ele-fashion at seeds 42 and
43. It verified the selected V1.2 same-seed `M_repaired` rows, the V7B.1
Text/Visual K=8 semantic cache, feature and physical graph fingerprints, split
fingerprints, model input dimensions, label universe, and Test masking. The
seed changes did not create a new split: each dataset uses one frozen split
fingerprint for both seeds. `data/frozen_baseline.json` and
`data/preflight.json` hold the per-dataset evidence.

The first CPU/GPU audit pass also found a V1.3 gradient-capture harness bug for
`sender_private` on the M backbone. Six initial sender-private GPU preflight
attempts failed while the audit helper asked for a nonexistent S/F-only
parameter; no formal training was started from those checks. The helper was
corrected and regression coverage added. The corrected implementation then
passed the full 12-cell GPU preflight, one real forward/backward/optimizer step
per Dataset × Variant × Seed, with finite gradients and unchanged Validation
RNG. First-attempt failures remain in `data/preflight.json` and
`data/failures.json`, separately from the corrected pass rows.

## Corrected-source GPU preflight

Preflight used physical GPU 1, NVIDIA GeForce RTX 3090. The peak reserved
memory measured for each dataset/variant determined its start threshold,
including the required additional 2 GiB:

| Dataset | Variant | Peak reserved | Required free memory |
|---|---|---:|---:|
| Movies | `joint_view` | 2,586 MiB | 4,634 MiB |
| Movies | `sender_private` | 2,260 MiB | 4,308 MiB |
| Grocery | `joint_view` | 2,530 MiB | 4,578 MiB |
| Grocery | `sender_private` | 2,204 MiB | 4,252 MiB |
| ele-fashion | `joint_view` | 13,308 MiB | 15,356 MiB |
| ele-fashion | `sender_private` | 11,478 MiB | 13,526 MiB |

At the last fresh pre-formal launch snapshot, GPU 1 had 22,999 MiB free,
19% utilization, and 63°C. Existing compute PID 72408 (1,244 MiB) was recorded
as the shared-device baseline. Formal attempts repeated the guard and free
memory assessment immediately before launch and monitored the device while
running. The highest formal reserved peak was 13,620 MiB (ele-fashion,
`joint_view`). Every formal attempt passed a fresh launch guard against the
measured preflight peak plus 2 GiB before starting; the formal peak and all
per-attempt samples are recorded in `data/runs.csv` and
`data/resource_summary.csv`.

## Guard interruptions and completed retries

Two formal Attempts were stopped when the live guard detected newly appearing
external compute PIDs: Grocery/`sender_private`/seed43 attempt 1 detected PID
87602; ele-fashion/`joint_view`/seed42 attempt 2 detected PID 89011. Seven
later starts were recorded as `blocked_resource` and not launched while those
guard snapshots were unresolved. A fresh GPU snapshot established the new
external PID state before work resumed. Only the V1.3 child process was
stopped; no external PID was signaled, paused, or modified. Retries succeeded,
so all 12 authorized unique cells are now complete. The raw attempt ledger,
snapshots, reasons, and timestamps remain under `data/` and the local
`outputs/mrcge_v13_screen/` tree.

The formal attempt ledger therefore has 12 successful rows, 2 failed formal
Attempts, and 7 blocked starts. The 7 blocked starts did not leave blocked
cells. No guard bypass, threshold change, or external-process intervention was
used. No Test evaluation or link-prediction work occurred.

## Records

- Per-step corrected and initial GPU preflight: `data/preflight.json`
- Frozen data/cache/split evidence: `data/frozen_baseline.json`
- Per-attempt GPU samples: `data/resource_summary.csv`
- Per-cell resource summary after analysis: `data/resource_by_cell.csv`
- All failed and blocked reasons: `data/failures.json`
- Final cell/attempt status counts: `data/manifest.json`
