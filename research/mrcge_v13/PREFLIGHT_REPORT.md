# MR-CGE V1.3 Preflight Report

## CPU and data preflight

The first CPU/data preflight passed for six dataset/seed groups: Movies,
Grocery, and ele-fashion with seeds 42 and 43. It verified the selected V1.2
same-seed reference rows, V7B.1 cache records, feature and physical graph
fingerprints, split fingerprints, Test-label masking, and initialization
compatibility audits. Each dataset reused the same split file and split
fingerprint for seeds 42 and 43.

The per-group record is in `data/preflight.json`; immutable dataset/cache/split
details are also in `data/frozen_baseline.json`. The CPU/data preflight was
rerun after the initial implementation freeze commit. A subsequent correction
to the gradient-audit harness changed the model source; the CPU/data preflight
will be refreshed again after that correction commit.

## GPU preflight

The first attempt used source commit `be5ec6b45d4ee7125d402b2751baa2169e546e50`.
All six `joint_view` cells passed the single-step checks. All six
`sender_private` cells failed in the diagnostic audit because the V1.3
gradient-capture helper requested nonexistent S/F-only parameters on the frozen
`M_repaired` backbone. No formal training started. The code has been corrected;
the first-attempt records remain in `data/preflight.json` and
`data/failures.json` and will not be counted as passes for the corrected source.

The corrected-source GPU preflight passed all 12 cells at source commit
`169cddd72aaaba00a5ce3fd25657981dd40c5e27`. Each cell completed one actual
forward/backward/optimizer step, produced finite gradients, passed the
evaluation RNG check, and passed the live GPU-1 guard. Each seed of the same
dataset/variant had the same measured peak:

| Dataset | Variant | Peak reserved | Required free memory (peak + 2 GiB) |
|---|---|---:|---:|
| Movies | `joint_view` | 2,586 MiB | 4,634 MiB |
| Movies | `sender_private` | 2,260 MiB | 4,308 MiB |
| Grocery | `joint_view` | 2,530 MiB | 4,578 MiB |
| Grocery | `sender_private` | 2,204 MiB | 4,252 MiB |
| ele-fashion | `joint_view` | 13,308 MiB | 15,356 MiB |
| ele-fashion | `sender_private` | 11,478 MiB | 13,526 MiB |

At the last fresh launch snapshot, GPU 1 was an RTX 3090 with 22,999 MiB free,
19% utilization, and 63°C. Existing compute PID 72408 (1,244 MiB) was recorded
as the shared-device baseline. No process was signaled or changed. The live
guard will take a new snapshot before each formal cell and monitor during each
child process.

The first partial attempt and its six audit-code failures remain recorded and
are not counted in the corrected 12/12 pass. The corrected-source GPU-preflight
data, per-attempt resource snapshots, and failures are retained under `data/`
and `outputs/mrcge_v13_screen/`.
