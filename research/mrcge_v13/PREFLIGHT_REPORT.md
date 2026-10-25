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

The corrected-source GPU preflight has not started yet. It requires a fresh
GPU-1 ownership and free-memory guard. Each authorized cell must pass one real
forward/backward/optimizer step, finite-gradient checks, evaluation RNG checks,
and the measured peak-reserved-memory plus 2 GiB start margin before formal
training can begin.
