# MR-CGE V1.3 Preflight Report

## CPU and data preflight

The first CPU/data preflight passed for six dataset/seed groups: Movies,
Grocery, and ele-fashion with seeds 42 and 43. It verified the selected V1.2
same-seed reference rows, V7B.1 cache records, feature and physical graph
fingerprints, split fingerprints, Test-label masking, and initialization
compatibility audits. Each dataset reused the same split file and split
fingerprint for seeds 42 and 43.

The per-group record is in `data/preflight.json`; immutable dataset/cache/split
details are also in `data/frozen_baseline.json`. This first pass preceded the
final diagnostics-only source adjustment. It will be refreshed after the
implementation freeze commit.

## GPU preflight

Not started. It requires a fresh GPU-1 ownership and free-memory guard after
the committed CPU/data preflight. Each authorized cell must pass one real
forward/backward/optimizer step, finite-gradient checks, evaluation RNG checks,
and the measured peak-reserved-memory plus 2 GiB start margin before formal
training can begin.
