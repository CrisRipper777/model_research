# MR-CGE V1.3 Two-Hop Cross — Preflight Report

## Status

- CPU preflight: **PASS**. Fixed-split, feature, graph, semantic-cache, candidate and V7B.1 fingerprints matched the historical V1.3-A records; both arms preserved common model and classifier initialization across 14 planned dataset/seed/arm combinations. Test labels were masked before Trainer entry. Historical checkpoint hashes and 2,096 protected historical file hashes were verified.
- Model tests: **15 passed** (`tests/test_mrcge_v13_two_hop_cross.py`).
- GPU preflight: **BLOCKED**. Five fresh GPU 1 preflight campaign attempts recorded zero passing cells. In each attempt the parent failed to obtain the fresh GPU snapshot from `nvidia-smi` (exit status 9 / NVIDIA driver communication error) before launching a worker. The first cell was Movies/Raw; the other five cells were not launched under the fail-closed guard.
- Formal training: **not started**. Stage 1 and Stage 2 were not run; no promotion gate was evaluated. No Test metric or link prediction was run.

## Registered GPU preflight cells

| Dataset | Arm | Seed | Result |
|---|---|---:|---|
| Movies | raw | 42 | Blocked before child launch in all five campaigns |
| Movies | complementary | 42 | Blocked before child launch in all five campaigns |
| Grocery | raw | 42 | Blocked before child launch in all five campaigns |
| Grocery | complementary | 42 | Blocked before child launch in all five campaigns |
| ele-fashion | raw | 42 | Blocked before child launch in all five campaigns |
| ele-fashion | complementary | 42 | Blocked before child launch in all five campaigns |

## Recorded failure behavior

For each campaign, the parent attempted to capture physical GPU 1 compute PIDs and device state. The required `nvidia-smi` query returned nonzero status 9, so the first cell was marked `blocked_resource` and the remaining cells were recorded as not launched. No process was killed; no GPU was selected other than physical GPU 1. Standalone status queries were intermittent, so a successful isolated read was not treated as sufficient evidence for launching the controlled preflight campaign.

Artifacts: `data/preflight.json`, `data/gpu_preflight.json`, `data/failures.json`, and `data/manifest.json`.
