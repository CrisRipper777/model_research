# MR-CGE V1.3 Two-Hop Cross — Preflight Report

## Status

- CPU preflight: **PASS**. Fixed-split, feature, graph, semantic-cache, candidate and V7B.1 fingerprints matched the historical V1.3-A records. Both arms preserved common model and classifier initialization across 14 planned dataset/seed/arm combinations. Test labels were masked before Trainer entry. The protected 2,096 historical file hashes still match.
- Model tests: **15 passed** in the recorded implementation check (`tests/test_mrcge_v13_two_hop_cross.py`).
- GPU preflight: **PASS, 6/6 cells** on physical GPU 1 (NVIDIA GeForce RTX 3090), in campaign 9. The user authorized shared-GPU execution after sandbox CUDA visibility and the ownership guard prevented progress. The override skips PID ownership, utilization and temperature checks. CUDA remains pinned to physical GPU 1; each worker must see one CUDA device and satisfy runtime free-memory thresholds (8 GiB at formal training start, 2 GiB during execution). The preflight also required the measured peak reservation plus a 2 GiB margin.
- Stage 1: **complete, 6/6 Movies runs** across raw and complementary arms and seeds 42, 43 and 44. Checkpoint reload verification passed for every run; Validation RNG was unchanged. Test evaluation and link prediction were disabled.
- Stage 2: **not authorized**. Neither Stage 1 arm passed the registered promotion gates.

## GPU preflight history

Campaigns 1–6 recorded no passing cells while CUDA/NVIDIA driver status was unavailable or intermittent from the execution environment. Campaigns 7 and 8 passed 1/6 and 4/6 cells before fresh ownership/resource checks stopped their remaining cells. Campaign 9 used the user-authorized shared-GPU mode and passed the complete registered six-cell matrix. Earlier records remain in `data/gpu_preflight.json` and `data/failures.json`.

| Dataset | Arm | Seed | Peak reserved (MiB) | Required preflight free memory (MiB) | Result |
|---|---|---:|---:|---:|---|
| Movies | raw | 42 | 2,458 | 4,506 | PASS |
| Movies | complementary | 42 | 2,544 | 4,592 | PASS |
| Grocery | raw | 42 | 2,462 | 4,510 | PASS |
| Grocery | complementary | 42 | 2,548 | 4,596 | PASS |
| ele-fashion | raw | 42 | 12,232 | 14,280 | PASS |
| ele-fashion | complementary | 42 | 12,618 | 14,666 | PASS |

The free-memory figures are the measured preflight peak reservation plus the required 2 GiB margin. Formal Stage 1 launches also enforced the 8 GiB minimum; active workers enforced the 2 GiB runtime floor. No unrelated process was stopped or modified.

## Promotion decision

The registered paired Validation screen failed for both arms. Raw had mean ΔAccuracy −0.1400 percentage points and 1/3 positive seeds; complementary had −0.1500 points and 1/3 positive seeds. Consequently, neither arm was selected and Stage 2 did not run. No Test metric or link-prediction result was collected.

Artifacts: `data/preflight.json`, `data/gpu_preflight.json`, `data/gpu_resources.csv`, `data/failures.json`, `data/runs.csv`, `data/stage1_screening_decision.json`, and `data/manifest.json`.
