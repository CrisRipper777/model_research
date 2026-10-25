# Preflight Report

## Implementation verification

The focused implementation suite passed **11 tests**. The requested historical regression set passed **65 tests** with 21 deprecation warnings. Source-level tests verify exact original-switch initialization/evaluation/training equivalence, NoP0 coefficient and gradient behavior, training/evaluation path agreement, checkpoint reload, support invariants and the corrected Hybrid SameNode positive path.

## CPU and data preflight

Status: **passed** on implementation freeze `cb641c4947c13d2305cfe83110cfdf9e35d42517`. All seven cells loaded the same authorized frozen split as their V1.3-A reference; automatic split generation and Test evaluation were disabled, and Test labels were masked before Trainer entry. Movies and Grocery feature, graph, semantic-cache, candidate and V7B.1 fingerprints matched. Four actual-dimension original-switch checks (Movies/Grocery × seeds42/43) passed bitwise for model state and classifier state. Four V1.3-A reference rows and the corrected Hybrid H2 Movies seed42 attempt02 row passed local checkpoint SHA256 checks. The corrected Hybrid commit and metric row were verified. A SHA256 inventory of **115** protected historical research files is recorded for final comparison.

Movies split fingerprint: `2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d`. Grocery split fingerprint: `7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056`.

## GPU preflight

Status: **passed for all seven cells** on the implementation-frozen source. Each isolated physical-GPU-1 preflight returned finite embeddings/loss/gradients and an unchanged Validation RNG state. Measured reserved peaks and minimum launch free-memory thresholds were:

| Cells | Reserved peak | Required free before training |
|---|---:|---:|
| Hybrid-CGE Movies seeds43/44 | 4,384 MiB | 6,432 MiB |
| V1.3-A Movies seed44 | 2,694 MiB | 4,742 MiB |
| NoP0 Movies seeds42/43 | 2,694 MiB | 4,742 MiB |
| NoP0 Grocery seeds42/43 | 2,638 MiB | 4,686 MiB |

All thresholds also obey the 8,192 MiB minimum launch rule. Fresh GPU 1 launch snapshots had 22,741 MiB free, 42–44% utilization and 74–75°C, with existing compute PID 119988 treated as the observed baseline. During child checks, the guard allowed only the task's own worker PID in addition to the baseline; no new foreign PID was observed. GPU 0 was not used.

Both Hybrid H2 preflights proved positive SameNode support for all 16,672 Movies nodes in both modalities. Candidate RMS was 0.249–0.316 and SameNode source weight was 0.398–0.403. All four NoP0 preflights reported exact Physical order-zero coefficients of zero for Private and Collaborative paths; the shared theta received finite, nonzero one-step gradients.

## Formal Validation training

All seven authorized cells completed on physical GPU 1: **7 complete, 0 failed, 0 resource-blocked, 0 unstarted**. Each cell ran in its own process after a fresh ownership/resource guard. Runtime monitoring found no new foreign GPU compute PID. The pre-existing baseline PID was not interrupted. GPU 0 was not used.

Every run kept Test evaluation off and Test labels masked before Trainer entry. Validation RNG state was unchanged. Both Best-Accuracy and Best-F1 checkpoints were strictly reloaded; their recorded SHA256 values and recomputed Train/Validation metrics are in `data/runs.csv`. Best-Accuracy mechanism diagnostics and effective coefficient matrices are in `data/view_diagnostics.csv` and `data/coefficients_diagnostics.csv`. The protected-history audit found all 115 historical research files unchanged.

The final requested core regression command passed **65 tests**, with 21 PyTorch Geometric/checkpoint deprecation warnings:

```text
conda run -n yhf_env python -m pytest -q \
  tests/test_mrcge_v13_physical_basis_screen.py \
  tests/test_mrcge_v13.py \
  tests/test_hybrid_cge_paper_evidence_v1.py \
  tests/test_hybrid_cge_v11.py \
  tests/test_mrcge_v12.py
65 passed, 21 warnings
```

CPU/GPU preflight, ownership snapshots, attempt evidence, and resource summaries are in `data/preflight.json`, `data/gpu_preflight_attempts/`, `data/manifest.json`, and `data/resource_summary.csv`. Training outputs, checkpoints and full logs remain under the local `outputs/mrcge_v13_physical_basis/` directory and are not included in the research-data commit.
