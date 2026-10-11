# V7B.1 feasibility and GPU preflight

## Environment

- Python 3.12.13
- PyTorch 2.4.0+cu121; PyG 2.7.0; FAISS 1.15.1
- Target: NVIDIA GeForce RTX 3090, physical GPU1
- GPU1 was selected with `CUDA_VISIBLE_DEVICES=1` and exposed to PyTorch as logical `cuda:0`.
- GPU0 was occupied by an unrelated process and was not used or stopped.
- Full graph, hidden size 256, four physical hops, two semantic hops, four experts, Top-2 routing, and K=8 per modality were preserved.

## Preflight coverage

The final preflight passed all 12 dataset/model combinations: R0, corrected B0, B3, and B4 on each of Movies, Grocery, and ele-fashion. Each performed a full-graph training forward, NC loss, backward pass, clipped optimizer update, and full validation inference. B0/B3/B4 relation tensors had finite nonzero gradients and nonzero updates. R0 has no semantic relation path.

| Dataset | Arm | Peak allocated GiB | Training step s | Relation scoring s | Semantic propagation s | Full validation inference s |
|---|---|---:|---:|---:|---:|---:|
| Movies | R0 | 2.27 | 0.696 | 0.000 | 0.000 | 0.026 |
| Movies | B0 | 3.31 | 0.144 | 0.005 | 0.016 | 0.042 |
| Movies | B3 | 3.31 | 0.123 | 0.005 | 0.016 | 0.041 |
| Movies | B4 | 3.31 | 0.123 | 0.005 | 0.016 | 0.041 |
| Grocery | R0 | 2.14 | 0.065 | 0.000 | 0.000 | 0.020 |
| Grocery | B0 | 3.15 | 0.120 | 0.004 | 0.016 | 0.040 |
| Grocery | B3 | 3.16 | 0.122 | 0.005 | 0.016 | 0.040 |
| Grocery | B4 | 3.16 | 0.121 | 0.005 | 0.016 | 0.040 |
| ele-fashion | R0 | 8.76 | 0.218 | 0.000 | 0.000 | 0.077 |
| ele-fashion | B0 | 13.79 | 0.518 | 0.015 | 0.074 | 0.171 |
| ele-fashion | B3 | 13.80 | 0.524 | 0.017 | 0.074 | 0.173 |
| ele-fashion | B4 | 13.82 | 0.525 | 0.016 | 0.074 | 0.173 |

Timings are single-step/inference measurements from the preflight run and are not campaign averages. The largest allocated peak was approximately 13.82 GiB on ele-fashion B4, below the 24 GiB device capacity. The recorded reserved peaks and per-run timings are in `data/preflight_summary.json`.

## Smoke and tests

Six one-epoch Movies smoke runs passed. The R0 arm used V7A `a0_raw`, required no candidate cache, and emitted no Test metrics. B1/B2 retained absent relation gradients; corrected B0 and B3/B4 had nonzero relation gradients.

The full repository test suite passed: 385 tests, including six new corrective parity tests. See `data/test_summary.json` and `data/smoke_summary.json`.
