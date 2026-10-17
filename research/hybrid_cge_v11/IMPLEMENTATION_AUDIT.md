# Hybrid-CGE V1.1 Implementation Freeze Audit

## Frozen source and scope

The implementation adds `src/models/hybrid_cge_v11.py`, its model/study configs, a data-and-GPU preflight, the validation-only runner, analysis code and focused tests. The design and experiment plan are frozen in `DESIGN.md` and `EXPERIMENT_PLAN.md`. The implementation reuses the CG-CGE V1 graph normalization, V7B.1 cache checks, four shared experts, Top-2 routing, auxiliary-balance accounting, fusion and custom receiver-sparse attention. `src/tasks/nc.py` and historical study records were not modified.

The two conditions have identical state-dict structure and initialization. H1 masks SameNode and has no late-residual dropout; H2 exposes SameNode and applies node-wise non-inverted dropout (`p=0.15`) only to the final Late residual during training. No Role-FiLM parameters are registered. The physical cross message enters both collaborative expert profiles and the inherited node-router correction; no Physical Cross candidate is added to Late Completion. The Private states feed the 0.3 unimodal training CE.

## Engineering checks

- `conda run -n yhf_env python -m py_compile ...`: passed for all new Python source and test files.
- Focused tests: `12 passed` (`tests/test_hybrid_cge_v11.py` and `tests/test_hybrid_cge_v11_analysis.py`).
- Full repository tests: `564 passed, 24 warnings` in 270.33 s. Warnings are upstream PyG/checkpoint deprecations; no failures.
- `git diff --check`: clean at freeze staging.

## Dataset and initialization preflight

`data/preflight_cpu.json` confirms the official dataset/split and cache fingerprints for Movies, Grocery and ele-fashion at seeds 42/43/44. Each cell matches the historical A2 and B1 fingerprint/run records. H1 and H2 have matching model and classifier initialization hashes for every Dataset×Seed and four shared expert objects. Model parameter count is dataset-dependent because the private auxiliary heads scale with class count: 2,268,134 on Movies/Grocery and 2,001,878 on ele-fashion. Test labels are sentinel-masked in the preflight training data object.

## GPU 0 preflight

`data/preflight.json` records the actual CUDA environment and six one-step runs on physical GPU 0. At launch the RTX 3090 reported 112 MiB used and 0% utilization; the runner restricted `CUDA_VISIBLE_DEVICES=0`. GPU 1 was not used. All losses, gradients and post-update parameters were finite; all required early/late parameter paths had finite, nonzero gradients; Train/Validation evaluation preserved CPU and CUDA RNG; and state-dict save/load preserved validation logits within the preflight all-close tolerance. Peak allocated memory ranged from 4.2–4.3 GiB on Movies/Grocery to 20.1–20.2 GiB on ele-fashion (about 20.6 GiB including the round-trip evaluation peak), below 24 GiB capacity.

The observed maximum absolute logit difference between two repeated post-update evaluations across save/load was `2.86e-6`. The check passed its declared `atol=rtol=1e-6` all-close rule. This is a measured floating-point difference; this audit does not attribute it to any single CUDA operation. The formal runner independently verifies validation metrics after checkpoint reload and records failures without modifying historical data.

## Runtime settings and evidence limits

The preflight environment was Python 3.12.13, NumPy 2.4.3, PyTorch 2.4.0+cu121, CUDA build 12.1, cuDNN 9.1.0, NVIDIA driver 535.309.01, and NVIDIA GeForce RTX 3090. Matmul TF32 was disabled; cuDNN TF32 was enabled; deterministic algorithms were disabled; `CUBLAS_WORKSPACE_CONFIG` was unset. Formal training keeps these recorded defaults; no determinism setting is silently substituted.

One optimizer step validates the active paths and execution budget but does not establish full-run stability or model utility. The formal seed-42 integrity gate and all 18 validation runs remain required. Test metrics are disabled, and no Test labels enter model selection.
