# Preflight Report

- CPU/data status: **pass** across 9 dataset×seed groups.
- Initialization audits: 9/9 passed.
- CPU attempt history:
- Attempt 1: **pass** — completed
- Attempt 2: **pass** — completed
- GPU preflight status counts: {'pass': 36, 'blocked_resource': 0, 'failed': 0}.
- Test evaluation: `False`.
- Source hashes at CPU preflight: `{"data_types": "531968f22725acc211d5906c66bf0189d35857cbbdb972d12ee4ebdd29d3d2f8", "dataset_config:Grocery": "18969c97a0f06ab33fd3094624d524f03de48139405159bb06d19cf107105e56", "dataset_config:Movies": "81b803b0ed79ad86d483aee5dceadac149ca9ba2263e31160b22951a7812f5c8", "dataset_config:Toys": "d27227abdd6a9f4f4b8138e506deaadbf613b52d50cf5847fe9d8f3714856a73", "dataset_config:ele-fashion": "50e683dcae43c505c999ecbc22a6b5b380a8b897d4942933c01f266f8169b917", "dataset_loader": "08fe47584069a8f690d87d20a2b892d008c43a1dad6bdb3d08bb2f7e22ddd173", "graph_utils": "8fa1c35c1897e25f2f30628ddc0312598fe9e581692a48cccb131a12a50f10df", "inference": "f6e0df653a03a6ab5be98016fdc495fcff7704f42a73bdd546c36f7fc63be0aa", "launcher": "cbf6b6eff5f772f4e3a5d024454fa76294f27774cd667dccce5e8bccab59739e", "model": "d9a7632f725a7c742dc083fe7737bc5c7384e9fa3bc91ffa64ead01e4a5d4ca6", "model_config": "c0e5782fee74bda70a6cc42c3cf50b57e169ccf00981e53f595bd68b04750eed", "nc_config": "c8df71f58e707b4c7df429acd62a4c1d91e9cfd80ba24bc6c6d1979307224849", "seed_utils": "3e423b11a7e59e9b51c48e176d08b05b8fbd6294f364e1895ed2f1413d2e02f4", "semantic_cache_loader": "d79f0f8bb72448701bb490cea57b7095a5b8e99ee2bf9b63fb552038977d9caf", "split_loader": "b352a1c558752cc82075e7405723d42a408cb416386ad412669f0339aacd1d22", "study_config": "4086b33fde0ca653676199382026b4936f9132b9da599e47a953e991068cf7b9", "trainer": "362350f07bb2d076b0f4f791872fbf0105ba695681bc15b3f2a09353bc6aacfd"}`.

## GPU single-step records

- `Movies/M_repaired/seed43`: **pass**; single-step preflight record present
- `Movies/M_repaired/seed43`: **pass**; single-step preflight record present
- `Movies/M_repaired/seed44`: **pass**; single-step preflight record present
- `Movies/M_repaired/seed44`: **pass**; single-step preflight record present
- `Movies/M_anchor/seed43`: **pass**; single-step preflight record present
- `Movies/M_anchor/seed43`: **pass**; single-step preflight record present
- `Movies/M_anchor/seed44`: **pass**; single-step preflight record present
- `Movies/M_anchor/seed44`: **pass**; single-step preflight record present
- `Grocery/M_repaired/seed43`: **pass**; single-step preflight record present
- `Grocery/M_repaired/seed43`: **pass**; single-step preflight record present
- `Grocery/M_repaired/seed44`: **pass**; single-step preflight record present
- `Grocery/M_repaired/seed44`: **pass**; single-step preflight record present
- `Grocery/M_anchor/seed43`: **pass**; single-step preflight record present
- `Grocery/M_anchor/seed43`: **pass**; single-step preflight record present
- `Grocery/M_anchor/seed44`: **pass**; single-step preflight record present
- `Grocery/M_anchor/seed44`: **pass**; single-step preflight record present
- `Toys/M_repaired/seed43`: **pass**; single-step preflight record present
- `Toys/M_repaired/seed43`: **pass**; single-step preflight record present
- `Toys/M_repaired/seed44`: **pass**; single-step preflight record present
- `Toys/M_repaired/seed44`: **pass**; single-step preflight record present
- `Toys/M_anchor/seed43`: **pass**; single-step preflight record present
- `Toys/M_anchor/seed43`: **pass**; single-step preflight record present
- `Toys/M_anchor/seed44`: **pass**; single-step preflight record present
- `Toys/M_anchor/seed44`: **pass**; single-step preflight record present
- `ele-fashion/M_repaired/seed43`: **pass**; single-step preflight record present
- `ele-fashion/M_repaired/seed43`: **pass**; single-step preflight record present
- `ele-fashion/M_repaired/seed44`: **pass**; single-step preflight record present
- `ele-fashion/M_repaired/seed44`: **pass**; single-step preflight record present
- `ele-fashion/M_anchor/seed43`: **pass**; single-step preflight record present
- `ele-fashion/M_anchor/seed43`: **pass**; single-step preflight record present
- `ele-fashion/M_anchor/seed44`: **pass**; single-step preflight record present
- `ele-fashion/M_anchor/seed44`: **pass**; single-step preflight record present
- `ele-fashion/S_repaired/seed42`: **pass**; single-step preflight record present
- `ele-fashion/S_repaired/seed42`: **pass**; single-step preflight record present
- `ele-fashion/F_sinit/seed42`: **pass**; single-step preflight record present
- `ele-fashion/F_sinit/seed42`: **pass**; single-step preflight record present

The GPU baseline snapshot is in `data/gpu_baseline.json`; per-phase snapshots and required free-memory thresholds are in `data/resources.csv`. A resource block leaves the associated formal cell incomplete and does not authorize use of another GPU or affect other processes.
