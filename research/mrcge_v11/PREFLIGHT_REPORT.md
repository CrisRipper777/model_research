# MR-CGE V1.1 Preflight Report

- Branch: `exp/mrcge_v11_shared_filter_pilot`; code commit at report generation: `1d462d9f46a880b27bfd93477fca7dae15b2f5f6`.
- CPU/data/cache/init preflight: **pass**.
- GPU single-step preflight: **blocked**.
- Formal Pilot cells started: **0/9**.
- Test evaluation: **disabled**; Test labels are masked to `-100` before Trainer entry.

## CPU and data checks

- **Movies:** nodes=16672, edges=160802, features=768+768, Test masked=True, V7B source flags=[205881, 2].
  S/F/M common initialization hashes match: `58aac40ea22e15a05c71a4aff50596aef7a150c54827b3baa99658ed3ee9218f`; classifier hash: `5b006a851b183c23311cf249404b4b5083d891214fd8f15e01336ec353f3606c`.
- **Grocery:** nodes=17074, edges=142262, features=768+768, Test masked=True, V7B source flags=[205557, 2].
  S/F/M common initialization hashes match: `58aac40ea22e15a05c71a4aff50596aef7a150c54827b3baa99658ed3ee9218f`; classifier hash: `5b006a851b183c23311cf249404b4b5083d891214fd8f15e01336ec353f3606c`.
- **ele-fashion:** nodes=97766, edges=399172, features=512+512, Test masked=True, V7B source flags=[1102200, 2].
  S/F/M common initialization hashes match: `28d4889e8e7e272a8235d2f1581189134e6e95f47f238566f91ca1aa18e79c8f`; classifier hash: `9745fd034afee26f9e6e6f1bda69cee40d4d7db11a19a9386e4f1f577a65d046`.

## Focused tests

Command: `PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python -m pytest -q tests/test_mrcge_v11.py`
Result: passed; 23 passed, 1 warning in 3.62s.

```text
.......................                                                  [100%]
=============================== warnings summary ===============================
../../../../home/m3/miniconda3/envs/yhf_env/lib/python3.12/site-packages/torch_geometric/llm/utils/backend_utils.py:26
  /home/m3/miniconda3/envs/yhf_env/lib/python3.12/site-packages/torch_geometric/llm/utils/backend_utils.py:26: DeprecationWarning: `torch_geometric.distributed` has been deprecated since 2.7.0 and will no longer be maintained. For distributed training, refer to our tutorials on distributed training at https://pytorch-geometric.readthedocs.io/en/latest/tutorial/distributed.html or cuGraph examples at https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg/cugraph_pyg/examples
    from torch_geometric.distributed import (

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
23 passed, 1 warning in 3.62s
```

## GPU guard outcome

The guarded GPU single-step attempt for Movies/S/seed42 did not pass.
Recorded stage: `gpu_single_step_attempt_failed`; reason: `RuntimeError('GPU0 baseline utilization 100% is too high to start a shared run')`.
See `data/failures.json` and the referenced worker log for the full trace. Formal Pilot training remains gated until all nine GPU single-step checks pass.

## Preserved failures

`data/failures.json` contains 7 recorded events.
