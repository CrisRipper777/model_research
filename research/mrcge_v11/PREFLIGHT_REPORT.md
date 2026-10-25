# MR-CGE V1.1 Preflight Report

- Branch: `exp/mrcge_v11_shared_filter_pilot`; code commit at report generation: `df4cf5a2a91c0bbda7c3765d987cc20e8a685237`.
- CPU/data/cache/init preflight: **pass**.
- GPU single-step preflight: **pass**.
- Formal Pilot cells attempted: **9/9**; completed: **8/9**.
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
Result: passed; 23 passed, 7 warnings in 5.53s.

```text
.......................                                                  [100%]
=============================== warnings summary ===============================
../../../../home/m3/miniconda3/envs/yhf_env/lib/python3.12/site-packages/torch_geometric/llm/utils/backend_utils.py:26
  /home/m3/miniconda3/envs/yhf_env/lib/python3.12/site-packages/torch_geometric/llm/utils/backend_utils.py:26: DeprecationWarning: `torch_geometric.distributed` has been deprecated since 2.7.0 and will no longer be maintained. For distributed training, refer to our tutorials on distributed training at https://pytorch-geometric.readthedocs.io/en/latest/tutorial/distributed.html or cuGraph examples at https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg/cugraph_pyg/examples
    from torch_geometric.distributed import (

tests/test_mrcge_v11.py::test_order_truncation_signed_coefficients_and_gradients
tests/test_mrcge_v11.py::test_trainer_loss_uses_main_private_and_balance_terms
tests/test_mrcge_v11.py::test_cpu_forward_backward_checkpoint_reload_and_rng_audit
tests/test_mrcge_v11.py::test_filter_router_and_cross_parameters_receive_gradients
tests/test_mrcge_v11.py::test_single_filter_gradient_diagnostics_match_trainable_parameters[S]
tests/test_mrcge_v11.py::test_single_filter_gradient_diagnostics_match_trainable_parameters[F]
  /home/m3/miniconda3/envs/yhf_env/lib/python3.12/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
    with device_autocast_ctx, torch.cpu.amp.autocast(**cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
23 passed, 7 warnings in 5.53s
```

## GPU guard outcome

All nine guarded GPU single-step checks passed on physical GPU 1; maximum measured allocated memory was 21173 MiB. Earlier blocked attempts remain in the audit history.
During formal Pilot ele-fashion/M/seed42 attempt 1, the runtime guard stopped only its own training child because `RuntimeError('GPU1 free memory 160 MiB below 1536 MiB guard threshold')`. See the matching `shared_gpu_runtime_guard` failure and `pilot_runtime_guard_blocked` resource row before resuming.

## Preserved failures

`data/failures.json` contains 15 recorded events.
GPU guard checks were explicitly overridden for formal ele-fashion/M/seed42 attempt 2; see `data/guard_overrides.json`.
GPU guard checks were explicitly overridden for formal ele-fashion/M/seed42 attempt 3; see `data/guard_overrides.json`.
Attempt 3 allocator environment: `{'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}`.
