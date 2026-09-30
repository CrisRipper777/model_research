# M0 — Adaptive Propagation Mechanism Screen

This stage compares five controlled one-step NC variants on Movies, Grocery and ele-fashion with seeds 42–44. All variants use the same end-to-end Text/Visual node projection and prior-retaining fusion. M0-A controls propagation extent, M0-B adds one learned correction direction, and M0-C uses four continuous basis coefficients with the same correction-factor parameter count as M0-B.

## Frozen protocol

- Source: `exp/p13_joint_readout_operator_probe` at `83487eada5ca4909a83140a3de498514f2d6d7d8`.
- Task protocol: `unified_full_graph_nc_v1`; AdamW, 300 epochs maximum, validation-accuracy selection, patience 30, minimum epoch 30, gradient clip 1.0.
- Seeds: 42, 43, 44. No per-dataset tuning, test evaluation or LP.
- Each message uses the original directed physical support after removing self loops. Aggregation always divides by the original non-self incoming degree.
- Edge relation computations are chunked at 100,000 messages. Only the leave-one-out context scalar is detached.
- The M0 loader requests train and validation split fields only, keeps only those labels, and does not attach test indices to the training data object.
- Large checkpoints and per-run JSON records live under ignored `outputs/m0_adaptive_propagation_screen/`.

## Commands

```bash
conda run --no-capture-output -n yhf_env python scripts/run_m0_adaptive_propagation.py --smoke --device cuda:0
conda run --no-capture-output -n yhf_env python scripts/run_m0_adaptive_propagation.py --campaign --device cuda:0
# Resume remaining unique records on GPU 1 if GPU 0 is occupied.
conda run --no-capture-output -n yhf_env python scripts/run_m0_adaptive_propagation.py --campaign --device cuda:1
conda run --no-capture-output -n yhf_env python scripts/analyze_m0_adaptive_propagation.py
conda run --no-capture-output -n yhf_env python -m pytest -q tests/test_m0_adaptive_propagation.py
```

The formal campaign completed 40 unique runs on GPU 0 and resumed the final 5 on GPU 1 after GPU 0 was occupied. One initial Movies/42/SEM attempt failed in the diagnostic summarizer; the empty-control edge case was fixed and that run completed on resume. All 45 final unique runs and validation-only intervention analyses are complete.
