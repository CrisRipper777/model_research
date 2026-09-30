# P1.3 — Joint-Readout Operator Counterfactual

P1.3 tests whether one physical message can have different marginal validation utility across the fixed `smooth`, `absdiff`, and `product` propagation channels when all channels share one jointly trained linear readout. It reuses the P1.2 operator and edge-selection functions and keeps the P0 semantic encoder frozen.

## Run protocol

The run script expects the ignored P0 and P1.2 artifacts to be present under `outputs/`, including all nine semantic-only checkpoints and all nine P1.2 edge tables. It fails before training if a P1.3 edge row, order, or original target degree differs from P0 or P1.2.

```bash
conda run --no-capture-output -n yhf_env python scripts/run_p13_joint_readout_operator_probe.py --smoke --device cuda:0
conda run --no-capture-output -n yhf_env python scripts/run_p13_joint_readout_operator_probe.py --datasets Movies Grocery ele-fashion --seeds 42 43 44 --device cuda:0
conda run --no-capture-output -n yhf_env python scripts/analyze_p13_joint_readout_operator_probe.py
conda run --no-capture-output -n yhf_env python -m pytest -q tests/test_p13_joint_readout_operator_probe.py tests/test_p11_p12_operator_rescue.py
```

Each formal run creates one `Linear(1024, num_classes)` head over `[H_T, H_V, S_T, D_T, P_T, S_V, D_V, P_V]`. It uses AdamW (`lr=1e-3`, `weight_decay=1e-4`), at most 300 epochs, patience 30, minimum epoch 30, gradient clipping at 1.0, and best validation-accuracy checkpoint selection. Each aligned edge row stores one shared `ce_full`, six `ce_removed` values, and six raw/relative utilities. Only the corresponding context block is removed, divided by the original graph degree.

The validation labels select the head and also define the counterfactual cross-entropy. Treat the results as a controlled diagnostic, not independent confirmation. The scripts do not evaluate test labels, link prediction, or any adaptive propagation mechanism.

## Artifacts

- `report.md`: full results, interpretation, and limitations.
- `run_manifest.json`: source provenance, software/GPU environment, campaign checks, and figure QA.
- `data/`: committed seed-level and three-seed summary tables; `within_node_function_diversity.csv` also contains one row per eligible target, and the modality preference table stores each cell of the 4×4 contingency matrix in long form.
- `figures/`: committed PNG figures. PDF/SVG/600-dpi TIFF files and figure QA reports are under ignored `outputs/p13_joint_readout_operator_probe/figure_qa/`.
- `outputs/p13_joint_readout_operator_probe/`: ignored full edge tables and trained checkpoints; do not add these large artifacts to Git.
