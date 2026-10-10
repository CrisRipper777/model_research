# MACE-MAG V6A experiment

The model and four frozen variants are described in [DESIGN.md](DESIGN.md). The formal protocol is `unified_full_graph_nc_v1`: Movies, Grocery, ele-fashion; seeds 42/43/44; A0–A3; 36 independent full-graph NC runs. Validation Accuracy selects checkpoints and Validation Macro-F1 is reported. Every runner command explicitly sets `task.evaluate_test=false`; do not add test evaluation.

Run from the repository root in `yhf_env`:

```bash
python scripts/run_mace_mag_v6a.py --mode preflight --device cuda:1
python scripts/run_mace_mag_v6a.py --mode smoke --device cuda:1
# Before freezing the implementation:
python -m pytest -q
# After full pytest and the freeze commit:
python scripts/run_mace_mag_v6a.py --mode campaign --device cuda:1
```

Formal runs are serial. To continue an interrupted campaign, keep `HEAD` at the recorded freeze commit and run:

```bash
python scripts/run_mace_mag_v6a.py --mode campaign --device cuda:1 --resume
```

The runner checks branch ancestry, freeze SHA, the configuration fingerprint, the preflight/smoke gates, each run's validation-only metrics, checkpoint, and saved Hydra `evaluate_test` value before reusing artifacts. Failed or incomplete runs are never counted as completed. Campaign summaries and reports are generated after all 36 cells validate.

Small reproducibility artifacts belong in `data/`. Full logs, checkpoints, and Hydra output remain server-local under the ignored `outputs/mace_mag_v6a_expert_collaboration/` directory; raw datasets, model checkpoints, and large intermediate tensors are not committed.
