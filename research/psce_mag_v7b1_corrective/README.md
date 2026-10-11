# PSCE-MAG V7B.1 corrective replication

This directory contains the implementation audit and a new six-arm validation-only full-graph node-classification campaign correcting the V7B B0 relation-parameter freeze bug.

## Current checkpoint

The model fix, regression tests, cache/parameter audits, real-graph parity audit, all 12 GPU preflight paths, and all six Movies smoke runs have passed. These files will be committed as the implementation freeze. The 54-run formal campaign starts only from that freeze SHA; see `data/campaign_manifest.json` after it completes.

## Run commands

From the repository root and `yhf_env`:

```bash
python scripts/run_psce_mag_v7b1_corrective.py --mode audit
python scripts/run_psce_mag_v7b1_corrective.py --mode tests
CUDA_VISIBLE_DEVICES=1 python scripts/run_psce_mag_v7b1_corrective.py --mode training-parity --device cuda:0
CUDA_VISIBLE_DEVICES=1 python scripts/run_psce_mag_v7b1_corrective.py --mode preflight --device cuda:0
CUDA_VISIBLE_DEVICES=1 python scripts/run_psce_mag_v7b1_corrective.py --mode smoke --device cuda:0
CUDA_VISIBLE_DEVICES=1 python scripts/run_psce_mag_v7b1_corrective.py --mode campaign --device cuda:0
```

Use `--mode resume` with the same freeze SHA and configuration fingerprint to recover validated runs. The runner rejects a different freeze or configuration.

## Artifacts

- `REPORT.md`: per-seed metrics, paired comparisons, diagnostics, and conclusions.
- `data/run_rows.json`: all formal run metrics, resources, checkpoint paths, and freeze/config provenance.
- `data/summary.csv`, `data/paired_comparisons.csv`, `data/semantic_diagnostics.csv`.
- `data/training_parity_audit.json`, `data/parameter_grad_audit.json`.
- `data/candidate_cache_audit.json`, `data/random_graph_audit.json`.
- `data/preflight_summary.json`, `data/smoke_summary.json`, `data/resource_profile.json`.

Formal checkpoints, logs, and Hydra snapshots are stored below `outputs/psce_mag_v7b1_corrective/` and are not committed.
