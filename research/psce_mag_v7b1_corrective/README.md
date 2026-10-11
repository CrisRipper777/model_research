# PSCE-MAG V7B.1 corrective replication

This directory contains the implementation audit and a new six-arm validation-only full-graph node-classification campaign correcting the V7B B0 relation-parameter freeze bug.

## Current checkpoint

The corrective implementation was frozen at `72d80b8a58cd1b10a5df6c5afab9528164296d8b`. The six-arm campaign completed all 54 validation-only full-graph NC runs with zero failures. Run rows, checkpoints, configuration fingerprints, and freeze SHA passed the final audit. The report and structured summaries are in this directory; no Test metrics were computed or recorded.

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
- `data/campaign_manifest.json`, `data/failures.json`, `data/resource_profile.csv`.

Formal checkpoints, logs, and Hydra snapshots are stored below `outputs/psce_mag_v7b1_corrective/` and are not committed.
