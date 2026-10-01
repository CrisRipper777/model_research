# L0 — Relation-to-Function Learnability Audit

This directory contains the target reliability, target-group cross-validation, function-utility predictability, relation-state capacity, frozen-state readout, and high-margin audit requested for L0.

## Reproduction

The campaign reused all nine full P1.3 utility tables and all nine P1.3 joint-head checkpoints. It did not regenerate P1.3 artifacts.

```bash
conda run --no-capture-output -n yhf_env python -m pytest -q tests/test_l0_relation_function_learnability.py
conda run --no-capture-output -n yhf_env python scripts/run_l0_relation_function_learnability.py --smoke --device cuda:1
conda run --no-capture-output -n yhf_env python scripts/run_l0_relation_function_learnability.py --campaign --device cuda:1 --max-epochs 150
conda run --no-capture-output -n yhf_env python scripts/analyze_l0_relation_function_learnability.py --figures --report
```

The smoke is Movies/42 only and is a correctness gate, not a formal estimate. The campaign includes Movies, Grocery and ele-fashion with seeds 42, 43 and 44. Metrics are averaged over outer folds within each dataset×seed before the three-seed mean and population SD are calculated.

## Artifacts

- `report.md` — answers the frozen L0 questions and records the diagnostic judgment.
- `run_manifest.json` — source provenance, protocol, environment, execution and deviation record.
- `data/` — source audit, cross-seed target reliability, fold assignments, fold-level metrics, seed summaries, within-target metrics, high-margin metrics, parameter summary and smoke audit.
- `figures/` — five PNGs with PDF/SVG/600 dpi TIFF companions and alignment/text/collision QA records.
- Large checkpoints, per-edge E0.1 state representations and resumable run data remain in ignored `outputs/l0_relation_function_learnability_audit/`.

## Scope boundary

The utility targets are validation-derived P1.3 diagnostics, not ground-truth edge roles. P1.3 AbsDiff/Product are not relabeled as E0 Relational/Cross-Modal. L0 does not evaluate the original test split, change E0/E0.1, design a new model, or implement M1.
