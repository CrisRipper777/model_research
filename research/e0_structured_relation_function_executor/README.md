# E0 — Structured Relation-Function Executor Screening

This package records the frozen E0 one-step executor screen. It compares StaticMix, TargetMix, and EdgeMix on Movies, Grocery, and ele-fashion for seeds 42–44, with M0 UNI/SEM as historical references. The three E0 variants share the same relation encoder, structured function bank, router, semantic backbone, fusion, parameter count, and per-seed initialization; only router pooling granularity changes.

## Result in brief

The screen does not establish incremental value for within-neighborhood edge-specific function composition. EdgeMix is close to TargetMix in the paired task metrics; within-target shuffle and target-mean interventions have very small average effects. The trained route is strongly Smooth-dominant on Movies and Grocery. Ele-fashion uses Relational and Cross-Modal more often and shows visible within-target and modality-specific variation, but this does not produce consistent EdgeMix-over-TargetMix performance gains. The pre-frozen category is **FUNCTION_BANK_NOT_CONVINCING** for this screen. This result does not revise P0–P1.3 evidence about propagation-function heterogeneity.

## Contents

- [`report.md`](report.md): protocol, correctness evidence, results, frozen-question answers, and self-audit.
- [`run_manifest.json`](run_manifest.json): provenance, commands, environment, training totals, and access boundaries.
- [`data/`](data/): run-level and summary performance, routing, function, parameter, and intervention tables.
- [`figures/`](figures/): six screen and diagnostic figures, each in PNG, editable SVG/PDF, and 600-dpi TIFF; render QA records are under `figures/qa/`.

Formal run JSON and checkpoints are in the ignored path `outputs/e0_structured_relation_function_executor/`.

## Reproduce analysis

From the repository root, activate `yhf_env` and run:

```bash
python scripts/analyze_e0_structured_executor.py
```

The 27-run campaign command and exact environment are recorded in the manifest. Do not interpret StaticMix−UNI as a pure routing effect: StaticMix adds the structured function bank. The matched-granularity comparisons are TargetMix−StaticMix and EdgeMix−TargetMix.
