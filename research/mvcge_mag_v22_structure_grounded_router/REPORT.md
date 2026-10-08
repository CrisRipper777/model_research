# MvCGE-MAG V2.2: Structure-Grounded Expert Routing

## Protocol and provenance

- Branch: `exp/mvcge_mag_v22_structure_grounded_router`; parent: `c57b57fcaf742f76416e9ed72df47d1f372a0f5b`; freeze commit: `f74013778d5c2c5f2f2abb35cf85cd93c2e2d66f`.
- Validation-only `unified_full_graph_nc_v1`; Movies, Grocery, `ele-fashion`; seeds 42–44; four variants; 36/36 runs completed.
- Every training command set `task.evaluate_test=false`; checkpoint selection used Validation Accuracy. Run metrics have no Test keys. The checkpoint audit accessed features, graph edges, model weights and validation-selected metadata only, without reading labels or indexing Test labels.
- No HPO, significance test, LP run, or post-freeze model/config edit. GPU: `cuda:0`. Unresolved failures: 0.
- V2.1's dead-expert wording correction is recorded in `V21_ERRATA.md`; prior V2.1 artifacts remain unchanged.

## Validation results

Run-level means and population standard deviations; paired deltas are descriptive percentage points across matched dataset-seed runs.

| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Best epoch mean | Trainable params (model + head) |
|---|---|---:|---:|---:|---:|
| Movies | R0_modality_static | 55.98% ± 0.47% | 48.77% ± 0.85% | 67.3 | 1,448,885 + 5,140 |
| Movies | R1_free_node | 55.91% ± 0.52% | 47.90% ± 2.22% | 69.3 | 1,448,885 + 5,140 |
| Movies | R2_structure_grounded | 56.02% ± 0.33% | 47.91% ± 2.02% | 75.3 | 1,449,783 + 5,140 |
| Movies | R3_expert_compatibility | 55.97% ± 0.38% | 48.23% ± 1.60% | 74.7 | 1,459,449 + 5,140 |
| Grocery | R0_modality_static | 83.21% ± 0.18% | 76.13% ± 1.00% | 86.7 | 1,448,885 + 5,140 |
| Grocery | R1_free_node | 83.24% ± 0.14% | 75.77% ± 1.06% | 81.7 | 1,448,885 + 5,140 |
| Grocery | R2_structure_grounded | 83.11% ± 0.24% | 75.38% ± 0.63% | 66.7 | 1,449,783 + 5,140 |
| Grocery | R3_expert_compatibility | 83.08% ± 0.26% | 75.13% ± 1.44% | 69.0 | 1,459,449 + 5,140 |
| ele-fashion | R0_modality_static | 87.31% ± 0.07% | 74.24% ± 1.22% | 174.0 | 1,186,741 + 3,084 |
| ele-fashion | R1_free_node | 87.59% ± 0.09% | 75.96% ± 0.45% | 191.0 | 1,186,741 + 3,084 |
| ele-fashion | R2_structure_grounded | 87.49% ± 0.15% | 75.47% ± 1.30% | 192.0 | 1,187,639 + 3,084 |
| ele-fashion | R3_expert_compatibility | 87.43% ± 0.08% | 74.70% ± 1.11% | 158.0 | 1,197,305 + 3,084 |

Overall equally weighted run means:

```json
{
  "R0_modality_static": {
    "val_accuracy": 0.7550129691759745,
    "val_macro_f1": 0.6637844923046151
  },
  "R1_free_node": {
    "val_accuracy": 0.7557864520284865,
    "val_macro_f1": 0.6654632392250837
  },
  "R2_structure_grounded": {
    "val_accuracy": 0.7554118699497647,
    "val_macro_f1": 0.6625513873016616
  },
  "R3_expert_compatibility": {
    "val_accuracy": 0.7549544307920668,
    "val_macro_f1": 0.6601970982607674
  }
}
```

## Required paired comparisons

| Comparison | Δ Accuracy | Δ Macro-F1 | Positive pairs (Acc / F1) |
|---|---:|---:|---:|
| R1 − R0: historical free-node router signal | +0.08 pp | +0.17 pp | 5/9 / 6/9 |
| R2 − R1: structural evidence and conservative residual | -0.04 pp | -0.29 pp | 3/9 / 3/9 |
| R2 − R0: structure-grounded routing vs static reuse | +0.04 pp | -0.12 pp | 6/9 / 4/9 |
| R3 − R2: explicit node–expert matching | -0.05 pp | -0.24 pp | 2/9 / 5/9 |
| R3 − R0: complete V2.2 router vs static reuse | -0.01 pp | -0.36 pp | 6/9 / 3/9 |

Per-dataset means and positive seed counts:

| Comparison | Dataset | Δ Accuracy mean (positive seeds) | Δ Macro-F1 mean (positive seeds) |
|---|---|---:|---:|
| R1 − R0: historical free-node router signal | Movies | -0.07 pp (0/3) | -0.87 pp (2/3) |
| R1 − R0: historical free-node router signal | Grocery | +0.03 pp (2/3) | -0.36 pp (1/3) |
| R1 − R0: historical free-node router signal | ele-fashion | +0.27 pp (3/3) | +1.73 pp (3/3) |
| R2 − R1: structural evidence and conservative residual | Movies | +0.11 pp (2/3) | +0.01 pp (1/3) |
| R2 − R1: structural evidence and conservative residual | Grocery | -0.13 pp (0/3) | -0.39 pp (1/3) |
| R2 − R1: structural evidence and conservative residual | ele-fashion | -0.10 pp (1/3) | -0.49 pp (1/3) |
| R2 − R0: structure-grounded routing vs static reuse | Movies | +0.04 pp (2/3) | -0.86 pp (1/3) |
| R2 − R0: structure-grounded routing vs static reuse | Grocery | -0.10 pp (1/3) | -0.75 pp (0/3) |
| R2 − R0: structure-grounded routing vs static reuse | ele-fashion | +0.18 pp (3/3) | +1.24 pp (3/3) |
| R3 − R2: explicit node–expert matching | Movies | -0.05 pp (1/3) | +0.32 pp (2/3) |
| R3 − R2: explicit node–expert matching | Grocery | -0.03 pp (0/3) | -0.25 pp (2/3) |
| R3 − R2: explicit node–expert matching | ele-fashion | -0.06 pp (1/3) | -0.78 pp (1/3) |
| R3 − R0: complete V2.2 router vs static reuse | Movies | -0.01 pp (2/3) | -0.54 pp (2/3) |
| R3 − R0: complete V2.2 router vs static reuse | Grocery | -0.13 pp (1/3) | -1.00 pp (0/3) |
| R3 − R0: complete V2.2 router vs static reuse | ele-fashion | +0.12 pp (3/3) | +0.46 pp (1/3) |

No significance testing was performed.

## Routing, experts, evidence, and strength

| Variant | Normalized Top-2 pair entropy | Route-to-mean JS | Route-to-static JS | Mean experts in Text ∪ Visual | Dead slots | Runs with ≥1 dead slot |
|---|---:|---:|---:|---:|---:|---:|
| R0_modality_static | 0.0000 | 0.000000 | 0.000000 | 3.111/4 | 8/9 | 7/9 |
| R1_free_node | 0.6601 | 0.163982 | 0.239410 | 3.889/4 | 1/9 | 1/9 |
| R2_structure_grounded | 0.3412 | 0.078133 | 0.128537 | 3.556/4 | 4/9 | 4/9 |
| R3_expert_compatibility | 0.2977 | 0.072432 | 0.128656 | 3.667/4 | 3/9 | 3/9 |


- Both-modality dead-expert counts distinguish the **number of dead slots** from the **number of run-checkpoints containing at least one dead slot**. A one-modality zero-load expert is not labeled a both-modality dead slot. R0's modality-static Top-2 control is not described as a dynamic-router collapse.
- Checkpoint R0/U0 and R1/U1 compatibility audits used `atol=1e-6, rtol=1e-5` on intermediate large-graph outputs and `atol=5e-6, rtol=1e-5` on fused z; the maximum observed absolute difference was `3.81e-06` for R0 and `1.19e-06` for R1. The requested toy-graph regression tests remain at `atol=1e-7, rtol=0`. This full-graph audit tolerance accommodates accumulated numerical roundoff and does not alter any trained model or configuration.
- Learned mean eta by variant: R0 `0.1000`, R1 `0.1000`, R2 `0.1037`, R3 `0.1019`. Mean R3 kappa: `0.1024`. Per-dataset/seed/modality values are in `data/routing_grounding.csv`.
- R2/R3 mean route-to-static JS is `0.128537` / `0.128656` nats. Mean normalized Top-2 pair entropy and route-to-modality-mean JS appear above; detailed pair counts and distributions are in `data/pair_diagnostics.csv`.
- Active-node reliability means average `0.9233` with mean across-checkpoint within-graph standard deviation `0.0196`; mean within-graph degree-norm standard deviation is `0.1452`. Mean within-graph hop-cosine and hop-displacement standard deviations across orders are `0.1352` and `0.1161`. See `data/evidence_diagnostics.csv` for each order and quantiles.
- Learned alpha pairwise cosine mean is `0.0543`. Functional expert-output mean-node cosine is `0.3858` and flattened cosine is `0.4503`. Pairwise records and checkpoint ranges are in `data/expert_similarity.csv`.
- All four variants use modality-static strength. Mean node-strength standard deviation across modality checkpoints: `0.0000000000`. Strength summaries, scaled correction RMS, prior RMS and their ratio are in `data/strength_diagnostics.csv`.

## Interpretation map

Thresholds and “approximately equal” definitions were fixed in `README.md` before the formal campaign. The patterns are descriptive and do not establish causal mechanisms.

- **A. Not observed.** R2 is above both R1 and R0 on overall Accuracy and Macro-F1. This pattern supports structural observation and conservative residualization as a promising explanation for the old free router's weak screen.
- **B. Not observed.** R2 improves on R1 while remaining approximately equal to R0; the structural evidence repairs the free router, but does not establish net value over static reuse.
- **C. Not observed.** R3 exceeds R2 on both overall metrics. Mean learned kappa is 0.1024; values above 0.05 are treated as clearly non-near-zero for this descriptive reading rule.
- **D. Not observed.** At least one of R2/R3 passes the stable-positive screen against R0, with positive overall Accuracy and Macro-F1, at least 6/9 positive Accuracy pairs, and positive Accuracy means on at least two datasets.
- **E. Not observed.** R2, R3, and R0 are approximately equal; learned residuals are near zero and route-to-static JS is negligible. The router is retaining modality-static routing in this screen.
- **F. Not observed.** R2/R3 are approximately equal to R0 despite non-near-zero residuals and material route changes. Routing changes do not translate into task gain with the current expert action space.
- **G. Observed.** At least one structure-grounded variant is below R0 on both overall metrics. The current structure-grounded routing design is not supported by this screen; do not further complexify the router.

## Success gate and scientific boundaries

The stable-positive gate is **not met**. Only a stable positive R2 or R3 comparison against R0 warrants a later study of node routing. If the gate is not met, stop router optimization and leave modality-conditioned topology/reliable structural expert actions for a separate future experiment. This screen does not implement them.

- This is a single-block MAG adaptation, not an exact MvCGE reproduction. It keeps one shared four-expert structural bank and fixed Top-2 routing.
- Reliability is evidence for router inputs only. Physical propagation remains the V2.1 symmetric normalized graph; no reliability-weighted message passing is used.
- There is no cross-modal feature/context, private expert pool, discrepancy/MMD, contrastive routing, confidence fusion, node-specific strength, edge routing, layer stacking, HPO, LP, or Test evaluation.
- The observed paired patterns and router diagnostics are descriptive; no causal claims or significance tests are made.

## Reproducibility artifacts

- `data/environment.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`
- `data/summary.csv`, `data/paired_comparisons.csv`, `data/routing_diagnostics.csv`, `data/pair_diagnostics.csv`
- `data/evidence_diagnostics.csv`, `data/routing_grounding.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/strength_diagnostics.csv`
- Checkpoints, raw outputs, embeddings, node tensors, and Hydra logs remain under ignored `outputs/` and are not committed.
