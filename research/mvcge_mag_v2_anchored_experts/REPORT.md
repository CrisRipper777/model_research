# MvCGE-MAG V2: Anchored Collaborative Structural Experts

## Protocol and provenance

- Branch: `exp/mvcge_mag_v2_anchored_experts`; parent: `412e69221a30fa3779e57e22fa42073697bdf4f8`; freeze commit: `7c76bc11b56c53439db005b5fb106c5526e0f6d5`.
- Validation-only protocol: `unified_full_graph_nc_v1`; datasets Movies, Grocery, ele-fashion; seeds 42, 43, 44; four fixed variants; 36/36 completed.
- `task.evaluate_test=false` for every run. Metrics JSON contains no Test metrics; diagnostics loaded features, graph edges, model weights and validation-selected metadata without reading label fields or indexing Test labels. The standard data loader materializes the full label vector as allowed by the protocol.
- Checkpoint selection used Validation Accuracy. No HPO, significance test, intervention, LP task, or model/config change occurred after freeze.
- GPU: `cuda:0`. Failures recorded: 0 unresolved.

## Validation results

Values are run-level means and population standard deviations. Paired deltas are descriptive percentage points across matched dataset-seed runs.

| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Mean best epoch | Trainable params (model + head) |
|---|---|---:|---:|---:|---:|
| Movies | M0_anchor | 54.18% ± 0.13% | 44.49% ± 0.12% | 56.7 | 1,249,029 + 5,140 |
| Movies | M1_direct_moe | 55.69% ± 0.15% | 48.60% ± 1.23% | 80.7 | 1,448,885 + 5,140 |
| Movies | M2_anchored_moe | 55.22% ± 0.26% | 46.37% ± 1.31% | 54.0 | 1,448,890 + 5,140 |
| Movies | M3_collaborative_moe | 55.52% ± 0.11% | 48.34% ± 1.08% | 61.3 | 1,448,890 + 5,140 |
| Grocery | M0_anchor | 81.89% ± 0.30% | 74.48% ± 1.12% | 90.0 | 1,249,029 + 5,140 |
| Grocery | M1_direct_moe | 83.36% ± 0.28% | 76.03% ± 1.13% | 70.3 | 1,448,885 + 5,140 |
| Grocery | M2_anchored_moe | 82.86% ± 0.34% | 75.25% ± 0.83% | 78.0 | 1,448,890 + 5,140 |
| Grocery | M3_collaborative_moe | 83.06% ± 0.36% | 75.82% ± 0.73% | 81.3 | 1,448,890 + 5,140 |
| ele-fashion | M0_anchor | 87.35% ± 0.25% | 74.53% ± 0.71% | 142.7 | 986,885 + 3,084 |
| ele-fashion | M1_direct_moe | 87.34% ± 0.16% | 74.42% ± 0.57% | 133.7 | 1,186,741 + 3,084 |
| ele-fashion | M2_anchored_moe | 87.44% ± 0.16% | 74.71% ± 0.98% | 147.0 | 1,186,746 + 3,084 |
| ele-fashion | M3_collaborative_moe | 87.44% ± 0.09% | 74.89% ± 0.64% | 143.3 | 1,186,746 + 3,084 |

Overall equally weighted run means:

```json
{
  "M0_anchor": {
    "acc": 0.7447581556108263,
    "f1": 0.6450026790931607
  },
  "M1_direct_moe": {
    "acc": 0.7546254528893365,
    "f1": 0.6634999796642717
  },
  "M2_anchored_moe": {
    "acc": 0.7517406874232822,
    "f1": 0.6544233479893812
  },
  "M3_collaborative_moe": {
    "acc": 0.7534123924043443,
    "f1": 0.6635030118300089
  }
}
```

## Paired comparisons

| Comparison | Δ Accuracy | Δ Macro-F1 | Positive accuracy pairs | Positive Macro-F1 pairs |
|---|---:|---:|---:|---:|
| M1_direct_moe - M0_anchor | +0.99 pp | +1.85 pp | 7/9 | 8/9 |
| M2_anchored_moe - M0_anchor | +0.70 pp | +0.94 pp | 8/9 | 7/9 |
| M2_anchored_moe - M1_direct_moe | -0.29 pp | -0.91 pp | 2/9 | 2/9 |
| M3_collaborative_moe - M2_anchored_moe | +0.17 pp | +0.91 pp | 7/9 | 7/9 |
| M3_collaborative_moe - M0_anchor | +0.87 pp | +1.85 pp | 8/9 | 8/9 |

Dataset means and positive seed counts:

| Comparison | Dataset | Δ Accuracy mean (positive seeds) | Δ Macro-F1 mean (positive seeds) |
|---|---|---:|---:|
| M1_direct_moe - M0_anchor | Movies | +1.51 pp (3/3) | +4.11 pp (3/3) |
| M1_direct_moe - M0_anchor | Grocery | +1.46 pp (3/3) | +1.55 pp (3/3) |
| M1_direct_moe - M0_anchor | ele-fashion | -0.01 pp (1/3) | -0.11 pp (2/3) |
| M2_anchored_moe - M0_anchor | Movies | +1.04 pp (3/3) | +1.88 pp (3/3) |
| M2_anchored_moe - M0_anchor | Grocery | +0.97 pp (3/3) | +0.77 pp (3/3) |
| M2_anchored_moe - M0_anchor | ele-fashion | +0.09 pp (2/3) | +0.18 pp (1/3) |
| M2_anchored_moe - M1_direct_moe | Movies | -0.47 pp (0/3) | -2.23 pp (0/3) |
| M2_anchored_moe - M1_direct_moe | Grocery | -0.50 pp (0/3) | -0.78 pp (0/3) |
| M2_anchored_moe - M1_direct_moe | ele-fashion | +0.10 pp (2/3) | +0.29 pp (2/3) |
| M3_collaborative_moe - M2_anchored_moe | Movies | +0.30 pp (3/3) | +1.97 pp (3/3) |
| M3_collaborative_moe - M2_anchored_moe | Grocery | +0.20 pp (3/3) | +0.57 pp (3/3) |
| M3_collaborative_moe - M2_anchored_moe | ele-fashion | -0.00 pp (1/3) | +0.18 pp (1/3) |
| M3_collaborative_moe - M0_anchor | Movies | +1.34 pp (3/3) | +3.85 pp (3/3) |
| M3_collaborative_moe - M0_anchor | Grocery | +1.17 pp (3/3) | +1.34 pp (3/3) |
| M3_collaborative_moe - M0_anchor | ele-fashion | +0.09 pp (2/3) | +0.36 pp (2/3) |

No inferential or significance testing was performed.

## Selected-checkpoint mechanism diagnostics

- Expert profiles were analyzed for M1/M2/M3: learned alpha pairwise cosine mean `0.0568`. Hadamard is only the initialization.
- Across 54 MoE modality/checkpoint records, all four experts had positive selection share in `43/54` records; `11` records had at least one zero-load expert. The median max/min ratio across positive loads was `1.576`.
- Functional expert output cosine: mean-node mean `0.4321`, pairwise range `-0.3891` to `0.9208`; flattened mean `0.4853`, range `-0.5463` to `0.9231`.
- Dense routing entropy mean ± mean within-checkpoint std: `0.8337 ± 0.1834` nats (maximum for four experts is `1.3863`); Top1−Top2 logit margin mean ± mean within-checkpoint std: `3.0980 ± 1.0826`.
- Mean fraction with `rho > 0.9*rho_max` across M2/M3 modality-checkpoints: `0.6253`.
- Mean scaled expert residual/base correction RMS ratio across M2/M3: `1.2196`.
- M2/M3 text-vs-visual sparse routing JS mean across selected checkpoints: `0.2734` nats.

Detailed per-run values are in `data/routing_diagnostics.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, and `data/strength_diagnostics.csv`.

## Interpretation map

The conditions below are applied as descriptive patterns; they do not establish causal mechanisms.

- **A. Not triggered.** The specified joint pattern was not observed in these descriptive summaries.
- **B. Not triggered.** The specified joint pattern was not observed in these descriptive summaries.
- **C. Observed.** M1 exceeds M0, M2 is below M1, and many residual strengths approach the cap. This can indicate that the bounded residual limits expert capacity; it does not by itself reject MoE.
- **D. Observed.** M3 exceeds M2 and their text/visual sparse routing distributions differ. This supports independent value from the DMCAR-inspired public cross-view routing context in this screen.
- **E. Not triggered.** The specified joint pattern was not observed in these descriptive summaries.
- **F. Not triggered.** The specified joint pattern was not observed in these descriptive summaries.
- **G. Observed.** At least one MoE modality-checkpoint does not use all four experts. Review the observed shares and max/min ratios before interpreting the result as evidence against shared experts; the balance objective/weight may need a dedicated study.
- **H. Not triggered.** The specified joint pattern was not observed in these descriptive summaries.

## Scientific boundaries

- This is an MvCGE-skeleton MAG-specific adaptation, not an exact MvCGE reproduction.
- It implements one collaborative expert block, not the full layer-wise architecture.
- Load balancing is an MvCGE-inspired per-modality surrogate, not an exact reproduction of Eq. (11).
- There is no graph discrepancy/MMD, DMCAR private expert pool, C2GMoE contrastive routing or confidence fusion, edge routing, topology learning, cross-modal attention, or modality transformer.
- Hadamard rows initialize trainable profiles; they do not define fixed expert semantics. RawPoly supplies a structural response substrate and is not described as a GPR model.
- No expert is assigned a fixed frequency band. Routing diagnostics do not support causal claims about node routing.

## Reproducibility artifacts

- `data/environment.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`
- `data/summary.csv`, `data/paired_comparisons.csv`, `data/routing_diagnostics.csv`
- `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/strength_diagnostics.csv`
- Raw outputs, checkpoints, and Hydra logs remain in the ignored `outputs/` tree and are not committed.
