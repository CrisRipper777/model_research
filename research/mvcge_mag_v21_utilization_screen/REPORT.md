# MvCGE-MAG V2.1: Expert Utilization Granularity Screen

## Protocol and provenance

- Branch: `exp/mvcge_mag_v21_utilization_screen`; parent: `40b2c3953f6647222ce07053b10a7500c1b6a856`; freeze commit: `4dd4f2d74ed20c1700dae66f0d411d4d01dde9da`. The final artifact commit contains this report and its data tables.
- Validation-only `unified_full_graph_nc_v1`; Movies, Grocery, ele-fashion; seeds 42–44; four variants; 36/36 runs completed.
- Every run used `task.evaluate_test=false`, selected checkpoints by Validation Accuracy, and had no Test metric keys. The checkpoint audit read features, edges, model weights and validation-selected metadata only; it did not read labels or Test labels.
- No HPO, significance test, LP run, or model/config change after freeze. GPU: `cuda:0`. Unresolved failures: 0.

## Validation results

Run-level means and population standard deviations; paired differences are descriptive percentage-point deltas over matched dataset-seed runs.

| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Best epoch mean | Params model + head |
|---|---|---:|---:|---:|---:|
| Movies | U0_modality_static | 55.92% ± 0.38% | 48.57% ± 0.99% | 83.3 | 1,448,885 + 5,140 |
| Movies | U1_node_selection | 56.03% ± 0.62% | 47.72% ± 2.18% | 68.3 | 1,448,885 + 5,140 |
| Movies | U2_node_selection_strength | 56.01% ± 0.13% | 48.70% ± 1.18% | 72.7 | 1,448,885 + 5,140 |
| Movies | U3_collaborative | 55.71% ± 0.36% | 46.67% ± 2.23% | 62.3 | 1,448,885 + 5,140 |
| Grocery | U0_modality_static | 83.13% ± 0.36% | 76.02% ± 1.45% | 77.3 | 1,448,885 + 5,140 |
| Grocery | U1_node_selection | 83.26% ± 0.19% | 76.13% ± 0.45% | 79.0 | 1,448,885 + 5,140 |
| Grocery | U2_node_selection_strength | 83.37% ± 0.30% | 75.70% ± 0.90% | 70.0 | 1,448,885 + 5,140 |
| Grocery | U3_collaborative | 83.27% ± 0.40% | 75.74% ± 0.33% | 83.0 | 1,448,885 + 5,140 |
| ele-fashion | U0_modality_static | 87.38% ± 0.10% | 74.37% ± 1.18% | 175.0 | 1,186,741 + 3,084 |
| ele-fashion | U1_node_selection | 87.48% ± 0.10% | 75.01% ± 0.55% | 157.0 | 1,186,741 + 3,084 |
| ele-fashion | U2_node_selection_strength | 87.31% ± 0.15% | 75.26% ± 0.97% | 158.3 | 1,186,741 + 3,084 |
| ele-fashion | U3_collaborative | 87.26% ± 0.12% | 74.28% ± 0.83% | 123.0 | 1,186,741 + 3,084 |

Overall equally weighted run means:

```json
{
  "U0_modality_static": {
    "acc": 0.7547800143559774,
    "f1": 0.663211738887957
  },
  "U1_node_selection": {
    "acc": 0.7559105157852173,
    "f1": 0.6628681561629247
  },
  "U2_node_selection_strength": {
    "acc": 0.7556221617592705,
    "f1": 0.6655442638039654
  },
  "U3_collaborative": {
    "acc": 0.7541151775254143,
    "f1": 0.6556656579361385
  }
}
```

## Required paired comparisons

| Comparison | Δ Accuracy | Δ Macro-F1 | Positive pairs (Acc / F1) |
|---|---:|---:|---:|
| U1 − U0: node-conditioned selection | +0.11 pp | -0.03 pp | 5/9 / 4/9 |
| U2 − U1: node-conditioned strength | -0.03 pp | +0.27 pp | 3/9 / 4/9 |
| U3 − U2: collaborative context | -0.15 pp | -0.99 pp | 2/9 / 2/9 |
| U2 − U0: total node-conditioned change | +0.08 pp | +0.23 pp | 6/9 / 5/9 |
| U3 − U0: total collaborative change | -0.07 pp | -0.75 pp | 4/9 / 2/9 |

Per-dataset means and positive seed counts:

| Comparison | Dataset | Δ Accuracy mean (positive seeds) | Δ Macro-F1 mean (positive seeds) |
|---|---|---:|---:|
| U1 − U0: node-conditioned selection | Movies | +0.11 pp (2/3) | -0.85 pp (2/3) |
| U1 − U0: node-conditioned selection | Grocery | +0.13 pp (2/3) | +0.11 pp (1/3) |
| U1 − U0: node-conditioned selection | ele-fashion | +0.10 pp (1/3) | +0.64 pp (1/3) |
| U2 − U1: node-conditioned strength | Movies | -0.02 pp (1/3) | +0.98 pp (1/3) |
| U2 − U1: node-conditioned strength | Grocery | +0.11 pp (2/3) | -0.43 pp (1/3) |
| U2 − U1: node-conditioned strength | ele-fashion | -0.17 pp (0/3) | +0.25 pp (2/3) |
| U3 − U2: collaborative context | Movies | -0.30 pp (0/3) | -2.03 pp (1/3) |
| U3 − U2: collaborative context | Grocery | -0.10 pp (1/3) | +0.04 pp (1/3) |
| U3 − U2: collaborative context | ele-fashion | -0.05 pp (1/3) | -0.98 pp (0/3) |
| U2 − U0: total node-conditioned change | Movies | +0.09 pp (2/3) | +0.13 pp (1/3) |
| U2 − U0: total node-conditioned change | Grocery | +0.23 pp (3/3) | -0.32 pp (2/3) |
| U2 − U0: total node-conditioned change | ele-fashion | -0.07 pp (1/3) | +0.89 pp (2/3) |
| U3 − U0: total collaborative change | Movies | -0.21 pp (1/3) | -1.90 pp (0/3) |
| U3 − U0: total collaborative change | Grocery | +0.14 pp (2/3) | -0.28 pp (1/3) |
| U3 − U0: total collaborative change | ele-fashion | -0.13 pp (1/3) | -0.09 pp (1/3) |

No significance testing was performed.

## Utilization and routing diagnostics

- Normalized Top-2 pair entropy (range 0–1), averaged over run × modality: U0 `0.0000`, U1 `0.6230`, U2 `0.7111`, U3 `0.7226`. U0 has exactly one unordered pair and entropy 0 in every modality checkpoint.
- Route-to-modality-mean JS (nats), averaged over active node × modality checkpoints: U0 `0.00000000`, U1 `0.149002`, U2 `0.174744`, U3 `0.170394`.
- Evaluation strength standard deviation, averaged over modality checkpoints: U0 `0.00000000`, U1 `0.00000000`, U2 `0.112798`, U3 `0.122611`. Static variants should be zero up to floating-point representation; U2/U3 can vary by node.
- Mean experts used by the Text ∪ Visual union: U0 `3.000` of 4, U1–U3 `4.000` of 4. Both-modality dead expert slots: U0 `9/9` run-checkpoints, U1–U3 `0/27`. A one-modality zero-load expert is not labeled collapse; the U0 matched static control does show a both-modality dead slot in each run.
- Functional expert-output cosine across U1–U3 checkpoints: mean-node mean `0.3778`, range `-0.3568` to `0.9204`; flattened mean `0.4440`, range `-0.1212` to `0.9332`. Learned alpha pairwise cosine mean `0.0491`. These are descriptive similarity diagnostics, not a diversity objective.
- Matched U2-vs-U3 routing JS mean `0.133789` nats (per-modality mean/std/p50/p90 in `cross_model_context_diagnostics.csv`); absolute strength difference mean `0.082234`.
- Per-modality strength summaries include mean/std/p10/p50/p90, fraction above 0.9, scaled MoE correction RMS, prior RMS and their ratio.

## Interpretation rules

The thresholds used for “approximately equal” are ±0.10 pp in overall Validation Accuracy; “low pair entropy” is normalized entropy ≤0.05; “nonzero matched routing JS” is mean >1e-7 nats. These are descriptive reading rules, not inferential thresholds.

- **A. Observed.** U1 的准确率 paired mean 高于 U0，node-conditioned expert selection 获得描述性支持；以 paired deltas 为证据，不作因果结论。
- **B. Not observed.** 该描述性模式未达到上述透明判断条件。
- **C. Observed.** U1 高于 U0，而 U2 未高于 U1；selection 有正向信号，node-specific strength 在此屏幕中未显示额外准确率收益。
- **D. Not observed.** 该描述性模式未达到上述透明判断条件。
- **E. Not observed.** 该描述性模式未达到上述透明判断条件。
- **F. Observed.** U3 未高于 U2；本次屏幕不支持保留 cross-modal context 以提升准确率。
- **G. Observed.** U2 与 U0 的准确率差在 ±0.10 pp 内；该结果与 shared expert bank 主要依赖 modality-level utilization 相容。
- **H. Not observed.** 该描述性模式未达到上述透明判断条件。
- **I. Observed.** 至少一个 checkpoint 出现同一 expert 在 Text 和 Visual 都 zero-load；本屏幕共有 9 个此类 run-checkpoint（其中 U0=9，U1–U3=0），按规则记为更强的 expert-starvation 提醒。单一 modality 的 zero-load 不计作 collapse。

## Layer-wise gate and scientific boundaries

The screening gate is 未满足下一阶段筛选门槛. For this descriptive screen, “clear U1 improvement” requires both overall Validation Accuracy and Macro-F1 deltas above zero; U3 also requires Accuracy above U2 and nonzero matched U2-vs-U3 routing JS. U1−U0 has +0.11 pp Accuracy but −0.03 pp Macro-F1 and Accuracy wins on only 5/9 pairs, so that small uneven signal does not pass the next-stage gate. This run stops here; it does not implement layer-wise stacking.

- This is not an exact MvCGE reproduction. It is a single-block MAG adaptation with one shared four-expert bank and fixed Top-2 routing.
- U0 is a modality-static matched expert control, not ordinary GPR. U1−U0 screens selection granularity, U2−U1 screens strength granularity, and U3−U2 screens collaborative context.
- There is no external structural anchor, beta/B base, residual cap, private expert pool, graph discrepancy/MMD, contrastive routing, confidence fusion, new fusion, layer-wise stacking, HPO, LP, or Test evaluation.
- The paired patterns are descriptive and do not establish causal routing effects.

## Reproducibility artifacts

- Compact records: `data/environment.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`.
- Tables: `data/summary.csv`, `data/paired_comparisons.csv`, `data/routing_diagnostics.csv`, `data/pair_diagnostics.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/strength_diagnostics.csv`, `data/cross_model_context_diagnostics.csv`.
- Checkpoints, raw outputs, tensors, embeddings and Hydra run logs remain under ignored `outputs/`.
