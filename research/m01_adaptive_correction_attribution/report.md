# M0.1 — Adaptive Correction Attribution Closure

## 结论摘要

A-wide 与原 B 的总可训练参数差为 **10**。在九个 dataset × seed 配对上，B 相对 A-wide 的 Accuracy 几乎持平（+0.03 pp），Macro-F1 有小幅优势（+0.78 pp，7/9 配对为正），但 CE 反而高 0.0145（7/9 更差）。B、B-static、B-target 的任务指标也非常接近。旧 B checkpoint 的 function-off 有稳定损失；但 c-only shuffle 和 c-target-mean 的影响都很小。当前数据不支持把旧 B 的 branch utility 进一步归因到同一 target 内的 edge-specific c 对应关系。

冻结决策为 **CAPACITY_EXPLANATION_REMAINS**：A-wide 已匹配 B 的参数量并达到相近表现，B 相对它没有跨指标一致优势。B-static/B-target 与 B 也没有显示可靠的性能分层，因此 correction granularity 仍未被任务结果明确区分。这个结论只描述当前 M0 learner，不否定 P0–P1.3 的 propagation utility/function heterogeneity 问题证据。

## 实验边界与统计口径

- 新训练仅含 A-wide、B-static、B-target：Movies/Grocery/ele-fashion × seeds 42/43/44，共 27 runs。SEM/UNI/A/B 直接读取已提交 M0 结果；旧 B 的九个 checkpoint 均复用，没有重训。
- 节点分类、validation-only；没有加载 test indices、没有把 test labels 提供给训练或评估，也没有 link prediction。
- 性能表给出 dataset × seed 原始值及其跨九个配对的 mean ± population SD。配对差始终先按同一 dataset 和 seed 计算。没有做显著性检验；这些描述性差值不能视为统计显著性结论。
- Intervention 是固定 checkpoint 的 validation-time reliance diagnostic；它不能代替重新训练的 B-static/B-target 比较。
- 正式协议无偏离：AdamW、lr=1e-3、weight_decay=1e-4、最多 300 epochs、patience=30、min_epoch=30、grad clip=1.0、按 validation Accuracy 选 checkpoint。没有 dataset-specific 调参。

## 参数与主性能

A-wide 的两个 gate 均为 `Linear(64,126)-GELU-Linear(126,1)`，variant-specific 参数为 16,634；B 为 gate 130 + correction head 130 + rank-32 basis 16,384 = 16,644。实际 `|trainable(A-wide)−trainable(B)|=10`。B-target 参数与 B 完全相同；B-static 比 B 少 128 个参数（两组 65-parameter correction heads 替换为两个 scalar logits）。逐 dataset/seed 的 backbone、relation encoder、default transform、gate、correction control、basis、classifier 和总参数数见 [parameter_attribution.csv](data/parameter_attribution.csv)。

下表为跨三个数据集、三个 seed 的 mean ± population SD；Accuracy/Macro-F1 用百分比，CE 保持原尺度。

| 方法 | Accuracy (%) | Macro-F1 (%) | CE |
|---|---:|---:|---:|
| SEM | 72.51 ± 14.69 | 57.98 ± 14.32 | 0.924 ± 0.441 |
| UNI | 75.03 ± 14.03 | 62.64 ± 12.05 | 0.848 ± 0.422 |
| A | 74.93 ± 14.04 | 62.40 ± 12.51 | 0.852 ± 0.429 |
| A-wide | 75.02 ± 14.02 | 61.94 ± 12.96 | 0.839 ± 0.409 |
| B-static | 75.10 ± 13.96 | 62.57 ± 12.47 | 0.831 ± 0.407 |
| B-target | 75.05 ± 14.00 | 62.56 ± 12.90 | 0.850 ± 0.408 |
| B | 75.05 ± 13.94 | 62.71 ± 12.55 | 0.854 ± 0.414 |

关键配对差为 left − right。分数列为 pp；CE 差为原尺度（正数代表左侧 CE 更高）。

| 配对差 | Accuracy (pp) | Macro-F1 (pp) | CE |
|---|---:|---:|---:|
| A-wide − A | +0.084 ± 0.254 | −0.463 ± 1.130 | −0.0132 ± 0.0508 |
| B − A-wide | +0.032 ± 0.180 | +0.776 ± 0.801 | +0.0145 ± 0.0369 |
| B − B-static | −0.048 ± 0.140 | +0.144 ± 0.938 | +0.0222 ± 0.0327 |
| B − B-target | −0.006 ± 0.139 | +0.154 ± 0.914 | +0.0034 ± 0.0224 |
| B-target − B-static | −0.042 ± 0.139 | −0.010 ± 0.862 | +0.0188 ± 0.0208 |
| A-wide − UNI | −0.012 ± 0.204 | −0.705 ± 1.612 | −0.0090 ± 0.0323 |
| B − UNI | +0.020 ± 0.178 | +0.070 ± 1.092 | +0.0055 ± 0.0379 |

完整九组配对、全部九个预先指定 comparisons、各数据集分层统计都在 [paired_attribution_by_run.csv](data/paired_attribution_by_run.csv) 与 [paired_attribution_summary.csv](data/paired_attribution_summary.csv)。B − A-wide 的 Macro-F1 信号没有伴随 Accuracy 改善，且 CE 方向更差；同时 B-static/B-target 的 Macro-F1 已接近 B。因此这不是 edge-specific c 的清楚证据。

B-static/B-target/B 的 Accuracy 按数据集汇总如下；逐 seed 数据保留在 full comparison 与 paired CSV。

| 数据集 | B-static | B-target | B |
|---|---:|---:|---:|
| Movies | 55.60% | 55.48% | 55.56% |
| Grocery | 82.21% | 82.28% | 82.25% |
| ele-fashion | 87.48% | 87.40% | 87.34% |

## 旧 M0-B checkpoint attribution

每项 intervention 在同一旧 B checkpoint、同一 H0、同一 classifier 上执行，只评估 validation。下表 mean ± population SD 是先在每个 checkpoint 内平均 shuffle seeds 1001–1005，再汇总九个 checkpoint；repeat-level SD 和逐 checkpoint 结果另存于 [b_intervention_summary.csv](data/b_intervention_summary.csv)。

| Intervention | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---:|---:|---:|
| function-off | −0.443 ± 0.340 | −0.855 ± 0.492 | +0.00934 ± 0.01086 |
| g-shuffle-only | −0.063 ± 0.093 | −0.084 ± 0.143 | +0.000876 ± 0.000693 |
| c-shuffle-only | −0.030 ± 0.044 | −0.010 ± 0.080 | +0.000138 ± 0.000429 |
| c-target-mean | −0.046 ± 0.061 | −0.029 ± 0.104 | +0.000069 ± 0.000330 |
| c-global-mean | −0.170 ± 0.259 | −0.293 ± 0.338 | +0.000096 ± 0.004253 |
| historical tuple shuffle `(g,c)` | −0.100 ± 0.114 | −0.116 ± 0.149 | +0.001474 ± 0.001253 |

Function-off lowered Accuracy and Macro-F1 in 9/9 old B checkpoints; CE worsened in 7/9. This supports that the trained B correction branch is used. It does not isolate adaptive c. The c-only interventions have tiny mean effects; c-target-mean is nearly a no-op at the task level. c-global-mean is somewhat larger but still small and mostly affects Accuracy/Macro-F1 rather than CE. The isolated g shuffle is larger than the c-only shuffle on each metric, and the tuple-shuffle effect is closer to the g effect; this is descriptive because interventions interact nonlinearly.

Historical reproduction was checked on normal, function-off, and all five tuple-shuffle repeats for all nine B checkpoints: **189 metric values** match the committed M0 records with maximum absolute difference `3.58e-7`. [m0_intervention_reproduction.csv](data/m0_intervention_reproduction.csv) contains each comparison. The new B-only intervention table is [b_intervention_by_run.csv](data/b_intervention_by_run.csv).

## Correction controls and c granularity

- **B-static:** learned scalar c averaged 0.1197 (Text) and 0.1202 (Visual), range 0.1174–0.1219. Its mean correction/default message norm ratio was about 0.0150/0.0168. Function-off on the independently trained B-static checkpoint was effectively neutral (Δ Accuracy −0.03 pp; Δ Macro-F1 −0.075 pp). This control tests a global linear neighbor transform/parameterization; it is not a new adaptive propagation function.
- **B-target:** target-level and edge-expanded c were saved separately. The maximum within-target deviation was exactly 0 across the nine runs, as required. Its correction/default norm ratio averaged about 0.156 (Text) and 0.138 (Visual); function-off lowered Accuracy by 0.350 pp and Macro-F1 by 0.748 pp on average. Despite using correction, B-target did not improve over B-static or B on the main metrics. c-target-mean is a prediction-level identity for this variant.
- **A-wide:** trained gate means averaged 0.886 (Text) and 0.926 (Visual); the fraction above 0.9 averaged 0.812/0.849. Gate diagnostics include quantiles, saturation fractions, modality gap, within-node spread, and extent-off/g-shuffle-only/modality-tied effects.

For original B, within-node `std_j(c_ij)` was computed on validation targets with indegree ≥5. Across runs, the mean of run-level target-standard-deviation quantiles was:

| Modality | Eligible targets/run (mean) | Mean | q10 | q25 | Median | q75 | q90 | Median within-target q90−q10 range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 1,652 | 0.116 | 0.009 | 0.028 | 0.104 | 0.193 | 0.249 | 0.144 |
| Visual | 1,652 | 0.136 | 0.010 | 0.034 | 0.134 | 0.223 | 0.273 | 0.196 |

The nonzero within-target variation is real in the learned controls, but the c-only interventions show little corresponding downstream effect. The descriptive one-way variance decomposition assigns about **63.9% (Text) / 64.7% (Visual)** to between-target variation and **36.1% / 35.3%** to within-target variation. These are descriptive fractions, not causal quantities. Per-run diagnostics and c quantiles are in [within_node_c_variation.csv](data/within_node_c_variation.csv) and [c_variance_decomposition.csv](data/c_variance_decomposition.csv).

## 按附件逐项回答

1. **A-wide 参数匹配：是。** A-wide 与 B 的总模型参数差 10；同 dataset × seed 的 total trainable 参数差也为 10，满足 ≤32。
2. **A-wide 是否明显改善 A：没有一致改善。** Accuracy +0.084 pp，Macro-F1 −0.463 pp，CE 降 0.013；方向混合且幅度小。
3. **B 相比 A-wide 的 function-specific 增量：部分指标有小信号，整体未确认。** Macro-F1 +0.776 pp，但 Accuracy +0.032 pp、CE +0.0145（较差）。
4. **B-static 与 B：近似。** B 的 Macro-F1 仅高 0.144 pp；B-static Accuracy 高 0.048 pp、CE 低 0.0222。
5. **B-target 与 B-static：近似。** Accuracy 差 −0.042 pp、Macro-F1 差 −0.010 pp；CE 高 0.0188。没有稳定 target-level 优势。
6. **B 与 B-target：近似。** B 的 Macro-F1 高 0.154 pp，Accuracy 低 0.006 pp，CE 高 0.0034。
7. **最有用的 correction granularity：无法由任务指标区分。** 没有清楚证据表明 target 或 edge 粒度优于 static/global；这表示当前 learner 未展示更细粒度的增量，不表示最优 granularity 已被证明为 global。
8. **B 的 within-target c variance：存在。** degree≥5 validation targets 上，两种模态的 std 分布都明显非零；但 variance 不自动代表 task utility。
9. **c variation 的主要来源：between-target。** 方差比例约 64% between、36% within。
10. **g-shuffle-only：小幅损害。** −0.063 pp Accuracy、−0.084 pp Macro-F1、CE +0.000876。
11. **c-shuffle-only：近乎无影响。** −0.030 pp Accuracy、−0.010 pp Macro-F1、CE +0.000138。
12. **tuple shuffle 更像由 g 驱动。** g-only 的平均影响大于 c-only；tuple shuffle 的影响接近 g-only。这里仅作描述性比较。
13. **c-target-mean：几乎不伤害 B。** −0.046 pp Accuracy、−0.029 pp Macro-F1。
14. **c-global-mean：小幅影响。** −0.170 pp Accuracy、−0.293 pp Macro-F1，CE 基本不变。
15. **重新解释 function-off：**它说明旧 B checkpoint 使用了 correction branch 的非零贡献；结合 B-static/B-target 和 c-only 干预，不能把它解释为 edge-conditioned c 对应关系有用。B-target checkpoint 也会依赖其 correction branch，但没有得到更好的主任务分数。
16. **B 的小增益可能只是 capacity/optimization：是，仍然成立。** 参数匹配的 A-wide 与 B 相近，且 B 对 B-static/B-target 无稳定全面优势。
17. **“edge-conditioned correction is task-useful beyond extent-only”：PARTIALLY。** 有 Macro-F1 的小幅 paired gain；Accuracy/CE 不一致，且 c-only 干预没有支持 edge correspondence 的 downstream value。
18. **“同一 target 内不同 physical relation 受益于不同 c”：NO。** 虽有 nonzero c variation，但 c shuffle/target mean 影响极小，B 与 B-target 也近似。
19. **决策：`CAPACITY_EXPLANATION_REMAINS`。** 当前 M0-B 不应被表述为 edge-specific functional adaptation 已获支持。
20. **下一步建议（仅建议，不实现）：**人工审阅后，若继续研究，优先重审 relation evidence/executor 如何将 within-neighborhood function heterogeneity 转成可学、可验证的信号；不要以增加传播深度替代归因。本报告不进入 M1，也不实现下一版模型。

## 自我审查

- **Capacity confound：**以 A-wide 直接控制，参数差 10；任务差异依指标而变，capacity/optimization 解释仍未排除。
- **Branch utility 与 adaptive-c utility：**function-off 只支持旧 checkpoint 的 branch reliance；c-shuffle/c-target-mean 几乎无影响，二者未混为一谈。
- **Global/target/edge：**通过 B-static、B-target、B 训练对照及原 B 的 global/target/shuffle inference intervention 分开检查。
- **Intervention 与训练基线：**报告明确区分固定 checkpoint 干预与独立训练的 B-static/B-target。
- **Problem 与 solution evidence：**M0.1 只检验当前 learner；结果不推翻 P0–P1.3 的经验问题证据。
- **实现公平：**五模型 common parameters bitwise equal；B/B-target correction head、B/B-static/B-target basis 初始化一致；B-target 与 B 参数相同；B-static 少 128；没有修改 M0-B 公式或 `research/m0_adaptive_propagation_screen/` 历史产物。

## 图表与质量检查

图形的核心 claim 是：capacity-matched A-wide 后，B 没有跨指标的一致优势，且 c 的细粒度变化未显示明显 downstream value。Performance 面板显示七种主要方法的 mean ± population SD（n=9 dataset × seed）；capacity/granularity 图按数据集展示同 seed 轨迹；intervention 图汇总九个旧 B checkpoint 的 paired Accuracy 变化。所有 quantitative panels 都对应 `data/` 中的 CSV。

四张图均导出 600-dpi PNG、LZW TIFF 与 editable PDF/SVG；多面板对齐审计 PASS（1.5 pt tolerance），PDF 最小字 6.2 pt，碰撞审计均为 0 fail/0 warn。PNG/PDF/SVG/TIFF 与 alignment/collision QA 文件位于 [figures/](figures/)。
