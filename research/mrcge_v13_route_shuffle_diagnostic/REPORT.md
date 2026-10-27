# MR-CGE V1.3-A Frozen Router-Shuffle 与专家差异诊断

## 范围与完整性

状态：**complete**。只分析 Movies/Grocery seed42 的既有 `mrcge_v13 / M_repaired / joint_view` Best-Validation-Accuracy checkpoint。未重新训练、未执行 Test 指标、未运行 LP。

计划包含 2 个无干预 baseline 与 24 个 Router-Shuffle 推理；另有每个数据集一次 pass-through Hook 控制，用来验证 Hook 不改变 baseline。所有干预仅置换同一视图内 Support=True 节点的四维 Router-logit 行；每个视图使用独立 CPU Generator。

## Baseline 复现

| Dataset | Val Acc | Macro-F1 | CE | 历史差异最大值 | Hook 控制一致 |
|---|---:|---:|---:|---:|---|
| Movies | 55.969% | 48.198% | 1.37037 | 0 | True |
| Grocery | 83.514% | 76.621% | 0.63923 | 0 | True |

## 专家最终输出差异

以下输出 RMS 差异使用 `RMS(Ea−Eb) / mean(RMS(Ea), RMS(Eb))`；cosine 只统计 Support=True 节点。Flattened cosine 在所有有效节点与通道上展平。它们描述表示差异，不赋予专家明确物理/频率语义。

| Dataset | Mean pair node cosine | Mean flattened cosine | Mean normalized RMS difference |
|---|---:|---:|---:|
| Movies | 0.2185 | 0.2185 | 1.1638 |
| Grocery | 0.1204 | 0.1205 | 1.2503 |

每视图的专家 RMS 与六对逐节点/全局 cosine、归一化 RMS 差异详见 `data/expert_diversity.csv`。路由选择份额、Top-2 组合数与 importance 见 `data/router_summary.csv`。

## 冻结 Router-Shuffle 变化

| Dataset | Condition | ΔAcc mean [min,max] pp | ΔMacro-F1 mean [min,max] pp | ΔCE mean [min,max] | 预测改变率 mean | Logit MAE mean |
|---|---|---:|---:|---:|---:|---:|
| Movies | shuffle_private_physical | -0.040 [-0.090, +0.000] | -0.099 [-0.125, -0.073] | -0.00019 [-0.00033, -0.00004] | 0.58% | 0.00907 |
| Movies | shuffle_private_semantic | -0.100 [-0.270, +0.090] | -0.558 [-0.896, -0.313] | +0.00275 [+0.00139, +0.00401] | 3.29% | 0.08075 |
| Movies | shuffle_collaborative_physical | -2.020 [-2.729, -1.440] | -2.183 [-2.595, -1.526] | +0.04801 [+0.04473, +0.05091] | 14.40% | 0.36273 |
| Movies | shuffle_all_six | -1.920 [-2.340, -1.410] | -2.425 [-2.508, -2.270] | +0.05172 [+0.04808, +0.05406] | 14.82% | 0.37995 |
| Grocery | shuffle_private_physical | +0.000 [-0.088, +0.088] | -0.053 [-0.175, +0.076] | -0.00036 [-0.00044, -0.00026] | 0.24% | 0.01579 |
| Grocery | shuffle_private_semantic | -0.371 [-0.381, -0.351] | -0.040 [-0.072, -0.014] | +0.00690 [+0.00656, +0.00755] | 2.41% | 0.14337 |
| Grocery | shuffle_collaborative_physical | -1.640 [-1.816, -1.493] | -1.412 [-1.802, -1.057] | +0.03436 [+0.03327, +0.03636] | 5.75% | 0.37314 |
| Grocery | shuffle_all_six | -1.981 [-2.284, -1.786] | -1.713 [-1.999, -1.533] | +0.04504 [+0.04292, +0.04653] | 6.87% | 0.41460 |

## 解释与建议

专家输出是否不同、冻结节点路由是否影响验证预测、以及是否保留节点 Router 是三个不同判断。Shuffle 是对已训练 checkpoint 的推理敏感性诊断，不是重训静态 Router、因果识别、显著性检验或 Test 泛化证据。

建议：**`KEEP_CURRENT_ROUTER`**。此建议仅供下一轮人工研究判断；本任务不据此修改模型。

## 核验与偏差

- Baseline 与历史 runs.csv 的 Accuracy/Macro-F1/CE 容差为 3e-6；先做无 Hook，再做 pass-through Hook 对照。Baseline 不匹配的数据集不会继续 Shuffle。
- 每次 Hook 都检查六次调用顺序、实际 modality/relation ID、Physical/Semantic Support Mask 与 hop；Hook 在 `finally` 中移除。
- 每个目标视图检查非恒等节点行置换、Logit 行多重集不变、Top-2 专家计数/组合不变、softmax importance 与 Top-2 权重统计不变；非目标视图 logits 必须逐元素不变。
- checkpoint/model/classifier 哈希与 RNG 在诊断前后核验；Test 标签为 -100，Test evaluation=false。

- 两个 checkpoint 的记录模型源码 SHA 与当前源码不同；strict state-dict load 通过，且两组 Baseline 指标均逐值复现历史记录。源码 provenance 差异仍记录为偏差。
- 首轮试跑完成 5 个单元后暴露 FP32 汇总容差与失败收尾问题；修复后从头执行全部 26 个单元。首轮部分结果未混入最终 CSV，过程记录见 Manifest。
- 本轮 GPU 守卫按用户明确指示绕过；GPU、空闲显存与启动快照记录于 Manifest。

实际 GPU：1。资源守卫状态：bypassed_by_user。执行 Commit：`9fae080b467a6120373f27172393a528d3f4ae40`。

单元完成/失败/未启动数量及其原因见 `data/manifest.json`；干预逐排列数据见 `data/interventions.csv`。
