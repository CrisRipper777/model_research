# MR-CGE V1.3-A Frozen Router-Shuffle 与专家差异诊断

## 范围与完整性

状态：**blocked_resource**。只分析 Movies/Grocery seed42 的既有 `mrcge_v13 / M_repaired / joint_view` Best-Validation-Accuracy checkpoint。未重新训练、未执行 Test 指标、未运行 LP。

计划包含 2 个无干预 baseline 与 24 个 Router-Shuffle 推理；另有每个数据集一次 pass-through Hook 控制，用来验证 Hook 不改变 baseline。所有干预仅置换同一视图内 Support=True 节点的四维 Router-logit 行；每个视图使用独立 CPU Generator。

## Baseline 复现

两个数据集均未执行 Baseline Forward，因此历史指标复现状态为**未确认**；所有指标表保留空值，不能据此比较 Shuffle 效应。CPU 预检已通过，两个 checkpoint 均可严格加载，但 checkpoint provenance 中的模型源码 SHA256 (`a70e2fd4…`) 与当前源码 (`be9272ad…`) 不同，故当前代码下的前向等价性仍待 Baseline 验证。

| Dataset | Val Acc | Macro-F1 | CE | 历史差异最大值 | Hook 控制一致 |
|---|---:|---:|---:|---:|---|

## 专家最终输出差异

以下输出 RMS 差异使用 `RMS(Ea−Eb) / mean(RMS(Ea), RMS(Eb))`；cosine 只统计 Support=True 节点。Flattened cosine 在所有有效节点与通道上展平。它们描述表示差异，不赋予专家明确物理/频率语义。

| Dataset | Mean pair node cosine | Mean flattened cosine | Mean normalized RMS difference |
|---|---:|---:|---:|
| Movies | 未运行 | 未运行 | 未运行 |
| Grocery | 未运行 | 未运行 | 未运行 |

每视图的专家 RMS 与六对逐节点/全局 cosine、归一化 RMS 差异详见 `data/expert_diversity.csv`。路由选择份额、Top-2 组合数与 importance 见 `data/router_summary.csv`。

## 冻结 Router-Shuffle 变化

| Dataset | Condition | ΔAcc mean [min,max] pp | ΔMacro-F1 mean [min,max] pp | ΔCE mean [min,max] | 预测改变率 mean | Logit MAE mean |
|---|---|---:|---:|---:|---:|---:|
| Movies | shuffle_private_physical | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| Movies | shuffle_private_semantic | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| Movies | shuffle_collaborative_physical | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| Movies | shuffle_all_six | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| Grocery | shuffle_private_physical | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| Grocery | shuffle_private_semantic | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| Grocery | shuffle_collaborative_physical | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| Grocery | shuffle_all_six | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |

## 解释与建议

专家输出是否不同、冻结节点路由是否影响验证预测、以及是否保留节点 Router 是三个不同判断。Shuffle 是对已训练 checkpoint 的推理敏感性诊断，不是重训静态 Router、因果识别、显著性检验或 Test 泛化证据。

建议：**`INCONCLUSIVE`**。此建议仅供下一轮人工研究判断；本任务不据此修改模型。

## 核验与偏差

- Baseline 与历史 runs.csv 的 Accuracy/Macro-F1/CE 容差为 3e-6；先做无 Hook，再做 pass-through Hook 对照。Baseline 不匹配的数据集不会继续 Shuffle。
- 每次 Hook 都检查六次调用顺序、实际 modality/relation ID、Physical/Semantic Support Mask 与 hop；Hook 在 `finally` 中移除。
- 每个目标视图检查非恒等节点行置换、Logit 行多重集不变、Top-2 专家计数/组合不变、softmax importance 与 Top-2 权重统计不变；非目标视图 logits 必须逐元素不变。
- checkpoint/model/classifier 哈希与 RNG 在诊断前后核验；Test 标签为 -100，Test evaluation=false。

GPU 启动守卫先在 GPU 1 利用率 87%、空闲显存 18150 MiB、温度 73°C 时通过；工作进程执行第二次启动前复核时读到 GPU 1 利用率 98%，守卫拒绝继续。工作进程在载入数据及模型 Forward 之前退出；没有 Baseline、Hook 控制或 Shuffle 推理。随后最后一次成功的资源快照为 GPU 0/1 利用率 99%/91%；13:07:31 UTC 的后续 `nvidia-smi` 查询无法连接 NVIDIA 驱动，因此未再次尝试。资源状态：`blocked_resource`；26 个正式 Forward 单元全部未启动。执行 Commit：`272f94894389b06eb9d051051c012b9926f77bb1`。

因此，专家最终输出差异、路由 Shuffle 对验证预测的影响及路由分布 Hook 核验均**没有实验结果**。模型源码 SHA 不一致也仍需通过历史 Baseline Forward 判断是否影响结果；当前不对 Router 是否应保留作实证结论，建议维持 `INCONCLUSIVE`。

单元完成/失败/未启动数量及其原因见 `data/manifest.json`；干预逐排列数据见 `data/interventions.csv`。
