# RM-CGE V1 首轮 Pilot 报告

## 结论摘要

按 Implementation Freeze 提交 `8b94d16c3b2a66b658419c7f79e0de688b51a23a` 完成三次 seed-42 Full-Graph NC 正式训练：Movies、Grocery、ele-fashion 各一次。三次均正常早停、checkpoint 有限且可复算验证指标；没有发生 OOM 或训练中断。模型选择只依据 Validation Accuracy，Test 评估关闭，运行记录和诊断都标记 `test_metrics_computed=false`、`test_labels_indexed=false`。

| 数据集 | Val Accuracy | Val Macro-F1 | 最佳 epoch / 完成 epoch | 训练时间 | 峰值显存 allocated / reserved |
|---|---:|---:|---:|---:|---:|
| Movies | 0.540492 | 0.429959 | 70 / 100 | 18.08 s | 1.002 / 1.274 GiB |
| Grocery | 0.823133 | 0.752066 | 91 / 121 | 20.90 s | 1.026 / 1.306 GiB |
| ele-fashion | 0.879309 | 0.741882 | 115 / 145 | 118.48 s | 5.456 / 6.406 GiB |

三组任务统一使用 `unified_full_graph_nc_v1`、AdamW、lr `1e-3`、weight decay `1e-4`、最多 300 epochs、patience 30、最早 epoch 30、gradient clip 1.0；每 epoch 记录 Train Loss、Val Accuracy、Val Macro-F1、耗时和学习率。模型固定为 128 维、两层、dropout 0.2、每层三个 `4d→64→d` 专家、残差尺度 0.5。没有调参或根据分数修改结构。

| 数据集 | 图节点 | 模型参数 | 分类头参数 | Cache / config SHA256 |
|---|---:|---:|---:|---|
| Movies | 16,672 | 892,242 | 2,580 | 见 `data/pilot_runs.csv` |
| Grocery | 17,074 | 892,242 | 2,580 | 见 `data/pilot_runs.csv` |
| ele-fashion | 97,766 | 761,170 | 1,548 | 见 `data/pilot_runs.csv` |

所有训练由 RTX 3090（物理 GPU 1）执行。Runner 启动前检查设备空闲量并只暴露被选 GPU；占用 GPU 0 的外部任务未被停止。完整环境快照、源提交、配置指纹、缓存指纹、耗时和显存数值在 `data/manifest.json`、`data/environment.json`、`data/pilot_runs.csv`、`data/resource_profile.csv`。

## 训练轨迹与数值状态

逐 epoch 完整轨迹保存于 `data/training_history.csv`。Train Loss 从首轮到末轮分别为：Movies `3.2587→0.3470`、Grocery `3.4181→0.1017`、ele-fashion `2.6711→0.2001`。选中 checkpoint 的重新计算验证指标与训练器记录相符。训练、checkpoint 参数、融合输出和实现预检中的梯度均为有限值；没有看到发散或数值异常。

Movies 的验证表现明显弱于另两个数据集；ele-fashion 的 Accuracy 与 Macro-F1 相差约 13.74 个百分点，说明总体准确率没有概括所有类别的表现。当前报告没有 class-wise 指标，也只有一个随机种子，不能据此定位类别原因或判断方差。结果是实现可运行性的首轮证据，不是 SOTA 或统计显著性结论。

## 支持集、路由和专家使用

三个数据集的物理图有效节点比例均约为 100%；ele-fashion 有 97,763/97,766 个节点拥有物理邻居。Text 和 Visual 语义支持的有效节点比例在三个数据集均为 100%。下表给出每一层、每个目标模态的 joint router 关系边缘概率、三个专家联合使用比例、路由熵（自然对数单位）及残差更新 RMS 相对当前状态 RMS。概率来自六个关系—专家候选的同一次联合 Softmax。

### Movies

| 层 / 目标模态 | Physical / Semantic 概率 | Expert 0 / 1 / 2 使用比例 | 路由熵 | 相对更新 RMS |
|---|---:|---:|---:|---:|
| 1 / Text | 0.643 / 0.357 | 0.399 / 0.295 / 0.306 | 1.690 | 0.159 |
| 1 / Visual | 0.636 / 0.364 | 0.432 / 0.275 / 0.294 | 1.680 | 0.174 |
| 2 / Text | 0.656 / 0.344 | 0.491 / 0.292 / 0.217 | 1.560 | 0.181 |
| 2 / Visual | 0.642 / 0.358 | 0.392 / 0.335 / 0.272 | 1.504 | 0.192 |

### Grocery

| 层 / 目标模态 | Physical / Semantic 概率 | Expert 0 / 1 / 2 使用比例 | 路由熵 | 相对更新 RMS |
|---|---:|---:|---:|---:|
| 1 / Text | 0.679 / 0.321 | 0.466 / 0.201 / 0.333 | 1.235 | 0.298 |
| 1 / Visual | 0.648 / 0.352 | 0.239 / 0.361 / 0.400 | 1.594 | 0.248 |
| 2 / Text | 0.672 / 0.328 | 0.301 / 0.232 / 0.467 | 1.245 | 0.259 |
| 2 / Visual | 0.671 / 0.329 | 0.294 / 0.477 / 0.229 | 1.282 | 0.269 |

### ele-fashion

| 层 / 目标模态 | Physical / Semantic 概率 | Expert 0 / 1 / 2 使用比例 | 路由熵 | 相对更新 RMS |
|---|---:|---:|---:|---:|
| 1 / Text | 0.597 / 0.403 | 0.203 / 0.333 / 0.464 | 1.068 | 0.259 |
| 1 / Visual | 0.615 / 0.385 | 0.510 / 0.418 / 0.072 | 1.079 | 0.236 |
| 2 / Text | 0.565 / 0.435 | 0.430 / 0.169 / 0.401 | 1.006 | 0.218 |
| 2 / Visual | 0.624 / 0.376 | 0.181 / 0.572 / 0.247 | 1.339 | 0.208 |

总体上，Physical 路由在所有数据集和层/模态组合中都高于 Semantic，但 Semantic 仍获得约 32%–44% 的候选概率；没有出现语义关系被全局关闭。三个专家都参与路由。较明显的局部偏斜是 ele-fashion 第 1 层 Visual 视图的 Expert 2（7.2%）；该专家在其他视图/层仍有质量，第 2 层 Visual 中占 24.7%，因此这是局部使用偏低而不是整个 bank 的专家坍塌。Expert 输出 RMS 范围为 Movies `0.423–0.554`、Grocery `0.621–0.782`、ele-fashion `0.513–0.720`；成对输出余弦均值范围分别为 `0.029–0.159`、`0.055–0.225`、`0.036–0.161`，没有证据表明各专家输出完全相同。逐关系、逐专家的明细见 `data/model_diagnostics.csv` 和各 dataset 的 JSON 诊断文件。

## 跨模态上下文、关系贡献与融合路径

按关系和视图统计的跨模态上下文 RMS / 同模态上下文 RMS 范围为：Movies `0.217–0.392`、Grocery `0.201–0.278`、ele-fashion `0.212–0.309`。跨模态信号幅度非零且没有压过同模态上下文。层/模态残差更新 RMS 比例分别落在 Movies `0.159–0.192`、Grocery `0.248–0.298`、ele-fashion `0.208–0.259`，显示图专家分支参与了状态更新，同时更新规模低于当前状态的 RMS。

| 数据集 | Graph term RMS | Intrinsic skip RMS | Graph / skip RMS 比 | 关闭 Cross-modal 后 Val Acc / Macro-F1 变化 |
|---|---:|---:|---:|---:|
| Movies | 0.852 | 0.664 | 1.283 | -1.470 / -3.161 pp |
| Grocery | 1.133 | 0.683 | 1.660 | -3.514 / -2.957 pp |
| ele-fashion | 1.070 | 0.786 | 1.362 | -0.624 / -0.905 pp |

Graph term RMS 在三个数据集都大于 intrinsic skip RMS；冻结关闭 skip 后 ele-fashion Accuracy 下降 9.256 pp，说明其预测仍明显依赖初始投影特征路径。关闭图更新也使三个数据集的 Val Accuracy 分别下降 2.999、9.312、2.076 pp，表明图分支对选中模型有可测贡献。

| 冻结前向干预 | Movies Acc / F1 变化 | Grocery Acc / F1 变化 | ele-fashion Acc / F1 变化 |
|---|---:|---:|---:|
| Cross-modal 关闭 | -1.470 / -3.161 pp | -3.514 / -2.957 pp | -0.624 / -0.905 pp |
| Semantic 关系关闭 | -0.270 / -0.598 pp | 0.000 / -0.705 pp | -1.125 / -0.807 pp |
| Physical 关系关闭 | -2.999 / -3.357 pp | -6.149 / -6.160 pp | -0.859 / -2.040 pp |
| 所有图更新关闭 | -2.999 / -6.567 pp | -9.312 / -8.257 pp | -2.076 / -3.342 pp |
| Intrinsic skip 关闭 | -3.929 / -6.100 pp | -2.958 / -4.679 pp | -9.256 / -7.126 pp |

这些是保持已训练权重和分类头不变的验证前向路径干预，不能解释为重新训练后的消融或因果贡献。Movies 和 Grocery 对 Physical 路径更敏感；Grocery 的 Semantic 关闭不改变 Accuracy，但 Macro-F1 下降 0.705 pp。ele-fashion 对两类关系的关闭都产生小幅下降。所有关系、专家、上下文 RMS、预测翻转比例、logit 变化和 embedding 变化保存在 `data/diagnostics/*_seed_42.json`。

## 实现核对与证据边界

- 每个 cache 均校验 feature、candidate 和 V7B 指纹；缓存使用 K=8 结果，没有重跑 FAISS。来源位 0/1/2 按 `flags[:, modality] > 0` 解读，Both 边在两种视图各保留完整边。Cache 不包含标签，候选生成阶段标签访问标记为 false。
- 算子单元测试、Cross-modal support 和 mask 测试、共享专家/路由对象检查、联合 masked-softmax 测试、第二层状态流测试全部通过。完整 ele-fashion 工程预检完成全图 forward/backward/AdamW update 和 checkpoint reload；所有核心参数组具有有限非零梯度并发生真实更新。
- PACE-MAG V1、其机制审计测试、PSCE V7A/V7B/V7B.1 回归测试通过；代码测试总计 **101 passed, 1 skipped**。实现冻结前详细记录见 `IMPLEMENTATION_AUDIT.md` 和 `data/implementation_checks.json`。
- 三个 checkpoint 均以 Validation Accuracy 选择，重算 Validation Accuracy/Macro-F1 与运行指标一致；没有 Test 指标或 Test 标签索引。
- 单种子、单次训练不能衡量方差。冻结前向分支关闭会改变归一化后的整条前向路径，结果只能说明已拟合模型对该信息路径的敏感性。跨方法历史分数若用于背景比较也不是严格配对实验。
- 本轮没有运行第二种子、额外数据集、Test、自动搜索或新架构；不从这些 Pilot 指标提出优越性或新颖性结论。与 MvCGE / ReCoG 的关系和潜在重合风险见 `DESIGN.md`。

## 可复审文件

- `data/pilot_runs.csv`：三次训练汇总、最佳 epoch、参数量、运行时间、峰值显存、config/cache 指纹和 checkpoint/log 路径。
- `data/training_history.csv`：全部 366 个训练 epoch 的逐 epoch 指标。
- `data/model_diagnostics.csv`：逐层、逐模态、逐关系诊断表。
- `data/diagnostics/`：包含支持比例、冻结分支干预、预测变化和详细机制指标的 JSON。
- `data/resource_profile.csv`、`data/manifest.json`、`data/environment.json`：运行资源、固定协议、环境和 Git 来源。
- `data/cache_audit.json`、`data/gradient_audit.json`、`data/implementation_checks.json`：缓存与实现预检证据。

训练 checkpoint、resolved config 和逐 epoch 原始日志保留在本地被 `.gitignore` 忽略的 `outputs/rmcge_v1_pilot/` 下；没有把特征、标签、全量表示或大型 checkpoint 提交到 Git。
