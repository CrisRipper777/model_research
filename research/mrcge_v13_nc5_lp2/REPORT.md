# MR-CGE V1.3-A：五数据集 NC 与两数据集 LP 汇总

**状态：** 完成。最终收集记录为 NC 15/15、LP 6/6；Test 未评估。

## 研究范围与结果来源

本轮评估冻结 MR-CGE V1.3-A Full `M_repaired` / `joint_view` 编码器。NC 包含 Movies、Grocery、Toys、Reddit-S、ele-fashion，各使用训练种子 42、43、44。Movies、Grocery、ele-fashion 的 9 个单元复用已完成的历史结果；Toys 与 Reddit-S 的 6 个单元为本轮新增训练。每个数据集的三个训练种子使用同一个 seed42 节点划分。NC 以 Best-Validation-Accuracy checkpoint 为主结果，Best-F1 checkpoint 单独保存。

LP 包含 sports-copurchase 与 cloth-copurchase，各使用种子 42、43、44。LP 采用全图三折正边留出协议 `mrcge_v13_full_graph_edge_holdout_lp_v1_official_split_as_is`，Best-Validation-MRR checkpoint 为主结果。此协议与仓库原有的 sampled LP 训练协议不同。

表中均为三个种子的均值 ± 样本标准差（SD，分母为 n−1）。逐 seed 结果、checkpoint 路径与哈希、数据指纹及历史尝试记录见 `data/nc_runs.csv`、`data/nc_new_runs.csv`、`data/lp_runs.csv` 和 `data/fingerprints.json`。

## 节点分类（NC）

| 数据集 | 结果来源 | Validation Accuracy | Validation Macro-F1 | Validation CE |
|---|---|---:|---:|---:|
| Movies | 复用历史结果 | 0.556789 ± 0.004028 | 0.484678 ± 0.006146 | 1.368133 ± 0.012950 |
| Grocery | 复用历史结果 | 0.835432 ± 0.001342 | 0.765778 ± 0.003813 | 0.642679 ± 0.003208 |
| Toys | 本轮新训练 | 0.800435 ± 0.000242 | 0.769927 ± 0.003275 | 0.686735 ± 0.001972 |
| Reddit-S | 本轮新训练 | 0.961518 ± 0.001011 | 0.927301 ± 0.005460 | 0.161392 ± 0.025348 |
| ele-fashion | 复用历史结果 | 0.873819 ± 0.002430 | 0.739549 ± 0.007657 | 0.414887 ± 0.021564 |

NC 使用历史全图训练协议，Test 标签在进入 Trainer 前屏蔽，`evaluate_test=false`。Test 指标均记为 `not_evaluated`。

## 链路预测（LP）

| 数据集 | Validation MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|
| sports-copurchase | 0.355888 ± 0.005952 | 0.200957 ± 0.003810 | 0.403092 ± 0.009053 | 0.706880 ± 0.012635 |
| cloth-copurchase | 0.262426 ± 0.004997 | 0.146702 ± 0.003859 | 0.281051 ± 0.005589 | 0.502903 ± 0.008828 |

LP 使用官方 `lp-edge-split.pt` 原文件，未重切分或修改。V7B.1、K=8 语义候选由全局节点属性生成；训练物理图由官方 Train 正边生成，训练阶段将当前折正边从传播图双向移除。Test 排名、Test 负例候选和 Test 指标均未读取或计算，Test 字段为 `not_evaluated`。

### 官方 LP split 的已知偏差

官方 split 存在自环正边以及按无向端点规范化后跨分区重叠。根据用户后续指示，保留官方 split 原样运行并记录偏差。因此，Train 图可能含有同时出现在 Validation 或 Test 分区的正边，Validation 指标**不是无泄漏的泛化估计**，也不应作为与 sampled LP 同协议的公平对照。

| 数据集 | Train / Validation / Test 正边数 | Train–Validation / Train–Test / Validation–Test 无向重叠数 |
|---|---:|---:|
| sports-copurchase | 356,202 / 45,557 / 37,413 | 8,264 / 6,621 / 2,035 |
| cloth-copurchase | 951,271 / 121,323 / 97,456 | 21,043 / 17,360 / 5,169 |

自环计数与更完整的分区审计见 `data/lp_split_preflight_audit.json`。LP 汇总应按“官方 split 原样下的全图三折验证结果”解释。

## 完整性与复现记录

- 最终 Manifest 为 `complete`：NC 15/15，其中 9 个历史复用、6 个新增训练；LP 6/6。
- 汇总选中的 21 个 NC/LP checkpoint 均存在，SHA256 与运行记录相符，且 checkpoint 重载核验通过。
- `data/protocol_audit.json` 状态为 `pass`，`issues` 为空。LP split 冲突作为已接受的协议偏差单独记录，不构成无泄漏声明。
- 各阶段早期失败或中断的 attempt 保留在 runs CSV / Manifest 中供审计；最终每个预期单元均有一个核验通过的成功结果。
- checkpoint 与大型模型权重保留在服务器 `outputs/`，不纳入 Git 提交。

## 机器可读结果

- NC 每数据集三种子统计：`data/nc_summary.csv`
- NC 全部运行记录：`data/nc_runs.csv`；本轮新训练 attempt：`data/nc_new_runs.csv`
- LP 每数据集三种子统计：`data/lp_summary.csv`
- LP 全部 attempt 与所选结果：`data/lp_runs.csv`
- 全局完成状态与指纹：`data/manifest.json`、`data/fingerprints.json`
- 协议检查与 LP split 审计：`data/protocol_audit.json`、`data/lp_split_preflight_audit.json`
