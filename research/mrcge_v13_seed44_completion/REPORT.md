# MR-CGE V1.3-A 主模型 Seed44 基线补齐

## 范围与协议

本补充实验仅完成 Grocery/seed44 与 ele-fashion/seed44 两个节点分类单元。正式模型为 `mrcge_v13`、`M_repaired`、`joint_view`、`sender_private_ratio=0.0`；使用原 seed42 节点划分，训练随机种子为 44。原编码器、NC Trainer 与历史运行文件未修改。

所有主结果来自 Best-Validation-Accuracy checkpoint。Best-F1 checkpoint 单独保存和核验，仅作为选择记录；它的指标未混入主结果。Private CE 权重为 0.3，Balance Loss 权重为 0.01，Trainer `aux_weight=1.0`。Test evaluation 关闭，进入 Trainer 前 Test 标签设为 `-100`。

训练配置：Physical/Semantic hops=3/1，4 experts、Top-K=2、hidden=256，V7B.1 semantic KNN K=8，AdamW (LR=0.001, weight decay=0.0001)，dropout=0.2/0.15，300 epoch 上限、patience=30。

## 九个单元

Accuracy 与 Macro-F1 以百分比表示；Gap 定义为 Train − Validation。七条历史记录是复用既有 checkpoint 元数据，不是本次重训。

| Dataset | Seed | 来源 | Val Acc | Val Macro-F1 | Val CE | Acc Gap | Macro-F1 Gap | CE Gap | Best Acc/F1 epoch |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | 复用历史 | 55.97% | 48.20% | 1.3704 | +16.79 pp | +23.56 pp | -0.5824 | 74/90 |
| Movies | 43 | 复用历史 | 55.22% | 49.17% | 1.3798 | +19.55 pp | +26.03 pp | -0.6396 | 73/66 |
| Movies | 44 | 复用历史 | 55.85% | 48.03% | 1.3542 | +14.81 pp | +19.85 pp | -0.5143 | 66/83 |
| Grocery | 42 | 复用历史 | 83.51% | 76.62% | 0.6392 | +10.79 pp | +15.63 pp | -0.4243 | 74/73 |
| Grocery | 43 | 复用历史 | 83.69% | 76.94% | 0.6456 | +10.45 pp | +14.86 pp | -0.4266 | 74/87 |
| Grocery | 44 | 新训练 | 83.43% | 76.18% | 0.6432 | +10.57 pp | +15.05 pp | -0.4288 | 83/108 |
| ele-fashion | 42 | 复用历史 | 87.10% | 73.28% | 0.4011 | +2.60 pp | +8.67 pp | -0.1042 | 111/137 |
| ele-fashion | 43 | 复用历史 | 87.50% | 73.79% | 0.4039 | +1.93 pp | +6.53 pp | -0.1043 | 114/112 |
| ele-fashion | 44 | 新训练 | 87.54% | 74.79% | 0.4397 | +3.46 pp | +9.64 pp | -0.1872 | 153/178 |

## 各数据集 Mean ± SD

SD 使用三个训练种子的样本标准差（ddof=1）。

| Dataset | Val Acc | Val Macro-F1 | Val CE | Train Acc | Train Macro-F1 | Train CE | Acc Gap | Macro-F1 Gap | CE Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 55.68 ± 0.40% | 48.47 ± 0.61% | 1.3681 ± 0.0129 | 72.73 ± 2.05% | 71.62 ± 3.66% | 0.7893 ± 0.0499 | 17.05 ± 2.38% | 23.15 ± 3.11% | -0.5788 ± 0.0627 |
| Grocery | 83.54 ± 0.13% | 76.58 ± 0.38% | 0.6427 ± 0.0032 | 94.15 ± 0.15% | 91.76 ± 0.51% | 0.2161 ± 0.0025 | 10.60 ± 0.17% | 15.18 ± 0.40% | -0.4266 ± 0.0022 |
| ele-fashion | 87.38 ± 0.24% | 73.95 ± 0.77% | 0.4149 ± 0.0216 | 90.04 ± 0.84% | 82.23 ± 2.06% | 0.2830 ± 0.0264 | 2.66 ± 0.77% | 8.28 ± 1.59% | -0.1319 ± 0.0479 |

## 来源、检查与偏差

- 本次新训练：Grocery/44、ele-fashion/44；各自的配置快照、run metrics、Best-Accuracy 与 Best-F1 权重保存在服务器 `outputs/mrcge_v13_seed44_baseline_completion/`。权重不纳入 Git。
- 复用历史：Movies/Grocery/ele-fashion 的 seed42/43，来源为 `research/mrcge_v13/data/runs.csv`；Movies/seed44 来自 `research/mrcge_v13_physical_basis/data/fixed_split_hybrid_comparison.csv`。历史 Train 指标从各自 Best-Accuracy checkpoint 的 `epoch_history` 对应 epoch 读取，并与存档 Validation 指标交叉核对。
- seed44 Grocery 与 ele-fashion 的特征、物理边、固定 Split、V7B.1 Semantic Cache 指纹均与 V1.2 repaired 同 seed 参考记录一致；CPU 配置/数据/初始化检查和 GPU 单步前向反向预检通过。
- GPU 资源守卫按用户本轮明确指示绕过；此项仅为资源准入偏差，不改变训练/评估协议。
- Test 关闭；Test 标签被掩码；checkpoint 重载后的 Train/Validation 指标与 Trainer 选择元数据一致；Best-Accuracy 与 Best-F1 分别核验；所有报告指标为有限数值。
- 协议偏差：本次两个单元无已知偏差。七个历史单元按其原始研究记录复用；本报告不将它们表述为重新训练。
- 详细来源、文件指纹、checkpoint SHA256 与每个新单元的运行状态见 `data/runs.csv` 和 `data/manifest.json`。
