# MR-CGE V1.3-A Early Cross / Late Completion 核心结构消融

状态：**complete**。本轮仅训练 Movies/Grocery/ele-fashion × seed42 × 两种独立消融；没有训练新的 Full，没有 Test 评估或 LP。

所有变化均相对相同数据集、相同 seed42 的历史 Full Best-Validation-Accuracy 记录计算。ΔAccuracy 与 ΔMacro-F1 使用百分点；ΔCE 为绝对差。Full 是复用的历史结果，不是本轮重训。

| Dataset | Condition | Val Acc | Macro-F1 | CE | ΔAcc (pp) | ΔMacro-F1 (pp) | ΔCE | Epochs | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | Full (historical) | 0.559688 | 0.481978 | 1.370375 | — | — | — | — | — |
| Movies | no_early_cross | 0.553089 | 0.465243 | 1.377257 | -0.660 | -1.673 | +0.006883 | 100 | 57.0 |
| Movies | no_late_completion | 0.559388 | 0.476022 | 1.413819 | -0.030 | -0.596 | +0.043444 | 107 | 54.1 |
| Grocery | Full (historical) | 0.835139 | 0.766214 | 0.639235 | — | — | — | — | — |
| Grocery | no_early_cross | 0.834553 | 0.770337 | 0.640190 | -0.059 | +0.412 | +0.000955 | 109 | 62.1 |
| Grocery | no_late_completion | 0.835139 | 0.760311 | 0.649940 | +0.000 | -0.590 | +0.010705 | 116 | 58.1 |
| ele-fashion | Full (historical) | 0.871024 | 0.732843 | 0.401065 | — | — | — | — | — |
| ele-fashion | no_early_cross | 0.873990 | 0.748825 | 0.405515 | +0.297 | +1.598 | +0.004450 | 156 | 353.7 |
| ele-fashion | no_late_completion | 0.873581 | 0.737741 | 0.400655 | +0.256 | +0.490 | -0.000410 | 155 | 304.2 |

## 结构判断

- Early Cross: Movies: ΔAcc -0.660 pp, ΔF1 -1.673 pp, ΔCE +0.00688; Grocery: ΔAcc -0.059 pp, ΔF1 +0.412 pp, ΔCE +0.00095; ele-fashion: ΔAcc +0.297 pp, ΔF1 +1.598 pp, ΔCE +0.00445. 建议：**`INCONCLUSIVE`**。这是单 seed 架构筛查，需结合变化方向与幅度解释。
  移除后 Movies 的 Acc/F1 下降且 CE 上升；Grocery 变化很小且指标方向混合；ele-fashion 的 Acc/F1 上升但 CE 略升。没有跨数据集的一致收益或损害，结果不足以普遍支持保留或移除 Early Cross。
- Late Completion: Movies: ΔAcc -0.030 pp, ΔF1 -0.596 pp, ΔCE +0.04344; Grocery: ΔAcc +0.000 pp, ΔF1 -0.590 pp, ΔCE +0.01071; ele-fashion: ΔAcc +0.256 pp, ΔF1 +0.490 pp, ΔCE -0.00041. 建议：**`INCONCLUSIVE`**。这是单 seed 架构筛查，需结合变化方向与幅度解释。
  移除后 Movies 与 Grocery 的 F1 下降、CE 上升，Acc 仅小幅下降或持平；ele-fashion 三项指标略有改善。前两者提供有限的保留信号，但总体仍呈数据集依赖，不能据此普遍去留。

## 实现与协议核验

- `none` 兼容性、历史 Movies seed42 Checkpoint strict-load、两种消融结构检查与一次无更新的训练损失/反向传播检查记录于 Manifest。
- 训练从 seed42 随机初始化开始；未将 Full Checkpoint 用作参数初始化。Private CE、六视图 Balance Loss、Joint View、专家和 Router 均保留。
- 两类消融保留相同注册参数表，但按定义关闭的路径不会参与计算/梯度；参数量不等价于实际计算量。
- 固定 NC Split 与 V7B.1 Cache 指纹逐数据集核对；Test 标签为 -100，`evaluate_test=false`。

执行提交：`4d1229da7da5a4ddd3895f3398882b08e4375794`。GPU：1；守卫绕过：True。

首轮建议用于人工架构审查；不自动删除模块，也不构成多种子显著性证据。
