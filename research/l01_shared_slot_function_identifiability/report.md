# L0.1 — Independent Shared-Slot Function Identifiability Audit

## 结论摘要

**最终标签：`DATASET_DEPENDENT_MIXED`。** 在相同的 512D 输入槽位和唯一共享线性分类头下，单边 Smooth→AbsDiff/Product 替换通常能改变 logits，但大多数变化没有改变 argmax，概率与 JS 位移也较小。utility 对三个独立 head repeat 的秩次有较好复现性，跨 seed 的复现性明显下降。Movies 与 Grocery 上，当前 endpoint/local evidence 几乎不能预测 clean gain；ele-fashion 有较明显的总秩相关和 within-target 信号，但严格 correspondence shuffle 仍保留一部分信号，且加入 endpoint/local 信息没有稳定超过所有基线。因此目前不支持推进 edge-wise latent-function router，也不支持引入 stochastic latent-function distribution。

本轮只给出 task-specific 的 empirical output-separation 结果。JS 或 logit shift 不是 Manenti 等人理论 injectivity 的证明；正 gain 是在 frozen shared head 下定义的 validation counterfactual utility，不是 ground-truth edge role。P1.3 的观察应限定为 operator-block marginal utility heterogeneity；它与本轮 clean substitution utility 有中等且依数据集变化的对应，不能直接解释为已识别的 edge function preference。

## 协议与数据边界

- 从 `exp/l0_relation_function_learnability_audit` 的已核实 SHA `9ce5f723fda0bf17264671a1e20e78f043727498` 开始；未合并 main，也未修改历史阶段目录。
- Movies、Grocery、ele-fashion × seeds 42/43/44。使用冻结 P0/P1.3 的 128D `H0_T/H0_V`，directed message `j→i`，移除 self message，所有 S/D/P aggregation 固定除以原始物理入度。
- 唯一 predictor 是 `Linear(512, C)`，输入顺序 `[H0_T,H0_V,C_T,C_V]`；同一 Text/Visual structural slot 分别始终使用 `[256:384]` 与 `[384:512]` 权重。一个 head 同时训练全部九种 S/D/P 模态组合，objective 是同一 target × 9 representations 的平均 CE。
- 每 dataset/seed 在原始 `train_idx` 内按 80/20 做三个确定性分层 head_train/head_select repeats。validation labels 未用于 head fit、early stopping 或 checkpoint selection；仅用于之后构造 utility。test split labels、metrics 和样本未访问。
- learnability probes 的 outer fold 是 validation-derived utility rows 上的 dst-group CV，并不是原 NC test split。outer/inner 按 dst 分组；feature 与 target scaler 仅拟合 inner-train；严格 shuffle 在每个 dst 内分别打乱 inner-train 和 inner-validation 的完整四维 gain tuple，outer fold utility 不打乱且拟合时不可见。
- 运行设备为 CUDA 1（RTX 3090）。正式 campaign：27 shared heads、108 real Evidence MLP、81 strict-shuffle MLP、81 fixed-alpha Ridge、27 DirectState、162 frozen Q/R/U readouts，共 486 个正式 fit。Raw edge utilities/checkpoints 保存在 gitignored `outputs/l01_shared_slot_function_identifiability/`。

## 正确性与审计

| 检查 | 结果 |
|---|---|
| 新增与历史回归测试 | 105 passed；2 个依赖库 warning（PyG deprecation、单类别 sklearn 指标提示） |
| Movies/42 smoke | passed，84.2 s；head、9 combinations、四种替换、one-fold MLP/shuffle/Ridge、DirectState、Frozen U 均覆盖 |
| H0 regression | Movies/42 `val_acc`, `val_macro_f1`, `val_ce` 与 P0 checkpoint 完全相同；对 P0 CSV 的最大绝对差 `2.22e-16`；128D、参数冻结 |
| P1.3 operator/context alignment | edge 顺序、`src/dst/degree` 与 P1.3 对齐；self-loop 数为 0；S/D/P whole-context 回归通过 |
| Shared-slot identity | feature dim 512；一个 trainable Linear；无 operator-specific head 参数；structural Text/Visual weight norms 均非零 |
| Validation/test 边界 | 27/27 heads 的 train/select 都是 `train_idx` 子集；validation nodes 不在其 split；未读取 NC test labels/metrics |
| Fast vs brute | synthetic 与 Movies/42 四种 modality/operator 替换均以 float64 逐项比较；formal 最大 logit 误差约 `2.2e-14`，阈值 `rtol=1e-7, atol=1e-8` |
| Probe split/scaling/null | outer 和 inner dst-disjoint；feature/target scaler inner-train-only；train 与 selection 两侧均 shuffle 完整 tuple；outer fold 未 shuffle |
| Figure QA | 六张图 alignment 均 0 fail/0 warn；PDF text 均可审计，最小 5.5 pt（门槛 5 pt）；六张 PDF collision 均 0 findings；源预检 20 PASS、0 FAIL、1 WARN |

唯一 figure-source 预检 WARN 是 `.dropna()` 检测到缺失相关系数过滤语句。代码记录每个相关性面板的 before/after；本次 24 个相关性检查均 `excluded=0`，不存在未报告的行排除。Ridge 固定 alpha=1 的 fits 都返回有限值，但 sklearn 发出高维共线输入的 ill-conditioned-matrix warning；未调整 alpha、重试或采用 fallback。

## 共享 head 与 output separation

Head-selection 质量（每 dataset 的 repeat 均值；数据集类别任务不同，不作跨数据集性能比较）：

| Dataset | 9-combo mean CE | S/S accuracy | S/S macro-F1 | 9-combo accuracy（描述性） |
|---|---:|---:|---:|---:|
| Movies | 1.201 | 0.593 | 0.536 | 0.587 |
| Grocery | 0.429 | 0.881 | 0.835 | 0.874 |
| ele-fashion | 0.374 | 0.873 | 0.700 | 0.872 |

两个 structural weight slice 在所有 head 中均非零（dataset 平均范数约 1.14–2.11）。这说明 head 并非结构槽完全空置，但不能单凭权重范数推断边替换有任务价值。

对所有 validation edges、seeds 和 head repeats 汇总，logit shift 的中位数为 Movies `0.097–0.159`、Grocery `0.182–0.243`、ele-fashion `0.175–0.293`（范围覆盖四个 modality/operator targets）。相应 probability-L1 中位数分别约 `0.005–0.011`、`0.0015–0.0025`、`0.0010–0.0016`。JS 中位数为 Movies `8.2e-6–3.1e-5`、Grocery `4.3e-6–9.8e-6`、ele-fashion `1.8e-6–4.2e-6`；JS 的 90th percentile 在各 dataset 大约 `2.0e-4–7.6e-4`。argmax flip 比例只有 Movies `0.82–1.54%`、Grocery `0.77–0.92%`、ele-fashion `0.33–0.42%`。所以替换不等于零效应，但通常只是小概率输出位移，且随数据集、模态和 operator 变化。

## Utility 稳定性与异质性

| Dataset | head-repeat median Spearman / centered Spearman / sign agreement | cross-seed median Spearman / centered Spearman / sign agreement |
|---|---|---|
| Movies | 0.722 / 0.713 / 0.801 | 0.428 / 0.381 / 0.674 |
| Grocery | 0.750 / 0.728 / 0.852 | 0.426 / 0.366 / 0.715 |
| ele-fashion | 0.703 / 0.680 / 0.818 | 0.334 / 0.289 / 0.676 |

跨 head 的稳定性明显好于跨 seed。edge gain 的三-head sign-consistency fraction 随 dataset/target 约为 `0.64–0.87`，但很多 gain 接近零，不能把符号当硬角色标签。跨 seed 对齐 edge overlap 较低：Movies 0.203、Grocery 0.188；ele-fashion 是 1.0。

Clean utility 仍有明显边际分布异质性，但中心常接近零。按 dataset 汇总四个 target，positive fraction 为 Movies `0.455`、Grocery `0.380`、ele-fashion `0.430`；degree≥5 recipient 中，至少同时出现正、负 incoming gains 的 target 比例均值分别为 `0.882`、`0.839`、`0.725`。同 target 内共存因此仍存在，Text/Visual 也有分歧：sign-disagreement fraction 均值为 Movies `0.454`、Grocery `0.365`、ele-fashion `0.408`，within-target centered cross-modality Spearman 仅 `0.040/0.131/0.046`。

## P1.3 historical utility 对比

按 dataset 对所有 seed、modality、operator 的边级相关作均值：

| Dataset | Spearman | sign agreement | centered Spearman | top-|effect| Q75 Jaccard |
|---|---:|---:|---:|---:|
| Movies | 0.395 | 0.666 | 0.491 | 0.477 |
| Grocery | 0.543 | 0.757 | 0.574 | 0.620 |
| ele-fashion | 0.279 | 0.646 | 0.314 | 0.605 |

对应并非完全消失，但由弱到中等且数据集依赖。新旧 target 定义不同：P1.3 是 joint-readout operator block 的 relative marginal contribution，本轮是同一 shared slot、同一 head 下的单边替换 CE gain。结论应收窄为：P1.3 支持 operator-block marginal heterogeneity；只有本轮测得的 clean substitution effects 才能回答这里的替换问题，旧 delta 不能直接作 edge-function preference evidence。

## Clean-utility learnability

表中数值先对四个 utility targets、fold 和 seed 做平均；`WT` 是 within-target residual Spearman，`AUC` 是 utility sign AUROC（chance=0.5）。R² 均为负值时，表示未超过对应的零均值标准化预测基准。

| Dataset | Probe | Total Spearman | WT Spearman | R² | sign AUC |
|---|---|---:|---:|---:|---:|
| Movies | TARGET_ONLY | 0.029 | — | -0.019 | 0.519 |
|  | ENDPOINT | 0.027 | 0.015 | -0.033 | 0.515 |
|  | ENDPOINT_LOCAL | 0.029 | 0.016 | -0.047 | 0.518 |
|  | strict shuffled null | 0.025 | 0.009 | -0.047 | 0.517 |
| Grocery | TARGET_ONLY | 0.057 | — | -0.011 | 0.493 |
|  | ENDPOINT | 0.059 | 0.007 | -0.023 | 0.502 |
|  | ENDPOINT_LOCAL | 0.053 | ~0.000 | -0.034 | 0.496 |
|  | strict shuffled null | 0.049 | -0.011 | -0.034 | 0.491 |
| ele-fashion | TARGET_ONLY | 0.134 | — | -0.012 | 0.558 |
|  | ENDPOINT | 0.154 | 0.116 | -0.023 | 0.575 |
|  | ENDPOINT_LOCAL | 0.146 | 0.104 | -0.032 | 0.576 |
|  | strict shuffled null | 0.129 | 0.065 | -0.030 | 0.564 |

Movies 与 Grocery 的真实 endpoint/local MLP 只比 strict shuffle 高零点几百分点或相当，且 local 相对 endpoint 没有一致改善。ele-fashion 的 endpoint/local 有更高 total 与 within-target rank；ENDPOINT_LOCAL 比 TARGET_ONLY 高 0.012 total Spearman、比严格 shuffle 高 0.017 total 和 0.039 within-target Spearman，但 ENDPOINT 优于 LOCAL，null 本身也有可观的 target-associated rank。因此该数据集只提供有限的局部 edge correspondence 线索，并非跨数据集可重复的 edge selector。

`SIM_ONLY` 在 Movies/Grocery 的总相关与 endpoint 类似或稍高；`TARGET_ONLY` 在前两者已解释了可见的微弱总排序。sign AUROC、balanced accuracy 接近 chance，所有 primary MLP R² 均为负。只看总 Spearman 会夸大可用预测信息。Ridge 与 primary MLP 的弱结论总体一致：三数据集 Ridge 的总 Spearman 约 `0.002–0.017`，R² 约 `-2.55` 到 `-0.33`，无稳定边级增益。因而没有证据认为 MLP 弱结果仅由非线性训练失败造成。

### 稳定与大效应子集

只在 evaluation 上选取三 head sign-consistent edges 后，ENDPOINT_LOCAL 的 total / WT Spearman 为 Movies `0.031/0.015`、Grocery `0.065/~0`、ele-fashion `0.167/0.116`，相对 full-set `0.029/0.016`、`0.053/~0`、`0.146/0.104` 只有小幅变化。每 fold 的 top-|gain| 四分位上，ENDPOINT_LOCAL total / WT Spearman 为 `0.036/0.006`、`0.051/0.012`、`0.116/0.055`；sign AUC 为 `0.519/0.525/0.590`。高效应和高稳定子集没有把两个弱数据集变为可预测，也没有消除 ele-fashion 与严格 null 的差别。

## DirectState 与 frozen Q/R/U

下表为总 Spearman / within-target Spearman 的均值：

| Dataset | DirectState | Frozen Q_PAIR | Frozen R_SHARED | Frozen U_MODAL |
|---|---:|---:|---:|---:|
| Movies | 0.033 / 0.007 | 0.037 / 0.006 | 0.026 / 0.003 | 0.032 / 0.006 |
| Grocery | 0.037 / -0.006 | 0.093 / -0.004 | 0.058 / -0.004 | 0.055 / 0.004 |
| ele-fashion | 0.122 / 0.062 | 0.141 / 0.054 | 0.101 / 0.022 | 0.095 / 0.023 |

utility-supervised DirectState 在 ele-fashion 有信号，但低于 observable endpoint MLP 的 `0.154/0.116`；Movies/Grocery 仍弱。冻结 U_MODAL 的 within-target rank 在三个数据集都小（最大 0.023），未保留 strong clean edge effect。Q_PAIR 在 ele-fashion 有一定秩信息但 sign AUC 约 0.54，且未显示为稳健的 edge-level function recovery。证据不足以把主要瓶颈归因于 q/r/u 压缩或 NC task-training misalignment；总体先表现为 dataset-dependent、edge evidence 弱且部分可由 recipient/neighborhood context 排序。

## 对应附件问题 Q1–Q24

1. **无 validation label 参与 head fit/selection。** 27 个 head 均只从原始 train_idx 内的 head_train/head_select 取标签。
2. **性能合理但非完美。** S/S 与九组合在 head_select 上均有可用分类性能；九组合共同训练避免把 D/P 当 OOD。各 dataset 结果见上表。
3. **替换通常会改变 logits，但 CE/probability 影响通常较小。** Logit shift 中位数非零，prediction flip 约 0.3–1.5%。
4. **output separation 随 dataset、modality、operator 而变。** Movies visual JS/flip 一般大于其 Text；ele-fashion flip 最少而 logit shift 并不总最小；详见 raw utilities 与 `output_separability.csv`。
5. **head repeats 内 utility 排序较稳定。** 中位 Spearman 约 0.70–0.75，sign agreement 约 0.80–0.85；Product target 相对较低。
6. **跨 seed 稳定性有限。** 中位 Spearman 0.33–0.43，within-target 0.29–0.38，明显低于 head-repeat reliability。
7. **clean gains 仍异质。** 大约 38–46% gain 为正，且 D/P 分布、模态与数据集不同；总体 mean/median 多接近零。
8. **同 target 内正负共存仍常见。** degree≥5 recipient coexistence ratio 约 0.73–0.88。
9. **Text/Visual disagreement 保留。** 平均 sign disagreement 约 0.37–0.45，中心化秩相关较低。
10. **P1.3 delta 与 clean gain 中度且依赖数据集地对应。** Spearman 均值 0.279–0.543；top-effect overlap 不等同于语义角色恢复。
11. **P1.3 claim 应收窄。** 它直接支持 operator-block marginal heterogeneity，不足以单独证明 edge substitution preference。
12. **TARGET_ONLY 信号：** Movies/Grocery total Spearman 约 0.03/0.06，sign AUC 接近 chance；ele-fashion 为 0.134、AUC 0.558。它主要是 target/neighborhood-associated 总排序的基线，而非 edge role。
13. **ENDPOINT 对 TARGET_ONLY：** 前两数据集增量极小；ele-fashion 有约 0.020 total Spearman 增量及 WT 0.116。
14. **LOCAL 对 ENDPOINT：** 无一致收益；Movies 微增 0.002，Grocery 与 ele-fashion 下降。
15. **LOCAL 对严格 shuffle：** Movies 近似、Grocery total 略高但 centered rank 近零、ele-fashion modestly better；没有跨 dataset 的稳健优势。
16. **within-target rank：** 主要集中在 ele-fashion；Movies 很弱，Grocery 近零。
17. **head-sign-consistent subset：** 只带来小幅提升；未改变整体判断。
18. **high-effect subset：** ele-fashion 稍容易预测；Movies/Grocery 仍弱，sign AUC 仅 0.52 左右。
19. **Ridge 与 MLP：** 都不支持通用 edge predictability；Ridge 更弱，未发现 optimizer-free 的替代解释。
20. **DirectState capacity：** dataset-specific；ele-fashion 有一定容量，其他数据集弱，整体低于 endpoint MLP。
21. **Frozen Q/R/U：** U_MODAL 没有清晰保留可预测的 within-target clean effect；Q_PAIR 在 ele-fashion 的总排序较高，但仍不足以支持通用 routing。
22. **瓶颈：** 最符合 dataset-dependent mixed，并同时包含 Movies/Grocery 的 local evidence insufficiency 与弱 target/neighborhood ordering；当前不支持优先诊断 relation-state compression 或 task-training misalignment。
23. **是否足以继续 edge-wise latent-function router：NO。** 个别 ele-fashion 信号不足以抵消其跨数据集不复现及 local 对 endpoint 无稳定改善。
24. **是否足以引入 stochastic latent-function distribution：NO。** 本轮没有识别出真实随机 latent function 生成过程，也没有进行 stochastic modeling。

## Manenti 解释边界与 Self-audit A–J

- 本轮没有复现 Manenti 等人的定理，也没有学习 calibrated latent graph distribution。仅采纳两点方法警示：point-task prediction 不保证 latent relation recovery；tested latent choice 要能在 downstream output 上产生可区分 effect 才谈得上 task-specific identifiability。这里 JS/logit shift 只是经验类比。
- **A** 没有把 JS 称为理论 injectivity。**B** 没有把 head-repeat uncertainty 称为 aleatoric latent-function distribution。**C** validation label 未用于 shared-head fit 或 selection。**D** utility 只叫 clean counterfactual gain，不叫 ground-truth role。**E** 独立测量并报告 P1.3/new correspondence，没有强迫一致。
- **F** clean gain弱没有推翻 P0/P1.3 的所有 structure-semantic interaction；只限定本轮测试的 operator、readout 与 utility target。**G** 使用 TARGET_ONLY 与 within-target residual 指标区分 recipient context 和 edge ranking。**H** strict shuffle 包含 inner-train 与 inner-validation，outer fold 保持原 tuple。**I** 两种 scaler 均为 inner-train-only。**J** 没有加入 router、sampler 或新模型。

## Artifacts

- 数据表：[`data/`](data/)；关键原始 head-repeat edge gains 在 gitignored `outputs/l01_shared_slot_function_identifiability/all_shared_slot_raw_utilities.csv.gz`。
- 六张图：[`figures/`](figures/)（PNG、editable SVG/PDF、600-dpi TIFF）。
- 配置、provenance、fit/split/H0 audit 与 figure QA 摘要：[`run_manifest.json`](run_manifest.json)；可复现命令及协议：[`README.md`](README.md)。
- 本轮到此结束；不启动下一模型阶段。
