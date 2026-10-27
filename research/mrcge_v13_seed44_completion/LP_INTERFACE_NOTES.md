# LP 接口静态检查

本文件仅根据现有代码记录适配边界；没有实现或运行 LP。

## 当前编码器的接口限制

`src/models/mrcge_v13.py` 显式声明 `supports_link_prediction=False`、`requires_global_semantic_candidates=True` 和 `requires_full_lp_sampler_depth=True`。`src/tasks/lp.py` 会拒绝这类声明。编码器 forward 还要求全局节点行顺序的 `x` 与全局物理 `edge_index`，语义候选缓存中的边端点也以全图节点 ID 表示。

LP 的 `LinkNeighborLoader` 为每个 batch 提供 sampled-subgraph 的局部 `x`/`edge_index`、局部 `edge_label_index` 和 `n_id`（局部节点到全局 ID 的映射）。因此把 batch 张量直接传给当前 V1.3 forward 会造成节点数/节点 ID 语义不匹配；全局 Semantic Cache 边也不能直接用于局部 batch，需要先按 `n_id` 选取诱导子图中的边，再将全局端点映射为局部端点。当前 V1.3 的全图传播依赖与 sampled-subgraph 的不同邻居边界也需要明确处理。

## 防止验证/测试正边泄漏

LP 适配应仅用训练正边构造消息传播图，再在采样前后断言消息图及其反向边不含验证或测试正边。现有 `src/data/loaders.py` 从 train split 构造 `data.edge_index`；未来任务适配仍应核验该保证在实际 sampler 输入上成立。验证/测试正边可用于对应阶段的监督标签与负采样过滤，但不能进入训练传播图；批内训练监督边的 mask 也应基于全局边 ID 检查，避免采样器把监督正边作为消息边重复泄漏。

## 任务适配与核心编码器边界

任务适配包括：LP split 的 train-only 消息图、`n_id` 全局/局部 ID 映射、Semantic Cache 的 batch 诱导与重编号、边评分头、正负样本采样、训练监督边屏蔽及验证/测试评估调用。若这些处理保持编码器在同一组输入节点与边上的运算不变，属于任务接口适配。

修改编码器的物理/语义传播深度、专家路由或 Top-K、跨视图融合、语义候选的构造/筛选语义，或者让编码器依赖局部采样深度改变传播结果，都会改变核心表示函数，应作为独立架构修改验证，而不是悄然并入 LP 适配。
