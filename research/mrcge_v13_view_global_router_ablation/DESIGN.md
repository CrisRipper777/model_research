# MR-CGE V1.3-A View-Global Router Ablation

This single-factor ablation compares the original `mrcge_v13 / M_repaired / joint_view` model with one condition that retains the same graph-aware router, four shared polynomial experts, Top-2, Balance Loss, private and collaborative views, late fusion, and classifier protocol.

The independent model subclasses `src.models.mrcge_v13.Model` without changing construction or state. It overrides `_filter()` and `_m_filter_impl()` because the original low-memory training path computes router logits through `self.router.net` directly. In each call, it averages the per-node logits over that call's active `support` nodes, broadcasts the mean, then applies the original dense softmax, Top-2, expert weights, and Balance Loss. Empty support uses finite zero logits. The mean remains attached to autograd. All six private/collaborative physical and semantic calls are handled independently.

No new parameter or state key is introduced. The Balance Loss formula and `balance_weight=0.01` remain fixed; its interaction with the more concentrated global Top-2 choice is part of this practical mechanism ablation. The comparison uses fixed full-graph transductive inputs, seed-42 splits, V7B.1 K=8 cache, Train-only labels, and Test-off evaluation. It does not establish inductive online behavior or isolate node adaptivity from Balance Loss coupling.
