# Framework-level ablation definitions

All ablations preserve H2 initialization/configuration, feature and graph input, semantic cache, split, optimizer, private auxiliary CE, and 0.15 late-cross dropout. The implementations live in `src/models/hybrid_cge_paper_v1.py`; the frozen `src/models/hybrid_cge_v11.py` remains unchanged.

| ID | Active change | Forward path removed | Retained path |
|---|---|---|---|
| A1 `no_early_physical_cross` | Set collaborative physical state equal to private state. | Physical cross propagation, early cross profile, second cross-conditioned node-router correction and duplicate collaborative physical expert call. | Private physical and same-source paths; full late semantic/same-node completion. |
| A2 `no_late_completion` | Set final state equal to collaborative state. | Late sparse semantic attention, same-node candidate, selector, channel gate, interaction MLP, complementary expert and custom late dropout. | Private states and early physical cross collaboration. |
| A3 `no_moe` | Use `mean_l Norm(P^l H)`, add fixed `0.25*C_P_hat` for early collaboration, use normalized `S_m H_m`, and use `D=lambda*p_accept*g*Cbar` late. | Physical, same-source and complementary shared experts; all associated Top-2 selection/balance paths and their parameters are removed from the module/optimizer tree. | Existing bounded late residual, source selector, channel gate, cross projections, same-node candidate, private heads and late dropout. No replacement expert bank or large MLP is introduced. |
| A4 `no_same_node` | Permanently mask Same-node selector logit. | Same-node projection and its candidate computation. | Semantic/None source choice, full remaining H2 path, including 0.15 late-cross dropout. |

## Required tests

The focused tests instrument actual module calls, parameter trees and gradients. They must establish that A1 never calls the early physical cross operator or collaborative physical route, while late cross projections retain task gradients; A2 never calls late modules and produces no late-only gradients; A3 has no expert or MoE router parameter in its module/optimizer set and no balance loss; A4 does not call the same-node projection and retains 0.15 dropout. All variants must pass finite forward/backward, isolated-node zero-support behavior, matched initialization, Train-only auxiliary supervision, and checkpoint round-trip checks. Full repository pytest is run before the implementation freeze.

The pair audit stores each variant's full initialized state hash and separately compares the shared public parameter intersection. The latter excludes the complete MoE/router parameter trees that A3 removes by design; the classifier initialization hash must match exactly for every Dataset × Seed group.
