# Same-protocol baseline comparison

Only project-run results appear in the main table; published benchmark numbers are not mixed into it. Each model sees identical dataset features, physical graph, Train/Validation/Test indices and labels (Test labels masked before Trainer), subject to its method-specific graph/feature pathway. All models use AdamW/LR 1e-3, up to 300 epochs, patience 30, min epoch 30 and gradient clip 1.0. MLP is feature-only by design.

| Dataset | Model | n | Val Accuracy | Val Macro-F1 | Val CE | Best-F1 Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| Movies | Hybrid-CGE (H2-Frozen) | 3/3 | 56.78% ± 0.44% | 49.78% ± 1.17% | 1.3497 ± 0.0226 | 50.85% |
| Movies | mlp | 3/3 | 52.22% ± 0.16% | 36.96% ± 1.23% | 1.4630 ± 0.0407 | — |
| Movies | sage | 3/3 | 54.76% ± 0.36% | 48.46% ± 0.97% | 2.2949 ± 0.6920 | 50.38% |
| Movies | mmgcn | 3/3 | 55.69% ± 0.81% | 46.90% ± 2.85% | 1.6308 ± 0.1430 | 49.16% |
| Movies | rolemag | 3/3 | 54.36% ± 0.36% | 42.27% ± 0.63% | 1.3960 ± 0.0254 | 44.74% |
| Grocery | Hybrid-CGE (H2-Frozen) | 3/3 | 83.86% ± 0.39% | 75.97% ± 1.11% | 0.6393 ± 0.0433 | 77.04% |
| Grocery | mlp | 3/3 | 79.08% ± 0.47% | 71.72% ± 0.22% | 1.4560 ± 0.1553 | — |
| Grocery | sage | 3/3 | 83.32% ± 0.17% | 76.23% ± 0.76% | 1.0564 ± 0.1021 | 76.44% |
| Grocery | mmgcn | 3/3 | 81.80% ± 0.62% | 73.29% ± 0.91% | 0.8474 ± 0.0982 | 73.84% |
| Grocery | rolemag | 3/3 | 80.42% ± 0.59% | 70.28% ± 1.81% | 0.7434 ± 0.0235 | 70.58% |
| ele-fashion | Hybrid-CGE (H2-Frozen) | 3/3 | 87.72% ± 0.06% | 74.65% ± 0.35% | 0.4113 ± 0.0035 | 76.16% |
| ele-fashion | mlp | 3/3 | 88.10% ± 0.05% | 77.40% ± 0.72% | 0.5690 ± 0.0972 | — |
| ele-fashion | sage | 3/3 | 87.68% ± 0.24% | 75.85% ± 2.00% | 0.4803 ± 0.1478 | 76.97% |
| ele-fashion | mmgcn | 3/3 | 87.60% ± 0.04% | 74.97% ± 0.22% | 0.4154 ± 0.0166 | 76.13% |
| ele-fashion | rolemag | 3/3 | 87.08% ± 0.15% | 73.73% ± 0.28% | 0.4122 ± 0.0037 | 74.84% |
| Toys | Hybrid-CGE (H2-Frozen) | 3/3 | 80.33% ± 0.31% | 77.47% ± 0.13% | 0.6724 ± 0.0075 | 77.77% |
| Toys | mlp | 3/3 | 75.78% ± 0.80% | 72.83% ± 1.60% | 1.2848 ± 0.4171 | — |
| Toys | sage | 3/3 | 78.84% ± 0.85% | 76.08% ± 1.21% | 1.1966 ± 0.3539 | 76.29% |
| Toys | mmgcn | 3/3 | 78.86% ± 0.38% | 75.57% ± 0.48% | 0.8225 ± 0.1155 | 76.06% |
| Toys | rolemag | 3/3 | 78.84% ± 0.23% | 75.85% ± 0.14% | 0.7290 ± 0.0106 | 75.86% |
| Reddit-S | Hybrid-CGE (H2-Frozen) | 3/3 | 96.60% ± 0.25% | 93.02% ± 0.50% | 0.1476 ± 0.0281 | 93.05% |
| Reddit-S | mlp | 3/3 | 93.07% ± 0.40% | 87.70% ± 0.83% | 0.3005 ± 0.0617 | — |
| Reddit-S | sage | 3/3 | 95.27% ± 0.33% | 91.24% ± 0.51% | 0.2369 ± 0.0154 | 91.30% |
| Reddit-S | mmgcn | 3/3 | 96.08% ± 0.38% | 92.18% ± 0.34% | 0.1720 ± 0.0030 | 92.28% |
| Reddit-S | rolemag | 3/3 | 95.71% ± 0.40% | 91.17% ± 0.27% | 0.1810 ± 0.0240 | 91.26% |

## Implementation fairness and limits

- `MLP` uses the same node features, labels and splits but no message passing, as required by the method. Its current feature-only trainer emits a Best-Accuracy checkpoint; a Best-F1 checkpoint is unavailable, so its Best-F1 column is blank.
- `GraphSAGE` uses the repository's supervised full-graph SAGEConv adaptation on the same transductive graph. This differs from sampled inductive GraphSAGE training in the original method family.
- `MMGCN` splits concatenated inputs in `[text, visual]` order and uses parallel modality branches. Its source-style `id_embedding` is a plain tensor rather than a registered parameter; it is therefore fixed at seeded initialization and not optimized or stored in `state_dict`. The run record stores its hash and evaluation recreates it from seed/config. Treat scores as a project adaptation, not exact official MAGB/MMGCN reproduction.
- `RoleMAG` is the project's supervised NC wrapper: it explicitly converts `[text, visual]` input to the core's modality order, forms nonzero/finite feature masks, uses default-zero optional routing regularizers, and applies the configured complementary/heterophily warmup/ramp. It is not described as an official repository reproduction.
- Best-F1 checkpoints for supported graph-model paths are the highest Validation Macro-F1 observed before Validation-Accuracy early stopping. Metrics for Best-Accuracy and Best-F1 are not interchangeable. MLP reports its Best-Accuracy checkpoint only.

The table uses mean and sample SD across three seeds. Public MAGB/OpenMAG numbers use different models, feature choices and/or split/training protocols, so they are contextual literature only and are not numerically compared here. See [MAGB](https://github.com/sktsherlock/MAGB), [OpenMAG](https://github.com/YUKI-N810/OpenMAG), the [GraphSAGE paper](https://arxiv.org/abs/1706.02216), and the [official MMGCN repository](https://github.com/iLearn-Lab/MM19-MMGCN).
