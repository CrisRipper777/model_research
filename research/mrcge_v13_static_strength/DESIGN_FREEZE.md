# MR-CGE V1.3 Static-Strength design freeze

## Scope

- Repository: `/hdd1/DataInHere/YHF/model_research`
- Base commit: `6d94c14d67e511aec34ff5fdcad3ce1a1cfd0701`
- Base branch: `exp/mrcge_v13_physical_basis_screen`
- Campaign branch: `exp/mrcge_v13_static_strength_screen`
- Authorized new cells: Static-Strength only; Movies seeds 42/43/44, Grocery 42/43, and ele-fashion 42/43.
- Read-only references: same dataset/seed V1.2 `M_repaired`; same dataset/seed V1.3-A `joint_view`; fixed-split Movies Hybrid-CGE H2 3-hop seeds 42/43/44.
- Explicit exclusions: Test evaluation, link prediction, NoP0, Early Cross disentangling, additional models, seeds, datasets, hyperparameter search, and V1.4.

## Frozen model and training protocol

The implementation inherits the V1.3-A `M_repaired` backbone: modality projectors, Physical orders 0–3, Semantic orders 0–1, shared four-expert signed polynomial bank, graph-aware Top-2 router, original theta/rho/gamma/beta initialization, Early Cross first-order injection, paired dropout and Private/Collaborative difference, SameNode/Late Semantic/None, sparse attention, channel gate, bounded residual, fusion, Private Auxiliary CE, and Expert Balance Loss.

Training is full-graph NC with AdamW, LR `0.001`, weight decay `0.0001`, up to 300 epochs, patience 30, minimum epoch 30, early-stop delta `0.0001`, gradient clipping 1.0, dropout 0.2, Late dropout 0.15, Private Auxiliary CE weight 0.3, and balance loss weight 0.01. Best Validation Accuracy is the primary checkpoint; Best Validation Macro-F1 is saved separately. Link prediction is unsupported.

## Single new mechanism

`static_logits` has shape `[2,2]`, with modality order Text/Visual and relation order Physical/Semantic. It is initialized to exact zeros and converted to `s = 2 * sigmoid(static_logits)`, giving four independent global modality/relation scales in `(0,2)`.

For modality `m`, retain learned V1.3-A values `rho_m=sigmoid(rho_raw[m])` and `gamma_m=sigmoid(gamma_raw[m])`:

```text
D_P = rho_m * R_P
D_S = gamma_m * R_S
U_priv = H_m + s[m,Physical] * D_P + s[m,Semantic] * D_S
U_collab = U_priv + rho_m * s[m,Physical] * (R_P_collab - R_P_priv)
```

The exact same modality-level Static scales are used by Private and Collaborative paths. Collaborative receives no additional Semantic response. Zero Cross context must preserve exact Private/Collaborative equality under paired Train dropout. `view_strength(modality_id, ...)` receives explicit modality identity; it does not infer it from hooks, global state, storage addresses, or dataset strings. The inherited Joint View MLP is absent from the final Static model.

## Data and evaluation controls

All three datasets use their existing seed-42 split, with automatic split generation disabled for all training seeds. Expected split fingerprints are:

| Dataset | Split fingerprint |
|---|---|
| Movies | `2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d` |
| Grocery | `7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056` |
| ele-fashion | `04d43160594823ab35d62b1bde6547225dc816405e55eb1f9c44790eabfae0c5` |

Use the existing V7B.1 Text/Visual Semantic KNN K=8 cache without rebuilding it. Before training, fail closed unless feature, physical edge, split fingerprint and file SHA, semantic cache SHA, candidate fingerprint, V7B fingerprint, node order, and input dimensions align to the same-seed V1.3-A row. Test labels are masked to `-100` before Trainer entry; no Test evaluation or LP is allowed.

## Compute and provenance controls

Only physical GPU 1 is authorized. The CPU/data and model equivalence preflight precedes per-cell GPU preflight. Each of the seven cells has an isolated subprocess and a live ownership guard. A new foreign Compute PID blocks or terminates only this task's child. The GPU must have at least 8 GiB free at launch and enough room for the measured peak reserved memory plus 2 GiB. Fashion must be measured with its own model and inputs. Every failed/blocked attempt is appended, never replaced.

Training source is committed and frozen before GPU preflight. Each run stores Best-Accuracy and Best-F1 checkpoints, resolved config, epoch history, source/config/data hashes, initialization hashes, reload metrics, RNG validation, elapsed time, and CUDA peak allocated/reserved memory. Large checkpoints and logs stay under the local output directory; only small records and reports are committed.

## Interpretation boundary

This screen asks whether four global modality/relation scales can reproduce the task behavior of a node-level Joint View strength network. Static adds four free scalars while retaining learned `rho` and `gamma`; the redundantly parameterized scales and resulting optimization paths mean Static-vs-Joint performance cannot isolate a pure causal effect of node-wise reliability. Report same-seed paired differences, per-dataset mean and population SD, costs, and node-wise strength diagnostics descriptively. Do not run significance tests or choose a winner by one seed or a preset threshold. Hybrid-CGE is Movies-only performance context, not architectural attribution.
