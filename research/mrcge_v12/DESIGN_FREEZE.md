# MR-CGE V1.2 design freeze

Date: 2026-10-25
Branch: `exp/mrcge_v12_anchor_performance_screen`
Base: `0e2496efe852c76fcd06c1477de12f41ac085e7c`

## Scope

This screening keeps the V1.1 model architecture and changes the physical/semantic hop configuration, training-time paired dropout, the M expert-bank activation lifetime, and the specified filter-theta initializations. It remains supervised transductive full-graph node classification on the existing features, physical graph, split, and V7B.1 Text/Visual K=8 semantic caches.

The model uses 256 hidden dimensions, four signed graph-filter experts, a shared Top-2 router, expert bottleneck 64, eta 0.1, dropout 0.2, physical highest order 3 (B0–B3), and semantic highest order 1 (B0–B1). The constructor reads and validates `physical_hops: 3` and `semantic_hops: 1`.

## Frozen conditions

| Condition | Filter computation | Initial theta |
|---|---|---|
| `S_repaired` | V1.1 S fixed single filter | `[1, 1, 1, 1]`, fixed |
| `F_sinit` | Same single-filter path and effective initial coefficients as S | `[1, 1, 1, 1]`, learnable |
| `M_repaired` | Four shared signed experts and shared Top-2 router | V1.1 M matrix |
| `M_anchor` | Same M path | Only row 0 changes to `[0.40, 0.30, 0.20, 0.10]` |

The M expert matrix for rows 1–3 remains the V1.1 matrix. The anchor is fully learnable and receives no special router bias, loss, or bottleneck initialization. No AMP, additional expert/router, Response Strength, or new objective is used.

## Training protocol

The screening is restricted to Movies, Grocery, and Toys, each with the four conditions and seed 42 (12 cells). The only allowed bonus cells are ele-fashion M_repaired and M_anchor, subject to their measured preflight peak reserved memory plus 2 GiB and the live GPU guard. Training uses AdamW (LR 0.001, WD 0.0001), up to 300 epochs, patience 30, minimum epoch 30, gradient clip 1.0, per-epoch Validation, private CE weight 0.3, and M balance weight 0.01. Best Validation Accuracy is primary and Best Macro-F1 is secondary.

Test labels are masked to `-100` before Trainer entry, `evaluate_test=false`, and no Test metric is calculated. No seeds 43/44, additional ablations, or hyperparameter search are permitted.

## Freeze boundary

The implementation, model/config files, training configuration, and tests were committed before formal training at `9cfe6e491d2ea2a58de01101af5032199c37cf14`. Any necessary correction to model math, optimizer, or data processing invalidates affected attempts and requires a new source commit and comparable reruns.

After the ownership-guard interruption, the failed attempt and partial audit were committed as `b024ae8dbe94259611fadb0165df55a67db631b6`; the user then authorized resumption after a fresh guard passed. The 12 resumed run rows record this actual repository HEAD. That commit changed only `research/mrcge_v12/` audit data and reports; model, trainer, launcher, and configuration files stayed byte-identical to the freeze commit. Thus the per-run Git SHAs differ, but the implementation source hashes and model/configuration definitions remained frozen. Both facts are retained in `data/runs.csv` and `data/manifest.json`.

The freeze SHA and preflight source-file hashes are retained in `data/manifest.json` and `data/preflight.json`, so the pre-commit GPU validation can be matched byte-for-byte to the frozen implementation.
