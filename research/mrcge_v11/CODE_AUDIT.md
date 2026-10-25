# MR-CGE V1.1 Code Audit

Source commit audited: `7636cfe354a885befb571b0d638f44f357fc3ce4`. The new model is a standalone `nn.Module`; it does not inherit from the frozen H2 model chain.

## Implemented paths

- Four views share one V7B.1-aware filter implementation. M owns one four-expert bank and one graph-aware Top-2 router.
- Physical and modality-specific semantic bases are built once per modality per forward. Isolated-node support masks zero all complete graph responses.
- Private states use only their own modality. Early Cross changes only physical order 1 and has an exact per-node zero-context fallback.
- Late completion uses receiver-sparse semantic attention, Semantic/Same-node/None selection, conditional candidate averaging, channel gating, relative residual capping and non-inverted Dropout 0.15.
- S/F keep all information paths while omitting M's expert router. S has fixed equal valid-order coefficients; F has one learnable signed filter; M has four learned signed filters.
- The only Trainer change is a guarded optional `collect_epoch_diagnostics()` hook; existing models without that method follow the same code path.

## Differences from historical H2

MR-CGE removes the H2 late complementary experts/router and its balance calls; it replaces relation/view-specific filter routing with one shared graph-aware router;
M's four expert objects are shared across all views and collaborative physical calls; Private/Cross state formulas and support handling are explicit in the new model. The historical H2 Late Complementary Balance Loss accounting defect remains outside scope. The historical H2 source, old configurations, reports, checkpoints and Final Test Lock remain unchanged.

## Validation

The focused suite result is recorded in `data/preflight.json` (23 passed, 7 warnings in 5.53s). The actual-data CPU preflight verifies cache fingerprints, source_flags, global node order, Test masking and condition initialization hashes.
GPU single-step preflight status: pass (9/9 cells). Full-epoch behavior and multi-seed performance remain separate questions.
Formal runs include per-epoch loss scalars, but `data/loss_gradient_diagnostics.csv` is header-only; single-step preflight gradients do not provide per-run training gradient magnitudes.

Parameter counts from CPU initialization preflight:

| Dataset | S model params | F model params | M model params |
|---|---:|---:|---:|
| Movies | 1654325 | 1654329 | 2021737 |
| Grocery | 1654325 | 1654329 | 2021737 |
| ele-fashion | 1388069 | 1388073 | 1755481 |

## Limits

A one-step GPU check does not establish full-training stability, long-run resource behavior, multi-seed performance or generalization.
