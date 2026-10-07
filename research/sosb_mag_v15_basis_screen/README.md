# SOSB-MAG V1.5 Structural Basis Screen

This directory contains the validation-only four-variant screen requested for the SOSB-MAG V1.5 structural response basis.

## Variants

- `A0_legacy_lg`: protected intrinsic path plus shared bias-free Local/Global response experts, static two-way mixture, and sigmoid gate.
- `A1_rawpoly_shared`: four per-order active-RMS normalized raw polynomial responses and shared L2-normalized coefficients/gate.
- `A2_sosb_shared`: four signal-conditioned, channel-wise orthogonal Krylov responses and shared coefficients/gate.
- `A3_sosb_modality`: same SOSB basis with separate Text and Visual coefficient vectors/gates.

All variants use the same modality projectors, hidden dimension 256, dropout 0.2, physical no-self-loop graph, protected intrinsic residual, and late fusion. Variant-specific modules are instantiated in a fixed order and frozen when inactive. No router, MoE, cross-modal attention, private expert, topology learning, auxiliary diversity loss, or GPR baseline is included.

## Protocol

Datasets: Movies, Grocery, ele-fashion. Seeds: 42, 43, 44. The full campaign comprises 36 runs under `unified_full_graph_nc_v1`; the selected checkpoint maximizes Validation Accuracy. `task.evaluate_test=false` for smoke and campaign. Test metrics and Test label indexing are prohibited.

## Numerical definitions

Active nodes have nonzero physical degree after self-loop removal. Active RMS and channel inner products only use active nodes; isolated structural coordinates are exact zeros. SOSB uses two-pass channel-wise modified Gram-Schmidt. Breakdown compares unregularized residual RMS against 1e-5, then uses `sqrt(mean(square)+1e-8)` for normalization. The distinction is necessary because adding eps before comparison floors the RMS at 1e-4.

The condition number is computed as `cond(G + jitter*I)` with `jitter=1e-6*max(1, abs(trace(G))/K)`. Paired comparisons are descriptive, with no significance tests.

## Run state

- Freeze commit: `53eb362c86c433f3322397aa1eee7bafcb070824`.
- Campaign completed: 36/36; unresolved failures: 0; failure attempts: 0.
- Smoke and formal run outputs/checkpoints remain under ignored `outputs/`; only compact JSON/CSV/Markdown research records are tracked here.

See `LEGACY_DIAGNOSTIC.md` for the prior V1 checkpoint analysis and `REPORT.md` for results and interpretation.
