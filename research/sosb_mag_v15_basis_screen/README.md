# SOSB-MAG V1.5 Structural Basis Screen

This directory records a validation-only comparison of four matched structural response variants. The experiment asks whether a K=4 response space improves on Local/Global responses, whether signal-conditioned orthogonal coordinates help over raw polynomial coordinates, and whether separate Text/Visual coefficients remain useful after SOSB.

## Variants

- `A0_legacy_lg`: protected intrinsic path plus shared bias-free Local/Global response experts, static two-way mixture, and sigmoid gate.
- `A1_rawpoly_shared`: four per-order active-RMS normalized raw polynomial responses and shared L2-normalized coefficients/gate.
- `A2_sosb_shared`: four signal-conditioned, channel-wise orthogonal Krylov responses and shared coefficients/gate.
- `A3_sosb_modality`: same SOSB basis with separate Text and Visual coefficient vectors/gates.

All variants use the same modality projectors, hidden dimension 256, dropout 0.2, physical graph with self-loops removed, protected intrinsic residual, and late fusion. Variant-specific modules are instantiated in a fixed order and frozen when inactive. No router, MoE, cross-modal attention, private expert, topology learning, auxiliary diversity loss, GPR baseline, or LP task is included.

## Protocol

Datasets are Movies, Grocery, and ele-fashion; seeds are 42, 43, and 44. The formal campaign has 36 runs under `unified_full_graph_nc_v1`. Checkpoints are selected by Validation Accuracy. Smoke and campaign use `task.evaluate_test=false`; Test metrics are not computed and Test label values are not indexed.

## Numerical definitions

Active nodes have nonzero physical degree after self-loop removal. Active RMS and channel inner products use active nodes only; isolated structural coordinates are exactly zero. SOSB uses two-pass channel-wise modified Gram-Schmidt.

Breakdown compares the unregularized residual RMS against `1e-5`, then normalizes with `sqrt(mean(square)+1e-8)`. Adding epsilon before the breakdown comparison would floor the RMS at `1e-4`, above the configured threshold. The jittered condition number is `cond(G + jitter*I)`, with `jitter=1e-6*max(1, abs(trace(G))/K)`. Paired comparisons are descriptive; no significance tests are performed.

The SOSB implementation is an **OptBasis-inspired signal-conditioned orthogonal Krylov basis**, not an exact OptBasisGNN reproduction. It evaluates coordinate conditioning, redundancy, and modality-conditioned structural response. It does not show that orthogonality expands the polynomial function space, that SOSB is task-optimal, that node routing is unnecessary, or how many experts a later model should use.

## Records

- `LEGACY_DIAGNOSTIC.md`: previous V1 checkpoint diagnostics.
- `REPORT.md`: formal validation results and interpretation.
- `data/`: environment, smoke, campaign, summary, paired comparison, basis, and coefficient records.

Training checkpoints and large outputs remain under ignored `outputs/` and are not tracked here.
