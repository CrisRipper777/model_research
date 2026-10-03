# PCRR-E0.1 design audit — Target-Only Active Residual Control

## Provenance and evidence boundary

- Parent branch: `exp/pcrr_e0_postgpr_paired_residual`
- Required parent SHA: `d1c49583e56af20063ed7274630b2df27cc612e7`
- E0.1 branch: `exp/pcrr_e01_target_only_control`, created directly from that SHA.
- Frozen E0 metrics are read from `research/pcrr_e0_postgpr_paired_residual/data/performance_by_run.csv`; no B/P/S formal jobs are repeated. The E0.1 formal campaign adds only nine T runs.
- Existing fixed NC splits are used unchanged. All NC runs set `task.evaluate_test=false` and `task.development_no_test=true`. No NC test or formal LP is in scope.

## A. What E0 established and left unresolved

E0's paired validation results have P−S accuracy positive for all 9 dataset-seed pairs. The dataset means were +0.770 pp on Movies, +0.537 pp on Grocery, and +0.409 pp on ele-fashion. P−B was +0.380 pp, −0.059 pp, and −0.266 pp respectively. This supports sensitivity to the correct correspondence over E0's wrong-source control, but it does not establish that cross-modal source information adds value beyond an active target-only nonlinear refinement. Because S injects an unrelated source into the residual, P>S could also result from corrupting an otherwise useful adapter.

## B. E0 paired representation

For target representation `t` and other-modality source `s`, E0 builds `F_P(t,s)=[t,s,abs(t-s),t*s]`. The target `t` is already the complete 256D, normalized RGD/GPR representation, so the adapter can in principle learn a response primarily from target-side information.

## C. E0.1 target-only control

T sets `source := target` independently for each modality. Therefore its pair features are `[G_T,G_T,0,G_T*G_T]` for the text branch and `[G_V,G_V,0,G_V*G_V]` for the visual branch. T does not pass the other modality into the residual response. Each modality self-refines, after which the unchanged late fusion still combines node i's text and visual branches.

T keeps the same 1024D pair input, shared `LayerNorm(1024) → Linear(1024,64) → GELU → Linear(64,256)` response, zero-initialized `pair_up`, no pair dropout, optimizer, and training protocol. It adds no parameter. B/P/S/T all instantiate the same modules and state layout; same-seed tensors are tested bitwise.

## D. Frozen backbone and legacy semantics

The raw direct GPR backbone, normalized physical graph operator, modality projectors, coefficients, modality LayerNorm, late fusion, streaming GPR, CPU feature staging, chunked projection/propagation/pair response, and large-graph checkpointing remain as in v0. The v0 implementation is not edited. In v1, B/P/S retain the v0 behavior, interventions, and deterministic S derangement; the only new behavior is `target_only` source selection.

## Regression gate and execution scope

Before training, v1 B/P/S are tested against v0 base/paired/shuffled after strict state-dict mapping at `atol ≤ 1e-6`. T is checked for exact self-source features and exact zero initial residual, and all four same-seed initialization outputs are compared in eval and train mode with reset RNG state. The only new smoke is Movies seed 42 T for two epochs. The formal campaign is Movies/Grocery/ele-fashion × seeds 42/43/44 for T only: nine validation-only runs.

## Attribution questions

- Primary: paired `P−T`, asking whether the correct other-modality source adds value beyond active self-refinement.
- Secondary: `T−B` (self-refinement relative to the RGD baseline) and `T−S` (self-refinement relative to wrong-source refinement).
- E0 `P−B`, `P−S`, and `S−B` are shown as frozen context, read from the committed parent results; they are not new E0.1 results.
- Paired deltas are matched by dataset and training seed. Population SD and direction counts are descriptive over three seeds; no pseudo-IID p-values or fixed minimum-gain threshold are used.

## Outputs and interpretation boundary

`target_performance_by_run.csv` contains only the nine newly trained T cells. Combined performance/residual/GPR tables distinguish E0.1 T from frozen E0 B/P/S. `P−T` is the attribution result; checkpoint interventions are not repeated. Residual and GPR diagnostics are descriptive. Interpretation will distinguish independent source value (`P>T`) from self-refinement gains (`T>B`) and from a net architecture gain over B.

## Completed evidence and interpretation

- The B/P/S v1-to-v0 regression gates passed at `atol=1e-6`; the full repository suite passed (189 tests). The T source/feature, initialization matching, gradient, chunking, and inference checks passed.
- The two-epoch Movies/seed42 T smoke passed; `pair_up` moved from its exact zero initialization, validation/checkpointing ran, and test evaluation was disabled.
- The nine planned T validation runs completed. Parent B/P/S validation rows were joined directly from the committed E0 performance CSV by dataset and seed; no parent formal runs were repeated.
- P−T Accuracy mean is positive on Movies (+0.210 pp) and Grocery (+0.059 pp), negative on ele-fashion (−0.286 pp). Across nine seeds, directions are 5 positive / 3 negative / 1 tied. Macro-F1 favors P on 5/9 seeds; CE favors P on 7/9, while Movies CE is effectively tied by its near-zero mean.
- This is limited, dataset-dependent evidence that the correct other-modality source can add validation value beyond target-only refinement. It is not a robust general result. Parent P−B is nonpositive on Grocery and ele-fashion, so E0.1 does not establish PCRR as a broad performance module.
- Recommendation: retain the source-correspondence observation for review, but stop expanding the residual architecture here. No rank sweep, feature ablation, extra mechanism, split change, test evaluation, or LP run was performed.

Detailed outputs and the 25-point final audit are in [report.md](report.md). This screen is complete and awaits human review.
