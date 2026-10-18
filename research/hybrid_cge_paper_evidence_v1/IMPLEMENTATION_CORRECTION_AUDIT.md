# Implementation correction audit

## Finding

The initial implementation freeze was `0db724e5eec704a1f3db92329bfbbac056df7202`. The historical `src/models/hybrid_cge_v11.py` correctly enabled its SameNode candidate only when `study_condition == "H2_hybrid_plus"`. The new paper wrapper preserved that parent forward but renamed the runtime condition to paper IDs (`H2_full`, `A1_no_early_physical_cross`, `H2_physical_3hop`, and Fashion IDs). As a result, inherited late-completion calls received an all-false SameNode mask.

The zero SameNode support and source weights in the first `PHYSICAL_SEMANTIC_DIAGNOSTICS.md` were direct evidence. That initial report and its CSV/JSON artifacts are preserved as `PHYSICAL_SEMANTIC_DIAGNOSTICS_INITIAL_DISABLED_SAME_NODE.md` and `data/cross_modal_diagnostics*_initial_disabled_same_node.*`. A source audit established the exact cause; it was not inferred from accelerator behavior. The first targeted test suite checked A4's negative path but lacked a positive assertion that ordinary paper conditions retained SameNode.

## Scope

The completed attempt-01 runs affected by the bug are:

- 15 A1 runs: A1 was meant to retain late Semantic and SameNode completion.
- 15 three-hop runs: 3-hop was meant to retain the complete H2 late path.
- 9 Fashion runs: F1, F2 and F4 were meant to retain the complete H2 late path.

These 39 old records, checkpoints and histories remain unchanged in `data/runs.csv` and the local output tree. The analyzer selects a later corrected attempt for these cells and lists the original records in `data/superseded_runs.csv`. They are not pooled into the corrected result tables. A2 intentionally removes late completion; A3 used the explicit controlled helper with SameNode enabled; A4 used the explicit controlled helper with SameNode disabled. The nine baseline families do not use this wrapper and are unaffected.

## Correction and verification

The paper wrapper now calls the explicit controlled late-completion helper for every condition. SameNode is enabled by default and disabled only for A4, which still skips the projection call and retains late-cross Dropout 0.15. The helper exposes SameNode support, candidate RMS and routing weights in diagnostics. A new positive-path test checks projection calls, support and candidate magnitude for H2, A1, 3-hop and all Fashion candidates; the existing A4 test checks that the projection is skipped. A second test loads identical parameters into the frozen H2 implementation and the corrected `H2_full` wrapper and compares their eval Forward and auxiliary-loss outputs.

After the correction, all ten focused paper-study tests pass and the full repository suite reports 581 passed, 26 warnings. The corrected implementation is committed and frozen before any corrective real-data preflight or training. Its commit SHA and rerun source SHA are recorded in `data/analysis_manifest.json` after completion.

## Evidence boundary

The correction does not change historical H2 checkpoints, reused baselines or unaffected attempt records. The three-hop and Fashion outcomes are recomputed from their corrected checkpoints. Historical H2 checkpoint diagnostics are also rerun through the corrected compatibility wrapper so the same-node source summaries describe the checkpoint's actual H2 path. No Test labels or metrics are used.

## Corrected rerun and post-correction diagnostics

The corrective stage completed 39/39 cells at attempt 02 from the corrected freeze. For all 39 superseded-to-corrected pairs, the recorded feature, raw-feature, physical-edge, split, semantic-cache, candidate, V7B and initial model/classifier/shared-parameter fingerprints are identical; only the intended implementation source revision and resulting training trajectory differ. The preflight stage passed 13/13 affected representative cells. The new stage audit is `data/final_evidence_audit.json`.

A fresh Train/Validation-only checkpoint audit reloaded 39 checkpoints and recorded 1,280 aggregated cross-modal scalar summaries. Unlike the initial disabled-path report, the corrected `PHYSICAL_SEMANTIC_DIAGNOSTICS.md` has nonzero SameNode selector weights for both modalities across the five datasets for H2 4-hop and corrected 3-hop. The per-seed table keeps dataset/modality variation visible. These are forward-path diagnostics, not causal utility estimates. The Fashion F1/F2/F4 candidate paths are also included in the detailed CSV/JSON. Test labels were masked and Test metrics were not computed.

All corrected cells have finite Train/Validation metrics, intact checkpoint records, unchanged Validation RNG, and Test evaluation disabled. The original attempt-01 rows, checkpoints, failure log and initial zero-support diagnostic remain preserved. The current implementation and reports do not attribute the initial issue to CUDA; the exact condition-name dispatch was found in source and covered by added positive-path tests.
