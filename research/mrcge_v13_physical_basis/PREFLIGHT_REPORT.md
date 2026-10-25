# Preflight Report

## Implementation verification

The focused implementation suite passed **11 tests**. The requested historical regression set passed **65 tests** with 21 deprecation warnings. Source-level tests verify exact original-switch initialization/evaluation/training equivalence, NoP0 coefficient and gradient behavior, training/evaluation path agreement, checkpoint reload, support invariants and the corrected Hybrid SameNode positive path.

## CPU and data preflight

Status: **pending committed-source CPU preflight**. The runner will resolve each frozen split using project configuration, compare input/graph/cache fingerprints with V1.3-A, check all historical baseline rows/checkpoint hashes, mask Test labels, and record original-switch initialization equality on Movies and Grocery dimensions. The protected historical research directory hash inventory will be stored in `data/preflight.json`.

## GPU preflight

Status: **not started**. It may start only after the implementation is committed and the CPU/data preflight passes. The runner will preflight each of the seven authorized cells individually on physical GPU 1 and record actual peak allocated/reserved memory, finite loss/gradients, Validation RNG invariance, the Hybrid SameNode positive-path audit, and the NoP0 coefficient-mask audit. Formal training is gated on all seven current-source GPU preflights passing. The launch guard requires `max(8192 MiB, measured peak reserved MiB + 2048 MiB)` free; runtime checks continue during each isolated worker.

No model training has been launched as of this report revision.
