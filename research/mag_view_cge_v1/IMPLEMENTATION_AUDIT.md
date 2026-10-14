# Implementation Audit (pre-run freeze)

## Code review conclusion

The implementation has one physical branch and one shared four-expert bank.
The same and cross-source semantic branches both call `self.experts[k]`; one
shared view router differentiates them using its view-type embedding. A1/A2
physical expert inputs contain physical multi-hop profiles and the inherited
directional physical cross projection only. Their Text/Visual KNN messages are
not also inserted into the physical profile.

The cross-source direction is explicit: a Text receiver reads the Visual KNN
operator after `visual_to_text` projection; a Visual receiver reads the Text KNN
operator after `text_to_visual` projection. KNN supports are built by the
existing V2 helper after canonical-cache validation. Semantic calibration uses
the matching target modality's detached physical first-hop scale. Separate
incoming-support masks are applied after expert response so empty rows remain
exactly zero and a physically isolated node can still receive valid semantic
context.

A0 temporarily selects the original V2.2 M0 discriminator and delegates its
forward without new arithmetic. New module initialization uses `fork_rng`,
keeping the legacy/common model and subsequent classifier initialization
paired. The model rejects sampled node counts and explicitly disables link
prediction. The runner masks Test labels before NC training and validation.

## Pre-run evidence

- Focused View-CGE tests: 11 passed.
- Full repository tests at freeze preparation: 539 passed, 0 skipped, 21
  warnings. `data/test_results.json` retains the first corrected test attempt.
- Real-data A0/V2.2 M0 maximum Forward error, full-graph one-step gradients,
  CUDA RNG checks, initialization hashes, and per-dataset allocated/reserved
  memory are in `data/preflight.json`.
- Preflight debugging and the two corrected implementation assumptions are
  retained in `data/preflight_attempts.json`.

This document is replaced by the post-run audit generator after the 27-run
campaign. No performance claim is made before formal training.
