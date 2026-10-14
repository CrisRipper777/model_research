# Experiment plan

## Matrix

Three datasets (Movies, Grocery, ele-fashion), three conditions
(`A0_legacy`, `A1_separate_intra`, `A2_cross_source`), and seeds 42, 43, 44
form 27 full-graph NC runs. Each Dataset × Seed uses the official split for
that seed. Conditions share the same V2.2 common initialization and the new
router/gamma initialization; the NC classifier is initialized after model
construction and must match across conditions.

## Gates

Before training: unit tests; real-data A0-to-V2.2-M0 forward compatibility;
operator provenance/direction; matched initialization; gradient/update checks;
empty-view masking; validation RNG preservation; a full-graph Forward/Backward
resource check on all datasets. Formal seed 42 runs are reviewed before seeds
43/44. No metric threshold determines whether later seeds proceed.

## Measurements

Record best Accuracy and best Macro-F1 checkpoint metrics, Train/Validation
CE/Accuracy/F1 gaps, epochs, parameters, runtime, peak allocated/reserved GPU
memory, split/cache fingerprints, initialization hashes, checkpoint hashes,
per-epoch histories, per-class Validation metrics and confusion matrices,
prediction transitions, and view/router/scale diagnostics. Frozen S/C shutoff
is labeled checkpoint sensitivity only; it is not a re-training ablation.

Test evaluation and Test-label access are disabled. Failed attempts remain in
the failure ledger. Checkpoints, raw inputs, and caches stay outside Git.
