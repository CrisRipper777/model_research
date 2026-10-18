# Physical propagation depth: three-hop versus four-hop

The H2 4-hop checkpoints remain unchanged and strict-load in the new wrapper for all five datasets. The independent three-hop model has `alpha_raw.shape == [4,3]`, retains four shared experts and Top-2, and runs exactly three physical propagation steps per modality. The mechanism test observes six propagation calls for a full two-modality forward.

The 15 new three-hop runs are pending their formal stage gate. Comparison uses paired Dataset × Seed Validation Accuracy as the primary metric and Macro-F1 as the secondary metric. A global depth change will be considered only if the five-dataset evidence supports it; a Fashion-only gain will be reported as a dataset-specific candidate.
