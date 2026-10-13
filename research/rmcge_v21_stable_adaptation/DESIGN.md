# RM-CGE V2.1 Stable Adaptation Study

## Objective

Determine whether propagation-before-selection can be adapted after a shared MAG model has acquired task performance, and whether selector-only warm-up followed by low-rate joint optimization improves stability or class balance. This is a study of the existing V2.1 C0/C1/C2 implementations; it does not introduce V2.2, another expert, a new router, an auxiliary objective, or a new semantic graph.

## Frozen source and data

- Source: `exp/rmcge_v21_message_purpose_selection` at `218d6d230e5afc16e259c09777f6363de5e0c6cf`.
- Datasets: Movies, Grocery, ele-fashion; seeds: 42, 43, 44.
- Data, official splits, V7B semantic candidate caches, and task protocol are inherited unchanged from the V2.1 study.
- Validation only is used for model selection and analysis. The analysis and training code never indexes Test label values.
- Historical V2.1 artifacts are read-only. New outputs are kept under `research/rmcge_v21_stable_adaptation/data/`; large checkpoints and per-node predictions stay in ignored local `outputs/`.

## Study phases

### A. Cold-start diagnosis

For the 27 historical dataset × seed × condition runs, reload both Best Validation Accuracy and Best Macro-F1 checkpoints. Report global and per-class metrics, confusion matrices, paired prediction changes and per-node CE changes, physical-degree subgroups, and the complete Train/Validation history with best epochs and stopping epochs. No Test evaluation is performed.

### B. Shared C0 warm start

For each dataset × seed, W0, W1, and W2 start from the exact same historical C0 Best Accuracy checkpoint and classifier:

- W0 continues C0.
- W1 adapts the C1 intra selector.
- W2 adapts the C2 intra and cross selectors.

W1/W2 receive 12 selector-only epochs with inherited C0 and classifier parameters frozen. All three conditions then receive a fresh AdamW optimizer and at most 90 joint epochs. Main checkpoint selection compares only the shared `t0` checkpoint and Stage 2 checkpoints; Stage 1 is diagnostic. Best Accuracy controls stopping, while Best Macro-F1 is tracked independently within the same stopping horizon.

## Interpretation boundary

W0/W1/W2 are paired comparisons from one common C0 checkpoint per dataset and seed. The comparison against historical random-initialization C0/C1/C2 also changes training budget, learning rate, optimizer state, and model-selection history, so it is descriptive rather than a causal isolation of initialization. Three seeds provide paired evidence and uncertainty summaries, not a significance claim.
