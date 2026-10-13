# RM-CGE V2.1 Experiment Plan

Frozen matrix: Movies/Grocery/ele-fashion × C0/C1/C2 × seeds 42/43/44. Every Dataset × Seed reuses identical official features, physical graph, V7B.1 K=8 semantic cache and Train/Validation splits; all conditions share the same public initial weights and classifier state.

Protocol: `unified_full_graph_nc_v1`, AdamW, lr 1e-3, weight decay 1e-4, dropout .2, up to 300 epochs, patience 30, minimum epoch 30, min delta 1e-4, gradient clip 1.0, validation every epoch, Validation Accuracy checkpoint selection, and existing balance loss. A separate Best Macro-F1 checkpoint is retained. `evaluate_test=false`.

Seed 42 forms the first integrity batch. Seeds 43/44 proceed after the first nine runs have finite histories, valid checkpoints, selector gradients, and intact data/cache fingerprints. No hyperparameter search is performed.

The audit compares paired Validation Accuracy, Macro-F1 and CE at each run's Best Accuracy checkpoint. Best Macro-F1 metrics are shown separately. Frozen interventions measure checkpoint sensitivity and are not retrained ablations.
