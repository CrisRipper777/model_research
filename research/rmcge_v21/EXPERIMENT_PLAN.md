# RM-CGE V2.1 Experiment Plan

## Matrix and protocol

Run Movies, Grocery and ele-fashion with C0/C1/C2 and seeds 42, 43 and 44: 27 runs. Within each Dataset × Seed, all conditions use the same official feature tensors, physical graph, V7B.1 K=8 semantic cache, Train/Validation split, public model initialization and classifier initialization.

Use `unified_full_graph_nc_v1`, full-graph propagation, AdamW, learning rate `1e-3`, weight decay `1e-4`, dropout `0.2`, maximum 300 epochs, patience 30, minimum epoch 30, minimum delta `1e-4`, gradient clip 1.0 and validation every epoch. Validation Accuracy selects the primary checkpoint; retain a separate Best Macro-F1 checkpoint. Keep the inherited balance loss and set `evaluate_test=false`.

Train seed 42 first. Inspect finite metrics, gradients, checkpoint reloads, row-mass conservation, resource use, histories and validation-only provenance for all nine runs. If they pass, train seeds 43 and 44. Do not change parameters based on interim validation scores.

## Evidence and diagnostics

Retain each epoch's Train/Validation CE, Accuracy and Macro-F1; checkpoint and classifier hashes; train time; peak GPU memory; model sizes; and paired seed differences. For each modality and purpose, measure row-mass error, eligible-node fraction, selected/original edge ratio, per-receiver weight variation, entropy, ESS, KL/JS divergence, degree-one behavior, score saturation and reverse-edge asymmetry. Compare the distributions across modality and purpose, and run a frozen receiver-mean query probe.

Measure physical, Intra, Cross Base, Cross Delta and KNN message scales, expert-weighted scales, graph correction, intrinsic representation and router use. Evaluate five frozen interventions: Intra selection off, Cross selection off, both off, Cross Base off and KNN off. Mark paths absent from a condition as Not Applicable. Frozen interventions are not retrained ablations.

Report C1−C0, C2−C1 and C2−C0 per seed for Validation Accuracy, Macro-F1 and CE. Use sample SD for three-seed summary tables and do not claim significance. Test labels and Test metrics are out of scope.
