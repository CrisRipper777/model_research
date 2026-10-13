# Experiment plan

## Frozen matrix

Three datasets (Movies, Grocery, ele-fashion), five conditions (M0–M4), and seeds 42/43/44 give 45 full-graph NC runs. Seed 42 is the first 15-run integrity batch. After its gate passes, seeds 43 and 44 add 30 runs. Within each Dataset × Seed, all arms use the same official split file, features, graph, V7B.1 cache and initial model/classifier state. Splits are seed-specific where the repository's official files differ; they are never regenerated or shared across seeds.

## Protocol

`unified_full_graph_nc_v1`; AdamW, lr `1e-3`, weight decay `1e-4`, dropout `.2`, max 300 epochs, patience 30, min epoch 30, min delta `1e-4`, gradient clip 1.0, validation each epoch, Accuracy-based stopping, separate Best Macro-F1 checkpoint, no warm-start, no KNN retuning, no new auxiliary loss. The inherited model returns `0.01 * balance`; the trainer adds it with `aux_weight=1`, so the total is CE plus one 0.01 balance term. Test evaluation is false.

The shared data loader returns labels for compatibility, but the runner immediately clones labels and writes sentinel `-100` at Test indices before passing data to training or diagnostics. The runner indexes labels only at Train and Validation indices. The frozen split fingerprint includes split indices and Train/Validation labels only. The semantic cache contains no labels.

## Seed 42 gate

Proceed to seeds 43/44 only if all 15 runs have finite Train/Validation histories, nonzero first-step final ContentGate gradients in M1–M4, strict checkpoint reload and metric reproduction, matched initial model/head hashes by Dataset × Seed, intact feature/split/cache fingerprints, no OOM, no label sentinel in measured indices, and no Test metrics. An accuracy delta alone is not a stopping criterion.

## Planned analysis

Report Best Accuracy and Best F1 checkpoints separately; paired deltas M1−M0, M2−M0, M3−M0, M3−M1, M3−M2, M4−M3, and M4−M0; Accuracy, Macro-F1, CE, epochs, Train–Validation gaps, parameter counts, GPU peaks and duration. Retain per-seed rows. Include gate channel statistics, residual scale/cosine and expert/router diagnostics, fixed-checkpoint interventions, per-class and degree groups, Class 11/1 fashion recall, and correct↔incorrect transitions. Frozen interventions are sensitivity checks, not retrained ablations.

No result-driven architecture or optimizer changes are allowed after seed 42 begins. A specific engineering defect requires a documented fix, new implementation freeze, and reruns for affected conditions.
