# RM-CGE V2.1 Stable Adaptation: Training Dynamics

## Frozen protocol and execution

The formal matrix was 3 datasets × 3 conditions × seeds 42, 43, and 44 (27 verified runs). For each dataset and seed, W0, W1, and W2 load the same historical C0 Best Validation Accuracy checkpoint and classifier. W1/W2 also receive matching selector initialization. The Stage 1 objective was Train CE plus the model's existing 0.01 balance loss; only new selector parameters were trainable for 12 epochs in eval mode with gradients enabled. W0 had no Stage 1 updates. Stage 2 used a new AdamW optimizer with empty state, no scheduler, common-model/classifier LR 1e-4, selector LR 5e-4, weight decay 1e-4, dropout 0.2, gradient clipping 1.0, and the frozen accuracy-based early-stop rule (patience 30, minimum epoch 20, min-delta 1e-4).

All 27 runs completed with finite Train/Validation metrics, verified checkpoint save/reload, and Test evaluation disabled. There were 1,357 Stage 2 joint updates total; runs stopped after 49–63 updates. The maximum allocated GPU memory was 12.11 GiB on the ele-fashion W2 run. W1/W2 use more time and memory than W0 because they add selector computation and a 12-epoch Stage 1. Exact runtime and memory for every run are in `data/resource_profile.csv`.

## Shared initialization and Stage 1 checks

The nine dataset × seed triplets passed the real-GPU initialization audit. Public C0 model and classifier states match exactly within each triplet, W1/W2 selector states match exactly, and W1/W2 initial selector scores are zero. The largest W1/W2-to-W0 Validation-logit difference across the nine pairs was `1.79e-6`; the maximum difference among t0 Validation Accuracy, Macro-F1, and CE was `2.38e-7`. The logit tolerance was `5e-5`. T0 evaluation preserved CPU/CUDA RNG state.

All 18 W1/W2 Stage 1 runs kept the inherited model and classifier state hashes unchanged. Every run updated the new edge scorer. All nine W2 runs received a nonzero cross-delta gradient during Stage 1 and changed `cross_delta_coefficients`. The per-run gradients and update RMS values are recorded in `data/epoch_history.csv` and `data/stage_transition_audit.json`.

Stage 1 produced measurable but modest weight redistribution. At warm end, mean absolute selector score was 0.104 for W1 text-intra and 0.215 for W1 visual-intra; W2 means were 0.114/0.196 for text/visual intra and 0.111/0.217 for text/visual cross. No Stage 1 edge score reached `|score| >= 0.99`. Mean receiver-distribution JS from the physical weights ranged from 0.00035 (W2 text-intra) to 0.00252 (W1 visual-intra) for degree ≥2 nodes. Mean effective-neighbor counts were about 8.0 on degree ≥2 nodes. These diagnostics show the scorer learned a bounded redistribution, not a wholesale replacement of physical propagation.

At warm end, Validation Macro-F1 changes from t0 were mixed across seeds and generally small. On ele-fashion, the mean warm-end change was −0.046 percentage points for W1 and −0.030 points for W2; W1/W2 Validation Accuracy moved by about −0.014 points. Stage 1 is exploratory and is not eligible for the main checkpoint.

## Stage 2 trajectories and checkpoint choice

The main Best Accuracy checkpoint is selected from t0 plus joint epochs; Best Macro-F1 is selected independently over the same horizon. All 27 Best Macro-F1 checkpoints came from a joint epoch. For ele-fashion, every Best Accuracy checkpoint remained t0 for W0, W1, and W2 across all three seeds. This is the intended fallback and means none of the conditions improved Validation Accuracy enough to displace the shared starting state.

On Movies, the Best Accuracy checkpoint came from a joint epoch for two of three seeds in each condition. On Grocery, W0 and W2 did so for two seeds, while W1 did so once. Across all datasets, accuracy-selected and F1-selected states differ in their Train/Validation fit and CE; `data/checkpoint_tradeoffs.csv` preserves the per-seed Train/Val Accuracy, Macro-F1, CE, gaps, epoch, and checkpoint hash. In ele-fashion, Best F1 improves Validation Macro-F1 over t0 by 0.70 pp for W0, 0.73 pp for W1, and 0.69 pp for W2 on average, while its Accuracy is 0.65–0.76 pp below t0. That tradeoff also appears in W0, so it is not a selector-specific effect.

The epoch table includes Train CE, Train Accuracy/Macro-F1, Validation CE, and Validation Accuracy/Macro-F1 at every Stage 1 and Stage 2 evaluation. `data/epoch_history.csv` is the run-level source; `data/warmstart_epoch_comparison.csv` adds checkpoint and stopping markers.

## What the configuration cannot isolate

Dropout (0.2 in Stage 2) and AdamW weight decay (1e-4 in both stages) were fixed across all conditions. Stage 1 used eval mode, so dropout was inactive while selector gradients were still computed. This experiment does not estimate a Dropout or weight-decay main effect or interaction. W0/W1/W2 share the same Stage 2 optimizer rules and maximum common-model update budget, but W1/W2 also have Stage 1 selector updates and additional selector parameters. Historical cold-start comparisons additionally change initialization, training budget, learning rate, optimizer state, and checkpoint history; they are descriptive only.

## Traceability

- Run-level metrics, timing, and GPU memory: `data/runs.csv`, `data/resource_profile.csv`.
- Per-epoch Train/Validation metrics and gradient/update norms: `data/epoch_history.csv`.
- Stage transitions, state hashes, and optimizer initialization: `data/stage_transition_audit.json`.
- Selector distribution and physical-neighbor comparisons: `data/selector_diagnostics.csv`.
- Physical, semantic, cross-modal, expert, and router contributions: `data/expert_router_diagnostics.csv`.
- Failed attempts and analysis/artifact corrections: `data/failures.json`.

No Test labels or Test metrics were used. Three seeds are paired descriptive evidence, not a significance test.
