# Experiment Plan

## Frozen matrix

| Dataset | Conditions | Seeds | Runs |
|---|---|---|---:|
| Movies | W0, W1, W2 | 42, 43, 44 | 9 |
| Grocery | W0, W1, W2 | 42, 43, 44 | 9 |
| ele-fashion | W0, W1, W2 | 42, 43, 44 | 9 |
| **Total** |  |  | **27** |

All three conditions for one dataset × seed use the same verified historical V2.1 C0 Best Accuracy checkpoint, classifier, official split, features, physical graph, and V7B semantic graph cache.

## Training schedule

| Stage | W0 | W1/W2 |
|---|---|---|
| t0 | Evaluate loaded C0 | Evaluate loaded C1/C2 with selector effect initialized to zero |
| Selector adaptation | Evaluate only; no update | 12 epochs, selector parameters only, eval-mode forward with gradients enabled, CE + 0.01 balance loss, LR 1e-3, weight decay 1e-4 |
| Joint optimization | Up to 90 epochs | Up to 90 epochs after Stage 1 |

Stage 2 uses a fresh AdamW optimizer with zero state, no scheduler, common-model and classifier LR 1e-4, selector LR 5e-4 where present, weight decay 1e-4, dropout 0.2, gradient clipping 1.0, validation every epoch, accuracy patience 30 after minimum epoch 20, and min-delta 1e-4. The three conditions have the same maximum number of common-parameter joint updates.

## Checkpoint policy

Main Best Accuracy and separate Best Macro-F1 are selected from `t0` plus Stage 2 epochs only. Stage 1 best metrics are exploratory diagnostics. If no Stage 2 epoch improves on `t0`, the selected checkpoint remains `t0` and is labeled accordingly. Early stopping follows Validation Accuracy. Test metrics are disabled.

## Batch gates

Run seed 42 first (nine runs). Continue seeds 43 and 44 only after checking finite train/validation metrics, nonzero selector updates for W1/W2, frozen inherited-state hashes during Stage 1, exact shared initialization, checkpoint reload, no OOM, correct split/cache identity, and absence of Test evaluation. Performance relative to historical single-run scores is not a gate.
