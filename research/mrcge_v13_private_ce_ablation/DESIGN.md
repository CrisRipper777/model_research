# MR-CGE V1.3-A Private Auxiliary CE Ablation

## Question and single variable

This ablation compares the original `src.models.mrcge_v13.Model` with the same-seed V1.3-A Joint View history. The only new-run configuration change is `task.loss.unimodal_weight`: the historical value is 0.3, and this run uses 0.0 (`no_private_ce`). The model, Private Text/Visual heads, projections, initialization order, data, split, cache, all other loss terms, optimizer, and training settings remain as in the frozen V1.3-A screen.

When the weight is zero, the existing trainer's objective is exactly `main_ce + aux_weight * aux_loss`. The model still computes and returns Private logits, but the trainer does not form their auxiliary CE. The Private representations remain connected to the final classifier and train through the main classification loss. Model parameters are not removed or reinitialized.

## Frozen model and training setup

- Model: `mrcge_v13`, `M_repaired`, `joint_view`, sender private ratio 0.0; the original V1.3 model implementation is loaded directly.
- Hidden dimension 256; 4 experts, Top-2; 3 physical and 1 semantic hop; expert bottleneck 64; Joint View and SameNode enabled; existing Early Cross and Late Completion unchanged.
- AdamW, learning rate 0.001, weight decay 0.0001, 300 epochs, patience 30, minimum stopping epoch 30, minimum delta 0.0001, gradient clip 1.0.
- Backbone and fusion dropout 0.2; Late Cross dropout 0.15; V7B.1 semantic cache K=8.
- `aux_weight=1.0`, `unimodal_weight=0.0`, and model `balance_weight=0.01`; the existing model already scales Balance Loss, so it is not scaled again by the trainer.
- Split seed 42, `auto_generate_splits=false`, `evaluate_test=false`; Test labels are masked to `-100` before Trainer entry. Validation selects checkpoints and early stopping. No Test metrics or Link Prediction.

## Fixed matrix and comparison

| Dataset | Seeds | Historical paired condition |
|---|---|---|
| Movies | 42, 43, 44 | V1.3-A Joint View, same seed and fixed split |
| Grocery | 42, 43 | V1.3-A Joint View, same seed and fixed split |

Movies seed 44 uses the V1.3-A `joint_view` row in the Physical Basis fixed-split comparison. Historical Best-Accuracy and Best-F1 checkpoint hashes are verified before training. No historical model is fine-tuned or retrained. The five new cells run serially, each in its own attempt directory and child process.

Report each seed's Validation Accuracy, Macro-F1, and CE, plus paired differences: `100 * (new - historical)` percentage points for Accuracy and Macro-F1, and unscaled CE difference. Report per-dataset means and population standard deviations. Also compare Train/Validation Accuracy, CE, and Train−Validation gaps. Do not pool Movies and Grocery into a single accuracy score. Conclusions use `KEEP_0.3`, `CONSIDER_0`, `MIXED`, or `NEAR_TIE`; no automatic threshold, statistical-significance claim, or additional weight search is used.

## Execution and evidence

Before formal training, run one CPU full-model forward/backward smoke check with the zero-weight loss path and verify finite loss/gradients while Private heads remain present. The runner verifies fixed data/cache fingerprints and historical checkpoint hashes, creates per-run config snapshots, keeps both existing Best-Accuracy and Best-F1 checkpoint paths, reloads both checkpoints (at minimum Best-Accuracy is mandatory), checks reloaded validation metrics against Trainer selection metadata, and writes compact CSV/JSON evidence. Training code is committed before any formal cell; all five runs must share that frozen source SHA.

Use physical GPU 1 only and the existing fail-closed ownership guard. Require a fresh snapshot, no unexpected process, at least 8 GiB free, acceptable temperature/utilization, and serialize all five cells. Never stop another process. Full checkpoints remain local and are excluded from the research commit.
