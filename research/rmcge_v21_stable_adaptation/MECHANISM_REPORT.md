# RM-CGE V2.1 Stable Adaptation: Mechanism Report

## Evidence boundaries

All selector and node-subgroup diagnostics are computed using Train/Validation protocol data only; node-level temporary tensors were not exported. Degree groups are observational and do not make low-degree receivers eligible for relative neighbor selection. Stage 1 and final-checkpoint statistics answer different questions: Stage 1 measures selector learning under a frozen shared representation, while the selected checkpoint diagnostics describe the later joint model.

## Common initialization and Stage 1

For each of nine Dataset × Seed pairs, W0/W1/W2 share exact inherited model and classifier states; W1/W2 selector parameters are matched. Across all nine triplets the maximum Validation-logit discrepancy from W0 is `1.79e-6` (audit tolerance `5e-5`), and the maximum t0 metric discrepancy is `2.38e-7`. Evaluation preserved CPU and CUDA RNG states.

All 18 W1/W2 runs kept inherited public model and classifier hashes unchanged through the 12 Stage 1 epochs. All 18 updated their edge scorer. W2 cross-delta coefficients had nonzero gradient and changed in all nine runs. Stage 1 used Eval mode with gradients enabled, so Dropout was inactive while the selector still received task loss gradients. These controls support that the selectors can be optimized on a stable physical/shared-expert representation.

| Dataset | W1 scorer grad norm (mean) | W2 scorer grad norm (mean) | W2 cross-delta grad norm | W2 cross-delta update RMS |
|---|---:|---:|---:|---:|
| Movies | 0.007513 | 0.011167 | 0.000806 | 0.004500 |
| Grocery | 0.002175 | 0.002739 | 0.000250 | 0.003995 |
| ele-fashion | 0.000498 | 0.001458 | 0.000181 | 0.004474 |

At warm end, mean absolute selector scores were W1 Text/Visual Intra 0.104/0.215, and W2 Text Intra/Cross 0.114/0.111 and Visual Intra/Cross 0.196/0.217. No measured warm-end score reached `|score| ≥ 0.99`. Receiver JS from physical weights on degree ≥2 nodes was only 0.00035–0.00252, with about eight effective neighbors. This is evidence of learned but modest redistribution, not replacement of the physical neighborhood.

## Selector movement at the selected Best-Accuracy checkpoints

Across selected Best-Accuracy checkpoints, saturation was rare: mean/max `|score| ≥ 0.99` fraction was 0.000607/0.01092 for W1 and 0.000095/0.00342 for W2. Mean absolute scores were about 0.076/0.150 for W1 Text/Visual Intra; for W2, 0.075 Text Intra, 0.032 Text Cross, 0.139 Visual Intra, and 0.161 Visual Cross. Mean receiver JS remained small (roughly 0.0010–0.0037); effective neighbors were roughly 7.9–8.1.

Final Stage 2 can produce larger score movement on Best-F1 checkpoints, including an isolated W1 maximum saturation fraction of 5.8%. The W1 aggregate Best-F1 saturation mean is 0.00323; W2 remains near zero in that selected subset. That isolated tail is not evidence of broad saturation, but it merits monitoring if the protocol is extended. Per-run, per-path time series remain in `data/selector_diagnostics.csv`.

## Physical, semantic, and cross-modal contribution scales

At selected Best-Accuracy checkpoints, the inherited intrinsic feature RMS is about 0.998 and first physical-profile contribution RMS about 0.773. The inherited cross-modal base has raw RMS about 0.55 and weighted per-expert RMS about 0.28; the KNN path is also present with weighted RMS about 0.12. In comparison, W1 selected intra residual RMS is about 0.019 raw and 0.0068 weighted. W2 intra residual is about 0.016 raw and 0.0061 weighted; its cross-delta residual is about 0.008 raw and 0.0033 weighted. These learned selector-conditioned residuals are small relative to the inherited paths. This supports describing V2.1 as a reweighting around a strong shared physical/semantic base.

Across modality rows, the selected Best-Accuracy expert output RMS stays near 1.10 in all conditions. W1/W2 therefore modify route-specific propagation while preserving a comparable output scale. Exact aggregate rows, expert coefficients, and contribution vectors are in `data/expert_router_diagnostics.csv`.

## Router behavior

Grocery Visual Node Router remains highly concentrated (maximum dense expert probability approximately 0.9997; entropy approximately 0.0023) in W0, W1, and W2, and the same concentration is visible at the inherited C0 state. It is not created by the new selector. Fashion routes remain more diffuse (maximum probability around 0.32; entropy around 1.08–1.17). Router entropy, dense probabilities, route shares, and Top-2 flips are exported per checkpoint. A low-entropy inherited router may constrain where a new selector’s message redistribution can act, but this study did not intervene on the router.

## Best Accuracy versus Best F1

For ele-fashion, all nine Best-Accuracy selections are t0 and consequently have no condition-level prediction flips or classwise changes. Best-F1 checkpoints are Stage 2 states and select different classification operating points: compared with Best Accuracy, mean Accuracy is lower by about 0.65–0.76 pp, Macro-F1 higher by 0.69–0.73 pp, and CE lower by about 0.010–0.013. This is a real Accuracy/Macro-F1 tradeoff seen in W0 as well as W1/W2.

## Fashion class balance and degree analysis

Historical cold-start failure (documented in `CLASSWISE_DIAGNOSTICS.md`) is concentrated in classes 11 and 1 for Best Accuracy. C1−C0 class F1 changes are about −10.1 pp and −4.5 pp; precision rises (+18.1/+8.0 pp) as recall falls (−8.7/−6.1 pp), and prediction counts fall by about 62/108 per seed. Best-F1 cold-start losses are led by classes 1, 7, and 11. The pattern supports a changed class prediction preference but does not identify its optimizer cause.

In the warm Best-F1 comparison, changes are much smaller and mixed: W1−W0 improves F1 most for classes 11 (+0.43 pp), 7 (+0.26 pp), and 10 (+0.23 pp), while class 3 (−0.23 pp), 0 (−0.20 pp), and 8 (−0.18 pp) fall most. For class 11, W1−W0 changes Precision by −1.08 pp and Recall by +0.96 pp; for class 1, Precision is −0.60 pp and Recall +0.50 pp. Thus the historical cold-start Precision-up/Recall-down pattern is not simply reproduced in warm W1. W2−W0 has its largest mean drops for class 4 (−0.27 pp) and class 1 (−0.06 pp), with small gains for classes 7/10/11. Fashion has 11 labels observed in Train/Validation; class 5 is absent. The smallest observed Validation support is 45 (class 3), followed by 95 (class 7), so those per-class estimates deserve caution even though no observed class has support below 10.

Warm Best-F1 prediction changes are correspondingly small: mean prediction flip rates are 0.60% for W1−W0, 0.61% for W2−W1, and 0.13% for W2−W0. For Best Accuracy they are zero because all methods retain the same t0 checkpoint. At Best-F1, W1−W0 class-level macro changes do not point to one single collapsing minority class.

At Best-F1, Fashion degree-1 nodes show W1−W0 Accuracy −0.23 pp and F1 −0.13 pp, while degree ≥2 groups combined show Accuracy −0.01 pp and F1 +0.21 pp. W2−W0 changes are near zero: degree-1 F1 −0.04 pp versus degree ≥2 F1 +0.03 pp. There is no sizeable degree-1-only deterioration in the warm comparison. Degree-1 nodes cannot choose among multiple physical neighbors, so any changes there are indirect effects of shared parameters or higher-order propagation. The historical cold-start failure also was not degree-1-only: the larger Fashion F1 loss was in degree 2–3 and degree ≥2 combined.

## Prediction and CE transitions

At Fashion Best-F1, CE improves for roughly 45.5% of nodes under W1−W0 and 45.8% under W2−W0, while CE worsens for roughly 54%; the mean CE difference is close to zero. Thus aggregate CE and macro-F1 move by small amounts even though a small subset of node labels changes. Per-seed wrong-to-right/right-to-wrong counts, flip rates, and CE-change tails are in `data/warmstart_prediction_transitions.csv`.

## Dropout and Weight Decay

Dropout 0.2 and AdamW weight decay 1e-4 were fixed, not experimentally varied. Dropout was inactive in Stage 1 eval mode and active in Stage 2. This is not a factorial study and cannot estimate a Dropout effect, Weight Decay effect, or their interaction. The protocol also resets AdamW for every Stage 2 run; historical optimizer momentum is absent by design.

## Remaining uncertainty

The study establishes successful selector learning and a reproducible common-start comparison, but it does not show a reliable Validation classification gain from the selector stages—especially for Fashion, where accuracy-selected checkpoints stay at t0. Small three-seed differences are descriptive. The run does not isolate scorer capacity, optimizer interaction, dropout, inherited router concentration, or insufficient content expressiveness as the unique cause.
