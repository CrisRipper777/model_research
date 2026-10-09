# V5A.1 Matched-Effect & Receiver-Local Ranking Repair

## Protocol

All reported effects are frozen-host exposed-message replay contrasts using the same local zero-change replay as their baseline. The V4A R0_raw seed-42 hosts, V5A receiver/edge sample, E0–E3 features, and train/validation/holdout receiver split are retained. No new MAG host or controller is trained.

Only training receiver indices and training labels enter effect construction and estimator fitting. EstimatorVal receivers select checkpoints through receiver-equal Huber or RankNet loss; EstimatorHoldout receivers are used for final descriptive metrics. Validation labels, test labels, and test metrics were not read.

The frozen source SHA is `76b5e208f2d23205a7cae24718ef441026aa3cf3`. The matched local zero-change replay reproduces the full-host cached baseline within 6.68e−6 maximum absolute logit error, 4.06e−6 fused-embedding error, and 6.20e−6 cross-entropy error over the datasets. The preflight zero-change effect was exactly zero.

## Main results

| Dataset | Target | Matched−old correlation | M edge pairwise | S edge pairwise | Ranker pairwise | Ranker top1 excess | Ranker gain vs random | Ranker oracle regret | Edge oracle gain | Modality oracle gain | Hop oracle gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | delete | 1 | 0.5034 | 0.5055 | 0.5165 | 0.03449 | 0.00068 | 0.00702 | 0.00709 | 0.00864 | 0.0116 |
| Movies | comp | 1 | 0.5011 | 0.5018 | 0.5127 | 0.02402 | 0.000217 | 0.0119 | 0.0102 | 0.00588 | 0.00927 |
| Grocery | delete | 1 | 0.524 | 0.5202 | 0.5596 | 0.08985 | -0.000274 | 0.00602 | 0.00641 | 0.00685 | 0.00759 |
| Grocery | comp | 1 | 0.5145 | 0.5208 | 0.5576 | 0.1021 | -0.00099 | 0.00984 | 0.00999 | 0.00535 | 0.00682 |
| ele-fashion | delete | 1 | 0.492 | 0.4939 | 0.5445 | 0.1008 | -2.05e-05 | 0.000723 | 0.000938 | 0.00333 | 0.00547 |
| ele-fashion | comp | 0.9999 | 0.4936 | 0.4926 | 0.5425 | 0.09328 | 0.000152 | 0.00531 | 0.00285 | 0.00149 | 0.00283 |

## Q1. Matched baseline effect

| Dataset | Target | Matched−old Spearman | Median absolute difference | P95 absolute difference | Sign disagreement |
|---|---|---:|---:|---:|---:|
| Movies | delete | 1 | 2.98e-07 | 1.609e-06 | 8.182e-05 |
| Movies | comp | 1 | 2.98e-07 | 1.609e-06 | 0.0002339 |
| Grocery | delete | 1 | 0 | 1.132e-06 | 0 |
| Grocery | comp | 1 | 0 | 1.192e-06 | 0.0005103 |
| ele-fashion | delete | 1 | 0 | 2.384e-07 | 0 |
| ele-fashion | comp | 0.9999 | 0 | 1.937e-07 | 0.0009035 |

## Q2–Q3. Magnitude-aware sign sensitivity

| Dataset | Target | τ | Text/Visual eligible fraction | Text/Visual sign disagreement | Hop strong mixed-sign fraction |
|---|---|---:|---:|---:|---:|
| Movies | delete | 0 | 1 | 0.4858 | 0.6971 |
| Movies | delete | 1e-06 | 0.9978 | 0.4855 | 0.6969 |
| Movies | delete | 1e-05 | 0.9821 | 0.4842 | 0.6937 |
| Movies | delete | 0.0001 | 0.8745 | 0.4838 | 0.6495 |
| Movies | delete | 0.001 | 0.5149 | 0.4706 | 0.4365 |
| Movies | comp | 0 | 1 | 0.4858 | 0.7407 |
| Movies | comp | 1e-06 | 0.9886 | 0.4861 | 0.738 |
| Movies | comp | 1e-05 | 0.939 | 0.4863 | 0.7204 |
| Movies | comp | 0.0001 | 0.733 | 0.4865 | 0.6112 |
| Movies | comp | 0.001 | 0.3357 | 0.487 | 0.3173 |
| Grocery | delete | 0 | 1 | 0.4253 | 0.416 |
| Grocery | delete | 1e-06 | 0.9677 | 0.4211 | 0.4052 |
| Grocery | delete | 1e-05 | 0.8288 | 0.4159 | 0.3529 |
| Grocery | delete | 0.0001 | 0.572 | 0.4062 | 0.2551 |
| Grocery | delete | 0.001 | 0.2872 | 0.3895 | 0.1346 |
| Grocery | comp | 0 | 1 | 0.3625 | 0.5177 |
| Grocery | comp | 1e-06 | 0.9374 | 0.3545 | 0.4967 |
| Grocery | comp | 1e-05 | 0.7634 | 0.3413 | 0.4238 |
| Grocery | comp | 0.0001 | 0.4919 | 0.329 | 0.283 |
| Grocery | comp | 0.001 | 0.237 | 0.3121 | 0.1463 |
| ele-fashion | delete | 0 | 1 | 0.5326 | 0.7106 |
| ele-fashion | delete | 1e-06 | 0.8201 | 0.5175 | 0.6379 |
| ele-fashion | delete | 1e-05 | 0.6224 | 0.5134 | 0.5123 |
| ele-fashion | delete | 0.0001 | 0.3956 | 0.5019 | 0.3368 |
| ele-fashion | delete | 0.001 | 0.1714 | 0.5084 | 0.171 |
| ele-fashion | comp | 0 | 1 | 0.5123 | 0.7363 |
| ele-fashion | comp | 1e-06 | 0.7 | 0.4838 | 0.6081 |
| ele-fashion | comp | 1e-05 | 0.4805 | 0.478 | 0.4443 |
| ele-fashion | comp | 0.0001 | 0.2594 | 0.4823 | 0.2522 |
| ele-fashion | comp | 0.001 | 0.08609 | 0.4705 | 0.08917 |

## Q4. Oracle leverage by action granularity

| Dataset | Target | Granularity | Groups | Oracle gain mean | Median | P90 | Fraction > 1e−4 |
|---|---|---|---:|---:|---:|---:|---:|
| Movies | delete | edge | 9600 | 0.00709 | 0.00141 | 0.0177 | 0.8631 |
| Movies | delete | modality | 18336 | 0.00864 | 0.00253 | 0.0204 | 0.9368 |
| Movies | delete | hop | 9168 | 0.0116 | 0.00393 | 0.0275 | 0.9769 |
| Movies | comp | edge | 9600 | 0.0102 | 0.00211 | 0.0252 | 0.9056 |
| Movies | comp | modality | 17136 | 0.00588 | 0.00139 | 0.014 | 0.8773 |
| Movies | comp | hop | 8568 | 0.00927 | 0.00271 | 0.022 | 0.9533 |
| Grocery | delete | edge | 9000 | 0.00641 | 0.000383 | 0.0165 | 0.6654 |
| Grocery | delete | modality | 17240 | 0.00685 | 0.000526 | 0.0167 | 0.7216 |
| Grocery | delete | hop | 8620 | 0.00759 | 0.000645 | 0.018 | 0.7396 |
| Grocery | comp | edge | 9000 | 0.00999 | 0.000568 | 0.0236 | 0.7048 |
| Grocery | comp | modality | 15740 | 0.00535 | 0.000326 | 0.0122 | 0.648 |
| Grocery | comp | hop | 7870 | 0.00682 | 0.000507 | 0.0161 | 0.7046 |
| ele-fashion | delete | edge | 6480 | 0.000938 | 1.94e-05 | 0.00128 | 0.3239 |
| ele-fashion | delete | modality | 12808 | 0.00333 | 8.56e-05 | 0.00494 | 0.4841 |
| ele-fashion | delete | hop | 6404 | 0.00547 | 0.000218 | 0.00838 | 0.5795 |
| ele-fashion | comp | edge | 6480 | 0.00285 | 3.79e-05 | 0.00248 | 0.3974 |
| ele-fashion | comp | modality | 10048 | 0.00149 | 1.74e-05 | 0.00143 | 0.3165 |
| ele-fashion | comp | hop | 5024 | 0.00283 | 6.58e-05 | 0.00349 | 0.4522 |

## Execution and audit

The frozen V5A.1 source is `76b5e208f2d23205a7cae24718ef441026aa3cf3`. Targeted V5A.1/replay tests passed (84); the full suite passed (306). Movies preflight used 10 receivers and 18 edges, reconstructed the local baseline within 3.34e−6 maximum logit error, produced exactly zero zero-change effect, and returned finite singleton, bundle, pointwise, and ranking paths. The 20-run Movies smoke campaign completed with finite outputs at no more than three epochs per model.

The formal campaign completed all 180 fixed estimator runs: 36 MatchedMultiTarget, 72 MatchedSingleTarget, and 72 receiver-local RankNet. Movies and Grocery preparation/training ran on `cuda:1`. The ele-fashion frozen-host replay cache and its 60 estimator runs ran on CPU with `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` after `cuda:1` ran out of memory during full-graph host preparation and CUDA was unavailable at the estimator-stage start. No scientific source or configuration changed after freeze. The manifest records this mixed execution; all three dataset samples cross-check exactly against V5A raw IDs.

No validation labels, test labels, or test metrics were read. The only supervised targets use `train_idx` and training labels; validation receiver data selected estimator checkpoints, and holdout receivers were used only for final descriptive metrics. Undefined correlation or pairwise statistics are stored as empty CSV cells (for example, a constant vector or an all-tied pair group), never as NaN/Inf tokens.

## Q5. MatchedSingleTarget versus MatchedMultiTarget

The main table above gives the edge pairwise means for M and S. The table below reports paired S−M differences over the same 12 feature-family × seed cells per dataset and target; standard deviations are across those fixed cells. A negative MAE difference favors S.

| Dataset | Target | S−M edge pairwise accuracy | S−M MAE |
|---|---|---:|---:|
| Movies | delete | +0.0021 ± 0.0083 | -0.00011 ± 0.00019 |
| Movies | comp | +0.0007 ± 0.0103 | -0.00005 ± 0.00009 |
| Grocery | delete | -0.0038 ± 0.0114 | -0.00007 ± 0.00020 |
| Grocery | comp | +0.0063 ± 0.0138 | -0.00008 ± 0.00011 |
| ele-fashion | delete | +0.0020 ± 0.0130 | +0.00003 ± 0.00008 |
| ele-fashion | comp | -0.0010 ± 0.0237 | -0.00004 ± 0.00003 |

S has slightly lower mean MAE in five of six dataset-target cells, but the paired edge-ranking differences are small and change sign across datasets and targets. This is mixed, weak descriptive evidence for negative transfer; the fixed single-target fits do not show a consistent ranking advantage over M.

## Q6. Receiver-local RankNet versus pointwise ranking

The paired differences below compare RankNet against S on identical dataset, feature family, target, and seed cells (12 pairs per row). `Top1 excess` is above each candidate group’s `1/group_size` chance rate.

| Dataset | Target | RankNet edge pairwise | S edge pairwise | RankNet−S pairwise Δ (mean ± SD) | RankNet top1 excess | RankNet−S top1 excess Δ (mean ± SD) |
|---|---|---:|---:|---:|---:|---:|
| Movies | delete | 0.5165 | 0.5055 | +0.0109 ± 0.0188 | 0.0345 | +0.0244 ± 0.0283 |
| Movies | comp | 0.5127 | 0.5018 | +0.0109 ± 0.0086 | 0.0240 | +0.0269 ± 0.0175 |
| Grocery | delete | 0.5596 | 0.5202 | +0.0393 ± 0.0274 | 0.0899 | +0.0589 ± 0.0430 |
| Grocery | comp | 0.5576 | 0.5208 | +0.0368 ± 0.0169 | 0.1021 | +0.0674 ± 0.0171 |
| ele-fashion | delete | 0.5445 | 0.4939 | +0.0506 ± 0.0151 | 0.1008 | +0.0690 ± 0.0134 |
| ele-fashion | comp | 0.5425 | 0.4926 | +0.0499 ± 0.0245 | 0.0933 | +0.0535 ± 0.0363 |

All six RankNet−S mean pairwise differences are positive (+0.0109 to +0.0506), and the paired top1-excess differences are also positive (+0.0244 to +0.0691). This is suggestive that aligning training with receiver-local edge comparisons improves the ranking metrics. Selection gain below remains near zero or negative in several cells, so the ranking improvement does not translate into a consistent reduction in selected effect.

## Q7–Q8. Selected effect gain and distance from the oracle

For each holdout `(receiver, modality, hop)` candidate group, `model_gain_vs_random = random_expected − model_selected_effect`; positive values mean the model selected a lower effect than the group mean. The table averages these group means equally over the 12 fixed estimator/seed runs per model. `RankNet regret` is selected effect minus the group oracle minimum.

| Dataset | Target | M gain vs random | S gain vs random | RankNet gain vs random | RankNet oracle regret |
|---|---|---:|---:|---:|---:|
| Movies | delete | -0.000376 | -0.000225 | +0.000680 | 0.007025 |
| Movies | comp | -0.001037 | -0.001299 | +0.000217 | 0.011914 |
| Grocery | delete | +0.000384 | +0.000221 | -0.000274 | 0.006019 |
| Grocery | comp | +0.000113 | +0.000312 | -0.000990 | 0.009842 |
| ele-fashion | delete | -0.000019 | -0.000039 | -0.000021 | 0.000723 |
| ele-fashion | comp | -0.003534 | -0.003033 | +0.000152 | 0.005307 |

RankNet selected-effect gain is positive but very small for Movies and fashion compensation, negative for both Grocery targets and nearly zero for fashion deletion. It does not show a consistent advantage over random selection. Mean RankNet oracle regret ranges from 0.00072 to 0.01191 in dataset-specific target units; see the full `oracle_leverage.csv` and `ranker_selection_metrics.csv` for distributions and group counts. No significance tests were run.

## Q9. Delete versus compensation predictability

The RankNet delete-minus-compensation edge pairwise difference is paired over the 12 fixed feature-family × seed cells. RankNet favors deletion modestly in each dataset, while the pointwise M/S target differences and selected-effect gains are mixed.

| Dataset | RankNet delete−comp edge pairwise Δ (mean ± SD) | RankNet delete gain | RankNet comp gain |
|---|---:|---:|---:|
| Movies | +0.0038 ± 0.0105 | +0.000680 | +0.000217 |
| Grocery | +0.0020 ± 0.0206 | -0.000274 | -0.000990 |
| ele-fashion | +0.0020 ± 0.0166 | -0.000021 | +0.000152 |

The direction of the small RankNet pairwise difference is consistent, but it is not a consistent gain-over-random pattern. Delete and compensation therefore remain distinct prediction targets; the evidence does not support treating one target’s predictability as a proxy for the other.

## E0–E3 feature ladder

Values are macro means of holdout edge pairwise accuracy across the six dataset × target cells and three seeds in each feature family (18 run-level values per family).

| Model | E0 heuristic | E1 unimodal pair | E2 multimodal pair | E3 host context |
|---|---:|---:|---:|---:|
| MatchedMultiTarget | 0.5063 | 0.5065 | 0.4999 | 0.5063 |
| MatchedSingleTarget | 0.5095 | 0.5060 | 0.5015 | 0.5063 |
| Receiver-local RankNet | 0.5325 | 0.5446 | 0.5400 | 0.5384 |

M/S pointwise results remain near chance in this macro summary. E1–E3 do not provide a stable monotonic improvement over E0; E2 is lower than E1 for all three model families in the macro means. RankNet has its highest macro score with E1, with dataset/target heterogeneity in the full rows. These are descriptive averages, not feature-family winner claims.

## Q10. Granularity and interpretation

The oracle tables show nonzero available leverage at edge, modality, and hop granularity. Modality/hop oracle gain is often comparable to or greater than edge oracle gain, especially for ele-fashion deletion; edge leverage is comparatively small there. The best post-hoc pointwise ordering summaries show some coarse modality/hop signal, but vary by dataset and target and are selected from fixed variants/seeds. RankNet improves edge pairwise ordering against S across all six target cells, while model-selected gains are close to zero or negative in several cells and oracle regret remains. Current evidence is therefore **inconclusive** about which action granularity should guide a controller: it is suggestive for receiver-local edge ranking as an objective, and it also leaves receiver-conditioned modality/hop adaptation plausible, but neither is supported as a consistently effective selection policy.

All quantities remain frozen-host exposed-message replay contrasts. They do not establish a causal effect or controller benefit. No V5B intervention, controller, new host, or main-model training was started.

Sensitivity thresholds and effect/pair-gap quartile boundaries are diagnostics only; quartiles are fitted on EstimatorTrain receivers and applied to EstimatorHoldout. RankNet is evaluated only within `(receiver, modality, hop)` candidate groups and has no cross-group score calibration. The audit sample contains up to 500 rows per dataset (1,500 total); complete singleton/bundle rows, features, checkpoints, and logs remain under ignored `outputs/mag_message_effect_v5a1/`.
