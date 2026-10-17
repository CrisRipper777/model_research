# CG-CGE V1 Decision Memo

## Decision

The 18-run Validation study is complete. The results do not establish a consistent winner across datasets. B1 is the simpler reference condition if this model line is revisited; B2's Role-FiLM adds 8,192 trainable parameters but does not show a stable benefit over B1. Stop at CG-CGE V1. These results do not authorize a successor model or changes to the shared Router or experts.

All figures below use three-seed mean paired differences and sample standard deviations. They are descriptive; no population-level significance claim is made. Test was disabled throughout.

## Comparison with matched A2 baseline

| Dataset | Contrast | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Direction across seeds |
|---|---|---:|---:|---|
| Movies | B1 − A2 | +0.560 ± 0.125 | +1.657 ± 1.489 | Accuracy + in 3/3; F1 + in 3/3 |
| Movies | B2 − A2 | +0.510 ± 0.159 | +1.628 ± 2.099 | Accuracy + in 3/3; F1 + in 2/3 |
| Grocery | B1 − A2 | +0.117 ± 0.183 | +0.140 ± 0.911 | Accuracy + in 2/3; F1 + in 1/3 |
| Grocery | B2 − A2 | +0.146 ± 0.077 | −0.118 ± 0.775 | Accuracy + in 3/3; F1 + in 1/3 |
| ele-fashion | B1 − A2 | −0.089 ± 0.287 | −0.808 ± 0.747 | Accuracy + in 1/3; F1 lower in 3/3 |
| ele-fashion | B2 − A2 | −0.232 ± 0.304 | −1.071 ± 0.887 | Accuracy + in 1/3; F1 lower in 3/3 |

CG-CGE has small positive paired Accuracy means on Movies and Grocery, with mixed Macro-F1 evidence on Grocery; both variants have lower Macro-F1 on ele-fashion. These outcomes do not support a general claim that complementary experts improve MAG classification.

## B2 versus B1

| Dataset | B2 − B1 Δ Accuracy (pp) | B2 − B1 Δ Macro-F1 (pp) | Interpretation |
|---|---:|---:|---|
| Movies | −0.050 ± 0.092 | −0.030 ± 0.954 | Essentially tied on means; seed-level F1 varies |
| Grocery | +0.029 ± 0.128 | −0.259 ± 0.141 | Accuracy is small and mixed; F1 lower in all three seeds |
| ele-fashion | −0.143 ± 0.118 | −0.263 ± 0.177 | Both metrics lower in all three seeds |

At the Best-F1 checkpoint, B2's mean Macro-F1 is also slightly below B1 on all three datasets (by 0.090, 0.044 and 0.626 pp, respectively). B2 does not earn the extra conditioning parameters in this matrix. B1 is the more defensible reference if the line is resumed, but the evidence is too small to call it a universal winner.

## Optimization and mechanism evidence

- Both variants used dropout 0.2, AdamW learning rate 1e-3 and weight decay 1e-4. Since these settings were fixed, this study does not estimate dropout or weight-decay main effects or their interaction.
- Movies shows large mean Train–Validation gaps at the Best-Accuracy checkpoint (about 17 pp Accuracy and 23 pp Macro-F1); Grocery gaps are also substantial (about 9 pp and 14 pp). ele-fashion has smaller Accuracy gaps (about 2–3 pp) but 7–10 pp Macro-F1 gaps. These indicate validation overfit and classwise/checkpoint sensitivity.
- Best-F1 checkpoints trade away Validation Accuracy for higher Macro-F1. Relative to Best-Accuracy checkpoints, mean Accuracy falls about 0.48–2.25 pp while Macro-F1 rises about 0.41–1.59 pp across dataset-condition cells. Reported checkpoint selection therefore matters.
- The No-completion weight varies per node from 0 to 0.424 and is not a constant option, although it is near zero in several cells. The measured Cross residual/unimodal RMS ratio is 0.020–0.254: the cross path remains smaller than the unimodal backbone, but is not numerically zero.
- Physical, Semantic and Complementary calls share the same four expert modules. Their Top-2 selection shares differ by call path, which is evidence of different routing activity but not causal expert specialization. B2 Role-FiLM scale and shift updates are nonzero in all nine runs.
- First-epoch and Best-Accuracy-epoch auxiliary CE, physical/semantic response RMS, source weights, channel gates, attention entropy, expert usage and Role-FiLM updates are retained in `PILOT_REPORT.md` and the CSV evidence. These observations show execution and activation, not causal utility.

## Reproducibility and evidence boundary

All Dataset × Seed cells have matched initial model/head parameters across B1/B2 and matched split, features and semantic-cache fingerprints. All 18 accepted runs have finite histories and metrics. The 36 selected checkpoints passed hash and size verification; no OOM occurred. Three seed pairs provide limited evidence about optimization variability.

One failed training attempt and its accepted retry had identical initial state and data fingerprints but different full-run Validation trajectories. Exact trajectory reproducibility remains unresolved. Dropout RNG behavior and sparse/index GPU operations are possible contributors; this study did not isolate the source. Historical failed records remain preserved in `data/failures.json` and `data/preflight_attempts.json`.

The evidence supports internal paired comparisons under the frozen V1 protocol. It does not support Test-set claims, significance claims, causal mechanism claims, dropout/weight-decay conclusions, or a claim of exact GPU trajectory replay. No Test evaluation, extra seeds, hyperparameter search, or successor implementation was performed.
