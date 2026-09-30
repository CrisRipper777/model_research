# P1.3 — Joint-Readout Operator Counterfactual

## Question and scope

P1.2 trained separate linear classifiers for `smooth`, `absdiff`, and `product`. P1.3 removes that readout confound: for each dataset and seed, all six modality/operator contexts and the same frozen semantic features are concatenated and evaluated by one joint linear classifier. The question is whether the same directed physical message still has different marginal validation utility across the three tested propagation functions under that common predictor.

This is a validation-only controlled diagnostic. The same validation labels select the joint head by accuracy and define the counterfactual CE utility, so the utility analysis is not independent confirmation. No test evaluation or link prediction was run. The six fixed contexts use the P1.2 formulas, incoming mean aggregation, the same no-self-loop physical graph, and the original target degree. The feature order is `[H_T, H_V, S_T, D_T, P_T, S_V, D_V, P_V]`, with 128 dimensions per block and 1024 total. Exactly one `Linear(1024, num_classes)` is trained per dataset×seed.

For each validation-directed edge, all six removals share the same full joint logit and `CE_full`. A removal changes one message contribution in one context block by `O_r(i,j,m)/d_i`; every other block is unchanged. Primary utility is raw `CE_removed − CE_full`. Positive utility means that this tested message contribution helps the fixed full joint prediction under validation CE; it is not a claim that the edge should be routed or retained by a future model.

For the complete four-cell transition matrices, “positive” means `U>0` and “nonpositive” means `U<=0`, so each edge enters exactly one cell. Exact zeros are reported separately. The requested joint prevalence and conditional transition use the strict predicates `U_smooth<0` and `U_alternative>0`. For the P1.2-compatible modality sign comparison, the sign split is positive versus nonpositive, matching P1.2’s `(U_text>0) != (U_visual>0)` implementation. The CSV also reports a zero-aware three-valued `sign(U)` disagreement.

## Execution and correctness

The source was fetched from `origin/exp/p11_p12_operator_rescue` at `17ca9420b614998f0192dac558c1a31d4c28a32e`, then checked out to `exp/p13_joint_readout_operator_probe`. The environment was Python 3.12.13, PyTorch 2.4.0+cu121, PyG 2.7.0, and NVIDIA GeForce RTX 3090 (`cuda:0`).

All nine dataset×seed runs completed: Movies, Grocery, and ele-fashion; seeds 42, 43, and 44. The Movies/42 smoke passed before the full campaign. Across all nine runs:

- P0 semantic metrics reproduced to maximum absolute difference `2.22e-16`; semantic parameters and `H_T/H_V` remained unchanged.
- Feature dimension/order, one-head training, no self-loops, and original-degree denominator checks passed.
- P1.3 validation edges matched P0 and P1.2 in row count, `(src,dst,dst_degree)` values, and order for every run.
- Fast removal logits matched explicit feature-edit/re-forward logits for all 3×2 channels at `rtol=1e-7`, `atol=1e-8`; maximum absolute error across runs was `1.60e-14`, well within the required `rtol=1e-7`, `atol=1e-8` check.
- No test indices or test labels were accessed. The P1.3 correctness tests and P1.2 regression tests passed: 17 passed, with one upstream PyG deprecation warning.

## 1. Same-model sign transitions

The strict S-negative/alternative-positive joint prevalence is nonzero in all six dataset×modality groups for both alternatives. Values below are three-seed means with population seed SD in parentheses. The conditional column is `P(U_alt>0 | U_smooth<0)`.

| Dataset | Modality | S−/D+ | P(D+\|S−) | S−/P+ | P(P+\|S−) |
|---|---|---:|---:|---:|---:|
| Movies | Text | 0.230 (0.008) | 0.558 (0.009) | 0.160 (0.016) | 0.390 (0.042) |
| Movies | Visual | 0.212 (0.019) | 0.573 (0.037) | 0.179 (0.018) | 0.481 (0.030) |
| Grocery | Text | 0.107 (0.002) | 0.508 (0.008) | 0.094 (0.009) | 0.448 (0.045) |
| Grocery | Visual | 0.118 (0.005) | 0.505 (0.015) | 0.101 (0.010) | 0.432 (0.036) |
| ele-fashion | Text | 0.129 (0.020) | 0.610 (0.032) | 0.128 (0.019) | 0.607 (0.031) |
| ele-fashion | Visual | 0.289 (0.088) | 0.544 (0.176) | 0.328 (0.034) | 0.613 (0.070) |

The complete four-cell transitions are shown below as fractions of all aligned edges. Cell order is `S>0 / alt>0`, `S>0 / alt<=0`, `S<=0 / alt>0`, `S<=0 / alt<=0`. Exact-zero utilities are in the nonpositive cells; strict joint metrics in the preceding table exclude exact-zero smooth values.

| Dataset | Modality | Alternative | S+ / A+ | S+ / A<=0 | S<=0 / A+ | S<=0 / A<=0 |
|---|---|---|---:|---:|---:|---:|
| Movies | Text | absdiff | 0.192 | 0.396 | 0.230 | 0.182 |
| Movies | Text | product | 0.448 | 0.140 | 0.160 | 0.251 |
| Movies | Visual | absdiff | 0.365 | 0.265 | 0.212 | 0.158 |
| Movies | Visual | product | 0.413 | 0.217 | 0.179 | 0.192 |
| Grocery | Text | absdiff | 0.269 | 0.520 | 0.107 | 0.104 |
| Grocery | Text | product | 0.681 | 0.109 | 0.094 | 0.116 |
| Grocery | Visual | absdiff | 0.304 | 0.462 | 0.118 | 0.116 |
| Grocery | Visual | product | 0.644 | 0.122 | 0.101 | 0.133 |
| ele-fashion | Text | absdiff | 0.286 | 0.503 | 0.129 | 0.081 |
| ele-fashion | Text | product | 0.550 | 0.239 | 0.128 | 0.082 |
| ele-fashion | Visual | absdiff | 0.109 | 0.355 | 0.289 | 0.247 |
| ele-fashion | Visual | product | 0.348 | 0.115 | 0.328 | 0.208 |

This answers question 1: sign transitions remain under a single common predictor, but their prevalence depends on dataset, modality, and tested function.

## 2. P1.2 to P1.3 comparison

The first two pairs below are P1.2 independent-head → P1.3 joint-head means. The last two pairs are the corresponding conditional probabilities. All transition prevalences remain present; the result is **largely retained** overall. The clearest attenuation is ele-fashion Visual joint prevalence, while its conditional transitions remain high. No dataset×modality transition disappears.

| Dataset | Modality | S−/D+ P1.2→P1.3 | P(D+\|S−) P1.2→P1.3 | S−/P+ P1.2→P1.3 | P(P+\|S−) P1.2→P1.3 |
|---|---|---:|---:|---:|---:|
| Movies | Text | 0.217→0.230 | 0.539→0.558 | 0.132→0.160 | 0.329→0.390 |
| Movies | Visual | 0.210→0.212 | 0.591→0.573 | 0.138→0.179 | 0.387→0.481 |
| Grocery | Text | 0.105→0.107 | 0.509→0.508 | 0.077→0.094 | 0.375→0.448 |
| Grocery | Visual | 0.111→0.118 | 0.474→0.505 | 0.096→0.101 | 0.411→0.432 |
| ele-fashion | Text | 0.104→0.129 | 0.547→0.610 | 0.064→0.128 | 0.338→0.607 |
| ele-fashion | Visual | 0.325→0.289 | 0.453→0.544 | 0.405→0.328 | 0.567→0.613 |

The “retained” reading applies to the sign-transition premise; it does not say the numerical utilities or selected channels are equal between different heads. This is not a performance comparison between the P1.2 and P1.3 probes.

## 3. Positive-best channel distribution

For each edge and modality, the preferred channel is the largest of the three raw utilities if that maximum is positive; otherwise it is `none`. `none` means only that none of the three tested channels has positive marginal utility. Values are three-seed means (population SD).

| Dataset | Modality | Smooth | AbsDiff | Product | None |
|---|---|---:|---:|---:|---:|
| Movies | Text | 0.226 (0.034) | 0.232 (0.014) | 0.443 (0.043) | 0.099 (0.006) |
| Movies | Visual | 0.439 (0.014) | 0.219 (0.024) | 0.255 (0.034) | 0.087 (0.010) |
| Grocery | Text | 0.533 (0.012) | 0.103 (0.002) | 0.309 (0.009) | 0.055 (0.007) |
| Grocery | Visual | 0.513 (0.033) | 0.137 (0.013) | 0.283 (0.024) | 0.067 (0.006) |
| ele-fashion | Text | 0.597 (0.082) | 0.100 (0.015) | 0.264 (0.067) | 0.039 (0.003) |
| ele-fashion | Visual | 0.190 (0.026) | 0.188 (0.061) | 0.516 (0.013) | 0.106 (0.058) |

The positive-best distributions vary substantially by dataset and modality. They are descriptive argmaxes, not ground-truth operator labels.

## 4. Within-node functional diversity

Eligibility is restricted to validation targets with original in-degree at least five. `none` edges do not count as positive preferences; targets with no positive-preferred incoming edge are recorded separately.

| Dataset | Modality | Multi-function neighborhood ratio | Median distinct positive functions | No-positive target fraction |
|---|---|---:|---:|---:|
| Movies | Text | 0.562 (0.046) | 1.667 (0.471) | 0.0065 (0.0011) |
| Movies | Visual | 0.817 (0.025) | 2.000 (0.000) | 0.0021 (0.0021) |
| Grocery | Text | 0.787 (0.009) | 2.000 (0.000) | 0.0016 (0.0009) |
| Grocery | Visual | 0.826 (0.018) | 2.000 (0.000) | 0.0002 (0.0003) |
| ele-fashion | Text | 0.452 (0.064) | 1.333 (0.471) | 0.0021 (0.0007) |
| ele-fashion | Visual | 0.597 (0.048) | 2.000 (0.000) | 0.0087 (0.0095) |

This provides within-target evidence: different physical neighbors of the same target often have different positive-best tested functions. The ratios range from 0.45 to 0.83 across groups.

## 5. Text/Visual preferred-function disagreement

Overall disagreement includes `none`; active disagreement is conditional on both modalities having a positive-best function.

| Dataset | Overall disagreement | Active-function disagreement |
|---|---:|---:|
| Movies | 0.741 | 0.702 |
| Grocery | 0.506 | 0.456 |
| ele-fashion | 0.724 | 0.678 |

The per-edge 4×4 table (rows Text: smooth/absdiff/product/none; columns Visual in the same order) is in `data/modality_preference_disagreement.csv`; it includes seed-level counts/fractions and three-seed cell means. The matrices below show the mean fraction of matched edges per cell.

**Movies**

| Text \\ Visual | Smooth | AbsDiff | Product | None |
|---|---:|---:|---:|---:|
| Smooth | 0.106 | 0.056 | 0.048 | 0.017 |
| AbsDiff | 0.082 | 0.037 | 0.083 | 0.030 |
| Product | 0.211 | 0.100 | 0.104 | 0.028 |
| None | 0.040 | 0.026 | 0.021 | 0.012 |

**Grocery**

| Text \\ Visual | Smooth | AbsDiff | Product | None |
|---|---:|---:|---:|---:|
| Smooth | 0.337 | 0.051 | 0.123 | 0.022 |
| AbsDiff | 0.028 | 0.028 | 0.028 | 0.020 |
| Product | 0.135 | 0.040 | 0.119 | 0.014 |
| None | 0.013 | 0.018 | 0.013 | 0.011 |

**ele-fashion**

| Text \\ Visual | Smooth | AbsDiff | Product | None |
|---|---:|---:|---:|---:|
| Smooth | 0.112 | 0.094 | 0.344 | 0.047 |
| AbsDiff | 0.022 | 0.030 | 0.031 | 0.018 |
| Product | 0.041 | 0.054 | 0.132 | 0.037 |
| None | 0.015 | 0.011 | 0.010 | 0.003 |

## 6. Operator-wise Text/Visual utility sign disagreement

P1.2 used a positive-versus-nonpositive split, so the middle column below preserves that comparison exactly. A zero-aware `sign(U)∈{-1,0,+1}` rate is also shown because P1.3 has exact zero utilities in some channels. The largest exact-zero fraction is 1.18% for ele-fashion Visual absdiff; smooth has no exact zero rows. Zero-aware disagreement remains similar to the P1.2-compatible rate.

| Dataset | Operator | P1.2 positive/nonpositive | P1.3 positive/nonpositive | P1.3 three-valued sign |
|---|---|---:|---:|---:|
| Movies | Smooth | 0.422 | 0.425 | 0.425 |
| Movies | AbsDiff | 0.660 | 0.575 | 0.576 |
| Movies | Product | 0.442 | 0.477 | 0.477 |
| Grocery | Smooth | 0.213 | 0.215 | 0.215 |
| Grocery | AbsDiff | 0.689 | 0.497 | 0.502 |
| Grocery | Product | 0.235 | 0.264 | 0.264 |
| ele-fashion | Smooth | 0.685 | 0.512 | 0.512 |
| ele-fashion | AbsDiff | 0.451 | 0.487 | 0.494 |
| ele-fashion | Product | 0.339 | 0.385 | 0.385 |

Thus Text/Visual sign disagreement persists but changes selectively: the P1.2-compatible absdiff disagreement is lower for Movies and Grocery, while product disagreement rises modestly; ele-fashion smooth disagreement falls, while alternative disagreements remain near or above the earlier values.

## 7. Preference margin

The margin is the largest raw utility minus the second-largest, before applying the `none` rule. Values are seed-mean quantiles; no threshold is imposed.

| Dataset | Modality | Median | IQR | Q90 |
|---|---|---:|---:|---:|
| Movies | Text | 0.0066 (0.0004) | 0.0174 (0.0011) | 0.0503 (0.0029) |
| Movies | Visual | 0.0075 (0.0006) | 0.0214 (0.0022) | 0.0629 (0.0060) |
| Grocery | Text | 0.0011 (0.0001) | 0.0082 (0.0001) | 0.0376 (0.0001) |
| Grocery | Visual | 0.0010 (0.0001) | 0.0075 (0.0002) | 0.0363 (0.0011) |
| ele-fashion | Text | 0.0018 (0.0003) | 0.0116 (0.0011) | 0.0537 (0.0035) |
| ele-fashion | Visual | 0.0009 (0.0001) | 0.0067 (0.0007) | 0.0341 (0.0025) |

The small median margins in Grocery and ele-fashion, and to a lesser extent Movies, mean many positive-best argmax assignments are near ties. That qualifies the strength of channel preference, while leaving the raw sign-transition calculation unchanged.

## 8–9. Channel use and whole-channel ablation

All six context blocks have nonzero fitted weight, validation activation, and effective logit contribution in all nine runs. Across dataset means, context-block mean `||W_block Z_block||₂` ranges from 0.99 to 3.00; no tested context is globally ignored. The committed channel audit also records weight Frobenius norm, feature RMS, and median logit contribution for all eight blocks (including `H_T/H_V`).

The table gives mean effective logit contribution and fixed-head validation CE change after zeroing one entire context block (positive ΔCE means that ablation worsened CE). Ablation is a sanity check, with no refit.

| Dataset | Block | Mean logit contribution | Δ validation CE |
|---|---|---:|---:|
| Movies | S_T | 2.059 | +0.0207 |
| Movies | D_T | 1.015 | +0.0125 |
| Movies | P_T | 2.968 | +0.0494 |
| Movies | S_V | 2.797 | +0.0594 |
| Movies | D_V | 1.544 | +0.0282 |
| Movies | P_V | 3.002 | +0.0191 |
| Grocery | S_T | 2.667 | +0.0379 |
| Grocery | D_T | 1.481 | +0.0144 |
| Grocery | P_T | 2.683 | +0.0144 |
| Grocery | S_V | 2.257 | +0.0213 |
| Grocery | D_V | 1.777 | +0.0138 |
| Grocery | P_V | 2.300 | −0.0060 |
| ele-fashion | S_T | 1.593 | +0.0161 |
| ele-fashion | D_T | 1.287 | +0.0063 |
| ele-fashion | P_T | 2.533 | +0.0133 |
| ele-fashion | S_V | 1.483 | +0.0060 |
| ele-fashion | D_V | 1.342 | +0.0042 |
| ele-fashion | P_V | 2.786 | +0.0056 |

Most ablations increase CE, consistent with the usage audit. Grocery `P_V` is a small exception: its full-block ablation slightly improves CE on average despite a nonzero channel contribution. Accuracy and Macro-F1 deltas are also included in the CSV and show some metric-specific variation; the ablation is not a benchmark result.

## 10. Validation performance sanity

These metrics only describe the selected joint probes; the 1024-dimensional joint features are not directly comparable to P1.2 single-operator heads as a model-performance benchmark.

| Dataset | Validation accuracy | Macro-F1 | CE |
|---|---:|---:|---:|
| Movies | 0.546 ± 0.010 | 0.468 ± 0.011 | 1.355 ± 0.005 |
| Grocery | 0.816 ± 0.006 | 0.739 ± 0.012 | 0.669 ± 0.002 |
| ele-fashion | 0.867 ± 0.002 | 0.670 ± 0.009 | 0.407 ± 0.006 |

## Assessment

**P1.3 status: SUPPORTED.** With one common jointly trained downstream predictor, the same physical message exhibits different marginal validation utility across the tested propagation-function channels. The P1.2 sign-transition pattern is largely retained, transitions also occur among different neighbors of the same target, and all six context blocks are measurably used. Results vary by dataset and modality; preference margins are often small and Text/Visual preferences frequently disagree.

I recommend freezing **Propagation Function Heterogeneity** as a bounded empirical problem premise for later method research. Keep the premise limited to these three fixed operators, this graph/feature setup, and a common joint linear readout. These results do not establish a final operator bank, ground-truth edge operator labels, a need for blocking, or superiority/necessity of routing, gates, or MoE. Validation reuse, three seeds, and shared targets among directed edges also limit the strength of inference.

## Files and QA

The seed-level and three-seed mean/population-SD tables are in `data/`; `within_node_function_diversity.csv` contains the target-level eligible-node records as well as dataset/seed/modality summaries. Four requested PNGs are in `figures/`. Final PDFs, SVGs, 600-dpi TIFFs, panel-alignment records, PDF text audits, and collision audits are in ignored `outputs/p13_joint_readout_operator_probe/figure_qa/`; all four final PDF collision audits passed, all panel-alignment gates passed at 1.5 pt, and minimum rendered text was at least 7 pt. The source preflight was ready with no failures and three nonblocking warnings: a generic figure-width check without a specified target journal, detection of the seeded edge-row sample, and a missing-value guard in seed aggregation. The random draw selects existing edge rows for the sample only; all edge analyses use the complete aligned tables. Seed metrics included in summaries were finite, so aggregation excluded no seed observations.

The complete per-edge tables and joint-head checkpoints remain in ignored `outputs/p13_joint_readout_operator_probe/`. No adaptive propagation screening was performed.
