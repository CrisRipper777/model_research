# R³-MAG H2-A relation-context response probe

Primary metrics are mean per-dimension Spearman, sign balanced accuracy, predicted-action utility, and within-modality regret. All targets come from ε=0.1 frozen-host own-structural ± actions. P1 is receiver-only with an all-zero padded context block; P1/P2/P3 therefore share exactly the same 2-layer MLP and parameter count. P3 context is shuffled inside ResponseTrain; Audit receives a separate Audit-only shuffle. Audit labels are evaluation-only.

## Audit metrics

| Dataset | Seed | Variant | Mean Spearman | Sign BA | Selected utility | Regret | Parameters |
|---|---:|---|---:|---:|---:|---:|---:|
| Movies | 42 | P1_ReceiverOnly | 0.0671 | 0.5025 | 0.00470 | 0.09103 | 73830 |
| Movies | 42 | P2_ReceiverPlusRealContext | 0.0663 | 0.5001 | 0.00417 | 0.09156 | 73830 |
| Movies | 42 | P3_ReceiverPlusShuffledContext | 0.0693 | 0.5022 | 0.00345 | 0.09228 | 73830 |
| Movies | 42 | P2_eval_shuffle | 0.0660 | 0.4991 | 0.00415 | 0.09158 | 73830 |
| Movies | 43 | P1_ReceiverOnly | 0.0029 | 0.4908 | 0.00345 | 0.08472 | 73830 |
| Movies | 43 | P2_ReceiverPlusRealContext | 0.0097 | 0.4998 | 0.00373 | 0.08444 | 73830 |
| Movies | 43 | P3_ReceiverPlusShuffledContext | 0.0083 | 0.5001 | 0.00373 | 0.08444 | 73830 |
| Movies | 43 | P2_eval_shuffle | 0.0093 | 0.5006 | 0.00370 | 0.08447 | 73830 |
| Movies | 44 | P1_ReceiverOnly | 0.0384 | 0.4993 | 0.00221 | 0.08331 | 73830 |
| Movies | 44 | P2_ReceiverPlusRealContext | 0.0390 | 0.5053 | 0.00245 | 0.08307 | 73830 |
| Movies | 44 | P3_ReceiverPlusShuffledContext | 0.0385 | 0.5052 | 0.00254 | 0.08298 | 73830 |
| Movies | 44 | P2_eval_shuffle | 0.0389 | 0.5056 | 0.00246 | 0.08306 | 73830 |
| Grocery | 42 | P1_ReceiverOnly | 0.0605 | 0.5098 | -0.00159 | 0.05198 | 73830 |
| Grocery | 42 | P2_ReceiverPlusRealContext | 0.0660 | 0.5129 | -0.00164 | 0.05202 | 73830 |
| Grocery | 42 | P3_ReceiverPlusShuffledContext | 0.0683 | 0.5104 | -0.00147 | 0.05185 | 73830 |
| Grocery | 42 | P2_eval_shuffle | 0.0650 | 0.5114 | -0.00165 | 0.05204 | 73830 |
| Grocery | 43 | P1_ReceiverOnly | 0.0450 | 0.5000 | -0.00234 | 0.05898 | 73830 |
| Grocery | 43 | P2_ReceiverPlusRealContext | 0.0512 | 0.4979 | -0.00241 | 0.05906 | 73830 |
| Grocery | 43 | P3_ReceiverPlusShuffledContext | 0.0496 | 0.4974 | -0.00237 | 0.05902 | 73830 |
| Grocery | 43 | P2_eval_shuffle | 0.0508 | 0.4975 | -0.00238 | 0.05903 | 73830 |
| Grocery | 44 | P1_ReceiverOnly | 0.0437 | 0.4822 | -0.00209 | 0.05547 | 73830 |
| Grocery | 44 | P2_ReceiverPlusRealContext | 0.0444 | 0.4888 | -0.00208 | 0.05546 | 73830 |
| Grocery | 44 | P3_ReceiverPlusShuffledContext | 0.0470 | 0.4875 | -0.00206 | 0.05544 | 73830 |
| Grocery | 44 | P2_eval_shuffle | 0.0461 | 0.4866 | -0.00206 | 0.05543 | 73830 |
| ele-fashion | 42 | P1_ReceiverOnly | 0.0558 | 0.4970 | -0.00130 | 0.02774 | 72790 |
| ele-fashion | 42 | P2_ReceiverPlusRealContext | 0.0445 | 0.4932 | -0.00134 | 0.02779 | 72790 |
| ele-fashion | 42 | P3_ReceiverPlusShuffledContext | 0.0475 | 0.4943 | -0.00136 | 0.02781 | 72790 |
| ele-fashion | 42 | P2_eval_shuffle | 0.0443 | 0.4930 | -0.00135 | 0.02779 | 72790 |
| ele-fashion | 43 | P1_ReceiverOnly | 0.0153 | 0.4953 | -0.00018 | 0.03080 | 72790 |
| ele-fashion | 43 | P2_ReceiverPlusRealContext | 0.0128 | 0.4913 | -0.00023 | 0.03084 | 72790 |
| ele-fashion | 43 | P3_ReceiverPlusShuffledContext | 0.0199 | 0.4914 | -0.00018 | 0.03079 | 72790 |
| ele-fashion | 43 | P2_eval_shuffle | 0.0123 | 0.4935 | -0.00024 | 0.03086 | 72790 |
| ele-fashion | 44 | P1_ReceiverOnly | 0.0329 | 0.4809 | -0.00114 | 0.03092 | 72790 |
| ele-fashion | 44 | P2_ReceiverPlusRealContext | 0.0239 | 0.4838 | -0.00143 | 0.03121 | 72790 |
| ele-fashion | 44 | P3_ReceiverPlusShuffledContext | 0.0213 | 0.4842 | -0.00143 | 0.03122 | 72790 |
| ele-fashion | 44 | P2_eval_shuffle | 0.0231 | 0.4836 | -0.00143 | 0.03121 | 72790 |

## Context correspondence contrasts

| Dataset | Seed | P2−P1 utility | P2−P1 regret | P2−P3 Spearman | P2−P3 utility | P2−eval-shuffle utility |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 42 | -0.00053 | +0.00053 | -0.0030 | +0.00072 | +0.00002 |
| Movies | 43 | +0.00028 | -0.00028 | 0.0014 | -0.00000 | +0.00003 |
| Movies | 44 | +0.00024 | -0.00024 | 0.0005 | -0.00009 | -0.00001 |
| Grocery | 42 | -0.00004 | +0.00004 | -0.0023 | -0.00017 | +0.00002 |
| Grocery | 43 | -0.00008 | +0.00008 | 0.0015 | -0.00004 | -0.00003 |
| Grocery | 44 | +0.00001 | -0.00001 | -0.0026 | -0.00002 | -0.00003 |
| ele-fashion | 42 | -0.00004 | +0.00004 | -0.0030 | +0.00002 | +0.00001 |
| ele-fashion | 43 | -0.00005 | +0.00005 | -0.0070 | -0.00005 | +0.00002 |
| ele-fashion | 44 | -0.00029 | +0.00029 | 0.0026 | +0.00000 | -0.00001 |

Across the nine runs, P2−P1 is -0.00006 selected utility and +0.00006 regret; P2−P3 is -0.0013 Spearman and +0.00004 utility. P2−P2-eval-shuffle is +0.00000 utility. These aggregate differences are small; inspect per-run rows for sign stability.

Under the requested qualitative reading, this probe is H2 Weak for the tested predictor: P2 is close to P1/P3, the predictive rank metrics do not consistently favor real context, and shuffling P2 at Audit barely changes utility. This does not test more expressive relation models or establish that context carries no signal.


## High/low disagreement subgroup

Groups are the bottom/top Audit quartiles of mean neighbor |H0 text cosine − H0 visual cosine|. Deltas are P2−P1.

| Dataset | Seed | Group | Selected utility delta | Regret delta | Nodes |
|---|---:|---|---:|---:|---:|
| Movies | 42 | high_disagreement | -0.00120 | +0.00120 | 249 |
| Movies | 42 | low_disagreement | -0.00106 | +0.00106 | 249 |
| Movies | 43 | high_disagreement | +0.00047 | -0.00047 | 249 |
| Movies | 43 | low_disagreement | +0.00009 | -0.00009 | 249 |
| Movies | 44 | high_disagreement | -0.00010 | +0.00010 | 249 |
| Movies | 44 | low_disagreement | +0.00006 | -0.00006 | 249 |
| Grocery | 42 | high_disagreement | -0.00004 | +0.00004 | 255 |
| Grocery | 42 | low_disagreement | -0.00004 | +0.00004 | 255 |
| Grocery | 43 | high_disagreement | +0.00001 | -0.00001 | 255 |
| Grocery | 43 | low_disagreement | -0.00029 | +0.00029 | 255 |
| Grocery | 44 | high_disagreement | +0.00047 | -0.00047 | 255 |
| Grocery | 44 | low_disagreement | -0.00044 | +0.00044 | 255 |
| ele-fashion | 42 | high_disagreement | -0.00012 | +0.00012 | 1466 |
| ele-fashion | 42 | low_disagreement | -0.00002 | +0.00002 | 1466 |
| ele-fashion | 43 | high_disagreement | -0.00002 | +0.00002 | 1466 |
| ele-fashion | 43 | low_disagreement | -0.00004 | +0.00004 | 1466 |
| ele-fashion | 44 | high_disagreement | -0.00047 | +0.00047 | 1466 |
| ele-fashion | 44 | low_disagreement | -0.00024 | +0.00024 | 1466 |

This subgroup is exploratory. No R² or discrete best-hop accuracy is a primary outcome.
