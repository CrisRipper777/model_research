# RM-CGE V2.1 Mechanism Report

The audit separates learned edge distributions, residual messages entering the shared experts and the effects of frozen checkpoint interventions. All edge statistics exclude isolated nodes; degree-one nodes are reported separately because they cannot change neighbor ranking.

## Receiver mass conservation and edge selection

| Dataset | Condition | Target | Purpose | Mean abs ratio deviation | Mean receiver JS | Mean ESS (degree≥2) | Score near-bound fraction | Max row-mass error | Selectable nodes (degree≥2) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Movies | C0 | text | intra | N/A | N/A | N/A | N/A | 1.19e-06 | 0.796 |
| Movies | C0 | text | cross | N/A | N/A | N/A | N/A | 1.67e-06 | 0.796 |
| Movies | C0 | visual | intra | N/A | N/A | N/A | N/A | 1.43e-06 | 0.796 |
| Movies | C0 | visual | cross | N/A | N/A | N/A | N/A | 1.91e-06 | 0.796 |
| Movies | C1 | text | intra | 0.5922 | 0.0533 | 7.244 | 0.7304 | 5.25e-06 | 0.796 |
| Movies | C1 | text | cross | N/A | N/A | N/A | N/A | 2.86e-06 | 0.796 |
| Movies | C1 | visual | intra | 0.4565 | 0.0369 | 7.664 | 0.5520 | 2.15e-06 | 0.796 |
| Movies | C1 | visual | cross | N/A | N/A | N/A | N/A | 2.38e-06 | 0.796 |
| Movies | C2 | text | intra | 0.5788 | 0.0524 | 7.381 | 0.6737 | 2.86e-06 | 0.796 |
| Movies | C2 | text | cross | 0.4315 | 0.0328 | 8.061 | 0.3228 | 3.34e-06 | 0.796 |
| Movies | C2 | visual | intra | 0.4312 | 0.0332 | 7.868 | 0.4685 | 2.86e-06 | 0.796 |
| Movies | C2 | visual | cross | 0.3224 | 0.0231 | 8.338 | 0.2750 | 2.86e-06 | 0.796 |
| Grocery | C0 | text | intra | N/A | N/A | N/A | N/A | 3.81e-06 | 0.750 |
| Grocery | C0 | text | cross | N/A | N/A | N/A | N/A | 6.68e-06 | 0.750 |
| Grocery | C0 | visual | intra | N/A | N/A | N/A | N/A | 5.72e-06 | 0.750 |
| Grocery | C0 | visual | cross | N/A | N/A | N/A | N/A | 1.05e-05 | 0.750 |
| Grocery | C1 | text | intra | 0.4511 | 0.0322 | 6.526 | 0.2461 | 9.54e-06 | 0.750 |
| Grocery | C1 | text | cross | N/A | N/A | N/A | N/A | 4.77e-06 | 0.750 |
| Grocery | C1 | visual | intra | 0.4782 | 0.0371 | 6.495 | 0.4331 | 1.05e-05 | 0.750 |
| Grocery | C1 | visual | cross | N/A | N/A | N/A | N/A | 1.05e-05 | 0.750 |
| Grocery | C2 | text | intra | 0.4286 | 0.0295 | 6.652 | 0.2188 | 1.53e-05 | 0.750 |
| Grocery | C2 | text | cross | 0.1961 | 0.0067 | 7.654 | 0.0005 | 6.68e-06 | 0.750 |
| Grocery | C2 | visual | intra | 0.4614 | 0.0350 | 6.581 | 0.3831 | 6.68e-06 | 0.750 |
| Grocery | C2 | visual | cross | 0.3584 | 0.0209 | 6.987 | 0.1029 | 9.54e-06 | 0.750 |
| ele-fashion | C0 | text | intra | N/A | N/A | N/A | N/A | 3.43e-05 | 0.535 |
| ele-fashion | C0 | text | cross | N/A | N/A | N/A | N/A | 3.05e-05 | 0.535 |
| ele-fashion | C0 | visual | intra | N/A | N/A | N/A | N/A | 3.81e-05 | 0.535 |
| ele-fashion | C0 | visual | cross | N/A | N/A | N/A | N/A | 2.67e-05 | 0.535 |
| ele-fashion | C1 | text | intra | 0.2948 | 0.0244 | 4.963 | 0.7024 | 6.68e-05 | 0.535 |
| ele-fashion | C1 | text | cross | N/A | N/A | N/A | N/A | 2.29e-05 | 0.535 |
| ele-fashion | C1 | visual | intra | 0.2548 | 0.0219 | 4.929 | 0.8560 | 1.13e-04 | 0.535 |
| ele-fashion | C1 | visual | cross | N/A | N/A | N/A | N/A | 5.53e-05 | 0.535 |
| ele-fashion | C2 | text | intra | 0.2823 | 0.0233 | 4.997 | 0.7079 | 6.68e-05 | 0.535 |
| ele-fashion | C2 | text | cross | 0.2919 | 0.0218 | 4.999 | 0.5304 | 5.72e-05 | 0.535 |
| ele-fashion | C2 | visual | intra | 0.2630 | 0.0223 | 4.908 | 0.8361 | 1.20e-04 | 0.535 |
| ele-fashion | C2 | visual | cross | 0.2220 | 0.0157 | 5.130 | 0.6088 | 8.39e-05 | 0.535 |

The selection formula preserves each receiver's total incoming physical coefficient mass in FP32. `Mean abs ratio deviation` is `mean |p_select/p − 1|`; receiver JS compares the selected and physical within-row distributions; ESS is `1 / sum_j q_ij²`. Scores near the tanh bound satisfy `|tanh(raw)| ≥ .99`. Maximum/mean mass error, ratio quantiles, entropies, KL, reverse-direction differences and per-seed values are in `data/edge_selection_diagnostics.csv`.

## Modality and message-purpose differences

| Dataset | Compared paths | Mean abs edge-probability difference | Mean correlation | Mean receiver JS | n |
|---|---|---:|---:|---:|---:|
| Movies | text_cross__visual_cross | 0.0390 | 0.9067 | 0.0531 | 3 |
| Movies | text_cross__visual_intra | 0.0365 | 0.9177 | 0.0469 | 3 |
| Movies | text_intra__text_cross | 0.0502 | 0.8669 | 0.0799 | 3 |
| Movies | text_intra__visual_cross | 0.0543 | 0.8433 | 0.0919 | 3 |
| Movies | text_intra__visual_intra | 0.0507 | 0.8614 | 0.0821 | 6 |
| Movies | visual_intra__visual_cross | 0.0387 | 0.9037 | 0.0530 | 3 |
| Grocery | text_cross__visual_cross | 0.0302 | 0.9587 | 0.0259 | 3 |
| Grocery | text_cross__visual_intra | 0.0379 | 0.9352 | 0.0402 | 3 |
| Grocery | text_intra__text_cross | 0.0344 | 0.9478 | 0.0330 | 3 |
| Grocery | text_intra__visual_cross | 0.0390 | 0.9321 | 0.0431 | 3 |
| Grocery | text_intra__visual_intra | 0.0491 | 0.8963 | 0.0649 | 6 |
| Grocery | visual_intra__visual_cross | 0.0441 | 0.9139 | 0.0539 | 3 |
| ele-fashion | text_cross__visual_cross | 0.0463 | 0.9523 | 0.0366 | 3 |
| ele-fashion | text_cross__visual_intra | 0.0502 | 0.9410 | 0.0455 | 3 |
| ele-fashion | text_intra__text_cross | 0.0510 | 0.9451 | 0.0435 | 3 |
| ele-fashion | text_intra__visual_cross | 0.0445 | 0.9504 | 0.0380 | 3 |
| ele-fashion | text_intra__visual_intra | 0.0476 | 0.9452 | 0.0424 | 6 |
| ele-fashion | visual_intra__visual_cross | 0.0387 | 0.9587 | 0.0317 | 3 |

These comparisons are receiver-level distribution diagnostics: edge probability difference, correlation and JS for the listed Intra/Cross paths. Small differences are not treated as failure; they are considered with prediction metrics and message contributions.

## Receiver query probe

| Dataset | Condition | Receiver mean-query ablation JS |
|---|---|---:|
| Movies | C0 | N/A |
| Movies | C1 | 0.0192 |
| Movies | C2 | 0.0115 |
| Grocery | C0 | N/A |
| Grocery | C1 | 0.0162 |
| Grocery | C2 | 0.0093 |
| ele-fashion | C0 | N/A |
| ele-fashion | C1 | 0.0195 |
| ele-fashion | C2 | 0.0196 |

This frozen probe replaces each trained receiver vector with its modality-wide active-node mean while leaving sender vectors, scorer weights, degrees and physical coefficients fixed. It measures checkpoint sensitivity to the receiver query; it is not a retrained ablation.

## Messages reaching the shared experts

Raw RMS and reference-normalized RMS are shown together. `U1` is ordinary physical propagation. `ΔIntra` and `ΔCross` are residuals from selection; `Cross base` remains the ordinary projected cross-modal message. KNN remains the inherited semantic context.

| Dataset | Condition | Modality | U1 RMS | Intra Δ RMS / normalized | Cross base RMS / normalized | Cross Δ RMS / normalized | KNN RMS / normalized | Graph correction RMS | Intrinsic RMS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | C0 | text | 0.8071 | 0.0000 / 0.0000 | 0.3822 / 0.7223 | N/A | 0.9393 / 1.2460 | 0.0745 | 0.9968 |
| Movies | C0 | visual | 0.7413 | 0.0000 / 0.0000 | 0.3743 / 0.6153 | N/A | 0.8069 / 1.0820 | 0.8757 | 0.9964 |
| Movies | C1 | text | 0.8097 | 0.1673 / 0.2839 | 0.3969 / 0.7270 | 0.0000 / 0.0000 | 0.9407 / 1.2390 | 0.0981 | 0.9967 |
| Movies | C1 | visual | 0.7379 | 0.1290 / 0.2067 | 0.4324 / 0.6988 | 0.0000 / 0.0000 | 0.8024 / 1.0810 | 0.7762 | 0.9967 |
| Movies | C2 | text | 0.7836 | 0.1852 / 0.3061 | 0.4477 / 0.7856 | 0.1106 / 0.2010 | 0.9201 / 1.2377 | 0.1178 | 0.9968 |
| Movies | C2 | visual | 0.7222 | 0.1459 / 0.2332 | 0.4287 / 0.6994 | 0.1205 / 0.2017 | 0.7834 / 1.0765 | 0.6773 | 0.9975 |
| Grocery | C0 | text | 0.7504 | 0.0000 / 0.0000 | 0.4138 / 0.7179 | N/A | 0.8618 / 1.1635 | 0.1838 | 0.9985 |
| Grocery | C0 | visual | 0.8055 | 0.0000 / 0.0000 | 0.6161 / 1.3190 | N/A | 0.9156 / 1.1750 | 1.4421 | 0.9945 |
| Grocery | C1 | text | 0.7486 | 0.1265 / 0.2008 | 0.3924 / 0.6920 | 0.0000 / 0.0000 | 0.8600 / 1.1632 | 0.2356 | 0.9987 |
| Grocery | C1 | visual | 0.7971 | 0.1109 / 0.2173 | 0.6306 / 1.2817 | 0.0000 / 0.0000 | 0.9089 / 1.1744 | 1.4888 | 0.9945 |
| Grocery | C2 | text | 0.7271 | 0.1372 / 0.2150 | 0.4310 / 0.7295 | 0.0237 / 0.0393 | 0.8417 / 1.1685 | 0.2783 | 0.9991 |
| Grocery | C2 | visual | 0.7864 | 0.1159 / 0.2151 | 0.6357 / 1.2409 | 0.0774 / 0.1544 | 0.8999 / 1.1734 | 1.5277 | 0.9944 |
| ele-fashion | C0 | text | 0.7901 | 0.0000 / 0.0000 | 0.6366 / 0.8649 | N/A | 0.9159 / 1.1641 | 0.4918 | 1.0030 |
| ele-fashion | C0 | visual | 0.7872 | 0.0000 / 0.0000 | 0.6665 / 1.0431 | N/A | 0.9063 / 1.1827 | 0.6788 | 0.9954 |
| ele-fashion | C1 | text | 0.8052 | 0.1447 / 0.1855 | 0.6082 / 0.8084 | 0.0000 / 0.0000 | 0.9280 / 1.1572 | 0.5311 | 1.0022 |
| ele-fashion | C1 | visual | 0.8115 | 0.1326 / 0.1924 | 0.6202 / 0.9719 | 0.0000 / 0.0000 | 0.9241 / 1.1706 | 0.5971 | 0.9942 |
| ele-fashion | C2 | text | 0.8034 | 0.1458 / 0.1864 | 0.6422 / 0.8513 | 0.1103 / 0.1473 | 0.9274 / 1.1586 | 0.5340 | 1.0023 |
| ele-fashion | C2 | visual | 0.8125 | 0.1375 / 0.2016 | 0.6738 / 1.0809 | 0.1112 / 0.1810 | 0.9250 / 1.1722 | 0.6409 | 0.9942 |
| Movies | C0 | text | 0.7711 | 0.0000 / 0.0000 | 0.5476 / 1.0115 | N/A | 0.9148 / 1.2600 | 0.1261 | 0.9957 |
| Movies | C0 | visual | 0.7267 | 0.0000 / 0.0000 | 0.4971 / 0.8261 | N/A | 0.7868 / 1.0762 | 0.6578 | 0.9976 |
| Movies | C1 | text | 0.7809 | 0.1881 / 0.3238 | 0.5000 / 0.9137 | 0.0000 / 0.0000 | 0.9225 / 1.2598 | 0.1813 | 0.9957 |
| Movies | C1 | visual | 0.7237 | 0.1566 / 0.2550 | 0.4920 / 0.8049 | 0.0000 / 0.0000 | 0.7844 / 1.0783 | 0.6104 | 0.9979 |
| Movies | C2 | text | 0.7817 | 0.1912 / 0.3311 | 0.4925 / 0.9198 | 0.1199 / 0.2222 | 0.9232 / 1.2630 | 0.1850 | 0.9957 |
| Movies | C2 | visual | 0.7254 | 0.1570 / 0.2566 | 0.4953 / 0.8185 | 0.1339 / 0.2264 | 0.7859 / 1.0776 | 0.6613 | 0.9979 |
| Grocery | C0 | text | 0.7963 | 0.0000 / 0.0000 | 0.3102 / 0.5685 | N/A | 0.9032 / 1.1595 | 0.4577 | 0.9965 |
| Grocery | C0 | visual | 0.8238 | 0.0000 / 0.0000 | 0.6741 / 1.6128 | N/A | 0.9304 / 1.1708 | 1.4125 | 0.9944 |
| Grocery | C1 | text | 0.7754 | 0.1164 / 0.1943 | 0.3282 / 0.5630 | 0.0000 / 0.0000 | 0.8857 / 1.1645 | 0.4442 | 0.9971 |
| Grocery | C1 | visual | 0.8147 | 0.1046 / 0.2226 | 0.6347 / 1.4779 | 0.0000 / 0.0000 | 0.9224 / 1.1701 | 1.4616 | 0.9946 |
| Grocery | C2 | text | 0.7499 | 0.1355 / 0.2204 | 0.3279 / 0.5469 | 0.0286 / 0.0499 | 0.8638 / 1.1703 | 0.4101 | 0.9978 |
| Grocery | C2 | visual | 0.8026 | 0.1155 / 0.2284 | 0.6357 / 1.3767 | 0.0851 / 0.1870 | 0.9119 / 1.1669 | 1.4764 | 0.9949 |
| ele-fashion | C0 | text | 0.7893 | 0.0000 / 0.0000 | 0.6126 / 0.8400 | N/A | 0.9141 / 1.1661 | 0.6592 | 1.0068 |
| ele-fashion | C0 | visual | 0.7739 | 0.0000 / 0.0000 | 0.6663 / 1.0543 | N/A | 0.8978 / 1.1932 | 0.6839 | 0.9965 |
| ele-fashion | C1 | text | 0.8138 | 0.1334 / 0.1704 | 0.6886 / 0.9151 | 0.0000 / 0.0000 | 0.9352 / 1.1562 | 0.6698 | 1.0031 |
| ele-fashion | C1 | visual | 0.8355 | 0.1407 / 0.2164 | 0.6718 / 1.1217 | 0.0000 / 0.0000 | 0.9397 / 1.1650 | 0.5770 | 0.9938 |
| ele-fashion | C2 | text | 0.8178 | 0.1336 / 0.1711 | 0.7924 / 1.0469 | 0.1091 / 0.1472 | 0.9368 / 1.1536 | 0.6367 | 1.0031 |
| ele-fashion | C2 | visual | 0.8381 | 0.1383 / 0.2117 | 0.7078 / 1.1749 | 0.0960 / 0.1596 | 0.9411 / 1.1606 | 0.6714 | 0.9945 |
| Movies | C0 | text | 0.7615 | 0.0000 / 0.0000 | 0.5830 / 1.0788 | N/A | 0.9147 / 1.2633 | 0.1920 | 0.9956 |
| Movies | C0 | visual | 0.7368 | 0.0000 / 0.0000 | 0.5103 / 0.8447 | N/A | 0.7979 / 1.0737 | 0.6372 | 0.9981 |
| Movies | C1 | text | 0.7398 | 0.2176 / 0.3477 | 0.5733 / 1.0287 | 0.0000 / 0.0000 | 0.8971 / 1.2649 | 0.2169 | 0.9962 |
| Movies | C1 | visual | 0.7111 | 0.1625 / 0.2594 | 0.4598 / 0.7553 | 0.0000 / 0.0000 | 0.7677 / 1.0708 | 0.7840 | 0.9994 |
| Movies | C2 | text | 0.8309 | 0.1654 / 0.2953 | 0.4203 / 0.8737 | 0.0397 / 0.0813 | 0.9572 / 1.2439 | 0.1150 | 0.9956 |
| Movies | C2 | visual | 0.7621 | 0.1218 / 0.1921 | 0.4867 / 0.7828 | 0.0378 / 0.0595 | 0.8287 / 1.0819 | 0.6388 | 0.9978 |
| Grocery | C0 | text | 0.7173 | 0.0000 / 0.0000 | 0.5277 / 0.8875 | N/A | 0.8352 / 1.1797 | 0.2954 | 0.9998 |
| Grocery | C0 | visual | 0.7983 | 0.0000 / 0.0000 | 0.5812 / 1.1977 | N/A | 0.9088 / 1.1679 | 1.6382 | 0.9943 |
| Grocery | C1 | text | 0.7176 | 0.1466 / 0.2325 | 0.4855 / 0.8128 | 0.0000 / 0.0000 | 0.8364 / 1.1794 | 0.3918 | 0.9998 |
| Grocery | C1 | visual | 0.7977 | 0.1240 / 0.2338 | 0.5646 / 1.1492 | 0.0000 / 0.0000 | 0.9085 / 1.1677 | 1.6812 | 0.9946 |
| Grocery | C2 | text | 0.7622 | 0.1009 / 0.1655 | 0.3822 / 0.6804 | 0.0150 / 0.0270 | 0.8756 / 1.1696 | 0.3129 | 0.9984 |
| Grocery | C2 | visual | 0.8197 | 0.0945 / 0.2002 | 0.5897 / 1.2943 | 0.0649 / 0.1442 | 0.9275 / 1.1696 | 1.6648 | 0.9945 |
| ele-fashion | C0 | text | 0.7893 | 0.0000 / 0.0000 | 0.6020 / 0.8285 | N/A | 0.9145 / 1.1643 | 0.6588 | 1.0038 |
| ele-fashion | C0 | visual | 0.7726 | 0.0000 / 0.0000 | 0.7020 / 1.1035 | N/A | 0.8984 / 1.1924 | 0.6355 | 0.9968 |
| ele-fashion | C1 | text | 0.8023 | 0.1382 / 0.1795 | 0.5938 / 0.8073 | 0.0000 / 0.0000 | 0.9246 / 1.1582 | 0.8417 | 1.0031 |
| ele-fashion | C1 | visual | 0.7930 | 0.1433 / 0.2070 | 0.6051 / 0.9451 | 0.0000 / 0.0000 | 0.9132 / 1.1835 | 0.5988 | 0.9956 |
| ele-fashion | C2 | text | 0.8099 | 0.1348 / 0.1735 | 0.6933 / 0.9423 | 0.1176 / 0.1582 | 0.9297 / 1.1542 | 0.7180 | 1.0025 |
| ele-fashion | C2 | visual | 0.8059 | 0.1409 / 0.2008 | 0.6422 / 0.9851 | 0.1211 / 0.1840 | 0.9220 / 1.1742 | 0.5321 | 0.9951 |

Mean coefficient-weighted RMS by learned message purpose, averaged over modalities, seeds and four experts:

| Dataset | Condition | Intra Δ coefficient-weighted RMS | Cross Δ coefficient-weighted RMS |
|---|---|---:|---:|
| Movies | C0 | 0.0000 | N/A |
| Movies | C1 | 0.0763 | 0.0000 |
| Movies | C2 | 0.0694 | 0.0393 |
| Grocery | C0 | 0.0000 | N/A |
| Grocery | C1 | 0.0583 | 0.0000 |
| Grocery | C2 | 0.0553 | 0.0236 |
| ele-fashion | C0 | 0.0000 | N/A |
| ele-fashion | C1 | 0.0635 | 0.0000 |
| ele-fashion | C2 | 0.0614 | 0.0511 |

The corresponding four expert-specific values, physical and combined profile RMS, projection/message scales, expert outputs and parameter update records are preserved in `data/message_scale_diagnostics.csv` and `data/gradient_audit.json`. In C2, Cross Delta feeds both shared expert profiles and the existing Node Router input. The Cross Projection, experts, Router and Fusion parameter gradients and updates are covered in the gradient audit.

## Existing Node Router behavior

| Dataset | Condition | Modality | Mean router entropy | Mean correction-logit RMS | Mean Top-2 flip fraction | Largest mean dense probability | Largest mean Top-2 share | Base largest Top-2 share |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Movies | C0 | text | 1.3803 | 0.0864 | 0.6231 | 0.2609 | 0.465 | 0.500 |
| Movies | C0 | visual | 1.1555 | 0.9363 | 0.5517 | 0.3620 | 0.440 | 0.500 |
| Movies | C1 | text | 1.3720 | 0.1337 | 0.6240 | 0.2688 | 0.446 | 0.500 |
| Movies | C1 | visual | 1.1304 | 1.1342 | 0.5950 | 0.3147 | 0.338 | 0.500 |
| Movies | C2 | text | 1.3823 | 0.0789 | 0.9102 | 0.2575 | 0.350 | 0.500 |
| Movies | C2 | visual | 1.1367 | 1.0430 | 0.5978 | 0.3349 | 0.312 | 0.500 |
| Grocery | C0 | text | 1.3533 | 0.6037 | 0.7401 | 0.2749 | 0.396 | 0.500 |
| Grocery | C0 | visual | 0.0022 | 3.5061 | 0.4821 | 0.9998 | 0.500 | 0.500 |
| Grocery | C1 | text | 1.3369 | 0.7280 | 0.6135 | 0.2699 | 0.383 | 0.500 |
| Grocery | C1 | visual | 0.0025 | 3.3209 | 0.4317 | 0.9997 | 0.500 | 0.500 |
| Grocery | C2 | text | 1.3513 | 0.6080 | 0.6856 | 0.2578 | 0.334 | 0.500 |
| Grocery | C2 | visual | 0.0032 | 3.1442 | 0.3897 | 0.9996 | 0.500 | 0.500 |
| ele-fashion | C0 | text | 1.1669 | 0.8849 | 0.8732 | 0.3170 | 0.339 | 0.500 |
| ele-fashion | C0 | visual | 1.0818 | 1.0697 | 0.9123 | 0.3191 | 0.323 | 0.333 |
| ele-fashion | C1 | text | 1.1915 | 0.8638 | 0.8792 | 0.3219 | 0.325 | 0.500 |
| ele-fashion | C1 | visual | 1.1602 | 0.8208 | 0.9188 | 0.3214 | 0.297 | 0.500 |
| ele-fashion | C2 | text | 1.1880 | 0.9061 | 0.8977 | 0.3280 | 0.355 | 0.500 |
| ele-fashion | C2 | visual | 1.1732 | 0.7728 | 0.9407 | 0.3405 | 0.346 | 0.500 |

Entropy is for the existing route distribution; largest expert shares summarize Top-2 selection counts. Grocery Visual is severely concentrated in every condition: its largest mean dense probability is 0.9996–0.9998, route entropy is 0.0022–0.0032, and correction-logit RMS is 3.14–3.51. This behavior is already present in C0, so these runs do not attribute it to V2.1. Movies routes are more distributed and ele-fashion is intermediate. Top-2 flips compare the trained node route with the base route; they describe route changes and are not a concentration measure. The Router architecture, Top-2 selection and balance loss were unchanged. Treat the Grocery Visual router as a serious inherited pathology for a separately scoped investigation.

## Frozen checkpoint interventions

| Dataset | Condition | Intervention | Mean Δ Val Accuracy (pp) | Mean Δ Macro-F1 (pp) | Mean logit RMS | Mean prediction flips |
|---|---|---|---:|---:|---:|---:|
| Movies | C0 | intra_selection_forced_physical | N/A | N/A | N/A | N/A |
| Movies | C0 | cross_selection_forced_physical | N/A | N/A | N/A | N/A |
| Movies | C0 | both_selections_forced_physical | N/A | N/A | N/A | N/A |
| Movies | C0 | cross_base_context_off | -0.09 | -0.08 | 0.3230 | 0.0708 |
| Movies | C0 | knn_context_off | -0.23 | -0.12 | 0.0468 | 0.0113 |
| Movies | C1 | intra_selection_forced_physical | -0.56 | -1.00 | 0.1868 | 0.0289 |
| Movies | C1 | cross_selection_forced_physical | N/A | N/A | N/A | N/A |
| Movies | C1 | both_selections_forced_physical | -0.56 | -1.00 | 0.1868 | 0.0289 |
| Movies | C1 | cross_base_context_off | -0.04 | +0.66 | 0.3861 | 0.0852 |
| Movies | C1 | knn_context_off | -0.19 | -0.04 | 0.0505 | 0.0107 |
| Movies | C2 | intra_selection_forced_physical | -0.45 | -0.64 | 0.1399 | 0.0245 |
| Movies | C2 | cross_selection_forced_physical | -0.16 | -0.40 | 0.0690 | 0.0108 |
| Movies | C2 | both_selections_forced_physical | -0.48 | -1.07 | 0.1616 | 0.0322 |
| Movies | C2 | cross_base_context_off | -0.18 | +0.20 | 0.3517 | 0.0806 |
| Movies | C2 | knn_context_off | -0.35 | -0.51 | 0.0490 | 0.0127 |
| Grocery | C0 | intra_selection_forced_physical | N/A | N/A | N/A | N/A |
| Grocery | C0 | cross_selection_forced_physical | N/A | N/A | N/A | N/A |
| Grocery | C0 | both_selections_forced_physical | N/A | N/A | N/A | N/A |
| Grocery | C0 | cross_base_context_off | -1.17 | -1.51 | 0.3106 | 0.0348 |
| Grocery | C0 | knn_context_off | -0.06 | -0.12 | 0.0504 | 0.0066 |
| Grocery | C1 | intra_selection_forced_physical | -0.03 | -0.03 | 0.1047 | 0.0066 |
| Grocery | C1 | cross_selection_forced_physical | N/A | N/A | N/A | N/A |
| Grocery | C1 | both_selections_forced_physical | -0.03 | -0.03 | 0.1047 | 0.0066 |
| Grocery | C1 | cross_base_context_off | -0.88 | -0.98 | 0.3042 | 0.0316 |
| Grocery | C1 | knn_context_off | -0.07 | -0.12 | 0.0497 | 0.0065 |
| Grocery | C2 | intra_selection_forced_physical | -0.11 | -0.35 | 0.0994 | 0.0074 |
| Grocery | C2 | cross_selection_forced_physical | -0.06 | -0.10 | 0.0380 | 0.0038 |
| Grocery | C2 | both_selections_forced_physical | -0.13 | -0.31 | 0.1068 | 0.0082 |
| Grocery | C2 | cross_base_context_off | -0.87 | -1.77 | 0.3033 | 0.0358 |
| Grocery | C2 | knn_context_off | -0.07 | -0.22 | 0.0483 | 0.0065 |
| ele-fashion | C0 | intra_selection_forced_physical | N/A | N/A | N/A | N/A |
| ele-fashion | C0 | cross_selection_forced_physical | N/A | N/A | N/A | N/A |
| ele-fashion | C0 | both_selections_forced_physical | N/A | N/A | N/A | N/A |
| ele-fashion | C0 | cross_base_context_off | -0.38 | -0.49 | 0.4021 | 0.0199 |
| ele-fashion | C0 | knn_context_off | -0.28 | -0.62 | 0.1373 | 0.0094 |
| ele-fashion | C1 | intra_selection_forced_physical | -0.03 | +0.07 | 0.1107 | 0.0028 |
| ele-fashion | C1 | cross_selection_forced_physical | N/A | N/A | N/A | N/A |
| ele-fashion | C1 | both_selections_forced_physical | -0.03 | +0.07 | 0.1107 | 0.0028 |
| ele-fashion | C1 | cross_base_context_off | -0.27 | -0.46 | 0.2733 | 0.0107 |
| ele-fashion | C1 | knn_context_off | -0.26 | -0.51 | 0.0906 | 0.0061 |
| ele-fashion | C2 | intra_selection_forced_physical | -0.04 | -0.19 | 0.0983 | 0.0025 |
| ele-fashion | C2 | cross_selection_forced_physical | -0.03 | -0.12 | 0.0592 | 0.0013 |
| ele-fashion | C2 | both_selections_forced_physical | -0.06 | -0.26 | 0.1212 | 0.0034 |
| ele-fashion | C2 | cross_base_context_off | -0.33 | +0.04 | 0.2860 | 0.0154 |
| ele-fashion | C2 | knn_context_off | -0.29 | -0.63 | 0.0839 | 0.0062 |

Values are mean intervention-minus-checkpoint changes over the three seeds. A prediction flip is the fraction of validation predictions changed. These interventions hold trained parameters fixed and measure checkpoint sensitivity; they do not replace retrained ablations. Paths absent from a condition are explicitly `not_applicable`.
