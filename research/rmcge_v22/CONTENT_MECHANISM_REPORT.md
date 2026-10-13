# Content mechanism report

These measurements are evaluated on each Best Accuracy checkpoint. Gate activity establishes that a pathway learned; frozen interventions show checkpoint sensitivity. Neither alone proves standalone causal utility. Pair input channels and active capacity differ across M1/M2/M3.

## Gate and residual summary

| Dataset | Arm | Mean channel SD | Saturated fraction | Query roll gate RMS | Normalized content RMS | Content/Cross cosine | Eta weighted RMS |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | M1 | 0.065706 | 0.0056661 | 0 | 0.25583 | 0.15033 | 0.06717 |
| Movies | M2 | 0.15183 | 0.034694 | 0 | 0.56319 | 0.11644 | 0.15628 |
| Movies | M3 | 0.13963 | 0.066796 | 0.17724 | 0.51496 | 0.11316 | 0.1391 |
| Movies | M4 | 0.17109 | 0.063044 | 0.21722 | 0.57424 | 0.10059 | 0.16027 |
| Grocery | M1 | 0.030088 | 0.0008456 | 0 | 0.12168 | 0.095638 | 0.032488 |
| Grocery | M2 | 0.1148 | 7.6004e-05 | 0 | 0.41426 | 0.063787 | 0.10977 |
| Grocery | M3 | 0.11464 | 0 | 0.15123 | 0.385 | 0.058385 | 0.10066 |
| Grocery | M4 | 0.1081 | 0 | 0.14086 | 0.34278 | 0.057511 | 0.089368 |
| ele-fashion | M1 | 0.29018 | 0.40683 | 0 | 0.8321 | 0.12139 | 0.25292 |
| ele-fashion | M2 | 0.31418 | 0.20191 | 0 | 1.3516 | 0.089973 | 0.44096 |
| ele-fashion | M3 | 0.37113 | 0.31396 | 0.38542 | 1.3215 | 0.074912 | 0.43501 |
| ele-fashion | M4 | 0.34243 | 0.39455 | 0.35576 | 1.1419 | 0.096559 | 0.36353 |

The CSV retains 32-channel means, standard deviations and correlation matrices per Dataset × Seed × Modality. M1 and M2 have no pair-query sensitivity by construction. M3/M4 use a fixed label-free receiver-query roll for both gate and frozen-logit sensitivity.

## Physical, cross-modal, and semantic scale

All entries are means over three seeds. Physical Strength is the scalar shared by active nodes for a modality at the frozen checkpoint, captured by a CPU eval-only probe. Physical Profile, eta-weighted Content, and expert-output columns retain the four expert-wise RMS values, in expert order. `u1` is the physical one-hop reference; `Cross` is projected cross-modal context; `KNN` is the semantic-neighbor context. `*_hat` values are the normalized inputs actually used by the expert path.

| Dataset | Arm | Modality | Physical Strength | Physical u1 RMS | Cross raw / normalized RMS | KNN raw / normalized RMS | Four-order Physical Profile RMS by expert | Content normalized RMS | Eta-weighted Content RMS by expert | Expert output RMS by expert |
|---|---|---|---:|---:|---:|---:|---|---:|---|---|
| Movies | M0 | text | 0.2409 | 0.7739 | 0.5201 / 0.9676 | 0.9174 / 1.257 | [1.915, 0.2756, 0.522, 0.2468] | 0 | [0, 0, 0, 0] | [1.978, 0.4531, 0.6114, 0.4283] |
| Movies | M0 | visual | 0.7363 | 0.7331 | 0.4534 / 0.7552 | 0.7948 / 1.077 | [1.91, 0.2853, 0.5343, 0.2465] | 0 | [0, 0, 0, 0] | [1.967, 0.4136, 0.6047, 0.3845] |
| Movies | M1 | text | 0.2152 | 0.7919 | 0.4671 / 0.8769 | 0.9301 / 1.254 | [1.922, 0.2464, 0.5012, 0.21] | 0.4324 | [0.1081, 0.1173, 0.1119, 0.1178] | [2, 0.4566, 0.61, 0.4224] |
| Movies | M1 | visual | 0.6678 | 0.7397 | 0.4414 / 0.7359 | 0.8028 / 1.078 | [1.914, 0.259, 0.5238, 0.2113] | 0.07929 | [0.01958, 0.02103, 0.02022, 0.02154] | [1.977, 0.3874, 0.5939, 0.349] |
| Movies | M2 | text | 0.2238 | 0.7915 | 0.4362 / 0.8112 | 0.9305 / 1.251 | [1.923, 0.2421, 0.4981, 0.2369] | 0.5991 | [0.1567, 0.1714, 0.1679, 0.169] | [2.006, 0.4697, 0.6283, 0.4568] |
| Movies | M2 | visual | 0.6723 | 0.7332 | 0.5334 / 0.8661 | 0.7954 / 1.078 | [1.913, 0.257, 0.5276, 0.2388] | 0.5273 | [0.1348, 0.1516, 0.1478, 0.151] | [1.975, 0.4321, 0.6385, 0.425] |
| Movies | M3 | text | 0.2176 | 0.8073 | 0.4183 / 0.795 | 0.9408 / 1.249 | [1.928, 0.2156, 0.4768, 0.1834] | 0.6202 | [0.1685, 0.1715, 0.1609, 0.1718] | [2.017, 0.4472, 0.5952, 0.4151] |
| Movies | M3 | visual | 0.6342 | 0.7464 | 0.4911 / 0.8039 | 0.8108 / 1.08 | [1.919, 0.231, 0.5042, 0.1862] | 0.4097 | [0.1092, 0.1119, 0.1059, 0.1131] | [1.985, 0.3869, 0.5902, 0.3562] |
| Movies | M4 | text | 0.2291 | 0.7935 | 0.4111 / 0.7772 | 0.931 / 1.252 | [1.924, 0.2526, 0.4916, 0.2358] | 0.6044 | [0.1632, 0.1756, 0.1632, 0.1766] | [2.009, 0.4836, 0.6157, 0.4645] |
| Movies | M4 | visual | 0.7483 | 0.7294 | 0.4998 / 0.8012 | 0.7906 / 1.078 | [1.913, 0.2656, 0.5218, 0.237] | 0.5441 | [0.1449, 0.1555, 0.1456, 0.1577] | [1.974, 0.4302, 0.618, 0.4156] |
| Grocery | M0 | text | 0.3905 | 0.7597 | 0.3958 / 0.6953 | 0.8716 / 1.166 | [1.931, 0.2062, 0.4621, 0.1598] | 0 | [0, 0, 0, 0] | [2.024, 0.3254, 0.5101, 0.2785] |
| Grocery | M0 | visual | 0.7586 | 0.8119 | 0.6185 / 1.383 | 0.9205 / 1.171 | [1.923, 0.205, 0.4906, 0.1633] | 0 | [0, 0, 0, 0] | [1.998, 0.4425, 0.6067, 0.4031] |
| Grocery | M1 | text | 0.41 | 0.7535 | 0.4351 / 0.7595 | 0.8665 / 1.168 | [1.928, 0.1931, 0.4672, 0.1501] | 0.05406 | [0.01404, 0.01455, 0.01389, 0.01471] | [2.028, 0.3216, 0.5248, 0.2824] |
| Grocery | M1 | visual | 0.7386 | 0.8086 | 0.611 / 1.342 | 0.9179 / 1.171 | [1.924, 0.189, 0.4851, 0.1519] | 0.1893 | [0.04956, 0.05142, 0.04884, 0.05289] | [1.992, 0.4334, 0.6031, 0.3978] |
| Grocery | M2 | text | 0.3954 | 0.7577 | 0.3821 / 0.6617 | 0.8699 / 1.166 | [1.931, 0.2054, 0.4629, 0.1555] | 0.2124 | [0.05795, 0.05755, 0.05518, 0.05639] | [2.029, 0.3328, 0.5155, 0.282] |
| Grocery | M2 | visual | 0.7509 | 0.8115 | 0.6596 / 1.467 | 0.9203 / 1.172 | [1.924, 0.2041, 0.49, 0.1591] | 0.6161 | [0.1661, 0.1638, 0.1583, 0.163] | [2.002, 0.4935, 0.642, 0.4545] |
| Grocery | M3 | text | 0.4123 | 0.7734 | 0.3632 / 0.6488 | 0.8839 / 1.165 | [1.933, 0.1967, 0.4557, 0.1499] | 0.1418 | [0.0377, 0.03793, 0.03669, 0.03625] | [2.026, 0.3124, 0.5007, 0.261] |
| Grocery | M3 | visual | 0.7411 | 0.8168 | 0.6449 / 1.45 | 0.925 / 1.172 | [1.925, 0.1973, 0.4862, 0.1544] | 0.6282 | [0.1665, 0.1677, 0.1623, 0.1602] | [2.005, 0.4842, 0.6347, 0.4431] |
| Grocery | M4 | text | 0.4219 | 0.772 | 0.3595 / 0.6452 | 0.8827 / 1.165 | [1.933, 0.1931, 0.4532, 0.1435] | 0.1464 | [0.03902, 0.03876, 0.03863, 0.03744] | [2.029, 0.313, 0.4987, 0.2592] |
| Grocery | M4 | visual | 0.7465 | 0.8132 | 0.6402 / 1.407 | 0.9222 / 1.173 | [1.926, 0.194, 0.4833, 0.1483] | 0.5392 | [0.1424, 0.1406, 0.1386, 0.1394] | [2.008, 0.4654, 0.6211, 0.4265] |
| ele-fashion | M0 | text | 0.6429 | 0.792 | 0.6215 / 0.853 | 0.9165 / 1.164 | [1.835, 0.262, 0.7986, 0.189] | 0 | [0, 0, 0, 0] | [2.008, 0.5516, 0.8588, 0.4812] |
| ele-fashion | M0 | visual | 0.6301 | 0.7822 | 0.6727 / 1.065 | 0.9039 / 1.188 | [1.816, 0.2622, 0.8384, 0.1935] | 0 | [0, 0, 0, 0] | [1.964, 0.5949, 0.9143, 0.5385] |
| ele-fashion | M1 | text | 0.5497 | 0.8047 | 0.5494 / 0.7433 | 0.9269 / 1.158 | [1.837, 0.2575, 0.7931, 0.2347] | 0.7805 | [0.2349, 0.2459, 0.2311, 0.2352] | [2.041, 0.592, 0.8815, 0.5596] |
| ele-fashion | M1 | visual | 0.4928 | 0.8085 | 0.5723 / 0.9048 | 0.9222 / 1.174 | [1.821, 0.2562, 0.8273, 0.2363] | 0.8837 | [0.2662, 0.2813, 0.261, 0.2678] | [2.011, 0.644, 0.9435, 0.6153] |
| ele-fashion | M2 | text | 0.5682 | 0.8021 | 0.8156 / 1.086 | 0.9259 / 1.16 | [1.838, 0.2492, 0.7886, 0.2317] | 1.51 | [0.511, 0.5095, 0.4639, 0.4886] | [2.146, 0.823, 1.018, 0.7845] |
| ele-fashion | M2 | visual | 0.537 | 0.8099 | 0.5357 / 0.846 | 0.9232 / 1.174 | [1.82, 0.2478, 0.8257, 0.2332] | 1.193 | [0.4013, 0.4014, 0.3662, 0.3858] | [2.027, 0.688, 0.9684, 0.6628] |
| ele-fashion | M3 | text | 0.5746 | 0.8012 | 0.6879 / 0.9262 | 0.9242 / 1.16 | [1.836, 0.2519, 0.7936, 0.2359] | 1.518 | [0.5213, 0.5239, 0.4706, 0.4852] | [2.111, 0.791, 0.992, 0.734] |
| ele-fashion | M3 | visual | 0.5483 | 0.8008 | 0.5198 / 0.8167 | 0.9173 / 1.178 | [1.819, 0.2512, 0.83, 0.2378] | 1.125 | [0.3845, 0.3875, 0.3464, 0.3607] | [2.024, 0.688, 0.9628, 0.6528] |
| ele-fashion | M4 | text | 0.656 | 0.806 | 0.6057 / 0.8046 | 0.9295 / 1.159 | [1.839, 0.2412, 0.7874, 0.2015] | 1.208 | [0.403, 0.3995, 0.3643, 0.369] | [2.079, 0.6756, 0.9315, 0.6199] |
| ele-fashion | M4 | visual | 0.5396 | 0.8153 | 0.5142 / 0.8139 | 0.9275 / 1.173 | [1.823, 0.2401, 0.8219, 0.2038] | 1.076 | [0.3576, 0.3578, 0.3265, 0.3304] | [2.021, 0.6531, 0.9408, 0.6065] |

These scale measurements show whether semantic KNN and cross-modal context reach a similar or different magnitude from physical propagation and whether the learned content correction survives normalization and expert-specific `eta`. Scale alone does not establish that a component supplies useful label information. Modality-specific per-seed values remain in `message_scale_diagnostics.csv`.

## Target-modality differences

Positive Text−Visual values mean the Text-target pathway has the larger scale. These are descriptive averages over the three paired seeds.

| Dataset | Arm | Normalized Content RMS Δ (Text−Visual) | Eta-weighted Content RMS Δ | Residual/Cross cosine Δ |
|---|---|---:|---:|---:|
| Movies | M1 | +0.35309 | +0.093159 | +0.0080012 |
| Movies | M2 | +0.071768 | +0.019967 | +0.059545 |
| Movies | M3 | +0.21049 | +0.058122 | +0.063509 |
| Movies | M4 | +0.060342 | +0.018717 | +0.086377 |
| Grocery | M1 | -0.13526 | -0.036378 | +0.017097 |
| Grocery | M2 | -0.40367 | -0.106 | +0.015431 |
| Grocery | M3 | -0.48644 | -0.12703 | +0.038186 |
| Grocery | M4 | -0.3928 | -0.10181 | +0.037709 |
| ele-fashion | M1 | -0.1032 | -0.032272 | -0.042044 |
| ele-fashion | M2 | +0.31676 | +0.10458 | -0.063969 |
| ele-fashion | M3 | +0.39269 | +0.13047 | -0.10026 |
| ele-fashion | M4 | +0.13203 | +0.04085 | -0.086226 |

Differences indicate that the two target directions operate at different trained scales; they do not establish that one modality is more useful. Per-seed modality-specific gate distributions, full 32-channel correlations, and content measurements remain available in the data tables.

## Frozen interventions

| Dataset | Arm | Intervention | Mean Δ Accuracy (pp) | Mean Δ Macro-F1 (pp) | Mean logit Δ RMS | Mean prediction flip fraction |
|---|---|---|---:|---:|---:|---:|
| Movies | M0 | content_off | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00000 ± 0.00000 | 0.00 ± 0.00 |
| Movies | M0 | cross_base_off | -0.31 ± 0.36 | -0.40 ± 0.57 | 0.34728 ± 0.08430 | 7.89 ± 1.82 |
| Movies | M0 | knn_off | -0.30 ± 0.03 | -0.57 ± 0.17 | 0.04838 ± 0.00833 | 1.36 ± 0.30 |
| Movies | M1 | content_off | -0.10 ± 0.14 | -0.21 ± 0.11 | 0.02916 ± 0.01682 | 0.92 ± 0.57 |
| Movies | M1 | cross_base_off | -0.04 ± 0.49 | -0.10 ± 0.71 | 0.30459 ± 0.09895 | 7.18 ± 2.11 |
| Movies | M1 | knn_off | -0.14 ± 0.26 | -0.25 ± 0.36 | 0.04250 ± 0.01154 | 1.14 ± 0.35 |
| Movies | M2 | content_off | -0.22 ± 0.19 | -0.13 ± 0.37 | 0.14086 ± 0.11369 | 3.41 ± 2.37 |
| Movies | M2 | cross_base_off | -0.19 ± 0.27 | -0.18 ± 0.37 | 0.34371 ± 0.10049 | 7.31 ± 1.54 |
| Movies | M2 | knn_off | -0.20 ± 0.14 | -0.18 ± 0.16 | 0.04457 ± 0.01141 | 1.27 ± 0.19 |
| Movies | M3 | content_off | -0.24 ± 0.39 | -0.10 ± 0.60 | 0.08622 ± 0.03570 | 2.30 ± 0.98 |
| Movies | M3 | cross_base_off | -0.40 ± 0.57 | -0.48 ± 0.20 | 0.30554 ± 0.07633 | 7.65 ± 1.14 |
| Movies | M3 | knn_off | -0.24 ± 0.20 | -0.34 ± 0.17 | 0.03771 ± 0.00575 | 1.08 ± 0.15 |
| Movies | M3 | receiver_query_roll | -0.31 ± 0.36 | -0.54 ± 0.91 | 0.09090 ± 0.04118 | 2.25 ± 1.01 |
| Movies | M4 | content_off | -0.44 ± 0.55 | -0.63 ± 0.69 | 0.17478 ± 0.13972 | 4.11 ± 3.00 |
| Movies | M4 | cross_base_off | -0.59 ± 0.32 | -0.09 ± 0.75 | 0.34450 ± 0.03993 | 8.38 ± 1.63 |
| Movies | M4 | intra_selection_off | -0.19 ± 0.24 | -0.83 ± 0.40 | 0.15665 ± 0.01088 | 2.76 ± 0.29 |
| Movies | M4 | knn_off | -0.26 ± 0.09 | -0.31 ± 0.19 | 0.05017 ± 0.00795 | 1.26 ± 0.25 |
| Movies | M4 | receiver_query_roll | -0.57 ± 0.66 | -0.87 ± 1.20 | 0.21678 ± 0.19788 | 4.66 ± 4.10 |
| Grocery | M0 | content_off | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00000 ± 0.00000 | 0.00 ± 0.00 |
| Grocery | M0 | cross_base_off | -0.87 ± 0.59 | -1.07 ± 0.76 | 0.30315 ± 0.02031 | 3.54 ± 0.46 |
| Grocery | M0 | knn_off | -0.19 ± 0.09 | -0.27 ± 0.09 | 0.05138 ± 0.00445 | 0.57 ± 0.07 |
| Grocery | M1 | content_off | -0.09 ± 0.13 | -0.08 ± 0.14 | 0.03740 ± 0.05242 | 0.30 ± 0.45 |
| Grocery | M1 | cross_base_off | -0.75 ± 0.19 | -0.92 ± 0.56 | 0.30167 ± 0.06136 | 3.15 ± 0.43 |
| Grocery | M1 | knn_off | -0.09 ± 0.06 | -0.17 ± 0.23 | 0.04906 ± 0.00176 | 0.60 ± 0.18 |
| Grocery | M2 | content_off | -0.11 ± 0.10 | -0.13 ± 0.19 | 0.12761 ± 0.00783 | 1.20 ± 0.26 |
| Grocery | M2 | cross_base_off | -0.74 ± 0.34 | -1.20 ± 0.05 | 0.29257 ± 0.02750 | 2.98 ± 0.13 |
| Grocery | M2 | knn_off | -0.06 ± 0.08 | -0.12 ± 0.14 | 0.05112 ± 0.00555 | 0.69 ± 0.23 |
| Grocery | M3 | content_off | -0.05 ± 0.06 | -0.13 ± 0.10 | 0.10550 ± 0.06643 | 0.96 ± 0.69 |
| Grocery | M3 | cross_base_off | -0.74 ± 0.15 | -1.42 ± 0.16 | 0.27561 ± 0.01038 | 3.09 ± 0.67 |
| Grocery | M3 | knn_off | 0.01 ± 0.13 | 0.05 ± 0.18 | 0.05187 ± 0.00441 | 0.86 ± 0.04 |
| Grocery | M3 | receiver_query_roll | -0.17 ± 0.19 | -0.11 ± 0.30 | 0.11209 ± 0.09465 | 0.97 ± 0.77 |
| Grocery | M4 | content_off | -0.08 ± 0.28 | -0.42 ± 0.43 | 0.08786 ± 0.01740 | 0.93 ± 0.29 |
| Grocery | M4 | cross_base_off | -0.90 ± 0.11 | -1.54 ± 0.27 | 0.27172 ± 0.02036 | 3.16 ± 0.31 |
| Grocery | M4 | intra_selection_off | -0.03 ± 0.13 | -0.07 ± 0.06 | 0.07914 ± 0.01397 | 0.51 ± 0.04 |
| Grocery | M4 | knn_off | -0.17 ± 0.06 | -0.25 ± 0.10 | 0.05088 ± 0.00382 | 0.68 ± 0.14 |
| Grocery | M4 | receiver_query_roll | -0.11 ± 0.18 | -0.23 ± 0.20 | 0.08926 ± 0.00806 | 0.92 ± 0.22 |
| ele-fashion | M0 | content_off | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00000 ± 0.00000 | 0.00 ± 0.00 |
| ele-fashion | M0 | cross_base_off | -0.40 ± 0.13 | -0.71 ± 0.38 | 0.38691 ± 0.08539 | 1.72 ± 0.39 |
| ele-fashion | M0 | knn_off | -0.29 ± 0.08 | -0.82 ± 0.18 | 0.13310 ± 0.02852 | 0.74 ± 0.01 |
| ele-fashion | M1 | content_off | -0.10 ± 0.04 | -0.04 ± 0.02 | 0.17868 ± 0.04956 | 0.79 ± 0.29 |
| ele-fashion | M1 | cross_base_off | -0.08 ± 0.07 | 0.03 ± 0.24 | 0.22323 ± 0.04526 | 0.93 ± 0.08 |
| ele-fashion | M1 | knn_off | -0.13 ± 0.02 | -0.29 ± 0.16 | 0.08002 ± 0.01852 | 0.51 ± 0.05 |
| ele-fashion | M2 | content_off | -0.22 ± 0.03 | 0.10 ± 0.07 | 0.33915 ± 0.05643 | 1.35 ± 0.22 |
| ele-fashion | M2 | cross_base_off | -0.15 ± 0.08 | 0.02 ± 0.19 | 0.24375 ± 0.01676 | 1.08 ± 0.02 |
| ele-fashion | M2 | knn_off | -0.20 ± 0.13 | -0.57 ± 0.25 | 0.07966 ± 0.00400 | 0.65 ± 0.08 |
| ele-fashion | M3 | content_off | -0.29 ± 0.15 | -0.27 ± 0.52 | 0.35882 ± 0.06365 | 1.63 ± 0.55 |
| ele-fashion | M3 | cross_base_off | -0.23 ± 0.04 | -0.26 ± 0.06 | 0.24615 ± 0.04664 | 1.05 ± 0.24 |
| ele-fashion | M3 | knn_off | -0.21 ± 0.06 | -0.55 ± 0.09 | 0.09305 ± 0.02531 | 0.58 ± 0.17 |
| ele-fashion | M3 | receiver_query_roll | -0.35 ± 0.11 | -0.81 ± 0.57 | 0.32803 ± 0.04609 | 1.17 ± 0.45 |
| ele-fashion | M4 | content_off | -0.17 ± 0.12 | -0.34 ± 0.43 | 0.25394 ± 0.02609 | 1.11 ± 0.27 |
| ele-fashion | M4 | cross_base_off | -0.12 ± 0.16 | -0.11 ± 0.11 | 0.20241 ± 0.02811 | 0.90 ± 0.26 |
| ele-fashion | M4 | intra_selection_off | -0.05 ± 0.02 | -0.25 ± 0.23 | 0.09492 ± 0.01251 | 0.24 ± 0.05 |
| ele-fashion | M4 | knn_off | -0.12 ± 0.09 | -0.48 ± 0.40 | 0.08375 ± 0.01525 | 0.72 ± 0.14 |
| ele-fashion | M4 | receiver_query_roll | -0.13 ± 0.11 | -0.25 ± 0.17 | 0.19431 ± 0.05107 | 0.59 ± 0.11 |

## Expert scale and router

`message_scale_diagnostics.csv` and `expert_router_diagnostics.csv` preserve each modality and seed's one-hop physical RMS, Cross Base, KNN, IntraDelta, low-rank/residual/normalized content, eta contribution by expert, profile/output RMS, Cross cosine, routing entropy and expert share. A large residual is not sufficient evidence of useful class information; compare with paired metrics and interventions.
