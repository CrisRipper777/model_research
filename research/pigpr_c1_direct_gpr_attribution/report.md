# PIGPR-C1 — Direct GPR Attribution Audit

## Execution protocol

- Required base SHA `66017c7a83af127d842e58238cd88cc2e0aa08ff`; branch `exp/pigpr_c1_direct_gpr_attribution`; 54/54 fixed-split NC runs completed.
- Datasets: Movies, Grocery, ele-fashion; variants RU/RUD/R0UD/RFD/RGD/AGD; seeds 42/43/44. Validation Accuracy selected checkpoints.
- `evaluate_test=false` and `development_no_test=true` for every NC run. No test metric was evaluated or recorded. No Toys, Reddit-S, formal LP, or mid-campaign tuning.
- Design audit preceded implementation. Full test suite, RU regression, 9 AGD repeats, six NC smokes, and two LP smokes are recorded in `smoke_status.json`.
- The first ele-fashion RU seed-42 attempt hit CUDA OOM while another process occupied GPU memory. The same cell passed after eliminating unused state computation; no hyperparameters or splits changed. Recovery details are in `run_manifest.json`.

## Validation performance

Accuracy and macro-F1 are percent; entries are mean ± population SD across three paired model seeds. CE and best epoch use mean ± population SD.

| Dataset | Variant | Val Acc % | Macro-F1 % | CE | Best epoch |
| --- | --- | --- | --- | --- | --- |
| Movies | RU | 55.38 ± 0.25 | 47.62 ± 0.78 | 1.4262 ± 0.0391 | 83.0 ± 21.0 |
| Movies | RUD | 54.75 ± 0.28 | 46.48 ± 0.11 | 1.3888 ± 0.0186 | 110.7 ± 11.3 |
| Movies | R0UD | 56.33 ± 0.38 | 49.52 ± 1.00 | 1.3368 ± 0.0083 | 84.0 ± 14.7 |
| Movies | RFD | 56.19 ± 0.26 | 49.20 ± 1.07 | 1.3489 ± 0.0107 | 113.7 ± 13.9 |
| Movies | RGD | 56.39 ± 0.17 | 49.70 ± 0.28 | 1.3642 ± 0.0412 | 92.7 ± 16.9 |
| Movies | AGD | 56.46 ± 0.09 | 48.99 ± 1.19 | 1.3345 ± 0.0016 | 88.0 ± 9.1 |
| Grocery | RU | 81.71 ± 0.08 | 75.40 ± 0.29 | 0.7035 ± 0.0226 | 83.0 ± 17.0 |
| Grocery | RUD | 81.96 ± 0.04 | 73.75 ± 0.20 | 0.6697 ± 0.0039 | 90.0 ± 3.6 |
| Grocery | R0UD | 82.94 ± 0.18 | 77.17 ± 0.55 | 0.6918 ± 0.0340 | 117.0 ± 26.4 |
| Grocery | RFD | 83.09 ± 0.20 | 76.09 ± 0.35 | 0.6610 ± 0.0135 | 108.7 ± 15.8 |
| Grocery | RGD | 83.09 ± 0.19 | 77.34 ± 0.23 | 0.6706 ± 0.0083 | 109.3 ± 9.7 |
| Grocery | AGD | 82.95 ± 0.09 | 76.34 ± 0.23 | 0.6545 ± 0.0194 | 101.3 ± 15.6 |
| ele-fashion | RU | 87.44 ± 0.08 | 74.56 ± 0.52 | 0.4094 ± 0.0073 | 149.7 ± 11.4 |
| ele-fashion | RUD | 85.57 ± 0.07 | 71.55 ± 0.30 | 0.4469 ± 0.0005 | 148.3 ± 1.2 |
| ele-fashion | R0UD | 87.16 ± 0.04 | 74.74 ± 0.50 | 0.4116 ± 0.0036 | 170.0 ± 8.6 |
| ele-fashion | RFD | 86.41 ± 0.42 | 72.19 ± 1.63 | 0.4246 ± 0.0145 | 116.7 ± 42.3 |
| ele-fashion | RGD | 87.32 ± 0.04 | 74.54 ± 0.25 | 0.4165 ± 0.0179 | 164.0 ± 28.6 |
| ele-fashion | AGD | 87.67 ± 0.12 | 75.75 ± 0.20 | 0.4427 ± 0.0180 | 253.7 ± 21.0 |

## Paired comparisons

Accuracy and F1 are percentage-point deltas; CE is raw CE difference. Signs are positive/negative/tie model seeds. Three model seeds on one fixed split are descriptive paired comparisons, not pseudo-IID inference.

| Dataset | Comparison | Δ Acc pp ± SD | +/−/tie | Δ F1 pp | Δ CE |
| --- | --- | --- | --- | --- | --- |
| Movies | RUD-RU | -0.630 ± 0.465 | 0/3/0 | -1.137 | -0.0375 |
| Movies | R0UD-RUD | +1.580 ± 0.655 | 3/0/0 | +3.037 | -0.0520 |
| Movies | RFD-R0UD | -0.140 ± 0.126 | 1/2/0 | -0.313 | +0.0121 |
| Movies | RGD-RFD | +0.200 ± 0.208 | 2/1/0 | +0.493 | +0.0153 |
| Movies | AGD-RGD | +0.070 ± 0.159 | 2/1/0 | -0.706 | -0.0297 |
| Movies | RGD-RU | +1.010 ± 0.099 | 3/0/0 | +2.081 | -0.0621 |
| Movies | AGD-RU | +1.080 ± 0.258 | 3/0/0 | +1.375 | -0.0917 |
| Movies | RGD-RUD | +1.640 ± 0.417 | 3/0/0 | +3.218 | -0.0246 |
| Movies | AGD-RUD | +1.710 ± 0.366 | 3/0/0 | +2.512 | -0.0543 |
| Grocery | RUD-RU | +0.254 ± 0.050 | 3/0/0 | -1.646 | -0.0338 |
| Grocery | R0UD-RUD | +0.976 ± 0.168 | 3/0/0 | +3.416 | +0.0221 |
| Grocery | RFD-R0UD | +0.156 ± 0.195 | 2/1/0 | -1.079 | -0.0308 |
| Grocery | RGD-RFD | +0.000 ± 0.291 | 1/2/0 | +1.251 | +0.0096 |
| Grocery | AGD-RGD | -0.146 ± 0.213 | 1/2/0 | -1.001 | -0.0161 |
| Grocery | RGD-RU | +1.386 ± 0.113 | 3/0/0 | +1.942 | -0.0329 |
| Grocery | AGD-RU | +1.240 ± 0.136 | 3/0/0 | +0.940 | -0.0490 |
| Grocery | RGD-RUD | +1.132 ± 0.163 | 3/0/0 | +3.588 | +0.0009 |
| Grocery | AGD-RUD | +0.986 ± 0.120 | 3/0/0 | +2.586 | -0.0152 |
| ele-fashion | RUD-RU | -1.875 ± 0.089 | 0/3/0 | -3.018 | +0.0376 |
| ele-fashion | R0UD-RUD | +1.589 ± 0.067 | 3/0/0 | +3.198 | -0.0353 |
| ele-fashion | RFD-R0UD | -0.743 ± 0.411 | 0/3/0 | -2.553 | +0.0129 |
| ele-fashion | RGD-RFD | +0.910 ± 0.416 | 3/0/0 | +2.346 | -0.0081 |
| ele-fashion | AGD-RGD | +0.348 ± 0.142 | 3/0/0 | +1.212 | +0.0262 |
| ele-fashion | RGD-RU | -0.119 ± 0.039 | 0/3/0 | -0.027 | +0.0071 |
| ele-fashion | AGD-RU | +0.228 ± 0.173 | 2/0/1 | +1.185 | +0.0333 |
| ele-fashion | RGD-RUD | +1.756 ± 0.071 | 3/0/0 | +2.991 | -0.0304 |
| ele-fashion | AGD-RUD | +2.104 ± 0.186 | 3/0/0 | +4.203 | -0.0042 |

- **RUD-RU:** Movies -0.630 pp (0/3 positive); Grocery +0.254 pp (3/3 positive); ele-fashion -1.875 pp (0/3 positive).
- **R0UD-RUD:** Movies +1.580 pp (3/3 positive); Grocery +0.976 pp (3/3 positive); ele-fashion +1.589 pp (3/3 positive).
- **RFD-R0UD:** Movies -0.140 pp (1/3 positive); Grocery +0.156 pp (2/3 positive); ele-fashion -0.743 pp (0/3 positive).
- **RGD-RFD:** Movies +0.200 pp (2/3 positive); Grocery +0.000 pp (1/3 positive); ele-fashion +0.910 pp (3/3 positive).
- **AGD-RGD:** Movies +0.070 pp (2/3 positive); Grocery -0.146 pp (1/3 positive); ele-fashion +0.348 pp (3/3 positive).
- **RGD-RU:** Movies +1.010 pp (3/3 positive); Grocery +1.386 pp (3/3 positive); ele-fashion -0.119 pp (0/3 positive).
- **AGD-RU:** Movies +1.080 pp (3/3 positive); Grocery +1.240 pp (3/3 positive); ele-fashion +0.228 pp (2/3 positive).
- **RGD-RUD:** Movies +1.640 pp (3/3 positive); Grocery +1.132 pp (3/3 positive); ele-fashion +1.756 pp (3/3 positive).
- **AGD-RUD:** Movies +1.710 pp (3/3 positive); Grocery +0.986 pp (3/3 positive); ele-fashion +2.104 pp (3/3 positive).

RUD−RU is interpreted as a direct-versus-protected parameterization/optimization comparison. C0 AGD−AGP has the same interpretation boundary: when polynomial coefficients are unrestricted, the protected skip can be reparameterized into effective polynomial coefficients.

RGD and AGD have the same polynomial function family because S=M H with invertible triangular M. Their initial effective coefficients, proposal, projector, and fusion are matched. AGD−RGD therefore tests basis-dependent optimization or implicit regularization, not greater expressive capacity.

## AGD same-seed execution repeatability

Each dataset uses three independent full executions of AGD seed 42. This estimates execution noise, not architecture variance.

| Dataset | Metric | Mean | Population SD | Range |
| --- | --- | --- | --- | --- |
| Movies | Accuracy (%) | 56.4987 | 0.0141389 | 0.0299931 |
| Movies | Macro-F1 (%) | 49.4172 | 0.0223644 | 0.0478842 |
| Movies | CE | 1.35322 | 0.0122673 | 0.0279882 |
| Movies | Best epoch | 100.333 | 1.88562 | 4 |
| Grocery | Accuracy (%) | 82.8502 | 0.0905194 | 0.20498 |
| Grocery | Macro-F1 (%) | 76.995 | 0.307335 | 0.748053 |
| Grocery | CE | 0.658985 | 0.0111286 | 0.0266617 |
| Grocery | Best epoch | 106.333 | 10.403 | 23 |
| ele-fashion | Accuracy (%) | 87.4297 | 0.167648 | 0.40912 |
| ele-fashion | Macro-F1 (%) | 75.143 | 0.983062 | 2.29987 |
| ele-fashion | CE | 0.425581 | 0.00888918 | 0.021774 |
| ele-fashion | Best epoch | 205 | 48.3322 | 108 |

Accuracy repeatability context: Movies Acc SD/range 0.014/0.030 pp; Grocery Acc SD/range 0.091/0.205 pp; ele-fashion Acc SD/range 0.168/0.409 pp.

## Coefficient diagnostics

RFD and RGD report raw monomial `c0..c3`; AGD reports both learned anchored `gamma0..gamma3` and `c=Mᵀgamma`. The CSV includes signed coefficients, negative counts, L1, sum, absolute-weighted effective order, and `c/(sum|c|+eps)`. Coefficients are not normalized by softmax or L1. Because the final modality embedding uses LayerNorm, the overall positive coefficient scale is weakly identifiable; compare normalized shape and functional contributions too.

| Dataset | Variant | c0 | c1 | c2 | c3 | γ0 | γ1 | γ2 | γ3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Movies | RFD | +0.1500 | +0.1275 | +0.7225 | +0.0000 | — | — | — | — |
| Movies | RGD | +0.2363 | +0.0689 | +0.6288 | -0.0911 | — | — | — | — |
| Movies | AGD | +0.2147 | +0.0740 | +0.6369 | -0.0665 | +0.1431 | +0.0119 | +0.7954 | -0.0913 |
| Grocery | RFD | +0.1500 | +0.1275 | +0.7225 | +0.0000 | — | — | — | — |
| Grocery | RGD | +0.2046 | +0.0706 | +0.6660 | -0.0533 | — | — | — | — |
| Grocery | AGD | +0.1903 | +0.1068 | +0.6727 | -0.0356 | +0.1077 | +0.0400 | +0.8354 | -0.0489 |
| ele-fashion | RFD | +0.1500 | +0.1275 | +0.7225 | +0.0000 | — | — | — | — |
| ele-fashion | RGD | +0.2654 | +0.0284 | +0.5819 | -0.1338 | — | — | — | — |
| ele-fashion | AGD | +0.2600 | +0.0551 | +0.5085 | -0.1593 | +0.2151 | +0.0181 | +0.6497 | -0.2186 |

Each displayed coefficient is averaged across the three seeds; the full checkpoint rows are in `coefficient_diagnostics.csv`.

## Effective monomial-order contributions

For RGD and AGD, `term_k=c_k H_k` and `G=sum c_k H_k` are computed in the same raw monomial basis. Rows aggregate three seeds per dataset, modality, and order. Contribution size is `RMS(term_k)/RMS(G)`; it is not inferred from coefficient magnitude alone.

| Dataset | Variant | Modality | k | RMS(Hk) | RMS(term) | term/G RMS | cos(term,G) | cos(Hk,P) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grocery | AGD | text | 0.0 | 1.0016 | 0.1906 | 0.2830 | 0.8997 | 1.0000 |
| Grocery | AGD | text | 1.0 | 0.7413 | 0.0792 | 0.1171 | 0.9723 | 0.8207 |
| Grocery | AGD | text | 2.0 | 0.6783 | 0.4565 | 0.6753 | 0.9838 | 0.8089 |
| Grocery | AGD | text | 3.0 | 0.6464 | 0.0228 | 0.0343 | -0.9611 | 0.7544 |
| Grocery | AGD | visual | 0.0 | 0.9992 | 0.1902 | 0.3129 | 0.8818 | 1.0000 |
| Grocery | AGD | visual | 1.0 | 0.6724 | 0.0719 | 0.1177 | 0.9660 | 0.7836 |
| Grocery | AGD | visual | 2.0 | 0.5955 | 0.4008 | 0.6570 | 0.9758 | 0.7598 |
| Grocery | AGD | visual | 3.0 | 0.5580 | 0.0197 | 0.0327 | -0.9457 | 0.6933 |
| Grocery | RGD | text | 0.0 | 1.0016 | 0.2049 | 0.3201 | 0.9065 | 1.0000 |
| Grocery | RGD | text | 1.0 | 0.7348 | 0.0519 | 0.0808 | 0.9675 | 0.8172 |
| Grocery | RGD | text | 2.0 | 0.6701 | 0.4463 | 0.6965 | 0.9796 | 0.8043 |
| Grocery | RGD | text | 3.0 | 0.6375 | 0.0340 | 0.0532 | -0.9545 | 0.7486 |
| Grocery | RGD | visual | 0.0 | 0.9991 | 0.2044 | 0.3513 | 0.8919 | 1.0000 |
| Grocery | RGD | visual | 1.0 | 0.6690 | 0.0473 | 0.0810 | 0.9602 | 0.7815 |
| Grocery | RGD | visual | 2.0 | 0.5915 | 0.3939 | 0.6766 | 0.9704 | 0.7573 |
| Grocery | RGD | visual | 3.0 | 0.5538 | 0.0295 | 0.0509 | -0.9378 | 0.6901 |
| Movies | AGD | text | 0.0 | 1.0032 | 0.2154 | 0.2933 | 0.9391 | 1.0000 |
| Movies | AGD | text | 1.0 | 0.8562 | 0.0634 | 0.0861 | 0.9844 | 0.8836 |
| Movies | AGD | text | 2.0 | 0.8313 | 0.5295 | 0.7205 | 0.9892 | 0.8790 |
| Movies | AGD | text | 3.0 | 0.8208 | 0.0546 | 0.0745 | -0.9796 | 0.8547 |
| Movies | AGD | visual | 0.0 | 0.9976 | 0.2142 | 0.3692 | 0.8935 | 1.0000 |
| Movies | AGD | visual | 1.0 | 0.6773 | 0.0502 | 0.0863 | 0.9613 | 0.7704 |
| Movies | AGD | visual | 2.0 | 0.6129 | 0.3904 | 0.6721 | 0.9668 | 0.7504 |
| Movies | AGD | visual | 3.0 | 0.5812 | 0.0386 | 0.0667 | -0.9404 | 0.6971 |
| Movies | RGD | text | 0.0 | 1.0030 | 0.2370 | 0.3287 | 0.9444 | 1.0000 |
| Movies | RGD | text | 1.0 | 0.8536 | 0.0589 | 0.0812 | 0.9817 | 0.8818 |
| Movies | RGD | text | 2.0 | 0.8283 | 0.5210 | 0.7208 | 0.9860 | 0.8769 |
| Movies | RGD | text | 3.0 | 0.8176 | 0.0744 | 0.1034 | -0.9753 | 0.8523 |
| Movies | RGD | visual | 0.0 | 0.9975 | 0.2357 | 0.4084 | 0.9055 | 1.0000 |
| Movies | RGD | visual | 1.0 | 0.6766 | 0.0467 | 0.0804 | 0.9550 | 0.7694 |
| Movies | RGD | visual | 2.0 | 0.6124 | 0.3852 | 0.6658 | 0.9588 | 0.7493 |
| Movies | RGD | visual | 3.0 | 0.5810 | 0.0528 | 0.0917 | -0.9302 | 0.6958 |
| ele-fashion | AGD | text | 0.0 | 1.0008 | 0.2602 | 0.4761 | 0.9636 | 1.0000 |
| ele-fashion | AGD | text | 1.0 | 0.7921 | 0.0437 | 0.0796 | 0.9106 | 0.8125 |
| ele-fashion | AGD | text | 2.0 | 0.7341 | 0.3734 | 0.6821 | 0.9614 | 0.8540 |
| ele-fashion | AGD | text | 3.0 | 0.7077 | 0.1127 | 0.2065 | -0.9065 | 0.7736 |
| ele-fashion | AGD | visual | 0.0 | 0.9976 | 0.2594 | 0.4780 | 0.9606 | 1.0000 |
| ele-fashion | AGD | visual | 1.0 | 0.7849 | 0.0433 | 0.0795 | 0.8972 | 0.7898 |
| ele-fashion | AGD | visual | 2.0 | 0.7324 | 0.3725 | 0.6855 | 0.9587 | 0.8433 |
| ele-fashion | AGD | visual | 3.0 | 0.7111 | 0.1132 | 0.2090 | -0.8992 | 0.7579 |
| ele-fashion | RGD | text | 0.0 | 1.0010 | 0.2657 | 0.4355 | 0.9593 | 1.0000 |
| ele-fashion | RGD | text | 1.0 | 0.8084 | 0.0231 | 0.0366 | 0.9143 | 0.8179 |
| ele-fashion | RGD | text | 2.0 | 0.7545 | 0.4391 | 0.7167 | 0.9707 | 0.8652 |
| ele-fashion | RGD | text | 3.0 | 0.7298 | 0.0975 | 0.1604 | -0.9203 | 0.7879 |
| ele-fashion | RGD | visual | 0.0 | 0.9985 | 0.2650 | 0.4287 | 0.9594 | 1.0000 |
| ele-fashion | RGD | visual | 1.0 | 0.8198 | 0.0236 | 0.0366 | 0.9020 | 0.8006 |
| ele-fashion | RGD | visual | 2.0 | 0.7753 | 0.4514 | 0.7260 | 0.9718 | 0.8682 |
| ele-fashion | RGD | visual | 3.0 | 0.7571 | 0.1010 | 0.1639 | -0.9190 | 0.7888 |

## Raw state similarities

Cosines use the flattened node-feature state tensors. `state_similarity.csv` retains per checkpoint and modality values.

| Dataset | Variant | Pair | Mean cosine | Population SD |
| --- | --- | --- | --- | --- |
| Grocery | AGD | H0-H1 | 0.8022 | 0.0195 |
| Grocery | AGD | H0-H2 | 0.7844 | 0.0258 |
| Grocery | AGD | H0-H3 | 0.7238 | 0.0322 |
| Grocery | AGD | H1-H2 | 0.9683 | 0.0033 |
| Grocery | AGD | H1-H3 | 0.9526 | 0.0078 |
| Grocery | AGD | H2-H3 | 0.9909 | 0.0009 |
| Grocery | RGD | H0-H1 | 0.7994 | 0.0180 |
| Grocery | RGD | H0-H2 | 0.7808 | 0.0236 |
| Grocery | RGD | H0-H3 | 0.7193 | 0.0294 |
| Grocery | RGD | H1-H2 | 0.9678 | 0.0030 |
| Grocery | RGD | H1-H3 | 0.9516 | 0.0071 |
| Grocery | RGD | H2-H3 | 0.9907 | 0.0008 |
| Movies | AGD | H0-H1 | 0.8270 | 0.0567 |
| Movies | AGD | H0-H2 | 0.8147 | 0.0645 |
| Movies | AGD | H0-H3 | 0.7759 | 0.0791 |
| Movies | AGD | H1-H2 | 0.9811 | 0.0076 |
| Movies | AGD | H1-H3 | 0.9689 | 0.0147 |
| Movies | AGD | H2-H3 | 0.9953 | 0.0022 |
| Movies | RGD | H0-H1 | 0.8256 | 0.0567 |
| Movies | RGD | H0-H2 | 0.8131 | 0.0644 |
| Movies | RGD | H0-H3 | 0.7741 | 0.0789 |
| Movies | RGD | H1-H2 | 0.9810 | 0.0077 |
| Movies | RGD | H1-H3 | 0.9686 | 0.0147 |
| Movies | RGD | H2-H3 | 0.9953 | 0.0022 |
| ele-fashion | AGD | H0-H1 | 0.8011 | 0.0114 |
| ele-fashion | AGD | H0-H2 | 0.8487 | 0.0062 |
| ele-fashion | AGD | H0-H3 | 0.7657 | 0.0088 |
| ele-fashion | AGD | H1-H2 | 0.9387 | 0.0036 |
| ele-fashion | AGD | H1-H3 | 0.9611 | 0.0010 |
| ele-fashion | AGD | H2-H3 | 0.9782 | 0.0014 |
| ele-fashion | RGD | H0-H1 | 0.8092 | 0.0095 |
| ele-fashion | RGD | H0-H2 | 0.8667 | 0.0081 |
| ele-fashion | RGD | H0-H3 | 0.7883 | 0.0099 |
| ele-fashion | RGD | H1-H2 | 0.9409 | 0.0030 |
| ele-fashion | RGD | H1-H3 | 0.9667 | 0.0029 |
| ele-fashion | RGD | H2-H3 | 0.9789 | 0.0012 |

Mean H2–H3 cosine across dataset/variant groups is 0.9882. Interpret order-3 coefficients alongside these state similarities and term contribution rows: correlated states can permit coefficient reparameterization without a correspondingly large functional correction.

## Monomial-space checkpoint interventions

The four interventions modify the selected RGD/AGD checkpoint's effective raw coefficient vector, recompose from raw `H`, then apply the same LayerNorm and frozen classifier. They measure checkpoint reliance and do not replace RGD−RFD or AGD−RGD retraining comparisons.

| Dataset | Variant | Intervention | Δ Acc pp | +/−/tie | Δ F1 pp | Δ CE |
| --- | --- | --- | --- | --- | --- | --- |
| Grocery | AGD | negative_terms_off | -0.010 | 0.0/1.0/2.0 | -0.032 | -0.0010 |
| Grocery | AGD | order0_off | -0.634 | 0.0/3.0/0.0 | -2.381 | +0.0415 |
| Grocery | AGD | order3_off | -0.010 | 0.0/1.0/2.0 | -0.032 | -0.0010 |
| Grocery | AGD | reset_prior | +0.049 | 2.0/1.0/0.0 | -0.015 | -0.0010 |
| Grocery | RGD | negative_terms_off | -0.088 | 0.0/2.0/1.0 | -0.208 | -0.0022 |
| Grocery | RGD | order0_off | -1.025 | 0.0/3.0/0.0 | -2.786 | +0.0485 |
| Grocery | RGD | order3_off | -0.088 | 0.0/2.0/1.0 | -0.208 | -0.0022 |
| Grocery | RGD | reset_prior | -0.195 | 1.0/2.0/0.0 | -0.865 | -0.0023 |
| Movies | AGD | negative_terms_off | +0.020 | 2.0/1.0/0.0 | -0.003 | -0.0034 |
| Movies | AGD | order0_off | -1.230 | 0.0/3.0/0.0 | -2.048 | +0.0351 |
| Movies | AGD | order3_off | +0.020 | 2.0/1.0/0.0 | -0.003 | -0.0034 |
| Movies | AGD | reset_prior | -0.280 | 0.0/3.0/0.0 | -0.611 | -0.0054 |
| Movies | RGD | negative_terms_off | -0.100 | 1.0/2.0/0.0 | -0.191 | -0.0053 |
| Movies | RGD | order0_off | -1.730 | 0.0/3.0/0.0 | -2.717 | +0.0406 |
| Movies | RGD | order3_off | -0.100 | 1.0/2.0/0.0 | -0.191 | -0.0053 |
| Movies | RGD | reset_prior | -0.580 | 0.0/3.0/0.0 | -0.835 | -0.0068 |
| ele-fashion | AGD | negative_terms_off | -0.065 | 1.0/2.0/0.0 | -0.317 | +0.0030 |
| ele-fashion | AGD | order0_off | -2.216 | 0.0/3.0/0.0 | -4.411 | +0.0859 |
| ele-fashion | AGD | order3_off | -0.065 | 1.0/2.0/0.0 | -0.317 | +0.0030 |
| ele-fashion | AGD | reset_prior | -0.944 | 0.0/3.0/0.0 | -2.445 | +0.0303 |
| ele-fashion | RGD | negative_terms_off | -0.092 | 0.0/3.0/0.0 | -0.125 | +0.0029 |
| ele-fashion | RGD | order0_off | -2.424 | 0.0/3.0/0.0 | -4.614 | +0.0853 |
| ele-fashion | RGD | order3_off | -0.092 | 0.0/3.0/0.0 | -0.125 | +0.0029 |
| ele-fashion | RGD | reset_prior | -0.726 | 0.0/3.0/0.0 | -1.540 | +0.0229 |

## Diagnosis

Pattern broadly matches Case C: RGD−RFD is positive or effectively tied by dataset mean, while AGD−RGD is mixed and remains below 0.35 pp in absolute size. Recommend raw direct GPR as the simpler audit basis; AGD retains the best nominal accuracy rank, but the current evidence does not show a stable cross-dataset anchored-basis gain.
Order-3/negative-term interventions show only small checkpoint changes; after comparison with the AGD repeatability SD, a repeated decrease larger than twice that noise floor appears in fewer than two datasets. A retrained signed/no3 audit is not yet compelled.

- Mean dataset rank by validation Accuracy: RU 4.33, RUD 5.67, R0UD 3.67, RFD 3.67, RGD 2.00, AGD 1.67. The descriptive rank selects **AGD** (1.67). RGD is the recommended working backbone: its learned raw GPR beats or ties RFD by dataset mean, and AGD's incremental accuracy effect is mixed and small. AGD has the best nominal mean rank, so retain it as a close comparator rather than claiming a basis advantage.
- Accuracy/F1/CE show a directional trade-off somewhere in the paired results: **yes**. For example, Movies RGD-RFD gains +0.200 pp accuracy while CE changes +0.0153. Read all three metrics in `paired_delta_summary.csv`.
- No semantic-drift-is-better claim is made. The evidence concerns task metrics, direct polynomial composition, basis coordinates, term contribution, and selected-checkpoint reliance.

## Self-audit

1. Started from `66017c7a83af127d842e58238cd88cc2e0aa08ff`: **yes**, branch ancestry and fetched remote were verified.
2. Used another experiment branch: **no**.
3. Ran or evaluated NC test: **no**; NC development mode masks test indices and all stored metrics are validation-only. LP smoke did not compute test metrics.
4. RU regression to C0 PIGPR-v0 RU: **passed**, maximum-error gate `1e-6`.
5. RFD and RGD initial function: **yes**, RGD delta starts at zero, so proposal and modality embeddings equal RFD at initialization.
6. RGD and AGD initial function: **yes**, `Mᵀ gamma_prior=c_prior`; matched same-seed projectors produce numerically equal proposals/embeddings within `1e-6`.
7. Same polynomial family: **yes**, anchored basis matrix is invertible triangular; differences indicate basis-dependent optimization/implicit regularization.
8. AGD repeatability: see the three-dataset population SD/range table above; this is execution noise, not model-seed variance.
9. RUD−RU: direct versus protected optimization/inductive-bias parameterization at matched uniform context; no function-capacity claim.
10. R0UD−RUD: effect of including the intrinsic order-0 state in direct uniform composition.
11. RFD−R0UD: effect of replacing uniform four-order context with the fixed CoSI/PPR-informed profile.
12. RGD−RFD supports learned GPR: **positive or effectively tied by dataset mean**; paired means by dataset are Movies +0.200 pp, Grocery +0.000 pp, ele-fashion +0.910 pp. The practical gain is concentrated in ele-fashion; Grocery is effectively tied.
13. AGD−RGD basis optimization effect: paired means are Movies +0.070 pp, Grocery -0.146 pp, ele-fashion +0.348 pp; see repeatability context and no capacity interpretation.
14. AGD/RGD versus RU: RGD−RU means Movies +1.010, Grocery +1.386, ele-fashion -0.119 pp; AGD−RU means Movies +1.080, Grocery +1.240, ele-fashion +0.228 pp.
15. Stable coefficient pattern across datasets: **yes in these checkpoints: c0–c2 positive and c3 negative for all nine RGD and all nine AGD fits; magnitudes vary and remain descriptive**.
16. Order-0 checkpoint reliance: **yes**; order0_off decreases accuracy for every RGD/AGD dataset mean. Deltas: RGD Movies -1.730 pp; RGD Grocery -1.025 pp; RGD ele-fashion -2.424 pp; AGD Movies -1.230 pp; AGD Grocery -0.634 pp; AGD ele-fashion -2.216 pp. These are frozen-checkpoint effects, not retrained ablations.
17. Negative/order-3 reliance: c3 is the only negative learned raw coefficient, so negative_terms_off and order3_off coincide. Accuracy deltas are RGD Movies -0.100 pp (2/3 negative); RGD Grocery -0.088 pp (2/3 negative); RGD ele-fashion -0.092 pp (3/3 negative); AGD Movies +0.020 pp (1/3 negative); AGD Grocery -0.010 pp (1/3 negative); AGD ele-fashion -0.065 pp (2/3 negative). The effects are small and mostly within the AGD repeatability floor; they do not establish a clear cross-dataset high-order dependency.
18. Coefficients versus functional contribution: **not assumed equivalent**; `order_contribution.csv` directly reports state and term RMS, normalized contribution, and cosine.
19. H2/H3 similarity: pooled group mean is 0.9882; dataset/variant values are in the state-similarity table.
20. Retrained signed/no3 audit: **not compelled yet**; fewer than two datasets show an order3_off decline exceeding twice the AGD repeatability SD with at least two of three model seeds declining. C1 stops here for review.
21. Current backbone: **RGD**, as the simpler raw direct parameterization with positive/tied RGD−RFD dataset means and no stable AGD−RGD cross-dataset gain. AGD has the best nominal validation-accuracy mean rank and remains a close comparator.
22. Accuracy/F1/CE trade-off: **yes**; For example, Movies RGD-RFD gains +0.200 pp accuracy while CE changes +0.0153. Full outcomes are in the paired table.
23. LP smoke compliant: **yes**; only sports-copurchase RGD/AGD, two epochs and two batches, test disabled.
24. Claims beyond evidence: **none intended**; findings are fixed-split validation descriptions with three model seeds and no p-values.
