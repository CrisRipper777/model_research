# R³-MAG v1 prototype: aggregate report

This report evaluates the first complete trainable prototype. Results are descriptive and use the requested qualitative reading; there is no automatic pass/fail threshold.

## A. Task Performance

Audit results are mean ± population standard deviation over model seeds 42, 43, and 44. Lower CE is better; higher Accuracy and Macro-F1 are better.

| Dataset | Variant | Audit CE | Audit Accuracy | Audit Macro-F1 |
|---|---|---:|---:|---:|
| Movies | G0 | 1.4610 ± 0.0520 | 0.5618 ± 0.0054 | 0.5029 ± 0.0056 |
| Movies | G0-FT | 1.4906 ± 0.0573 | 0.5541 ± 0.0087 | 0.4998 ± 0.0033 |
| Movies | G1 | 1.4827 ± 0.0517 | 0.5558 ± 0.0049 | 0.4995 ± 0.0037 |
| Movies | G2 | 1.4898 ± 0.0459 | 0.5595 ± 0.0054 | 0.5025 ± 0.0030 |
| Movies | G3 | 1.4897 ± 0.0459 | 0.5598 ± 0.0058 | 0.5026 ± 0.0032 |
| Movies | G4 | 1.4896 ± 0.0459 | 0.5601 ± 0.0062 | 0.5027 ± 0.0033 |
| Grocery | G0 | 0.6605 ± 0.0412 | 0.8408 ± 0.0038 | 0.7452 ± 0.0054 |
| Grocery | G0-FT | 0.7073 ± 0.0456 | 0.8382 ± 0.0047 | 0.7465 ± 0.0085 |
| Grocery | G1 | 0.7081 ± 0.0559 | 0.8392 ± 0.0018 | 0.7476 ± 0.0080 |
| Grocery | G2 | 0.7205 ± 0.0231 | 0.8395 ± 0.0028 | 0.7484 ± 0.0056 |
| Grocery | G3 | 0.6967 ± 0.0320 | 0.8392 ± 0.0026 | 0.7511 ± 0.0048 |
| Grocery | G4 | 0.6967 ± 0.0320 | 0.8392 ± 0.0026 | 0.7511 ± 0.0048 |
| ele-fashion | G0 | 0.4187 ± 0.0208 | 0.8791 ± 0.0009 | 0.7064 ± 0.0031 |
| ele-fashion | G0-FT | 0.4091 ± 0.0130 | 0.8788 ± 0.0020 | 0.7050 ± 0.0027 |
| ele-fashion | G1 | 0.4092 ± 0.0128 | 0.8784 ± 0.0019 | 0.7068 ± 0.0052 |
| ele-fashion | G2 | 0.4097 ± 0.0128 | 0.8784 ± 0.0020 | 0.7050 ± 0.0025 |
| ele-fashion | G3 | 0.4088 ± 0.0134 | 0.8784 ± 0.0018 | 0.7047 ± 0.0015 |
| ele-fashion | G4 | 0.4092 ± 0.0132 | 0.8788 ± 0.0019 | 0.7058 ± 0.0021 |

Pairwise contrasts are left minus right for each metric; lower CE and higher Accuracy/Macro-F1 favor the left variant.

| Dataset | Contrast | Δ CE | Δ Accuracy | Δ Macro-F1 |
|---|---|---:|---:|---:|
| Movies | G0-FT - G0 | 0.0296 ± 0.0116 | -0.0077 ± 0.0070 | -0.0031 ± 0.0041 |
| Movies | G1 - G0-FT | -0.0080 ± 0.0108 | 0.0017 ± 0.0047 | -0.0003 ± 0.0004 |
| Movies | G2 - G1 | 0.0071 ± 0.0066 | 0.0037 ± 0.0017 | 0.0030 ± 0.0035 |
| Movies | G3 - G2 | -0.0001 ± 0.0001 | 0.0003 ± 0.0005 | 0.0001 ± 0.0002 |
| Movies | G4 - G3 | -0.0001 ± 0.0001 | 0.0003 ± 0.0005 | 0.0001 ± 0.0001 |
| Movies | G4 - G0-FT | -0.0010 ± 0.0125 | 0.0060 ± 0.0029 | 0.0029 ± 0.0036 |
| Grocery | G0-FT - G0 | 0.0468 ± 0.0426 | -0.0026 ± 0.0009 | 0.0013 ± 0.0037 |
| Grocery | G1 - G0-FT | 0.0008 ± 0.0147 | 0.0010 ± 0.0056 | 0.0011 ± 0.0053 |
| Grocery | G2 - G1 | 0.0124 ± 0.0780 | 0.0003 ± 0.0040 | 0.0008 ± 0.0045 |
| Grocery | G3 - G2 | -0.0238 ± 0.0525 | -0.0003 ± 0.0005 | 0.0027 ± 0.0023 |
| Grocery | G4 - G3 | 0.0001 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Grocery | G4 - G0-FT | -0.0105 ± 0.0228 | 0.0010 ± 0.0021 | 0.0046 ± 0.0062 |
| ele-fashion | G0-FT - G0 | -0.0096 ± 0.0079 | -0.0003 ± 0.0018 | -0.0014 ± 0.0010 |
| ele-fashion | G1 - G0-FT | 0.0001 ± 0.0004 | -0.0003 ± 0.0006 | 0.0018 ± 0.0035 |
| ele-fashion | G2 - G1 | 0.0005 ± 0.0003 | -0.0001 ± 0.0005 | -0.0018 ± 0.0034 |
| ele-fashion | G3 - G2 | -0.0009 ± 0.0007 | 0.0001 ± 0.0004 | -0.0004 ± 0.0019 |
| ele-fashion | G4 - G3 | 0.0005 ± 0.0007 | 0.0003 ± 0.0005 | 0.0011 ± 0.0016 |
| ele-fashion | G4 - G0-FT | 0.0001 ± 0.0004 | 0.0000 ± 0.0003 | 0.0008 ± 0.0006 |

## B. Reusable Bank Behavior

Atom coefficient vectors and pairwise cosine, per-router mean use, routing entropy, and effective usage are in `mechanism_metrics.csv` and each per-run JSON. Effective usage is `exp(mean raw routing entropy)`.

| Dataset | Variant | Bank | Mean pairwise cosine | Effective usage |
|---|---|---|---:|---:|
| Movies | G1 | flat | -0.0190 | 1.9347 |
| Movies | G2 | shared + private | -0.0167 | 1.1805 |
| Movies | G3 | shared + private | -0.0166 | 1.1828 |
| Movies | G4 | shared + private | -0.0163 | 1.1845 |
| Grocery | G1 | flat | -0.0491 | 2.0851 |
| Grocery | G2 | shared + private | -0.0535 | 1.2704 |
| Grocery | G3 | shared + private | -0.0267 | 1.2519 |
| Grocery | G4 | shared + private | -0.0259 | 1.2553 |
| ele-fashion | G1 | flat | -0.0182 | 2.5697 |
| ele-fashion | G2 | shared + private | -0.0067 | 1.6902 |
| ele-fashion | G3 | shared + private | -0.0062 | 1.6859 |
| ele-fashion | G4 | shared + private | -0.0076 | 1.6612 |

Response atoms are signed DCT-initialized structural-order profiles. No semantic labels are assigned to individual atoms. A nonuniform route is not required for a valid run; report the measured behavior without imposing a uniform-usage target.

## C. Shared/Private Behavior

G2–G4 path norms and shared/private cosine are reported in `mechanism_metrics.csv`. Normal-vs-SharedOff and Normal-vs-PrivateOff deltas use intervention metric minus normal metric; positive CE delta or negative Accuracy/F1 delta indicates degradation under that intervention.

| Dataset | Variant | Shared-off ΔCE / ΔAcc / ΔF1 | Private-off ΔCE / ΔAcc / ΔF1 |
|---|---|---|---|
| Movies | G2 | -0.0001 / +0.0007 / +0.0005 | +0.0001 / -0.0003 / -0.0004 |
| Movies | G3 | -0.0001 / +0.0003 / +0.0004 | +0.0001 / -0.0003 / -0.0004 |
| Movies | G4 | -0.0000 / +0.0003 / +0.0003 | +0.0001 / -0.0003 / -0.0004 |
| Grocery | G2 | +0.0001 / +0.0000 / -0.0000 | -0.0002 / -0.0003 / -0.0005 |
| Grocery | G3 | +0.0000 / +0.0003 / +0.0002 | -0.0001 / -0.0003 / -0.0006 |
| Grocery | G4 | +0.0000 / +0.0003 / +0.0002 | -0.0001 / -0.0003 / -0.0006 |
| ele-fashion | G2 | +0.0001 / +0.0000 / +0.0001 | -0.0002 / +0.0001 / +0.0001 |
| ele-fashion | G3 | +0.0000 / -0.0003 / -0.0002 | -0.0002 / -0.0001 / +0.0006 |
| ele-fashion | G4 | +0.0000 / -0.0003 / -0.0002 | -0.0001 / -0.0003 / -0.0002 |

Same-checkpoint adaptation utility is reported for each adapted variant. This is an intervention diagnostic, not a causal effect.

| Dataset | Variant | Mean utility | Positive fraction | Harmful fraction |
|---|---|---:|---:|---:|
| Movies | G1 | 0.0000 ± 0.0003 | 0.4516 ± 0.0225 | 0.5484 ± 0.0225 |
| Movies | G2 | 0.0001 ± 0.0003 | 0.4412 ± 0.0208 | 0.5588 ± 0.0208 |
| Movies | G3 | 0.0001 ± 0.0002 | 0.4409 ± 0.0182 | 0.5591 ± 0.0182 |
| Movies | G4 | 0.0001 ± 0.0002 | 0.4422 ± 0.0176 | 0.5574 ± 0.0172 |
| Grocery | G1 | 0.0002 ± 0.0001 | 0.3790 ± 0.0704 | 0.6132 ± 0.0750 |
| Grocery | G2 | -0.0001 ± 0.0001 | 0.3249 ± 0.0511 | 0.6712 ± 0.0499 |
| Grocery | G3 | -0.0001 ± 0.0001 | 0.2997 ± 0.0615 | 0.6924 ± 0.0579 |
| Grocery | G4 | -0.0001 ± 0.0001 | 0.2981 ± 0.0634 | 0.6856 ± 0.0538 |
| ele-fashion | G1 | -0.0001 ± 0.0001 | 0.4089 ± 0.0713 | 0.5817 ± 0.0749 |
| ele-fashion | G2 | -0.0002 ± 0.0002 | 0.3873 ± 0.0925 | 0.6053 ± 0.0941 |
| ele-fashion | G3 | -0.0002 ± 0.0002 | 0.3849 ± 0.0946 | 0.6071 ± 0.0955 |
| ele-fashion | G4 | -0.0002 ± 0.0002 | 0.3839 ± 0.0951 | 0.6055 ± 0.0944 |

## D. Reliability Behavior

G3/G4 reliability quartiles sort Audit nodes by mean of Text and Visual rho. Adaptation utility is `CE_preserve - CE_normal`; positive values mean normal adaptation helped. Spearman is functional diagnostic evidence only; no significance threshold is imposed.

| Dataset | Variant | Mean rho | Rho std | Harmful fraction | Spearman(rho, utility) |
|---|---|---:|---:|---:|---:|
| Movies | G3 | 0.1027 ± 0.0074 | 0.0010 ± 0.0004 | 0.5591 ± 0.0182 | 0.0243 |
| Movies | G4 | 0.0832 ± 0.0224 | 0.0009 ± 0.0003 | 0.5574 ± 0.0172 | 0.0096 |
| Grocery | G3 | 0.0544 ± 0.0436 | 0.0010 ± 0.0007 | 0.6924 ± 0.0579 | 0.0219 |
| Grocery | G4 | 0.0477 ± 0.0481 | 0.0004 ± 0.0001 | 0.6856 ± 0.0538 | 0.0448 |
| ele-fashion | G3 | 0.0956 ± 0.0200 | 0.0011 ± 0.0009 | 0.6071 ± 0.0955 | -0.0178 |
| ele-fashion | G4 | 0.0757 ± 0.0308 | 0.0008 ± 0.0005 | 0.6055 ± 0.0944 | -0.1045 |

| Dataset | Variant / quartile | Mean adaptation utility | Harmful fraction | Mean CE | Accuracy |
|---|---|---:|---:|---:|---:|
| Movies | G3 Q1 | -0.0012 ± 0.0004 | 0.5573 ± 0.0236 | 1.6796 ± 0.0479 | 0.4880 ± 0.0142 |
| Movies | G3 Q2 | -0.0000 ± 0.0003 | 0.5280 ± 0.0341 | 1.5468 ± 0.0448 | 0.5320 ± 0.0408 |
| Movies | G3 Q3 | 0.0008 ± 0.0003 | 0.5408 ± 0.0068 | 1.5427 ± 0.0357 | 0.5301 ± 0.0118 |
| Movies | G3 Q4 | 0.0007 ± 0.0002 | 0.6104 ± 0.0205 | 1.1887 ± 0.1507 | 0.6894 ± 0.0561 |
| Movies | G4 Q1 | -0.0008 ± 0.0003 | 0.5360 ± 0.0285 | 1.6324 ± 0.0331 | 0.5067 ± 0.0068 |
| Movies | G4 Q2 | -0.0003 ± 0.0001 | 0.5547 ± 0.0180 | 1.5917 ± 0.1068 | 0.5320 ± 0.0226 |
| Movies | G4 Q3 | 0.0007 ± 0.0005 | 0.5435 ± 0.0148 | 1.4868 ± 0.0581 | 0.5475 ± 0.0360 |
| Movies | G4 Q4 | 0.0006 ± 0.0001 | 0.5957 ± 0.0165 | 1.2466 ± 0.1923 | 0.6546 ± 0.0623 |
| Grocery | G3 Q1 | -0.0001 ± 0.0003 | 0.6484 ± 0.0431 | 0.8714 ± 0.2267 | 0.8151 ± 0.0453 |
| Grocery | G3 Q2 | -0.0001 ± 0.0003 | 0.6914 ± 0.0609 | 0.7952 ± 0.0362 | 0.8138 ± 0.0037 |
| Grocery | G3 Q3 | -0.0002 ± 0.0001 | 0.7163 ± 0.0691 | 0.6379 ± 0.0971 | 0.8562 ± 0.0287 |
| Grocery | G3 Q4 | 0.0001 ± 0.0002 | 0.7137 ± 0.0666 | 0.4812 ± 0.1867 | 0.8719 ± 0.0304 |
| Grocery | G4 Q1 | -0.0000 ± 0.0002 | 0.6211 ± 0.0194 | 1.0248 ± 0.3203 | 0.7656 ± 0.0776 |
| Grocery | G4 Q2 | -0.0002 ± 0.0004 | 0.6953 ± 0.0556 | 0.7160 ± 0.0177 | 0.8398 ± 0.0096 |
| Grocery | G4 Q3 | -0.0001 ± 0.0000 | 0.7150 ± 0.0716 | 0.5379 ± 0.1949 | 0.8771 ± 0.0463 |
| Grocery | G4 Q4 | 0.0000 ± 0.0001 | 0.7111 ± 0.0720 | 0.5069 ± 0.1769 | 0.8745 ± 0.0333 |
| ele-fashion | G3 Q1 | -0.0002 ± 0.0003 | 0.5823 ± 0.0780 | 0.4730 ± 0.0426 | 0.8624 ± 0.0232 |
| ele-fashion | G3 Q2 | -0.0002 ± 0.0003 | 0.6078 ± 0.1088 | 0.3949 ± 0.0173 | 0.8856 ± 0.0076 |
| ele-fashion | G3 Q3 | 0.0001 ± 0.0002 | 0.6082 ± 0.1055 | 0.3705 ± 0.0326 | 0.8893 ± 0.0109 |
| ele-fashion | G3 Q4 | -0.0004 ± 0.0007 | 0.6301 ± 0.0897 | 0.3967 ± 0.0236 | 0.8763 ± 0.0114 |
| ele-fashion | G4 Q1 | -0.0002 ± 0.0001 | 0.5882 ± 0.0907 | 0.3299 ± 0.0955 | 0.9047 ± 0.0262 |
| ele-fashion | G4 Q2 | -0.0002 ± 0.0002 | 0.5889 ± 0.1005 | 0.3721 ± 0.0361 | 0.8918 ± 0.0100 |
| ele-fashion | G4 Q3 | 0.0001 ± 0.0001 | 0.6107 ± 0.0963 | 0.4342 ± 0.0446 | 0.8681 ± 0.0142 |
| ele-fashion | G4 Q4 | -0.0004 ± 0.0007 | 0.6342 ± 0.0905 | 0.5007 ± 0.0323 | 0.8504 ± 0.0288 |

## E. Intervention Analysis

All intervention deltas are intervention minus normal on the same best checkpoint. `preserve` sets adaptation scale to zero; `uniform_route` sets every response router distribution to uniform. Intervention results do not participate in checkpoint selection.

| Dataset | Variant | Intervention | Δ Audit CE | Δ Accuracy | Δ Macro-F1 |
|---|---|---|---:|---:|---:|
| Movies seed42 | G0 | preserve | +0.00000 | +0.00000 | +0.00000 |
| Movies seed43 | G0 | preserve | +0.00000 | +0.00000 | +0.00000 |
| Movies seed44 | G0 | preserve | +0.00000 | +0.00000 | +0.00000 |
| Movies seed42 | G0-FT | preserve | +0.00000 | +0.00000 | +0.00000 |
| Movies seed43 | G0-FT | preserve | +0.00000 | +0.00000 | +0.00000 |
| Movies seed44 | G0-FT | preserve | +0.00000 | +0.00000 | +0.00000 |
| Movies seed42 | G1 | preserve | -0.00003 | +0.00000 | +0.00003 |
| Movies seed42 | G1 | uniform_route | -0.00005 | -0.00200 | +0.00012 |
| Movies seed43 | G1 | preserve | +0.00037 | +0.00200 | +0.00117 |
| Movies seed43 | G1 | uniform_route | +0.00012 | +0.00100 | +0.00103 |
| Movies seed44 | G1 | preserve | -0.00032 | +0.00000 | -0.00112 |
| Movies seed44 | G1 | uniform_route | -0.00046 | +0.00200 | +0.00078 |
| Movies seed42 | G2 | preserve | +0.00012 | +0.00100 | +0.00043 |
| Movies seed42 | G2 | uniform_route | +0.00012 | +0.00100 | +0.00085 |
| Movies seed42 | G2 | shared_off | +0.00000 | +0.00100 | +0.00046 |
| Movies seed42 | G2 | private_off | -0.00000 | +0.00000 | +0.00000 |
| Movies seed43 | G2 | preserve | +0.00035 | -0.00200 | -0.00154 |
| Movies seed43 | G2 | uniform_route | -0.00005 | +0.00100 | +0.00019 |
| Movies seed43 | G2 | shared_off | -0.00007 | +0.00100 | +0.00105 |
| Movies seed43 | G2 | private_off | +0.00013 | -0.00100 | -0.00110 |
| Movies seed44 | G2 | preserve | -0.00032 | +0.00200 | +0.00191 |
| Movies seed44 | G2 | uniform_route | -0.00057 | +0.00301 | +0.00265 |
| Movies seed44 | G2 | shared_off | -0.00011 | +0.00000 | +0.00000 |
| Movies seed44 | G2 | private_off | +0.00009 | +0.00000 | +0.00000 |
| Movies seed42 | G3 | preserve | +0.00008 | +0.00000 | -0.00007 |
| Movies seed42 | G3 | uniform_route | -0.00024 | +0.00000 | +0.00019 |
| Movies seed42 | G3 | shared_off | +0.00000 | +0.00000 | +0.00000 |
| Movies seed42 | G3 | private_off | -0.00000 | +0.00000 | +0.00000 |
| Movies seed43 | G3 | preserve | +0.00034 | -0.00200 | -0.00154 |
| Movies seed43 | G3 | uniform_route | -0.00009 | +0.00100 | +0.00013 |
| Movies seed43 | G3 | shared_off | -0.00007 | +0.00100 | +0.00105 |
| Movies seed43 | G3 | private_off | +0.00013 | -0.00100 | -0.00110 |
| Movies seed44 | G3 | preserve | -0.00027 | +0.00200 | +0.00191 |
| Movies seed44 | G3 | uniform_route | -0.00052 | +0.00301 | +0.00265 |
| Movies seed44 | G3 | shared_off | -0.00010 | +0.00000 | +0.00000 |
| Movies seed44 | G3 | private_off | +0.00007 | +0.00000 | +0.00000 |
| Movies seed42 | G4 | preserve | +0.00005 | -0.00100 | -0.00023 |
| Movies seed42 | G4 | uniform_route | -0.00023 | -0.00100 | -0.00019 |
| Movies seed42 | G4 | shared_off | +0.00000 | +0.00000 | +0.00000 |
| Movies seed42 | G4 | private_off | +0.00000 | +0.00000 | +0.00000 |
| Movies seed43 | G4 | preserve | +0.00034 | -0.00200 | -0.00155 |
| Movies seed43 | G4 | uniform_route | -0.00008 | +0.00100 | +0.00012 |
| Movies seed43 | G4 | shared_off | -0.00007 | +0.00100 | +0.00104 |
| Movies seed43 | G4 | private_off | +0.00013 | -0.00100 | -0.00111 |
| Movies seed44 | G4 | preserve | -0.00021 | +0.00200 | +0.00191 |
| Movies seed44 | G4 | uniform_route | -0.00043 | +0.00301 | +0.00265 |
| Movies seed44 | G4 | shared_off | -0.00008 | +0.00000 | +0.00000 |
| Movies seed44 | G4 | private_off | +0.00006 | +0.00000 | +0.00000 |
| Grocery seed42 | G0 | preserve | +0.00000 | +0.00000 | +0.00000 |
| Grocery seed43 | G0 | preserve | +0.00000 | +0.00000 | +0.00000 |
| Grocery seed44 | G0 | preserve | +0.00000 | +0.00000 | +0.00000 |
| Grocery seed42 | G0-FT | preserve | +0.00000 | +0.00000 | +0.00000 |
| Grocery seed43 | G0-FT | preserve | +0.00000 | +0.00000 | +0.00000 |
| Grocery seed44 | G0-FT | preserve | +0.00000 | +0.00000 | +0.00000 |
| Grocery seed42 | G1 | preserve | +0.00034 | +0.00000 | +0.00018 |
| Grocery seed42 | G1 | uniform_route | +0.00116 | +0.00098 | +0.00077 |
| Grocery seed43 | G1 | preserve | +0.00001 | +0.00000 | -0.00043 |
| Grocery seed43 | G1 | uniform_route | +0.00047 | +0.00098 | +0.00080 |
| Grocery seed44 | G1 | preserve | +0.00022 | +0.00000 | +0.00000 |
| Grocery seed44 | G1 | uniform_route | +0.00038 | +0.00000 | +0.00080 |
| Grocery seed42 | G2 | preserve | -0.00017 | -0.00196 | -0.00630 |
| Grocery seed42 | G2 | uniform_route | +0.00128 | +0.00098 | +0.00066 |
| Grocery seed42 | G2 | shared_off | +0.00000 | +0.00000 | +0.00000 |
| Grocery seed42 | G2 | private_off | -0.00000 | +0.00000 | +0.00000 |
| Grocery seed43 | G2 | preserve | -0.00026 | +0.00196 | +0.00138 |
| Grocery seed43 | G2 | uniform_route | +0.00083 | +0.00391 | +0.00341 |
| Grocery seed43 | G2 | shared_off | +0.00011 | +0.00098 | +0.00069 |
| Grocery seed43 | G2 | private_off | -0.00021 | -0.00098 | -0.00168 |
| Grocery seed44 | G2 | preserve | +0.00001 | -0.00196 | -0.00132 |
| Grocery seed44 | G2 | uniform_route | +0.00086 | -0.00098 | -0.00065 |
| Grocery seed44 | G2 | shared_off | +0.00008 | -0.00098 | -0.00077 |
| Grocery seed44 | G2 | private_off | -0.00034 | +0.00000 | +0.00010 |
| Grocery seed42 | G3 | preserve | -0.00008 | +0.00098 | +0.00043 |
| Grocery seed42 | G3 | uniform_route | +0.00017 | +0.00098 | +0.00043 |
| Grocery seed42 | G3 | shared_off | -0.00000 | +0.00000 | +0.00000 |
| Grocery seed42 | G3 | private_off | -0.00000 | +0.00000 | +0.00000 |
| Grocery seed43 | G3 | preserve | -0.00025 | +0.00196 | +0.00138 |
| Grocery seed43 | G3 | uniform_route | +0.00081 | +0.00391 | +0.00341 |
| Grocery seed43 | G3 | shared_off | +0.00011 | +0.00098 | +0.00069 |
| Grocery seed43 | G3 | private_off | -0.00021 | -0.00098 | -0.00168 |
| Grocery seed44 | G3 | preserve | +0.00012 | +0.00000 | +0.00000 |
| Grocery seed44 | G3 | uniform_route | +0.00023 | +0.00000 | +0.00000 |
| Grocery seed44 | G3 | shared_off | +0.00000 | +0.00000 | +0.00000 |
| Grocery seed44 | G3 | private_off | +0.00000 | +0.00000 | +0.00000 |
| Grocery seed42 | G4 | preserve | -0.00004 | +0.00098 | +0.00043 |
| Grocery seed42 | G4 | uniform_route | +0.00012 | +0.00098 | +0.00043 |
| Grocery seed42 | G4 | shared_off | -0.00000 | +0.00000 | +0.00000 |
| Grocery seed42 | G4 | private_off | -0.00000 | +0.00000 | +0.00000 |
| Grocery seed43 | G4 | preserve | -0.00025 | +0.00196 | +0.00138 |
| Grocery seed43 | G4 | uniform_route | +0.00080 | +0.00391 | +0.00341 |
| Grocery seed43 | G4 | shared_off | +0.00011 | +0.00098 | +0.00069 |
| Grocery seed43 | G4 | private_off | -0.00021 | -0.00098 | -0.00168 |
| Grocery seed44 | G4 | preserve | +0.00006 | +0.00000 | +0.00000 |
| Grocery seed44 | G4 | uniform_route | +0.00012 | +0.00000 | +0.00000 |
| Grocery seed44 | G4 | shared_off | +0.00000 | +0.00000 | +0.00000 |
| Grocery seed44 | G4 | private_off | +0.00000 | +0.00000 | +0.00000 |
| ele-fashion seed42 | G0 | preserve | +0.00000 | +0.00000 | +0.00000 |
| ele-fashion seed43 | G0 | preserve | +0.00000 | +0.00000 | +0.00000 |
| ele-fashion seed44 | G0 | preserve | +0.00000 | +0.00000 | +0.00000 |
| ele-fashion seed42 | G0-FT | preserve | +0.00000 | +0.00000 | +0.00000 |
| ele-fashion seed43 | G0-FT | preserve | +0.00000 | +0.00000 | +0.00000 |
| ele-fashion seed44 | G0-FT | preserve | +0.00000 | +0.00000 | +0.00000 |
| ele-fashion seed42 | G1 | preserve | +0.00007 | +0.00017 | -0.00008 |
| ele-fashion seed42 | G1 | uniform_route | +0.00028 | -0.00017 | -0.00007 |
| ele-fashion seed43 | G1 | preserve | -0.00018 | -0.00017 | -0.00315 |
| ele-fashion seed43 | G1 | uniform_route | +0.00017 | -0.00034 | -0.00323 |
| ele-fashion seed44 | G1 | preserve | -0.00021 | +0.00017 | +0.00019 |
| ele-fashion seed44 | G1 | uniform_route | +0.00021 | +0.00017 | +0.00009 |
| ele-fashion seed42 | G2 | preserve | +0.00010 | +0.00068 | -0.00005 |
| ele-fashion seed42 | G2 | uniform_route | +0.00063 | +0.00017 | -0.00083 |
| ele-fashion seed42 | G2 | shared_off | +0.00013 | -0.00017 | -0.00006 |
| ele-fashion seed42 | G2 | private_off | -0.00010 | +0.00017 | +0.00018 |
| ele-fashion seed43 | G2 | preserve | -0.00014 | +0.00068 | +0.00220 |
| ele-fashion seed43 | G2 | uniform_route | +0.00067 | +0.00034 | +0.00177 |
| ele-fashion seed43 | G2 | shared_off | +0.00008 | +0.00068 | +0.00043 |
| ele-fashion seed43 | G2 | private_off | -0.00009 | +0.00000 | +0.00000 |
| ele-fashion seed44 | G2 | preserve | -0.00049 | +0.00000 | +0.00027 |
| ele-fashion seed44 | G2 | uniform_route | +0.00049 | -0.00017 | -0.00037 |
| ele-fashion seed44 | G2 | shared_off | -0.00003 | -0.00051 | -0.00011 |
| ele-fashion seed44 | G2 | private_off | -0.00037 | +0.00000 | +0.00027 |
| ele-fashion seed42 | G3 | preserve | +0.00008 | -0.00034 | -0.00048 |
| ele-fashion seed42 | G3 | uniform_route | +0.00033 | -0.00068 | -0.00108 |
| ele-fashion seed42 | G3 | shared_off | +0.00005 | -0.00017 | -0.00031 |
| ele-fashion seed42 | G3 | private_off | -0.00003 | -0.00017 | -0.00008 |
| ele-fashion seed43 | G3 | preserve | -0.00014 | -0.00017 | +0.00117 |
| ele-fashion seed43 | G3 | uniform_route | +0.00048 | -0.00017 | +0.00073 |
| ele-fashion seed43 | G3 | shared_off | +0.00004 | -0.00017 | +0.00016 |
| ele-fashion seed43 | G3 | private_off | -0.00006 | +0.00000 | +0.00209 |
| ele-fashion seed44 | G3 | preserve | -0.00048 | -0.00017 | -0.00018 |
| ele-fashion seed44 | G3 | uniform_route | +0.00049 | -0.00034 | -0.00082 |
| ele-fashion seed44 | G3 | shared_off | -0.00003 | -0.00068 | -0.00056 |
| ele-fashion seed44 | G3 | private_off | -0.00036 | -0.00017 | -0.00018 |
| ele-fashion seed42 | G4 | preserve | +0.00005 | -0.00034 | -0.00034 |
| ele-fashion seed42 | G4 | uniform_route | +0.00020 | -0.00034 | -0.00022 |
| ele-fashion seed42 | G4 | shared_off | +0.00004 | +0.00000 | +0.00000 |
| ele-fashion seed42 | G4 | private_off | -0.00002 | -0.00017 | -0.00008 |
| ele-fashion seed43 | G4 | preserve | -0.00007 | +0.00017 | +0.00216 |
| ele-fashion seed43 | G4 | uniform_route | +0.00037 | +0.00034 | +0.00243 |
| ele-fashion seed43 | G4 | shared_off | +0.00004 | -0.00017 | -0.00019 |
| ele-fashion seed43 | G4 | private_off | -0.00005 | -0.00051 | -0.00040 |
| ele-fashion seed44 | G4 | preserve | -0.00048 | -0.00017 | -0.00018 |
| ele-fashion seed44 | G4 | uniform_route | +0.00049 | -0.00034 | -0.00082 |
| ele-fashion seed44 | G4 | shared_off | -0.00003 | -0.00068 | -0.00056 |
| ele-fashion seed44 | G4 | private_off | -0.00036 | -0.00017 | -0.00018 |

## First-pass qualitative reading

- Reusable response bank: **MIXED**. Read G1 vs G0-FT on Movies/Grocery together with uniform-route interventions; no minimum percentage-point gain is required.
- Shared/private organization: **MIXED**. The reading includes G2 vs G1 task direction and whether both paths are active.
- Preserve-or-adapt reliability: **MIXED**. Read task metrics, harmful adaptation, rho quartiles, and rho/utility correlation together; no significance cutoff is applied.
- G3 vs G4 has no consistent two-dataset pattern isolating the budget setting; retain the per-dataset differences for review.
- These first-screen grades do not automatically reject the Structural Response direction.


## F. Boundary / Negative Results

- Each Stage-1 checkpoint is trained once per dataset and model seed. G0-FT/G1/G2/G3/G4 reuse that same best checkpoint and external classifier initialization.
- DevTrain is HostTrain union ResponseTrain (90% of the original train labels); Audit is the remaining 10%. Original validation selects checkpoints.
- Split hashes are checked against the tracked H1 records. Test indices are retained only as split metadata. **TEST SPLIT UNTOUCHED.** No test labels are indexed and no test metrics are calculated.
- Prototype absolute values are not directly compared with H1/H2 host values, which used 80% HostTrain.
- Training settings and architecture are dataset-independent. Smoke runs are not included in formal reports.
- Negative and mixed results remain visible in the raw per-run data; the report makes no automatic +1 percentage-point failure judgment.
- `protocol_deviations` is recorded per run; an empty list means none were detected.
