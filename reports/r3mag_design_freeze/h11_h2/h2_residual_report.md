# R³-MAG H2-B minimal functional residual screen

The host is fully frozen. The adapter predicts bounded Text/Visual residuals and scalar gates, then uses the original modality LayerNorm, fusion, and classifier. B1 uses receiver features plus a zero context block; B2 uses real context; B3 uses a split-restricted shuffled context. Their architectures and parameter counts match. B2-eval-shuffle evaluates the trained B2 on independently shuffled Audit context.

## Audit metrics

| Dataset | Seed | Variant | CE | Accuracy | Macro-F1 | CE delta vs B0 | Positive improvement | Harmful | Parameters |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | B0_GlobalFrozen | 1.60051 | 0.55611 | 0.48378 | 0.00000 |  |  | None |
| Movies | 42 | B1_ReceiverResidual | 1.52883 | 0.55511 | 0.49398 | -0.07168 | 0.4599 | 0.5401 | 106080 |
| Movies | 42 | B2_ContextResidual | 1.52963 | 0.55711 | 0.49679 | -0.07089 | 0.4569 | 0.5431 | 106080 |
| Movies | 42 | B3_ShuffledContextResidual | 1.52637 | 0.55511 | 0.49493 | -0.07415 | 0.4599 | 0.5401 | 106080 |
| Movies | 42 | B2_eval_shuffle | 1.53081 | 0.55511 | 0.49641 | -0.06970 | 0.4639 | 0.5361 | None |
| Movies | 43 | B0_GlobalFrozen | 1.49603 | 0.54709 | 0.45217 | 0.00000 |  |  | None |
| Movies | 43 | B1_ReceiverResidual | 1.48326 | 0.53206 | 0.47261 | -0.01277 | 0.4208 | 0.5792 | 106080 |
| Movies | 43 | B2_ContextResidual | 1.47338 | 0.54609 | 0.48468 | -0.02265 | 0.4259 | 0.5741 | 106080 |
| Movies | 43 | B3_ShuffledContextResidual | 1.47547 | 0.54509 | 0.48089 | -0.02056 | 0.4228 | 0.5772 | 106080 |
| Movies | 43 | B2_eval_shuffle | 1.47270 | 0.54609 | 0.48421 | -0.02333 | 0.4359 | 0.5641 | None |
| Movies | 44 | B0_GlobalFrozen | 1.46912 | 0.56112 | 0.49099 | 0.00000 |  |  | None |
| Movies | 44 | B1_ReceiverResidual | 1.44687 | 0.55511 | 0.48940 | -0.02226 | 0.4429 | 0.5571 | 106080 |
| Movies | 44 | B2_ContextResidual | 1.44562 | 0.55511 | 0.48972 | -0.02350 | 0.4559 | 0.5441 | 106080 |
| Movies | 44 | B3_ShuffledContextResidual | 1.44643 | 0.55611 | 0.49064 | -0.02269 | 0.4479 | 0.5521 | 106080 |
| Movies | 44 | B2_eval_shuffle | 1.44615 | 0.55511 | 0.48968 | -0.02297 | 0.4529 | 0.5471 | None |
| Grocery | 42 | B0_GlobalFrozen | 0.61236 | 0.83562 | 0.72494 | 0.00000 |  |  | None |
| Grocery | 42 | B1_ReceiverResidual | 0.60131 | 0.83562 | 0.72701 | -0.01105 | 0.3258 | 0.6742 | 106080 |
| Grocery | 42 | B2_ContextResidual | 0.60169 | 0.83659 | 0.72778 | -0.01067 | 0.3249 | 0.6751 | 106080 |
| Grocery | 42 | B3_ShuffledContextResidual | 0.60187 | 0.83659 | 0.72778 | -0.01049 | 0.3249 | 0.6751 | 106080 |
| Grocery | 42 | B2_eval_shuffle | 0.60176 | 0.83659 | 0.72778 | -0.01060 | 0.3239 | 0.6761 | None |
| Grocery | 43 | B0_GlobalFrozen | 0.62901 | 0.83366 | 0.73855 | 0.00000 |  |  | None |
| Grocery | 43 | B1_ReceiverResidual | 0.62440 | 0.83170 | 0.74103 | -0.00462 | 0.3014 | 0.6986 | 106080 |
| Grocery | 43 | B2_ContextResidual | 0.62446 | 0.83170 | 0.74103 | -0.00456 | 0.2994 | 0.7006 | 106080 |
| Grocery | 43 | B3_ShuffledContextResidual | 0.62439 | 0.83170 | 0.74103 | -0.00462 | 0.2994 | 0.7006 | 106080 |
| Grocery | 43 | B2_eval_shuffle | 0.62443 | 0.83170 | 0.74103 | -0.00458 | 0.2994 | 0.7006 | None |
| Grocery | 44 | B0_GlobalFrozen | 0.61100 | 0.84344 | 0.74427 | 0.00000 |  |  | None |
| Grocery | 44 | B1_ReceiverResidual | 0.61466 | 0.84051 | 0.73977 | 0.00366 | 0.3493 | 0.6507 | 106080 |
| Grocery | 44 | B2_ContextResidual | 0.61541 | 0.83757 | 0.73737 | 0.00441 | 0.3611 | 0.6389 | 106080 |
| Grocery | 44 | B3_ShuffledContextResidual | 0.61423 | 0.83659 | 0.73608 | 0.00323 | 0.3571 | 0.6429 | 106080 |
| Grocery | 44 | B2_eval_shuffle | 0.61521 | 0.83659 | 0.73659 | 0.00421 | 0.3659 | 0.6341 | None |
| ele-fashion | 42 | B0_GlobalFrozen | 0.40229 | 0.87364 | 0.68324 | 0.00000 |  |  | None |
| ele-fashion | 42 | B1_ReceiverResidual | 0.38154 | 0.87517 | 0.69464 | -0.02075 | 0.1589 | 0.8411 | 105040 |
| ele-fashion | 42 | B2_ContextResidual | 0.38451 | 0.87074 | 0.69015 | -0.01779 | 0.1639 | 0.8361 | 105040 |
| ele-fashion | 42 | B3_ShuffledContextResidual | 0.38502 | 0.87125 | 0.69050 | -0.01728 | 0.1642 | 0.8358 | 105040 |
| ele-fashion | 42 | B2_eval_shuffle | 0.38456 | 0.87057 | 0.69137 | -0.01774 | 0.1647 | 0.8353 | None |
| ele-fashion | 43 | B0_GlobalFrozen | 0.43689 | 0.87585 | 0.68724 | 0.00000 |  |  | None |
| ele-fashion | 43 | B1_ReceiverResidual | 0.38818 | 0.87210 | 0.70475 | -0.04872 | 0.1598 | 0.8402 | 105040 |
| ele-fashion | 43 | B2_ContextResidual | 0.38958 | 0.87193 | 0.70385 | -0.04731 | 0.1646 | 0.8354 | 105040 |
| ele-fashion | 43 | B3_ShuffledContextResidual | 0.38773 | 0.87278 | 0.70915 | -0.04916 | 0.1594 | 0.8406 | 105040 |
| ele-fashion | 43 | B2_eval_shuffle | 0.38969 | 0.87142 | 0.70358 | -0.04720 | 0.1642 | 0.8358 | None |
| ele-fashion | 44 | B0_GlobalFrozen | 0.42993 | 0.87619 | 0.70780 | 0.00000 |  |  | None |
| ele-fashion | 44 | B1_ReceiverResidual | 0.38188 | 0.87466 | 0.71097 | -0.04805 | 0.1448 | 0.8552 | 105040 |
| ele-fashion | 44 | B2_ContextResidual | 0.38137 | 0.87449 | 0.71112 | -0.04856 | 0.1427 | 0.8573 | 105040 |
| ele-fashion | 44 | B3_ShuffledContextResidual | 0.38147 | 0.87466 | 0.71277 | -0.04846 | 0.1444 | 0.8556 | 105040 |
| ele-fashion | 44 | B2_eval_shuffle | 0.38142 | 0.87415 | 0.71217 | -0.04851 | 0.1448 | 0.8552 | None |

## Pairwise contrasts

Negative CE favors the left variant; positive Accuracy/Macro-F1 favors the left variant.

| Dataset | Seed | Contrast | Δ CE | Δ Accuracy | Δ Macro-F1 |
|---|---:|---|---:|---:|---:|
| Movies | 42 | B1_minus_B0 | -0.07168 | -0.00100 | +0.01020 |
| Movies | 42 | B2_minus_B0 | -0.07089 | +0.00100 | +0.01301 |
| Movies | 42 | B2_minus_B1 | +0.00080 | +0.00200 | +0.00281 |
| Movies | 42 | B2_minus_B2_eval_shuffle | -0.00118 | +0.00200 | +0.00039 |
| Movies | 42 | B2_minus_B3 | +0.00326 | +0.00200 | +0.00186 |
| Movies | 43 | B1_minus_B0 | -0.01277 | -0.01503 | +0.02045 |
| Movies | 43 | B2_minus_B0 | -0.02265 | -0.00100 | +0.03251 |
| Movies | 43 | B2_minus_B1 | -0.00988 | +0.01403 | +0.01206 |
| Movies | 43 | B2_minus_B2_eval_shuffle | +0.00068 | +0.00000 | +0.00047 |
| Movies | 43 | B2_minus_B3 | -0.00209 | +0.00100 | +0.00379 |
| Movies | 44 | B1_minus_B0 | -0.02226 | -0.00601 | -0.00159 |
| Movies | 44 | B2_minus_B0 | -0.02350 | -0.00601 | -0.00127 |
| Movies | 44 | B2_minus_B1 | -0.00124 | +0.00000 | +0.00031 |
| Movies | 44 | B2_minus_B2_eval_shuffle | -0.00053 | +0.00000 | +0.00003 |
| Movies | 44 | B2_minus_B3 | -0.00081 | -0.00100 | -0.00092 |
| Grocery | 42 | B1_minus_B0 | -0.01105 | +0.00000 | +0.00207 |
| Grocery | 42 | B2_minus_B0 | -0.01067 | +0.00098 | +0.00284 |
| Grocery | 42 | B2_minus_B1 | +0.00038 | +0.00098 | +0.00077 |
| Grocery | 42 | B2_minus_B2_eval_shuffle | -0.00007 | +0.00000 | +0.00000 |
| Grocery | 42 | B2_minus_B3 | -0.00018 | +0.00000 | +0.00000 |
| Grocery | 43 | B1_minus_B0 | -0.00462 | -0.00196 | +0.00247 |
| Grocery | 43 | B2_minus_B0 | -0.00456 | -0.00196 | +0.00247 |
| Grocery | 43 | B2_minus_B1 | +0.00006 | +0.00000 | +0.00000 |
| Grocery | 43 | B2_minus_B2_eval_shuffle | +0.00003 | +0.00000 | +0.00000 |
| Grocery | 43 | B2_minus_B3 | +0.00006 | +0.00000 | +0.00000 |
| Grocery | 44 | B1_minus_B0 | +0.00366 | -0.00294 | -0.00449 |
| Grocery | 44 | B2_minus_B0 | +0.00441 | -0.00587 | -0.00690 |
| Grocery | 44 | B2_minus_B1 | +0.00075 | -0.00294 | -0.00240 |
| Grocery | 44 | B2_minus_B2_eval_shuffle | +0.00020 | +0.00098 | +0.00078 |
| Grocery | 44 | B2_minus_B3 | +0.00118 | +0.00098 | +0.00129 |
| ele-fashion | 42 | B1_minus_B0 | -0.02075 | +0.00153 | +0.01140 |
| ele-fashion | 42 | B2_minus_B0 | -0.01779 | -0.00290 | +0.00691 |
| ele-fashion | 42 | B2_minus_B1 | +0.00297 | -0.00443 | -0.00449 |
| ele-fashion | 42 | B2_minus_B2_eval_shuffle | -0.00005 | +0.00017 | -0.00121 |
| ele-fashion | 42 | B2_minus_B3 | -0.00051 | -0.00051 | -0.00034 |
| ele-fashion | 43 | B1_minus_B0 | -0.04872 | -0.00375 | +0.01751 |
| ele-fashion | 43 | B2_minus_B0 | -0.04731 | -0.00392 | +0.01661 |
| ele-fashion | 43 | B2_minus_B1 | +0.00140 | -0.00017 | -0.00090 |
| ele-fashion | 43 | B2_minus_B2_eval_shuffle | -0.00011 | +0.00051 | +0.00027 |
| ele-fashion | 43 | B2_minus_B3 | +0.00185 | -0.00085 | -0.00530 |
| ele-fashion | 44 | B1_minus_B0 | -0.04805 | -0.00153 | +0.00317 |
| ele-fashion | 44 | B2_minus_B0 | -0.04856 | -0.00171 | +0.00332 |
| ele-fashion | 44 | B2_minus_B1 | -0.00052 | -0.00017 | +0.00014 |
| ele-fashion | 44 | B2_minus_B2_eval_shuffle | -0.00005 | +0.00034 | -0.00106 |
| ele-fashion | 44 | B2_minus_B3 | -0.00010 | -0.00017 | -0.00165 |

Across the nine runs, B2−B1 averages ΔCE -0.00059, ΔAccuracy +0.00103, and ΔMacro-F1 +0.00092. B2−B3 averages ΔCE +0.00030; B2−eval-shuffle averages ΔCE -0.00012. B2's per-node ΔCE is positive for 31.1% and harmful for 68.9% of Audit nodes on average. Read alongside the per-run variation; the small differences do not establish a reliable context effect.

Taken together with H2-A, the residual screen also fits H2 Weak for this minimal adapter: B2 is very close to receiver-only and shuffled controls, with no clear real-context correspondence advantage. Adapters can reduce mean CE relative to B0 while helping fewer than half of nodes, so that global reduction alone is not evidence that relation context supplied it.

Per-modality gate mean/median, residual norm ratio, and per-node ΔCE positive/harmful fractions are retained in per-run JSON. These are mechanism diagnostics and no automatic significance gate is applied.
