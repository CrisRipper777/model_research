# Physical and semantic cross-modal diagnostics

Best-Validation-Accuracy checkpoints from historical H2 4-hop and current 3-hop runs were reloaded with the frozen compatible wrapper. The model was set to evaluation mode and returned its forward diagnostics. Train and Validation metrics index only their own rows; Test labels were masked before model construction and never indexed. All reported numbers are descriptive forward activations/routing summaries, not causal effects.

For `late_completion`, source weights average the node-wise selector probabilities. Semantic and same-node candidates can both be available, while `none` remains a third option. `same_semantic_weighted_rms` is the within-modality semantic graph contribution; early calibrated cross RMS is the other modality's physical-neighbor message after scale calibration. Residual ratios summarize the late contribution relative to its collaborative state.

| Dataset | Depth | Early cross RMS (target text) | Early cross RMS (target visual) | Text same-semantic RMS | Visual same-semantic RMS | Text semantic source weight | Visual semantic source weight | Text same-node source weight | Visual same-node source weight | Text late residual/state RMS | Visual late residual/state RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | H2_full | 0.6659 ± 0.1873 | 0.4709 ± 0.0587 | 0.0973 ± 0.0020 | 0.0934 ± 0.0032 | 0.2255 ± 0.2751 | 0.0461 ± 0.0302 | 0.2995 ± 0.2645 | 0.0884 ± 0.0810 | 0.0620 ± 0.0277 | 0.0122 ± 0.0033 |
| Movies | H2_physical_3hop | 0.5694 ± 0.1005 | 0.4435 ± 0.0354 | 0.0983 ± 0.0014 | 0.0938 ± 0.0042 | 0.2799 ± 0.3997 | 0.0663 ± 0.0638 | 0.3329 ± 0.3105 | 0.1883 ± 0.1122 | 0.0543 ± 0.0285 | 0.0189 ± 0.0018 |
| Grocery | H2_full | 0.4222 ± 0.0849 | 0.7204 ± 0.1461 | 0.0970 ± 0.0036 | 0.1112 ± 0.0115 | 0.5932 ± 0.1091 | 0.2748 ± 0.2257 | 0.3860 ± 0.0980 | 0.5389 ± 0.3193 | 0.0745 ± 0.0251 | 0.0644 ± 0.0208 |
| Grocery | H2_physical_3hop | 0.4689 ± 0.1203 | 0.6699 ± 0.1534 | 0.0974 ± 0.0045 | 0.1126 ± 0.0135 | 0.6116 ± 0.0979 | 0.2969 ± 0.2525 | 0.3261 ± 0.0559 | 0.5757 ± 0.3383 | 0.0705 ± 0.0268 | 0.0647 ± 0.0135 |
| ele-fashion | H2_full | 0.7415 ± 0.0058 | 1.1778 ± 0.0309 | 0.2075 ± 0.0258 | 0.1468 ± 0.0096 | 0.5909 ± 0.0291 | 0.9531 ± 0.0810 | 0.1802 ± 0.1484 | 0.0040 ± 0.0068 | 0.1228 ± 0.0314 | 0.1510 ± 0.0153 |
| ele-fashion | H2_physical_3hop | 0.7123 ± 0.0480 | 1.2055 ± 0.0779 | 0.2233 ± 0.0167 | 0.1429 ± 0.0168 | 0.6664 ± 0.0217 | 0.9894 ± 0.0157 | 0.1591 ± 0.1481 | 0.0011 ± 0.0009 | 0.1492 ± 0.0244 | 0.1686 ± 0.0192 |
| Toys | H2_full | 0.3352 ± 0.0132 | 0.4789 ± 0.0276 | 0.0871 ± 0.0004 | 0.0881 ± 0.0003 | 0.2009 ± 0.0752 | 0.2354 ± 0.1100 | 0.6250 ± 0.0416 | 0.6733 ± 0.0727 | 0.0524 ± 0.0025 | 0.0440 ± 0.0051 |
| Toys | H2_physical_3hop | 0.3452 ± 0.0195 | 0.4642 ± 0.0499 | 0.0878 ± 0.0006 | 0.0887 ± 0.0004 | 0.1897 ± 0.1253 | 0.1489 ± 0.0980 | 0.5999 ± 0.1565 | 0.7786 ± 0.1076 | 0.0501 ± 0.0069 | 0.0473 ± 0.0004 |
| Reddit-S | H2_full | 0.5470 ± 0.0476 | 0.3522 ± 0.0628 | 0.0759 ± 0.0010 | 0.0874 ± 0.0023 | 0.3061 ± 0.5024 | 0.0290 ± 0.0291 | 0.6509 ± 0.5488 | 0.7164 ± 0.2209 | 0.0544 ± 0.0201 | 0.0726 ± 0.0240 |
| Reddit-S | H2_physical_3hop | 0.4808 ± 0.0516 | 0.3402 ± 0.0852 | 0.0761 ± 0.0012 | 0.0870 ± 0.0013 | 0.6382 ± 0.2853 | 0.1253 ± 0.0617 | 0.2971 ± 0.2899 | 0.6158 ± 0.1438 | 0.0490 ± 0.0152 | 0.0666 ± 0.0236 |

Full scalar paths and per-seed checkpoint details are in `data/cross_modal_diagnostics.csv`, `data/cross_modal_diagnostics_summary.csv` and `data/cross_modal_diagnostics.json`. Seed-to-seed variation and dataset shifts should be read directly; activation scale alone does not establish utility or mechanism importance.
