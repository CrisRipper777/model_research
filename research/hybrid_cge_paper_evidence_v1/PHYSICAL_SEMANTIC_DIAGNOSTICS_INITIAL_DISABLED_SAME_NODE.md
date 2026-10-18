# Physical and semantic cross-modal diagnostics

Best-Validation-Accuracy checkpoints from historical H2 4-hop and current 3-hop runs were reloaded with the frozen compatible wrapper. The model was set to evaluation mode and returned its forward diagnostics. Train and Validation metrics index only their own rows; Test labels were masked before model construction and never indexed. All reported numbers are descriptive forward activations/routing summaries, not causal effects.

For `late_completion`, source weights average the node-wise selector probabilities. Semantic and same-node candidates can both be available, while `none` remains a third option. `same_semantic_weighted_rms` is the within-modality semantic graph contribution; early calibrated cross RMS is the other modality's physical-neighbor message after scale calibration. Residual ratios summarize the late contribution relative to its collaborative state.

| Dataset | Depth | Early cross RMS (target text) | Early cross RMS (target visual) | Text same-semantic RMS | Visual same-semantic RMS | Text semantic source weight | Visual semantic source weight | Text same-node source weight | Visual same-node source weight | Text late residual/state RMS | Visual late residual/state RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | H2_full | 0.6659 ± 0.1873 | 0.4709 ± 0.0587 | 0.0973 ± 0.0020 | 0.0934 ± 0.0032 | 0.3761 ± 0.2616 | 0.0917 ± 0.0252 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0405 ± 0.0278 | 0.0090 ± 0.0027 |
| Movies | H2_physical_3hop | 0.5902 ± 0.0818 | 0.4823 ± 0.0074 | 0.0979 ± 0.0008 | 0.0920 ± 0.0021 | 0.5330 ± 0.4038 | 0.1562 ± 0.1285 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0368 ± 0.0325 | 0.0106 ± 0.0022 |
| Grocery | H2_full | 0.4222 ± 0.0849 | 0.7204 ± 0.1461 | 0.0970 ± 0.0036 | 0.1112 ± 0.0115 | 0.9226 ± 0.0407 | 0.5475 ± 0.2252 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0520 ± 0.0189 | 0.0288 ± 0.0057 |
| Grocery | H2_physical_3hop | 0.5606 ± 0.1338 | 0.7372 ± 0.0663 | 0.0964 ± 0.0029 | 0.1130 ± 0.0122 | 0.9109 ± 0.0774 | 0.7603 ± 0.1908 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0671 ± 0.0232 | 0.0427 ± 0.0087 |
| ele-fashion | H2_full | 0.7415 ± 0.0058 | 1.1778 ± 0.0309 | 0.2075 ± 0.0258 | 0.1468 ± 0.0096 | 0.6479 ± 0.1000 | 0.9679 ± 0.0550 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1053 ± 0.0224 | 0.1514 ± 0.0148 |
| ele-fashion | H2_physical_3hop | 0.8637 ± 0.0494 | 1.3196 ± 0.0437 | 0.1924 ± 0.0269 | 0.1352 ± 0.0275 | 0.6318 ± 0.0792 | 0.9935 ± 0.0102 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1169 ± 0.0315 | 0.1596 ± 0.0253 |
| Toys | H2_full | 0.3352 ± 0.0132 | 0.4789 ± 0.0276 | 0.0871 ± 0.0004 | 0.0881 ± 0.0003 | 0.5484 ± 0.0787 | 0.7098 ± 0.1566 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0188 ± 0.0020 | 0.0216 ± 0.0055 |
| Toys | H2_physical_3hop | 0.3766 ± 0.0194 | 0.5369 ± 0.0399 | 0.0883 ± 0.0014 | 0.0893 ± 0.0017 | 0.7952 ± 0.1244 | 0.9353 ± 0.0885 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0321 ± 0.0065 | 0.0377 ± 0.0037 |
| Reddit-S | H2_full | 0.5470 ± 0.0476 | 0.3522 ± 0.0628 | 0.0759 ± 0.0010 | 0.0874 ± 0.0023 | 0.7924 ± 0.0962 | 0.3901 ± 0.0826 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0301 ± 0.0049 | 0.0205 ± 0.0034 |
| Reddit-S | H2_physical_3hop | 0.4759 ± 0.0454 | 0.2911 ± 0.0252 | 0.0769 ± 0.0004 | 0.0863 ± 0.0005 | 0.9217 ± 0.0588 | 0.7322 ± 0.0636 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0341 ± 0.0009 | 0.0320 ± 0.0045 |

Full scalar paths and per-seed checkpoint details are in `data/cross_modal_diagnostics.csv`, `data/cross_modal_diagnostics_summary.csv` and `data/cross_modal_diagnostics.json`. Seed-to-seed variation and dataset shifts should be read directly; activation scale alone does not establish utility or mechanism importance.
