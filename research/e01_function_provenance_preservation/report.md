# E0.1 — Function-Provenance Preservation Screen

## Executive decision

**Primary frozen label: `PROVENANCE_ACTIVE_BUT_NO_MODEL_GAIN`.** This screen isolates whether preserving the existing E0 Smooth/Relational/Cross-Modal function identity after edge aggregation adds validation value when all four variants have the same E0 modules, router, function bank, composer architecture, initialization, and parameter count.

**Premature function mixing was a major E0 bottleneck: PARTIALLY.** KeepEdge−PremixEdgeControl, the same-capacity provenance contrast, is Accuracy -0.003 ± 0.135 pp; Macro-F1 -0.070 ± 0.691 pp; CE -0.0110 ± 0.0199. KeepEdge has checkpoint reliance: provenance-collapse changes Accuracy by -0.283 pp on average (SD 0.162) and the five provenance permutations by -0.329 pp (SD 0.247); however, reliance did not translate into an Accuracy or Macro-F1 gain over the matched Premix control. KeepEdge's CE is lower in aggregate, so evidence is partial rather than a broad performance win.

**Preserving function provenance enables useful edge-specific routing: NO.** KeepEdge−KeepTarget is Accuracy 0.039 ± 0.176 pp; Macro-F1 0.459 ± 0.836 pp; CE 0.0094 ± 0.0163; this must be read together with KeepEdge−PremixEdgeControl and the edge-routing interventions below.

The next screened bottleneck indicated by this evidence is **function-context degeneration**. Four of six dataset×modality groups have low node-level R/X mass; ele-fashion retains both alternatives. The evidence points to dataset-specific function-context degeneration, while the nonzero collapse/permutation effects show that contexts surviving in trained KeepEdge are used. This E0.1 result is an executor attribution screen only. P1.3's joint readout motivated the hypothesis by retaining multiple operator contexts until one shared predictor; it did not establish that this E0.1 composer must preserve provenance. A success here supports only the tested E0 function contexts and this composer, not the necessity of P1.3's specific representation.

## Provenance, implementation, and protocol

- Source: E0 branch `exp/e0_structured_relation_function_executor` at `4905962df8c548c457b0d663ac7f473279f9f0e9`; this branch was created from that commit without merging `main`.
- Four variants: `premix_edge_control` (E0 EdgeMix router, composer input `[Cmix/3,Cmix/3,Cmix/3]`), `keep_static` (E0 StaticMix router, `[CS,CR,CX]`), `keep_target` (E0 TargetMix router, `[CS,CR,CX]`), `keep_edge` (E0 EdgeMix router, `[CS,CR,CX]`). Every variant adds the unchanged base `Cmix` to its composer output before the original modality residual LayerNorm and original four-way Text/Visual fusion.
- PremixEdgeControl has the **same composer parameter count and hidden width** as every Keep variant. Its repeated active input blocks permit the first Linear layer to represent any 128D-to-128D map through the sum of its three block weights. It is not a one-third-capacity baseline; it differs by lack of function identity in its input.
- All E0 common modules are constructed before the two modality-specific composers. E0 function definitions, Smooth-RMS stop-gradient calibration, softmax router, router prior, relation state, physical support, LOO context, degree denominator, residual norms, fusion, and one-hop protocol remain fixed.
- Formal NC campaign: 36/36 unique runs complete, 1 failed attempt and 1 retry; datasets Movies/Grocery/ele-fashion, seeds 42/43/44, four variants; AdamW 1e−3, weight decay 1e−4, max 300 epochs, patience 30, minimum epoch 30, gradient clipping 1.0, best validation accuracy. The sole failed attempt was a too-tight GPU target-mean numerical assertion at Grocery/42; after widening only that floating-point audit tolerance, the same run completed. No model/protocol change or other failure occurred.
- Device: cuda:0; summed training time across the 36 successful runs 1467.9s (the failed attempt duration was not captured), intervention time 133.4s, peak allocated training GPU memory 17.83 GiB.
- Only train/validation labels were made available. No test evaluation, test-index/test-label access, link prediction, or test-based checkpoint choice.
- The historical UNI and E0 EdgeMix rows are read from committed E0/M0 performance CSVs; no historical model was retrained. SEM and E0 Static/Target are not part of the required E0.1 main table.
- Parameter summary has exactly 36 unique dataset×seed×variant rows; the duplicated E0 parameter-summary rows were not propagated.

## Validation performance

Dataset cells are mean ± population SD over seeds 42–44. Accuracy and Macro-F1 are percentages; CE is native. The aggregate is an equal-weight descriptive summary of nine dataset×seed runs, not a pooled-node score. No significance tests were run.

### Movies

| Method | Accuracy (%) | Macro-F1 (%) | CE |
|---|---:|---:|---:|
| UNI | 55.419 ± 0.338 | 45.867 ± 0.802 | 1.4193 ± 0.0273 |
| Historical E0 EdgeMix | 55.139 ± 0.387 | 45.010 ± 0.407 | 1.3995 ± 0.0252 |
| PremixEdgeControl | 55.369 ± 0.185 | 44.791 ± 0.274 | 1.3886 ± 0.0121 |
| KeepStatic | 55.199 ± 0.157 | 45.193 ± 0.647 | 1.3909 ± 0.0077 |
| KeepTarget | 55.419 ± 0.163 | 44.060 ± 0.490 | 1.3794 ± 0.0063 |
| KeepEdge | 55.389 ± 0.331 | 44.942 ± 0.440 | 1.3847 ± 0.0098 |

### Grocery

| Method | Accuracy (%) | Macro-F1 (%) | CE |
|---|---:|---:|---:|
| UNI | 82.255 ± 0.231 | 73.405 ± 1.072 | 0.7085 ± 0.0293 |
| Historical E0 EdgeMix | 81.942 ± 0.193 | 73.496 ± 0.526 | 0.7033 ± 0.0329 |
| PremixEdgeControl | 82.186 ± 0.444 | 73.804 ± 0.483 | 0.7178 ± 0.0505 |
| KeepStatic | 81.991 ± 0.167 | 73.419 ± 0.437 | 0.7145 ± 0.0051 |
| KeepTarget | 81.972 ± 0.258 | 72.965 ± 1.264 | 0.6933 ± 0.0130 |
| KeepEdge | 82.089 ± 0.360 | 73.293 ± 0.392 | 0.7024 ± 0.0209 |

### ele-fashion

| Method | Accuracy (%) | Macro-F1 (%) | CE |
|---|---:|---:|---:|
| UNI | 87.413 ± 0.084 | 68.660 ± 0.798 | 0.4162 ± 0.0032 |
| Historical E0 EdgeMix | 87.331 ± 0.046 | 68.496 ± 0.145 | 0.4208 ± 0.0081 |
| PremixEdgeControl | 87.375 ± 0.065 | 68.616 ± 0.378 | 0.4441 ± 0.0057 |
| KeepStatic | 87.501 ± 0.087 | 69.372 ± 0.360 | 0.4141 ± 0.0034 |
| KeepTarget | 87.413 ± 0.046 | 68.600 ± 0.694 | 0.4168 ± 0.0109 |
| KeepEdge | 87.443 ± 0.092 | 68.766 ± 0.256 | 0.4305 ± 0.0115 |

### Equal-weight aggregate across nine runs

| Method | Accuracy (%) | Macro-F1 (%) | CE |
|---|---:|---:|---:|
| UNI | 75.029 ± 14.027 | 62.644 ± 12.054 | 0.8480 ± 0.4219 |
| Historical E0 EdgeMix | 74.804 ± 14.080 | 62.334 ± 12.425 | 0.8412 ± 0.4120 |
| PremixEdgeControl | 74.977 ± 14.029 | 62.404 ± 12.639 | 0.8502 ± 0.3979 |
| KeepStatic | 74.897 ± 14.110 | 62.661 ± 12.472 | 0.8398 ± 0.4085 |
| KeepTarget | 74.934 ± 13.978 | 61.875 ± 12.753 | 0.8298 ± 0.4048 |
| KeepEdge | 74.974 ± 14.023 | 62.334 ± 12.442 | 0.8392 ± 0.4017 |

## Paired same-seed comparisons

Positive Accuracy/Macro-F1 favors the left method; positive CE is unfavorable. Each dataset uses three paired seeds; `ALL` contains nine dataset×seed pairs and is descriptive.

| Dataset | Comparison | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---|---:|---:|---:|
| Movies | KeepEdge-PremixEdgeControl | 0.020 ± 0.150 | 0.150 ± 0.169 | -0.0039 ± 0.0057 |
| Movies | KeepTarget-KeepStatic | 0.220 ± 0.181 | -1.133 ± 0.777 | -0.0114 ± 0.0015 |
| Movies | KeepEdge-KeepTarget | -0.030 ± 0.255 | 0.882 ± 0.405 | 0.0053 ± 0.0066 |
| Movies | KeepEdge-KeepStatic | 0.190 ± 0.190 | -0.251 ± 0.373 | -0.0062 ± 0.0072 |
| Movies | PremixEdgeControl-HistoricalE0EdgeMix | 0.230 ± 0.220 | -0.219 ± 0.147 | -0.0109 ± 0.0133 |
| Movies | KeepEdge-UNI | -0.030 ± 0.625 | -0.926 ± 0.388 | -0.0346 ± 0.0327 |
| Grocery | KeepEdge-PremixEdgeControl | -0.098 ± 0.118 | -0.511 ± 0.845 | -0.0154 ± 0.0296 |
| Grocery | KeepTarget-KeepStatic | -0.020 ± 0.108 | -0.453 ± 0.866 | -0.0212 ± 0.0115 |
| Grocery | KeepEdge-KeepTarget | 0.117 ± 0.120 | 0.328 ± 0.919 | 0.0091 ± 0.0228 |
| Grocery | KeepEdge-KeepStatic | 0.098 ± 0.193 | -0.125 ± 0.310 | -0.0122 ± 0.0252 |
| Grocery | PremixEdgeControl-HistoricalE0EdgeMix | 0.244 ± 0.276 | 0.308 ± 0.773 | 0.0145 ± 0.0182 |
| Grocery | KeepEdge-UNI | -0.166 ± 0.144 | -0.112 ± 1.310 | -0.0061 ± 0.0166 |
| ele-fashion | KeepEdge-PremixEdgeControl | 0.068 ± 0.059 | 0.150 ± 0.631 | -0.0136 ± 0.0143 |
| ele-fashion | KeepTarget-KeepStatic | -0.089 ± 0.054 | -0.772 ± 0.340 | 0.0027 ± 0.0084 |
| ele-fashion | KeepEdge-KeepTarget | 0.031 ± 0.052 | 0.166 ± 0.898 | 0.0137 ± 0.0141 |
| ele-fashion | KeepEdge-KeepStatic | -0.058 ± 0.017 | -0.606 ± 0.559 | 0.0164 ± 0.0132 |
| ele-fashion | PremixEdgeControl-HistoricalE0EdgeMix | 0.044 ± 0.021 | 0.120 ± 0.329 | 0.0233 ± 0.0137 |
| ele-fashion | KeepEdge-UNI | 0.031 ± 0.115 | 0.107 ± 1.053 | 0.0143 ± 0.0141 |
| ALL | KeepEdge-PremixEdgeControl | -0.003 ± 0.135 | -0.070 ± 0.691 | -0.0110 ± 0.0199 |
| ALL | KeepTarget-KeepStatic | 0.037 ± 0.182 | -0.786 ± 0.753 | -0.0100 ± 0.0128 |
| ALL | KeepEdge-KeepTarget | 0.039 ± 0.176 | 0.459 ± 0.836 | 0.0094 ± 0.0163 |
| ALL | KeepEdge-KeepStatic | 0.077 ± 0.187 | -0.327 ± 0.473 | -0.0006 ± 0.0209 |
| ALL | PremixEdgeControl-HistoricalE0EdgeMix | 0.173 ± 0.223 | 0.070 ± 0.539 | 0.0090 ± 0.0210 |
| ALL | KeepEdge-UNI | -0.055 ± 0.385 | -0.310 ± 1.091 | -0.0088 ± 0.0303 |

The decisive provenance comparison is KeepEdge−PremixEdgeControl: Accuracy -0.003 ± 0.135 pp; Macro-F1 -0.070 ± 0.691 pp; CE -0.0110 ± 0.0199. Composer capacity relative to historical E0 EdgeMix is PremixEdgeControl−E0 EdgeMix: Accuracy 0.173 ± 0.223 pp; Macro-F1 0.070 ± 0.539 pp; CE 0.0090 ± 0.0210. Target-level granularity is KeepTarget−KeepStatic: Accuracy 0.037 ± 0.182 pp; Macro-F1 -0.786 ± 0.753 pp; CE -0.0100 ± 0.0128; edge-level granularity is KeepEdge−KeepTarget: Accuracy 0.039 ± 0.176 pp; Macro-F1 0.459 ± 0.836 pp; CE 0.0094 ± 0.0163. KeepEdge−KeepStatic is Accuracy 0.077 ± 0.187 pp; Macro-F1 -0.327 ± 0.473 pp; CE -0.0006 ± 0.0209; KeepEdge−UNI is Accuracy -0.055 ± 0.385 pp; Macro-F1 -0.310 ± 1.091 pp; CE -0.0088 ± 0.0303.

## Aggregated function-context diagnostics

The context statistics below are computed on validation nodes after full-neighborhood channel aggregation. `Cmix` is defined as `CS + CR + CX`; the measured maximum identity error across the 36 runs is 0, below 1e−6. Node RMS distributions include mean, population SD and q10/q25/median/q75/q90 in `data/node_channel_norms.csv`.

### KeepEdge node-channel RMS

- KeepEdge Movies/text: smooth: node RMS median 0.875, node RMS mean 0.8532; relational: node RMS median 0.007683, node RMS mean 0.02945; cross_modal: node RMS median 0.007609, node RMS mean 0.02866; mix: node RMS median 0.8742, node RMS mean 0.8715.
- KeepEdge Movies/visual: smooth: node RMS median 0.8955, node RMS mean 0.898; relational: node RMS median 0.001287, node RMS mean 0.005699; cross_modal: node RMS median 0.001367, node RMS mean 0.00557; mix: node RMS median 0.8962, node RMS mean 0.901.
- KeepEdge Grocery/text: smooth: node RMS median 0.916, node RMS mean 0.8813; relational: node RMS median 0.004057, node RMS mean 0.04152; cross_modal: node RMS median 0.003705, node RMS mean 0.02792; mix: node RMS median 0.9164, node RMS mean 0.9038.
- KeepEdge Grocery/visual: smooth: node RMS median 0.8856, node RMS mean 0.87; relational: node RMS median 0.004682, node RMS mean 0.03801; cross_modal: node RMS median 0.003993, node RMS mean 0.02513; mix: node RMS median 0.8887, node RMS mean 0.8985.
- KeepEdge ele-fashion/text: smooth: node RMS median 0.4612, node RMS mean 0.5111; relational: node RMS median 0.2937, node RMS mean 0.3046; cross_modal: node RMS median 0.2664, node RMS mean 0.2606; mix: node RMS median 0.795, node RMS mean 0.8207.
- KeepEdge ele-fashion/visual: smooth: node RMS median 0.7067, node RMS mean 0.6131; relational: node RMS median 0.1177, node RMS mean 0.1885; cross_modal: node RMS median 0.134, node RMS mean 0.1828; mix: node RMS median 0.8083, node RMS mean 0.8066.

### Relative context mass by variant

Each value is the across-seed mean of a per-run validation-node median; full per-run distributions are in `data/channel_mass_summary.csv`.

- Movies/text/PremixEdgeControl median relative mass: S=0.975, R=0.013, C=0.013.
- Movies/text/KeepStatic median relative mass: S=0.999, R=0.001, C=0.001.
- Movies/text/KeepTarget median relative mass: S=0.992, R=0.004, C=0.004.
- Movies/text/KeepEdge median relative mass: S=0.982, R=0.009, C=0.009.
- Movies/visual/PremixEdgeControl median relative mass: S=0.997, R=0.001, C=0.001.
- Movies/visual/KeepStatic median relative mass: S=0.999, R=0.001, C=0.001.
- Movies/visual/KeepTarget median relative mass: S=0.997, R=0.002, C=0.002.
- Movies/visual/KeepEdge median relative mass: S=0.997, R=0.001, C=0.001.
- Grocery/text/PremixEdgeControl median relative mass: S=0.950, R=0.028, C=0.021.
- Grocery/text/KeepStatic median relative mass: S=0.997, R=0.001, C=0.001.
- Grocery/text/KeepTarget median relative mass: S=0.995, R=0.002, C=0.002.
- Grocery/text/KeepEdge median relative mass: S=0.992, R=0.004, C=0.004.
- Grocery/visual/PremixEdgeControl median relative mass: S=0.953, R=0.025, C=0.022.
- Grocery/visual/KeepStatic median relative mass: S=0.997, R=0.001, C=0.001.
- Grocery/visual/KeepTarget median relative mass: S=0.992, R=0.004, C=0.004.
- Grocery/visual/KeepEdge median relative mass: S=0.990, R=0.005, C=0.004.
- ele-fashion/text/PremixEdgeControl median relative mass: S=0.369, R=0.324, C=0.262.
- ele-fashion/text/KeepStatic median relative mass: S=0.109, R=0.572, C=0.319.
- ele-fashion/text/KeepTarget median relative mass: S=0.494, R=0.302, C=0.191.
- ele-fashion/text/KeepEdge median relative mass: S=0.440, R=0.280, C=0.252.
- ele-fashion/visual/PremixEdgeControl median relative mass: S=0.645, R=0.162, C=0.165.
- ele-fashion/visual/KeepStatic median relative mass: S=0.216, R=0.372, C=0.412.
- ele-fashion/visual/KeepTarget median relative mass: S=0.785, R=0.116, C=0.096.
- ele-fashion/visual/KeepEdge median relative mass: S=0.731, R=0.122, C=0.139.

KeepEdge alternative channel mass by modality group: Movies/text R+X median mass 0.009; Movies/visual R+X median mass 0.001; Grocery/text R+X median mass 0.004; Grocery/visual R+X median mass 0.005; ele-fashion/text R+X median mass 0.266; ele-fashion/visual R+X median mass 0.131.

### KeepEdge node-context cosine distinctness

Cosines include only validation nodes where both channel vector norms exceed `eps`. Each line reports the mean of three run medians and the mean of run q10/q90 values; valid counts are recorded per run.

- Movies/text: smooth_relational: median 0.061, q10–q90 -0.032–0.188, valid nodes/run 3334; smooth_cross_modal: median -0.024, q10–q90 -0.103–0.088, valid nodes/run 3334; relational_cross_modal: median 0.271, q10–q90 0.216–0.326, valid nodes/run 3334.
- Movies/visual: smooth_relational: median 0.126, q10–q90 0.046–0.215, valid nodes/run 3334; smooth_cross_modal: median 0.093, q10–q90 -0.009–0.188, valid nodes/run 3334; relational_cross_modal: median 0.322, q10–q90 0.290–0.348, valid nodes/run 3334.
- Grocery/text: smooth_relational: median 0.018, q10–q90 -0.090–0.139, valid nodes/run 3415; smooth_cross_modal: median 0.019, q10–q90 -0.083–0.118, valid nodes/run 3415; relational_cross_modal: median 0.230, q10–q90 0.133–0.318, valid nodes/run 3415.
- Grocery/visual: smooth_relational: median 0.231, q10–q90 0.138–0.328, valid nodes/run 3415; smooth_cross_modal: median 0.173, q10–q90 0.068–0.262, valid nodes/run 3415; relational_cross_modal: median 0.319, q10–q90 0.239–0.398, valid nodes/run 3415.
- ele-fashion/text: smooth_relational: median 0.030, q10–q90 -0.126–0.203, valid nodes/run 9776; smooth_cross_modal: median -0.005, q10–q90 -0.173–0.172, valid nodes/run 9776; relational_cross_modal: median 0.090, q10–q90 -0.061–0.251, valid nodes/run 9776.
- ele-fashion/visual: smooth_relational: median 0.044, q10–q90 -0.093–0.166, valid nodes/run 9776; smooth_cross_modal: median 0.019, q10–q90 -0.123–0.158, valid nodes/run 9776; relational_cross_modal: median 0.170, q10–q90 0.014–0.312, valid nodes/run 9776.

## Composer usage

`composer/base` is the node-wise RMS ratio `RMS(Ccomp)/(RMS(Cmix)+eps)`. Values summarize the per-run node distribution across three seeds; cosine is between composer correction and Cmix. Composer nonzero magnitude is evidence of branch usage only, not proof of gain.

- Movies/text: PremixEdgeControl ratio median=0.119, IQR=0.048, q90=0.185; cos median=0.255; KeepStatic ratio median=0.195, IQR=0.102, q90=0.311; cos median=0.185; KeepTarget ratio median=0.182, IQR=0.094, q90=0.282; cos median=0.203; KeepEdge ratio median=0.184, IQR=0.092, q90=0.281; cos median=0.195.
- Movies/visual: PremixEdgeControl ratio median=0.168, IQR=0.059, q90=0.227; cos median=0.151; KeepStatic ratio median=0.167, IQR=0.062, q90=0.249; cos median=0.206; KeepTarget ratio median=0.169, IQR=0.069, q90=0.268; cos median=0.211; KeepEdge ratio median=0.177, IQR=0.070, q90=0.271; cos median=0.197.
- Grocery/text: PremixEdgeControl ratio median=0.178, IQR=0.107, q90=0.301; cos median=0.242; KeepStatic ratio median=0.229, IQR=0.100, q90=0.330; cos median=0.139; KeepTarget ratio median=0.214, IQR=0.094, q90=0.312; cos median=0.198; KeepEdge ratio median=0.234, IQR=0.115, q90=0.348; cos median=0.202.
- Grocery/visual: PremixEdgeControl ratio median=0.241, IQR=0.110, q90=0.333; cos median=0.279; KeepStatic ratio median=0.236, IQR=0.072, q90=0.310; cos median=0.252; KeepTarget ratio median=0.247, IQR=0.078, q90=0.322; cos median=0.290; KeepEdge ratio median=0.248, IQR=0.075, q90=0.324; cos median=0.301.
- ele-fashion/text: PremixEdgeControl ratio median=0.252, IQR=0.197, q90=0.605; cos median=0.385; KeepStatic ratio median=0.235, IQR=0.129, q90=0.383; cos median=0.310; KeepTarget ratio median=0.261, IQR=0.243, q90=0.653; cos median=0.292; KeepEdge ratio median=0.284, IQR=0.206, q90=0.674; cos median=0.309.
- ele-fashion/visual: PremixEdgeControl ratio median=0.158, IQR=0.086, q90=0.262; cos median=0.171; KeepStatic ratio median=0.176, IQR=0.068, q90=0.243; cos median=0.273; KeepTarget ratio median=0.237, IQR=0.102, q90=0.331; cos median=0.082; KeepEdge ratio median=0.249, IQR=0.103, q90=0.348; cos median=0.088.

## Frozen E0 calibration-scale audit

The selected checkpoints were reloaded for a forward-only audit on non-self edges whose destination is a validation node. These diagnostics record E0's existing Smooth-reference calibration; no calibration change or clamp was applied. All 36 checkpoints and all 504 modality×diagnostic rows were finite. Maximum validation-edge q99 across Relational/Cross-Modal calibration scales was 14.98.

| Dataset | Modality | Diagnostic | Mean run q99 ± SD | Maximum run q99 |
|---|---|---|---:|---:|
| Movies | text | relational_calibration_scale | 2.463 ± 0.1952 | 2.732 |
| Movies | text | cross_modal_calibration_scale | 1.998 ± 0.3371 | 2.436 |
| Movies | text | calibrated_relational_rms_ratio | 1 ± 0 | 1 |
| Movies | text | calibrated_cross_modal_rms_ratio | 1 ± 0 | 1 |
| Movies | visual | relational_calibration_scale | 1.275 ± 0.2273 | 1.592 |
| Movies | visual | cross_modal_calibration_scale | 1.627 ± 0.2633 | 1.857 |
| Movies | visual | calibrated_relational_rms_ratio | 1 ± 0 | 1 |
| Movies | visual | calibrated_cross_modal_rms_ratio | 1 ± 0 | 1 |
| Grocery | text | relational_calibration_scale | 5.413 ± 2.679 | 8.571 |
| Grocery | text | cross_modal_calibration_scale | 3.652 ± 1.441 | 5.676 |
| Grocery | text | calibrated_relational_rms_ratio | 1 ± 5.62e-08 | 1 |
| Grocery | text | calibrated_cross_modal_rms_ratio | 1 ± 5.62e-08 | 1 |
| Grocery | visual | relational_calibration_scale | 1.723 ± 0.5264 | 2.131 |
| Grocery | visual | cross_modal_calibration_scale | 2.907 ± 1.618 | 5.193 |
| Grocery | visual | calibrated_relational_rms_ratio | 1 ± 0 | 1 |
| Grocery | visual | calibrated_cross_modal_rms_ratio | 1 ± 0 | 1 |
| ele-fashion | text | relational_calibration_scale | 12.48 ± 0.2414 | 12.79 |
| ele-fashion | text | cross_modal_calibration_scale | 11.05 ± 0.9344 | 11.94 |
| ele-fashion | text | calibrated_relational_rms_ratio | 1 ± 0 | 1 |
| ele-fashion | text | calibrated_cross_modal_rms_ratio | 1 ± 5.62e-08 | 1 |
| ele-fashion | visual | relational_calibration_scale | 6.409 ± 1.143 | 7.339 |
| ele-fashion | visual | cross_modal_calibration_scale | 8.848 ± 1.212 | 10.14 |
| ele-fashion | visual | calibrated_relational_rms_ratio | 1 ± 5.62e-08 | 1 |
| ele-fashion | visual | calibrated_cross_modal_rms_ratio | 1 ± 5.62e-08 | 1 |

All seven distributions per modality, including Smooth reference RMS, raw Relational/Cross-Modal RMS, calibrated RMS ratios, and calibration scales, are in `data/expert_scale_diagnostics.csv`. No NaN/Inf occurred, so the frozen calibration was retained without clamping.

## KeepEdge fixed-checkpoint interventions

All rows below are validation-only changes to a trained KeepEdge checkpoint. Collapse replaces `[CS,CR,CX]` by three copies of `Cmix/3`; five permutations keep `Cmix` fixed while permuting channel identity; composer-off uses `delta=Cmix`; routing interventions recompute channel contexts before the composer. Accuracy/Macro-F1 are percentage-point deltas. The report gives both pooled record mean ± SD (including repeats) and mean ± SD after averaging repeats within each run; adverse direction counts are across run-level means (Accuracy/F1 lower or CE higher). These are reliance diagnostics, not standalone causal proof.

- `provenance_collapse` (n=9 evaluations over 9 checkpoints): pooled record mean±SD acc -0.283±0.162 pp, F1 -0.684±0.404 pp, CE 0.0004±0.0035; record ranges acc [-0.540,0.000] pp, F1 [-1.552,-0.077] pp, CE [-0.0040,0.0077]; Movies: acc -0.440±0.075 pp (3/3 adverse), F1 -1.115±0.332 pp (3/3 adverse), CE 0.0046±0.0023 (3/3 adverse); Grocery: acc -0.224±0.179 pp (2/3 adverse), F1 -0.362±0.228 pp (3/3 adverse), CE 0.0000±0.0010 (1/3 adverse); ele-fashion: acc -0.184±0.055 pp (3/3 adverse), F1 -0.575±0.158 pp (3/3 adverse), CE -0.0033±0.0008 (0/3 adverse).
- `provenance_permutation` (n=45 evaluations over 9 checkpoints): pooled record mean±SD acc -0.329±0.247 pp, F1 -0.848±0.900 pp, CE 0.0007±0.0051; record ranges acc [-1.110,0.000] pp, F1 [-3.126,0.037] pp, CE [-0.0096,0.0126]; Movies: acc -0.460±0.169 pp (3/3 adverse), F1 -1.471±0.837 pp (3/3 adverse), CE 0.0046±0.0029 (3/3 adverse); Grocery: acc -0.295±0.076 pp (3/3 adverse), F1 -0.552±0.389 pp (3/3 adverse), CE 0.0022±0.0016 (3/3 adverse); ele-fashion: acc -0.232±0.105 pp (3/3 adverse), F1 -0.519±0.264 pp (3/3 adverse), CE -0.0047±0.0020 (0/3 adverse).
- `composer_off` (n=9 evaluations over 9 checkpoints): pooled record mean±SD acc -0.472±0.250 pp, F1 -1.076±0.622 pp, CE 0.0015±0.0065; record ranges acc [-1.020,-0.088] pp, F1 [-2.420,-0.477] pp, CE [-0.0086,0.0099]; Movies: acc -0.630±0.282 pp (3/3 adverse), F1 -1.294±0.805 pp (3/3 adverse), CE 0.0087±0.0016 (3/3 adverse); Grocery: acc -0.371±0.227 pp (3/3 adverse), F1 -0.595±0.120 pp (3/3 adverse), CE 0.0020±0.0022 (2/3 adverse); ele-fashion: acc -0.416±0.134 pp (3/3 adverse), F1 -1.339±0.389 pp (3/3 adverse), CE -0.0063±0.0021 (0/3 adverse).
- `pi_shuffle_within_target` (n=45 evaluations over 9 checkpoints): pooled record mean±SD acc -0.078±0.118 pp, F1 -0.125±0.246 pp, CE 0.0016±0.0027; record ranges acc [-0.498,0.059] pp, F1 [-1.057,0.135] pp, CE [-0.0023,0.0106]; Movies: acc -0.002±0.003 pp (1/3 adverse), F1 -0.007±0.010 pp (1/3 adverse), CE -0.0003±0.0004 (0/3 adverse); Grocery: acc -0.125±0.160 pp (2/3 adverse), F1 -0.257±0.298 pp (2/3 adverse), CE 0.0033±0.0033 (3/3 adverse); ele-fashion: acc -0.108±0.033 pp (3/3 adverse), F1 -0.111±0.079 pp (3/3 adverse), CE 0.0019±0.0012 (3/3 adverse).
- `pi_target_mean` (n=9 evaluations over 9 checkpoints): pooled record mean±SD acc -0.051±0.093 pp, F1 -0.081±0.196 pp, CE 0.0006±0.0017; record ranges acc [-0.293,0.030] pp, F1 [-0.548,0.200] pp, CE [-0.0013,0.0050]; Movies: acc 0.010±0.014 pp (0/3 adverse), F1 0.013±0.018 pp (0/3 adverse), CE -0.0004±0.0006 (0/3 adverse); Grocery: acc -0.107±0.132 pp (2/3 adverse), F1 -0.233±0.231 pp (2/3 adverse), CE 0.0021±0.0021 (3/3 adverse); ele-fashion: acc -0.055±0.041 pp (3/3 adverse), F1 -0.023±0.160 pp (2/3 adverse), CE 0.0003±0.0009 (2/3 adverse).
- `pi_global_mean` (n=9 evaluations over 9 checkpoints): pooled record mean±SD acc -0.734±0.643 pp, F1 -1.408±1.461 pp, CE 0.0137±0.0120; record ranges acc [-2.035,0.000] pp, F1 [-4.018,0.000] pp, CE [0.0000,0.0331]; Movies: acc -0.300±0.424 pp (1/3 adverse), F1 -0.206±0.291 pp (1/3 adverse), CE 0.0025±0.0036 (3/3 adverse); Grocery: acc -0.498±0.376 pp (2/3 adverse), F1 -0.682±0.551 pp (2/3 adverse), CE 0.0119±0.0093 (3/3 adverse); ele-fashion: acc -1.405±0.475 pp (3/3 adverse), F1 -3.336±0.576 pp (3/3 adverse), CE 0.0266±0.0061 (3/3 adverse).
- `modality_tied_pi` (n=9 evaluations over 9 checkpoints): pooled record mean±SD acc -0.132±0.210 pp, F1 -0.185±0.234 pp, CE 0.0024±0.0030; record ranges acc [-0.660,0.051] pp, F1 [-0.596,0.038] pp, CE [-0.0000,0.0095]; Movies: acc -0.220±0.311 pp (1/3 adverse), F1 -0.155±0.220 pp (1/3 adverse), CE 0.0011±0.0015 (2/3 adverse); Grocery: acc -0.049±0.050 pp (2/3 adverse), F1 -0.063±0.062 pp (2/3 adverse), CE 0.0003±0.0005 (2/3 adverse); ele-fashion: acc -0.126±0.135 pp (2/3 adverse), F1 -0.336±0.271 pp (2/3 adverse), CE 0.0058±0.0027 (3/3 adverse).
- `smooth_only` (n=9 evaluations over 9 checkpoints): pooled record mean±SD acc -2.401±3.580 pp, F1 -3.946±5.960 pp, CE 0.0533±0.0782; record ranges acc [-11.793,0.029] pp, F1 [-19.202,0.045] pp, CE [-0.0003,0.2544]; Movies: acc -0.330±0.467 pp (1/3 adverse), F1 -0.217±0.339 pp (1/3 adverse), CE 0.0039±0.0057 (1/3 adverse); Grocery: acc -0.556±0.456 pp (2/3 adverse), F1 -0.812±0.639 pp (2/3 adverse), CE 0.0140±0.0117 (2/3 adverse); ele-fashion: acc -6.318±3.872 pp (3/3 adverse), F1 -10.807±5.937 pp (3/3 adverse), CE 0.1421±0.0794 (3/3 adverse).

`pi_shuffle_within_target` used repeat seeds 1001–1005, with Text and Visual independently permuted. The permutation intervention used all five non-identity orders: SRX→SXR, RSX, RXS, XSR, XRS. The exact order, per-run values, min/max, and all CE deltas are in the intervention CSVs. KeepStatic's global-mean and KeepTarget's target-mean routing identity checks passed; PremixEdgeControl's collapsed-input identity passed. Smooth-only retains the composer with `[CS,0,0]`.

## Answers to the frozen questions

1. **E0 relation state and function bank preserved?** Yes. The E0 module implementation is reused. M0/E0 relation-state, experts and router have explicit regression tests; the Movies/42 GPU smoke passed, including bitwise expert and router outputs under common weights. All 9 dataset×seed E0 common parameter initializations are bitwise equal.
2. **Four variants parameter-identical?** Yes; all counts and initialization hashes are in the audit CSVs. Each dataset×seed has one unique row per variant.
3. **Same-seed initialization identical?** Yes for every model parameter and classifier across all four variants; composers are initialized after all shared E0 modules.
4. **Is PremixEdgeControl a fair collapsed-provenance control?** Yes. It has identical architecture, parameter count, initialization, edge router and `Cmix` base; all three input blocks are active and repeated. It has full composer capacity but no function identity.
5. **Does `CS+CR+CX` recover `Cmix`?** Yes by construction, and the explicit numerical identity is checked per modality and run.
6. **Do function contexts remain distinct after aggregation?** See node-level cosine/mass tables above; per-edge distinctness from E0 is not substituted for these aggregated diagnostics.
7. **Do Movies/Grocery R/X channels vanish?** Their validation-node relative mass and RMS are listed separately; no inference is based only on E0 router averages.
8. **Does ele-fashion retain multiple channels?** Its channel RMS/mass/cosine rows above provide the direct node-context evidence.
9. **Is the composer used?** Ratio and cosine diagnostics above quantify output relative to the base and alignment with it; `composer_off` independently measures fixed-checkpoint reliance.
10. **Does composer capacity alone change E0?** PremixEdgeControl−historical E0 EdgeMix is Accuracy 0.173 ± 0.223 pp; Macro-F1 0.070 ± 0.539 pp; CE 0.0090 ± 0.0210; descriptive only because training seeds and model capacity context differ from the E0 reference.
11. **Does provenance add value over matched Premix?** KeepEdge−PremixEdgeControl is Accuracy -0.003 ± 0.135 pp; Macro-F1 -0.070 ± 0.691 pp; CE -0.0110 ± 0.0199, interpreted with collapse/permutation checkpoint reliance and node-context diagnostics.
12. **Does target routing add value?** KeepTarget−KeepStatic is Accuracy 0.037 ± 0.182 pp; Macro-F1 -0.786 ± 0.753 pp; CE -0.0100 ± 0.0128.
13. **Does edge routing add value after provenance?** KeepEdge−KeepTarget is Accuracy 0.039 ± 0.176 pp; Macro-F1 0.459 ± 0.836 pp; CE 0.0094 ± 0.0163.
14–15. **Do collapse/permutation hurt?** See per-dataset run-level direction counts and five-order min/max in the intervention tables. Reliance and incremental performance remain separate.
16. **Are target-mean/shuffle more damaging than in E0?** E0.1 and historical E0 per-dataset/run summaries are retained in paired intervention files; this report compares their signs and magnitudes descriptively only.
17. **What explains any improvement?** The matched control separates composer capacity from information preservation; router granularity contrasts target/edge; the fixed-checkpoint interventions check reliance. A single contrast is not used to attribute all changes.
18. **Was premature mixing a major E0 bottleneck?** PARTIALLY. Provenance-collapse/permutation and composer-off show mechanism use, and KeepEdge improves CE relative to Premix on average; however, Accuracy/Macro-F1 are essentially unchanged versus the matched control. This supports partial mechanism activity, not a demonstrated classification bottleneck.
19. **Does provenance make useful edge routing possible?** NO under the combined performance and checkpoint-reliance evidence.
20. **Next bottleneck:** function-context degeneration (Movies/Grocery have near-vanishing node-level R/X mass; ele-fashion retains multiple contexts; screen-level indication only).

## Frozen interpretation and limitations

Final label: **`PROVENANCE_ACTIVE_BUT_NO_MODEL_GAIN`**. The label combines (i) KeepEdge vs PremixEdgeControl and KeepEdge vs KeepTarget, (ii) provenance collapse/permutation and routing interventions, and (iii) node-level RMS/mass/cosine diagnostics. It is not selected from any one intervention. Comparisons use three seeds per dataset and validation labels for model selection and diagnostics; no significance tests are claimed. Historical E0 and UNI are descriptive references. P1.3 motivates the hypothesis only and is neither validated nor overturned by E0.1. No new expert, routing family, relation evidence, calibration, test evaluation, link prediction, or M1 component was added.

## Figure QA

The plotting source passed strict static preflight (21 PASS, 0 WARN, 0 FAIL). All six panels/figures passed the 1.5 pt strict alignment gate. All PDFs passed the 5 pt text floor (smallest text 5.3 pt); the rendered collision audit found 0 text collisions, clipping failures, or warnings across the six figures. The six final PNGs were visually inspected after the last render. The initial audit caught crowded paired-comparison and intervention tick labels; the final render uses compact/vertical labels and passes.

## Artifacts

- `README.md`, `run_manifest.json`, this `report.md`
- Run-level and summary CSVs under `data/`, including matched-pair and intervention records
- `data/expert_scale_diagnostics.csv` records all seven frozen E0 scale distributions by modality, variant, and run
- Six requested figures under `figures/` with PDF/SVG/600-dpi TIFF companion exports
- Figure alignment and rendered QA notes under `figures/qa/`
- Raw run JSONs and checkpoints under ignored `outputs/e01_function_provenance_preservation/`
