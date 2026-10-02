# CARE-MAG A0–A2 context-adapter architecture screen

- Base SHA: `579d8dcde6d9bf6d39efb4f9d88bf70a78acc38e`
- Formal status: `complete`; 36 requested runs across 3 datasets, 4 variants, and seeds 42–44.
- NC protocol: full-graph, frozen main task optimization, best checkpoint selected by validation accuracy.
- Test evaluation: disabled. Formal metrics contain validation fields only; development mode masks test-index labels on the training device.
- Mechanism analysis device: `cuda:0`; all interventions use validation nodes only.

## Formal validation performance

Values are mean ± population SD over the three paired seeds. Accuracy and Macro-F1 are fractions; CE is unscaled.

| Dataset | Variant | Val Accuracy | Val Macro-F1 | Val CE | Best epoch |
|---|---|---:|---:|---:|---:|
| Movies | prior_only | 0.5155 ± 0.0009 | 0.4001 ± 0.0102 | 1.5489 ± 0.0718 | 62.0 ± 16.1 |
| Movies | structural_base | 0.5551 ± 0.0010 | 0.4825 ± 0.0047 | 1.4338 ± 0.0415 | 90.0 ± 14.7 |
| Movies | static_adapter | 0.5580 ± 0.0032 | 0.4795 ± 0.0051 | 1.4351 ± 0.0448 | 90.3 ± 3.8 |
| Movies | context_adapter | 0.5525 ± 0.0007 | 0.4911 ± 0.0046 | 1.4078 ± 0.0269 | 74.0 ± 10.2 |
| Grocery | prior_only | 0.7765 ± 0.0014 | 0.7058 ± 0.0039 | 0.8340 ± 0.0017 | 57.7 ± 2.5 |
| Grocery | structural_base | 0.8200 ± 0.0028 | 0.7584 ± 0.0045 | 0.7155 ± 0.0221 | 89.3 ± 8.7 |
| Grocery | static_adapter | 0.8209 ± 0.0012 | 0.7572 ± 0.0039 | 0.6850 ± 0.0072 | 73.7 ± 5.9 |
| Grocery | context_adapter | 0.8219 ± 0.0006 | 0.7608 ± 0.0018 | 0.7119 ± 0.0086 | 91.7 ± 12.5 |
| ele-fashion | prior_only | 0.8713 ± 0.0008 | 0.7399 ± 0.0044 | 0.4301 ± 0.0049 | 146.0 ± 5.7 |
| ele-fashion | structural_base | 0.8739 ± 0.0007 | 0.7414 ± 0.0066 | 0.4122 ± 0.0135 | 158.3 ± 24.7 |
| ele-fashion | static_adapter | 0.8749 ± 0.0027 | 0.7469 ± 0.0108 | 0.4505 ± 0.0473 | 174.0 ± 58.0 |
| ele-fashion | context_adapter | 0.8737 ± 0.0006 | 0.7430 ± 0.0032 | 0.4101 ± 0.0027 | 122.0 ± 3.6 |

## Paired deltas

Deltas are first variant minus baseline at the same seed. SD is population SD; counts give positive / negative / tie runs. No IID p-values are used. For CE, negative values favor the first variant.

| Dataset | Comparison | Metric | Mean Δ ± SD | + / − / tie |
|---|---|---|---:|---:|
| Movies | structural_base - prior_only | Val Accuracy | +0.03959 ± 0.00136 | 3 / 0 / 0 |
| Movies | structural_base - prior_only | Val Macro-F1 | +0.08238 ± 0.01226 | 3 / 0 / 0 |
| Movies | structural_base - prior_only | Val CE | -0.11509 ± 0.09695 | 1 / 2 / 0 |
| Movies | static_adapter - structural_base | Val Accuracy | +0.00290 ± 0.00222 | 2 / 0 / 1 |
| Movies | static_adapter - structural_base | Val Macro-F1 | -0.00301 ± 0.00531 | 1 / 2 / 0 |
| Movies | static_adapter - structural_base | Val CE | +0.00136 ± 0.00666 | 2 / 1 / 0 |
| Movies | context_adapter - static_adapter | Val Accuracy | -0.00550 ± 0.00364 | 0 / 3 / 0 |
| Movies | context_adapter - static_adapter | Val Macro-F1 | +0.01162 ± 0.00357 | 3 / 0 / 0 |
| Movies | context_adapter - static_adapter | Val CE | -0.02728 ± 0.01925 | 0 / 3 / 0 |
| Movies | context_adapter - structural_base | Val Accuracy | -0.00260 ± 0.00143 | 0 / 3 / 0 |
| Movies | context_adapter - structural_base | Val Macro-F1 | +0.00861 ± 0.00183 | 3 / 0 / 0 |
| Movies | context_adapter - structural_base | Val CE | -0.02592 ± 0.01857 | 1 / 2 / 0 |
| Grocery | structural_base - prior_only | Val Accuracy | +0.04353 ± 0.00144 | 3 / 0 / 0 |
| Grocery | structural_base - prior_only | Val Macro-F1 | +0.05261 ± 0.00800 | 3 / 0 / 0 |
| Grocery | structural_base - prior_only | Val CE | -0.11850 ± 0.02141 | 0 / 3 / 0 |
| Grocery | static_adapter - structural_base | Val Accuracy | +0.00088 ± 0.00250 | 2 / 1 / 0 |
| Grocery | static_adapter - structural_base | Val Macro-F1 | -0.00119 ± 0.00163 | 1 / 2 / 0 |
| Grocery | static_adapter - structural_base | Val CE | -0.03048 ± 0.02922 | 1 / 2 / 0 |
| Grocery | context_adapter - static_adapter | Val Accuracy | +0.00098 ± 0.00060 | 3 / 0 / 0 |
| Grocery | context_adapter - static_adapter | Val Macro-F1 | +0.00355 ± 0.00479 | 2 / 1 / 0 |
| Grocery | context_adapter - static_adapter | Val CE | +0.02688 ± 0.01078 | 3 / 0 / 0 |
| Grocery | context_adapter - structural_base | Val Accuracy | +0.00185 ± 0.00258 | 2 / 1 / 0 |
| Grocery | context_adapter - structural_base | Val Macro-F1 | +0.00236 ± 0.00471 | 2 / 1 / 0 |
| Grocery | context_adapter - structural_base | Val CE | -0.00360 ± 0.02336 | 1 / 2 / 0 |
| ele-fashion | structural_base - prior_only | Val Accuracy | +0.00259 ± 0.00076 | 3 / 0 / 0 |
| ele-fashion | structural_base - prior_only | Val Macro-F1 | +0.00152 ± 0.00875 | 2 / 1 / 0 |
| ele-fashion | structural_base - prior_only | Val CE | -0.01796 ± 0.01808 | 1 / 2 / 0 |
| ele-fashion | static_adapter - structural_base | Val Accuracy | +0.00102 ± 0.00317 | 2 / 1 / 0 |
| ele-fashion | static_adapter - structural_base | Val Macro-F1 | +0.00541 ± 0.01643 | 2 / 1 / 0 |
| ele-fashion | static_adapter - structural_base | Val CE | +0.03833 ± 0.05627 | 2 / 1 / 0 |
| ele-fashion | context_adapter - static_adapter | Val Accuracy | -0.00119 ± 0.00320 | 1 / 2 / 0 |
| ele-fashion | context_adapter - static_adapter | Val Macro-F1 | -0.00382 ± 0.01158 | 1 / 2 / 0 |
| ele-fashion | context_adapter - static_adapter | Val CE | -0.04039 ± 0.04984 | 0 / 3 / 0 |
| ele-fashion | context_adapter - structural_base | Val Accuracy | -0.00017 ± 0.00089 | 2 / 1 / 0 |
| ele-fashion | context_adapter - structural_base | Val Macro-F1 | +0.00159 ± 0.00486 | 2 / 1 / 0 |
| ele-fashion | context_adapter - structural_base | Val CE | -0.00206 ± 0.01284 | 2 / 1 / 0 |

## Context-checkpoint mechanism diagnostics

Rows below aggregate the selected context checkpoints by dataset and modality. Hop attention means are followed by across-node alpha SD. Coefficient vectors are retained rank-wise in `data/mechanism_diagnostics.csv`.

| Dataset | Modality | λ | γ | α k1 / k2 / k3 | Effective hop | R RMS | Δ RMS / R RMS | γΔ RMS / R RMS | λγΔ RMS / R RMS | cos(Δ,R) | Node coefficient SD | Text/visual disagreement RMS |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | text | 0.486 | 0.104 | 0.429/0.278/0.294 | 1.865 | 0.4839 | 2.5416 | 0.2653 | 0.1290 | -0.0221 | 0.31258 | 0.46706 |
| Movies | visual | 0.510 | 0.106 | 0.421/0.220/0.359 | 1.937 | 0.6583 | 1.7410 | 0.1853 | 0.0946 | 0.2212 | 0.23782 | 0.46706 |
| Grocery | text | 0.510 | 0.109 | 0.388/0.248/0.364 | 1.976 | 0.5811 | 2.1149 | 0.2319 | 0.1183 | 0.1303 | 0.30992 | 0.43578 |
| Grocery | visual | 0.510 | 0.108 | 0.255/0.284/0.461 | 2.206 | 0.6458 | 1.5276 | 0.1655 | 0.0845 | 0.3225 | 0.27651 | 0.43578 |
| ele-fashion | text | 0.507 | 0.113 | 0.385/0.376/0.239 | 1.853 | 0.5126 | 3.6051 | 0.4072 | 0.2066 | 0.0688 | 0.39421 | 0.56552 |
| ele-fashion | visual | 0.509 | 0.114 | 0.480/0.189/0.331 | 1.852 | 0.5275 | 2.9899 | 0.3401 | 0.1735 | 0.0812 | 0.25486 | 0.56552 |

## Validation checkpoint interventions

Intervention deltas are relative to the same checkpoint's normal context forward. `coeff_node_shuffle` reports five deterministic shuffles per checkpoint; its SD includes seeds and shuffle repeats. These interventions assess checkpoint reliance on correspondence and do not replace the retrained context-versus-static comparison.

| Dataset | Intervention | Val Accuracy | Val Macro-F1 | Val CE | Δ Accuracy | Δ Macro-F1 | Δ CE |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | normal | 0.5525 ± 0.0007 | 0.4911 ± 0.0046 | 1.4078 ± 0.0269 |  |  |  |
| Movies | adapter_off | 0.5436 ± 0.0031 | 0.4768 ± 0.0079 | 1.4393 ± 0.0203 | -0.00890 | -0.01431 | +0.03146 |
| Movies | coeff_global_mean | 0.5505 ± 0.0005 | 0.4900 ± 0.0042 | 1.4070 ± 0.0246 | -0.00200 | -0.00105 | -0.00088 |
| Movies | coeff_node_shuffle | 0.5500 ± 0.0021 | 0.4876 ± 0.0037 | 1.4085 ± 0.0257 | -0.00254 | -0.00348 | +0.00067 |
| Grocery | normal | 0.8219 ± 0.0006 | 0.7608 ± 0.0018 | 0.7119 ± 0.0086 |  |  |  |
| Grocery | adapter_off | 0.8160 ± 0.0019 | 0.7529 ± 0.0017 | 0.7368 ± 0.0152 | -0.00586 | -0.00791 | +0.02490 |
| Grocery | coeff_global_mean | 0.8209 ± 0.0014 | 0.7615 ± 0.0028 | 0.7113 ± 0.0075 | -0.00098 | +0.00076 | -0.00059 |
| Grocery | coeff_node_shuffle | 0.8202 ± 0.0010 | 0.7601 ± 0.0025 | 0.7121 ± 0.0080 | -0.00162 | -0.00064 | +0.00029 |
| ele-fashion | normal | 0.8737 ± 0.0006 | 0.7430 ± 0.0032 | 0.4101 ± 0.0027 |  |  |  |
| ele-fashion | adapter_off | 0.8591 ± 0.0066 | 0.7121 ± 0.0079 | 0.4443 ± 0.0191 | -0.01463 | -0.03097 | +0.03415 |
| ele-fashion | coeff_global_mean | 0.8700 ± 0.0015 | 0.7340 ± 0.0015 | 0.4134 ± 0.0032 | -0.00368 | -0.00901 | +0.00332 |
| ele-fashion | coeff_node_shuffle | 0.8653 ± 0.0053 | 0.7307 ± 0.0039 | 0.4270 ± 0.0137 | -0.00843 | -0.01234 | +0.01685 |

## Conservative diagnostic label: `DATASET_DEPENDENT_MIXED`

- Structural base vs prior only: 3/3 datasets improve Accuracy, Macro-F1, and CE together; 0/3 show mixed directions.
- Static adapter vs structural base: 0/3 datasets improve Accuracy, Macro-F1, and CE together; 3/3 show mixed directions.
- Context adapter vs static adapter: 0/3 datasets improve Accuracy, Macro-F1, and CE together; 3/3 show mixed directions.
- Context coefficients are node-varying; inspect rank-wise standard deviations in the mechanism CSV.
- Meaningful degradation (at least two metrics moving by 0.5 percentage points or 0.005 CE) occurs under node shuffling in 1/3 datasets and global-mean replacement in 0/3. Directional changes occur for shuffle in 3/3; checkpoint correspondence sensitivity is dataset-dependent. Global mean removes node variation; shuffle preserves each checkpoint's coefficient rows while breaking their node assignment. These are checkpoint-reliance probes.
- Checkpoint intervention reliance does not establish a retrained context-over-static gain.
- Evidence framing for `DATASET_DEPENDENT_MIXED`: The incremental adapter evidence varies by dataset or validation metric; it is not a uniform context-adaptation result. Intervention reliance may coexist with mixed retrained results and does not resolve them.
- A3 decision: do not proceed to A3 on this screen alone.
- Trajectory collapse and correction scale are summarized numerically in the mechanism table; inspect per-checkpoint and per-rank values in `data/mechanism_diagnostics.csv` before interpreting them.
- Dataset regime: read each paired table row separately; pooled seeds across datasets are not treated as IID.

## Self-audit

1. **Only based on main?** Yes. The experiment branch was created from fetched `origin/main` at `579d8dcde6d9bf6d39efb4f9d88bf70a78acc38e`; no code or history from another branch was used.
2. **Any test evaluation or test metric access?** No test evaluation was enabled. Analyzer rejects any `test_*` metric keys; only train/validation labels are used for validation bookkeeping.
3. **Did development_no_test isolate test labels?** Yes. The NC runner errors if test evaluation is enabled, builds Macro-F1 labels from train+validation indices only, and masks test-index labels to -1 before moving labels to the training device. Per-run metadata records the mode.
4. **Are all four variants parameter matched?** Yes, parameter counts and state-dict layouts are tested equal.
5. **Are A1/A2 parameters and initialization matched?** Yes. Tests reconstruct each variant with the same seed and verify all same-named state tensors bitwise equal.
6. **Is structural_base clearly better than prior_only?** - Structural base vs prior only: 3/3 datasets improve Accuracy, Macro-F1, and CE together; 0/3 show mixed directions.
7. **Does the static adapter add independent value?** - Static adapter vs structural base: 0/3 datasets improve Accuracy, Macro-F1, and CE together; 3/3 show mixed directions.
8. **Does context add over static?** - Context adapter vs static adapter: 0/3 datasets improve Accuracy, Macro-F1, and CE together; 3/3 show mixed directions.
9. **Are context coefficients node-varying?** Yes; see per-rank and node SDs in the mechanism CSV.
10. **Do global-mean/shuffle interventions show correspondence use?** - Meaningful degradation (at least two metrics moving by 0.5 percentage points or 0.005 CE) occurs under node shuffling in 1/3 datasets and global-mean replacement in 0/3. Directional changes occur for shuffle in 3/3; checkpoint correspondence sensitivity is dataset-dependent. Global mean removes node variation; shuffle preserves each checkpoint's coefficient rows while breaking their node assignment. These are checkpoint-reliance probes.
11. **Does checkpoint reliance agree with retrained architecture gain?** - Checkpoint intervention reliance does not establish a retrained context-over-static gain.
12. **Did trajectory readout collapse to a fixed hop?** No strong fixed-hop collapse (mean max-hop weight 0.434; dominant k=1 in 72% of rows).
13. **Is correction too large relative to response?** The maximum trust- and gamma-scaled Δ/R RMS ratio was 0.25369 (mean 0.13442); raw Δ/R remains in the mechanism CSV. Interpret with validation comparisons and cosine diagnostics.
14. **Is there a dataset-dependent regime?** Yes; the final label is mixed.
15. **Enough evidence for A3 shared/private residual MoE?** - A3 decision: do not proceed to A3 on this screen alone.
16. **Does LP smoke establish sampled-protocol compatibility?** NC GPU smoke `passed`; sports-copurchase LP smoke `passed`. LinkNeighborLoader=True, fanouts=[5, 5, 5], positive edges removed per epoch=[2306, 2214], validation inference=True, checkpoint saved=True. This establishes execution compatibility only, not LP quality.
17. **Any claim beyond direct support?** No. This screen supports validation-only comparisons for these three datasets and this frozen optimization setup; it does not establish test/generalization gains, formal LP quality, or full CARE-MAG effectiveness.

## Limitations

- Three seeds give a compact paired screen, not high-precision estimates.
- All reported NC metrics are validation metrics; test results were intentionally neither evaluated nor read.
- The LP check is only a short sampled execution smoke and is not a performance campaign.
- Intervention outcomes describe reliance of the selected checkpoints on coefficient-to-node correspondence; they are not retrained architecture comparisons.
- The architecture is limited to the specified three-hop diffusion, global trust scalars, and concat residual fusion.

## A3 recommendation

- A3 decision: do not proceed to A3 on this screen alone.
