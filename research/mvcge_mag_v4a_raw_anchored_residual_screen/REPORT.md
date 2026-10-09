# MvCGE-MAG V4A: Raw-Anchored Adaptive Role Residual Screen

## Protocol

Validation-only full-graph node classification; 3 datasets × 3 seeds × 4 variants (36 runs). Checkpoints are selected by validation Accuracy; Macro-F1 is diagnostic. Test evaluation was disabled.

Preflight passed: **True**. Smoke: **4/4**. Formal rows: **36/36**.

The fixed V3C role partition is used only to form a Support-minus-Discrepant residual. The complete Raw trajectory and its expert MLP remain the backbone. Reported role contrast is not described as a heterophily or high-pass signal.

## Validation summary

| Dataset | Variant | Accuracy mean ± SD | Macro-F1 mean ± SD | Best epoch mean |
|---|---|---:|---:|---:|
| Movies | R0_raw | 56.029 ± 0.385% | 47.997 ± 1.637% | 83.0 |
| Movies | R1_global_residual | 56.069 ± 0.506% | 48.954 ± 0.572% | 68.7 |
| Movies | R2_expert_residual | 56.039 ± 0.248% | 49.076 ± 0.590% | 77.3 |
| Movies | R3_adaptive_residual | 56.029 ± 0.343% | 48.534 ± 0.215% | 84.7 |
| Grocery | R0_raw | 83.094 ± 0.325% | 75.766 ± 1.387% | 66.7 |
| Grocery | R1_global_residual | 83.133 ± 0.259% | 75.331 ± 1.194% | 67.3 |
| Grocery | R2_expert_residual | 83.153 ± 0.249% | 75.858 ± 0.627% | 80.0 |
| Grocery | R3_adaptive_residual | 83.133 ± 0.235% | 75.301 ± 1.149% | 69.3 |
| ele-fashion | R0_raw | 87.413 ± 0.128% | 74.778 ± 1.560% | 184.7 |
| ele-fashion | R1_global_residual | 87.460 ± 0.063% | 74.581 ± 0.152% | 169.0 |
| ele-fashion | R2_expert_residual | 87.348 ± 0.163% | 74.445 ± 0.245% | 158.7 |
| ele-fashion | R3_adaptive_residual | 87.409 ± 0.189% | 74.484 ± 1.568% | 165.7 |
| ALL | R0_raw | 75.512 ± 13.892% | 66.180 ± 12.954% | 111.4 |
| ALL | R1_global_residual | 75.554 ± 13.895% | 66.289 ± 12.285% | 101.7 |
| ALL | R2_expert_residual | 75.513 ± 13.878% | 66.460 ± 12.317% | 105.3 |
| ALL | R3_adaptive_residual | 75.524 ± 13.898% | 66.107 ± 12.481% | 106.6 |

## Paired descriptive comparisons

Paired deltas use the same dataset and seed. No significance tests were run. Stable-positive and approximately-equal labels follow the preregistered descriptive cutoffs and are not inferential claims.

| Comparison | Accuracy Δ (pp) | Macro-F1 Δ (pp) | Acc positive /9 | F1 positive /9 | Stable-positive | Approximately equal |
|---|---:|---:|---:|---:|:---:|:---:|
| R1_global_residual - R0_raw | +0.042 | +0.108 | 6/9 | 3/9 | True | True |
| R2_expert_residual - R1_global_residual | -0.041 | +0.171 | 5/9 | 6/9 | False | True |
| R2_expert_residual - R0_raw | +0.001 | +0.279 | 5/9 | 4/9 | False | True |
| R3_adaptive_residual - R2_expert_residual | +0.011 | -0.353 | 3/9 | 1/9 | False | True |
| R3_adaptive_residual - R0_raw | +0.012 | -0.074 | 5/9 | 3/9 | False | True |
| R3_adaptive_residual - R1_global_residual | -0.030 | -0.182 | 4/9 | 3/9 | False | True |

## Interpretation map

Use the paired table together with beta, realized residual magnitude, safety scales, and R3 gamma diagnostics. A stable-positive R1/R2/R3 indicates evidence for the corresponding residual complexity. If all arms are approximately equal and beta stays near zero, the model declined the fixed residual. Nonzero beta with no gain indicates use without validation benefit. R2 vs R3 distinguishes a shared Raw hop profile from an independent residual profile. These outcomes guide a later decision; this report does not launch V4B.

### Observed readout

The preregistered stable-positive rule classifies **R1−R0** as stable-positive (+0.042 pp Accuracy, +0.108 pp Macro-F1; Accuracy positive on 6/9 pairs, with positive per-dataset mean on all three datasets). It is also approximately-equal under the separate frozen cutoff because both mean deltas lie inside that descriptive band. These labels overlap by design and are neither significance nor equivalence tests; the small effect and mixed per-dataset Macro-F1 changes warrant a cautious reading.

None of the other primary comparisons is stable-positive: R2−R1 is −0.041/+0.171 pp (Accuracy/F1), R2−R0 is +0.001/+0.279 pp, R3−R2 is +0.011/−0.353 pp, and R3−R0 is +0.012/−0.074 pp. All five comparisons, including R3−R1, fall inside the approximately-equal cutoffs. Per-dataset paired means are in `data/paired_comparisons.csv`. Thus the result is consistent with a small descriptive benefit from the simplest global residual, but does not show a stable-positive benefit for expert-specific gates, an independent residual profile, or the full R3 mechanism.

Among these runs, R1 is the only residual variant that meets the frozen stable-positive rule against R0 and is therefore the simplest qualifying candidate **if a later V4B study is separately authorized**. This campaign stops here; V4B was not launched. R2−R0 does not meet the rule, so the expert-specific residual is not selected by the gate. R3 does not improve stably over R2 or R0. The R3 normalized gamma stays close to alpha (mean cosine 0.989), with mean normalized drift 0.067, no sign changes, and mean low/high-order absolute masses 1.016/0.976 (absolute-order centroid 2.479); this is limited profile movement, not evidence of a distinct response grammar.

Most learned gates remain near zero: R1 has 8/9 run-level global beta values with |beta|<0.01 (range −0.0076 to +0.0102); R2 has 32/36 and R3 has 31/36 expert gates below that threshold. No gate is near the 0.25 cap. The mean absolute beta is 0.0072 for R1, 0.0053 for R2, and 0.0060 for R3. Across active residual variants, safety scales average 0.948–0.954 and range from 0.655 to 1.0, so calibration only downscales. The scaled residual-to-Raw expert-mixture RMS ratio averages 0.0046 (0.0023–0.0136), and mean scaled residual correction RMS is 0.00209 versus 0.50986 for the Raw correction. This indicates small realized residual use overall, despite clear role-contrast action-space novelty.

This pattern does not meet the stronger branch in which all residuals are approximately equal to Raw *and* every gate is near zero: R1 is stable-positive descriptively and some gates exceed the near-zero cutoff. Nor is there evidence here for the branch requiring a substantial nonzero residual with no task benefit. The evidence favors retaining the fixed role contrast only as a small, bounded residual hypothesis; it does not motivate node-conditioned routing in this completed screen.

| Comparison | Movies Acc/F1 Δ (pp) | Grocery Acc/F1 Δ (pp) | ele-fashion Acc/F1 Δ (pp) |
|---|---:|---:|---:|
| R1−R0 | +0.040 / +0.957 | +0.039 / −0.435 | +0.048 / −0.197 |
| R2−R1 | −0.030 / +0.122 | +0.020 / +0.527 | −0.113 / −0.136 |
| R2−R0 | +0.010 / +1.078 | +0.059 / +0.092 | −0.065 / −0.333 |
| R3−R2 | −0.010 / −0.542 | −0.020 / −0.557 | +0.061 / +0.040 |
| R3−R0 | +0.000 / +0.537 | +0.039 / −0.465 | −0.003 / −0.293 |
| R3−R1 | −0.040 / −0.420 | +0.000 / −0.030 | −0.051 / −0.096 |

## Diagnostics and provenance

- Preflight max decomposition identity error: 3.81e-06
- Selected checkpoint audits: 36; finite=True; role partition valid=True.
- Same-checkpoint V4A R0 / V3C F0 forward allclose pairs: 9/9.
- Preflight reads frozen modality features and physical edges only; it does not load labels or split indices.
- Formal validation metrics contain no Test metrics; `task.evaluate_test=false` was verified from each saved Hydra configuration.
