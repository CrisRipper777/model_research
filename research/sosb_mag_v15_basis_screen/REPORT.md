# SOSB-MAG V1.5 Structural Basis Screen

## Protocol and scope

- Branch: `exp/sosb_mag_v15_basis_screen`; freeze commit: `53eb362c86c433f3322397aa1eee7bafcb070824`.
- Formal runs: 36/36; unresolved failures: 0 (failure attempts recorded: 0).
- Validation Accuracy selects the checkpoint. `task.evaluate_test=false`; no Test metrics were computed and no Test labels were indexed.
- This screen compares structural response coordinates with a protected intrinsic residual; it does not test node routing or expert specialization.
- SOSB is an **OptBasis-inspired signal-conditioned orthogonal Krylov basis**, not an exact OptBasisGNN reproduction.
- Breakdown uses unregularized residual RMS for the `<1e-5` decision because adding `eps=1e-8` first would floor RMS at `1e-4`; normalization uses `sqrt(mean(square)+eps)`. This makes the configured breakdown threshold operative.
- Gram condition number is `cond(G + jitter*I)`, where `jitter=1e-6*max(1, abs(trace(G))/K)`; RawPoly and SOSB use the same Gram definition.
- No HPO, Test evaluation, significance testing, or causal intervention was performed.

## Validation results

Values below average the three dataset-level seed means equally. They are descriptive summaries, not pooled node metrics.

| Variant | Validation Accuracy (%) | Validation Macro-F1 (%) |
|---|---:|---:|
| A0_legacy_lg | 73.29 | 62.19 |
| A1_rawpoly_shared | 74.50 | 64.83 |
| A2_sosb_shared | 72.94 | 62.02 |
| A3_sosb_modality | 72.84 | 61.67 |

## Paired comparisons

Positive counts use paired dataset-seed runs, with no inferential test.

| Comparison | Paired Δ Accuracy | Paired Δ Macro-F1 | Positive Acc. pairs | Positive F1 pairs |
|---|---:|---:|---:|---:|
| A1_rawpoly_shared-minus-A0_legacy_lg | +1.21 pp | +2.64 pp | 8/9 | 8/9 |
| A2_sosb_shared-minus-A1_rawpoly_shared | -1.56 pp | -2.81 pp | 0/9 | 0/9 |
| A2_sosb_shared-minus-A0_legacy_lg | -0.35 pp | -0.16 pp | 2/9 | 4/9 |
| A3_sosb_modality-minus-A2_sosb_shared | -0.11 pp | -0.36 pp | 3/9 | 3/9 |

### Paired means by dataset

Entries are mean paired accuracy / Macro-F1 differences in percentage points; per-dataset positive seed counts are in `data/paired_comparisons.csv`.

| Dataset | A1−A0 | A2−A1 | A2−A0 | A3−A2 |
|---|---:|---:|---:|---:|
| Movies | +1.15 pp / +3.51 pp | -1.85 pp / -4.20 pp | -0.70 pp / -0.69 pp | -0.10 pp / -0.60 pp |
| Grocery | +2.13 pp / +3.07 pp | -2.62 pp / -2.94 pp | -0.49 pp / +0.13 pp | -0.08 pp / -0.37 pp |
| ele-fashion | +0.36 pp / +1.36 pp | -0.21 pp / -1.28 pp | +0.15 pp / +0.08 pp | -0.14 pp / -0.10 pp |

## Conditioning and breakdown

Across selected A1/A2/A3 modality-checkpoints, mean Gram diagonal was 0.999998 for RawPoly and 0.999999 for SOSB; mean absolute off-diagonal entry was 0.86256 and 1.9859e-09, respectively.
Mean jittered Gram condition number was 371.34 for RawPoly and 1 for SOSB. These describe coordinate conditioning; they do not establish a performance cause.
No SOSB breakdown was observed: all selected-checkpoint breakdown fractions were zero at orders 1–4. The per dataset, seed, modality, and order values are in `data/basis_diagnostics.csv`.

## Learned coefficients and response scale

Means below cover selected dataset/seed checkpoints; A3 is split by modality. The correction-to-prior ratio uses active-node RMS.

| Variant / profile | Effective β[1..4] mean | Gate mean | Prior RMS | Response RMS | Scaled correction / prior RMS |
|---|---|---:|---:|---:|---:|
| A1_rawpoly_shared | [0.536, 0.492, 0.503, 0.463] | 0.1268 | 0.9937 | 1.8875 | 0.2407 |
| A2_sosb_shared | [0.581, 0.567, 0.409, 0.415] | 0.1180 | 0.9990 | 1.0000 | 0.1181 |
| A3_sosb_modality/text | [0.583, 0.570, 0.404, 0.409] | 0.1183 | 0.9998 | 1.0000 | 0.1183 |
| A3_sosb_modality/visual | [0.574, 0.548, 0.426, 0.433] | 0.1170 | 0.9982 | 1.0000 | 0.1172 |

## Interpretation rules applied to these results

1. **K=4 response space:** A1−A0 is +1.21 pp Accuracy and +2.64 pp Macro-F1, positive in 8/9 pairs for both metrics. This supports the richer raw K=4 response space over the current Local/Global control in this screen.
2. **Orthogonal basis vs raw polynomial:** A2−A1 is -1.56 pp Accuracy and -2.81 pp Macro-F1, with 0/9 positive pairs. SOSB has much healthier Gram conditioning, but no task advantage here; the conditioning change is not evidence that orthogonality caused the metric change.
3. **Modality-specific coefficients:** A3−A2 is -0.11 pp Accuracy and -0.36 pp Macro-F1, positive in 3/9 pairs for each. The screen finds no performance gain from separate modality coefficients/gates.
4. **Shared SOSB coefficients:** A3 is close to A2 in overall mean, though slightly lower; this keeps shared coefficients viable for these data without proving modality structural differences are absent.
5. **Local/Global control:** the all-new-variants-weaker pattern does not hold because A1 exceeds A0. A2 and A3 are slightly below A0 on the equally weighted overall means; no hybrid bank or next-stage MoE is justified by this screen.

A2−A0 averages -0.35 pp Accuracy and -0.16 pp Macro-F1, with only 2/9 positive Accuracy pairs. All comparisons remain descriptive and are limited to three datasets and three seeds per dataset.

## Scientific boundaries

This experiment evaluates coordinate conditioning, redundancy, and modality-conditioned structural response. It does not show that orthogonalization expands the polynomial function space, that SOSB is task-optimal, that node routing is unnecessary, or how many experts a later model should use. No M=4 expert bank or router was implemented. Stop this stage here.

## Reproducibility artifacts

- Run-level results: `data/run_rows.json`; aggregate manifest: `data/campaign_manifest.json`.
- Validation summaries and paired comparisons: `data/summary.csv`, `data/paired_comparisons.csv`.
- Selected-checkpoint basis/profile records: `data/basis_diagnostics.csv`, `data/coefficient_profiles.csv`.
- Legacy V1 readout: `LEGACY_DIAGNOSTIC.md` and `data/legacy_v1_diagnostics.json`.
