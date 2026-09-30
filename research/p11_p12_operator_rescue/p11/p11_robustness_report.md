# P1.1 robustness audit

This analysis reuses only the committed P0/P1 validation edge evidence; no model was trained. Raw utility is reconstructed as `U_raw = U_relative × CE_full`, which equals `CE_removed − CE_full` under the P0 signed-utility definition. All association, sign, and set-overlap summaries are descriptive across directed messages; no p-values are reported.

## H1: utility spread and mixed-sign neighborhoods

The sign fractions are invariant to the positive per-edge `CE_full` scale. Within-node spread is recalculated from raw ΔCE, so its magnitude is not directly comparable to relative utility units.

### Movies

| Modality | Helpful edge fraction | Harmful edge fraction | Median node IQR (raw ΔCE) | Median node q90–q10 | Mixed-sign target ratio |
|---|---:|---:|---:|---:|---:|
| Text | 0.620 | 0.380 | 0.015142 | 0.028814 | 0.389 |
| Visual | 0.630 | 0.370 | 0.028633 | 0.054222 | 0.549 |

### Grocery

| Modality | Helpful edge fraction | Harmful edge fraction | Median node IQR (raw ΔCE) | Median node q90–q10 | Mixed-sign target ratio |
|---|---:|---:|---:|---:|---:|
| Text | 0.781 | 0.219 | 0.0020573 | 0.0036494 | 0.396 |
| Visual | 0.765 | 0.235 | 0.0024954 | 0.0044262 | 0.429 |

### ele-fashion

| Modality | Helpful edge fraction | Harmful edge fraction | Median node IQR (raw ΔCE) | Median node q90–q10 | Mixed-sign target ratio |
|---|---:|---:|---:|---:|---:|
| Text | 0.777 | 0.223 | 7.0472e-05 | 0.00014341 | 0.156 |
| Visual | 0.267 | 0.733 | 5.4308e-05 | 0.00010853 | 0.165 |

## H2: Text and Visual utility on the same edge

| Dataset | Raw ΔCE Spearman | Raw ΔCE Pearson | Sign disagreement | Median absolute difference | Degree≥5 top Jaccard | Degree≥5 bottom Jaccard | Relative Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.067 | 0.093 | 0.414 | 0.017804 | 0.187 | 0.195 | 0.168 |
| Grocery | 0.416 | 0.346 | 0.223 | 0.0022072 | 0.229 | 0.266 | 0.576 |
| ele-fashion | -0.211 | -0.144 | 0.672 | 0.0043306 | 0.199 | 0.201 | -0.303 |

The same-edge sign-disagreement rate remains unchanged when moving from relative utility to raw ΔCE because each row is multiplied by its positive `CE_full`. Correlation and utility-rank overlaps can move because edge scales differ.

## H3: raw and task-projected cosine

The table reports seed means for raw ΔCE. Spearman is the primary rank association; Pearson and AUROC are secondary. Degree≥5 Jaccards average the per-target top/bottom 20% sets.

| Dataset | Modality | Raw cosine Spearman | Projected cosine Spearman | Raw cosine AUROC | Projected cosine AUROC | Raw top Jaccard | Projected top Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | Text | 0.013 | 0.026 | 0.520 | 0.556 | 0.164 | 0.162 |
| Movies | Visual | 0.016 | 0.029 | 0.523 | 0.565 | 0.163 | 0.165 |
| Grocery | Text | 0.022 | 0.088 | 0.546 | 0.684 | 0.182 | 0.216 |
| Grocery | Visual | 0.043 | 0.055 | 0.583 | 0.688 | 0.188 | 0.215 |
| ele-fashion | Text | 0.044 | 0.069 | 0.513 | 0.611 | 0.196 | 0.201 |
| ele-fashion | Visual | -0.072 | -0.003 | 0.496 | 0.475 | 0.175 | 0.186 |

Across the six dataset×modality groups, task-projected cosine raises mean sign AUROC over frozen raw cosine in five groups (the exception is ele-fashion Visual); gains are clearest in Grocery (about 0.68 AUROC). Its Spearman correlations remain modest (roughly −0.003 to 0.088), and the picture is not consistent across modalities or datasets. Projected similarity therefore helps classify positive utility in some settings but is not sufficient to determine propagation utility. The full seed-level comparison, including Pearson and bottom-tail overlaps and relative-utility sensitivity rows, is in `raw_vs_projected_similarity_summary.csv`.

## H1 after restricting to degree ≥ 5

| Dataset | Modality | Mixed-sign ratio | Median node IQR (raw ΔCE) | Median node q90–q10 | Relative mixed-sign ratio |
|---|---|---:|---:|---:|---:|
| Movies | Text | 0.594 | 0.017864 | 0.035056 | 0.594 |
| Movies | Visual | 0.832 | 0.033344 | 0.064977 | 0.832 |
| Grocery | Text | 0.678 | 0.0043263 | 0.0081335 | 0.678 |
| Grocery | Visual | 0.734 | 0.005312 | 0.009774 | 0.734 |
| ele-fashion | Text | 0.400 | 0.0012651 | 0.0027172 | 0.400 |
| ele-fashion | Visual | 0.473 | 0.00111 | 0.0022776 | 0.473 |

In ele-fashion, degree≥5 raises the mixed-sign ratio from 0.156 to 0.400 for Text and from 0.165 to 0.473 for Visual. Low-degree targets therefore explain a substantial part of the earlier weak aggregate, although degree≥5 heterogeneity remains below Movies and Grocery for at least one modality. Raw spread magnitudes also depend on each dataset’s CE scale.

## P1.1 assessment

Raw ΔCE preserves H1’s global sign mix and within-node mixed-sign neighborhoods, and it preserves H2’s same-edge sign disagreement and dataset ordering. Text/Visual raw-utility correlations are smaller than relative-utility correlations in all three datasets, but remain positive in Movies/Grocery and negative in ele-fashion. H3 remains weak for frozen cosine; task-projected cosine improves sign AUROC in five of six dataset×modality groups but still has modest rank correlation and inconsistent overlap. Thus raw-scale robustness retains the P0 propagation-utility heterogeneity claim, while P1.1 alone does not establish function heterogeneity.
