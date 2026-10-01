# N1 — Recipient-Conditioned Function-Strength Mixing Screen

## Summary

**Final label: `DYNAMIC_MIXING_ACTIVE_NO_GAIN`.** Whole-neighborhood S/D/P channels are used by the trained CrossState checkpoints, and same/cross-state strengths become node-dependent. Removing the channels, replacing validation-node strengths with global means, shuffling recipient assignments, or tying Text/Visual strengths harms those checkpoints. The retrained controls do not show a stable multi-metric increment for recipient conditioning or cross-modal conditioning across datasets. The N0 target-level modulation result therefore motivates an active mechanism here, but does not translate into reliable incremental NC value for this S/D/P mixer.

This is a validation-only mechanism screen. Beta is a learned nonnegative channel strength, not a utility estimate or ground-truth function preference. The channels are whole-neighborhood aggregates; nothing in this result identifies an edge function.

## Provenance, protocol and correctness

- Started from `exp/n0_recipient_state_function_context`, SHA `98081ab965b1bb832218659862a0f3e3ba288dea`, with local `origin/exp/n0_recipient_state_function_context` at the same SHA. The experiment branch is `exp/n1_recipient_function_strength_mixer`; `main` was not merged. `git fetch origin` was attempted before work, but DNS resolution for `github.com` failed in the execution environment.
- Campaign: Movies, Grocery and ele-fashion × seeds 42/43/44 × four variants = **36/36 successful runs**, zero failed runs or retries. Full-graph `unified_full_graph_nc_v1`, train labels only, validation accuracy selection, validation CE and Macro-F1 recorded at each candidate checkpoint. M0/E0 optimizer, AdamW settings, dropout, epoch limit, early stopping, gradient clipping and null scheduler were retained.
- `evaluate_test=false`; the NC loader supplies no test indices, and labels outside train/validation are masked before training. No test metrics, LP or LP datasets were used.
- Every variant constructs the same projectors, Smooth weights, fusion, residual norms, D/P transforms and two independent strength mixers. Per dataset×seed model and classifier hashes match bitwise; parameter counts match exactly. The classifier uses the E0.1 seed isolation convention (`seed + 1907`), then the training RNG is reset to the run seed. Runtime hashes are checked against the audit before every run. See `data/parameter_init_audit.csv`.
- `tests/test_n1_recipient_function_strength_mixer.py`: **25 passed**, one upstream PyG deprecation warning. Coverage includes modality split, directed self-loop-free support and fixed degree; all three message formulas; brute-force/chunk equivalence; SmoothOnly/M0 regression; exact masks and zero static input; initialization and classifier fairness; strength constraints; residual/fusion formula; gradient behavior; beta interventions; validation-only overrides; channel and diagnostic finiteness; frozen task configuration and test-label masking.
- Movies/42 SmoothOnly and M0 UNI use the same common initialization and training protocol. Best epoch was 74 for both; N1 ran 104 epochs as did M0. Validation differences (N1 minus M0) were accuracy **−0.090 pp**, Macro-F1 **−0.030 pp**, CE **+0.000044**. The real-graph forward regression measured maximum absolute error **1.79e−6** for final `z`, **7.15e−7** for `C_S`, and 0 for projected `H0`. It passes `rtol=1e−6, atol=1e−5`; this measured tolerance accounts for CUDA `index_add` atomic accumulation order. The channels use the exact same equations and mapped common weights.
- Accumulated training time was **385.2 s**; total campaign time **407.0 s**. Peak allocated GPU memory was **6.73 GiB** (ele-fashion); no OOM or non-finite run occurred.

## Validation performance

Values are mean ± population SD over the three seeds. Accuracy and Macro-F1 are percentages; CE is native scale. Each dataset×seed comparison is paired in `paired_delta_by_run.csv`.

### Movies

| Variant | Accuracy (%) | Macro-F1 (%) | CE |
|---|---:|---:|---:|
| SmoothOnly | 55.43 ± 0.32 | 45.87 ± 0.57 | 1.4461 ± 0.0599 |
| StaticStrength | 55.32 ± 0.53 | 44.67 ± 1.01 | 1.4239 ± 0.0168 |
| SameStateStrength | 55.47 ± 0.28 | 45.46 ± 1.40 | 1.4069 ± 0.0112 |
| CrossStateStrength | 55.53 ± 0.50 | 45.38 ± 0.39 | 1.4219 ± 0.0053 |

### Grocery

| Variant | Accuracy (%) | Macro-F1 (%) | CE |
|---|---:|---:|---:|
| SmoothOnly | 82.22 ± 0.21 | 73.28 ± 1.09 | 0.7053 ± 0.0268 |
| StaticStrength | 82.33 ± 0.37 | 74.43 ± 0.44 | 0.6985 ± 0.0223 |
| SameStateStrength | 82.45 ± 0.29 | 74.01 ± 0.80 | 0.6982 ± 0.0027 |
| CrossStateStrength | 82.65 ± 0.25 | 74.33 ± 0.68 | 0.7266 ± 0.0308 |

### ele-fashion

| Variant | Accuracy (%) | Macro-F1 (%) | CE |
|---|---:|---:|---:|
| SmoothOnly | 87.40 ± 0.06 | 68.74 ± 0.82 | 0.4130 ± 0.0057 |
| StaticStrength | 87.45 ± 0.06 | 68.52 ± 0.91 | 0.4181 ± 0.0175 |
| SameStateStrength | 87.44 ± 0.21 | 68.44 ± 1.41 | 0.4148 ± 0.0080 |
| CrossStateStrength | 87.41 ± 0.08 | 68.12 ± 0.07 | 0.4127 ± 0.0152 |

### Paired comparisons

Positive values are left minus right. Accuracy and Macro-F1 deltas are percentage points; CE deltas are native scale. The table shows per-dataset means over three matched seeds; the CSV also reports population SD and positive/negative/tie run counts.

| Dataset | Comparison | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---|---:|---:|---:|
| Movies | Static − Smooth | −0.110 | −1.193 | −0.02220 |
| Movies | Same − Static | +0.150 | +0.785 | −0.01700 |
| Movies | Cross − Same | +0.060 | −0.074 | +0.01500 |
| Movies | Cross − Smooth | +0.100 | −0.482 | −0.02420 |
| Grocery | Static − Smooth | +0.117 | +1.148 | −0.00683 |
| Grocery | Same − Static | +0.117 | −0.420 | −0.00034 |
| Grocery | Cross − Same | +0.205 | +0.316 | +0.02845 |
| Grocery | Cross − Smooth | +0.439 | +1.044 | +0.02128 |
| ele-fashion | Static − Smooth | +0.058 | −0.226 | +0.00507 |
| ele-fashion | Same − Static | −0.014 | −0.073 | −0.00331 |
| ele-fashion | Cross − Same | −0.031 | −0.319 | −0.00205 |
| ele-fashion | Cross − Smooth | +0.014 | −0.618 | −0.00029 |

The equal-weight nine-pair summaries are descriptive only (not IID significance tests): Static−Smooth is **+0.022 pp accuracy, −0.090 pp Macro-F1, −0.00799 CE**; Same−Static **+0.084 pp, +0.098 pp, −0.00688 CE**; Cross−Same **+0.078 pp, −0.026 pp, +0.01380 CE**; Cross−Smooth **+0.184 pp, −0.018 pp, −0.00107 CE**. Full SDs and run-direction counts are in `data/paired_delta_summary.csv`.

## Function strengths and channel diagnostics

- Static strengths remain global constants per modality and run. Their trained means are generally near 0.10–0.14, with exactly zero node-wise spread. Grocery shows the clearest static-bank signal: all three seeds improve Macro-F1 over SmoothOnly and the mean accuracy/CE directions are favorable. Movies loses Macro-F1 in all three seeds; ele-fashion has conflicting metrics. The bank is therefore not uniformly useful.
- Same/Cross strengths are node-dependent. Across seeds, validation-target standard deviations for beta range from about **0.07–0.17** on Movies and **0.10–0.24** on Grocery. They are larger and more uneven on ele-fashion: D-strength SD is about **0.71–0.99** in three Text/Visual cases, while Visual P-strength SD is only **0.025–0.064**. Thus the dynamic models are not static in disguise, but their variation depends strongly on modality, function and dataset.
- Effective median alternative-to-Smooth contributions (`beta × RMS(C_alt)/RMS(C_S)`) are nonzero. Typical dynamic medians are about **0.29–0.50** on Movies and **0.32–0.63** on Grocery. ele-fashion is uneven: D medians span **0.55–1.05** for active modality/function pairs, while Visual P is around **0.15–0.17**. Some D contributions match or exceed Smooth locally. Beta alone would hide this channel-scale dependence.
- Raw channel scale is not uniform. Across variants/seeds, median `RMS(C_D)/RMS(C_S)` is roughly **0.56–0.98** by dataset/modality, while `RMS(C_P)/RMS(C_S)` is roughly **1.36–1.78**. The 99th percentile ratios reach about 1.2–2.14. This is a visible Product-versus-Smooth scale skew; effective-contribution summaries account for it. The residual LayerNorm remains part of the fixed architecture.
- Pairwise channel cosine medians are close to zero overall (approximately **−0.21 to +0.24** across dataset, modality and variant summaries), rather than near ±1. The S/D/P contexts did not collapse to the same representation in these cosine diagnostics. Full median/q25/q75 tables are in `data/channel_distinctness.csv`.
- Text and Visual strengths differ, especially on Grocery and ele-fashion, but correlation is not uniformly high. Mean validation-node L1 distance between the two strength pairs ranges by dataset/state from about **0.27–0.33** (Movies), **0.50–0.82** (Grocery), and **1.76–2.04** (ele-fashion). D/P cross-modality Spearman correlations vary by run and are often weak or negative. These are recipient-strength differences, not edge modality roles.

## CrossState checkpoint interventions

All deltas below are intervention minus normal CrossState checkpoint; negative accuracy/F1 and positive CE indicate degradation. `node_beta_shuffle` reports the mean over five deterministic repeats per run. The CSV has per-run and per-repeat values, population SD, and within-run shuffle SD.

| Dataset | Intervention | Δ Accuracy (pp) | Δ Macro-F1 (pp) | Δ CE |
|---|---|---:|---:|---:|
| Movies | D off | −2.509 | −6.660 | +0.07521 |
| Movies | P off | −1.030 | −0.742 | +0.00708 |
| Movies | DP off | −2.779 | −6.963 | +0.06160 |
| Movies | Validation global mean | −0.370 | −0.673 | +0.00356 |
| Movies | Node beta shuffle | −0.520 | −0.909 | +0.00809 |
| Movies | Modality tied | −0.320 | −0.511 | +0.00192 |
| Grocery | D off | −0.810 | −2.239 | +0.01690 |
| Grocery | P off | −2.401 | −3.884 | +0.06457 |
| Grocery | DP off | −2.460 | −5.643 | +0.05904 |
| Grocery | Validation global mean | −0.508 | −0.825 | +0.00111 |
| Grocery | Node beta shuffle | −0.896 | −1.086 | +0.01495 |
| Grocery | Modality tied | −0.429 | −1.102 | +0.01053 |
| ele-fashion | D off | −5.312 | −9.367 | +0.10606 |
| ele-fashion | P off | −0.617 | −1.285 | +0.02089 |
| ele-fashion | DP off | −6.849 | −11.396 | +0.14741 |
| ele-fashion | Validation global mean | −0.440 | −0.904 | +0.01287 |
| ele-fashion | Node beta shuffle | −0.920 | −1.911 | +0.02503 |
| ele-fashion | Modality tied | −0.273 | −0.885 | +0.00887 |

D-off and DP-off degrade all three metrics in all nine runs. P-off also degrades accuracy/F1 in all nine and raises CE in eight. All 45 node-shuffle repeats reduce accuracy; 44/45 reduce Macro-F1 and 43/45 increase CE. Validation global-mean assignment and modality tying also degrade accuracy and Macro-F1 in all nine CrossState runs. These interventions show checkpoint reliance and recipient/modality correspondence. They do not establish that retraining the dynamic architecture beats its matched controls.

## Answers to the frozen questions

1. **SmoothOnly vs M0 UNI:** implementation-compatible. The full Movies/42 training reproduction selected epoch 74 in both; metric differences are small and consistent with measured CUDA accumulation roundoff. Forward regression passed at the measured tolerance above.
2. **Does the S/D/P bank add value over SmoothOnly?** Not consistently across datasets and metrics. StaticStrength has a useful Grocery pattern, but Movies Macro-F1 worsens and ele-fashion metrics conflict.
3. **What did StaticStrength learn?** Small global positive strengths near initialization (roughly 0.10–0.14); Text and Visual constants can differ, but values are node-invariant by construction.
4. **Is SameState beta node-dependent?** Yes, clearly on all datasets, with much larger and less balanced variation on ele-fashion.
5. **Does Same−Static have stable task gain?** No. Movies improves accuracy/F1 while lowering CE; Grocery improves accuracy but lowers Macro-F1; ele-fashion changes are small and mixed.
6. **Does CrossState use cross-modal state?** Its full-state input produces node-varying strengths, Text/Visual strengths differ, and modality tying degrades all metrics in all nine checkpoints. The cross-modal input still lacks a stable retrained gain over SameState.
7. **Does Cross−Same have stable gain?** No. Grocery accuracy rises in all seeds but CE worsens in all three; Movies and ele-fashion are mixed or small.
8. **Do Text/Visual strengths differ?** Yes, most visibly on ele-fashion; cross-modality rank correlations vary by dataset and seed and do not provide a consistent shared ordering.
9. **Are D/P contributions nondegenerate?** Yes. Effective contributions are nonzero, with D and P occupying different ranges. Several ele-fashion D medians approach or exceed Smooth, while Visual P remains relatively small.
10. **Did channels collapse?** No evidence of pairwise cosine collapse; medians stay near zero rather than ±1.
11. **Is channel scale imbalanced?** Product RMS is systematically above Smooth RMS (median ratio around 1.36–1.78); D is closer to or below Smooth. This should qualify interpretation of beta.
12. **Do D/P/DP-off interventions show reliance?** Yes. Effects are adverse in nearly all run-level metrics and fully consistent for D-off and DP-off.
13. **Does global-mean replacement matter?** Yes, with modest consistent accuracy/F1 degradation and CE increase in eight of nine runs.
14. **Does node correspondence matter?** Yes in the trained CrossState checkpoint: five-repeat tuple shuffles have adverse mean changes for every dataset; accuracy is adverse in 45/45 repeats.
15. **Does modality tying matter?** Yes for trained CrossState reliance: all nine runs lose accuracy/F1 and CE rises in eight of nine.
16. **Do retrained comparisons and interventions agree?** No. Interventions show the trained dynamic model uses its strengths; matched retrained Same/Cross gains are mixed and small.
17. **Did N0 modulation become model increment?** The learned mixer is active and checkpoint correspondence matters, but this does not become reliable incremental task value under the present S/D/P model screen.
18. **Mechanism label:** `DYNAMIC_MIXING_ACTIVE_NO_GAIN`.
19. **Expand to Toys / Reddit-S now?** **NO.** First review whether the tested function bank is a useful basis for this task; the dynamic mixer does not yet justify a larger dataset campaign.
20. **Next phase:** after human review, reconsider whole-neighborhood channel design and scale behavior, keeping Smooth as the compatibility baseline. Any new screen should retain matched retrained controls and validation-only analysis. No further model was implemented in this stage.

## Scientific self-audit

- No edge router, relation encoder, edge gate, attention, expert routing, stochastic sampler, set model, multi-hop path, topology edit, LP objective or utility supervision was added.
- AbsDiff and Product remain prototype function channels, not claims about independent relation sources or final best operators.
- Beta is interpreted as a learned strength. Checkpoint interventions are described as reliance; architecture value is judged only from retrained matched comparisons.
- Channel RMS is used alongside beta. Metric disagreement and dataset-specific patterns remain visible; the pooled nine-run table is descriptive.
- No test or LP access occurred. Strong SmoothOnly/M0 performance is not evidence against propagation-function-sensitive heterogeneity; the conclusion is limited to this S/D/P bank and recipient mixer.

## Artifacts

- Protocol, provenance, run status and per-run records: `run_manifest.json`.
- Performance and paired analyses: `data/performance_by_run.csv`, `data/performance_summary.csv`, `data/paired_delta_by_run.csv`, `data/paired_delta_summary.csv`.
- Fairness and diagnostics: `data/parameter_init_audit.csv`, `data/strength_summary.csv`, `data/strength_node_variation.csv`, `data/effective_contribution_summary.csv`, `data/channel_rms_summary.csv`, `data/channel_distinctness.csv`, `data/modality_strength_disagreement.csv`.
- CrossState interventions: `data/intervention_by_run.csv`, `data/intervention_summary.csv`.
- Figures: PNG/TIFF 600 dpi and editable SVG/PDF exports for all five plots in `figures/`; methods, uncertainty definitions and audit records are summarized in `figures/figure_qa.md`.
- Large per-run records and selected checkpoints are under gitignored `outputs/n1_recipient_function_strength_mixer/`.

This phase is complete. No Toys, Reddit-S, test or LP work is included; stop here for human review.
