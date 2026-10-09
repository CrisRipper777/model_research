# V5A Task-Grounded Multimodal Message-Effect Atlas

## Protocol and interpretation boundary

All effects are frozen-host exposed-message singleton contrasts. A directed incoming message is removed only from one receiver's cached normalized Raw hop basis; baseline upstream states, other receivers, other edges/hops, denominators, router, strength, and classifier stay fixed. This is not recursive graph deletion and does not establish a causal graph-edge effect. The pilot is descriptive, uses V4A R0 seed-42 hosts, and has no performance pass/fail gate.

Main Atlas and estimator generation used training receiver indices and training labels only. Validation labels were read only by the separate V4A same-checkpoint gate deletion audit. Test labels and Test metrics were not read.

## V4A same-checkpoint residual-gate deletion audit

For 27 R1/R2/R3 checkpoints, zeroing the active raw residual-gate parameter changed Validation accuracy by mean -0.0089 pp and macro-F1 by mean 0.0014 pp. This is a same-checkpoint direct-dependence audit only, not causal proof or a training-noise diagnosis. Per-variant dataset means and positive counts are in the accompanying CSV.

## Effect distributions by dataset

| Dataset | Singleton rows | Receivers | Directed edges | Delete mean ± SD | Delete median (P05, P95) | Delete negative fraction | Comp feasible fraction | Comp mean ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 36672 | 1500 | 4584 | 0.000432933 ± 0.0316299 | -1.31386e-05 (-0.0245208, 0.0256226) | 0.506 | 0.935 | 0.0004357 ± 0.0242412 |
| Grocery | 34480 | 1500 | 4310 | 0.00104698 ± 0.0337784 | 2.46174e-05 (-0.0179322, 0.0221017) | 0.394 | 0.913 | 0.000610816 ± 0.033149 |
| ele-fashion | 25616 | 1500 | 3202 | -0.000328714 ± 0.0208095 | -1.19151e-07 (-0.00405181, 0.00691717) | 0.502 | 0.785 | 0.000586128 ± 0.0460481 |

## Descriptive synthesis

- **Clear descriptive pattern:** mean singleton deletion effects sit close to zero while their distributions have visible spread. Within-receiver edge effects and same-edge hop effects vary across the sampled messages; hop sign changes occur for 41.6%–71.1% of receiver-edge-modality groups.
- **Mixed pattern:** Text and Visual deletion effects disagree in sign for 42.5%–52.2% of paired rows, while their effect correlations remain weak (Spearman -0.025 to 0.178). The direction and size of modality differences vary by dataset.
- **Suggestive but dataset-dependent:** mass-preserving compensation retains moderate rank association with deletion on Movies/Grocery (Spearman 0.458/0.522) and weaker association on ele-fashion (0.206); sign flips remain common (28.1%–39.3%).
- **Weak heuristic/prediction pattern:** most scalar-cue associations are near zero, and estimator global holdout Spearman is at most 0.18 in this pilot. Edge pairwise ranking accuracy generally stays near 0.5. E1 and E3 show some dataset-specific descriptive gains, but those gains do not repeat across targets and datasets. E2 does not show a consistent gain over E1, so the pilot gives inconclusive evidence for additional joint multimodal pair information.
- **Bundle pattern:** median relative interaction for the full eight-message deletion bundle is 3.3%–8.6%, with larger dispersion than the medians; joint effects are therefore often close to singleton sums in the middle of these samples but can differ substantially on some rows.

## Required research questions

### Movies

- **Q1 Edge heterogeneity:** within-receiver group SD averaged 0.00602862 across modality-hop cells; group counts and ranges are in heterogeneity_diagnostics.csv.
- **Q2 Text/Visual:** paired sign disagreement was 0.486; effect Spearman was 0.033.
- **Q3 Hop variation:** mixed-sign-across-hop fraction was 0.697; mean effect SD across hops was 0.00940817.
- **Q4 Compensation:** delete-versus-compensated Spearman was 0.458, sign flips 0.327, and median |comp|/|delete| 0.557.
- **Q9 Bundle non-additivity:** full eight-message median absolute interaction was 0.00102532; median relative interaction was 0.033.
### Grocery

- **Q1 Edge heterogeneity:** within-receiver group SD averaged 0.00555484 across modality-hop cells; group counts and ranges are in heterogeneity_diagnostics.csv.
- **Q2 Text/Visual:** paired sign disagreement was 0.425; effect Spearman was 0.178.
- **Q3 Hop variation:** mixed-sign-across-hop fraction was 0.416; mean effect SD across hops was 0.00632284.
- **Q4 Compensation:** delete-versus-compensated Spearman was 0.522, sign flips 0.281, and median |comp|/|delete| 0.765.
- **Q9 Bundle non-additivity:** full eight-message median absolute interaction was 0.000629; median relative interaction was 0.086.
### ele-fashion

- **Q1 Edge heterogeneity:** within-receiver group SD averaged 0.000832679 across modality-hop cells; group counts and ranges are in heterogeneity_diagnostics.csv.
- **Q2 Text/Visual:** paired sign disagreement was 0.522; effect Spearman was -0.025.
- **Q3 Hop variation:** mixed-sign-across-hop fraction was 0.711; mean effect SD across hops was 0.00436282.
- **Q4 Compensation:** delete-versus-compensated Spearman was 0.206, sign flips 0.393, and median |comp|/|delete| 0.435.
- **Q9 Bundle non-additivity:** full eight-message median absolute interaction was 6.55949e-05; median relative interaction was 0.048.

### Q5 Heuristic alignment

Per-dataset/modality/hop Spearman associations for semantic cosine, V3C role score, router reliability, edge weight, degree, and message magnitude are in heuristic_alignment.csv. These associations are descriptive and do not imply role success or failure.

Mean cue/effect Spearman across the 8 modality-hop cells for each dataset:
- Movies: semantic_cosine_text/delete=0.002; semantic_cosine_text/comp=-0.009; semantic_cosine_visual/delete=0.002; semantic_cosine_visual/comp=0.005; role_score_text/delete=-0.003; role_score_text/comp=-0.014; role_score_visual/delete=0.007; role_score_visual/comp=0.011; router_reliability_active_modality/delete=0.006; router_reliability_active_modality/comp=0.010
- Grocery: semantic_cosine_text/delete=0.003; semantic_cosine_text/comp=-0.022; semantic_cosine_visual/delete=0.006; semantic_cosine_visual/comp=-0.018; role_score_text/delete=-0.003; role_score_text/comp=-0.011; role_score_visual/delete=0.001; role_score_visual/comp=-0.018; router_reliability_active_modality/delete=0.034; router_reliability_active_modality/comp=-0.004
- ele-fashion: semantic_cosine_text/delete=-0.039; semantic_cosine_text/comp=-0.011; semantic_cosine_visual/delete=-0.046; semantic_cosine_visual/comp=-0.010; role_score_text/delete=-0.006; role_score_text/comp=-0.019; role_score_visual/delete=-0.014; role_score_visual/comp=-0.018; router_reliability_active_modality/delete=-0.058; router_reliability_active_modality/comp=-0.017

Support/discrepant negative-effect fractions are in heuristic_alignment.csv under P(delta_delete<0 | role_partition).

### Q6–Q8 Estimator comparisons

Estimator metrics use EstimatorHoldout receivers only. Absolute values are seed mean ± population SD over estimator seeds 0/1/2, separately for deletion and compensation targets.

### E0–E3 absolute holdout metrics

Values are seed mean ± population SD over estimator seeds 0/1/2. Ranking scores are receiver-group metrics.

| Dataset | Target | Estimator | Global Spearman | Edge ranking Spearman | Edge pairwise accuracy | Modality ordering accuracy | Hop ranking Spearman | Hop pairwise accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Movies | delete | E0_heuristic | 0.017 ± 0.019 | 0.006 ± 0.011 | 0.501 ± 0.004 | 0.470 ± 0.002 | 0.032 ± 0.034 | 0.516 ± 0.016 |
| Movies | delete | E1_unimodal_pair | 0.037 ± 0.015 | 0.030 ± 0.007 | 0.516 ± 0.005 | 0.514 ± 0.015 | 0.059 ± 0.030 | 0.527 ± 0.015 |
| Movies | delete | E2_multimodal_pair | 0.039 ± 0.004 | 0.023 ± 0.002 | 0.516 ± 0.002 | 0.472 ± 0.004 | 0.044 ± 0.020 | 0.521 ± 0.009 |
| Movies | delete | E3_host_context | 0.027 ± 0.004 | -0.001 ± 0.008 | 0.504 ± 0.002 | 0.510 ± 0.009 | 0.012 ± 0.012 | 0.504 ± 0.006 |
| Movies | comp | E0_heuristic | 0.011 ± 0.009 | 0.033 ± 0.015 | 0.518 ± 0.004 | 0.501 ± 0.013 | -0.015 ± 0.007 | 0.493 ± 0.004 |
| Movies | comp | E1_unimodal_pair | 0.003 ± 0.005 | 0.014 ± 0.006 | 0.516 ± 0.005 | 0.498 ± 0.003 | -0.014 ± 0.013 | 0.495 ± 0.006 |
| Movies | comp | E2_multimodal_pair | 0.002 ± 0.007 | 0.006 ± 0.019 | 0.511 ± 0.006 | 0.494 ± 0.006 | -0.013 ± 0.002 | 0.494 ± 0.000 |
| Movies | comp | E3_host_context | -0.007 ± 0.011 | 0.001 ± 0.019 | 0.508 ± 0.009 | 0.493 ± 0.001 | -0.026 ± 0.011 | 0.488 ± 0.003 |
| Grocery | delete | E0_heuristic | 0.043 ± 0.024 | 0.068 ± 0.017 | 0.533 ± 0.009 | 0.514 ± 0.027 | 0.069 ± 0.033 | 0.530 ± 0.013 |
| Grocery | delete | E1_unimodal_pair | 0.084 ± 0.014 | 0.052 ± 0.004 | 0.524 ± 0.004 | 0.542 ± 0.019 | 0.041 ± 0.009 | 0.519 ± 0.005 |
| Grocery | delete | E2_multimodal_pair | 0.052 ± 0.022 | 0.033 ± 0.026 | 0.518 ± 0.008 | 0.492 ± 0.004 | 0.044 ± 0.016 | 0.517 ± 0.008 |
| Grocery | delete | E3_host_context | 0.082 ± 0.017 | 0.034 ± 0.010 | 0.522 ± 0.005 | 0.546 ± 0.012 | 0.058 ± 0.025 | 0.524 ± 0.011 |
| Grocery | comp | E0_heuristic | 0.018 ± 0.011 | 0.051 ± 0.015 | 0.523 ± 0.004 | 0.515 ± 0.004 | -0.002 ± 0.012 | 0.500 ± 0.005 |
| Grocery | comp | E1_unimodal_pair | 0.001 ± 0.009 | 0.058 ± 0.040 | 0.515 ± 0.015 | 0.499 ± 0.008 | -0.010 ± 0.013 | 0.497 ± 0.006 |
| Grocery | comp | E2_multimodal_pair | 0.023 ± 0.015 | 0.038 ± 0.020 | 0.513 ± 0.008 | 0.501 ± 0.014 | -0.001 ± 0.007 | 0.499 ± 0.003 |
| Grocery | comp | E3_host_context | -0.027 ± 0.019 | 0.011 ± 0.044 | 0.506 ± 0.017 | 0.498 ± 0.005 | 0.006 ± 0.009 | 0.503 ± 0.005 |
| ele-fashion | delete | E0_heuristic | 0.082 ± 0.011 | 0.011 ± 0.011 | 0.508 ± 0.002 | 0.510 ± 0.029 | 0.091 ± 0.004 | 0.538 ± 0.002 |
| ele-fashion | delete | E1_unimodal_pair | 0.144 ± 0.034 | -0.036 ± 0.012 | 0.485 ± 0.004 | 0.568 ± 0.026 | 0.169 ± 0.010 | 0.570 ± 0.005 |
| ele-fashion | delete | E2_multimodal_pair | 0.101 ± 0.034 | -0.061 ± 0.032 | 0.486 ± 0.009 | 0.502 ± 0.000 | 0.165 ± 0.021 | 0.564 ± 0.012 |
| ele-fashion | delete | E3_host_context | 0.180 ± 0.106 | -0.003 ± 0.066 | 0.498 ± 0.032 | 0.574 ± 0.086 | 0.180 ± 0.054 | 0.575 ± 0.029 |
| ele-fashion | comp | E0_heuristic | -0.023 ± 0.003 | -0.025 ± 0.010 | 0.512 ± 0.004 | 0.496 ± 0.002 | -0.010 ± 0.020 | 0.496 ± 0.009 |
| ele-fashion | comp | E1_unimodal_pair | -0.036 ± 0.007 | -0.048 ± 0.023 | 0.499 ± 0.010 | 0.494 ± 0.016 | -0.026 ± 0.027 | 0.488 ± 0.011 |
| ele-fashion | comp | E2_multimodal_pair | -0.013 ± 0.020 | -0.019 ± 0.027 | 0.500 ± 0.006 | 0.498 ± 0.010 | -0.007 ± 0.027 | 0.496 ± 0.012 |
| ele-fashion | comp | E3_host_context | 0.003 ± 0.011 | 0.010 ± 0.053 | 0.518 ± 0.021 | 0.496 ± 0.009 | 0.013 ± 0.013 | 0.506 ± 0.005 |

### E1–E0, E2–E1, and E3–E2 paired seed differences

| Dataset | Target | Comparison | Global Spearman | Edge ranking Spearman | Edge pairwise accuracy | Modality ordering accuracy | Hop ranking Spearman |
|---|---|---|---:|---:|---:|---:|---:|
| Movies | delete | E1_unimodal_pair-E0_heuristic | 0.021 (0.007) | 0.024 (0.017) | 0.015 (0.006) | 0.044 (0.013) | 0.027 (0.031) |
| Movies | delete | E2_multimodal_pair-E1_unimodal_pair | 0.002 (0.013) | -0.007 (0.005) | -0.000 (0.003) | -0.042 (0.011) | -0.015 (0.035) |
| Movies | delete | E3_host_context-E2_multimodal_pair | -0.012 (0.002) | -0.024 (0.008) | -0.012 (0.003) | 0.037 (0.010) | -0.032 (0.011) |
| Movies | comp | E1_unimodal_pair-E0_heuristic | -0.008 (0.014) | -0.019 (0.010) | -0.002 (0.002) | -0.003 (0.011) | 0.001 (0.020) |
| Movies | comp | E2_multimodal_pair-E1_unimodal_pair | -0.001 (0.004) | -0.009 (0.015) | -0.005 (0.003) | -0.004 (0.007) | 0.001 (0.012) |
| Movies | comp | E3_host_context-E2_multimodal_pair | -0.010 (0.016) | -0.005 (0.011) | -0.003 (0.003) | -0.001 (0.006) | -0.013 (0.012) |
| Grocery | delete | E1_unimodal_pair-E0_heuristic | 0.041 (0.013) | -0.016 (0.013) | -0.010 (0.012) | 0.028 (0.018) | -0.028 (0.028) |
| Grocery | delete | E2_multimodal_pair-E1_unimodal_pair | -0.031 (0.018) | -0.018 (0.024) | -0.006 (0.009) | -0.050 (0.016) | 0.003 (0.011) |
| Grocery | delete | E3_host_context-E2_multimodal_pair | 0.029 (0.013) | 0.001 (0.020) | 0.004 (0.003) | 0.054 (0.013) | 0.014 (0.029) |
| Grocery | comp | E1_unimodal_pair-E0_heuristic | -0.017 (0.003) | 0.006 (0.033) | -0.007 (0.013) | -0.016 (0.008) | -0.008 (0.025) |
| Grocery | comp | E2_multimodal_pair-E1_unimodal_pair | 0.022 (0.012) | -0.020 (0.036) | -0.003 (0.009) | 0.001 (0.006) | 0.009 (0.008) |
| Grocery | comp | E3_host_context-E2_multimodal_pair | -0.050 (0.028) | -0.027 (0.046) | -0.006 (0.012) | -0.002 (0.015) | 0.007 (0.009) |
| ele-fashion | delete | E1_unimodal_pair-E0_heuristic | 0.062 (0.044) | -0.047 (0.019) | -0.023 (0.006) | 0.058 (0.055) | 0.079 (0.007) |
| ele-fashion | delete | E2_multimodal_pair-E1_unimodal_pair | -0.042 (0.019) | -0.026 (0.020) | 0.001 (0.006) | -0.066 (0.026) | -0.005 (0.016) |
| ele-fashion | delete | E3_host_context-E2_multimodal_pair | 0.079 (0.073) | 0.059 (0.034) | 0.012 (0.024) | 0.072 (0.087) | 0.015 (0.036) |
| ele-fashion | comp | E1_unimodal_pair-E0_heuristic | -0.013 (0.009) | -0.023 (0.031) | -0.013 (0.013) | -0.002 (0.018) | -0.016 (0.047) |
| ele-fashion | comp | E2_multimodal_pair-E1_unimodal_pair | 0.023 (0.015) | 0.029 (0.048) | 0.001 (0.017) | 0.004 (0.007) | 0.020 (0.046) |
| ele-fashion | comp | E3_host_context-E2_multimodal_pair | 0.016 (0.020) | 0.029 (0.076) | 0.018 (0.025) | -0.002 (0.018) | 0.019 (0.037) |

E1 adds a same-modality receiver/sender pair representation. E2 adds separate pair blocks from both modalities; a descriptive change does not prove cross-modal interaction. E3 adds frozen host context, so a change is consistent with additional predictive information in that context. No estimator is interpreted as a controller or a strong model-selection result.

### Q10 Implications for a later message-controller study

Observed within-receiver, modality, and hop variation can suggest preserving receiver-directed message granularity in a later controller characterization. Bundle interactions can suggest measuring joint interventions alongside singleton sums. These observations do not specify or automatically create a final controller architecture.

## Implementation audit

Baseline local replay reconstruction, finite-effect checks, compensation mass preservation, and receiver split disjointness are summarized in preflight_summary.json. Preflight and estimator smoke are implementation checks, not scientific gates. The preflight/smoke ran on cuda:1. A first formal cuda:1 attempt encountered out-of-memory while a concurrent workload occupied GPU memory; the unchanged frozen protocol then completed on CPU. Estimator architecture, split, optimizer, targets, and epoch/patience settings did not change. Formal estimator training completed 36/36 runs. Archived effect feature caches store infeasible compensation targets as finite zero placeholders with an explicit false target_comp_mask; training used the equivalent infeasible-target mask. Undefined group statistics (for example, a metric that does not apply to that ranking group or constant-valued ranks) are blank in the CSVs. Full features, raw effects, estimator checkpoints, and logs remain under ignored outputs/mag_message_effect_v5a.
