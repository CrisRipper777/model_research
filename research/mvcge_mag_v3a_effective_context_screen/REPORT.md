# MvCGE-MAG V3A: Effective Structural Action-Space Screen

## Protocol and provenance

- Branch `exp/mvcge_mag_v3a_effective_context_screen`; parent `bb52895da5863c737fc47c1f88637be12a069445`; validation-only `unified_full_graph_nc_v1`; Movies, Grocery, `ele-fashion`; seeds 42–44; C0–C3; 36 runs.
- C0 is the V2.2 R0 modality-static selection and strength controller. The raw and effective contexts use modality-static Top-2 selection and the same static strength; no node-conditioned router is active.
- Semantic edge weights use only detached pretrained raw modality features on existing physical edges. No labels were used in operator or selected-checkpoint audits.
- `task.evaluate_test=false`; checkpoints are selected by Validation Accuracy. No Test metrics, HPO, LP, significance testing, or V3B implementation.
- Model/config were frozen before the formal campaign. Formal campaign provenance and selected-checkpoint audits are recorded in `data/campaign_manifest.json`.

## Validation results

Run-level means and population standard deviations; paired deltas are descriptive percentage points across matched dataset-seed runs.

| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Best epoch mean | Trainable parameters (model + head) |
|---|---|---:|---:|---:|---:|
| Movies | C0_raw | 55.89% ± 0.41% | 48.23% ± 1.29% | 77.3 | 1,448,885 + 5,140 |
| Movies | C1_effective | 55.96% ± 0.37% | 48.21% ± 1.85% | 67.3 | 1,448,885 + 5,140 |
| Movies | C2_dual_shared_profile | 55.82% ± 0.40% | 48.78% ± 0.72% | 85.7 | 1,448,889 + 5,140 |
| Movies | C3_dual_context_profile | 55.89% ± 0.35% | 47.84% ± 2.16% | 69.0 | 1,448,905 + 5,140 |
| Grocery | C0_raw | 83.08% ± 0.32% | 75.76% ± 1.38% | 66.7 | 1,448,885 + 5,140 |
| Grocery | C1_effective | 83.08% ± 0.26% | 75.37% ± 0.26% | 84.7 | 1,448,885 + 5,140 |
| Grocery | C2_dual_shared_profile | 83.18% ± 0.38% | 75.56% ± 2.01% | 67.7 | 1,448,889 + 5,140 |
| Grocery | C3_dual_context_profile | 83.33% ± 0.20% | 76.25% ± 0.40% | 82.0 | 1,448,905 + 5,140 |
| ele-fashion | C0_raw | 87.32% ± 0.07% | 74.14% ± 1.11% | 169.3 | 1,186,741 + 3,084 |
| ele-fashion | C1_effective | 86.85% ± 0.14% | 72.66% ± 0.72% | 88.7 | 1,186,741 + 3,084 |
| ele-fashion | C2_dual_shared_profile | 87.31% ± 0.09% | 73.63% ± 0.44% | 140.0 | 1,186,745 + 3,084 |
| ele-fashion | C3_dual_context_profile | 87.33% ± 0.11% | 73.75% ± 0.46% | 144.7 | 1,186,761 + 3,084 |

## Primary paired comparisons

| Comparison | Δ Accuracy | Δ Macro-F1 | Positive pairs (Accuracy / F1) |
|---|---:|---:|---:|
| C1 − C0: effective-only vs raw-only | -0.134 pp | -0.629 pp | 4/9 / 4/9 |
| C2 − C0: dual shared-profile vs raw-only | +0.007 pp | -0.054 pp | 6/9 / 4/9 |
| C2 − C1: raw context added to effective-only | +0.140 pp | +0.576 pp | 6/9 / 6/9 |
| C3 − C2: context-specific effective hop profile | +0.080 pp | -0.041 pp | 6/9 / 5/9 |
| C3 − C0: full action-space expansion vs raw-only | +0.087 pp | -0.095 pp | 6/9 / 5/9 |

Per-dataset deltas and positive-seed counts are in `data/paired_comparisons.csv`. No significance testing was performed.

## Semantic edge and operator diagnostics

| Dataset | Modality | Mean semantic cosine | p10–p90 | Mean relative raw/effective operator ΔL2 | Text/Visual effective ΔL2 |
|---|---|---:|---:|---:|---:|
| Movies | text | 0.9655 | 0.9274–0.9906 | 0.0062 | 1.8357 |
| Movies | visual | 0.6186 | 0.4672–0.7611 | 0.0321 | 1.8357 |
| Grocery | text | 0.9722 | 0.9446–0.9914 | 0.0048 | 2.1547 |
| Grocery | visual | 0.6443 | 0.4856–0.8285 | 0.0403 | 2.1547 |
| ele-fashion | text | 0.7119 | 0.4979–0.9004 | 0.0321 | 4.4872 |
| ele-fashion | visual | 0.7540 | 0.5685–0.9091 | 0.0279 | 4.4872 |

The edge-score cosine distribution, threshold fractions, normalized raw/effective sparse edge-vector norms/cosines, effective-degree quantiles, and text/visual operator discrepancies are recorded in `data/operator_diagnostics.csv`. No dense adjacency was constructed.

## Trajectory and action-space novelty

| Variant | Modality | Hop | Raw/effective cosine | Relative RMS delta | Raw RMS | Effective RMS |
|---|---|---:|---:|---:|---:|---:|
| C0_raw | text | 1 | 1.0000 | 0.0100 | 1.0000 | 1.0000 |
| C0_raw | text | 2 | 1.0000 | 0.0114 | 1.0000 | 1.0000 |
| C0_raw | text | 3 | 1.0000 | 0.0101 | 1.0000 | 1.0000 |
| C0_raw | text | 4 | 1.0000 | 0.0118 | 1.0000 | 1.0000 |
| C0_raw | visual | 1 | 1.0000 | 0.0234 | 1.0000 | 1.0000 |
| C0_raw | visual | 2 | 0.9999 | 0.0230 | 1.0000 | 1.0000 |
| C0_raw | visual | 3 | 0.9999 | 0.0228 | 1.0000 | 1.0000 |
| C0_raw | visual | 4 | 0.9999 | 0.0239 | 1.0000 | 1.0000 |
| C1_effective | text | 1 | 1.0000 | 0.0095 | 1.0000 | 1.0000 |
| C1_effective | text | 2 | 1.0000 | 0.0113 | 1.0000 | 1.0000 |
| C1_effective | text | 3 | 1.0000 | 0.0098 | 1.0000 | 1.0000 |
| C1_effective | text | 4 | 1.0000 | 0.0117 | 1.0000 | 1.0000 |
| C1_effective | visual | 1 | 1.0000 | 0.0227 | 1.0000 | 1.0000 |
| C1_effective | visual | 2 | 0.9999 | 0.0226 | 1.0000 | 1.0000 |
| C1_effective | visual | 3 | 0.9999 | 0.0223 | 1.0000 | 1.0000 |
| C1_effective | visual | 4 | 0.9999 | 0.0236 | 1.0000 | 1.0000 |
| C2_dual_shared_profile | text | 1 | 1.0000 | 0.0098 | 1.0000 | 1.0000 |
| C2_dual_shared_profile | text | 2 | 1.0000 | 0.0114 | 1.0000 | 1.0000 |
| C2_dual_shared_profile | text | 3 | 1.0000 | 0.0100 | 1.0000 | 1.0000 |
| C2_dual_shared_profile | text | 4 | 1.0000 | 0.0118 | 1.0000 | 1.0000 |
| C2_dual_shared_profile | visual | 1 | 1.0000 | 0.0232 | 1.0000 | 1.0000 |
| C2_dual_shared_profile | visual | 2 | 0.9999 | 0.0229 | 1.0000 | 1.0000 |
| C2_dual_shared_profile | visual | 3 | 0.9999 | 0.0227 | 1.0000 | 1.0000 |
| C2_dual_shared_profile | visual | 4 | 0.9999 | 0.0238 | 1.0000 | 1.0000 |
| C3_dual_context_profile | text | 1 | 1.0000 | 0.0099 | 1.0000 | 1.0000 |
| C3_dual_context_profile | text | 2 | 1.0000 | 0.0114 | 1.0000 | 1.0000 |
| C3_dual_context_profile | text | 3 | 1.0000 | 0.0100 | 1.0000 | 1.0000 |
| C3_dual_context_profile | text | 4 | 1.0000 | 0.0118 | 1.0000 | 1.0000 |
| C3_dual_context_profile | visual | 1 | 1.0000 | 0.0232 | 1.0000 | 1.0000 |
| C3_dual_context_profile | visual | 2 | 0.9999 | 0.0229 | 1.0000 | 1.0000 |
| C3_dual_context_profile | visual | 3 | 0.9999 | 0.0227 | 1.0000 | 1.0000 |
| C3_dual_context_profile | visual | 4 | 0.9999 | 0.0238 | 1.0000 | 1.0000 |

`action_space_novelty.csv` uses the C0-selected projector state to isolate operator-context span novelty. It reports effective-outside-raw and raw-outside-effective ratios plus Gram condition diagnostics, using 4×4 Gram identities rather than explicit projection residuals.

- text: mean effective-outside-raw-span ratio 0.0107; mean raw-outside-effective-span ratio 0.0108.
- visual: mean effective-outside-raw-span ratio 0.0224; mean raw-outside-effective-span ratio 0.0225.

## Context mixing, alpha profiles, and functional experts

- C2_dual_shared_profile learned λ mean/range by expert: E0: 0.1017 [0.0966, 0.1064]; E1: 0.0978 [0.0964, 0.0993]; E2: 0.1006 [0.0985, 0.1038]; E3: 0.0979 [0.0943, 0.1007].
- C3_dual_context_profile learned λ mean/range by expert: E0: 0.1019 [0.0972, 0.1067]; E1: 0.0976 [0.0949, 0.0992]; E2: 0.1003 [0.0985, 0.1028]; E3: 0.0979 [0.0914, 0.1017].
- C3 raw/effective alpha cosine mean: 1.0000; raw-alpha drift from initialization: 0.0810; effective-alpha drift: 0.0806.
- Expert output cosine: flattened 0.4428; mean-node 0.3675. C2/C3 raw/effective pre-transform profile cosine mean: 0.9998.
- Full per-expert λ, RMS contributions, alpha values/drifts, within-bank alpha cosine, and output/profile pair similarities are in `data/context_mix_diagnostics.csv`, `data/expert_profiles.csv`, and `data/expert_similarity.csv`.

## Static routing and correction scale

| Variant | Most common Text pair | Most common Visual pair | Text/Visual same-pair rate | Mean expert union | Runs with dead expert slot |
|---|---|---|---:|---:|---:|
| C0_raw | 0-3 (3/9) | 0-3 (3/9) | 0.111 | 3.111/4 | 7/9 |
| C1_effective | 1-3 (3/9) | 0-1 (5/9) | 0.333 | 2.778/4 | 8/9 |
| C2_dual_shared_profile | 0-1 (4/9) | 0-1 (5/9) | 0.333 | 2.778/4 | 8/9 |
| C3_dual_context_profile | 0-3 (5/9) | 0-1 (5/9) | 0.333 | 2.778/4 | 8/9 |
- C0_raw correction RMS/prior RMS ratio mean: 0.5047; scaled correction RMS 0.5029; prior RMS 0.9975.
- C1_effective correction RMS/prior RMS ratio mean: 0.5223; scaled correction RMS 0.5204; prior RMS 0.9973.
- C2_dual_shared_profile correction RMS/prior RMS ratio mean: 0.5180; scaled correction RMS 0.5161; prior RMS 0.9973.
- C3_dual_context_profile correction RMS/prior RMS ratio mean: 0.5153; scaled correction RMS 0.5134; prior RMS 0.9972.

Static unused experts are sparse global compositions, not node-router collapse. Per-checkpoint routes, dense probabilities, Top-2 weights, strength and correction RMS are in `data/routing_diagnostics.csv` and `data/strength_diagnostics.csv`.

## Historical C0 regression against V2.2 R0

Across nine matched dataset-seed runs, C0−old R0 mean deltas were Accuracy -0.071 pp (2/9 positive; range -0.420 to +0.150; direction mixed or zero) and Macro-F1 -0.337 pp (0/9 positive; range -1.143 to +0.000; direction mixed or zero); best-epoch difference mean -4.89. The mixed-or-zero directions show no uniform signed shift; this is descriptive and not a significance claim. Compatible historical/new checkpoint state-to-output audits exact: 8/9 historical states and 7/9 new C0 states. At atol=1e-7, allclose counts were 8/9 historical and 7/9 new C0; maximum z differences were 1.19e-06 historical and 1.43e-06 new C0. Any failed exact/allclose pair is reported in the CSV rather than suppressing the analysis. See `data/historical_c0_regression.csv` for each matched pair and maximum output difference.

## Frozen interpretation rules and decision map

Stable-positive means positive overall Accuracy and Macro-F1 deltas, at least 6/9 positive Accuracy pairs, and positive Accuracy mean on at least 2/3 datasets. Approximately equal means |Δ Accuracy| ≤ 0.15 pp and |Δ Macro-F1| ≤ 0.50 pp. These are descriptive labels, not significance or equivalence tests.

- **A. Not observed.** C1−C0 is -0.134/-0.629 pp Accuracy/Macro-F1 with 4/9 positive Accuracy pairs. Effective-only does not meet the stable-positive gate as a standalone replacement.
- **B. Not observed.** C1−C0 does not pass stable-positive; C2−C0 is +0.007/-0.054 pp. The rule requiring C1 not to pass and C2 to pass is not met.
- **C. Not observed.** C2−C0 stable-positive=False; C3−C2 approximate-equal=True. Both parts of the C2-positive/C3-near-equal pattern are not satisfied together.
- **D. Not observed.** C3−C2 stable-positive=False; context-specific multi-hop profiles are not supported by a stable-positive C3−C2 result.
- **E. Not observed.** C2/C3 near-equality to C0=True; maximum operator relative ΔL2=0.04026; maximum C0 effective-outside-raw span ratio=0.02511. The full near-equal/near-zero operator-and-span pattern is not present.
- **F. Observed.** C2/C3 stable-positive against C0=False/False; maximum operator relative ΔL2=0.0403; maximum span novelty=0.0251. The contexts differ, but this NC screen does not show task need; do not respond by sharpening similarity.
- **G. Not observed.** C2/C3 learned λ range is 0.0914–0.1067 (initial 0.10); not all learned values are near zero under the frozen ≤0.05 descriptive flag.
- **H. Not observed.** Maximum within-checkpoint expert λ range=0.0153; positive C2/C3-vs-C0 gate=False. The rule of differentiated expert mixing plus positive performance is not established.
- **I. Not observed.** C3−C2 stable-positive=False; mean raw/effective alpha cosine=1.0000; mean effective-alpha drift=0.0806. Performance and alpha separation do not jointly establish context-specific multi-hop response.

## V3B gate and boundary

C2 stable-positive vs C0: False; C3 stable-positive vs C0: False. The V3B gate is not met; stop effective-context screen work here.

No V3B or cross-modal operator was implemented. The observed comparisons do not establish statistical significance or causal mechanism.

## Reproducibility artifacts

- `data/environment.json`, `data/preflight_operator_diagnostics.csv`, `data/preflight_summary.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`
- `data/summary.csv`, `data/paired_comparisons.csv`, `data/operator_diagnostics.csv`, `data/trajectory_diagnostics.csv`, `data/action_space_novelty.csv`
- `data/context_mix_diagnostics.csv`, `data/routing_diagnostics.csv`, `data/strength_diagnostics.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/historical_c0_regression.csv`
- Checkpoints, logs, raw features, raw edge tensors and caches remain under ignored server-local `outputs/`.
