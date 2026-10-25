# MR-CGE V1.2 Multi-seed Confirmation

Status: **18/18 new formal cells complete**. The seed-42 V1.2 rows remain read-only historical evidence. New training uses the frozen V1.2 model, trainer, configuration, data and seed-42 split; no V1.3 changes were made.

## Cell accounting

| Completed cells | Failed cells | Resource-blocked cells | Unstarted cells | GPU preflights pass / blocked / failed | New checkpoint pairs reverified |
| --- | --- | --- | --- | --- | --- |
| 18 | 0 | 0 | 0 | 36 / 0 / 0 | 18 |

This report uses Best Validation Accuracy checkpoints for its primary Accuracy/Macro-F1/CE values. Best-F1 checkpoint values remain in their separate columns and are not mixed into the primary table. Test evaluation is disabled throughout.

## Three-seed M summary

Values are mean ± sample SD across available seeds 42, 43 and 44, shown as Accuracy % / Macro-F1 %. Missing seeds are not imputed.

| Dataset | M_repaired Acc / F1 | M_anchor Acc / F1 |
| --- | --- | --- |
| Movies | 55.079 ± 0.279 / 47.445 ± 0.841 | 55.589 ± 0.125 / 47.318 ± 1.175 |
| Grocery | 82.645 ± 0.244 / 76.336 ± 0.093 | 82.704 ± 0.212 / 75.970 ± 0.407 |
| Toys | 79.367 ± 0.147 / 76.329 ± 0.486 | 79.303 ± 0.328 / 76.537 ± 0.344 |
| ele-fashion | 87.368 ± 0.072 / 73.772 ± 0.373 | 87.372 ± 0.180 / 74.071 ± 0.977 |

Mean Validation CE ± sample SD:

| Dataset | M_repaired CE | M_anchor CE |
| --- | --- | --- |
| Movies | 1.4119 ± 0.0348 | 1.3867 ± 0.0344 |
| Grocery | 0.7004 ± 0.0076 | 0.6902 ± 0.0185 |
| Toys | 0.7203 ± 0.0027 | 0.7201 ± 0.0157 |
| ele-fashion | 0.4101 ± 0.0051 | 0.4077 ± 0.0097 |

## Paired M_anchor − M_repaired differences

Accuracy and Macro-F1 deltas are percentage points. The `positive pairs` count reports seeds with a positive anchor-minus-repaired delta. These are descriptive paired differences; three seeds do not establish statistical significance.

| Dataset | Acc mean Δ pp | Acc range | Acc + / − seeds | F1 mean Δ pp | F1 range | F1 + / − seeds | CE mean Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Movies | 0.510 | 0.330 … 0.780 | +3 / −0 | -0.127 | -2.057 … 1.924 | +1 / −2 | -0.025 |
| Grocery | 0.059 | -0.029 … 0.117 | +2 / −1 | -0.365 | -0.752 … 0.111 | +1 / −2 | -0.010 |
| Toys | -0.064 | -0.242 … 0.145 | +1 / −2 | 0.208 | -0.036 … 0.356 | +2 / −1 | -0.000 |
| ele-fashion | 0.003 | -0.113 … 0.102 | +2 / −1 | 0.298 | -0.190 … 1.000 | +2 / −1 | -0.002 |

Equal-weighted mean of available dataset-level paired deltas: Accuracy **0.127 pp**; Macro-F1 **0.004 pp**. This is descriptive and does not determine the model choice.

## ele-fashion seed-42 S/F/M comparison

| Condition | Val Accuracy % | Val Macro-F1 % | Val CE | Source | Accuracy checkpoint SHA verified |
| --- | --- | --- | --- | --- | --- |
| S_repaired | 87.399 | 74.248 | 0.43403297662734985 | confirmation_new_run | True |
| F_sinit | 87.358 | 73.716 | 0.4438895583152771 | confirmation_new_run | True |
| M_repaired | 87.368 | 73.498 | 0.4130326509475708 | V1.2_seed42_history | True |
| M_anchor | 87.389 | 73.584 | 0.4062482714653015 | V1.2_seed42_history | True |

## Historical H2 / DiP context

| Dataset | H2 reference | H2 Acc % | H2 F1 % | DiP Acc % | DiP F1 % |
| --- | --- | --- | --- | --- | --- |
| Movies | H2_hybrid_plus | 56.779 | 49.780 | 56.569 | 48.330 |
| Grocery | H2_hybrid_plus | 83.856 | 75.972 | 83.953 | 76.899 |
| Toys | H2_physical_3hop (historical 4-hop: 80.325 / 77.472) | 80.269 | 77.052 | 80.317 | 77.287 |
| ele-fashion | H2_hybrid_plus | 87.719 | 74.653 | 88.033 | 76.171 |

These are cross-campaign reference means, not matched controls. H2/DiP runs came from other campaigns; some baseline implementations were adapted to the shared protocol. They do not support SOTA or significance claims. Source records: `research/hybrid_cge_v11/PILOT_REPORT.md`, `research/hybrid_cge_paper_evidence_v1/BASELINE_REPORT.md`, and its `data/condition_summary.csv`.

## Expert and view diagnostics

The descriptive summary contains 80 dataset/condition/view rows. Rows with a zero Top-2 share in at least one selected checkpoint/view: 0. `expert_route_paired_differences.csv` reports same-seed Top-2 share changes; 17 of 64 fully paired dataset/view/expert combinations keep the same direction across all three seeds. `response_comparison.csv` and `expert_view_summary.csv` retain expert theta, effective coefficients, physical/semantic response RMS, rho/gamma, Early Physical Cross, and Late selector/residual summaries where recorded. These are descriptive diagnostics and do not identify causal contributions.

## Protocol and limits

- Fixed baseline commit: `712bf477085b8e9022a746330e3ae48e12fa0841`. Confirmation branch: `exp/mrcge_v12_multiseed_confirmation`.
- Seeds 43/44 use the existing seed-42 splits; observed split fingerprints: `04d43160594823ab35d62b1bde6547225dc816405e55eb1f9c44790eabfae0c5, 2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d, 7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056, bc29ceb8378cd130a5e8bd7e2c4630e20cf65915c5028b033bc2cab594d80b93`.
- The paired initialization audit compares actual state tensors: M conditions differ only at `filter_bank.theta[0]`; Fashion S/F states and initial logits agree within the recorded tolerance.
- Checkpoint SHA, reload metrics, model/trainer/config/data provenance, Validation RNG, and Test-off are recorded per completed cell. Historical checkpoint availability is reported separately.
- No expert collaboration causal claim, statistical significance claim, Test metric, or V1.3 experiment is made.

## Raw evidence

`data/new_runs.csv`, `data/combined_selected_runs.csv`, `data/epoch_history.csv`, `data/paired_differences.csv`, `data/dataset_summary.csv`, `data/ele_fashion_comparison.csv`, `data/expert_view_summary.csv`, `data/expert_route_paired_differences.csv`, `data/response_comparison.csv`, `data/resources.csv`, `data/preflight.json`, `data/failures.json`, and `data/manifest.json`.
