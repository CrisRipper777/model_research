# PACE-MAG V1 first-round pilot report

## Scope and protocol

This is the first, validation-only pilot for the frozen implementation at commit `739f340bb6858b7280dede1dfaaf37740c2de3fa`. It contains the requested three datasets, seed 42, and all five variants: 15 runs total. All 15 completed; none failed. Each row records `evaluate_test=False`; Test labels and Test metrics were not used by training or the Full-checkpoint diagnostics.

The runs use `unified_full_graph_nc_v1`, the same frozen model configuration and task protocol, the existing official train/validation split and V7B semantic cache, AdamW, validation-accuracy checkpoint selection, and validation Macro-F1 as a parallel descriptive metric. Full configs and their SHA-256 hashes, source commit, cache fingerprints, parameter counts, finite checks, timing, and CUDA memory are recorded in `data/pilot_runs.csv`. The compact aggregation is `data/pilot_summary.csv`; per-run resource values are also in `data/resource_profile.csv`. Training logs and checkpoints remain in the ignored `outputs/pace_mag_v1_pilot/runs/` tree and are not committed.

## Run results

Accuracy and Macro-F1 are the raw validation metrics on a 0–1 scale. `Train s` is the recorded training time. CUDA memory is peak allocated / reserved GiB. Each cell has one seed, so these are observations rather than statistical comparisons.

| Dataset | Variant | Val Accuracy | Val Macro-F1 | Best epoch | Train s | Peak CUDA GiB (alloc / reserved) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Movies | E0 source-agnostic | 0.541392 | 0.465155 | 81 | 17.2 | 1.48 / 1.72 |
| Movies | E1 no cross-modal | 0.540792 | 0.459571 | 72 | 15.0 | 1.41 / 1.66 |
| Movies | E2 full | 0.542292 | 0.452725 | 88 | 17.9 | 1.48 / 1.72 |
| Movies | E3 single-expert control | 0.553389 | 0.431350 | 88 | 14.3 | 1.11 / 1.41 |
| Movies | E4 static routing | 0.542891 | 0.443252 | 88 | 16.7 | 1.23 / 1.51 |
| Grocery | E0 source-agnostic | 0.820498 | 0.748808 | 133 | 25.0 | 1.52 / 1.78 |
| Grocery | E1 no cross-modal | 0.819327 | 0.749017 | 129 | 24.0 | 1.45 / 1.72 |
| Grocery | E2 full | 0.817862 | 0.735134 | 91 | 18.6 | 1.52 / 1.78 |
| Grocery | E3 single-expert control | 0.821083 | 0.746410 | 87 | 14.4 | 1.13 / 1.40 |
| Grocery | E4 static routing | 0.820791 | 0.742957 | 180 | 30.8 | 1.27 / 1.52 |
| ele-fashion | E0 source-agnostic | 0.875626 | 0.745662 | 123 | 110.3 | 7.33 / 8.34 |
| ele-fashion | E1 no cross-modal | 0.876649 | 0.750326 | 134 | 115.2 | 6.94 / 7.95 |
| ele-fashion | E2 full | 0.876445 | 0.747244 | 135 | 119.1 | 7.33 / 8.35 |
| ele-fashion | E3 single-expert control | 0.878081 | 0.753353 | 142 | 97.3 | 4.93 / 5.71 |
| ele-fashion | E4 static routing | 0.879002 | 0.746930 | 137 | 112.4 | 5.88 / 6.85 |

The 15 recorded training times sum to 748.2 seconds; per-run training time ranges from 14.3 to 119.1 seconds. Peak allocated memory ranges from 1.11 to 7.33 GiB; peak reserved memory ranges from 1.40 to 8.35 GiB. All runs used the RTX 3090 at `cuda:1`. GPU 0 was left alone because it had a pre-existing VLLM workload.

## Descriptive comparison

- E3 had the highest validation Accuracy in all three one-seed dataset comparisons. Macro-F1 winners differed: E0 on Movies, E1 on Grocery, and E3 on ele-fashion. On Movies, E3's higher Accuracy coincided with the lowest Macro-F1 among the five variants.
- Compared with E1, E2's Accuracy changed by +0.15 percentage points on Movies, −0.15 points on Grocery, and −0.02 points on ele-fashion. Its Macro-F1 changed by −0.68, −1.39, and −0.31 points, respectively. This pilot does not show a consistent validation benefit from enabling cross-modal messages.
- E2 was not the top Accuracy or Macro-F1 variant on any of the three datasets. E4 and E3 comparisons are also mixed by metric and dataset; these observations do not establish that dynamic routing or multiple experts are unnecessary. In particular, E3 is a simple control and does not test every possible expert design.
- E3's single MLP uses bottleneck width 246. Its single-network processing parameter count is 157,814 versus the 157,611 target for the Full expert-plus-router processing block, a difference of +203 parameters (+0.13%). The exact per-run counts and signed delta are in the CSV.

These are single-seed validation observations with no uncertainty estimate or significance claim. The official Test set remains untouched. They are not sufficient to select a final method or to infer generalization beyond these splits.

## Full E2 checkpoint diagnostics

`data/diagnostics/{Movies,Grocery,ele-fashion}_seed_42_e2_full.json` contains the full label-free diagnostics for each selected E2 validation checkpoint. The analyzer used the checkpoint chosen by validation Accuracy and only node Text/Visual features and physical/candidate graph topology. It did not load NC split files or labels; each JSON records `diagnostic_label_access=false`, `test_labels_loaded=false`, and `test_metrics_computed=false`.

- Effective source-node fractions were 1.0 for Text and Visual views on all three datasets. Physical was 1.0 on Movies and Grocery and 0.999969 on ele-fashion.
- Learned `rho` values stayed near their 0.1 initialization: across the four target/source combinations and datasets, 0.0997–0.1079. Learned `beta_phy` ranged 0.510–0.526 and `beta_sem` 0.298–0.321. The fixed `gamma` was 0.2.
- Physical response RMS was 0.375–0.596 for Text targets and 0.522–0.739 for Visual targets. Semantic response RMS was 0.330–0.586 for Text and 0.475–0.618 for Visual. All final embeddings were finite; their RMS was 1.051–1.058. `aux_loss` was zero.
- Across the relation/modality views, expert-output RMS ranged 0.426–0.806 (Movies), 0.566–0.912 (Grocery), and 0.574–1.069 (ele-fashion). Pairwise expert-output cosine ranged −0.025–0.166, −0.074–0.168, and −0.128–0.146, respectively. The mean router probabilities varied by view and dataset rather than remaining uniform: across views their component values ranged 0.130–0.734, 0.208–0.549, and 0.049–0.866. Mean router entropy ranged 0.625–0.845, 0.559–0.891, and 0.359–0.588, respectively.

The learned coefficients, nonzero response magnitudes, differentiated expert outputs, and nonuniform routing show that the corresponding computation paths were active at these checkpoints. They do not demonstrate that these mechanisms improved prediction quality. The E1-vs-E2 retrained pilot comparison is the relevant available cross-modal control here; no frozen-checkpoint intervention was run.

## Engineering and credibility notes

All 15 rows have finite validation metrics and model parameters, no error text, and CUDA peak telemetry. E1's pilot gradient summaries show no task gradient in the disabled cross-modal modules or `rho`; the implementation's unit audit checks these parameters do not change under AdamW. The real-graph Movies E2 preflight also completed forward, backward, AdamW update, save/reload, and inference, with finite nonzero gradients and parameter updates in the intended modules. See `IMPLEMENTATION_AUDIT.md`, `data/implementation_checks.json`, and `data/gradient_audit.json`.

No run failed or ran out of memory. During setup, standalone GPU-management probes intermittently failed to initialize while the active training process continued normally; all pilot jobs completed and recorded per-run PyTorch CUDA telemetry. This did not alter device selection or the saved run results. The checkpoint library emitted a CPU autocast deprecation warning during training; it did not affect completion or metrics.

The model and task behavior match the frozen design; the explicit limits remain that this is Full-Graph NC only, `PT + PV = PU` is an operator-level identity, and row-quality normalization plus nonlinear experts do not imply whole-model linear reconstruction. For the next research decision, review these pilot signals and the frozen implementation before deciding whether to define a confirmatory multi-seed protocol. Do not treat this pilot as formal evidence or as an automatic tuning signal.
