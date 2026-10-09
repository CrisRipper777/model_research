# MvCGE-MAG V3C: Functional Role Action-Space Screen

## Protocol and provenance

- Parent `075b12b497d619d0b86d25f264fa97402ac74778`; validation-only `unified_full_graph_nc_v1`; Movies, Grocery, `ele-fashion`; seeds 42–44; four variants; 36 runs.
- F0 follows V3A C0 / V2.2 R0 construction and output. Every variant uses modality-static Top-2 selection and strength, four shared experts, and one shared alpha profile.
- Supportive and Discrepant roles use detached raw modality features and observed physical edges only. Role edge weights are masks of the raw normalized operator; no channel-specific renormalization or new edges.
- `task.evaluate_test=false`; checkpoints selected by Validation Accuracy. No Test metrics, labels in preflight, HPO, LP, significance testing, learned edge router, node routing, or cross-modal operators.
- Discrepant is a fixed local-relative semantic-consistency label; it does not assert ground-truth heterophily.

## Validation results

Run-level means ± population standard deviations; paired differences below are descriptive percentage points.

| Dataset | Variant | Val Accuracy mean ± std | Val Macro-F1 mean ± std | Best epoch mean | Model parameters |
|---|---|---:|---:|---:|---:|
| Movies | F0_raw | 55.97% ± 0.45% | 48.03% ± 1.64% | 72.0 | 1448885 |
| Movies | F1_support | 54.08% ± 0.34% | 45.11% ± 0.81% | 56.7 | 1448885 |
| Movies | F2_role_dual_smooth | 54.79% ± 0.31% | 45.95% ± 1.19% | 60.7 | 1448889 |
| Movies | F3_role_functional | 54.50% ± 0.11% | 45.52% ± 0.92% | 54.0 | 1448889 |
| Grocery | F0_raw | 83.10% ± 0.33% | 75.82% ± 1.39% | 67.3 | 1448885 |
| Grocery | F1_support | 82.29% ± 0.45% | 75.14% ± 1.31% | 73.3 | 1448885 |
| Grocery | F2_role_dual_smooth | 82.63% ± 0.39% | 75.09% ± 1.71% | 70.3 | 1448889 |
| Grocery | F3_role_functional | 82.40% ± 0.40% | 75.17% ± 0.87% | 80.3 | 1448889 |
| ele-fashion | F0_raw | 87.36% ± 0.09% | 74.13% ± 1.04% | 171.0 | 1186741 |
| ele-fashion | F1_support | 87.30% ± 0.09% | 74.29% ± 0.49% | 160.0 | 1186741 |
| ele-fashion | F2_role_dual_smooth | 87.34% ± 0.12% | 74.86% ± 0.54% | 166.3 | 1186745 |
| ele-fashion | F3_role_functional | 87.17% ± 0.12% | 73.79% ± 0.89% | 141.3 | 1186745 |
| ALL | F0_raw | 75.48% ± 13.91% | 65.99% ± 12.79% | 103.4 | 1361504 |
| ALL | F1_support | 74.56% ± 14.63% | 64.85% ± 13.99% | 96.7 | 1361504 |
| ALL | F2_role_dual_smooth | 74.92% ± 14.37% | 65.30% ± 13.74% | 99.1 | 1361508 |
| ALL | F3_role_functional | 74.69% ± 14.41% | 64.83% ± 13.69% | 91.9 | 1361508 |

## Required paired comparisons

| Comparison | Δ Accuracy | Δ Macro-F1 | Positive pairs (Accuracy / Macro-F1) |
|---|---:|---:|---:|
| F1 − F0: Supportive-only smoothing vs raw physical context | -0.919 pp | -1.144 pp | 1/9 / 2/9 |
| F2 − F1: add Discrepant ordinary smoothing | +0.360 pp | +0.452 pp | 7/9 / 6/9 |
| F2 − F0: dual-role smoothing vs raw context | -0.560 pp | -0.693 pp | 2/9 / 2/9 |
| F3 − F2: signed difference vs Discrepant smoothing | -0.226 pp | -0.471 pp | 0/9 / 3/9 |
| F3 − F1: functional role vs Supportive-only | +0.134 pp | -0.020 pp | 4/9 / 4/9 |
| F3 − F0: full functional-role action space vs raw | -0.786 pp | -1.164 pp | 1/9 / 1/9 |

Per-dataset means and positive-seed counts are in `data/paired_comparisons.csv`; no significance testing was performed.

## Fixed role partition and cross-modal diagnostics

Statistics below are from raw modality features and physical edges only. Local-μ summaries use physical-active nodes; node coverage is the fraction of physical-active nodes receiving at least one incoming edge in that role.

| Dataset | Modality | Support edges | Discrepant edges | Support node coverage | Discrepant node coverage | Support/raw weight L2 | Discrepant/raw weight L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | text | 0.584 | 0.416 | 0.903 | 0.819 | 0.823 | 0.568 |
| Movies | visual | 0.499 | 0.501 | 0.896 | 0.824 | 0.821 | 0.572 |
| Grocery | text | 0.572 | 0.428 | 0.886 | 0.772 | 0.820 | 0.572 |
| Grocery | visual | 0.469 | 0.531 | 0.858 | 0.805 | 0.796 | 0.605 |
| ele-fashion | text | 0.532 | 0.468 | 0.764 | 0.669 | 0.819 | 0.573 |
| ele-fashion | visual | 0.528 | 0.472 | 0.757 | 0.670 | 0.813 | 0.582 |

The same-edge Text/Visual role contingency, disagreement fraction, and Supportive/Discrepant mask Jaccard values are recorded in `data/role_partition_diagnostics.csv`. Partition maximum absolute error is checked against 1e-7. Role score, semantic cosine and local μ quantiles are in the same file.

## Trajectory comparisons

For each hop, cosine is computed on physical-active nodes; relative RMS delta uses the first named action family as denominator. Full per-checkpoint values are in `data/trajectory_diagnostics.csv`.

- F0_raw/text: six comparison families in order raw/support, raw/disc-smooth, raw/disc-signed, support/disc-smooth, support/disc-signed, disc-smooth/disc-signed; each value vector is hop 1→4. Mean cosine vectors: `[0.8922, 0.838, 0.7751, 0.6787, 0.7778, 0.7428, 0.678, 0.5938, 0.2511, 0.2502, 0.2896, 0.2365, 0.4131, 0.3966, 0.2325, 0.1639, 0.3495, 0.12, 0.2029, 0.0677, 0.0259, 0.0534, 0.0215, -0.0386]`; corresponding relative RMS-delta vectors: `[0.4539, 0.5508, 0.6494, 0.7738, 0.6359, 0.6807, 0.76, 0.8503, 1.1427, 1.1423, 1.1084, 1.1522, 1.0517, 1.0651, 1.2029, 1.2552, 1.1028, 1.2875, 1.2237, 1.3254, 1.3949, 1.3732, 1.3954, 1.4375]`.
- F0_raw/visual: six comparison families in order raw/support, raw/disc-smooth, raw/disc-signed, support/disc-smooth, support/disc-signed, disc-smooth/disc-signed; each value vector is hop 1→4. Mean cosine vectors: `[0.8774, 0.8041, 0.7226, 0.6204, 0.797, 0.7751, 0.7409, 0.6835, 0.1959, 0.2227, 0.2878, 0.2631, 0.4123, 0.3669, 0.2217, 0.1449, 0.2981, 0.0965, 0.1564, 0.0523, 0.0017, 0.0766, 0.0873, 0.0559]`; corresponding relative RMS-delta vectors: `[0.4824, 0.6053, 0.7176, 0.8379, 0.6091, 0.6357, 0.6832, 0.751, 1.1882, 1.1654, 1.1114, 1.133, 1.0607, 1.0999, 1.2206, 1.2793, 1.156, 1.3144, 1.2697, 1.3466, 1.4116, 1.3533, 1.3436, 1.3667]`.
- F1_support/text: six comparison families in order raw/support, raw/disc-smooth, raw/disc-signed, support/disc-smooth, support/disc-signed, disc-smooth/disc-signed; each value vector is hop 1→4. Mean cosine vectors: `[0.8948, 0.839, 0.7759, 0.6783, 0.7851, 0.748, 0.6846, 0.5991, 0.262, 0.2667, 0.3043, 0.2519, 0.429, 0.4004, 0.2412, 0.1671, 0.3522, 0.1353, 0.2081, 0.0757, 0.0484, 0.0822, 0.0507, -0.0104]`; corresponding relative RMS-delta vectors: `[0.4483, 0.5491, 0.6481, 0.7743, 0.6258, 0.6738, 0.7523, 0.8448, 1.1337, 1.1287, 1.0955, 1.1396, 1.0376, 1.0616, 1.196, 1.2528, 1.1001, 1.276, 1.2195, 1.3196, 1.3783, 1.3511, 1.3731, 1.4167]`.
- F1_support/visual: six comparison families in order raw/support, raw/disc-smooth, raw/disc-signed, support/disc-smooth, support/disc-signed, disc-smooth/disc-signed; each value vector is hop 1→4. Mean cosine vectors: `[0.8781, 0.8038, 0.7223, 0.6198, 0.7996, 0.7769, 0.7431, 0.6845, 0.1972, 0.2264, 0.2902, 0.2649, 0.4173, 0.3674, 0.2237, 0.1451, 0.2963, 0.0987, 0.1556, 0.0526, 0.0064, 0.0836, 0.0933, 0.06]`; corresponding relative RMS-delta vectors: `[0.481, 0.6057, 0.7181, 0.8387, 0.6059, 0.6339, 0.6813, 0.7506, 1.1874, 1.1629, 1.1098, 1.1322, 1.0565, 1.0996, 1.2191, 1.2792, 1.1574, 1.3128, 1.2703, 1.3464, 1.4083, 1.3486, 1.3398, 1.3646]`.
- F2_role_dual_smooth/text: six comparison families in order raw/support, raw/disc-smooth, raw/disc-signed, support/disc-smooth, support/disc-signed, disc-smooth/disc-signed; each value vector is hop 1→4. Mean cosine vectors: `[0.8949, 0.8393, 0.7763, 0.6786, 0.7849, 0.7482, 0.6847, 0.5993, 0.2614, 0.2665, 0.3047, 0.2524, 0.429, 0.4013, 0.242, 0.168, 0.3521, 0.1354, 0.2089, 0.0764, 0.0473, 0.0815, 0.0504, -0.0106]`; corresponding relative RMS-delta vectors: `[0.448, 0.5485, 0.6476, 0.774, 0.6259, 0.6735, 0.7521, 0.8446, 1.1343, 1.1287, 1.0952, 1.1391, 1.0374, 1.0607, 1.1953, 1.2521, 1.1004, 1.2759, 1.2189, 1.3191, 1.379, 1.3513, 1.3731, 1.4165]`.
- F2_role_dual_smooth/visual: six comparison families in order raw/support, raw/disc-smooth, raw/disc-signed, support/disc-smooth, support/disc-signed, disc-smooth/disc-signed; each value vector is hop 1→4. Mean cosine vectors: `[0.8771, 0.8042, 0.7235, 0.6224, 0.7961, 0.7737, 0.7388, 0.6803, 0.1944, 0.2206, 0.2843, 0.2583, 0.4102, 0.3648, 0.2198, 0.1436, 0.2956, 0.0951, 0.1542, 0.0509, 0.0008, 0.0738, 0.0822, 0.0486]`; corresponding relative RMS-delta vectors: `[0.4828, 0.6051, 0.7165, 0.8359, 0.6109, 0.6384, 0.6868, 0.7556, 1.1895, 1.1673, 1.1147, 1.1374, 1.0628, 1.1019, 1.2221, 1.2804, 1.1581, 1.3154, 1.2714, 1.3477, 1.4123, 1.3559, 1.3482, 1.373]`.
- F3_role_functional/text: six comparison families in order raw/support, raw/disc-smooth, raw/disc-signed, support/disc-smooth, support/disc-signed, disc-smooth/disc-signed; each value vector is hop 1→4. Mean cosine vectors: `[0.8971, 0.8401, 0.777, 0.6778, 0.7908, 0.7527, 0.6909, 0.6045, 0.2704, 0.2799, 0.3173, 0.2664, 0.4421, 0.4058, 0.2503, 0.1716, 0.3558, 0.1467, 0.2137, 0.0828, 0.0632, 0.1039, 0.0747, 0.0143]`; corresponding relative RMS-delta vectors: `[0.4433, 0.5471, 0.6465, 0.7749, 0.6176, 0.6675, 0.7449, 0.8391, 1.1269, 1.1178, 1.0843, 1.1278, 1.0252, 1.0567, 1.1887, 1.2494, 1.0968, 1.2674, 1.215, 1.3145, 1.3671, 1.3339, 1.3544, 1.3981]`.
- F3_role_functional/visual: six comparison families in order raw/support, raw/disc-smooth, raw/disc-signed, support/disc-smooth, support/disc-signed, disc-smooth/disc-signed; each value vector is hop 1→4. Mean cosine vectors: `[0.8786, 0.804, 0.7227, 0.6199, 0.8006, 0.7776, 0.744, 0.6852, 0.1994, 0.2291, 0.2927, 0.2675, 0.4198, 0.3681, 0.2253, 0.1457, 0.2974, 0.1006, 0.1562, 0.0536, 0.0095, 0.0878, 0.0979, 0.0647]`; corresponding relative RMS-delta vectors: `[0.48, 0.6054, 0.7175, 0.8386, 0.6045, 0.6332, 0.6803, 0.7501, 1.1857, 1.1607, 1.1076, 1.1301, 1.0541, 1.099, 1.2178, 1.2788, 1.1564, 1.3114, 1.2697, 1.3457, 1.4061, 1.3453, 1.3362, 1.3611]`.

## Action-space span novelty

Ratios use flattened selected-checkpoint trajectories and small Gram/pseudoinverse identities; no large projection matrix is formed. Ratios are summarized across nine dataset-seed checkpoints by variant and modality.

| Variant | Modality | support outside raw | discrepant smoothing outside raw | discrepant signed outside raw | dual-smooth union outside raw | functional union outside raw | signed outside dual-smooth |
|---|---|---:|---:|---:|---:|---:|---:|
| F0_raw | text | 0.5896 | 0.6977 | 0.7518 | 0.6427 | 0.6708 | 0.3221 |
| F0_raw | visual | 0.6325 | 0.6475 | 0.7347 | 0.6401 | 0.6833 | 0.3328 |
| F1_support | text | 0.5888 | 0.6926 | 0.7470 | 0.6397 | 0.6679 | 0.3227 |
| F1_support | visual | 0.6324 | 0.6464 | 0.7344 | 0.6394 | 0.6830 | 0.3309 |
| F2_role_dual_smooth | text | 0.5883 | 0.6923 | 0.7469 | 0.6393 | 0.6676 | 0.3221 |
| F2_role_dual_smooth | visual | 0.6317 | 0.6504 | 0.7367 | 0.6409 | 0.6838 | 0.3327 |
| F3_role_functional | text | 0.5877 | 0.6872 | 0.7414 | 0.6364 | 0.6644 | 0.3210 |
| F3_role_functional | visual | 0.6322 | 0.6458 | 0.7334 | 0.6389 | 0.6824 | 0.3300 |
Per-checkpoint novelty and Gram condition numbers for all variants are in `data/action_space_novelty.csv`.

## Role profiles, λ, experts and correction scale

- F2_role_dual_smooth λ mean/range by expert: E0: 0.1076 [0.1052, 0.1127]; E1: 0.1031 [0.0986, 0.1090]; E2: 0.1049 [0.1003, 0.1132]; E3: 0.1050 [0.0988, 0.1168]. Text/visual share the same λ parameter.
- F3_role_functional λ mean/range by expert: E0: 0.1043 [0.1019, 0.1063]; E1: 0.0954 [0.0893, 0.0987]; E2: 0.0974 [0.0939, 0.0996]; E3: 0.0964 [0.0896, 0.0996]. Text/visual share the same λ parameter.
- F0_raw correction/prior RMS ratio mean 0.4928; correction RMS 0.4910; prior RMS 0.9975; scaled-correction/prior cosine 0.5584.
- F1_support correction/prior RMS ratio mean 0.6489; correction RMS 0.6470; prior RMS 0.9977; scaled-correction/prior cosine 0.5542.
- F2_role_dual_smooth correction/prior RMS ratio mean 0.6903; correction RMS 0.6879; prior RMS 0.9972; scaled-correction/prior cosine 0.5866.
- F3_role_functional correction/prior RMS ratio mean 0.7409; correction RMS 0.7384; prior RMS 0.9973; scaled-correction/prior cosine 0.6343.
- Shared expert output pair cosine: flattened mean 0.3249; mean-node mean 0.4027. Alpha values/drifts, per-profile channel cosine and RMS, and per-pair similarities are in `data/expert_profiles.csv`, `data/role_profile_diagnostics.csv`, and `data/expert_similarity.csv`.

## Static routing

| Variant | Most common Text pair | Most common Visual pair | Same-pair rate | Mean expert union | Runs with a dead slot |
|---|---|---|---:|---:|---:|
| F0_raw | 1-3 (3/9) | 0-1 (4/9) | 0.111 | 3.222/4 | 6/9 |
| F1_support | 0-1 (3/9) | 0-1 (5/9) | 0.333 | 2.667/4 | 9/9 |
| F2_role_dual_smooth | 0-1 (6/9) | 0-1 (6/9) | 0.667 | 2.333/4 | 9/9 |
| F3_role_functional | 0-1 (6/9) | 0-1 (6/9) | 0.556 | 2.444/4 | 9/9 |
Static unused experts are sparse global compositions, not node-router collapse; per-checkpoint Top-2 weights, dense probabilities, and strength are in `data/routing_diagnostics.csv`.

## Historical F0 comparison against V3A C0

Across nine matched runs, F0−C0 mean deltas were Accuracy +0.047 pp (4/9 positive; range -0.120 to +0.360) and Macro-F1 -0.049 pp (4/9 positive; range -0.926 to +0.847). Mean best-epoch difference was -1.00. Shared state exact: 0/9; forward allclose at 1e-7: 0/9. Maximum z/aux differences were 2.01/0.000624. See `data/historical_f0_regression.csv` for each matched checkpoint.

## Frozen descriptive interpretation and next-stage gate

Stable-positive: overall Accuracy and Macro-F1 deltas both >0, at least 6/9 positive Accuracy pairs, and positive Accuracy means on at least 2/3 datasets. Approximately equal: absolute Accuracy delta ≤0.15 pp and absolute Macro-F1 delta ≤0.50 pp. For interpretation labels only, functional novelty ≤1e-3 is treated as near-zero; cross-modal disagreement >0.5 means a majority of physical edges change role. These are descriptive cutoffs, not statistical tests or tuning targets.

- **A: Not observed.** F1−F0 stable-positive=False; Supportive-only smoothing does not pass the stable-positive gate.
- **B: Not observed.** F1−F0 is below zero on both overall metrics=True; F2≈F0=False.
- **C: Not observed.** F3−F2 stable-positive=False; F3−F0 stable-positive=False.
- **D: Not observed.** F3−F2 stable-positive=False; F3−F0 stable-positive=False. If observed, prefer testing Raw + functional-role residual before node routing.
- **E: Not observed.** F2−F0 stable-positive=False; F3−F2 approximately equal or below on both metrics=True.
- **F: Observed.** F2/F3 stable-positive vs F0=False/False; mean functional-union novelty=0.6734. If observed, favor learned role assignment over static threshold changes.
- **G: Not observed.** Mean functional-union outside-raw-span ratio=0.6734; descriptive near-zero cutoff=0.001.
- **H: Not observed.** F3−F0 stable-positive=False; mean Text/Visual role disagreement=0.4328; majority-disagreement flag=False.

## Next-stage decision

F3 vs F0 stable-positive: False; functional action novelty near-zero: False. The next-stage gate is not met. Stop this screen here; do not automatically start node-conditioned routing, role learning, or context augmentation.

No ground-truth homophily/heterophily claim, causal claim, or significance claim is made.

## Reproducibility artifacts

- `data/environment.json`, `data/preflight_role_diagnostics.csv`, `data/preflight_summary.json`, `data/smoke_summary.json`, `data/campaign_manifest.json`, `data/run_rows.json`
- `data/summary.csv`, `data/paired_comparisons.csv`, `data/role_partition_diagnostics.csv`, `data/trajectory_diagnostics.csv`, `data/action_space_novelty.csv`
- `data/role_profile_diagnostics.csv`, `data/routing_diagnostics.csv`, `data/strength_diagnostics.csv`, `data/expert_profiles.csv`, `data/expert_similarity.csv`, `data/historical_f0_regression.csv`
- Checkpoints, training/Hydra logs, raw features and role masks remain under ignored server-local `outputs/`.
