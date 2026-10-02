# PIGPR-C0 — Prior-Anchored GPR Backbone Audit

## Execution and protocol

- Required base: `2fca8b31dc85edb8c28f9aa4066318b1e8557ad3`; branch: `exp/pigpr_c0_prior_anchored_backbone_audit`; campaign contains 54 run records (54 complete).
- NC uses the fixed dataset split and training seeds 42/43/44. All runs used `evaluate_test=false` and `development_no_test=true`; there are no `test_*` metrics in the run records.
- The six same-seed RU repeatability runs, focused RU regression gate, full repository test suite, six NC smoke runs, and two LP smoke runs are recorded in `smoke_status.json`.
- LP smoke was limited to sports-copurchase, AGP/AGD, two epochs and two training batches with `evaluate_test=false`. The existing LP loader reads the frozen edge split to filter held-out positive message edges; no LP test metric was evaluated.
- No NC test evaluation, formal LP, secondary dataset, or hyperparameter change was performed.

## Validation performance

Accuracy and macro-F1 are percentages. Entries are mean ± population SD across three model seeds; CE is mean ± population SD.

| Dataset | Variant | Val Acc % | Macro-F1 % | Val CE | Seeds |
| --- | --- | --- | --- | --- | --- |
| Movies | PO | 51.49 ± 0.03 | 40.70 ± 0.55 | 1.4630 ± 0.0056 | 3 |
| Movies | RU | 55.61 ± 0.26 | 47.57 ± 0.73 | 1.4394 ± 0.0527 | 3 |
| Movies | AU | 54.94 ± 0.20 | 46.47 ± 0.43 | 1.4212 ± 0.0042 | 3 |
| Movies | AP | 54.69 ± 0.23 | 46.09 ± 0.52 | 1.4333 ± 0.0149 | 3 |
| Movies | AGP | 55.05 ± 0.21 | 46.83 ± 0.39 | 1.4860 ± 0.0372 | 3 |
| Movies | AGD | 56.52 ± 0.18 | 48.83 ± 1.31 | 1.3501 ± 0.0130 | 3 |
| Grocery | PO | 77.55 ± 0.13 | 70.22 ± 0.84 | 0.8524 ± 0.0164 | 3 |
| Grocery | RU | 81.73 ± 0.13 | 75.38 ± 0.30 | 0.7015 ± 0.0198 | 3 |
| Grocery | AU | 81.35 ± 0.09 | 74.98 ± 0.44 | 0.7115 ± 0.0270 | 3 |
| Grocery | AP | 81.35 ± 0.13 | 74.89 ± 0.62 | 0.7178 ± 0.0186 | 3 |
| Grocery | AGP | 81.60 ± 0.10 | 75.33 ± 0.05 | 0.7096 ± 0.0265 | 3 |
| Grocery | AGD | 83.05 ± 0.09 | 77.24 ± 0.61 | 0.6836 ± 0.0287 | 3 |
| ele-fashion | PO | 87.06 ± 0.13 | 73.55 ± 0.59 | 0.4443 ± 0.0250 | 3 |
| ele-fashion | RU | 87.44 ± 0.06 | 74.44 ± 0.56 | 0.4108 ± 0.0155 | 3 |
| ele-fashion | AU | 87.42 ± 0.09 | 74.37 ± 0.51 | 0.4157 ± 0.0051 | 3 |
| ele-fashion | AP | 87.39 ± 0.01 | 74.26 ± 0.25 | 0.4105 ± 0.0096 | 3 |
| ele-fashion | AGP | 87.36 ± 0.08 | 74.20 ± 0.77 | 0.4186 ± 0.0121 | 3 |
| ele-fashion | AGD | 87.33 ± 0.19 | 74.22 ± 0.20 | 0.4106 ± 0.0054 | 3 |

## Paired comparisons

Accuracy and macro-F1 deltas are percentage points. The sign column is positive / negative / tie model seeds out of three. These are paired fixed-split model-seed comparisons, not pseudo-IID tests.

| Dataset | Comparison | Δ Acc pp | +/-/tie | Δ F1 pp | Δ CE |
| --- | --- | --- | --- | --- | --- |
| Movies | RU-PO | 4.119 | 3/0/0 | 6.872 | -0.0236 |
| Movies | AU-RU | -0.670 | 0/3/0 | -1.100 | -0.0182 |
| Movies | AP-AU | -0.250 | 1/2/0 | -0.382 | 0.0122 |
| Movies | AGP-AP | 0.360 | 2/1/0 | 0.740 | 0.0527 |
| Movies | AGP-AGD | -1.470 | 0/3/0 | -1.996 | 0.1359 |
| Movies | AGP-RU | -0.560 | 0/3/0 | -0.743 | 0.0466 |
| Movies | AP-RU | -0.920 | 0/3/0 | -1.482 | -0.0061 |
| Movies | AGD-RU | 0.910 | 3/0/0 | 1.253 | -0.0893 |
| Grocery | RU-PO | 4.178 | 3/0/0 | 5.157 | -0.1510 |
| Grocery | AU-RU | -0.381 | 0/3/0 | -0.394 | 0.0100 |
| Grocery | AP-AU | -0.000 | 2/1/0 | -0.097 | 0.0063 |
| Grocery | AGP-AP | 0.254 | 3/0/0 | 0.442 | -0.0082 |
| Grocery | AGP-AGD | -1.445 | 0/3/0 | -1.914 | 0.0260 |
| Grocery | AGP-RU | -0.127 | 1/2/0 | -0.050 | 0.0081 |
| Grocery | AP-RU | -0.381 | 0/3/0 | -0.492 | 0.0163 |
| Grocery | AGD-RU | 1.318 | 3/0/0 | 1.865 | -0.0179 |
| ele-fashion | RU-PO | 0.378 | 3/0/0 | 0.899 | -0.0335 |
| ele-fashion | AU-RU | -0.014 | 1/1/1 | -0.077 | 0.0049 |
| ele-fashion | AP-AU | -0.038 | 1/1/1 | -0.113 | -0.0052 |
| ele-fashion | AGP-AP | -0.024 | 1/2/0 | -0.055 | 0.0081 |
| ele-fashion | AGP-AGD | 0.027 | 1/2/0 | -0.020 | 0.0081 |
| ele-fashion | AGP-RU | -0.075 | 0/3/0 | -0.244 | 0.0078 |
| ele-fashion | AP-RU | -0.051 | 1/2/0 | -0.190 | -0.0003 |
| ele-fashion | AGD-RU | -0.102 | 2/1/0 | -0.225 | -0.0002 |

- **RU-PO:** dataset mean deltas (pp) Movies +4.119 (3/3 seeds positive), Grocery +4.178 (3/3 seeds positive), ele-fashion +0.378 (3/3 seeds positive).
- **AU-RU:** dataset mean deltas (pp) Movies -0.670 (0/3 seeds positive), Grocery -0.381 (0/3 seeds positive), ele-fashion -0.014 (1/3 seeds positive).
- **AP-AU:** dataset mean deltas (pp) Movies -0.250 (1/3 seeds positive), Grocery -0.000 (2/3 seeds positive), ele-fashion -0.038 (1/3 seeds positive).
- **AGP-AP:** dataset mean deltas (pp) Movies +0.360 (2/3 seeds positive), Grocery +0.254 (3/3 seeds positive), ele-fashion -0.024 (1/3 seeds positive).
- **AGP-AGD:** dataset mean deltas (pp) Movies -1.470 (0/3 seeds positive), Grocery -1.445 (0/3 seeds positive), ele-fashion +0.027 (1/3 seeds positive).

## Same-seed repeatability floor

These runs hold dataset, split, variant RU, and seed 42 fixed while repeating execution. SD is population SD and range is max-min.

| Dataset | Metric | Mean | Population SD | Range |
| --- | --- | --- | --- | --- |
| Movies | val_accuracy | 55.70886 pp | 0.06163 pp | 0.14997 pp |
| Movies | val_macro_f1 | 46.91293 pp | 0.68408 pp | 1.62695 pp |
| Movies | val_ce | 1.44595  | 0.03475  | 0.07768  |
| Movies | best_epoch | 104.00000  | 9.93311  | 22.00000  |
| Grocery | val_accuracy | 81.60078 pp | 0.01380 pp | 0.02928 pp |
| Grocery | val_macro_f1 | 75.05174 pp | 0.02028 pp | 0.04302 pp |
| Grocery | val_ce | 0.68232  | 0.00000  | 0.00001  |
| Grocery | best_epoch | 71.00000  | 0.00000  | 0.00000  |

## Anchoring, drift, and semantic retention

For each checkpoint and modality, raw and anchored states were recomputed from the same projected prior and physical graph operator. Drift is `1 - mean_node_cosine(P, state)`; it describes representation movement, not task quality.

| Dataset | Variant | Raw H3 drift | Anchored S3 drift | Anchored − raw |
| --- | --- | --- | --- | --- |
| Movies | PO | 0.12169 | 0.08410 | -0.03759 |
| Movies | RU | 0.18873 | 0.12452 | -0.06421 |
| Movies | AU | 0.18558 | 0.12329 | -0.06229 |
| Movies | AP | 0.18232 | 0.12169 | -0.06063 |
| Movies | AGP | 0.19677 | 0.13005 | -0.06673 |
| Movies | AGD | 0.18616 | 0.12206 | -0.06410 |
| Grocery | PO | 0.17470 | 0.11368 | -0.06102 |
| Grocery | RU | 0.18649 | 0.12055 | -0.06593 |
| Grocery | AU | 0.17830 | 0.11597 | -0.06233 |
| Grocery | AP | 0.18846 | 0.12178 | -0.06667 |
| Grocery | AGP | 0.18473 | 0.11967 | -0.06507 |
| Grocery | AGD | 0.22916 | 0.14145 | -0.08772 |
| ele-fashion | PO | 0.12626 | 0.07892 | -0.04734 |
| ele-fashion | RU | 0.11617 | 0.07295 | -0.04322 |
| ele-fashion | AU | 0.11741 | 0.07377 | -0.04364 |
| ele-fashion | AP | 0.11813 | 0.07409 | -0.04405 |
| ele-fashion | AGP | 0.12188 | 0.07637 | -0.04551 |
| ele-fashion | AGD | 0.11983 | 0.07456 | -0.04528 |

Across datasets, order-3 anchored state drift is lower than raw order-3 drift: 3/3 datasets. `semantic_drift.csv` also contains each order, each proposal, and each pre-normalization output. `propagation_diagnostics.csv` contains per-modality lambda, RMS, proposal delta, and cosines.

For explicit protection, the retrained checkpoint means are:

| Dataset | AGP cos(P,Zpre) | AGD cos(P,Zpre) | AGP ||Zpre-P||/||P|| | AGD ||Zpre-P||/||P|| | AGP lambda |
| --- | --- | --- | --- | --- | --- |
| Movies | 0.98093 | 0.94014 | 0.25747 | 0.48155 | 0.4962 |
| Grocery | 0.98101 | 0.92396 | 0.26308 | 0.52771 | 0.5094 |
| ele-fashion | 0.99267 | 0.97191 | 0.22454 | 0.43247 | 0.4905 |

AGP's pre-norm output is closer to `P` by both cosine and relative displacement on all three datasets. This is the intended mechanism, separate from paired AGP−AGD validation performance.

## Learned GPR coefficients

AP uses the fixed CoSI anchored coefficients. AGP/AGD start from the same vector and share an unconstrained global delta across modalities. Equivalent monomial coefficients are reconstructed as `M(anchor_alpha).T @ gamma`; coefficients are polynomial terms, not spectral energy.

- Across AGP/AGD modality rows, 144/144 delta coefficients differ from zero by more than `1e-8`.
- Negative anchored coefficient rows: 24/36; nonzero order-3 rows: 36/36.
- Full `gamma_k0..k3`, sums, L1 norms, negative counts, deltas, monomial coefficients, and absolute-coefficient effective orders are in `filter_diagnostics.csv`.

Mean coefficients by dataset, with each shared checkpoint counted once:

| Dataset | Variant | γ0 | γ1 | γ2 | γ3 | Δγ0 | Δγ1 | Δγ2 | Δγ3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Movies | AGP | 0.0153 | 0.0962 | 0.9292 | 0.0311 | -0.0403 | 0.0437 | 0.0372 | 0.0311 |
| Movies | AGD | 0.1493 | 0.0055 | 0.7859 | -0.1001 | 0.0937 | -0.0470 | -0.1061 | -0.1001 |
| Grocery | AGP | 0.0107 | 0.1014 | 0.9346 | 0.0411 | -0.0448 | 0.0490 | 0.0427 | 0.0411 |
| Grocery | AGD | 0.1221 | 0.0295 | 0.8130 | -0.0683 | 0.0666 | -0.0230 | -0.0789 | -0.0683 |
| ele-fashion | AGP | 0.0878 | -0.0458 | 0.8587 | -0.0320 | 0.0322 | -0.0983 | -0.0333 | -0.0320 |
| ele-fashion | AGD | 0.1808 | 0.0550 | 0.7244 | -0.1547 | 0.1252 | 0.0025 | -0.1675 | -0.1547 |

## AGP checkpoint interventions

These alter an already trained AGP checkpoint and measure reliance; they do not replace retrained AGP−AP or AGP−AGD comparisons.

- Movies / gamma_reset_prior: accuracy -0.420 pp (SD 0.258, +/-/tie 0/3/0), CE +0.0133.
- Movies / graph_injection_off: accuracy -5.379 pp (SD 0.440, +/-/tie 0/3/0), CE +0.2085.
- Movies / lambda_one: accuracy +0.780 pp (SD 0.098, +/-/tie 3/0/0), CE -0.0487.
- Grocery / gamma_reset_prior: accuracy -0.303 pp (SD 0.132, +/-/tie 0/3/0), CE +0.0117.
- Grocery / graph_injection_off: accuracy -6.179 pp (SD 0.269, +/-/tie 0/3/0), CE +0.2024.
- Grocery / lambda_one: accuracy +0.576 pp (SD 0.073, +/-/tie 3/0/0), CE -0.0289.
- ele-fashion / gamma_reset_prior: accuracy -0.017 pp (SD 0.067, +/-/tie 1/2/0), CE -0.0008.
- ele-fashion / graph_injection_off: accuracy -1.057 pp (SD 0.212, +/-/tie 0/3/0), CE +0.0287.
- ele-fashion / lambda_one: accuracy -1.156 pp (SD 0.117, +/-/tie 0/3/0), CE +0.0391.

## Conservative diagnosis

The repeatability SD/range gives execution noise context for Movies and Grocery. With three model seeds on one fixed split, results support directional and dataset-specific judgments only; no p-values or population generalization claims are made.

- **RU-PO:** dataset mean deltas (pp) Movies +4.119 (3/3 seeds positive), Grocery +4.178 (3/3 seeds positive), ele-fashion +0.378 (3/3 seeds positive).
- **AU-RU:** dataset mean deltas (pp) Movies -0.670 (0/3 seeds positive), Grocery -0.381 (0/3 seeds positive), ele-fashion -0.014 (1/3 seeds positive).
- **AP-AU:** dataset mean deltas (pp) Movies -0.250 (1/3 seeds positive), Grocery -0.000 (2/3 seeds positive), ele-fashion -0.038 (1/3 seeds positive).
- **AGP-AP:** dataset mean deltas (pp) Movies +0.360 (2/3 seeds positive), Grocery +0.254 (3/3 seeds positive), ele-fashion -0.024 (1/3 seeds positive).
- **AGP-AGD:** dataset mean deltas (pp) Movies -1.470 (0/3 seeds positive), Grocery -1.445 (0/3 seeds positive), ele-fashion +0.027 (1/3 seeds positive).

The validation-accuracy mean-rank rule across the three datasets selects **RU** (mean dataset rank 1.67) as the compact cross-dataset candidate. Dataset-specific differences and macro-F1/CE trade-offs remain visible in the tables; this rank alone does not establish a universal winner. Retain that candidate only as the current evidence-backed backbone, and wait for review before adding another module.

Structural context is valuable on all three fixed splits: RU−PO is positive for accuracy in 9/9 paired seeds, with mean gains +4.119 pp on Movies, +4.178 pp on Grocery, and +0.378 pp on ele-fashion. Repeated anchoring lowers the measured order-3 drift on every dataset, but AU−RU accuracy is negative on Movies and Grocery and effectively tied on ele-fashion. AP−AU is also non-positive at the dataset-mean level, so neither anchoring nor the fixed CoSI profile adds a consistent retrained accuracy gain here. AGP moves substantially away from the fixed coefficients, including order-3 terms, but AGP−AP is modest and changes direction on ele-fashion. AGD beats AGP by about 1.45 pp on both Movies and Grocery; ele-fashion is nearly tied. Thus hard protection reduces semantic drift but has no consistent retrained task advantage in this campaign.

## Self-audit

1. Started strictly from `2fca8b31dc85edb8c28f9aa4066318b1e8557ad3`: **yes**, verified against the fetched previous branch and ancestry.
2. Used other historical experiment branches: **no**.
3. Ran or evaluated NC test: **no**. The shared dataset loader materializes the label tensor and fixed split indices; development mode masks `test_idx` before the training loss and derives checkpoint/metrics only from train/validation labels. No NC test metric was computed or inspected. The LP smoke's existing loader reads its frozen edge split to filter held-out positive message edges, but did not evaluate LP test metrics.
4. RU regression to SPGPR-U: **passed**, with mapped projector/lambda/fusion weights and the `1e-6` maximum-error gate.
5. CoSI prior conversion: **yes**, triangular solve recovers monomial prior `[0.15, 0.1275, 0.7225, 0]`; synthetic graph composition test passed.
6. Repeatability floor: see same-seed population SD and range table above.
7. RU−PO graph value: **yes**; +4.119/+4.178/+0.378 pp on Movies/Grocery/ele-fashion, with 3/3 positive seeds on each.
8. AU−RU semantic anchoring value: **no retrained accuracy gain**; means are −0.670/−0.381/−0.014 pp. Anchoring only supports the representation-drift mechanism result.
9. Anchored states reduce drift: **yes**, H3 drift is lower in 3/3 datasets (by about 0.043–0.066 for RU); this does not imply task improvement.
10. AP−AU fixed CoSI/PPR prior value: **not supported**; means are −0.250/approximately 0/−0.038 pp.
11. AGP−AP learned GPR value: **small and dataset-dependent**; +0.360 Movies, +0.254 Grocery, −0.024 ele-fashion pp. Accuracy/CE trade off on Movies.
12. Learned gamma departure: **yes**; all 144/144 displayed AGP/AGD delta entries differ from zero by >`1e-8`, and the per-dataset means are shown above.
13. Negative/order-3 coefficients: **yes**; AGP develops negative terms on ele-fashion; AGD has negative order-3 means across datasets. Order 3 is nonzero for 36/36 displayed AGP/AGD modality rows.
14. AGP−AGD explicit protection value: **no consistent task gain**; AGP trails AGD by 1.470/1.445 pp on Movies/Grocery and is near-tied on ele-fashion.
15. Protected output's closeness to P: **yes as a mechanism**; AGP has higher cos(P,Zpre) and lower relative displacement than AGD in all 3 datasets (table above).
16. Intervention consistency: **gamma-reset aligns with retrained AGP−AP on 2/3 dataset means** (Movies/Grocery, not ele-fashion). `lambda_one` and retrained AGP−AGD align at the dataset-mean direction on 3/3 datasets; ele-fashion's retrained mean is near zero and only 1/3 paired seeds favors AGP. Intervention reliance and retrained effects are not interchangeable.
17. Dataset-specific regime: **yes**; Movies/Grocery favor AGD in raw accuracy while ele-fashion ranks RU first; AU/AP add no consistent gain.
18. Accuracy/F1/CE trade-offs: **yes**; Movies AGP−AP raises mean accuracy by 0.360 pp and macro-F1 by 0.740 pp, while mean CE worsens by 0.0527; accuracy/F1 and CE therefore do not always move together.
19. Current candidate: **RU**, chosen by cross-dataset mean validation-accuracy rank with the stated limitations.
20. Evidence for another MAG-specific module: **not established by C0 alone**; stop here for human review.
21. LP smoke: **compliant** with AGP/AGD, two epochs/two batches, fanouts `[5,5,5]`, positive message-edge removal, validation inference, checkpoint save, and no test evaluation.
22. Claims: restricted to fixed-split validation comparisons and descriptive mechanisms; no claim of exact original GPR-GNN equivalence or universal/theoretical optimality.
