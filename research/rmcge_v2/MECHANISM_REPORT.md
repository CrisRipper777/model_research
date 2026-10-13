# RM-CGE V2 Mechanism Report

All entries below come from the Validation-Accuracy-selected checkpoint. Frozen interventions hold fitted weights fixed and measure sensitivity; they are not retrained ablations or edge-level causal effects.

## Edge-conditioned physical weights

The per-run distributions, Text/Visual differences, reverse-direction differences, and target-degree strata are in `data/edge_weight_diagnostics.csv`. V0/V1 use unit weights; their between-modality correlation is mathematically undefined and is stored as zero with a `correlation_defined=false` flag.

Across the three V2 seeds, mean Text/Visual weights were 1.9977/1.9981 on Movies and 1.9971/1.9975 on Grocery; every measured edge was more than 0.2 from initialization on both datasets. Ele-fashion means were 1.9182/1.9168, with about 99.1% of edges more than 0.2 from one. The target-modality correlations were 0.9997–1.0000, and Text/Visual mean absolute differences were 0.00039–0.00168. Thus the scorer trained and changed weights, but its final outputs were almost modality-independent and close to the upper bound of 2. Movies/Grocery weights also varied little within a modality (mean SD about 0.01/0.004); this looks more like global propagation amplification than sharply selective neighborhoods. This is an observed optimization outcome; its cause is unresolved.

Mean absolute reverse-direction differences were about 0.00021 on Movies/Grocery and 0.093 on ele-fashion. Direction asymmetry therefore appeared on ele-fashion but was small on the other two datasets. The edge table retains percentiles and degree strata for each run.

## Context and expert scale

`data/message_scale_diagnostics.csv` records raw and reference-normalized Delta, cross-modal C, and KNN S RMS, their expert-coefficient-weighted scales, physical and combined expert-profile RMS, per-expert output RMS and pairwise cosine, final graph correction, and the reference-scale floor count. `data/frozen_interventions.csv` records validation metric and logit changes when each path is closed.

For V2, mean normalized Delta RMS was 0.95–1.00. After expert coefficients, Delta contributed about 0.21–0.24 RMS per expert; cross-modal context contributed about 0.36 on Movies, 0.42 on Grocery, and 0.50 on ele-fashion; KNN contributed about 0.09, 0.09, and 0.17. The resulting weighted graph correction RMS was about 0.45, 0.96, and 0.68 against intrinsic RMS near 1.0. The new paths reached the expert profiles and fused prediction; none exploded by orders of magnitude. Cross context was the largest added contribution, while KNN was the smallest.

With a fixed V2 checkpoint, forcing dynamic edge weights to one changed Validation Accuracy by −0.59 percentage points on Movies, −0.76 on Grocery, and −0.18 on ele-fashion on average; Macro-F1 changed by −0.84, −1.31, and −0.20 points. Cross-context-off changed Accuracy by −1.00, −0.99, and −0.54 points. KNN-off changed it by −0.25, −0.13, and −0.21 points. These are checkpoint sensitivities only. In particular, the apparent edge-off benefit is compatible with a learned near-global scale increase, not evidence that edge ranking itself helped.

## Router and optimization

`data/expert_router_diagnostics.csv` and `data/gradient_audit.json` retain corrected Top-2 shares, dense entropy, node-logit corrections, route flips, first-backward gradient checks, and parameter update norms. V1's edge scorer is intentionally inactive. Initial zero Delta gives the `a` coefficients no data gradient; their later learned values and updates are retained in the diagnostics.

V2's Edge Scorer, edge projections, Cross Projections, Node Router, Delta coefficients, Experts, Projectors, and Fusion all updated in 9/9 runs. V1's inactive Edge Scorer remained unchanged in 9/9, as designed. Top-2 selection shares did not show a universally dead expert, but Grocery Visual dense-route entropy was only 0.003 for V2 (0.002 for V1), with Node Router correction RMS 4.60 (3.46). Node corrections changed at least one Top-2 choice for 34% of V2 Grocery Visual nodes; route flips exceeded 85% on both ele-fashion modalities. This points to a router calibration concern, not a missing-gradient problem.

## Limits

V2−V1 Validation differences are mixed or near zero, and V2 weights are nearly saturated and highly correlated across target modalities. The current evidence therefore supports edge-path participation but not useful per-edge or modality-specific selection. Three seeds per dataset are descriptive. Message RMS and frozen sensitivity do not establish utility or causal attribution; read them with paired Validation results and the parameter-activity boundary.
