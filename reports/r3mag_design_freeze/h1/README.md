# R³-MAG H1 design-freeze preflight

## Scientific question

This audit tests only H1: whether a validation-selected global dual-modality structural-response host still has useful node-specific and modality-specific response heterogeneity beyond matched shuffled directions. It does not implement H2, H3, or the final R³-MAG model.

The primary utility is the per-Audit-node cross-entropy reduction, in nats:

\[
U_{i,a}^{m}=CE_{baseline}(i)-CE_{action}(i,m,a).
\]

This is a frozen-host local response diagnostic, not a causal effect. Action argmaxes are preferences within the registered action set, not latent node roles.

## Exact protocol

- Datasets: Movies and Grocery are primary; ele-fashion is a stress dataset.
- Movies/Grocery always load the existing NC split at data seed 42. Host seeds 42, 43, and 44 affect only model initialization/training. ele-fashion uses its official split.
- The original train indices are partitioned once with seed `20261006`, by class, into HostTrain 80%, ResponseTrain 10%, and Audit 10%. Per-class counts use deterministic Hamilton largest-remainder allocation; every nonempty class contributes at least one HostTrain node. Classes with fewer than 10 examples are listed as small-class allocation fallbacks. The split is generated once and reused for all host seeds.
- Host loss indexes only HostTrain labels. Original validation labels select the best-accuracy checkpoint and report validation accuracy, macro-F1, and CE. ResponseTrain labels do not enter host fitting or response utility. The original-train labels are necessarily used once to stratify internal membership and derive partition histograms. Audit target labels are passed to the utility stage only after the best-validation checkpoint has been restored and frozen.
- No test metric, threshold, action, or label histogram is computed. Test count and index hash are recorded; its class histogram is marked `NOT READ — TEST SPLIT UNTOUCHED`.
- The physical graph operator is symmetric normalized `A+I`, reused by both modalities. Each projector is Linear → LayerNorm → ReLU → Dropout. `K=3`; signed global coefficients initialize as truncated PPR weights `gamma_k = alpha * (1-alpha)^k`, with `alpha=0.2`, and train freely. Each modality response receives LayerNorm; concatenated responses pass through a one-hidden-layer fusion MLP and classifier.
- Fixed host settings across datasets: hidden size 128, dropout 0.2, AdamW, learning rate `1e-3`, weight decay `5e-4`, max 1000 epochs, patience 100, best validation accuracy. Early stopping follows the NC runner's `min_epoch=30` and `min_delta=1e-4`.
- Each modality has exactly nine actions: NOOP, positive and negative direction for each order 0–3. Epsilon is fixed at 0.1 or 0.2. Directions are rescaled to the node's global response norm; each candidate is then NormMatched to that norm.
- The matched control borrows the donor's raw direction `R_j−C_j`, then rescales it to the target's `||C_i||`. Donors match degree-quantile × frozen predicted class where possible, then fall back to degree bucket and global pool. It preserves action count, order, epsilon, and per-target norm. Twenty fixed control seeds are used in formal runs. Structural and shuffled joint actions are both evaluated as actual pairs through frozen normalization, fusion, and classifier.
- Node-specific headroom is mean per-node best action utility minus the best single shared action's mean utility. Modality headroom is mean per-node best separate Text/Visual pair utility minus the best shared action pair (same action type on both modalities). Each is compared with its matched shuffled counterpart.
- Paired bootstrap resamples Audit target nodes at least 2000 times. These intervals describe target-sample uncertainty only; they are not causal or independent-graph confidence intervals. Sign stability is reported across host seeds and the two registered epsilons.
- Evidence tiers are set before formal runs from Movies and Grocery. The practical threshold for a CE-excess estimate is 0.01 nats/node: STRONG requires every seed × epsilon estimate to meet it and every within-run 95% interval to exclude zero; MIXED / MARGINAL requires at least 70% positive estimates (including detectable but sub-threshold effects); otherwise WEAK / FAIL. Raw estimates and intervals remain fully reported.

## Correctness checks

The synthetic tests and runtime checks cover split disjointness/reproducibility, NOOP identity, epsilon-zero identity for structural and shuffled candidates, NormMatch tolerance, action cardinality, joint-oracle dominance, vectorized versus row-wise candidate logits/CE on 10 sampled Audit nodes and multiple actions, and repeated frozen candidate utility determinism. NOOP identity remains `<1e-6`; NormMatch remains `<1e-5`; non-NOOP float32 vectorized/repeat logits and CE use a fixed `<1e-5` tolerance after formal CUDA QA measured up to `4.8e-6` batch-shape differences. Actual maxima and pass flags are written to `qa_report.json`.

## Run commands

From the repository root, in `yhf_env`:

```bash
PYTHONPATH=. conda run --no-capture-output -n yhf_env python -m pytest -q tests/test_r3mag_response_core.py
PYTHONPATH=. conda run --no-capture-output -n yhf_env python -m src.analysis.r3mag_h1_response_audit --smoke
PYTHONPATH=. conda run --no-capture-output -n yhf_env python -m src.analysis.r3mag_h1_response_audit
```

The smoke uses Movies/seed42, three epochs, two shuffled repeats, and 50 bootstrap repeats for implementation QA only. Do not interpret it as an H1 result. Formal defaults run all three datasets, all three host seeds, both epsilons, 20 controls, and 2000 bootstrap repeats. Checkpoints and complete run details go to ignored `outputs/r3mag_design_freeze/h1/`; compact review files are written here.

## Files and known limits

- `aggregate_report.md`: results, predeclared evidence tiers, stability, and protocol notes.
- `aggregate_metrics.csv`: dataset × host seed × epsilon metrics and within-run paired intervals.
- `per_run/`: split hashes/histograms, training/checkpoint metrics, signed gamma, basis diagnostics, shuffled-repeat distributions, and QA.
- `per_node/`: deterministic sample of up to 256 Audit nodes per run and epsilon, with all structural single-modality action utilities and shuffled/joint summaries.
- `qa_report.json`: correctness and label-boundary assertions.

The graph model is transductive and computes on the graph/features while training and auditing, consistent with the repository's full-graph NC protocol. No test-label slice is indexed for any metric or report. Internal partition histograms are derived from the original-train labels used for the required stratification; test histograms are withheld. Bootstrap uncertainty does not account for graph sampling or dataset-to-dataset variation.
