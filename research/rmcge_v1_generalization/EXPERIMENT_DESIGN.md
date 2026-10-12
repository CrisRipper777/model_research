# RM-CGE V1 Generalization & Optimization Study

## Study question and frozen boundary

This study tests whether two optimizer regularizers improve the validation generalization of the frozen RM-CGE V1 under the existing full-graph node-classification protocol. It does not alter the model computation graph, relation supports, routing objective, residual scale, fusion, label split, or K=8 semantic candidates. Only model dropout (`0.20` versus `0.35`) and the existing AdamW weight-decay value (`1e-4` versus `0.05`) vary.

Frozen model settings: hidden dimension 128, two layers, three experts per layer, expert bottleneck 64, router hidden dimension 64, residual scale 0.5, and K=8. O0 is the canonical internal reference. The archived V1 Pilot remains a historical reference, not an exact score or trajectory target.

## Factorial design and experimental unit

The complete matrix is a balanced 2 × 2 factorial on each of Movies, Grocery, and ele-fashion:

| ID | Dropout | AdamW weight decay |
|---|---:|---:|
| O0 | 0.20 | 0.0001 |
| O1 | 0.35 | 0.0001 |
| O2 | 0.20 | 0.05 |
| O3 | 0.35 | 0.05 |

Every dataset × treatment combination uses seeds 42, 43, and 44, for 36 planned runs. Within a dataset, these are three paired seed blocks; each seed-level run is the experimental unit (`n=3` for every treatment contrast within that dataset). Nodes and epochs are repeated observations inside a run and are never treated as independent replicate runs.

For each dataset and seed, all four treatments must produce identical pre-optimization model and classifier state dictionaries. Each run reseeds Python, NumPy, CPU Torch, and the selected CUDA device independently. Existing optimizer grouping remains unchanged: inspection found no RM-CGE no-decay exclusions, so the current builder creates one AdamW parameter group containing every encoder and classifier parameter. The study records group membership, learning rate, and decay for each run.

All other values remain: Full-Graph NC, `unified_full_graph_nc_v1`, AdamW, learning rate `1e-3`, at most 300 epochs, patience 30, minimum stopping epoch 30, `early_stop_min_delta=1e-4`, gradient clipping 1.0, evaluation every epoch, null scheduler, and Cross-Entropy loss. Accuracy remains the sole early-stopping signal. Test evaluation stays disabled.

## Revised correctness and reproducibility standard

The original Movies/O0/seed-42 Pilot replay is retained as historical evidence. A different best Epoch, stopping horizon, or validation score does not block the factorial study. It is classified as `historical_full_training_replay_not_exact` and is reported without changing the original artifacts.

Admission to the formal matrix instead requires: identical official data/splits/cache fingerprints; matching frozen O0 configuration and optimizer semantics; identical O0–O3 initialization for each Dataset × Seed; short-run frozen-source versus study-code Forward/Backward/AdamW agreement within documented floating-point tolerance; RNG-invariant extra diagnostics; finite training/validation histories; and successful dual-checkpoint save/reload validation. Test evaluation stays disabled and Test labels are never indexed.

The retained two-update probe provides a maximum parameter difference of about `5.05e-7`, equal step losses, identical initialization, and unchanged CPU/CUDA RNG around the extra CPU metrics. This is short-run evidence, not proof that every later training operation is identical. CUDA/cuDNN/TF32/determinism settings and sparse propagation behavior are recorded; the formal environment is not silently changed to force determinism.

The frozen implementation is committed before training. The first formal batch is the 12 seed-42 runs across all datasets and treatments. After run-level finiteness, pairing, Test-isolation, and Checkpoint reload checks pass, continue seeds 43 and 44. All formal O0 results are internal to this frozen-code matrix; the historical Pilot does not provide an acceptance threshold.

## Metrics and checkpoint selection

At every epoch, record the loss from the ordinary stochastic training-mode forward, plus a post-update Train Accuracy computed with the model in eval mode, and Validation CE, Accuracy, and Macro-F1 computed in eval mode without gradients. The existing full-graph inference tensor is reused for Train Accuracy and Validation CE so added diagnostics require no extra model forward and do not consume dropout randomness. Index-based classifier scoring is chunked without a shuffled loader. The evaluation helper has an RNG-state invariance test.

Maintain two immutable state snapshots over exactly the epochs observed before Accuracy-driven early stopping:

- `best_acc.pt`: best Validation Accuracy under the existing `early_stop_min_delta`; this alone determines patience and the primary result.
- `best_f1.pt`: highest Validation Macro-F1 over the same observed epoch range; ties retain the earliest epoch and this choice never affects stopping.

Each checkpoint stores model and classifier weights, its selection metric/epoch and Validation metrics, resolved configuration and config SHA256, source Git SHA, cache fingerprints, and the Accuracy stopping horizon. Validation class counts, precision, recall, F1 and predicted counts are calculated separately for both checkpoints using Validation indices only. The compatibility metrics `best.pt` are not required; no metric-selection rules are mixed in summary tables.

## Stage B frozen-checkpoint diagnostics

Before interpreting the original V1 mechanism, reload the three seed-42 best-Accuracy checkpoints and verify that recomputed embeddings/logits are repeatable and Validation Accuracy/Macro-F1 match the original Pilot. Save only checksums and numerical differences, not node embeddings or logits.

For each checkpoint, compare Original, All Cross-modal Off, Physical Cross-modal Off, and Semantic Cross-modal Off. A relation-specific intervention zeros only that relation's cross-modal context after its normal support aggregation/conversion; same-modal context, validity masks, expert bank, router, classifier, and learned weights remain in normal forward execution. Since router inputs change when cross context is zeroed, results are frozen-path sensitivity checks, not retrained ablations or independent causal effects. Record Validation CE/Accuracy/Macro-F1, prediction flip fraction, relative logit RMS change, and relative embedding RMS change.

For each Dataset × Layer × Modality, summarize node-level Physical/Semantic route mass (mean, SD, and 5/25/50/75/95 percentiles), joint route entropy, and the winning relation-expert candidate's expert ID proportions. Pair Text and Visual views on common valid nodes to quantify route-mass disagreement. For each relation, compare the current normalized node state with its same-modality propagated context using cosine similarity and state-normalized L2 distance, then report descriptive Pearson and Spearman correlations of those quantities with node-level Semantic route mass. These node-level correlations describe this graph/checkpoint only; nodes do not increase the experimental `n`, and no node-level p-values are used.

For relation contexts before Expert input LayerNorm, report raw RMS and nodewise low-variance fraction using the prespecified threshold `population feature variance < 1e-6`. Also report RMS after the existing shared non-affine block LayerNorm. A small raw cross-context RMS is not interpreted as an equally small post-normalization Expert input.

## Estimands and reporting

For the Accuracy-selected checkpoint, report per Dataset × Treatment mean ± sample SD (`ddof=1`) across the three seeds for Validation Accuracy, Macro-F1, Validation CE, Best Epoch, Train-to-Validation gaps, training duration, and peak CUDA memory. For the F1-selected checkpoint, report its corresponding Accuracy, Macro-F1, Validation CE, epoch, and per-class metrics separately.

Use within-seed paired treatment differences. For Accuracy and Macro-F1, positive favors the first treatment; for Validation CE, negative favors the first treatment:

- Dropout at low decay: `O1 − O0`.
- Weight decay at low dropout: `O2 − O0`.
- Weight decay at high dropout: `O3 − O1`.
- Dropout at high decay: `O3 − O2`.
- Additive interaction: `(O3 − O2) − (O1 − O0)`.

Show the three paired seed differences, their mean and sample SD. Effects are descriptive; no p-value, node-level inference, or significance claim is planned. A favorable direction in three seeds alone is not statistical significance.

Learning plots show every seed trajectory and treatment summaries with the available run count by epoch; early stopping makes later-epoch sample counts smaller. Checkpoint-selection plots connect Best-Accuracy and Best-F1 outcomes from the same dataset/treatment/seed run. Report seed-level evidence alongside aggregates to expose heterogeneity.

## Artifact and provenance plan

The study has independent code, resolved configs and outputs under `research/rmcge_v1_generalization/` and `outputs/rmcge_v1_generalization/`. Machine-readable files include the manifest, run table, exact epoch history, initialization/gradient/parameter audits, paired contrasts, factorial effects, dual-checkpoint tradeoffs, class metrics, frozen cross-relation interventions, router node summaries, failures, and validation-only cache/provenance checks. Checkpoints, original features/labels, per-node embeddings, and raw logits stay out of Git.

Plots use the established Python/Matplotlib backend. The planned report figure set is: (1) train/validation CE and accuracy trajectories by dataset and treatment, (2) paired factorial effects by seed with mean ± SD, (3) Accuracy-versus-Macro-F1 checkpoint trade-offs with paired checkpoint choices, and (4) stopping/selection epoch differences. Each plot has a preserved script and CSV source; comparable multi-panel axes share scales and are checked after rendering.

## Interpretation limits and next-version rule

If regularization changes the seed-paired outcomes and the train/validation curves in a consistent, descriptive way, this supports a training-fit explanation but does not establish it as the sole bottleneck. Frozen relation interventions establish only fitted-checkpoint path sensitivity. Router distribution spread and view disagreement establish node-conditional variation, not reliable source identification. A Reliability-aware router, different treatment of zero-update candidates, semantic relation change, expert change, or fusion change remains a future hypothesis and requires a separately designed study; no V1.1 work starts in this task.
