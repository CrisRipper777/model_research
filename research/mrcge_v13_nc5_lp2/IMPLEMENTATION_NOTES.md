# Implementation Notes and Self-Audit

## Code audit

- The formal encoder is `src/models/mrcge_v13.py`; it declares `supports_link_prediction=False`, requires a global V7B.1 semantic cache and global node order, and consumes one global feature matrix plus the corresponding full physical edge index. `src/tasks/lp.py` uses `LinkNeighborLoader` sampled-subgraph IDs and also evaluates Test by default. Directly changing the capability flag would therefore be unsafe.
- The added LP adapter is independent. It does not change `mrcge_v13.py`, the model factory, the NC Trainer, the existing sampled LP Trainer, or any prior experiment runner/result.
- MM-Graph `clip_feat.pt` is split by the existing loader as Text in columns 0:512 and Visual in columns 512:1024. Preparation and the LP worker both assert exact tensor equality with the loader's modality views.
- The LP official split is read from each configured `lp-edge-split.pt`. Train/Validation/Test positive partitions are compared after canonicalizing undirected edges. The Train graph is rebuilt only from Train positives and compared with the loader graph. Both directions, duplicate edges, invalid IDs, self-loops, fold masking, and held-out positive overlap have explicit checks.
- LP KNN preparation is wired to the existing V7B.1 builder with global features and a Train-only graph for provenance/overlap statistics. However, preparation validates the official split before invoking that builder. The current official split files fail validation, so no LP semantic cache was built or marked prepared. If a reviewed split later passes, the resulting semantic candidates are fixed across folds and seeds. LP Train negative rejection uses the existing repository sampler; known positive edges, including Validation/Test positives, serve only as rejection filters.
- The detach-leaf path is tested against direct mean BCE + one auxiliary loss on a synthetic graph. Per-batch summed BCE is divided by the fold's total supervised count. Decoder gradients accumulate at the leaf, then pass through the real encoder output; the Balance Auxiliary Loss is backpropagated once.
- Validation reuses the existing fixed candidates, MRR/Hits definitions, and pessimistic `>=` tie ranking. The LP adapter has no Test evaluation call. Its Checkpoint includes encoder/projection/decoder state and model/config/cache/split/Train-graph provenance; a fresh module load with `strict=True` must reproduce Validation metrics.

## Current implementation checks

- Baseline was clean at `4bb7253ac9dffeca57686771c139002d6e84a19f`; the requested branch was created from that exact commit.
- Both fixed seed42 NC splits and both official LP split files are present. NC preparation passed for all five datasets. The preparation script never silently replaces a split. A read-only audit found that both current LP splits conflict with the required undirected positive-partition disjointness and contain positive self-loops:

  | Dataset | Train / Valid / Test raw positives | Self-loops by partition | Undirected overlap Train-Valid / Train-Test / Valid-Test |
  |---|---:|---:|---:|
  | sports-copurchase | 356,202 / 45,557 / 37,413 | 26 / 3 / 5 | 8,264 / 6,621 / 2,035 |
  | cloth-copurchase | 951,271 / 121,323 / 97,456 | 38 / 7 / 6 | 21,043 / 17,360 / 5,169 |

  The audit canonicalizes positives as undirected endpoint pairs. `prepare --phase lp` records the audit, skips cache construction, and exits 2. No split was repaired, and no LP worker was started.
- Historical Movies/Grocery/ele-fashion NC records are imported as `reused_historical`, with no retraining path for those datasets. The LP command gates on all six new NC cells and their checkpoint SHA256s.
- CPU tests are synthetic only. The V1.3 smoke test uses the actual frozen Full encoder, actual V7B.1 cache builder/loader, global forward/backward, and strict state reload on a 12-node graph. It does not load a formal trainable dataset or perform an official experiment.
- GPU execution is isolated in a per-cell worker with the existing V1.3 ownership/memory/utilization guard. No bypass flag exists. The collector does not launch a worker.

## Self-audit answers

1. **Historical Full results:** yes. The collector reuses and hash-checks all nine historical Full records; only Toys and Reddit-S are configured for new NC training.
2. **Fixed NC split:** yes. Preparation validates each seed42 split and the runner resolves the split path before changing the training seed; automatic split generation is disabled in formal NC workers.
3. **Encoder unchanged:** yes. No production encoder or NC training file is changed; LP calls the frozen encoder with global node order and graph semantics.
4. **LP semantic KNN:** the adapter/preparation is designed to use global node features via the existing V7B.1 builder without labels or held-out relation edges. The current split gate stops before cache construction, so no LP cache exists yet.
5. **LP physical graph:** the graph builder uses Train positives only and adds both directions. The current official splits fail earlier validation due to self-loops and cross-partition undirected overlaps, so no accepted LP graph or training run was produced.
6. **Current fold mask:** yes. Each fold removes canonical positives and both directions before its full-graph encoder forward; duplicates are canonicalized before assignment.
7. **Encoder gradient:** yes. Synthetic gradient equivalence verifies the decoder-to-leaf-to-encoder gradient chain; LP training performs one full encoder graph per fold.
8. **Batch weighting / auxiliary:** yes. Each batch BCE sum is divided by the whole fold supervision count, and auxiliary loss is differentiated once per fold.
9. **Sampled LP distinction:** yes. The protocol is named `mrcge_v13_full_graph_edge_holdout_lp_v1` and is documented as a distinct optimization protocol.
10. **Test evaluation:** disabled in NC and LP configs/workers. NC Test targets are masked to `-100`; LP adapter only evaluates the fixed Validation split. Test metric output is `not_evaluated`.
11. **Formal training launched by implementation:** no. Only CPU unit/synthetic tests and data-integrity preparation/preflight are allowed in this phase.
12. **Executable commands:** supplied in `RUNBOOK.md`. NC preparation is verified. The LP preparation command is executable but currently returns code 2 with a recorded protocol blocker; the LP training command refuses to launch until that gate is resolved.

The authoritative machine-readable blocker is `data/lp_split_preflight_audit.json`. The current LP results are intentionally absent; the implementation must not be interpreted as completed LP evaluation.

No architecture search, hyperparameter search, sampled/full protocol substitution, split regeneration, or Test evaluation is part of this implementation stage.
