# Implementation Notes and Self-Audit

## Code audit

- The formal encoder is `src/models/mrcge_v13.py`; it declares `supports_link_prediction=False`, requires a global V7B.1 semantic cache and global node order, and consumes one global feature matrix plus the corresponding full physical edge index. `src/tasks/lp.py` uses `LinkNeighborLoader` sampled-subgraph IDs and also evaluates Test by default. Directly changing the capability flag would therefore be unsafe.
- The added LP adapter is independent. It does not change `mrcge_v13.py`, the model factory, the NC Trainer, the existing sampled LP Trainer, or any prior experiment runner/result.
- MM-Graph `clip_feat.pt` is split by the existing loader as Text in columns 0:512 and Visual in columns 512:1024. Preparation and the LP worker both assert exact tensor equality with the loader's modality views.
- The LP official split is read from each configured `lp-edge-split.pt` and is never rewritten. The audit reports raw counts, undirected duplicates, self-loops, and cross-partition overlaps. The Train graph is rebuilt only from Train positives and compared with the loader graph. Under the user's direction, the current self-loops and overlaps are accepted under protocol `mrcge_v13_full_graph_edge_holdout_lp_v1_official_split_as_is`.
- This acceptance is a material protocol deviation. Validation positives that overlap Train may be included in training supervision and/or Train message passing, so Validation metrics are not leakage-safe estimates. Train self-loop positives remain decoder supervision; the repository's usual graph preprocessing excludes self-loops from propagation. Both behaviors are stated in per-run metadata and `data/lp_split_preflight_audit.json`.
- LP KNN preparation uses the existing V7B.1 builder with global features and a Train-only graph for provenance/overlap statistics. Candidate construction uses no LP edge labels. LP Train negative rejection uses the existing repository sampler; known positive edges, including Validation/Test positives, serve only as rejection filters.
- The detach-leaf path is tested against direct mean BCE + one auxiliary loss on a synthetic graph. Per-batch summed BCE is divided by the fold's total supervised count. Decoder gradients accumulate at the leaf, then pass through the real encoder output; the Balance Auxiliary Loss is backpropagated once.
- Validation reuses the existing fixed candidates, MRR/Hits definitions, and pessimistic `>=` tie ranking. The LP adapter has no Test evaluation call. Its Checkpoint includes encoder/projection/decoder state and model/config/cache/split/Train-graph provenance; a fresh module load with `strict=True` must reproduce Validation metrics.

## Current implementation checks

- Baseline was clean at `4bb7253ac9dffeca57686771c139002d6e84a19f`; the requested branch was created from that exact commit.
- Both fixed seed42 NC splits and both official LP split files are present. NC preparation passed for all five datasets. A read-only audit found that the current LP files contain positive self-loops and undirected cross-partition overlap:

  | Dataset | Train / Valid / Test raw positives | Self-loops by partition | Undirected overlap Train-Valid / Train-Test / Valid-Test |
  |---|---:|---:|---:|
  | sports-copurchase | 356,202 / 45,557 / 37,413 | 26 / 3 / 5 | 8,264 / 6,621 / 2,035 |
  | cloth-copurchase | 951,271 / 121,323 / 97,456 | 38 / 7 / 6 | 21,043 / 17,360 / 5,169 |

- The audit canonicalizes positives as undirected endpoint pairs. The official files remain byte-for-byte unchanged; LP preparation records the conflict audit and builds the Train-graph semantic caches.
- Historical Movies/Grocery/ele-fashion NC records are imported as `reused_historical`. Toys/Reddit-S seeds 42/43/44 were trained on the fixed seed42 split. When the outer runner raised after training, all six existing checkpoints were recovered by strict reloading without retraining; the LP command gates on their checkpoint SHA256s.
- CPU tests are synthetic only. The V1.3 smoke test uses the actual frozen Full encoder, actual V7B.1 cache builder/loader, global forward/backward, and strict state reload on a 12-node graph. It does not load a formal trainable dataset or perform an official experiment.
- GPU execution is isolated in per-cell workers. The initial guard-enabled LP attempt saw a new external PID and stopped; following the user's explicit authorization to continue on GPU 1, the resumed run uses `--skip-gpu-guard` and records that bypass in attempt/run artifacts. The collector does not launch a worker.

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
12. **LP official split deviation:** both official files are used unchanged. The audit and each run record preserve split hashes and disclose validation overlap with Train. No claim of leakage-safe Validation results is made.
13. **Executable commands:** supplied in `RUNBOOK.md`. NC preparation and all six new NC checkpoint reloads are verified. LP preparation builds the two Train-graph caches; LP workers use the official split as-is and keep Test evaluation disabled.

The authoritative machine-readable split audit is `data/lp_split_preflight_audit.json`. It reports overlap/self-loop counts and the accepted protocol deviation. LP results, when present, remain Validation-only and must not be interpreted as leakage-safe test performance.

No architecture search, hyperparameter search, sampled/full protocol substitution, split regeneration, or Test evaluation is part of this implementation stage.
