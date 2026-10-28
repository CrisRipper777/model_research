# MR-CGE V1.3-A NC5 / LP2 Protocol

## Scope and frozen encoder

This campaign reuses the frozen `mrcge_v13` Full encoder: `M_repaired`, `joint_view`, `sender_private_ratio=0`, hidden width 256, physical/semantic hops 3/1, four shared experts with Top-2 routing, and V7B.1 semantic KNN with K=8. The encoder source and its propagation, filtering, cross-view, and completion equations are not modified.

NC covers Movies, Grocery, Toys, Reddit-S, and ele-fashion with training seeds 42/43/44. The nine Movies/Grocery/ele-fashion records are reused from the audited historical V1.3-A Full campaign. Only Toys and Reddit-S are new NC training: three seeds each. All seeds use the one fixed seed42 NC split per dataset. Missing split files stop preparation by default; `--allow-create-fixed-split` is an explicit opt-in to the existing stratified split generator. Formal NC workers always set `auto_generate_splits=false`, mask Test labels to `-100` before `run_nc`, and set `evaluate_test=false`.

NC uses the historical full-graph Trainer and settings: AdamW, LR 0.001, weight decay 0.0001, at most 300 epochs, patience 30, minimum epoch 30, minimum delta 1e-4, gradient clipping 1.0, full-graph inference, private/unimodal CE weight 0.3, balance auxiliary weight 1.0 (model balance coefficient 0.01), best Validation Accuracy as the primary checkpoint, and a separate Best-F1 checkpoint.

## LP official split and accepted protocol deviation

LP covers only sports-copurchase and cloth-copurchase with seeds 42/43/44. It uses each dataset's existing `lp-edge-split.pt`; the split is never generated or repaired. The loaded CLIP feature is checked as `[Text(512), Visual(512)]`.

The physical propagation graph is made only from official Train Positive edges, canonicalized as undirected edges, expanded to both directions, and deduplicated. The official split files are never modified. Invalid node IDs and malformed Validation candidates still stop the run. Duplicate positive rows are grouped by their undirected canonical edge before fold assignment so they cannot be assigned to separate folds or receive repeated supervision.

The official files contain self-loop positives and Train/Validation/Test overlap after undirected canonicalization (counts are in `data/lp_split_preflight_audit.json`). Per the user's instruction, these files are accepted as-is under protocol `mrcge_v13_full_graph_edge_holdout_lp_v1_official_split_as_is`. This is a material protocol deviation: Validation positives that also occur in Train may be supervised during training or present in the Train message graph. Therefore Validation MRR/Hits are not leakage-safe estimates and must be described as official-split-as-is results. The code does not remove, redistribute, or rewrite overlapping split entries.

Official Train self-loop positives remain in decoder supervision. The repository's standard graph preprocessing removes self-loops from message passing, so those positives do not appear in physical propagation graphs. This handling is recorded in each run's audit.

V7B.1 Text and Visual KNN are built from global node attributes only, with K=8 and the repository's existing feature normalization, FAISS search, provenance, and cache fingerprint implementation. The cache builder receives only the Train-only physical graph for physical-overlap/provenance statistics. It does not receive LP positive labels or Test/Validation edges for candidate generation. A semantic candidate may coincide with a supervised positive by chance because the candidates are attribute-derived; this is not a label-derived edge.

Unique undirected Train Positive pairs, including self-loops as supervision pairs, are partitioned once into three deterministic folds using `fold_seed=42`. For fold k, its non-loop positives and reverse directions are removed from the physical propagation graph and all fold positives are used for decoder supervision. Other Train Positive edges remain as messages. Each epoch visits all three folds. Validation uses the complete unique Train-only physical graph; due to official cross-partition overlaps, this graph may contain Validation or Test positive pairs as supplied in Train.

Training negatives use the existing LP rejection sampler with one negative per positive. The full set of known positive Train/Validation/Test edges is used only to reject false-negative candidates; only the generated Train negatives enter the training loss. Test negative candidate lists and Test rankings are not read by the LP adapter. Validation uses the official fixed positive/negative candidates, MRR, Hits@1/3/10, and pessimistic tie ranking (`negative_score >= positive_score`). Best Validation MRR is the only selection criterion. Test metrics are `not_evaluated`.

## LP optimization

The task protocol is `mrcge_v13_full_graph_edge_holdout_lp_v1_official_split_as_is`, not `unified_sampled_lp_v1`. Each fold performs one full-graph encoder forward. A detached embedding leaf receives gradients from the MLP decoder's edge batches; batch BCE sums are divided by the fold's total supervised-edge count, making their accumulated gradient equal to the full-fold mean BCE gradient. The accumulated embedding gradient is propagated through the original encoder output, and the encoder Balance Auxiliary Loss is backpropagated exactly once per fold. Encoder, projection, and decoder gradients are checked, clipped to 1.0, and followed by one optimizer step per fold.

LP settings: Adam, LR 0.001, weight decay 1e-5, at most 150 epochs, patience 10, minimum epoch 20, minimum delta 1e-4, gradient clipping 1.0, Validation every two epochs, projection width 128, existing `LinkPredictor` MLP with hidden width 256 / 3 layers / dropout 0.02, supervision edge batch 2048, BCE-with-logits, Balance Auxiliary Loss weight 1.0, and one filtered Train negative per positive. Encoder settings remain at the frozen V1.3-A values.

## Artifacts and status semantics

Prepared caches and per-cell outputs live below `outputs/mrcge_v13_nc5_lp2/`. Lightweight tracked records live below `research/mrcge_v13_nc5_lp2/data/`. Checkpoints, original feature tensors, raw graphs, and dense logits are not committed. Per-cell records include source commit, config, split/cache/data fingerprints, checkpoint SHA256, resource/wall-time details, and explicit Test status. Collector status is `partial` until all expected formal cells are present and verified; it cannot infer or fill missing seeds.
