# MR-CGE Final Benchmark (NC5 + LP2)

This directory contains the frozen final benchmark protocol, empty result schemas, and the two paper-table templates. No final benchmark results are included yet.

## Frozen model and data

- Model: **MR-CGE Final (V1.3-A Full)**, copied from `src/models/mrcge_v13.py` at source commit `4177ec8aec4e05ad513c0357f36da7b926a5c040`.
- NC: Movies, Grocery, Toys, Reddit-S, and ele-fashion; seeds 42/43/44. MAGB datasets pin all seeds to the prepared seed42 split; ele-fashion pins to its official `split.pt`. Split generation is disabled.
- LP: sports-copurchase and cloth-copurchase; seeds 42/43/44. The original `lp-edge-split.pt` files and their fixed negative candidates are used unchanged. No folds are introduced.
- NC selects by Best-Validation-Accuracy. Test labels are masked during Trainer execution. A fresh evaluator reloads the frozen checkpoint and the original labels, computes Validation for reload verification, then evaluates Test once. Test Macro-F1 uses every configured class ID.
- LP selects by Best-Validation-MRR. The original `LinkNeighborLoader`, `[5,5,5]` physical fanouts, `global_eid` positive-edge mask, training negative sampler, and `_evaluate_split` pessimistic tie rule are retained. The Trainer copy omits Test negative candidates; the original split is reloaded after checkpoint selection for one Test evaluation.

## LP semantic context boundary

The adapter retains every cached V7B.1 semantic candidate for the endpoint queries and every original semantic weight and source flag. It maps global node IDs to the appended local node order; candidates are neither re-ranked nor truncated, and their weights are not renormalized. Late-Completion source nodes are added with their required semantic candidate rows. Their supplementary physical context is sampled only from the official Train physical graph with a fixed fanout of `[1,1,1]` for each of the three physical hops. This is an explicit approximation to the full physical receptive field and is not described as full-graph-equivalent. Supervised positive edges and reverse orientations are removed from the original sampled graph by the repository `global_eid` logic and filtered again from the supplementary context.

## Existing LP baseline audit

The repository contains completed MR-CGE V1.3 NC5/LP2 Validation records for the same official split and feature/cache fingerprints. Those records use the separate three-fold full-graph edge-holdout protocol, do not report Test metrics, and therefore are not strict comparisons for this final sampled-LP Test table. A repository-wide search found no other model's Test result for these two datasets under this exact split, feature version, metric definition, and protocol. No baseline score or missing standard deviation is fabricated here.

The final runners write one `result.json` per completed cell below `outputs/mrcge_final_benchmark/{nc,lp}/...`. The CSV files in `data/` are the run-level collection interface. Summary standard deviations must use the sample estimator (`ddof=1`) across the three seeds, never the best seed.
