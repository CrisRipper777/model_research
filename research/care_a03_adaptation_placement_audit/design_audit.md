# CARE-MAG A0.3 design and protocol audit

## Provenance

- Required parent SHA: `a44a556517112f15a8550f81c4b916906756d77f`.
- Experiment branch: `exp/care_a03_adaptation_placement_audit`.
- The runner verified that the branch is descended from the required SHA before tests, smokes, and the campaign.
- The work used the authorized A0.2 source branch only; no other historical experiment branch was inspected or used.

## Question and design

The 2×3 factorial separates recipient-specific hop utilization from residual feature correction. Trajectory factors are `global` and `node`; adapter factors are `off`, `static`, and `context`. G0/GS/GC and N0/NS/NC share the v0 module construction, parameter layout, propagation, initialization and fusion. No trainable module was added. Global hop scores reuse each modality's v0 `Wp`, `Wr`, hop embeddings, and score vector on graph means, then broadcast the resulting hop probabilities to all nodes.

The node × adapter cells are regression mapped to the v0 structural, static and context variants with strict state loading and numerical equality checks.

## Frozen training protocol

- Datasets and split: Movies, Grocery and ele-fashion; model/training seeds 42, 43, 44; each dataset's configured NC split stays fixed.
- Task: full-graph NC, validation Accuracy checkpoint selection, AdamW, and the repository's unchanged learning rate, weight decay, stopping, clipping and evaluation configuration.
- `task.evaluate_test=false` and `task.development_no_test=true` for every formal run. The formal analyzer rejects any `test_*` metric key.
- 18 dataset × architecture configurations × 3 paired seeds = 54 runs. The launcher stops on the first failed configuration and can resume completed configurations.
- No NC test, formal LP campaign, hyperparameter adjustment or A3/MoE experiment is part of this stage.

## Required gates

- Full repository tests: passed (exit 0).
- Movies, seed 42: six architecture cells, two epochs each; each must train, validate and save a checkpoint.
- LP is execution smoke only: sports-copurchase, GC and NC, two epochs and at most two sampled training batches, with test evaluation disabled. It must report `LinkNeighborLoader`, fanouts `[5,5,5]`, positive supervision edge removal, forward/backward, validation inference and a saved checkpoint.
- Formal NC campaign status: complete.

## Analysis plan

Validation Accuracy is primary because it selects checkpoints. Paired effects are calculated within dataset and seed. Accuracy and Macro-F1 deltas are expressed in percentage points; CE deltas remain on the original scale. Means and population SDs are descriptive; no p-values are reported. Macro-F1 and CE are secondary consistency signals, and conflicts are labeled as trade-offs.

Direct retraining comparisons estimate architecture value. Trajectory interventions measure reliance of an already-selected checkpoint on node-to-hop correspondence. Adapter interventions analogously probe coefficient correspondence at GC/NC selected checkpoints. These evidence types remain separate. The descriptive “clear positive” rule is a mean Accuracy gain of at least 0.5 points with at least two of three paired seeds positive; values within ±0.5 points are called approximately zero. This rule is for readable conservative interpretation, not a significance threshold.

## No tuning / stop boundary

Model widths, graph processing, optimizer and all training settings remain frozen. Results are not used to modify the model or continue the campaign with different settings. After this audit, stop for human review; do not start A3 MoE, dynamic trust, LP campaign or NC test.
