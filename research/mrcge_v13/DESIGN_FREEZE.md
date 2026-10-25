# MR-CGE V1.3 Design Freeze

## Scope

This screening study starts from V1.2 confirmation commit
`a7930442e85ff9450fded54d5869915b434c7431` on
`exp/mrcge_v12_multiseed_confirmation`. The implementation branch is
`exp/mrcge_v13_view_cross_screen`.

The only backbone is `M_repaired`. The two formal conditions are independent:

- `joint_view`: V1.3-A only.
- `sender_private`: V1.3-B only, with a fixed sender-private ratio of `0.3`.

The authorized matrix is Movies, Grocery, and ele-fashion × those two variants
× seeds 42 and 43 (12 cells). `base` is an initialization and compatibility
condition only. A+B, seed 44, Toys, Reddit-S, link prediction, and Test
evaluation are outside this screen.

## Frozen backbone and training protocol

V1.3 preserves the V1.2 `M_repaired` implementation: Text/Visual private
projectors; Physical order 3 and Semantic order 1; four shared signed
polynomial experts; the graph-aware Top-2 router; `eta=0.1`, bottleneck 64,
hidden dimension 256; paired Private/Collaborative dropout; Early Physical
Cross projection, calibration, and beta; Late Semantic/Same-node/None,
Sparse Attention, Channel Gate, residual cap, and dropout 0.15; Fusion; private
auxiliary CE and expert-balance loss. `supports_link_prediction` remains false.

The run protocol matches V1.2: full-graph node classification, AdamW with
learning rate `0.001` and weight decay `0.0001`, model dropout `0.2`, at most
300 epochs, patience 30, minimum epoch 30, minimum accuracy improvement
`1e-4`, gradient clipping 1.0, private CE weight 0.3, balance weight 0.01,
Best Validation Accuracy selection, and a separately saved Best Macro-F1
checkpoint. Test labels are set to `-100` before trainer entry and Test
evaluation is disabled.

The seed changes initialization and training randomness only. Movies and
Grocery use the frozen seed-42 node split for both seeds; ele-fashion uses its
frozen `split.pt`. All conditions use the verified V7B.1 Text/Visual K=8
semantic cache and the same feature, physical graph, semantic cache, and split
fingerprints as the corresponding V1.2 seed reference.

## V1.3-A: Joint View Contribution

For each modality, the module receives
`[H, rho*R_P, gamma*R_S, abs(rho*R_P-gamma*R_S), (rho*R_P)*(gamma*R_S)]` and
returns `2*sigmoid(Linear(32,2)(GELU(Linear(5H,32)(Q))))`. The same module is
shared by Text and Visual. Its output layer starts at zero, so both initial
scales equal one. Its construction is inside a forked CPU RNG context, keeping
the V1.2 module initialization and the later external classifier initialization
unchanged.

The first linear projection is evaluated as five sliced linear products and
summed. This is mathematically equivalent to the concatenated linear layer and
avoids allocating an `[N,5H]` input. No dropout, normalization, regularizer, or
additional output is added.

Private state is `H + sP_priv*(rho*R_P_priv) + sS_priv*(gamma*R_S_priv)`. After
the unchanged Early Cross and collaborative Physical expert recomputation,
`sP_collab` is evaluated from the collaborative Physical response and private
Semantic response. The Collaborative update is
`U_priv + rho*(sP_collab*R_P_collab - sP_priv*R_P_priv)`; it adds no second
Semantic update. Existing support masks are retained. A zero Cross context
uses the existing paired-dropout replay and fallback, preserving the
Private/Collaborative equality invariant in training mode.

## V1.3-B: Private-informed Physical Cross

Only the source tensor passed to each unchanged Early Cross projection changes:
`Sender = 0.7*H_source + 0.3*U_priv_source`. The same rule applies in both
directions, with no detach. The source is the source modality's Private state;
the receiving side still uses the V1.2 physical graph, normalization,
calibration, beta, expert, router, and Private/Collaborative difference. Ratio
zero is accepted only for compatibility checks; formal B runs use 0.3.

## Comparison and interpretation

Each V1.3 result is compared to the corresponding V1.2 `M_repaired` dataset and
seed row from `combined_selected_runs.csv`. The historical V1.2 result is a
separate campaign, not a same-job re-run. Two-seed averages are descriptive.
This screen will not claim statistical significance, strict causal attribution,
SOTA, or that view strengths recover true graph reliability. No automatic A/B
winner will be selected.
