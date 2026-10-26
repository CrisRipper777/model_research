# MR-CGE V1.3 Intrinsic-Prior Fusion — Design Freeze

## Objective and frozen comparison

This screen tests one change to the V1.3-A `M_repaired` / `joint_view` model: a zero-initialized scalar controls how the original intrinsic projector outputs enter the existing final linear skip. All private, collaborative, joint-view, completion, router, classifier, and loss paths remain frozen.

The historical comparison is the same-dataset, same-training-seed V1.3-A Joint View row, selected by Best Validation Accuracy. Best-F1 checkpoint metrics remain a separate secondary comparison. No historical checkpoint is used to initialize a new run.

## Frozen forward equation and insertion point

The parent model computes the two intrinsic projector outputs once:

`h_text = projectors.text(raw_text)` and `h_visual = projectors.visual(raw_visual)`.

After unchanged Private, Collaborative, Joint View, and Late Completion paths, it forms:

`F = cat(final[text], final[visual])`

`H0 = cat(h[text], h[visual])`

`MLP_f(F) = fusion_linear2(fusion_dropout(GELU(fusion_linear1(F))))`

The frozen output is `z_old = fusion_norm(fusion_skip(F) + MLP_f(F))`. The sole change is a deterministic scalar parameter `prior_skip_raw = 0`, with `lambda_prior = tanh(prior_skip_raw)`, and:

`z_new = fusion_norm(fusion_skip(F + lambda_prior * H0) + MLP_f(F))`.

The existing `fusion_skip` is called once in the output path, so its bias is added once. `MLP_f` continues to receive `F`; the same already-computed `H0` is used without detach. No other model state or intermediate is changed. A finite copied `forward` override is allowed because the frozen parent has no Fusion hook. A mechanical source comparison will document that the computational change is limited to the scalar, `H0` concatenation, and final skip input; all other copied statements must match the frozen parent.

The interpretation is limited to separately reweighting intrinsic projector output in the final skip relative to the graph/cross residual already present in `F`. It does not add information unavailable to the parent. Negative lambda means suppression of the original intrinsic contribution.

## Initialization, gradients, and parameter count

The subclass calls the frozen parent constructor without changing module order, then adds `nn.Parameter(torch.zeros(()))`. This consumes no random numbers. The model must have exactly one more scalar parameter, preserve every parent parameter and buffer byte-for-byte at fixed seed, and leave the next external NC classifier initialization unchanged.

Hard pre-training tests compare evaluation embedding/logits/losses/routes/late outputs, training loss and all shared gradients under replayed RNG, RNG state, classifier state, intermediate state invariants, and strict reload. The new scalar must have finite, nonzero gradient on a non-degenerate synthetic graph and be present in the optimizer. Tests also cover finite outputs for empty physical support, zero cross, and isolated nodes.

## Frozen data and training protocol

- Base commit: `184deaaf12b21c9ee01d3328aef15d86bcc50346`; base branch: `exp/mrcge_v13_two_hop_cross_screen`.
- New branch: `exp/mrcge_v13_intrinsic_prior_skip_screen`.
- Model config: `M_repaired`, `joint_view`, `sender_private_ratio=0`, hidden 256, Physical Hops 3, Semantic Hops 1, 4 shared experts, Top-2, expert bottleneck 64, SameNode and Joint View enabled.
- Train configuration is constructed by reusing `scripts/run_mrcge_v13_screen.py`: AdamW, LR `1e-3`, weight decay `1e-4`, 300 epochs, patience 30, minimum epoch 30, min delta `1e-4`, clip 1.0, existing dropout and K=8.
- Loss is unchanged: main CE + `0.3 * private auxiliary CE` + the model's existing `0.01 * balance loss` exactly once.
- Fixed split seed 42; train seeds are Movies 42/43/44 and Grocery 42/43; ele-fashion 42/43 is conditional. Reuse V7B.1 K=8 caches and verify feature, graph, split, split file, semantic cache, candidate, V7B, node order, class count, and historical checkpoint fingerprints.
- `auto_generate_splits=false`, Test labels masked to `-100` before Trainer, `evaluate_test=false`, and no link prediction. Validation alone controls early stopping, checkpoint selection, and the screen.
- Each run saves separate Best-Accuracy and Best-Macro-F1 checkpoints, full epoch history, run metrics, scalar trajectory, resource metrics, and both checkpoint hashes; both checkpoints are reloaded and compared with Trainer selection metadata.

## Fixed experiment matrix and promotion gates

Stage 1 is five full training runs, all with `intrinsic_prior_skip`:

| Dataset | Training seeds | Split seed |
|---|---|---:|
| Movies | 42, 43, 44 | 42 |
| Grocery | 42, 43 | 42 |

Pair each Best-Accuracy result to the verified same-seed V1.3-A reference. Report per-seed deltas and dataset-specific means and population SDs.

Movies passes only if all are true: mean Delta Accuracy at least +0.20 pp; at least two of three seeds improve Accuracy; mean Delta Macro-F1 at least −0.10 pp; mean Delta CE no more than +0.020. Grocery must have mean Delta Accuracy at least −0.10 pp, mean Delta Macro-F1 at least −0.30 pp, and mean Delta CE no more than +0.020. Also require all five runs complete, finite, strict-reload verified, Test-off, hash protected, exact data/cache fingerprint matches, and the new scalar demonstrably trainable. These are screening criteria, not significance tests.

Stage 2 is two ele-fashion runs (seeds 42 and 43) and is authorized only if every Stage 1 criterion passes. Otherwise record No-Go and stop. The total training cap is seven.

## Required mechanism diagnostics

At each epoch retain `prior_skip_raw` and `tanh(prior_skip_raw)`. At Best-Accuracy checkpoint reload, compute on the same Eval forward: `RMS(H0)`, `RMS(F)`, `RMS(F-H0)`, `RMS(W_s F)`, `RMS(lambda W_s H0)`, `RMS(MLP_f(F))`, prior-to-old-skip RMS ratio, and their cosine. For these diagnostics, use `fusion_skip.weight` without bias. Report Train and Validation Accuracy, Macro-F1, and CE and the Train−Validation Accuracy/CE gaps. Lambda nonzero or a large prior RMS alone is not a causal explanation or evidence of performance benefit.

## GPU access and fail-closed behavior

Only physical GPU 1 is permitted, at most one isolated training child at a time. Before launch, verify GPU UUID, external compute PIDs, free memory and measured-peak margin, and pass a real single-step forward/backward for each formal dataset/configuration, including both checkpoint paths as relevant. Do not use GPU 0, stop other processes, or alter the frozen training configuration. This study does not inherit the shared-GPU guard exception from the prior study. If the sandbox cannot provide a verifiable fresh GPU snapshot, record the access block and request the required execution permission; never record an unverified guard pass.

## Stop and interpretation rule

After the Stage 1 decision, run only the conditionally authorized Stage 2 or record No-Go and stop. Do not adjust the model or thresholds after validation, merge this variant into the default model, or begin another ablation. Any positive screen remains exploratory and requires independent review.
