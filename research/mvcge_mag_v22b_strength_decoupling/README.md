# MvCGE-MAG V2.2b: Strength-Decoupled Routing Repair

## Purpose and frozen protocol

This is an interpretation-repair screen, not a router-complexity extension. It separates (1) whether node-specific selection remains useful when modality strength is an independent parameter and (2) whether structural evidence and static-centered residualization help free node routing.

- Parent: V2.2 final commit `d1d7bbd4b795abd014cec6fdc194dc056f6d058c`.
- Task: `unified_full_graph_nc_v1`; Movies, Grocery, `ele-fashion`; seeds 42–44; four variants; 36 runs.
- Select checkpoints by Validation Accuracy. Set `task.evaluate_test=false`; do not evaluate Test labels or metrics.
- No HPO, significance test, LP, fifth variant, or post-freeze model/config edit.
- Freeze the model/config after targeted tests, full tests, and a 4/4 smoke; launch the formal campaign only at the freeze commit with a clean worktree.

## Fixed architecture

All four variants share the V2.2 four-expert structural bank, normalized order-4 physical trajectory, 256-dimensional hidden state, 64-dimensional router state, fixed Top-2, and active-only V2.2 balance loss. Expert definitions and physical propagation are copied from V2.2. Reliability and hop statistics are router evidence only; they never reweight propagation. There is no cross-modal input or expert compatibility behavior.

The legacy `strength_head` and compatibility modules are constructed in V2.2 order to retain initialization/RNG compatibility, then frozen. The only strength used by the forward pass is `sigmoid(direct_strength_raw[m])`, with two constant-initialized trainable scalars for Text and Visual. Thus strength has no input, graph, evidence, or router dependency.

| Variant | Selection logits | Evidence | Residual |
|---|---|---|---|
| D0_static | Broadcast modality-static logits | No | No |
| D1_free_node | V2.2 R1 old free-node context | No | No |
| D2_struct_free | Structure-grounded free-node logits | Yes | No |
| D3_struct_residual | Static + sigmoid(eta) × (structure-free − static) | Yes | Yes |

Every variant uses the same direct modality strength and V2.2 balance formula.

## Interpretation rules fixed before the campaign

“Approximately equal” is a descriptive label when both `abs(overall Δ Accuracy) <= 0.15 pp` and `abs(overall Δ Macro-F1) <= 0.50 pp`; it is not an equivalence test. “Stable positive” requires overall Accuracy and Macro-F1 deltas above zero, at least 6/9 positive Accuracy pairs, and positive Accuracy means on at least 2/3 datasets. No significance claims are made.

- A: A stable-positive D1−D0 supports node-specific selection after strength decoupling.
- B: D1−D0 approximately equal to or below zero means the old node-routing signal is unstable or absent after decoupling.
- C/D: D2−D1 above zero on both headline metrics supports incremental structural observation; approximate equality indicates no stable task value from this evidence.
- E/F: D3 below/above D2 describes whether static-centered residualization hurts or may help, interpreted alongside learned eta and hard-Top-2 pair-change fractions.
- G: If neither D1 nor D2 is stable-positive versus D0, stop router-centric optimization. The next research question is whether modality-conditioned effective structural contexts can improve a shared structural expert action space. This experiment does not implement that mechanism.

Regardless of outcome, stop after this screen. Do not automatically begin V3. If a D1 or D2 stable-positive signal appears, retain node routing only as a future candidate after validating the effective structural action space.

## Artifacts

Tracked validation summaries and diagnostics are under `data/`. Checkpoints, model weights, raw tensors, embeddings, caches, and training/Hydra logs stay under ignored `outputs/` on the server.
