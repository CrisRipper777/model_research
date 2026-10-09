# MvCGE-MAG V3C: Functional Role Action-Space Screen

V3C asks whether two fixed, modality-local functional roles on observed
physical edges create useful structural actions: Supportive smoothing and a
Discrepant channel tested as either ordinary smoothing (F2) or signed
representation difference (F3).

The role score uses detached raw modality features after parameter-free
per-node LayerNorm and L2 normalization:

```text
cos_ij = cosine(x_i, x_j)
mu_i = incoming physical-neighbor mean cosine
role_score_ij = cos_ij - 0.5 * (mu_i + mu_j)
Supportive: role_score >= 0
Discrepant: role_score < 0
```

Ties belong to Supportive. The two role weights are masks of the V2.2/V3A raw
normalized physical edge weights, with no role-specific renormalization. The
partition identity is checked edge by edge. Discrepant is a descriptive
relative-consistency label, not ground-truth heterophily.

Variants are fixed: `F0_raw`, `F1_support`, `F2_role_dual_smooth`, and
`F3_role_functional`. All use the V3A C0/V2.2 R0 static Top-2 router and
strength controller, four shared experts, and the same alpha profile. F3
differs from F2 only by replacing Discrepant smoothing with the signed action
`T_disc_(k-1) - T_disc_k`.

## Protocol

- Validation-only `unified_full_graph_nc_v1`; Movies, Grocery, `ele-fashion`;
  seeds 42–44; 36 runs total.
- `task.evaluate_test=false`; checkpoints selected by Validation Accuracy.
- No labels/splits in role preflight; no node routing, learned edge router,
  new edges, cross-modal operations, HPO, LP, Test evaluation, or significance
  testing.
- Run the V3C targeted tests, full test suite, label-free role preflight, and
  four-variant Movies/seed 42 one-epoch smoke before freezing the model/config.
- Start formal training only at the recorded freeze SHA with a clean worktree.

The final `REPORT.md` and CSVs report role fractions and coverage, cross-modal
role disagreement, trajectory similarity/displacement, action-space span
novelty, profile and routing diagnostics, correction scale/cosine, and the
historical F0 comparison. Checkpoints and logs remain under ignored
`outputs/` and are not committed.

For the qualitative interpretation map only, functional-union novelty at or
below `1e-3` is called near-zero, and a Text/Visual role disagreement fraction
above `0.5` means a majority of physical edges change role. These are
descriptive labels fixed before the campaign, not tuning targets.

See [`V3A_INTERPRETATION_NOTE.md`](V3A_INTERPRETATION_NOTE.md) for the narrowed
V3A conclusion that motivated this screen.
