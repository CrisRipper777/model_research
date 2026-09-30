# M0.1 — Adaptive Correction Attribution Closure

M0.1 closes the attribution questions left open by the frozen M0 screen. It compares parameter-matched extent-only capacity (A-wide) with learned correction, then separates correction granularity into global modality-specific (B-static), target-specific (B-target), and edge-specific (M0-B) controls. All three new variants use the frozen M0 backbone, relation evidence, edge support, optimizer, splits, and validation-selected checkpoint protocol.

This is an attribution experiment, not an architecture search. In particular, B-static has a global linear neighbor transform and tests parameterization/optimization; it does not add a new propagation function. Inference interventions measure reliance of an already trained checkpoint. They do not substitute for the independently trained B-target comparison.

## Design note

The external sanity review used the public papers as boundary checks: RoleMAG makes neighbor roles explicit, PLANET studies node-level granularity, CAMPA uses decoupled multi-hop propagation/alignment, and CoMAG adapts context to task. M0.1 imports none of those mechanisms. The purpose here is only to distinguish capacity, static correction, target-level correction, and within-target edge correction while keeping the problem evidence from P0–P1.3 separate from evidence that this particular learner converts it into task utility.

- [RoleMAG](https://arxiv.org/abs/2604.12271)
- [PLANET](https://arxiv.org/abs/2602.04116)
- [CAMPA](https://arxiv.org/abs/2605.11468)
- [Context-aware CoMAG](https://arxiv.org/abs/2606.14172)

## Variants

- `extent_wide`: two modality-specific `Linear(64,126)-GELU-Linear(126,1)` gates; extent control only.
- `single_basis_static`: original edge-specific gate and rank-32 correction basis, with one learned correction logit per modality.
- `single_basis_target`: original edge-specific gate, correction head, and rank-32 basis; correction head receives the mean functional state over each target's original non-self incoming edges, computed in two differentiable chunked passes.
- `single_basis`: original M0-B, frozen and reused from its nine committed checkpoints for attribution.

## Protocol and outputs

The formal campaign trains only the 27 new runs (3 datasets × 3 seeds × 3 variants). The nine M0-B checkpoints and committed SEM/UNI/A/B run metrics are reused. Node classification is validation-only; test indices are not attached to the loaded graph, test labels are masked before training/evaluation, and no link prediction is run.

Large run JSON, checkpoints, and smoke outputs are stored under the ignored `outputs/m01_adaptive_correction_attribution/`. Committed tables, figures, and the final interpretation are in this directory. Reproduce with:

```bash
conda run --no-capture-output -n yhf_env python scripts/run_m01_adaptive_correction_attribution.py --smoke --device cuda:0
conda run --no-capture-output -n yhf_env python scripts/run_m01_adaptive_correction_attribution.py --campaign --device cuda:0
conda run --no-capture-output -n yhf_env python scripts/run_m01_adaptive_correction_attribution.py --refresh-diagnostics --device cuda:0
conda run --no-capture-output -n yhf_env python scripts/analyze_m01_adaptive_correction_attribution.py
```

The frozen parent M0 artifacts in `research/m0_adaptive_propagation_screen/` are not modified. See [report.md](report.md), [run_manifest.json](run_manifest.json), `data/`, and `figures/` for results and provenance. Figure alignment, typography, collision, and panel-by-panel checks are recorded in [figure_qa_notes.md](figures/qa/figure_qa_notes.md).
