# MvCGE-MAG V2.1: Expert Utilization Granularity Screen

This branch screens four matched variants using the same protected intrinsic path, RawPoly order-4 trajectory, four shared structural experts, Top-2 routing, late fusion, and active-only load-balance loss:

- `U0_modality_static`: selection and strength share one modality-level context.
- `U1_node_selection`: selection is node-conditioned; strength stays modality-static.
- `U2_node_selection_strength`: selection and strength share a node context and regress to V2 `M1_direct_moe`.
- `U3_collaborative`: selection and strength share a context that also includes the opposite modality's router state.

The frozen campaign is validation-only: Movies, Grocery, and ele-fashion; seeds 42–44; 36 total runs; checkpoint selection by Validation Accuracy; `task.evaluate_test=false`. It does not use an external anchor, private experts, discrepancy or contrastive losses, HPO, LP, or layer-wise stacking.

See [REPORT.md](REPORT.md) and `data/` for compact results and diagnostics. Checkpoints and raw Hydra outputs stay under the ignored `outputs/` directory.
