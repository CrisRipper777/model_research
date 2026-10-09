# MvCGE-MAG V4A: Raw-Anchored Adaptive Role Residual Screen

This experiment tests whether the fixed modality-conditioned V3C role contrast is useful as a bounded residual on top of the complete Raw backbone, and whether that residual benefits from expert-specific gates or an independent four-hop profile.

## Frozen design

- Parent: V3C final commit `a9853de789fd52297b717e79c4fdb13ede738e0b`.
- Variants: `R0_raw`, `R1_global_residual`, `R2_expert_residual`, `R3_adaptive_residual`.
- Role assignment is inherited unchanged from V3C: modality-private frozen raw features, per-node LayerNorm, L2 normalization, physical-edge cosine, local incoming mean, and the fixed threshold `q >= 0` for Supportive versus `q < 0` for Discrepant.
- The Raw trajectory remains the model action. At each Raw hop, the previous Raw state is propagated over supportive and discrepant masked raw weights. Both messages share that Raw hop's channelwise ActiveRMS denominator. The role contrast is `S_support - S_discrepant`.
- The Raw expert MLP always receives the original Raw expert profile. The role contrast is added only after the Raw expert output, with zero-initialized `0.25*tanh(gate)` and detached, scalar, one-sided RMS safety calibration.
- R1 shares one residual gate; R2 uses one shared-across-modalities gate per expert; R3 also learns one shared-across-modalities normalized four-hop profile per expert.
- Node-conditioned gates, learned roles, HPO, Test evaluation, and all changes listed as prohibited in the experiment brief are excluded.

## Protocol

The preflight uses raw modality features, physical edges, and untrained model instances only. Smoke runs Movies/seed 42 for one epoch in each arm. The formal screen has 3 datasets (`Movies`, `Grocery`, `ele-fashion`) × seeds 42–44 × 4 variants = 36 full-graph NC runs. Checkpoints are selected by validation Accuracy; Macro-F1 is diagnostic. `task.evaluate_test=false` is required throughout.

Run entry points:

```bash
python scripts/run_mvcge_mag_v4a_screen.py --mode preflight --device cuda:0
python scripts/run_mvcge_mag_v4a_screen.py --mode smoke --device cuda:0
python scripts/run_mvcge_mag_v4a_screen.py --mode campaign --device cuda:0
```

Formal output and validation-selected checkpoint files are kept below ignored `outputs/mvcge_mag_v4a_raw_anchored_residual_screen/`; compact diagnostics and the final report are committed under this research directory.

## Interpretation boundary

Role contrast means relative Supportive-versus-Discrepant contribution. It is not labeled a heterophily or high-pass basis. All paired comparisons are descriptive; no significance or equivalence tests are run. The experiment ends after V4A analysis and does not automatically start V4B.
