# E0.1 — Function-Provenance Preservation Screen

This is a validation-only, one-hop NC screen of whether the existing E0 Smooth, Relational, and Cross-Modal aggregated contexts provide incremental value when their function identity is retained until node-level composition.

The decisive comparison is `keep_edge` versus the same-capacity `premix_edge_control`: both use E0 EdgeMix routing and the same `Cmix` residual base, but the control feeds three repeated copies of `Cmix/3` to its composer. `keep_static`, `keep_target`, and `keep_edge` retain `[CS, CR, CX]` and differ only in the E0 routing granularity.

The campaign uses Movies, Grocery, and ele-fashion with seeds 42, 43, and 44. It exposes train/validation labels only; test evaluation and link prediction remain disabled. After the 36 runs completed, all selected checkpoints received a forward-only audit of E0's frozen calibration scales on validation-target edges. All 504 modality×diagnostic distributions were finite; no calibration change or clamp was applied.

Formal results, node-context diagnostics, matched interventions, calibration scales, and figure QA are documented in `report.md`, `run_manifest.json`, and `data/`.

Raw run records and checkpoints are written under the ignored `outputs/e01_function_provenance_preservation/` directory.
