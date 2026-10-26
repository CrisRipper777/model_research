# Implementation audit

Base `788c35392f049fea32f42f5081656924888f7082`; analysis freeze `63b8f94a4d44a47099c56a544ad0a488f2fd7403`. Independent operator reconstruction is in `scripts/mrcge_v13_propagation_audit_ops.py`; stable blockwise FP64 diagnostics are in `scripts/mrcge_v13_propagation_audit_math.py`. The seven B0–B3/X trajectories use frozen V1.3 methods. Hybrid uses its own `Model._normalized_operator`, `_physical_states`, `_active_rms_norm`, `_physical_cross_message`, and trained alpha path. No model source/config was edited.

CPU suite: 13 passed. Small-graph tests compare frozen physical operator, basis, early Cross, X2/X3, full eval forward state capture, empty graph, zero Cross, parameter and RNG immutability.

Independent X1 reconstruction is checked against `_last_state_audit['early_cross']` at `1e-6` relative Frobenius and `1e-4` maximum absolute tolerance. Every full cell checks checkpoint state immutability and CPU/CUDA RNG invariance.

Historical V1.3 source difference: early Movies/Grocery/Fashion seeds use model SHA `a70e2f…`, while the current base uses `be9272…`; Movies seed44 uses `be9272…`. `git diff` shows only the later addition of audit/contribution captures and algebraically identical assignments in `forward`; `_physical_operator`, `_basis`, `_early_cross`, `_filter`, `_m_filter_impl`, and `_late` are unchanged. The later capture fields enable the true private SameNode check. Historical SHA identities are retained per checkpoint.
