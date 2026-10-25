# Implementation Audit

## Frozen source boundary

The implementation is isolated in:

- `src/models/mrcge_v13_basis_screen.py`
- `configs/model/mrcge_v13_basis_screen.yaml`
- `configs/study/mrcge_v13_physical_basis_screen.yaml`
- `scripts/run_mrcge_v13_physical_basis_screen.py`
- `scripts/analyze_mrcge_v13_physical_basis_screen.py`
- `tests/test_mrcge_v13_physical_basis_screen.py`

The parent V1.3 model and original Hybrid-CGE models/wrapper, shared NC Trainer, historical model configs and historical campaign runners are hash-checked against the required base. The wrapper re-registers the original expert bank's existing Parameter and module objects without creating layers or consuming random state. Model Factory loading is covered by the new test.

## Correctness coverage

The focused tests cover bitwise initial state and classifier equality with frozen V1.3-A when the switch is off; Evaluation embedding/logit/auxiliary-loss equality; same-RNG training output and gradient equality; Physical-only signed-L1 NoP0 coefficients; unchanged Semantic orders 0–1; theta trainability; P0 retained in propagation; Private/Collaborative diagnostic masking; full expert-stack versus low-memory checkpointed output, auxiliary loss and gradients; paired dropout at zero Cross; no-support behavior; both Early Cross directions; strict checkpoint reload and Validation metrics; and positive SameNode support, candidate RMS and source weight for corrected Hybrid H2 3-hop.

The launch preflight additionally checks finite one-step loss/gradients, Test masking, Validation RNG stability, model-specific memory use, live SameNode evidence for Hybrid, and masked effective coefficients for NoP0. Formal run records contain resolved config, code/data/cache/split hashes, init hashes, epoch metrics, checkpoint hashes and reload audits.

## Verification performed before freeze

`conda run -n yhf_env python -m pytest -q tests/test_mrcge_v13_physical_basis_screen.py`: **11 passed**.

`conda run -n yhf_env python -m pytest -q tests/test_mrcge_v13_physical_basis_screen.py tests/test_mrcge_v13.py tests/test_hybrid_cge_paper_evidence_v1.py tests/test_hybrid_cge_v11.py tests/test_mrcge_v12.py`: **65 passed**, 21 warnings (PyTorch Geometric/checkpoint deprecation warnings).

No historical source, record, or checkpoint was edited. The CPU preflight records a SHA256 inventory of all files under the three protected historical research directories and rechecks it during final analysis.

Implementation freeze commit: `cb641c4947c13d2305cfe83110cfdf9e35d42517`.
