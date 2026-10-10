# PSCE-MAG V7A

Branch: `exp/v7a_physical_semantic_collaborative_experts`
Baseline: `af848e6c241994d34827fab8d37611159d167965`

This directory contains the design/feasibility review, label-free candidate graph summary, full-suite test record, GPU preflight, and GPU smoke records. Candidate caches and checkpoints live outside Git under `outputs/psce_mag_v7a/`.

## Run sequence

```bash
conda run -n yhf_env python scripts/run_psce_mag_v7a.py --mode build-candidates
conda run -n yhf_env python scripts/run_psce_mag_v7a.py --mode raw-regression --device cpu
conda run -n yhf_env python scripts/run_psce_mag_v7a.py --mode tests
conda run -n yhf_env python scripts/run_psce_mag_v7a.py --mode preflight --device cuda:1
conda run -n yhf_env python scripts/run_psce_mag_v7a.py --mode smoke --device cuda:1
```

Formal training is gated on the full GPU preflight, all three variant smokes, and the full pytest record. It runs on a clean frozen commit:

```bash
conda run -n yhf_env python scripts/run_psce_mag_v7a.py --mode campaign --device cuda:1
```

Resume only after a recorded failed/interrupted formal cell:

```bash
conda run -n yhf_env python scripts/run_psce_mag_v7a.py --mode resume --device cuda:1
```

The three-dataset GPU preflight and A0/A1/A2 Movies GPU smoke gates passed on `cuda:1`; detailed resource and audit records are in `data/preflight_summary.json` and `data/smoke_summary.json`. The formal 27-run campaign starts only after the freeze commit is created and the worktree is clean.
