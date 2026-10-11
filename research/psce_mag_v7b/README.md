# PSCE-MAG V7B

V7B evaluates relation provenance and relative semantic edge weights under the frozen V7A full-graph MAG node-classification body.

Read `FEASIBILITY_REVIEW.md`, `DESIGN.md`, then `REPORT.md`. Reproduction commands from the repository root use the `yhf_env` interpreter:

```bash
python scripts/run_psce_mag_v7b.py --mode build-candidates
python scripts/run_psce_mag_v7b.py --mode build-random-graphs
python scripts/run_psce_mag_v7b.py --mode raw-regression --device cuda:1
python scripts/run_psce_mag_v7b.py --mode tests
python scripts/run_psce_mag_v7b.py --mode preflight --device cuda:1
python scripts/run_psce_mag_v7b.py --mode smoke --device cuda:1
python scripts/run_psce_mag_v7b.py --mode campaign --device cuda:1
```

Formal artifacts are under `data/`. Model checkpoints and FAISS caches are local under `outputs/psce_mag_v7b/` and are not committed.
