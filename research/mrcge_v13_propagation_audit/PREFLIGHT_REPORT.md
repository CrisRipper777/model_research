# Preflight report

- Branch/base: `exp/mrcge_v13_propagation_mechanism_audit` / `788c35392f049fea32f42f5081656924888f7082`.
- Analysis source freeze: `63b8f94a4d44a47099c56a544ad0a488f2fd7403`.
- CPU data/cache/split preflight: PASS.
- GPU1 ownership guard: pass.
- GPU cells complete: 10/10; failed/blocked: 0.
- Total measured forward+diagnostic time: 7.7 s; maximum CUDA reserved: 8.27 GiB.
- Per-cell VRAM free baseline, UUID/PIDs, peak allocated/reserved, and worker logs are recorded under `data/`.
- Launch guard requires 8 GiB free; during-run guard requires at least 2 GiB free. Only physical GPU 1 was targeted.
