# Preflight report

- Branch/base: `exp/mrcge_v13_propagation_mechanism_audit` / `788c35392f049fea32f42f5081656924888f7082`.
- Analysis source freeze: `fe0e390e2d2282f8dcb761ab327b381f942f5933`.
- CPU data/cache/split preflight: PASS.
- GPU1 ownership guard: pass.
- GPU cells complete: 5/10; failed/blocked: 1.
- Total measured forward+diagnostic time: 3.7 s; maximum CUDA reserved: 1.90 GiB.
- Per-cell VRAM free baseline, UUID/PIDs, peak allocated/reserved, and worker logs are recorded under `data/`.
- Launch guard requires 8 GiB free; during-run guard requires at least 2 GiB free. Only physical GPU 1 was targeted.
