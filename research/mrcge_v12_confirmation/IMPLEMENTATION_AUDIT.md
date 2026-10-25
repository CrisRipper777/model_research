# Implementation Audit

The independent runner is `scripts/run_mrcge_v12_confirmation.py`; the analyzer is `scripts/analyze_mrcge_v12_confirmation.py`. New writes use `outputs/mrcge_v12_confirmation/` and `research/mrcge_v12_confirmation/`. The seed-42 V1.2 output and research folders are read-only inputs.

The runner accepts dataset, condition, and seed filters, reserves unique attempt directories, verifies checkpoints before skipping completed cells, and records failures/guard blocks without overwriting attempts. All formal and preflight GPU work uses physical GPU 1 under the copied V1.2 ownership and 2-GiB headroom guard.

The CPU preflight compares actual M state tensors and Fashion S/F initial logits. The GPU single-step preflight checks finite losses/gradients, initialization-seeded forward/backward behavior, RNG, split/cache fingerprints, Test-off, and resource peaks. Each formal worker reloads both selected checkpoints and reproduces Validation metrics before recording completion.
