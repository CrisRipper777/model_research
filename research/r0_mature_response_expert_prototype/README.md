# R0 — Mature Structural-Response Expert Prototype

This directory contains the validation-only report bundle for the fixed R0 campaign. The four variants share exact initialized modules and total parameter counts; only the active forward path and CrossMoE orthogonality objective differ.

- `report.md`: results, bounded interpretation, and final route decision.
- `figure_qa.md`: source, alignment, collision, text-size, and visual-review results.
- `run_manifest.json`: source SHA, protocol, run count, failures, device, and execution metadata.
- `data/`: per-run performance and validation diagnostics, paired comparisons, initialization/capacity audits, orthogonality records, and checkpoint interventions.
- `figures/`: Python-rendered PNG/TIFF previews and editable vector SVG/PDF figures with alignment QA manifests.

All metrics are validation-only. Three-seed summaries use population SD. Pooled nine-pair summaries are descriptive and are not IID significance tests. Checkpoints and large per-node routing logs are stored in the ignored `outputs/r0_mature_response_expert_prototype/` directory.
