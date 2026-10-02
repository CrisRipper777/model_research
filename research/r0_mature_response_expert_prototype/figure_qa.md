# Figure QA

Final QA of the six Python/Matplotlib figures in this report bundle, 2026-10-02.

- Plotting-source preflight: 21 PASS, 0 WARN, 0 FAIL.
- Strict panel alignment: all six PASS at 1.5 pt tolerance. Comparable-panel checks: performance 2, paired deltas 2, response bank 1, MoE specialization 4, protected attention 4, interventions 2.
- PDF collision audit: all six PASS, with 0 warnings and 0 failures.
- PDF text audit at a 5 pt minimum: all six pass; the smallest glyph is 5.7 pt.
- Visual review: inspected every panel in each final exported figure. The protected-attention y-axis label was shortened after the first audit found an overlap with its panel label; the revised PDF passes collision and alignment audits.
- Exports: 600 dpi PNG and losslessly LZW-compressed TIFF, editable vector PDF and SVG, plus the Matplotlib alignment manifests.

Figures use population standard deviations across three seeds for performance uncertainty and matched seed-level changes for paired comparisons. Source data are the CSV files in `data/`.
