# N1 figure QA record

## Evidence and figure contracts

These are validation-only diagnostic figures for the N1 mechanism screen. Source data come from the completed 36-run campaign summaries in `../data/`; the plot code is `scripts/analyze_n1_recipient_function_strength_mixer.py`.

| Figure | Evidence shown | Summary and uncertainty |
|---|---|---|
| `n1_performance` | Validation accuracy, Macro-F1 and cross-entropy by dataset and trained mixer variant | Mean ± population SD (`ddof=0`) across three seeds per dataset/variant. Accuracy and Macro-F1 are fractions; CE is native scale. |
| `n1_paired_deltas` | Four seed-matched variant comparisons for each dataset | Mean ± population SD across three matched seeds. Accuracy and Macro-F1 are percentage points; CE is native scale. Every metric has its own axes. |
| `n1_strength_usage` | Text-modality D/P strength, recipient-wise variation and effective alternative-to-Smooth contribution for Same/Cross variants | Each dot is the mean across seeds and each error bar is population SD across the three seeds. The plotted diagnostic within each run is strength mean, validation-node SD, or effective-contribution median, respectively. |
| `n1_channel_scale` | D/Smooth and P/Smooth channel RMS ratios by dataset and variant | Mean ± population SD across three seeds, using each run’s validation-node median ratio. |
| `n1_interventions` | Inference-only intervention deltas on trained CrossState checkpoints | Mean ± population SD across the three runs. Node-beta shuffle aggregates five deterministic repeats per run (15 records per dataset); other interventions have one record per run. Accuracy and Macro-F1 use percentage points; CE is native scale. |

Points are unconnected because the x positions are categorical model variants, metrics, or interventions. No confidence intervals or significance tests are implied. The paired comparisons and checkpoint interventions answer different questions; see `../report.md` for interpretation.

## Rendering and audit

- Backend: Matplotlib Python (`Agg`); editable text retained in PDF and SVG.
- Raster exports: PNG and LZW-compressed TIFF at 600 dpi. Vector exports: PDF and SVG.
- Strict panel-alignment gate: all five figures passed at 1.5 pt tolerance; 18 panel comparisons, zero failures and warnings. The gate records are `*.alignment.json`.
- Static source preflight: 20 passes, zero failures. Its one warning is the wide report-scale layout; it found 317.5 mm from the source `figsize`. Final PDF page widths are 304.2–354.7 mm. No target journal or column width was specified for this internal screen, so the report-scale dimensions are retained. Resize and re-audit if preparing a journal submission.
- PDF text audit: all five pass a 7 pt minimum; the smallest observed run is 7 pt and none falls below the threshold. Machine-readable records are `qa/*.pdf-text.json`.
- PDF collision audit: all five pass with zero text collisions, clipping findings, or warnings. Machine-readable records are `qa/*.collision.json`.
- Visual inspection: all five final 600 dpi PNG exports were inspected. Legends and tick labels are clear; the strength chart aggregates all three seeds; unlike-unit metrics are unconnected and shown on separate axes.

The collision overlays used for review are temporary files under `/tmp/n1-collision/`; only the audit records are part of this research artifact.
