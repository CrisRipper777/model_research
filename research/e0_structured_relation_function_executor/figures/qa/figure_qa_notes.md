# E0 figure quality assurance notes

## Figure contract

- **Archetype:** quantitative grid; each figure has a single diagnostic claim.
- **Core claim:** tested edge-specific function composition adds little over target-level composition on this benchmark, while route use and within-target variability are dataset-dependent.
- **Backend/export:** Python/matplotlib; editable PDF/SVG and 600-dpi PNG/TIFF, width 182.9 mm. Every PDF is text-selectable and passed the 5 pt glyph-floor audit.
- **Source data:** run-level and diagnostic CSVs in `../../data/`; E0 metrics are validation-only. UNI/SEM are historical M0 references. No significance tests were run.
- **Replicate unit:** dataset×seed training run; n=3 seeds for dataset panels and n=9 equally weighted dataset×seed runs in pooled diagnostics. SD is population SD unless the panel is a box plot of run-level summaries.

## Panel roles and uncertainty

| Figure | Panel role | Summary/spread | QA result |
| --- | --- | --- | --- |
| `e0_performance_screen` | Validation Accuracy, Macro-F1, and CE for SEM/UNI/Static/Target/Edge | Mean ± population SD over 3 seeds; no inferential test | Alignment PASS; text-floor PASS; collisions PASS |
| `e0_routing_usage` | Mean function composition by dataset, variant, and modality | Stacked mean composition over seeds. Segment-wise uncertainty is not drawn because components are constrained to sum to one; per-run SD and quantiles are in `router_diagnostics.csv`. This is descriptive. | Alignment PASS; text-floor PASS; collisions PASS |
| `e0_within_node_routing` | EdgeMix within-target L1 variation, Text and Visual | Box plots of three seed-level medians, separately for each dataset; full target/seed quantiles are in CSV | Alignment PASS; text-floor PASS; collisions PASS |
| `e0_modality_disagreement` | Text/Visual pi L1 disagreement on shared physical edges | Mean ± population SD over 3 seeds | Single panel; alignment NOT APPLICABLE; text-floor PASS; collisions PASS |
| `e0_function_distinctness` | Per-edge executor cosine similarities, EdgeMix | Box plots of three seed-level medians for each dataset×modality; full edge quantiles are in CSV | Alignment PASS; text-floor PASS; collisions PASS |
| `e0_interventions` | Fixed EdgeMix checkpoint reliance effects | Mean ± population SD over run records; within-target shuffle pools five repeats per run (45 records), all other interventions have 9 records | Alignment PASS; text-floor PASS; collisions PASS |

The source preflight reports 21 PASS, 0 WARN, 0 FAIL. Five multi-panel figures passed the 1.5 pt alignment gate; the one single-panel figure is correctly marked not applicable. Six PDF text audits found zero glyphs below 5 pt (minimum found 5.5 pt). Six rendered collision audits report zero failures and zero warnings. The PNGs were inspected panel by panel and as complete figures after the last geometry changes.
