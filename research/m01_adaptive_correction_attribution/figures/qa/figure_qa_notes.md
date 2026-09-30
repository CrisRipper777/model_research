# Figure contract and rendered QA

## Figure contract

- **Results-level question:** Does the edge-conditioned M0-B correction retain task value after parameter-matched extent capacity, and do target/edge c granularity controls explain that value?
- **Figure-level claim:** M0-B has no consistent all-metric advantage over A-wide; c-only correspondence interventions are small, so edge-specific c utility is not established.
- **Archetype:** Quantitative grid.
- **Backend:** Python / Matplotlib only.
- **Data and statistics:** validation metrics from the CSV files in `../data/`; training variants use three seeds per dataset; global performance bars use nine dataset × seed values with mean ± population SD; no inferential test. Old B interventions are within-checkpoint deltas, with five deterministic shuffle seeds and the spread definition stated below.
- **Export:** 182.9 mm wide; PNG/TIFF at 600 dpi, editable PDF/SVG. No image manipulation or image panels.

## Panel audit

| Figure/panel | Unique evidence role | Center/spread | Replicate unit | Labels/alignment | Collision audit | Result |
|---|---|---|---|---|---|---|
| `m01_performance_attribution` a | Compare Accuracy across seven primary methods | mean ± population SD | 9 dataset × seed runs | direct method labels; 1×3 alignment PASS | 0 fail, 0 warn | pass |
| `m01_performance_attribution` b | Compare Macro-F1 across the same methods | mean ± population SD | 9 dataset × seed runs | same mapping; 1×3 alignment PASS | 0 fail, 0 warn | pass |
| `m01_performance_attribution` c | Compare validation CE and expose metric disagreement | mean ± population SD | 9 dataset × seed runs | same mapping; 1×3 alignment PASS | 0 fail, 0 warn | pass |
| `m01_capacity_control` a/b/c | Show matched-seed Accuracy trajectories for Movies/Grocery/ele-fashion | individual validation runs; no aggregate interval | seeds 42/43/44 within dataset | shared y scale; 1×3 alignment PASS | 0 fail, 0 warn | pass |
| `m01_c_granularity` a/b/c | Compare B-static/B-target/B trajectories within each dataset | individual validation runs; no aggregate interval | seeds 42/43/44 within dataset | shared y scale; 1×3 alignment PASS | 0 fail, 0 warn | pass |
| `m01_b_interventions` a | Show trained M0-B intervention reliance | mean delta ± population SD across nine checkpoint means; shuffle means first average repeats 1001–1005 | old checkpoint × dataset × seed | single-panel alignment not applicable | 0 fail, 0 warn | pass |

Every figure panel maps to the same attribution claim; the capacity and granularity figures retain per-dataset seed trajectories so dataset-level performance differences are not hidden by the aggregate plot. The paired CSV remains the numerical source for all within-dataset × seed comparisons.

## Rendered preflight

- Figure source preflight: 21 pass, 0 warn, 0 fail.
- Panel alignment: three 1×3 figures PASS at 1.5 pt; single-panel intervention figure NOT APPLICABLE.
- PDF text-size audit: all four PDFs auditable; smallest rendered text 6.2 pt (5 pt floor).
- PDF collision audit: all four figures 0 fail / 0 warn.
- QA alignment JSON, measured-rectangle SVGs and collision JSON are stored beside this note. No collision overlay is retained because the final PDFs have no findings.
