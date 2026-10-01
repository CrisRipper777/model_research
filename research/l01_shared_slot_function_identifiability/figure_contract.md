# Figure plan and QA contract

**Core conclusion:** effects are output-separable in logits but usually small in probability space; repeat reliability exceeds cross-seed reliability; Movies/Grocery lack edge-level predictability, while ele-fashion retains a modest, non-exclusive local signal. The result is dataset-dependent and does not support a router or stochastic sampler.

**Results question:** does a single edge-level S→D/P substitution have a stable, distinguishable task effect that existing local evidence or frozen relation states can identify?

**Archetype:** six quantitative evidence grids; each figure is a distinct part of the ordered identifiability argument. No panel will present empirical JS as theorem-level injectivity, or positive gain as a ground-truth operator label.

**Backend and export:** Python/matplotlib, established backend preference. Required PNGs plus editable SVG/PDF and 600-dpi TIFF for QA; per-figure source-data CSVs and audit manifests remain traceable in this repository. Dense but legible report figures; 5 pt minimum rendered glyph, editable text, white background, restrained neutral/blue/teal plus directional gain colors.

## Panel map

1. **Utility reliability.** Head-repeat and cross-seed edge-aligned Spearman distributions, separately encoded; points show individual pairwise estimates and bars show median/IQR. Exact sign agreement, centered ranks and edge overlap are in the companion data tables/report.
2. **P1.3 transfer.** Historical `Delta_D/Delta_P` versus clean shared-slot gain Spearman, faceted by dataset and target. Points are seeds, black tick/whisker is mean ± seed SD; sign agreement and top-effect overlap are tabulated.
3. **Output separation.** Per-target JS median and interquartile span across edge distributions for each head/seed. Logit/probability shifts, flip fractions, upper quantiles and gain-to-separation correlations are tabulated; no injectivity claim.
4. **Total predictability.** Evidence MLP variants, strict shuffled null, Ridge, DirectState and frozen Q/R/U; show total Spearman and across-fold variability, faceted by dataset and clean target.
5. **Within-target predictability.** Within-target residual Spearman for local and state/readout probes, to show whether edge order survives removal of recipient mean effects.
6. **State capacity.** Endpoint-local MLP, utility-supervised DirectState and frozen Q/R/U residual Spearman, faceted by dataset and clean target.

## Evidence and reviewer-risk checks

- All plotted rows come from the committed `data/*.csv` tables or documented raw outputs; no unreported row sampling or dropping. Reliability-correlation NA filtering had equal before/after counts (zero exclusions).
- Across-seed and fold summaries show the defined variation (seed/fold SD) consistently. Head-repeat spread is not substituted for across-seed variability.
- Frozen validation protocol only; no test rows or metrics.
- Distinguish total rank correlation from within-target centered rank. TARGET_ONLY is a recipient/neighborhood baseline.
- Historical and new utilities must be joined on exact dataset, seed, source, destination and modality before paired comparison.
- Strict shuffle permutes full four-gain tuples separately inside each destination for inner-train and inner-validation only; outer-test targets remain unshuffled and unevaluated during fit.
- Prediction plots use compact model codes, decoded in the figure note. Six figures each pass render-time panel alignment, PDF text and collision audits plus final-size visual review. QA overlays are diagnostic and never replace figure exports.
