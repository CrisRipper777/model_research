# N0 figure panel claims and QA

Each figure has three aligned dataset panels: **a** Movies, **b** Grocery, **c** ele-fashion. D denotes AbsDiff, P denotes Product, and T/V denote Text/Visual candidate modality. Every error bar reports the three-seed mean ± population SD unless a panel note below defines an additional reference band. No significance tests are shown.

| Figure | Panel | Dataset | Quantity and claim | Uncertainty / definition |
|---|---:|---|---|---|
| `n0_context_gain_range` | a | Movies | Mean per-edge gain range over the nine recipient backgrounds, separately for D·T, P·T, D·V, P·V. | Across-seed population SD (3 seeds). |
| `n0_context_gain_range` | b | Grocery | Mean per-edge gain range over the nine recipient backgrounds, separately for D·T, P·T, D·V, P·V. | Across-seed population SD (3 seeds). |
| `n0_context_gain_range` | c | ele-fashion | Mean per-edge gain range over the nine recipient backgrounds, separately for D·T, P·T, D·V, P·V. | Across-seed population SD (3 seeds). |
| `n0_sign_switch` | a | Movies | Fraction of validation edges whose gain changes sign across the nine backgrounds; ordinary versus all-head robust switching. | Across-seed population SD (3 seeds); robust requires all three heads to agree in each of two opposite-sign states. |
| `n0_sign_switch` | b | Grocery | Fraction of validation edges whose gain changes sign across the nine backgrounds; ordinary versus all-head robust switching. | Across-seed population SD (3 seeds); robust requires all three heads to agree in each of two opposite-sign states. |
| `n0_sign_switch` | c | ele-fashion | Fraction of validation edges whose gain changes sign across the nine backgrounds; ordinary versus all-head robust switching. | Across-seed population SD (3 seeds); robust requires all three heads to agree in each of two opposite-sign states. |
| `n0_rank_stability` | a | Movies | Mean target-level Spearman correlation between SS edge gains and each alternative background, for degree ≥5 targets; colored lines are candidate/operator pairs. | Across-seed population SD (3 seeds); dashed line and band show the SS head-repeat rank mean and population SD. |
| `n0_rank_stability` | b | Grocery | Mean target-level Spearman correlation between SS edge gains and each alternative background, for degree ≥5 targets; colored lines are candidate/operator pairs. | Across-seed population SD (3 seeds); dashed line and band show the SS head-repeat rank mean and population SD. |
| `n0_rank_stability` | c | ele-fashion | Mean target-level Spearman correlation between SS edge gains and each alternative background, for degree ≥5 targets; colored lines are candidate/operator pairs. | Across-seed population SD (3 seeds); dashed line and band show the SS head-repeat rank mean and population SD. |
| `n0_same_vs_cross_modal` | a | Movies | Mean absolute gain change from SS when changing the candidate modality’s other incoming edges or the opposite modality neighborhood. | Across-seed population SD (3 seeds). |
| `n0_same_vs_cross_modal` | b | Grocery | Mean absolute gain change from SS when changing the candidate modality’s other incoming edges or the opposite modality neighborhood. | Across-seed population SD (3 seeds). |
| `n0_same_vs_cross_modal` | c | ele-fashion | Mean absolute gain change from SS when changing the candidate modality’s other incoming edges or the opposite modality neighborhood. | Across-seed population SD (3 seeds). |
| `n0_context_vs_head_uncertainty` | a | Movies | Full-grid SD of head-mean conditional gain divided by mean head-repeat SD; dashed line marks ratio 1 as a descriptive reference. | Across-seed population SD (3 seeds). |
| `n0_context_vs_head_uncertainty` | b | Grocery | Full-grid SD of head-mean conditional gain divided by mean head-repeat SD; dashed line marks ratio 1 as a descriptive reference. | Across-seed population SD (3 seeds). |
| `n0_context_vs_head_uncertainty` | c | ele-fashion | Full-grid SD of head-mean conditional gain divided by mean head-repeat SD; dashed line marks ratio 1 as a descriptive reference. | Across-seed population SD (3 seeds). |
| `n0_gradient_decomposition` | a | Movies | Median relative absolute error for first- and second-order CE-gradient/Hessian gain approximations, plotted on a log scale. | Across-seed population SD (3 seeds); relative error denominator includes 1e-8 and all plotted means are asserted positive. |
| `n0_gradient_decomposition` | b | Grocery | Median relative absolute error for first- and second-order CE-gradient/Hessian gain approximations, plotted on a log scale. | Across-seed population SD (3 seeds); relative error denominator includes 1e-8 and all plotted means are asserted positive. |
| `n0_gradient_decomposition` | c | ele-fashion | Median relative absolute error for first- and second-order CE-gradient/Hessian gain approximations, plotted on a log scale. | Across-seed population SD (3 seeds); relative error denominator includes 1e-8 and all plotted means are asserted positive. |

## Final QA

- Static preflight on `scripts/plot_n0_recipient_state_function_context_impl.py`: **21 PASS, 0 WARN, 0 FAIL**.
- Panel alignment: **6/6 PASS**, with 1.5 pt tolerance; per-figure alignment manifests are in the gitignored QA directory.
- PDF collision audit: **6/6 PASS**, zero warnings and zero failures.
- PDF text audit: minimum observed font **6 pt** across all six; zero text runs below the 5 pt floor and no audit warnings.
- Visual inspection: all 18 panels inspected after final export; axis labels, legends, seed spread, panel titles, and footnotes are legible at rendered size. The sign-switch panel leaves headroom above the largest bar; the gradient panel uses labeled log ticks without minor-tick clutter.
- Export: each figure is available as 600 dpi PNG and TIFF plus editable SVG and PDF.

Machine-readable QA details are in `outputs/n0_recipient_state_function_context/figure_qa/` (gitignored).
