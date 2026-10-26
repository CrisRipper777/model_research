# Decision Handoff — Intrinsic-Prior Skip

Branch: `exp/mrcge_v13_intrinsic_prior_skip_screen`
Base: `184deaaf12b21c9ee01d3328aef15d86bcc50346`
Frozen training source: `c7f9209c60f18059d330ac23d7b3ab1b13af2e6e`

Stage 1 status: `no_go`
Stage 2 status: `not authorized` (Movies failed its preregistered Stage 1 gate; ele-fashion was not run)
Recommendation: `NO-GO`
Fashion conclusion: not evaluated; the conditional stage was not authorized. No failure is inferred for ele-fashion.

All five Stage 1 cells completed on physical GPU 1 and their audits passed. The preregistered Movies metric gate failed, so the conditional ele-fashion Stage 2 cells were not authorized. The training source is frozen. Review the implementation audit, data protocol audit, reports, compact CSV/JSON evidence, and GPU resource record. Checkpoint files remain local under `outputs/mrcge_v13_intrinsic_prior_skip/runs/` and are not tracked. No default-model integration or further ablation is authorized in this task.
