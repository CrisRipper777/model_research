# CSE-MAG V1 Local/Global Screen

Branch: `exp/cse_mag_v1_local_global_screen`

Base: `main@579d8dcde6d9bf6d39efb4f9d88bf70a78acc38e`

## Research question

MAG modalities share one physical topology but have heterogeneous pretrained
semantic spaces. The screen asks whether structural processing should be
organized as a reusable cross-modality capability space rather than either:

1. fully independent modality-specific graph encoders, or
2. one fixed structural strategy shared by every node and modality.

The candidate V1 uses two functionally defined structural responses:

- **Local response**: one-hop smoothing displacement `PH - H`.
- **Global contrastive response**: a truncated PC-Conv-inspired
  `(exp(-tP) - I)H` direction,
  `sum_{k=1}^K (-t)^k/k! P^k H`.

The intrinsic modality representation is a protected residual path and cannot
be removed by the structural composer.

## Four-model minimum screen

All variants use the same:

- modality-specific intrinsic projectors,
- input feature layout,
- physical graph,
- hidden width,
- late residual fusion,
- NC classifier and unified training protocol.

The model file instantiates all variant-specific modules in a fixed order and
then freezes inactive modules. Under the same seed this preserves initialization
of common projectors, fusion modules, and the downstream classifier.

### B0_independent — `variant=independent`

Local/Global structural response definitions are the same as V1, but Text and
Visual use separate expert transforms and separate node routers.

Question: does cross-modality structural parameter sharing help or hurt?

### B1_shared_static — `variant=shared_static`

Text and Visual share the functional Local/Global experts, but use one
graph-wide Local/Global mixture and one graph-wide structural gate.

Question: is a single shared structural strategy sufficient?

### B2_mvcge_style — `variant=mvcge_style`

Direct transfer control inspired by MvCGE:

- homogeneous shared graph experts,
- shared graph-context-aware node/modality router,
- Top-K expert selection,
- lightweight load-balancing loss,
- no protected intrinsic residual.

This is intentionally labeled **MvCGE-style**, not an exact reproduction of
the original paper.

Question: is generic shared-expert routing already sufficient after replacing
views with modalities?

### B3_v1 — `variant=v1`

Shared functional Local/Global experts + shared node/modality-conditioned
router + protected intrinsic residual.

Question: does the MAG-specific redesign improve on independent, static-shared,
and generic shared-MoE controls?

## Screening protocol

Datasets:

- Movies
- Grocery
- ele-fashion

Seeds:

- 42
- 43
- 44

Task:

- node classification only
- repository protocol `unified_full_graph_nc_v1`
- best checkpoint selected by Validation Accuracy
- Test evaluation disabled during this screen

Total formal runs: `3 datasets x 3 seeds x 4 variants = 36`.

No LP run, no architecture search, no Test-based selection, and no additional
auxiliary objective are authorized for V1 in this first screen.

## Commands

Smoke test all four variants on Movies / seed 42 / one epoch:

```bash
python scripts/run_cse_mag_v1_screen.py --smoke --device cuda:0
```

Formal validation-only screen:

```bash
python scripts/run_cse_mag_v1_screen.py --campaign --device cuda:0
```

Resume completed cells:

```bash
python scripts/run_cse_mag_v1_screen.py --campaign --device cuda:0 --resume
```

Outputs are written under:

```
outputs/cse_mag_v1_local_global_screen/
```

The launcher writes paired V1-minus-control validation summaries to
`campaign_manifest.json`.

## Interpretation boundary

This screen is deliberately coarse. It is intended to decide whether the
**shared-capability / conditional-utilization** mother hypothesis is worth
continuing.

A positive V1 result would not establish that:

- the two response operators are uniquely optimal,
- PC-Conv's full filter theory has been reproduced,
- MvCGE has been exactly reproduced,
- node-wise routing is universally necessary,
- homophily/heterophily can be identified from the learned routing weights.

A negative V1 result would reject this concrete prototype, not the general
idea of collaborative structural experts.

## Next stage only if the screen is promising

Do not add more model components before inspecting the four-model screen.

If B3 is competitive/promising, the next stage should be limited to:

1. Local-only vs Global-only vs static mixture vs full V1;
2. response cosine/effective-rank diagnostics;
3. Text-vs-Visual routing distribution comparison;
4. structural-gate distribution and gate-shuffle functional check.

Only after those checks should we consider changing the response dictionary,
learning the heat parameter `t`, or adding stronger anti-collapse objectives.
