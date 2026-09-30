# P0/P1: Physical-relation propagation utility heterogeneity

This artifact records a validation-only NC study of whether incoming messages on the observed physical graph have different, modality-dependent effects on a neutral one-hop probe, and whether raw text/image cosine similarity is enough to identify those effects.

## Protocol

- Datasets: Movies, Grocery, ele-fashion; seeds: 42, 43, 44.
- Frozen features and existing NC splits from `/hdd1/DataInHere/YHF/data`.
- Separate `Linear → LayerNorm` Text and Visual projections (128 dimensions), uniform directed incoming-neighbor means with self-loops removed, and a linear classifier.
- The semantic-only control uses the same projected intrinsic node features without neighbor context.
- AdamW, learning rate `1e-3`, weight decay `1e-4`, maximum 300 epochs, validation-accuracy checkpoint selection, patience 30, minimum epoch 30. No per-dataset tuning.
- Only train labels optimize parameters. Validation labels select checkpoints and support the requested analysis. No test metrics are computed; the probe data object exposes only train and validation indices and only copies those labels into its visible label tensor.
- The graph is the configured full physical graph, made undirected and stripped of self-loops. This is transductive NC: all graph-node features can contribute to one-hop context, while test labels and test-index values are not used. Split tensors are memory-mapped, and the split accessor requests only `train_idx` and `val_idx`; requesting a test field raises.
- Per-edge CE utility is the signed relative change `(CE_removed - CE_full) / max(CE_full, 1e-12)`. The attached text displayed `CE_removed / CE_full`, which cannot have the stated positive/helpful and negative/harmful signs. The relative-change interpretation follows those stated signs. Counterfactual classifier arithmetic is evaluated in float64 after the trained float32 projection to make the fast and brute-force paths numerically comparable.

See [report.md](report.md) for results and interpretation. The exact environment, data paths, commands, and provenance are in [run_manifest.json](run_manifest.json).

## Files

- `probe_performance.csv`: semantic-only and uniform-one-hop validation metrics for all 9 runs.
- `utility_summary_by_seed.csv`: global CE-utility distributions by dataset, seed, and modality.
- `node_heterogeneity_summary.csv`: within-target-node spread and mixed-sign neighborhood ratios.
- `modality_disagreement_summary.csv`: Text/Visual utility association, sign disagreement, and per-target top/bottom Jaccard.
- `margin_utility_sanity_summary.csv`: directional and rank agreement between the secondary margin and primary CE utilities.
- `similarity_utility_summary.csv`: raw-cosine association, descriptive AUROC, and similarity/utility set overlap.
- `similarity_utility_bins.csv`: raw-cosine quantile-bin counts, ranges, and utility distributions.
- `group_intervention_summary.csv`: 10-repeat random-control averages and utility-tail interventions.
- `edge_sample.csv`: deterministic sample of up to 8,000 directed validation messages for each dataset × seed.
- `data/node_heterogeneity_per_target.csv`: per-target spread values used by the neighborhood figure.
- `data/group_intervention_repeats.csv`: individual deterministic random-control repeats.
- `figures/`: four requested figures.

Complete per-run edge tables and checkpoints are in the ignored `outputs/p0p1_propagation_heterogeneity/` directory and are not part of this commit.

## Reproduction

From the repository root with `yhf_env` active:

```bash
python -m pytest -q tests/test_p0p1_propagation_probe.py
python scripts/run_p0p1_propagation_heterogeneity.py --smoke --device cuda:0
python scripts/run_p0p1_propagation_heterogeneity.py --datasets Movies Grocery ele-fashion --seeds 42 43 44 --device cuda:0
python scripts/analyze_p0p1_propagation_heterogeneity.py --output-dir outputs/p0p1_propagation_heterogeneity --research-dir research/p0p1_propagation_heterogeneity --datasets Movies Grocery ele-fashion --seeds 42 43 44
```

No V0 model or routing mechanism is implemented in this branch.
