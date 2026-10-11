# Corrective implementation and training-dynamics audit

## Finding and minimal fix

At the V7B baseline, `src/models/psce_mag_v7b.py` treated every non-relative variant as a fixed-weight control. The `else` branch therefore set `requires_grad=False` on `relation_projections` and `relation_scorer` for B0 as well as B1/B2. That left B0 with a randomly initialized, frozen Sigmoid scorer, contrary to its definition as V7A A2.

The corrected constructor freezes relation modules only for `b1_fixed_knn` and `b2_random_fixed`. B0 falls through with the inherited V7A A2 relation modules trainable. B3/B4 construction and all other model operations are unchanged.

## Parameter permissions and count

On the frozen Movies configuration:

- Old erroneous B0: 1,551,682 trainable model parameters.
- Corrected B0 / V7A A2: 1,576,387 trainable model parameters.
- Increase: 24,705, exactly the two relation projections plus V7A scorer.
- B1/B2 remain at 1,551,682 trainable parameters.
- B3/B4 remain at 1,576,578 trainable parameters.

The V7A A2 and B0 public parameter names, shapes, and `requires_grad` flags match exactly. B1/B2 relation modules remain frozen; B3/B4 relation scoring heads remain trainable. Per-parameter evidence is in `data/parameter_grad_audit.json`.

## Training parity

The CPU regression tests use matching seeds and Dropout RNG, train mode, the NC cross-entropy plus auxiliary objective, gradient clipping, AdamW parameter groups, and a three-step trajectory. They compare common initialization, logits, losses, gradients, parameters, and optimizer state. All six corrective tests passed.

The real Movies graph audit used identical features, physical graph, KNN candidate cache, train nodes, model/head initialization, dropout RNG, loss, clipping, and AdamW update for V7A A2 and corrected B0:

- Model and classifier initialization max error: 0.
- Embedding max error: 1.19e-6.
- Total loss error: 2.38e-7.
- Public model/classifier gradient max error: 1.19e-7.
- Post-AdamW parameter max difference: 1.24e-6.
- All six B0 relation projection/scorer tensors had finite, nonzero gradients and nonzero parameter updates.

See `data/training_parity_audit.json` for each relation tensor's gradient norm and update size.

## Cache and topology

Feature fingerprints were recomputed from source feature files for Movies, Grocery, and ele-fashion. V7B KNN `edge_index`, unique edge list, degree sequence, source counts, and V7B provenance fingerprints passed direct checks; the KNN tensors matched V7A exactly. Candidate construction/audit used feature arrays and physical edges without label or split access.

All nine B2 caches were reused in place. The training seed mapping, source KNN fingerprint, unique symmetric edges, degree equality, absence of self-loops/duplicates, and physical overlap counts were checked. No B2 topology was regenerated. See `data/candidate_cache_audit.json` and `data/random_graph_audit.json`.

## Historical consequence

The prior 45 V7B runs are preserved. Their B0 was not a trainable V7A A2. Accordingly, the old B0−B1, B3−B0, and B4−B0 comparisons cannot answer the intended learnable-versus-fixed questions. The new campaign trains all six groups from scratch under one freeze commit; no old metric row or checkpoint is imported.
