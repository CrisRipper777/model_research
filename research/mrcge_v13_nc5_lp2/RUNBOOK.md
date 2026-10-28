# NC5 / LP2 Runbook

Repository: `/hdd1/DataInHere/YHF/model_research`

Environment: `yhf_env`

Branch: `exp/mrcge_v13_nc5_lp2_fullgraph`
Physical GPU default: 1 (normal ownership, memory, and utilization guard enabled)

Run each command from the repository root. Use `conda activate yhf_env` first, or prefix commands with `conda run --no-capture-output -n yhf_env`.

## 1. Prepare NC splits and semantic caches

```bash
python scripts/prepare_mrcge_v13_nc5_lp2.py --phase nc
```

This validates the existing seed42 Toys and Reddit-S splits and reuses only a V7B.1 cache whose feature, physical graph, candidate, V7B, and file SHA256 checks pass. The current data tree has both fixed split files. If a split is missing on another machine, preparation stops by default. Creating one requires the operator to explicitly run:

```bash
python scripts/prepare_mrcge_v13_nc5_lp2.py --phase nc --allow-create-fixed-split
```

That option uses the existing stratified split generator once at seed42. Do not use it to replace a present but invalid split.

## 2. Run only the six new NC cells

```bash
python scripts/run_mrcge_v13_nc5_lp2.py --phase nc --datasets Toys Reddit-S --seeds 42 43 44 --physical-gpu 1
```

The runner is sequential and creates a separate attempt directory, config snapshot, log, status, resource record, and checkpoints for each dataset/seed. It skips cells whose recorded checkpoint SHA256 verifies. It never changes the fixed split or evaluates Test. A resource guard block is recorded and stops that phase; after resolving the resource conflict, rerun the same command to resume remaining cells.

Optional interim collection (still `partial` until LP finishes):

```bash
python scripts/collect_mrcge_v13_nc5_lp2.py --finalize
```

## 3. Prepare LP using the official split files unchanged

Run this after NC preparation (it can be run before NC training if cache preparation is desired):

```bash
python scripts/prepare_mrcge_v13_nc5_lp2.py --phase lp
```

This validates the two existing official `lp-edge-split.pt` files and builds/reuses the V7B.1 caches from global features and the Train-only graph. The official split files remain unchanged. The accepted protocol is named `mrcge_v13_full_graph_edge_holdout_lp_v1_official_split_as_is`.

Current read-only audit:

| Dataset | Train / Valid / Test raw positives | Self-loop positives (Train / Valid / Test) | Undirected cross-partition overlaps (Train-Valid / Train-Test / Valid-Test) |
|---|---:|---:|---:|
| sports-copurchase | 356,202 / 45,557 / 37,413 | 26 / 3 / 5 | 8,264 / 6,621 / 2,035 |
| cloth-copurchase | 951,271 / 121,323 / 97,456 | 38 / 7 / 6 | 21,043 / 17,360 / 5,169 |

Counts use raw positive rows; overlap counts compare canonical undirected endpoint pairs, and self-loops are tabulated separately. Train/Validation/Test positive overlaps are retained and can make Validation metrics optimistic because some Validation edges also occur in Train supervision or the Train message graph. These results are protocol-deviation results, not leakage-safe LP estimates. Train self-loops remain in decoder supervision and are removed from propagation by the repository's standard graph preprocessing.

## 4. Run the six LP cells

```bash
python scripts/run_mrcge_v13_nc5_lp2.py --phase lp --datasets sports-copurchase cloth-copurchase --seeds 42 43 44 --physical-gpu 1
```

The runner starts cells sequentially after checking the six new NC checkpoints and the prepared split/cache fingerprints. It does not call the existing sampled LP Trainer and does not evaluate Test metrics. Results include the split-overlap counts and explicit protocol-deviation text.

## 5. Collect final artifacts

```bash
python scripts/collect_mrcge_v13_nc5_lp2.py --finalize
```

The collector only verifies and summarizes files; it never starts training. Until all expected NC and LP cells pass checkpoint, fingerprint, and protocol checks, it records `status=partial`, preserves failures, and leaves missing seed values absent. After formal runs, ask Codex to review the generated CSVs and prepare the core scientific report before submitting the second-stage results.

## Before starting either training phase

- Check that `prepare --phase nc` and `prepare --phase lp` completed and `research/mrcge_v13_nc5_lp2/data/fingerprints.json` records both phases before their respective runs.
- Do not alter `src/models/mrcge_v13.py`, the official LP splits, prepared caches, or NC seed42 splits between preparation and training. The runner checks hashes before use.
- LP is intentionally a different optimization protocol from sampled LP. Treat its results as a separate Full-Graph Three-Fold evaluation.
- Checkpoint files remain in `outputs/`; never add them to Git.
