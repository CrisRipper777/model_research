# RM-CGE V2 Post-Freeze Analysis and Artifact Fixes

Implementation Freeze is `08d7a16b9f2f9201a7ec12c9ce5dab979c7a2f02`. The
27 formal training runs used that frozen model and protocol. The corrections
below affect analysis and evidence serialization only; they do not change the
model forward, optimizer, data, checkpoints, or training schedule. No model
run was repeated.

## Validation CE lookup

The first report-generation attempt stopped with `KeyError: 'val_ce'`. The
checkpoint diagnostic stores Validation CE in the selected epoch history and
dual-checkpoint metadata rather than at the top level. The report generator now
reads CE from the Validation-Accuracy-selected epoch. It was checked with four
focused analysis/artifact tests, and the analyzer completed over all 27 runs.

## CSV schema and mechanism rows

The precreated header-only tables retained an older schema after the final
diagnostic fields were added. The runner then wrote rows with the current field
lists. This made the row widths disagree with the headers for `runs.csv`,
`epoch_history.csv`, `message_scale_diagnostics.csv`, and
`expert_router_diagnostics.csv`. The source checkpoint JSON remained intact.
The mechanism writer also applied a V0-only remapping that blanked fields
already present in the checkpoint diagnostics.

The repair script regenerated derived tables from the per-run
`checkpoint_diagnostics.json` and `run_metrics.json` files, retaining the
source records and restoring the V0 expert output, selection-share, and graph
correction diagnostics. It now verifies table widths, unique run keys,
finite epoch metrics, both checkpoint reloads, Test-disabled flags, and
mechanism values against their source JSON. The completed audit is recorded in
`data/artifact_integrity.json` with row counts and SHA-256 hashes.

The runner now validates an existing CSV header before appending. Header-only
files can be migrated to the current schema; populated mismatched files fail
closed instead of silently receiving misaligned rows. Successful and failed
run rows share the same `error` column.

## Verification and rerun scope

The artifact audit verified 27 unique completed runs, 3,757 epoch rows, and
30,056 finite history metric values. Both Accuracy-selected and F1-selected
checkpoints reloaded for every run; Test metrics were absent. It compared 1,944
mechanism fields against the per-run JSON sources and checked all CSV row
widths. The training failure ledger remains empty.

The implementation Freeze suite recorded 461 passed and 1 skipped. After the
report and artifact fixes, the full repository regression suite recorded
**467 passed, 0 skipped**. The V2 model and focused post-freeze test command
recorded **15 passed**. The five focused artifact/report tests cover
selected-epoch CE lookup, missing CE failure, header-only schema migration,
refusal to rewrite populated mismatched files, and preservation of V0
mechanism values. The affected scope is report generation and derived-table
serialization only; the 27 model training runs do not need to be repeated.
