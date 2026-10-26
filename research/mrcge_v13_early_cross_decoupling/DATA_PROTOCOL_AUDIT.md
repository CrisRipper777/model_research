# Data and protocol audit

- Dataset split seed: 42, fixed project split files, automatic split generation disabled.
- Training seeds and authorized cells: Movies 42/43/44; Grocery 42/43; ele-fashion 42/43.
- CPU preflight status: `pass`; GPU preflight status: `pass`.
- V7B.1 K=8 semantic cache, feature, candidate and graph fingerprints were checked against paired V1.3-A records for every cell.
- Historical V1.3-A source commit/model hash, fixed split file, and both checkpoint hashes were rechecked; Movies/44 came from the fixed-split campaign. Hybrid Movies 42–44 model hashes and checkpoints were rechecked.
- Test labels were masked before Trainer; Test evaluation and Link Prediction stayed disabled.
- Independent Best-Accuracy and Best-Macro-F1 checkpoints were reloaded and hash-verified for all 7 completed runs.
- Historical V1.2 and fixed-split Hybrid records, when included, are context only. Hybrid fixed-split Movies seeds 42–44 are not a paired architecture comparison.
- Historical model source SHA audit: 17 references pass; see `data/historical_source_audit.csv`.
- `failures.json` retains all failed and blocked attempts (1 entries); no failure was overwritten.
