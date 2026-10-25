# Data and Protocol Audit

## Frozen data inputs

All new experiments load Movies or Grocery through `load_mag_data` and the repository's existing project MAGB configuration/path resolver. The training seed is preserved as 42, 43 or 44 while the resolved node split path is forced to the existing seed42 split. `auto_generate_splits=false`; a missing or mismatched split fails closed.

Movies split fingerprint: `2b544e174253b95facf6d91015723b843c240db70dcdfb9b3fe6cc23afa0366d`; split-file SHA256: `9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211`. Grocery split fingerprint: `7ae489257ba4aa1e1690f795e93076fa778aa6e9a3e5239b8862eb0d1aec1056`; split-file SHA256: `3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e`. CPU preflight passed for all seven target cells. Movies has 16,672 nodes / 160,802 physical edges; Grocery has 17,074 nodes / 142,262 physical edges. Every target cell matches the corresponding V1.3-A feature, graph, split, semantic-cache, candidate and V7B.1 fingerprints. Dimensions and node count are recorded in `data/preflight.json`.

Semantic inputs use the already verified V7B.1 candidate cache, with k=8 as in V1.3-A. Cache file SHA256 and its feature/candidate/V7B fingerprints are captured. No new candidate construction or dataset-specific data transformation is introduced.

## Labels and evaluation

`mask_test_labels` clones labels and replaces every Test label with `-100` before any Trainer, loss or checkpoint evaluation receives the data. Train and Validation labels are checked unchanged. `task.evaluate_test=false`; no Test metric is recorded. Validation uses the repository's established class convention and dual-checkpoint rule. The reload audit recomputes Train/Validation Accuracy, Macro-F1 and CE from strict-loaded Best-Accuracy and Best-F1 checkpoints and checks that Validation does not advance CPU or CUDA RNG state.

## Read-only historical references

The fixed-split Hybrid table uses only corrected `H2_physical_3hop` Movies seed42 attempt 02 plus new seeds43/44 on the seed42 split. Historical Hybrid seeds43/44 with other splits are excluded. The V1.3-A Movies comparison reuses historical seeds42/43 and adds seed44. NoP0 baselines are the matching V1.3-A dataset/seed records for Movies and Grocery.

The corrected Hybrid seed42 reference is checked for correction commit `5e052dbc26237ddb534e0f779d91951f26079748`, corrected wrapper SHA, attempt 02, split fingerprint, recorded Validation metrics and local Best-Accuracy/Best-F1 checkpoint hashes. V1.3-A reference checkpoint files and hashes are also checked. The fixed-split Hybrid comparison requires all six model/seed rows to carry the authorized Movies split fingerprint.

All data rows are Validation-only. The fixed split and same training seed improve comparability, but the two model families use different parameterizations and initialization computation order. NoP0 also changes the normalized higher-order Physical mixture and its coupled Early Cross response. These constraints rule out strict causal claims.
