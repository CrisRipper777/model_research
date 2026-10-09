# V5A.1 experiment package

This package repairs the V5A target baseline with a matched local zero-change replay and compares receiver-equal matched multi-target/single-target pointwise models with a receiver-local RankNet scorer. The host, sample, feature ladder, and receiver split are inherited from V5A.

- `REPORT.md`: protocol, aggregate results, and interpretation boundary.
- `data/`: committed summaries, audits, diagnostics, and campaign records.
- Full singleton and bundle rows, high-dimensional feature tensors, checkpoints, and logs remain under ignored `outputs/mag_message_effect_v5a1/`.

The fixed formal campaign completed 180 estimator runs across Movies, Grocery, and ele-fashion. Movies and Grocery ran on `cuda:1`; ele-fashion preparation and estimation used CPU after the GPU became unavailable. Per-dataset devices and the fallback are recorded in `data/campaign_manifest.json` and `data/environment.json`.

All effects remain frozen-host exposed-message replay contrasts. The receiver-local ranker is evaluated only within `(receiver, modality, hop)` candidate sets. Undefined statistics appear as empty CSV cells.
