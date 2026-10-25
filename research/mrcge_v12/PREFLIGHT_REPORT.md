# MR-CGE V1.2 preflight report

Date: 2026-10-25. Raw records: `data/preflight.json`, `data/v11_checkpoint_regression.json`, `data/failures.json`, `data/resources.csv`.

## CPU/data preflight

Status: **pass**. All 16 dataset/condition cache checks (Movies/Grocery/Toys/ele-fashion × four conditions) passed for the existing feature, split, physical graph, and V7B.1 cache fingerprints. Initialization checks passed for S/F and M_repaired/M_anchor. The safe copy masks Test labels to `-100`; Test evaluation is disabled.

## Guarded GPU preflight

Physical GPU 1 was an NVIDIA RTX 3090. The first attempt completed Movies/S_repaired forward/backward, then observed a newly appeared external compute PID. The ownership guard terminated only its own child and blocked the remaining checks. No formal training began. That event and the failed attempt remain in the raw audit trail.

A fresh guarded attempt then passed **14/14** checks: all four conditions on Movies/Grocery/Toys and M_repaired/M_anchor on ele-fashion. Every one-step loss and gradient was finite, the no-grad diagnostic left Validation RNG unchanged, and all six M CUDA filter equivalence audits passed. The attempt started with about 17.2 GiB free on GPU 1; every recorded M preflight required about 4.1–12.8 GiB free including the 2 GiB margin. Live memory and ownership are still rechecked before every formal cell.

Peak GPU memory for one full-graph training forward/backward (allocated / reserved):

| Dataset | S_repaired | F_sinit | M_repaired | M_anchor |
|---|---:|---:|---:|---:|
| Movies | 1.67 / 2.18 GiB | 1.98 / 2.46 GiB | 1.51 / 1.83 GiB | 1.51 / 1.83 GiB |
| Grocery | 1.70 / 2.20 GiB | 2.03 / 2.47 GiB | 1.55 / 1.83 GiB | 1.55 / 1.83 GiB |
| Toys | 2.06 / 2.45 GiB | 2.45 / 2.87 GiB | 1.87 / 2.13 GiB | 1.87 / 2.13 GiB |
| ele-fashion | — | — | 8.49 / 8.95 GiB | 8.49 / 8.95 GiB |

The ele-fashion M measurement is below the V1.2 engineering target of roughly 16–17 GiB allocated. Its measured reserved peak plus the required 2 GiB margin is 13,124 MiB. This makes the two bonus cells eligible for a fresh live-guard check; it does not guarantee later resource availability.

For comparison, the preserved V1.1 ele-fashion M preflight used 22,201,272,320 allocated bytes (20.68 GiB) and 23,089,643,520 reserved bytes (21.50 GiB). V1.2's measured full-graph allocated peak is lower by 58.95%, and reserved peak lower by 58.40%. The V1.1 and V1.2 measurements are separate preflight runs on shared hardware.

## Failed and superseded attempts

- GPU preflight attempt 1 stopped after an external process appeared during the Movies/S_repaired check; remaining cells were blocked. The successful attempt 2 is the only preflight used to authorize training.
- Early V1.1 regression attempts exposed a checkpoint-row lookup issue and a small eval arithmetic difference in an earlier segmented implementation. Both were corrected before the final successful three-dataset regression; the unsuccessful attempts remain in `data/failures.json`.
- No Test labels or metrics were used. No training occurred during preflight. No other GPU process was stopped, signalled, or modified.
