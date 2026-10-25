# MR-CGE V1.1 Stage-A Pilot Report

## Status

Planned: 9 Dataset × Condition × Seed cells (Movies/Grocery/ele-fashion × S/F/M × seed 42).
Completed: 8/9; formal cells with a recorded attempt: 9.
The runtime GPU ownership guard interrupted ele-fashion/M/seed42 attempt 1 because `RuntimeError('GPU1 free memory 160 MiB below 1536 MiB guard threshold')`. The guard stopped only its own child. 8 cells completed; the interrupted attempt has no valid metric, and 1 authorized cell(s) remain incomplete.
A user-directed override disabled GPU guard checks for ele-fashion/M/seed42 attempt 2; see `data/guard_overrides.json` for the device snapshot and scope.
That overridden attempt ended without a valid metric: `OutOfMemoryError('CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 23.69 GiB of which 64.88 MiB is free. Process 3949460 has 2.89 GiB memory in use. Including non-PyTorch memory, this process has 20.72 GiB memory in use. Of the allocated memory 20.39 GiB is allocated by PyTorch, and 32.92 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)')`.
A user-directed override disabled GPU guard checks for ele-fashion/M/seed42 attempt 3; see `data/guard_overrides.json` for the device snapshot and scope.
Attempt 3 used allocator environment `{'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}`; model and optimizer settings were unchanged.
That overridden attempt ended without a valid metric: `OutOfMemoryError('CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total capacity of 23.69 GiB of which 70.88 MiB is free. Process 3949460 has 2.89 GiB memory in use. Including non-PyTorch memory, this process has 20.71 GiB memory in use. Of the allocated memory 20.33 GiB is allocated by PyTorch, and 87.49 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)')`.

## Validation metrics

These are one-seed descriptive Validation metrics and do not support inferential claims.

| Dataset | Condition | Val Accuracy | Val Macro-F1 | Val CE |
|---|---|---:|---:|---:|
| Movies | S | 0.5302939414978027 | 0.43265517358409744 | 1.4439923763275146 |
| Movies | F | 0.5107978582382202 | 0.36692950842054267 | 1.4827326536178589 |
| Movies | M | 0.5527894496917725 | 0.4838016985835905 | 1.4057756662368774 |
| Grocery | S | 0.8005856275558472 | 0.7206894206885341 | 0.7428837418556213 |
| Grocery | F | 0.772767186164856 | 0.7032902155017174 | 0.8816630840301514 |
| Grocery | M | 0.8269400000572205 | 0.7567569553235868 | 0.6694358587265015 |
| ele-fashion | S | 0.8741945624351501 | 0.7495736657863662 | 0.4363113343715668 |
| ele-fashion | F | 0.871126115322113 | 0.7394391214024839 | 0.4261397421360016 |

See the expert, route, response and resource diagnostics alongside epoch-level loss values.
- Formal gradient traces: `data/loss_gradient_diagnostics.csv` is header-only, so per-run training gradient magnitudes were not recorded. The nine GPU single-step preflights and focused tests checked finite gradients, but do not replace formal-run gradient traces.

## Mechanism diagnostics in completed M cells

Diagnostics are available for 2 completed M cells (Movies and Grocery), across 12 recorded private/collaborative router views.
- Routing: all four experts received nonzero Top-2 selection share in every recorded view; shares ranged 0.087–0.447. No complete route collapse appears in these two one-seed runs.
- Filter responses: the four experts retained distinct signed theta vectors in each recorded dataset (yes). Across dataset/view/expert pairs, G cosine ranged -0.998–0.939 and E cosine ranged -0.996–0.930; expert responses were not all identical, though these measures do not establish useful specialization.
- Early Cross: nonzero cross-message RMS was recorded (0.242–0.588); beta was 0.245–0.260. Collaborative/private response RMS ratios were 0.964–1.034.
- Late completion: mean p_accept was 0.665–0.989, gate mean 0.638–0.743, and late delta/RMS ratios 0.040–0.086; the path was active but its predictive contribution is not isolated by this Pilot.
These are end-of-run descriptive diagnostics for two datasets and one seed; ele-fashion/M has no completed run, and no causal or generalization claim follows.

Expert and routing observations are descriptive for this seed. Generalization and statistical claims require later approved seeds.

## Scope guard

No Test metric or model selection used Test labels. Test labels were masked to `-100` before Trainer entry. Seeds 43/44, the remaining 18 formal runs, full 27-run training, tuning and ablations were not started.

Resume the remaining M cell only when GPU 1 has enough headroom and a fresh ownership/resource guard passes. The runner will preserve every failed attempt, skip verified completed cells, and create the next attempt for the interrupted cell.
