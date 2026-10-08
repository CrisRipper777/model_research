# MvCGE-MAG V3A: Effective Structural Action-Space Screen

## Research question

Can fixed, modality-conditioned semantic reweighting of existing physical edges create useful structural actions for the shared expert dictionary? This screen compares raw physical trajectories, effective modality-specific trajectories, their convex mixture, and a mixture with a separately learned effective hop profile.

## Frozen design

- Parent: `bb52895da5863c737fc47c1f88637be12a069445`; branch: `exp/mvcge_mag_v3a_effective_context_screen`.
- Datasets: Movies, Grocery, and `ele-fashion`; seeds 42, 43, and 44; four variants; 36 formal runs.
- Variants: `C0_raw`, `C1_effective`, `C2_dual_shared_profile`, `C3_dual_context_profile`.
- All variants retain the V2.2 R0 modality-static Top-2 selection and strong static strength controller. Hidden width is 256, there are four shared experts, Top-2 routing, and four trajectory hops.
- Effective edge scores use only detached raw pretrained features from the corresponding modality. They reweight existing physical edges; no labels, splits, learned edge parameters, or added edges enter the operator.
- The common intrinsic modality paths, expert transforms, static router, strength controller, and late fusion follow V2.2 R0. Context mixing is convex with a shared per-expert coefficient initialized to 0.10. Its parameter has no weight decay.
- Training uses `task=nc` and `task.evaluate_test=false`. Validation Accuracy selects checkpoints. No Test evaluation, HPO, LP, significance testing, cross-modal operator, or V3B is part of this screen.

## Execution order

From the repository root, with the project environment active:

```bash
python -m pytest tests/test_mvcge_mag_v3a.py -q
python -m pytest -q
python scripts/run_mvcge_mag_v3a_screen.py --mode preflight --device cuda:0
python scripts/run_mvcge_mag_v3a_screen.py --mode smoke --device cuda:0
```

The preflight reads only raw modality features and physical edges for each dataset. It saves six modality records before stopping if every normalized operator delta is at most `1e-6`. Do not tune the method from preflight results. Freeze the implementation only after the tests, preflight, and four-run one-epoch Movies smoke pass.

Freeze commit message:

```text
Freeze MvCGE-MAG V3A effective structural action-space screen
```

The formal campaign requires HEAD to equal that freeze commit and a clean worktree:

```bash
python scripts/run_mvcge_mag_v3a_screen.py --mode campaign --device cuda:0
```

After interruption, resume only from the same freeze SHA and with changes limited to campaign result artifacts:

```bash
python scripts/run_mvcge_mag_v3a_screen.py --mode campaign --device cuda:0 --resume
```

## Primary comparisons and interpretation

Report paired Accuracy and Macro-F1 deltas in percentage points for C1−C0, C2−C0, C2−C1, C3−C2, and C3−C0, with nine matched pairs and per-dataset seed counts. Stable-positive is the frozen descriptive rule: both overall deltas are positive, at least 6/9 Accuracy pairs are positive, and mean Accuracy is positive on at least two of three datasets. Approximate equality is `|Δ Accuracy| ≤ 0.15 pp` and `|Δ Macro-F1| ≤ 0.50 pp`. These are descriptive labels, not significance or equivalence tests.

The report also audits sparse operator weights, raw/effective trajectories, trajectory-span novelty, context mixing, hop profiles, expert function similarity, static routes, correction strength, and C0 checkpoint compatibility with V2.2 R0. A positive C2 or C3 result versus C0 is only a gate for considering a separately authorized V3B study; this campaign stops after V3A.

## Artifacts

Tracked reproducibility outputs live in `data/` and the generated `REPORT.md`. Checkpoints, Hydra directories, training logs, raw tensors, and feature caches remain in the ignored server-local `outputs/mvcge_mag_v3a_effective_context_screen/` tree and must not be committed.
