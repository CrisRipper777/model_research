# Hybrid-CGE Validation Expansion Protocol

## Task and data

- Task: transductive full-graph node classification. Full graph features and topology may be used; Test labels are replaced by `-100` before training and evaluation is disabled (`task.evaluate_test=false`). Train and Validation are the only labeled splits used for fitting, checkpoint selection, reporting and diagnosis.
- MAGB datasets use the repository loader's official feature paths and project-generated stratified 60% Train / 20% Validation / 20% Test split keyed by seed. If a split file exists it is reused and SHA256 recorded; otherwise the existing `generate_nc_split` rule is used and the generated split is persisted. A persisted project split is not described as an upstream official split.
- ele-fashion uses the existing fixed `split.pt` unchanged.
- Modalities are text then visual in loaded concatenated input. Toys: `Toys_roberta_base_512_mean.npy` and `Toys_openai_clip-vit-large-patch14.npy`; Reddit-S: `RedditS_roberta_base_100_mean.npy` and `RedditS_openai_clip-vit-large-patch14.npy`. No fallback feature encoder is allowed.
- Hybrid semantic candidates follow V7B.1: Text and Visual sources searched independently at K=8, project mean-centered row-L2 feature normalization, symmetric union/canonical de-duplication, and provenance source flags. Candidate building reads no labels or split outcomes.

## Model and training

Shared training protocol: full-batch graph update for graph encoders; MLP uses feature-only node minibatches as its method property. AdamW, LR `1e-3`, 300 maximum epochs, patience 30, minimum epoch 30, early-stop minimum delta `1e-4`, gradient clip 1.0, evaluation every epoch, and Validation-Accuracy checkpoint selection. Where the trainer supports it, also retain the best Validation Macro-F1 checkpoint observed before the Accuracy stopping horizon. All checkpoint metrics are Validation-only.

Optimization conditions:

| Condition | Model dropout | AdamW weight decay | Private auxiliary CE |
|---|---:|---:|---:|
| H2-Frozen (historical reference) | 0.20 | 1e-4 | 0.30 |
| H2-R1 | 0.35 | 1e-4 | 0.30 |
| H2-R2 | 0.30 | 1e-3 | 0.10 |

The Hybrid-CGE `late_cross_dropout=0.15` is fixed for all three. Actual `nn.Dropout` layers in the Hybrid-CGE projector, expert, router and fusion paths must be checked against `model.dropout` before training. Within each Dataset × Seed, model and classifier initial-state SHA256 values must match across R1 and R2. All non-research settings, data fingerprints, cache fingerprints and split fingerprints must match.

Baselines use repository model configs and documented necessary interface adaptations. They share input arrays, physical edge index, split indices, class count, AdamW/LR/epoch/early-stop/gradient-clip budget where supported, and the same Test-off contract. MLP does not receive graph messages. MMGCN receives `[text, visual]`, uses the existing model's two modality branches and its reference-style ID embedding (the implementation's tensor is not registered as an `nn.Parameter`, so it is not optimized); this limitation is surfaced in the report. RoleMAG is the project's NC wrapper and role-core adaptation, including its source-order conversion, missing-modality masks, auxiliary-loss defaults and warmup/ramp schedule; it is not called an exact official reproduction.

## Evaluation, checkpoints and RNG

- Best-Accuracy: highest Validation Accuracy with configured minimum delta governing early stopping; report its same-checkpoint Macro-F1, CE, and Train metrics/gaps.
- Best-F1: highest Validation Macro-F1 within the Accuracy-driven stopping horizon where dual-checkpoint support is available; report its Accuracy and CE too. For trainer/model paths without dual-checkpoint support, report the documented limitation and do not synthesize a Best-F1 model selection after the run.
- Checkpoint reload must reproduce saved Validation metrics within numeric tolerance; store checkpoint SHA256 and source/config/data/split/cache fingerprints.
- Validation uses `eval()`/`no_grad()` and the Hybrid runner's evaluation RNG check. Any evaluation that can consume training RNG is treated as a defect and blocks that run.
- Finite Train/Validation loss and metrics are required. OOM, numerical, alignment, data or GPU-guard failures are recorded, not overwritten or silently retried as successful.

## Analysis and selection

Use paired Dataset × Seed differences. First compare H2-R1 and H2-R2 against historical H2-Frozen across the three old datasets, emphasizing mean Validation Accuracy and checking Macro-F1, CE and Train–Validation gaps. Select one global config; if neither candidate improves overall, retain H2-Frozen. Toys and Reddit-S do not influence this choice. Baselines are compared on the same five datasets using Validation Accuracy and Macro-F1, with per-seed values and uncertainty retained. Test is not opened during this campaign.

## External reference boundary

MAGB defines the dataset/benchmark context; OpenMAG is a broader multi-dataset and multi-task benchmark. Their public scores use their own implementations, features and protocols and therefore are literature context only, never cells in this campaign's same-protocol result table. GraphSAGE, MMGCN and RoleMAG reports describe the method family, while project code and this protocol define the actual executable comparison.
