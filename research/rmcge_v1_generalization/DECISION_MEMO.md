# RM-CGE V1 Decision Memo

## What the completed evidence supports

The full matrix completed 36 seed-level runs. Best-Accuracy remains the primary selection rule; Best-Macro-F1 is reported independently. The descriptive tables and seed-paired contrasts are in `GENERALIZATION_REPORT.md` and the CSV evidence directory. Three seeds per cell do not establish statistical significance.

## Findings by question

- **Movies:** highest Best-Accuracy treatment mean is O3 (Val Accuracy 0.5448; Macro-F1 0.4318). This identifies the observed treatment leader in this small fixed screen, not a generally optimal setting.
- **Grocery:** highest Best-Accuracy treatment mean is O1 (Val Accuracy 0.8250; Macro-F1 0.7533). This identifies the observed treatment leader in this small fixed screen, not a generally optimal setting.
- **ele-fashion:** highest Best-Accuracy treatment mean is O2 (Val Accuracy 0.8796; Macro-F1 0.7545). This identifies the observed treatment leader in this small fixed screen, not a generally optimal setting.

The estimates answer whether regularization helps on these splits and seeds, not whether any treatment generalizes to a new dataset. Weight Decay uses the original AdamW grouping: one group contains the model and classifier parameters. Its per-step decoupled shrink multiplier is approximately `1−lr×wd`; for `lr=1e−3`, the nominal factors are 0.9999999 per step at `wd=1e−4` and 0.99995 at `wd=0.05`, applied uniformly to parameters on optimizer steps. Total effect depends on the number of updates and gradient-driven updates; final norm changes are not themselves a causal estimate of decay.

### 1. Generalization adaptation or model structure?

Compare held-out validation CE/accuracy curves and paired treatment effects before attributing the performance limit. If Train CE falls while Validation CE rises and stronger dropout shifts validation metrics consistently across all three paired seeds, that supports a training-generalization contribution on the current split. If best validation stays low despite reduced train/validation divergence, the remaining limit is compatible with relation/view information organization or label/feature limits; this screen cannot isolate those causes.

### 2. Which regularization is more stable?

Use per-treatment sample SDs and seed-level paired differences in the report. With n=3, rank stability cautiously and avoid interpreting a single best seed as evidence of robustness.

### 3. Physical and Semantic cross-modal paths

Frozen intervention changes are reported in the mechanism report. They show checkpoint sensitivity when the chosen relation's cross-modal context is set to zero. Since this also changes router inputs and is not retrained, it cannot establish a causal generalization benefit of training with that path.

### 4. Router node conditioning

Use the node-level distribution and correlations, not just average relation mass. The recorded coefficients characterize how routing mass co-varies with the measured current-state/neighborhood similarity within available support; they do not assess class predictiveness or relation truth.

### 5. Next architecture priority

Treat Relation Reliability, zero-update candidates, Semantic support handling, Expert diversity, and final fusion as hypotheses for a separately designed controlled study. This study does not train those variants. A Reliability-aware Router is justified only if the node diagnostics reveal a stable mismatch between relation support quality and routed contribution, and a pre-registered matched experiment shows improved validation behavior without hiding optimization or class trade-offs.

## Recommended next experiment

After independent audit, design a small matched mechanism screen around the strongest observed diagnostic: preserve the current V1 checkpoint and training protocol, vary one MAG-specific factor at a time, pair initializations by seed, report both checkpoint-selection rules, and keep Test sealed until a final model-selection decision. Do not infer that a next model revision is automatically warranted from one validation improvement.

Implementation freeze: `e002af6a9c57de21986129d2b1c6e8478bcc74e3`. Evidence source revision for the matrix is recorded per run in `data/runs.csv`.
