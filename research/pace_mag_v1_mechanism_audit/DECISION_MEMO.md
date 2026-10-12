# Decision memo: what to test after independent review

## A. Are the source-view messages distinct?

Yes. Text-only and Visual-only edges account for 91.9–96.6% of unique semantic edges, and global Text/Visual support Jaccard is only 3.4–8.1%. With the target modality fixed, nearly all nodes have source contexts with relative L2 above 0.1, both with cross-modal injection on and off. The contexts are most aligned on ele-fashion, but remain measurably different. This supports testing source adaptation; it does not establish that it will improve Validation metrics.

## B. What happens to cross-modal information?

The learned injection is small relative to the target message: `rho*Phi` is 1.8–3.7% of `H` on Movies, 2.4–3.1% on Grocery, and 8.3–10.4% on ele-fashion. Relation aggregation does not halve that relative change; aggregate/send ratios are 0.94–1.54. After the full expert, fusion, and classifier path, however, cross-off shifts Validation predictions very little: at most 0.03 percentage points in Accuracy and 0.27% prediction flips. The branch is small at entry and has little checkpoint-level prediction effect, rather than being erased by aggregation. This cannot distinguish whether another training setup would learn to use it.

## C. Does the Router or expert bank matter?

The Router is node-conditioned: route probabilities have nonzero nodewise spread, different relation/target cells have different expert preferences, and same-node route distributions differ across semantic views and target modalities. Node-permuting the trained probabilities consistently lowers Validation Accuracy by 1.69–2.78 pp and flips 5.07–13.29% of Validation predictions. Replacing them with uniform probabilities also lowers Accuracy on every dataset. These are direct observations of frozen-checkpoint dependence on route assignment.

Expert outputs have different scales and low pairwise cosine, and route changes affect predictions. The current evidence does not show whether each expert has a distinct task function or whether a retrained alternative would do as well. Keep the expert and Router design as a working component for the next isolated test; do not redesign it based only on cosine geometry.

## D. Which frozen interventions move predictions most?

The largest observed effects are intrinsic-skip removal on Movies and ele-fashion (Accuracy -5.73/-5.15 pp; Macro-F1 -4.88/-7.07 pp), physical-response removal on Grocery (-4.22/-2.18 pp), and node-permuted routing across all three datasets (Accuracy -1.69 to -2.78 pp). Semantic-response removal has smaller effects (Accuracy -0.38 to -0.72 pp). Cross-modal removal changes Accuracy by only +0.02 to +0.03 pp, with fewer than 0.3% of predictions flipping.

These are frozen counterfactual forwards. Large changes show that the fitted representation and classifier depend on a branch under current weights; they do not predict how the remaining components would adapt if retrained. Small changes show limited dependence of these checkpoints, not lack of learning capacity.

## E. Is an optimization defect established?

No. The final `rho` values remain near 0.1 and their effective derivatives remain near 0.06. Recorded first-epoch cross-modal and `rho_raw` gradients are finite and nonzero; the existing one-step real-graph preflight changed every checked parameter. The cross-modal injection is small, so its initialization and learned scale deserve a targeted follow-up if that mechanism is important, but the evidence does not establish a broken gradient path or saturated scalar.

The logs show declining train loss after the best-Accuracy checkpoint, and Movies' Validation metrics fall by the final epoch. Grocery and ele-fashion end close in Accuracy while Macro-F1 rises after the selected epoch. Without train Accuracy/Macro-F1, optimizer state, per-epoch gradients, and intermediate weights, the audit cannot diagnose full-run overfitting, underfitting, or early-stopping quality.

## F. How should E3/E4 be interpreted?

Correct the old Pilot prose: E3 has the highest single-seed Validation Accuracy on Movies and Grocery; E4 is highest on ele-fashion. E2/E3 have close total parameter counts (+203 parameters for E3) and close estimated expert/Router linear MACs (+0.33% for E3), but they are different computation and optimization structures. Their common pre-expert, scalar, and LayerNorm initializations match; all common Fusion/Skip tensors differ because random-number consumption diverges during construction. E4 is another single-seed static-routing result. Dataset-specific wins cannot establish general superiority or isolate one mechanism from seed and initialization effects.

## G. V1.1 priority

Prioritize **one** candidate for the first post-review training comparison: a lightweight Text/Visual source-view selector around the existing semantic-response fusion. The evidence for distinct source contexts is strong, while V1 currently fuses source responses using fixed row-quality proportions. Keep the proposal narrow: preserve the existing graph, E2 experts, Router, cross-modal path, and skip; change only how the two semantic source responses are mixed. Treat it as a hypothesis because no source-mixer intervention was performed here.

For a minimal discriminating validation, compare the frozen-protocol E2 baseline with one selector variant over at least three seeds on the same three datasets. Report each seed's Validation Accuracy and Macro-F1, the per-class Validation results, learned source weights, parameter count, and early-stop epoch. Do not use Test labels or select a design from Test performance. The independent review requested for this branch should happen before launching that training.

Do **not** prioritize cross-modal rescaling yet: the frozen branch is small and prediction-insensitive, while E1/E2 Pilot differences are small and mixed. Revisit it only if the source-selection study or a separately preregistered validation question gives a reason to spend training budget there. Do **not** prioritize an expert/fusion redesign: routing changes currently matter to predictions, and skip removal has substantial effects on two datasets. Those results support preserving these mechanisms while isolating the source-mixing question.

## First files to review

1. `IMPLEMENTATION_CHECKS.md` and `data/forward_parity.json` for reproduction and label boundaries.
2. `MECHANISM_REPORT.md` and the intervention/source/router CSVs for measured information flow.
3. `OPTIMIZATION_REVIEW.md`, `data/optimization_evidence.json`, and `data/initialization_audit.json` for parameter and comparison limits.
4. `AUDIT_DESIGN.md` for intervention definitions and interpretation boundaries.
