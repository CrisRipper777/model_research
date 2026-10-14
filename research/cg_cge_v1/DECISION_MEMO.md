# CG-CGE V1 Decision Memo

## Decision status

No decision about retaining B1 or B2 is justified before the frozen 18-run Validation study is complete. No next model version is authorized by this memo.

## Questions the evidence will answer

1. Do B1 or B2 improve Validation Accuracy, Macro-F1 or CE over strictly matched frozen View-CGE A2 on each dataset and seed?
2. Does B2 Role-FiLM add a stable benefit over B1, or does the added complexity fail to pay for itself?
3. Are Cross candidates selected, gated and propagated at material scale, or are they mostly rejected/attenuated?
4. Do auxiliary Text/Visual heads receive training-only supervision and improve the single-modality backbone's learning without leaking validation or Test labels?
5. Are run-to-run variance, overfitting, memory and time acceptable for the measured gain?

## Decision rule

Use only the three-seed paired Validation results plus implementation and resource audits. Report all individual seeds and sign counts; do not present three seeds as precise uncertainty. Distinguish code facts, measurements and explanations. A useful mechanism statistic is not sufficient reason to retain complexity. If B1 is competitive and B2 is not, prefer the simpler B1. If neither is competitive, document the limits and stop at this frozen study rather than automatically designing a successor.

Current evidence: implementation under audit; formal run results pending. Test remains disabled.
