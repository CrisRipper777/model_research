# Decision handoff

The screen finished 7/7 authorized cells. On the primary Movies target, Best-Accuracy checkpoint mean differences are Acc -0.170 pp, Macro-F1 -0.634 pp and CE +0.04082; this does not reach the suggested +0.3 pp Acc empirical signal. Grocery and ele-fashion show small positive mean Acc/F1 differences with mixed CE changes, so the overall result is mixed and does not support promoting the condition.

Review the per-seed paired metrics and parameter/resource cost in `SCREENING_REPORT.md` before deciding whether this narrowly scoped change merits a separately authorized follow-up. This screen alone does not establish a general improvement: sample counts are small, optimization trajectories differ, and the changed expert input can affect the router and Joint View downstream.

No default-model change or follow-on run is made by this screen. Test metrics were not computed.
