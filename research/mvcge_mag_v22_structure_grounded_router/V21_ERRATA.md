# V2.1 report erratum

The V2.1 report says that a both-modality dead expert appeared in “9/9
run-checkpoints” for U0. This wording conflates the number of dead expert slots
with the number of run-checkpoints containing at least one such slot.

The committed V2.1 `routing_diagnostics.csv` gives the corrected distribution
for the nine U0 dataset-seed checkpoints:

- 9 both-modality dead expert slots in total;
- those slots occur across 7/9 run-checkpoints;
- 2 checkpoints have 0 dead slots, 5 have 1, and 2 have 2.

U0 is modality-static Top-2 routing. A both-modality zero-load slot in this
control therefore describes its selected static expert pair; it must not be
automatically interpreted as dynamic-router collapse. The prior V2.1 report
and results are left unchanged, and V2.1 was not rerun.
