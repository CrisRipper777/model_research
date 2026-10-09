# V3A interpretation note for the V3C design

This note narrows the V3A conclusion and records the motivation for screening
local-relative functional edge roles. V3A outputs and artifacts remain
unchanged; no V3A analysis or training is rerun here.

## Scale-free comparison of modality operators

The V3A Text-versus-Visual effective-operator absolute L2 differences were
approximately 1.84 (Movies), 2.15 (Grocery), and 4.49 (`ele-fashion`). These
absolute norms depend on graph size and operator scale, so they are not a
scale-free measure. The corresponding relative L2 differences were about
0.0320, 0.0396, and 0.0342; operator cosines were about 0.9995, 0.9992, and
0.9994.

## Revised conclusion

V3A fixed semantic edge reweighting was numerically non-identical, but its
effective operator and resulting expert profiles were functionally near
redundant with the raw structural context in this screen. Effective-outside-
raw-span novelty was generally about 0.3%–2.5%, and raw/effective profile
cosine was approximately 0.9998. C2/C3 did not pass the frozen stable-positive
validation gate. This supports investigating a different physical-edge action
type; it does not establish that the NC task rejects all distinct contexts.

## Historical C0 Macro-F1 direction

For V3A C0 versus historical V2.2 R0, Macro-F1 was positive in 0/9 paired
runs, so the observed direction is **non-positive** (negative or tied), not a
mixture of positive and negative deltas. The mean difference was about
-0.337 pp. This is descriptive and was not significance-tested.

## V3C terminology

V3C calls edges Supportive or Discrepant according to a fixed local-relative
semantic score. Discrepant means lower cosine consistency relative to the
endpoints' incoming-neighbor baselines. It is not a label-derived role and
must not be called ground-truth heterophily.
