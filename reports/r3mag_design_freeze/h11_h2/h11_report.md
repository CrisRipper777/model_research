# R³-MAG H1.1 action-bank control screen

Descriptive analysis only; no H1 pass/fail gate is applied. Test labels and metrics were not read.

## Findings

The report compares own structural directions, coherent single-donor trajectories, the previous independent-donor null, and shared signed-permutation geometry scrambles. All banks use eight unique directions per node across the two modalities, nine actions per modality including NOOP, NormMatch, and full joint frozen-host evaluation.

Geometry fields report the mean across Audit targets separately for Text and Visual. Utility fields are nats of CE reduction. For each registered epsilon, every dataset/host-seed appears in h11_metrics.csv.

## Direction geometry

| Dataset | Seed | Bank | Text effective rank | Visual effective rank | Text mean |cos| | Visual mean |cos| |
|---|---:|---|---:|---:|---:|---:|
| Movies | 42 | CoherentDonor | 1.7272 | 2.0202 | 0.7015 | 0.5686 |
| Movies | 42 | GeometryScrambled | 1.7264 | 2.0192 | 0.7024 | 0.5678 |
| Movies | 42 | IndependentDonor | 2.5389 | 3.0557 | 0.5704 | 0.4091 |
| Movies | 42 | OwnStructural | 1.7264 | 2.0192 | 0.7024 | 0.5678 |
| Movies | 43 | CoherentDonor | 1.6822 | 2.0581 | 0.7242 | 0.5584 |
| Movies | 43 | GeometryScrambled | 1.6778 | 2.0582 | 0.7262 | 0.5577 |
| Movies | 43 | IndependentDonor | 2.3591 | 3.0790 | 0.6154 | 0.3971 |
| Movies | 43 | OwnStructural | 1.6778 | 2.0582 | 0.7262 | 0.5577 |
| Movies | 44 | CoherentDonor | 1.7787 | 2.0461 | 0.6760 | 0.5639 |
| Movies | 44 | GeometryScrambled | 1.7751 | 2.0460 | 0.6775 | 0.5624 |
| Movies | 44 | IndependentDonor | 2.6036 | 3.0400 | 0.5427 | 0.4075 |
| Movies | 44 | OwnStructural | 1.7751 | 2.0460 | 0.6775 | 0.5624 |
| Grocery | 42 | CoherentDonor | 2.0160 | 2.0947 | 0.5893 | 0.5608 |
| Grocery | 42 | GeometryScrambled | 2.0103 | 2.1016 | 0.5903 | 0.5570 |
| Grocery | 42 | IndependentDonor | 2.7437 | 2.9097 | 0.4871 | 0.4341 |
| Grocery | 42 | OwnStructural | 2.0103 | 2.1016 | 0.5903 | 0.5570 |
| Grocery | 43 | CoherentDonor | 2.0115 | 2.1077 | 0.5881 | 0.5536 |
| Grocery | 43 | GeometryScrambled | 2.0047 | 2.1158 | 0.5893 | 0.5507 |
| Grocery | 43 | IndependentDonor | 2.7899 | 2.9494 | 0.4769 | 0.4252 |
| Grocery | 43 | OwnStructural | 2.0047 | 2.1158 | 0.5893 | 0.5507 |
| Grocery | 44 | CoherentDonor | 2.0027 | 2.1107 | 0.5929 | 0.5518 |
| Grocery | 44 | GeometryScrambled | 1.9922 | 2.1129 | 0.5967 | 0.5505 |
| Grocery | 44 | IndependentDonor | 2.7733 | 2.9349 | 0.4819 | 0.4292 |
| Grocery | 44 | OwnStructural | 1.9922 | 2.1129 | 0.5967 | 0.5505 |
| ele-fashion | 42 | CoherentDonor | 1.5882 | 1.8049 | 0.7762 | 0.6805 |
| ele-fashion | 42 | GeometryScrambled | 1.5897 | 1.8040 | 0.7757 | 0.6820 |
| ele-fashion | 42 | IndependentDonor | 2.3893 | 2.5793 | 0.6126 | 0.5545 |
| ele-fashion | 42 | OwnStructural | 1.5897 | 1.8040 | 0.7757 | 0.6820 |
| ele-fashion | 43 | CoherentDonor | 1.5970 | 1.7918 | 0.7739 | 0.6877 |
| ele-fashion | 43 | GeometryScrambled | 1.5967 | 1.7893 | 0.7743 | 0.6895 |
| ele-fashion | 43 | IndependentDonor | 2.4512 | 2.5881 | 0.5990 | 0.5566 |
| ele-fashion | 43 | OwnStructural | 1.5967 | 1.7893 | 0.7743 | 0.6895 |
| ele-fashion | 44 | CoherentDonor | 1.6027 | 1.8263 | 0.7713 | 0.6709 |
| ele-fashion | 44 | GeometryScrambled | 1.6025 | 1.8240 | 0.7715 | 0.6729 |
| ele-fashion | 44 | IndependentDonor | 2.4454 | 2.6413 | 0.6009 | 0.5425 |
| ele-fashion | 44 | OwnStructural | 1.6025 | 1.8240 | 0.7715 | 0.6729 |

## Utility comparison

| Dataset | Seed | ε | Bank | Text oracle | Text headroom | Visual oracle | Visual headroom | Joint separate−shared |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Movies | 42 | 0.1 | CoherentDonor | 0.10538 | 0.07842 | 0.17990 | 0.14859 | 0.03693 |
| Movies | 42 | 0.1 | GeometryScrambled | 0.05944 | 0.05289 | 0.10503 | 0.09667 | 0.03911 |
| Movies | 42 | 0.1 | IndependentDonor | 0.12201 | 0.09547 | 0.20115 | 0.16806 | 0.05048 |
| Movies | 42 | 0.1 | OwnStructural | 0.07025 | 0.06085 | 0.12121 | 0.10697 | 0.03122 |
| Movies | 42 | 0.2 | CoherentDonor | 0.20351 | 0.15527 | 0.33193 | 0.29570 | 0.06900 |
| Movies | 42 | 0.2 | GeometryScrambled | 0.11801 | 0.10246 | 0.20477 | 0.18584 | 0.07186 |
| Movies | 42 | 0.2 | IndependentDonor | 0.23504 | 0.18737 | 0.37066 | 0.33127 | 0.08984 |
| Movies | 42 | 0.2 | OwnStructural | 0.13842 | 0.12104 | 0.23184 | 0.21046 | 0.05982 |
| Movies | 43 | 0.1 | CoherentDonor | 0.08934 | 0.07204 | 0.16409 | 0.14128 | 0.03529 |
| Movies | 43 | 0.1 | GeometryScrambled | 0.04734 | 0.04279 | 0.09132 | 0.08408 | 0.03136 |
| Movies | 43 | 0.1 | IndependentDonor | 0.10262 | 0.08524 | 0.18319 | 0.16058 | 0.04754 |
| Movies | 43 | 0.1 | OwnStructural | 0.06156 | 0.05507 | 0.11480 | 0.10152 | 0.03208 |
| Movies | 43 | 0.2 | CoherentDonor | 0.17321 | 0.14344 | 0.30221 | 0.28019 | 0.06566 |
| Movies | 43 | 0.2 | GeometryScrambled | 0.09434 | 0.08319 | 0.17723 | 0.16207 | 0.05836 |
| Movies | 43 | 0.2 | IndependentDonor | 0.19874 | 0.16852 | 0.33748 | 0.31564 | 0.08520 |
| Movies | 43 | 0.2 | OwnStructural | 0.12169 | 0.11010 | 0.22018 | 0.20112 | 0.06191 |
| Movies | 44 | 0.1 | CoherentDonor | 0.09311 | 0.07938 | 0.15323 | 0.13981 | 0.03518 |
| Movies | 44 | 0.1 | GeometryScrambled | 0.04984 | 0.04630 | 0.08236 | 0.07747 | 0.03236 |
| Movies | 44 | 0.1 | IndependentDonor | 0.10690 | 0.09454 | 0.17058 | 0.15677 | 0.04811 |
| Movies | 44 | 0.1 | OwnStructural | 0.06291 | 0.06079 | 0.10814 | 0.09728 | 0.03202 |
| Movies | 44 | 0.2 | CoherentDonor | 0.17921 | 0.15684 | 0.28305 | 0.27542 | 0.06507 |
| Movies | 44 | 0.2 | GeometryScrambled | 0.09867 | 0.08969 | 0.15941 | 0.14920 | 0.06016 |
| Movies | 44 | 0.2 | IndependentDonor | 0.20519 | 0.18532 | 0.31475 | 0.30599 | 0.08576 |
| Movies | 44 | 0.2 | OwnStructural | 0.12439 | 0.12090 | 0.20761 | 0.19193 | 0.06188 |
| Grocery | 42 | 0.1 | CoherentDonor | 0.06160 | 0.05397 | 0.08306 | 0.07553 | 0.01357 |
| Grocery | 42 | 0.1 | GeometryScrambled | 0.02805 | 0.02556 | 0.03765 | 0.03364 | 0.01763 |
| Grocery | 42 | 0.1 | IndependentDonor | 0.06750 | 0.05949 | 0.09078 | 0.08263 | 0.02042 |
| Grocery | 42 | 0.1 | OwnStructural | 0.04473 | 0.04231 | 0.05604 | 0.05508 | 0.01224 |
| Grocery | 42 | 0.2 | CoherentDonor | 0.11545 | 0.10847 | 0.15203 | 0.14991 | 0.02454 |
| Grocery | 42 | 0.2 | GeometryScrambled | 0.05454 | 0.04972 | 0.07387 | 0.06477 | 0.03208 |
| Grocery | 42 | 0.2 | IndependentDonor | 0.12620 | 0.11874 | 0.16587 | 0.16279 | 0.03494 |
| Grocery | 42 | 0.2 | OwnStructural | 0.08547 | 0.08258 | 0.10545 | 0.10545 | 0.02404 |
| Grocery | 43 | 0.1 | CoherentDonor | 0.06989 | 0.05670 | 0.09005 | 0.07575 | 0.01564 |
| Grocery | 43 | 0.1 | GeometryScrambled | 0.03085 | 0.02806 | 0.04301 | 0.03860 | 0.01884 |
| Grocery | 43 | 0.1 | IndependentDonor | 0.07725 | 0.06318 | 0.09866 | 0.08436 | 0.02326 |
| Grocery | 43 | 0.1 | OwnStructural | 0.04814 | 0.04614 | 0.06515 | 0.06515 | 0.01246 |
| Grocery | 43 | 0.2 | CoherentDonor | 0.12985 | 0.11466 | 0.16354 | 0.15250 | 0.02774 |
| Grocery | 43 | 0.2 | GeometryScrambled | 0.06007 | 0.05435 | 0.08416 | 0.07392 | 0.03392 |
| Grocery | 43 | 0.2 | IndependentDonor | 0.14315 | 0.12673 | 0.17867 | 0.16720 | 0.03848 |
| Grocery | 43 | 0.2 | OwnStructural | 0.09220 | 0.08974 | 0.12203 | 0.12203 | 0.02433 |
| Grocery | 44 | 0.1 | CoherentDonor | 0.06628 | 0.05678 | 0.08700 | 0.07811 | 0.01438 |
| Grocery | 44 | 0.1 | GeometryScrambled | 0.03116 | 0.02893 | 0.03817 | 0.03432 | 0.01904 |
| Grocery | 44 | 0.1 | IndependentDonor | 0.07301 | 0.06282 | 0.09445 | 0.08441 | 0.02118 |
| Grocery | 44 | 0.1 | OwnStructural | 0.04572 | 0.04433 | 0.06104 | 0.06104 | 0.01200 |
| Grocery | 44 | 0.2 | CoherentDonor | 0.12339 | 0.11473 | 0.15817 | 0.15525 | 0.02578 |
| Grocery | 44 | 0.2 | GeometryScrambled | 0.06054 | 0.05606 | 0.07489 | 0.06591 | 0.03442 |
| Grocery | 44 | 0.2 | IndependentDonor | 0.13544 | 0.12572 | 0.17090 | 0.16680 | 0.03519 |
| Grocery | 44 | 0.2 | OwnStructural | 0.08699 | 0.08575 | 0.11449 | 0.11449 | 0.02285 |
| ele-fashion | 42 | 0.1 | CoherentDonor | 0.04803 | 0.03877 | 0.03199 | 0.02706 | 0.00813 |
| ele-fashion | 42 | 0.1 | GeometryScrambled | 0.02454 | 0.02221 | 0.02151 | 0.01936 | 0.01202 |
| ele-fashion | 42 | 0.1 | IndependentDonor | 0.05592 | 0.04686 | 0.03668 | 0.03171 | 0.01215 |
| ele-fashion | 42 | 0.1 | OwnStructural | 0.03086 | 0.02987 | 0.02203 | 0.01970 | 0.00620 |
| ele-fashion | 42 | 0.2 | CoherentDonor | 0.09028 | 0.08113 | 0.06140 | 0.05355 | 0.01444 |
| ele-fashion | 42 | 0.2 | GeometryScrambled | 0.04845 | 0.04364 | 0.04173 | 0.03749 | 0.02192 |
| ele-fashion | 42 | 0.2 | IndependentDonor | 0.10453 | 0.09536 | 0.07005 | 0.06216 | 0.02043 |
| ele-fashion | 42 | 0.2 | OwnStructural | 0.05880 | 0.05779 | 0.04286 | 0.03957 | 0.01165 |
| ele-fashion | 43 | 0.1 | CoherentDonor | 0.05510 | 0.03950 | 0.03806 | 0.02808 | 0.00880 |
| ele-fashion | 43 | 0.1 | GeometryScrambled | 0.02988 | 0.02544 | 0.02703 | 0.02319 | 0.01526 |
| ele-fashion | 43 | 0.1 | IndependentDonor | 0.06402 | 0.04788 | 0.04369 | 0.03368 | 0.01305 |
| ele-fashion | 43 | 0.1 | OwnStructural | 0.03479 | 0.03086 | 0.02645 | 0.02070 | 0.00730 |
| ele-fashion | 43 | 0.2 | CoherentDonor | 0.10378 | 0.08238 | 0.07321 | 0.05526 | 0.01546 |
| ele-fashion | 43 | 0.2 | GeometryScrambled | 0.05900 | 0.04965 | 0.05256 | 0.04471 | 0.02736 |
| ele-fashion | 43 | 0.2 | IndependentDonor | 0.11981 | 0.09718 | 0.08353 | 0.06559 | 0.02171 |
| ele-fashion | 43 | 0.2 | OwnStructural | 0.06643 | 0.06301 | 0.05134 | 0.04130 | 0.01360 |
| ele-fashion | 44 | 0.1 | CoherentDonor | 0.05503 | 0.03934 | 0.03736 | 0.02851 | 0.00878 |
| ele-fashion | 44 | 0.1 | GeometryScrambled | 0.02935 | 0.02510 | 0.02786 | 0.02460 | 0.01531 |
| ele-fashion | 44 | 0.1 | IndependentDonor | 0.06406 | 0.04830 | 0.04285 | 0.03369 | 0.01303 |
| ele-fashion | 44 | 0.1 | OwnStructural | 0.03417 | 0.03074 | 0.02540 | 0.02097 | 0.00594 |
| ele-fashion | 44 | 0.2 | CoherentDonor | 0.10379 | 0.08282 | 0.07184 | 0.05622 | 0.01555 |
| ele-fashion | 44 | 0.2 | GeometryScrambled | 0.05828 | 0.04926 | 0.05413 | 0.04751 | 0.02752 |
| ele-fashion | 44 | 0.2 | IndependentDonor | 0.11985 | 0.09882 | 0.08192 | 0.06574 | 0.02166 |
| ele-fashion | 44 | 0.2 | OwnStructural | 0.06528 | 0.06282 | 0.04950 | 0.04212 | 0.01126 |

## Four requested interpretation points

A. Independent minus own effective rank is +0.835 on average; mean absolute pairwise cosine changes by -0.136. Positive rank and negative |cos| shifts indicate a more diverse independent-donor bank.
B. Own minus coherent-donor headroom averages -0.0200 Text, -0.0333 Visual, and -0.0032 joint separate-vs-shared utility.
C. Own minus geometry-scrambled oracle utility averages +0.0161 Text, +0.0194 Visual, and +0.0313 joint pair utility; compare with preserved Gram/effective-rank QA.
D. Higher independent-donor rank/lower correlation together with a narrowed own-vs-coherent gap would make action-bank diversity a plausible contributor to the previous negative structural-minus-independent excess. These controls are descriptive and cannot assign a causal share.
Observed here, independent-donor rank is much higher and |cos| lower, but coherent donors also beat own directions on average despite nearly matching own-bank rank/correlation. Geometry scrambling preserves own Gram geometry while lowering average single-modality oracle utility. This makes action-bank diversity a plausible part of the independent-null advantage, but the coherent result shows diversity alone does not account for the full gap; donor alignment and host-coordinate sensitivity remain possible contributors.
