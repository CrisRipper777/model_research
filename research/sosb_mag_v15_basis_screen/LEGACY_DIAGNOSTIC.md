# Legacy CSE-MAG V1 Diagnostics

- Status: **available** (18/18 checkpoints).
- Device: `cuda:1`.
- Label policy: Only feature arrays, edge indices, checkpoint weights, and model metadata were consulted. No label fields were read/indexed and no split indices or Test metrics were used. MAGB graph deserialization may materialize node fields internally, but this script accesses only graph edges and node count.
- This is descriptive checkpoint analysis only: no shuffle, intervention, retraining, or causal claim.

| Dataset | Seed | Variant | wL | wG | Gate | Text local p mean | Visual local p mean | Text gate mean | Visual gate mean | routing JS mean |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | B1_shared_static | 0.5168 | 0.4832 | 0.1218 | 0.5168 | 0.5168 | 0.1218 | 0.1218 | — |
| Movies | 42 | B3_v1 | — | — | — | 0.8056 | 0.6158 | 0.2036 | 0.5014 | 0.0508 |
| Movies | 43 | B1_shared_static | 0.5156 | 0.4844 | 0.1233 | 0.5156 | 0.5156 | 0.1233 | 0.1233 | — |
| Movies | 43 | B3_v1 | — | — | — | 0.8539 | 0.6895 | 0.2340 | 0.5188 | 0.0285 |
| Movies | 44 | B1_shared_static | 0.5212 | 0.4788 | 0.1232 | 0.5212 | 0.5212 | 0.1232 | 0.1232 | — |
| Movies | 44 | B3_v1 | — | — | — | 0.9949 | 0.6000 | 0.4622 | 0.6439 | 0.1799 |
| Grocery | 42 | B1_shared_static | 0.4792 | 0.5208 | 0.1264 | 0.4792 | 0.4792 | 0.1264 | 0.1264 | — |
| Grocery | 42 | B3_v1 | — | — | — | 0.0993 | 0.1068 | 0.5125 | 0.6414 | 0.0445 |
| Grocery | 43 | B1_shared_static | 0.4761 | 0.5239 | 0.1273 | 0.4761 | 0.4761 | 0.1273 | 0.1273 | — |
| Grocery | 43 | B3_v1 | — | — | — | 0.3591 | 0.2173 | 0.5362 | 0.7676 | 0.0891 |
| Grocery | 44 | B1_shared_static | 0.4777 | 0.5223 | 0.1262 | 0.4777 | 0.4777 | 0.1262 | 0.1262 | — |
| Grocery | 44 | B3_v1 | — | — | — | 0.2863 | 0.2333 | 0.5161 | 0.6671 | 0.0376 |
| ele-fashion | 42 | B1_shared_static | 0.4638 | 0.5362 | 0.1387 | 0.4638 | 0.4638 | 0.1387 | 0.1387 | — |
| ele-fashion | 42 | B3_v1 | — | — | — | 0.4876 | 0.4723 | 0.7185 | 0.7083 | 0.1234 |
| ele-fashion | 43 | B1_shared_static | 0.4751 | 0.5249 | 0.1387 | 0.4751 | 0.4751 | 0.1387 | 0.1387 | — |
| ele-fashion | 43 | B3_v1 | — | — | — | 0.5707 | 0.5724 | 0.6262 | 0.6750 | 0.1410 |
| ele-fashion | 44 | B1_shared_static | 0.4774 | 0.5226 | 0.1361 | 0.4774 | 0.4774 | 0.1361 | 0.1361 | — |
| ele-fashion | 44 | B3_v1 | — | — | — | 0.4504 | 0.4017 | 0.6605 | 0.6389 | 0.1308 |

## Response-space cosine and scale

All response comparisons use active nodes only; cosine distributions are per node, and flattened cosine/RMS use the same active-node subset.

| Dataset | Seed | Variant | Modality | cos(DL,DG) node mean | cos(DL,DG) flattened | RMS(DL) | RMS(DG) | cos(RL,RG) node mean | cos(RL,RG) flattened | RMS(RL) | RMS(RG) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | B1_shared_static | text | 0.1255 | -0.1964 | 0.5696 | 0.6074 | 0.0478 | 0.0479 | 0.9990 | 0.9972 |
| Movies | 42 | B1_shared_static | visual | -0.0562 | -0.2454 | 0.7192 | 0.5720 | -0.0035 | -0.0034 | 0.9996 | 0.9985 |
| Movies | 42 | B3_v1 | text | 0.1575 | -0.1885 | 0.5512 | 0.6113 | 0.0277 | 0.0277 | 0.9987 | 0.9969 |
| Movies | 42 | B3_v1 | visual | -0.0464 | -0.2416 | 0.7164 | 0.5730 | -0.0080 | -0.0080 | 0.9996 | 0.9983 |
| Movies | 43 | B1_shared_static | text | 0.0917 | -0.2066 | 0.5926 | 0.6034 | 0.0297 | 0.0297 | 0.9992 | 0.9973 |
| Movies | 43 | B1_shared_static | visual | -0.0726 | -0.2526 | 0.7332 | 0.5679 | 0.0136 | 0.0137 | 0.9997 | 0.9987 |
| Movies | 43 | B3_v1 | text | 0.1448 | -0.1926 | 0.5600 | 0.6105 | 0.0356 | 0.0355 | 0.9989 | 0.9971 |
| Movies | 43 | B3_v1 | visual | -0.0576 | -0.2458 | 0.7272 | 0.5701 | 0.0023 | 0.0023 | 0.9996 | 0.9986 |
| Movies | 44 | B1_shared_static | text | 0.0826 | -0.2080 | 0.5988 | 0.6014 | 0.0189 | 0.0189 | 0.9993 | 0.9976 |
| Movies | 44 | B1_shared_static | visual | -0.0759 | -0.2543 | 0.7340 | 0.5675 | 0.0040 | 0.0041 | 0.9997 | 0.9986 |
| Movies | 44 | B3_v1 | text | 0.0699 | -0.2122 | 0.6110 | 0.5995 | 0.0046 | 0.0045 | 0.9993 | 0.9977 |
| Movies | 44 | B3_v1 | visual | -0.0809 | -0.2564 | 0.7439 | 0.5655 | -0.0001 | -0.0001 | 0.9996 | 0.9984 |
| Grocery | 42 | B1_shared_static | text | 0.0631 | -0.3000 | 0.6762 | 0.6095 | -0.0144 | -0.0143 | 0.9995 | 0.9952 |
| Grocery | 42 | B1_shared_static | visual | 0.0273 | -0.3120 | 0.6974 | 0.6094 | 0.0314 | 0.0317 | 0.9987 | 0.9965 |
| Grocery | 42 | B3_v1 | text | -0.0353 | -0.2780 | 0.7518 | 0.5702 | 0.0005 | 0.0005 | 0.9996 | 0.9971 |
| Grocery | 42 | B3_v1 | visual | -0.0872 | -0.3024 | 0.7923 | 0.5626 | -0.0042 | -0.0041 | 0.9989 | 0.9974 |
| Grocery | 43 | B1_shared_static | text | 0.0818 | -0.3038 | 0.6662 | 0.6151 | -0.0104 | -0.0102 | 0.9994 | 0.9946 |
| Grocery | 43 | B1_shared_static | visual | 0.0287 | -0.3142 | 0.6962 | 0.6104 | 0.0394 | 0.0397 | 0.9987 | 0.9959 |
| Grocery | 43 | B3_v1 | text | -0.0437 | -0.2811 | 0.7567 | 0.5695 | 0.0052 | 0.0053 | 0.9997 | 0.9973 |
| Grocery | 43 | B3_v1 | visual | -0.1060 | -0.3060 | 0.8069 | 0.5548 | 0.0061 | 0.0062 | 0.9989 | 0.9978 |
| Grocery | 44 | B1_shared_static | text | 0.0835 | -0.3040 | 0.6657 | 0.6158 | -0.0118 | -0.0117 | 0.9994 | 0.9947 |
| Grocery | 44 | B1_shared_static | visual | 0.0397 | -0.3110 | 0.6912 | 0.6119 | 0.0320 | 0.0323 | 0.9987 | 0.9961 |
| Grocery | 44 | B3_v1 | text | -0.0180 | -0.2825 | 0.7364 | 0.5801 | -0.0051 | -0.0051 | 0.9997 | 0.9967 |
| Grocery | 44 | B3_v1 | visual | -0.0738 | -0.2999 | 0.7811 | 0.5669 | -0.0004 | -0.0003 | 0.9988 | 0.9973 |
| ele-fashion | 42 | B1_shared_static | text | -0.3842 | -0.5740 | 0.8930 | 0.7253 | 0.0017 | 0.0018 | 0.9993 | 0.9960 |
| ele-fashion | 42 | B1_shared_static | visual | -0.4335 | -0.6113 | 0.9043 | 0.7484 | 0.0199 | 0.0200 | 0.9977 | 0.9956 |
| ele-fashion | 42 | B3_v1 | text | -0.3789 | -0.5620 | 0.9016 | 0.7182 | -0.0203 | -0.0203 | 0.9993 | 0.9973 |
| ele-fashion | 42 | B3_v1 | visual | -0.4284 | -0.5945 | 0.9134 | 0.7314 | -0.0046 | -0.0046 | 0.9979 | 0.9966 |
| ele-fashion | 43 | B1_shared_static | text | -0.3813 | -0.5741 | 0.8915 | 0.7261 | 0.0109 | 0.0110 | 0.9993 | 0.9958 |
| ele-fashion | 43 | B1_shared_static | visual | -0.4368 | -0.6149 | 0.9060 | 0.7526 | 0.0331 | 0.0332 | 0.9977 | 0.9955 |
| ele-fashion | 43 | B3_v1 | text | -0.3723 | -0.5692 | 0.8953 | 0.7259 | -0.0155 | -0.0155 | 0.9993 | 0.9971 |
| ele-fashion | 43 | B3_v1 | visual | -0.4321 | -0.6088 | 0.9086 | 0.7481 | -0.0010 | -0.0010 | 0.9978 | 0.9967 |
| ele-fashion | 44 | B1_shared_static | text | -0.3803 | -0.5830 | 0.8883 | 0.7363 | 0.0073 | 0.0074 | 0.9993 | 0.9955 |
| ele-fashion | 44 | B1_shared_static | visual | -0.4270 | -0.6160 | 0.8986 | 0.7551 | -0.0191 | -0.0191 | 0.9977 | 0.9955 |
| ele-fashion | 44 | B3_v1 | text | -0.3724 | -0.5643 | 0.8981 | 0.7225 | -0.0171 | -0.0171 | 0.9993 | 0.9973 |
| ele-fashion | 44 | B3_v1 | visual | -0.4282 | -0.5945 | 0.9148 | 0.7311 | -0.0272 | -0.0273 | 0.9979 | 0.9972 |
