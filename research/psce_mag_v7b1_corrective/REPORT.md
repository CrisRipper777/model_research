# PSCE-MAG V7B.1 corrective replication report

Branch `exp/v7b1_corrective_replication`; baseline `75157f052a50c60b791566b2c91207e372956d7b`; freeze `72d80b8a58cd1b10a5df6c5afab9528164296d8b`.
The validation-only campaign completed 54/54 full-graph NC runs under `unified_full_graph_nc_v1`. `task.evaluate_test=false`; no Test metrics were computed, loaded, saved, or reported. Checkpoint selection used Validation Accuracy. Macro-F1 is diagnostic only.

## Formal results by dataset and seed

Each row is one dataset × seed × variant run. Accuracy and Macro-F1 are validation values. Timings and peak GPU memory come from run metadata.

| Dataset | Seed | Group | Val Accuracy | Val Macro-F1 | Best epoch | Epochs | Train seconds | Trainable model params | Peak allocated / reserved GiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | R0 Raw | 55.609% | 47.562% | 87 | 117 | 19.3 | 1,448,885 | 2.28 / 2.57 |
| Movies | 42 | B0 learned V7A A2 | 55.999% | 48.481% | 72 | 102 | 27.1 | 1,576,387 | 3.32 / 3.68 |
| Movies | 42 | B1 fixed KNN | 55.969% | 47.883% | 76 | 106 | 27.1 | 1,551,682 | 3.32 / 3.68 |
| Movies | 42 | B2 random fixed | 56.029% | 47.543% | 109 | 139 | 35.0 | 1,551,682 | 3.32 / 3.68 |
| Movies | 42 | B3 relative shared | 56.209% | 48.625% | 100 | 130 | 34.7 | 1,576,578 | 3.32 / 3.69 |
| Movies | 42 | B4 relative modality | 56.089% | 47.422% | 105 | 135 | 35.6 | 1,576,578 | 3.32 / 3.69 |
| Movies | 43 | R0 Raw | 55.399% | 45.980% | 57 | 87 | 14.8 | 1,448,885 | 2.28 / 2.57 |
| Movies | 43 | B0 learned V7A A2 | 55.759% | 48.138% | 54 | 84 | 22.7 | 1,576,387 | 3.32 / 3.68 |
| Movies | 43 | B1 fixed KNN | 56.149% | 48.338% | 86 | 116 | 29.6 | 1,551,682 | 3.32 / 3.68 |
| Movies | 43 | B2 random fixed | 55.519% | 46.949% | 57 | 87 | 22.3 | 1,551,682 | 3.32 / 3.68 |
| Movies | 43 | B3 relative shared | 55.759% | 47.490% | 55 | 85 | 22.9 | 1,576,578 | 3.32 / 3.69 |
| Movies | 43 | B4 relative modality | 55.609% | 49.164% | 71 | 101 | 26.9 | 1,576,578 | 3.32 / 3.69 |
| Movies | 44 | R0 Raw | 56.449% | 49.959% | 70 | 100 | 16.9 | 1,448,885 | 2.28 / 2.57 |
| Movies | 44 | B0 learned V7A A2 | 56.569% | 50.762% | 77 | 107 | 28.3 | 1,576,387 | 3.32 / 3.68 |
| Movies | 44 | B1 fixed KNN | 56.359% | 50.167% | 71 | 101 | 25.8 | 1,551,682 | 3.32 / 3.68 |
| Movies | 44 | B2 random fixed | 56.869% | 49.801% | 79 | 109 | 27.8 | 1,551,682 | 3.32 / 3.68 |
| Movies | 44 | B3 relative shared | 56.809% | 50.199% | 73 | 103 | 27.5 | 1,576,578 | 3.32 / 3.69 |
| Movies | 44 | B4 relative modality | 56.719% | 50.429% | 80 | 110 | 29.4 | 1,576,578 | 3.32 / 3.69 |
| Grocery | 42 | R0 Raw | 83.309% | 77.433% | 70 | 100 | 16.7 | 1,448,885 | 2.15 / 2.46 |
| Grocery | 42 | B0 learned V7A A2 | 83.338% | 77.675% | 74 | 104 | 27.9 | 1,576,387 | 3.16 / 3.58 |
| Grocery | 42 | B1 fixed KNN | 83.367% | 78.076% | 94 | 124 | 31.5 | 1,551,682 | 3.16 / 3.58 |
| Grocery | 42 | B2 random fixed | 83.163% | 76.914% | 66 | 96 | 24.6 | 1,551,682 | 3.16 / 3.58 |
| Grocery | 42 | B3 relative shared | 83.455% | 77.136% | 79 | 109 | 29.8 | 1,576,578 | 3.17 / 3.58 |
| Grocery | 42 | B4 relative modality | 83.543% | 77.192% | 76 | 106 | 28.7 | 1,576,578 | 3.17 / 3.58 |
| Grocery | 43 | R0 Raw | 83.367% | 75.966% | 71 | 101 | 17.3 | 1,448,885 | 2.15 / 2.46 |
| Grocery | 43 | B0 learned V7A A2 | 83.836% | 76.394% | 82 | 112 | 30.0 | 1,576,387 | 3.16 / 3.58 |
| Grocery | 43 | B1 fixed KNN | 83.836% | 75.424% | 89 | 119 | 30.4 | 1,551,682 | 3.16 / 3.58 |
| Grocery | 43 | B2 random fixed | 83.953% | 76.445% | 71 | 101 | 25.8 | 1,551,682 | 3.16 / 3.58 |
| Grocery | 43 | B3 relative shared | 84.100% | 76.394% | 82 | 112 | 30.4 | 1,576,578 | 3.17 / 3.58 |
| Grocery | 43 | B4 relative modality | 83.836% | 75.720% | 96 | 126 | 33.8 | 1,576,578 | 3.17 / 3.58 |
| Grocery | 44 | R0 Raw | 82.635% | 74.038% | 61 | 91 | 15.7 | 1,448,885 | 2.15 / 2.46 |
| Grocery | 44 | B0 learned V7A A2 | 83.104% | 75.312% | 111 | 141 | 38.0 | 1,576,387 | 3.16 / 3.58 |
| Grocery | 44 | B1 fixed KNN | 82.928% | 73.568% | 69 | 99 | 25.4 | 1,551,682 | 3.16 / 3.58 |
| Grocery | 44 | B2 random fixed | 82.928% | 72.447% | 60 | 90 | 23.6 | 1,551,682 | 3.16 / 3.58 |
| Grocery | 44 | B3 relative shared | 82.958% | 73.161% | 63 | 93 | 25.2 | 1,576,578 | 3.17 / 3.58 |
| Grocery | 44 | B4 relative modality | 83.045% | 74.402% | 71 | 101 | 27.5 | 1,576,578 | 3.17 / 3.58 |
| ele-fashion | 42 | R0 Raw | 87.644% | 75.859% | 217 | 247 | 142.6 | 1,186,741 | 8.77 / 10.46 |
| ele-fashion | 42 | B0 learned V7A A2 | 87.542% | 74.243% | 151 | 181 | 182.6 | 1,314,243 | 13.80 / 15.13 |
| ele-fashion | 42 | B1 fixed KNN | 87.829% | 74.511% | 225 | 255 | 231.8 | 1,289,538 | 13.79 / 15.14 |
| ele-fashion | 42 | B2 random fixed | 87.184% | 73.599% | 125 | 155 | 142.1 | 1,289,538 | 13.79 / 15.14 |
| ele-fashion | 42 | B3 relative shared | 87.696% | 74.776% | 134 | 164 | 167.8 | 1,314,434 | 13.82 / 15.15 |
| ele-fashion | 42 | B4 relative modality | 87.440% | 75.029% | 132 | 162 | 166.0 | 1,314,434 | 13.83 / 15.14 |
| ele-fashion | 43 | R0 Raw | 87.235% | 72.702% | 133 | 163 | 94.6 | 1,186,741 | 8.77 / 10.46 |
| ele-fashion | 43 | B0 learned V7A A2 | 87.235% | 72.707% | 98 | 128 | 129.3 | 1,314,243 | 13.80 / 15.13 |
| ele-fashion | 43 | B1 fixed KNN | 87.358% | 73.561% | 114 | 144 | 131.9 | 1,289,538 | 13.79 / 15.14 |
| ele-fashion | 43 | B2 random fixed | 87.399% | 74.780% | 169 | 199 | 182.5 | 1,289,538 | 13.79 / 15.14 |
| ele-fashion | 43 | B3 relative shared | 87.491% | 74.517% | 152 | 182 | 185.3 | 1,314,434 | 13.82 / 15.15 |
| ele-fashion | 43 | B4 relative modality | 87.307% | 73.342% | 98 | 128 | 131.3 | 1,314,434 | 13.83 / 15.14 |
| ele-fashion | 44 | R0 Raw | 87.460% | 75.156% | 204 | 234 | 136.2 | 1,186,741 | 8.77 / 10.46 |
| ele-fashion | 44 | B0 learned V7A A2 | 87.737% | 74.693% | 163 | 193 | 194.6 | 1,314,243 | 13.80 / 15.13 |
| ele-fashion | 44 | B1 fixed KNN | 87.542% | 74.628% | 158 | 188 | 171.5 | 1,289,538 | 13.79 / 15.14 |
| ele-fashion | 44 | B2 random fixed | 87.266% | 73.751% | 173 | 203 | 185.8 | 1,289,538 | 13.79 / 15.14 |
| ele-fashion | 44 | B3 relative shared | 87.491% | 74.181% | 115 | 145 | 147.8 | 1,314,434 | 13.82 / 15.15 |
| ele-fashion | 44 | B4 relative modality | 87.399% | 73.881% | 125 | 155 | 159.0 | 1,314,434 | 13.83 / 15.14 |

## Seed mean ± population SD

Seeds are repeated runs within each dataset. They are not independent datasets.

| Dataset | Group | Val Accuracy (%) | Val Macro-F1 (%) | Best epoch | Epochs completed | Trainable params | Max allocated / reserved GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | R0 Raw | 55.819 ± 0.454 | 47.834 ± 1.636 | 71.3 ± 12.3 | 101.3 | 1,448,885 | 2.28 / 2.57 |
| Movies | B0 learned V7A A2 | 56.109 ± 0.340 | 49.127 ± 1.165 | 67.7 ± 9.9 | 97.7 | 1,576,387 | 3.32 / 3.68 |
| Movies | B1 fixed KNN | 56.159 ± 0.159 | 48.796 ± 0.987 | 77.7 ± 6.2 | 107.7 | 1,551,682 | 3.32 / 3.68 |
| Movies | B2 random fixed | 56.139 ± 0.556 | 48.098 ± 1.229 | 81.7 ± 21.3 | 111.7 | 1,551,682 | 3.32 / 3.68 |
| Movies | B3 relative shared | 56.259 ± 0.430 | 48.771 ± 1.111 | 76.0 ± 18.5 | 106.0 | 1,576,578 | 3.32 / 3.69 |
| Movies | B4 relative modality | 56.139 ± 0.454 | 49.005 ± 1.233 | 85.3 ± 14.4 | 115.3 | 1,576,578 | 3.32 / 3.69 |
| Grocery | R0 Raw | 83.104 ± 0.332 | 75.812 ± 1.390 | 67.3 ± 4.5 | 97.3 | 1,448,885 | 2.15 / 2.46 |
| Grocery | B0 learned V7A A2 | 83.426 ± 0.305 | 76.461 ± 0.966 | 89.0 ± 15.9 | 119.0 | 1,576,387 | 3.16 / 3.58 |
| Grocery | B1 fixed KNN | 83.377 ± 0.371 | 75.689 ± 1.850 | 84.0 ± 10.8 | 114.0 | 1,551,682 | 3.16 / 3.58 |
| Grocery | B2 random fixed | 83.348 ± 0.438 | 75.269 ± 2.004 | 65.7 ± 4.5 | 95.7 | 1,551,682 | 3.16 / 3.58 |
| Grocery | B3 relative shared | 83.504 ± 0.468 | 75.564 ± 1.726 | 74.7 ± 8.3 | 104.7 | 1,576,578 | 3.17 / 3.58 |
| Grocery | B4 relative modality | 83.475 ± 0.326 | 75.771 ± 1.140 | 81.0 ± 10.8 | 111.0 | 1,576,578 | 3.17 / 3.58 |
| ele-fashion | R0 Raw | 87.447 ± 0.167 | 74.572 ± 1.353 | 184.7 ± 36.9 | 214.7 | 1,186,741 | 8.77 / 10.46 |
| ele-fashion | B0 learned V7A A2 | 87.505 ± 0.206 | 73.881 ± 0.850 | 137.3 ± 28.2 | 167.3 | 1,314,243 | 13.80 / 15.13 |
| ele-fashion | B1 fixed KNN | 87.576 ± 0.194 | 74.233 ± 0.478 | 165.7 ± 45.6 | 195.7 | 1,289,538 | 13.79 / 15.14 |
| ele-fashion | B2 random fixed | 87.283 ± 0.089 | 74.043 ± 0.524 | 155.7 ± 21.7 | 185.7 | 1,289,538 | 13.79 / 15.14 |
| ele-fashion | B3 relative shared | 87.559 ± 0.096 | 74.492 ± 0.244 | 133.7 ± 15.1 | 163.7 | 1,314,434 | 13.82 / 15.15 |
| ele-fashion | B4 relative modality | 87.382 ± 0.056 | 74.084 ± 0.703 | 118.3 ± 14.7 | 148.3 | 1,314,434 | 13.83 / 15.14 |

## Eight paired comparisons

Differences are left minus right, paired by dataset and training seed. Each dataset row shows all three seed-paired deltas, the mean ± population SD, and the number of positive seed pairs. The final row weights the three dataset means equally and is descriptive only. No significance test is applied.

### B1 minus R0

| Dataset | Accuracy Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds | Macro-F1 Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds |
|---|---|---:|---:|---|---:|---:|
| Movies | +0.360, +0.750, -0.090 | +0.340 ± 0.343 | 2/3 | +0.321, +2.357, +0.208 | +0.962 ± 0.988 | 3/3 |
| Grocery | +0.059, +0.469, +0.293 | +0.273 ± 0.168 | 3/3 | +0.643, -0.542, -0.470 | -0.123 ± 0.542 | 1/3 |
| ele-fashion | +0.184, +0.123, +0.082 | +0.130 ± 0.042 | 3/3 | -1.348, +0.859, -0.528 | -0.339 ± 0.911 | 1/3 |
| Equal-weight dataset means | {"Grocery": 0.27330517768859863, "Movies": 0.3399312496185303, "ele-fashion": 0.12955466906229657} | +0.248 ± 0.088 | descriptive | {"Grocery": -0.12283925064881702, "Movies": 0.9620350317970668, "ele-fashion": -0.33910103965226757} | +0.167 ± 0.569 | descriptive |

### B2 minus R0

| Dataset | Accuracy Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds | Macro-F1 Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds |
|---|---|---:|---:|---|---:|---:|
| Movies | +0.420, +0.120, +0.420 | +0.320 ± 0.141 | 3/3 | -0.019, +0.969, -0.158 | +0.264 ± 0.502 | 1/3 |
| Grocery | -0.146, +0.586, +0.293 | +0.244 ± 0.301 | 2/3 | -0.519, +0.479, -1.591 | -0.543 ± 0.845 | 1/3 |
| ele-fashion | -0.460, +0.164, -0.194 | -0.164 ± 0.256 | 1/3 | -2.260, +2.078, -1.405 | -0.529 ± 1.876 | 1/3 |
| Equal-weight dataset means | {"Grocery": 0.24402141571044922, "Movies": 0.3199338912963867, "ele-fashion": -0.16364852587382} | +0.133 ± 0.212 | descriptive | {"Grocery": -0.5434341296805645, "Movies": 0.2640393531254226, "ele-fashion": -0.5289371495688384} | -0.269 ± 0.377 | descriptive |

### B1 minus B2

| Dataset | Accuracy Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds | Macro-F1 Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds |
|---|---|---:|---:|---|---:|---:|
| Movies | -0.060, +0.630, -0.510 | +0.020 ± 0.469 | 1/3 | +0.340, +1.389, +0.366 | +0.698 ± 0.488 | 3/3 |
| Grocery | +0.205, -0.117, +0.000 | +0.029 ± 0.133 | 1/3 | +1.162, -1.021, +1.121 | +0.421 ± 1.019 | 2/3 |
| ele-fashion | +0.644, -0.041, +0.276 | +0.293 ± 0.280 | 2/3 | +0.912, -1.219, +0.876 | +0.190 ± 0.996 | 2/3 |
| Equal-weight dataset means | {"Grocery": 0.029283761978149414, "Movies": 0.019997358322143555, "ele-fashion": 0.2932031949361166} | +0.114 ± 0.127 | descriptive | {"Grocery": 0.42059487903174747, "Movies": 0.6979956786716441, "ele-fashion": 0.1898361099165709} | +0.436 ± 0.208 | descriptive |

### B0 minus B1

| Dataset | Accuracy Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds | Macro-F1 Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds |
|---|---|---:|---:|---|---:|---:|
| Movies | +0.030, -0.390, +0.210 | -0.050 ± 0.251 | 2/3 | +0.599, -0.200, +0.595 | +0.331 ± 0.376 | 2/3 |
| Grocery | -0.029, +0.000, +0.176 | +0.049 ± 0.091 | 1/3 | -0.400, +0.970, +1.745 | +0.771 ± 0.887 | 2/3 |
| ele-fashion | -0.286, -0.123, +0.194 | -0.072 ± 0.200 | 1/3 | -0.268, -0.854, +0.065 | -0.352 ± 0.380 | 1/3 |
| Equal-weight dataset means | {"Grocery": 0.048804283142089844, "Movies": -0.04999041557312012, "ele-fashion": -0.07159709930419922} | -0.024 ± 0.052 | descriptive | {"Grocery": 0.7714317916585389, "Movies": 0.33139623994722, "ele-fashion": -0.35200922470403073} | +0.250 ± 0.462 | descriptive |

### B3 minus B0

| Dataset | Accuracy Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds | Macro-F1 Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds |
|---|---|---:|---:|---|---:|---:|
| Movies | +0.210, +0.000, +0.240 | +0.150 ± 0.107 | 2/3 | +0.144, -0.648, -0.563 | -0.356 ± 0.355 | 1/3 |
| Grocery | +0.117, +0.264, -0.146 | +0.078 ± 0.170 | 2/3 | -0.539, +0.000, -2.151 | -0.897 ± 0.914 | 1/3 |
| ele-fashion | +0.153, +0.256, -0.245 | +0.055 ± 0.216 | 2/3 | +0.533, +1.810, -0.512 | +0.610 ± 0.949 | 2/3 |
| Equal-weight dataset means | {"Grocery": 0.07808605829874674, "Movies": 0.14996925989786786, "ele-fashion": 0.054552157719930015} | +0.094 ± 0.041 | descriptive | {"Grocery": -0.8966329678861203, "Movies": -0.3556799801574232, "ele-fashion": 0.6103851278045136} | -0.214 ± 0.623 | descriptive |

### B3 minus B1

| Dataset | Accuracy Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds | Macro-F1 Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds |
|---|---|---:|---:|---|---:|---:|
| Movies | +0.240, -0.390, +0.450 | +0.100 ± 0.357 | 2/3 | +0.743, -0.848, +0.033 | -0.024 ± 0.651 | 2/3 |
| Grocery | +0.088, +0.264, +0.029 | +0.127 ± 0.100 | 3/3 | -0.939, +0.970, -0.406 | -0.125 ± 0.805 | 1/3 |
| ele-fashion | -0.133, +0.133, -0.051 | -0.017 ± 0.111 | 1/3 | +0.265, +0.956, -0.446 | +0.258 ± 0.572 | 2/3 |
| Equal-weight dataset means | {"Grocery": 0.1268903414408366, "Movies": 0.09997884432474773, "ele-fashion": -0.017044941584269203} | +0.070 ± 0.062 | descriptive | {"Grocery": -0.12520117622758137, "Movies": -0.024283740210203263, "ele-fashion": 0.25837590310048286} | +0.036 ± 0.162 | descriptive |

### B4 minus B3

| Dataset | Accuracy Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds | Macro-F1 Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds |
|---|---|---:|---:|---|---:|---:|
| Movies | -0.120, -0.150, -0.090 | -0.120 ± 0.024 | 0/3 | -1.204, +1.674, +0.229 | +0.233 ± 1.175 | 2/3 |
| Grocery | +0.088, -0.264, +0.088 | -0.029 ± 0.166 | 2/3 | +0.055, -0.674, +1.240 | +0.207 ± 0.789 | 2/3 |
| ele-fashion | -0.256, -0.184, -0.092 | -0.177 ± 0.067 | 0/3 | +0.253, -1.174, -0.301 | -0.407 ± 0.587 | 1/3 |
| Equal-weight dataset means | {"Grocery": -0.029281775156656902, "Movies": -0.11997620264689128, "ele-fashion": -0.17728606859842938} | -0.109 ± 0.061 | descriptive | {"Grocery": 0.20701973867052192, "Movies": 0.2332538015843396, "ele-fashion": -0.407432830752356} | +0.011 ± 0.296 | descriptive |

### B4 minus B0

| Dataset | Accuracy Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds | Macro-F1 Δ (pp), seeds 42/43/44 | Mean ± SD (pp) | Positive seeds |
|---|---|---:|---:|---|---:|---:|
| Movies | +0.090, -0.150, +0.150 | +0.030 ± 0.130 | 2/3 | -1.060, +1.026, -0.334 | -0.122 ± 0.865 | 1/3 |
| Grocery | +0.205, +0.000, -0.059 | +0.049 ± 0.113 | 1/3 | -0.484, -0.674, -0.911 | -0.690 ± 0.175 | 0/3 |
| ele-fashion | -0.102, +0.072, -0.338 | -0.123 ± 0.168 | 1/3 | +0.786, +0.635, -0.813 | +0.203 ± 0.721 | 2/3 |
| Equal-weight dataset means | {"Grocery": 0.048804283142089844, "Movies": 0.029993057250976562, "ele-fashion": -0.12273391087849934} | -0.015 ± 0.077 | descriptive | {"Grocery": -0.6896132292155983, "Movies": -0.12242617857308365, "ele-fashion": 0.20295229705215764} | -0.203 ± 0.369 | descriptive |

## Original V7B B0 bug and corrective findings

In `src/models/psce_mag_v7b.py`, the constructor previously froze `relation_projections` and `relation_scorer` for every non-relative arm. That incorrectly included B0, whose definition is the complete V7A A2 with a trainable shared Sigmoid scorer. The fix limits freezing to B1/B2. B3/B4 scorer construction is unchanged.
The model audit measured 1,551,682 trainable B0 parameters under the old permissions and 1,576,387 after correction, a +24,705 parameter change. The full Movies NC configuration has the same corrected count; formal run rows record model and classifier counts separately.
Real Movies train-mode parity passed with identical initialization, shared candidate graph, features, physical topology, train nodes, dropout RNG, CE + auxiliary objective, gradient clipping, AdamW, and one update. Maximum embedding error=1.19e-06; total-loss error=2.38e-07; common gradient error=1.19e-07; post-update error=1.24e-06.
The real-graph B0 relation gradients were finite and nonzero for every audited projection/scorer tensor. Their one-step L1 update sizes are recorded in `data/training_parity_audit.json`; all are positive. CPU initialization, dropout-matched losses/gradients, AdamW and three-step optimizer trajectories also match in the corrective pytest module.

| Dataset | Old buggy B0 edge-weight mean ± SD | Corrected B0 edge-weight mean ± SD |
|---|---:|---:|
| Movies | 0.80035 ± 0.00061 | 0.14865 ± 0.26857 |
| Grocery | 0.80018 ± 0.00053 | 0.69320 ± 0.22462 |
| ele-fashion | 0.80007 ± 0.00075 | 0.95091 ± 0.13214 |

The old V7B B0 used a random initialized but frozen relation scorer; it was not a trainable V7A A2. Therefore the old B0−B1, B3−B0, and B4−B0 comparisons do not answer the intended learning-versus-fixed questions. The old runs remain untouched and are not included in this campaign. New results only support the corrected comparisons in this report.

## Candidate, random topology, and training audits

The cache audit passed for all three datasets. Feature fingerprints were recomputed from source feature files; KNN edge index, unique edges, and degree exactly matched V7A; V7B source flags and fingerprints matched; label access during candidate construction/audit was false. All nine B2 caches were reused with the original training-seed mapping 42→2026, 43→2027, 44→2028, exact degree sequence, undirected unique edges, and no self-loops or duplicates. Physical overlap details are in `data/candidate_cache_audit.json` and `data/random_graph_audit.json`.
Full repository pytest: True (exit 0, 27.1s). GPU preflight: True (12 dataset/variant paths). Six-group Movies smoke: True.
Target was physical GPU1, RTX 3090. The runner used `CUDA_VISIBLE_DEVICES=1` (logical CUDA device 0); GPU0 was left untouched. Preflight reports one-step time, relation-scoring and semantic-propagation timings, full validation inference time, peak allocated/reserved VRAM and trainable parameter count.

## Relation, gate, expert, and scale diagnostics

`data/semantic_diagnostics.csv` contains per dataset/seed/checkpoint edge-weight means, SDs, quantiles, saturation fractions, Text/Visual correlation and absolute differences, source-group distributions, relation projection/scorer gradients, physical/semantic expert selection shares, semantic strength gates, semantic contribution RMS relative to physical and intrinsic pathways, calibration scales, semantic/Raw cosine, and closed-semantic-path validation changes. B1/B2 fixed edge weights are exactly 0.8; their semantic gates and topology controls are included. These diagnostics describe trained behavior and are not evidence that an individual edge is causally useful.

## Findings against the research questions

The compact values below are equal-weight averages of the three dataset means. Seed-pair counts are descriptive totals out of nine; they are not significance tests.

| Paired comparison | Accuracy Δ (pp) | Macro-F1 Δ (pp) | Positive Accuracy seed pairs | Positive Macro-F1 seed pairs |
|---|---:|---:|---:|---:|
| B1 − R0 | +0.248 | +0.167 | 8/9 | 5/9 |
| B2 − R0 | +0.133 | -0.269 | 6/9 | 3/9 |
| B1 − B2 | +0.114 | +0.436 | 4/9 | 7/9 |
| B0 − B1 | -0.024 | +0.250 | 4/9 | 5/9 |
| B3 − B0 | +0.094 | -0.214 | 6/9 | 4/9 |
| B3 − B1 | +0.070 | +0.036 | 6/9 | 5/9 |
| B4 − B3 | -0.109 | +0.011 | 2/9 | 5/9 |
| B4 − B0 | -0.015 | -0.203 | 4/9 | 3/9 |

**A. Fixed KNN.** B1 exceeded Raw in mean validation accuracy on all three datasets and in 8/9 seed pairs, with an equal-weight gain of +0.248 pp. Its Macro-F1 direction varied by dataset. B1 over the degree-preserving random B2 was small (+0.114 pp accuracy) and inconsistent across seeds (4/9 positive accuracy pairs; 7/9 for Macro-F1). Keep B1 as the simple semantic control; the evidence for a strong topology-specific gain over B2 is limited.

**B. V7A learned edge weights.** Corrected B0 did not improve mean accuracy over fixed KNN (B0−B1: -0.024 pp; 4/9 positive accuracy pairs), although Macro-F1 was +0.250 pp descriptively. This does not establish a robust accuracy gain from learned weighting. Correct training produced markedly dataset-dependent edge distributions: Movie weights contracted toward zero, while ele-fashion weights concentrated near the upper Sigmoid boundary. These are behavior diagnostics, not evidence of task utility.

**C. Relative scoring.** B3 gained only +0.094 pp accuracy over corrected B0 and +0.070 pp over B1 on equal-weight dataset means. The seed directions and Macro-F1 directions were mixed. The campaign does not show a clear incremental benefit from the relative score mechanism.

**D. Modality-specific relations.** B4 had -0.109 pp mean accuracy versus shared B3 (positive in only 2/9 seed pairs); Macro-F1 was essentially unchanged at +0.011 pp. B4 learned different Text/Visual weights, but that separation did not yield a consistent validation advantage.

**E. Next-step choice.** The simplest useful semantic arm is fixed KNN B1: it gave a small, recurring accuracy improvement over Raw, while learned B0 and the more complex B3/B4 did not show clear accuracy gains over B1. Retain corrected B0 as the valid learnable-weight reference. Do not treat B4 as the default main model or start mechanism/parameter searches from these small, inconsistent effects.

## Selected relation and gate diagnostics

Values are means over the three seeds within each dataset. B0 near-bound fractions use its Sigmoid interval [0, 1] (near zero ≤0.005; near one ≥0.995). B3 fractions use the relative-weight interval [0.2, 1.4] (near lower ≤0.205; near upper ≥1.395). Per-seed quantiles, source groups, expert usage, gate distributions, contribution scales, gradients, and closed-path results are in `data/semantic_diagnostics.csv`.

| Dataset | Corrected B0 weight mean ± SD | B0 near 0 / near 1 | B3 weight mean | B3 near lower / near upper | B4 Text/Visual correlation | B4 mean absolute Text−Visual difference |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0.149 ± 0.269 | 36.4% / 2.0% | 0.805 | 9.4% / 8.7% | 0.140 | 0.458 |
| Grocery | 0.693 ± 0.225 | 1.7% / 3.3% | 0.793 | 17.1% / 16.9% | 0.913 | 0.179 |
| ele-fashion | 0.951 ± 0.132 | 0.3% / 84.0% | 0.828 | 20.2% / 21.7% | 0.064 | 0.505 |

The corrected B0 result directly answers the historical concern: Movies shows strong edge shrinkage, while 84% of ele-fashion edges are within 0.005 of the upper Sigmoid boundary. B3 also places roughly 9–22% of edges near each relative-weight bound, depending on dataset. B4 yields the largest modality separation on Movies and ele-fashion, yet no consistent accuracy gain.

| Dataset | B2 Text / Visual semantic gate means | B0 closed-path Δ Val Accuracy | B3 closed-path Δ Val Accuracy | B4 closed-path Δ Val Accuracy |
|---|---:|---:|---:|---:|
| Movies | 0.010 / 0.022 | -0.16 pp | -0.57 pp | -0.63 pp |
| Grocery | 0.005 / 0.011 | -0.48 pp | -0.80 pp | -0.69 pp |
| ele-fashion | 0.121 / 0.060 | -2.26 pp | -4.45 pp | -2.19 pp |

B2's semantic gate was near zero on Movies and Grocery, consistent with the model reducing reliance on those randomized structures; it was higher on ele-fashion but remained below its fixed-KNN control. Closing the semantic path lowered validation accuracy for B0/B3/B4 in all three dataset averages. These checkpoint ablations quantify reliance on the pathway, not causal utility of individual edges.

All eight detailed comparisons remain paired by dataset and seed in the preceding section. Small and inconsistent differences do not justify more mechanisms or a hyperparameter search in this correction study. No significance inference treats the nine dataset/seed pairs as independent datasets.

## Artifacts and limits

- Per-run raw rows and provenance: `data/run_rows.json` and `data/campaign_manifest.json`
- Per-dataset summaries: `data/summary.csv`
- Seed-paired deltas and equal-weight descriptive summaries: `data/paired_comparisons.csv`
- Relation/gate/expert diagnostics: `data/semantic_diagnostics.csv`
- Training parity: `data/training_parity_audit.json` and `data/parameter_grad_audit.json`
- Candidate and random graph audit: `data/candidate_cache_audit.json`, `data/random_graph_audit.json`
- GPU preflight/resource records: `data/preflight_summary.json`, `data/resource_profile.json`, `data/resource_profile.csv`
- Test and smoke records: `data/test_summary.json`, `data/smoke_summary.json`

This was full-graph node classification only. It did not perform formal link prediction, Test evaluation, split regeneration, or hyperparameter search. The conclusions cover these three datasets and three training seeds only.
