# Hybrid-CGE V1.1 training-curve figure QA

## Figure contract

- Three dataset rows × three columns: Validation Accuracy, Validation Macro-F1, and Train/Validation CE.
- Curves show the mean and sample SD across three matched seeds. Every plotted epoch has `n=3`; each dataset is truncated to its common completed epoch horizon without interpolation or imputation.
- Source data: `../data/epoch_history.csv` and `../data/runs.csv`.
- Exports: PDF, SVG, PNG, and TIFF under `training_curves/`.

## Checks

- Strict source validation: **PASS**, 21 checks passed, 0 warnings, 0 failures.
- Strict Matplotlib panel alignment: **PASS**, nine panel comparisons, 0 warnings, 0 failures.
- PDF text-size audit: **PASS**, minimum found font size 5.8 pt against a 5 pt floor; 0 below-floor text runs among 111 runs.
- Rendered collision audit: **PASS**, 0 warnings and 0 failures. PyMuPDF 1.28.2 from the base Python 3.11 environment was used because it is not installed in `yhf_env`; the figure itself was rendered by Matplotlib in `yhf_env`.
- PNG and TIFF dimensions: 4320 × 4290 px at 600 dpi. TIFF uses LZW compression.

Machine-readable records are `qa/source_validation.json`, `qa/pdf_text_audit.json`, `qa/collision_audit.json`, and `training_curves/hybrid_cge_v11_training_curves.alignment.json`.
