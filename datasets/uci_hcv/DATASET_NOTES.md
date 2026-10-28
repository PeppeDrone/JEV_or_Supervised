# uci_hcv

- Observations: 608
- Predictors: 12
- Outcome: hcv_category
- Class counts: {'Blood donor': 533, 'Hepatitis': 24, 'Fibrosis': 21, 'Cirrhosis': 30}
- Identifier note: Source ID values from the UCI HCV release are retained as record_id prefixes; the seven suspect blood-donor rows were excluded because that class has fewer than ten observations.

Raw sources and SHA256 checksums are recorded in PREPARATION.json.
No imputation, scaling or outcome-guided feature selection was applied.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/571/hcv+data
- Changes: serum units added (albumin g/L, enzymes U/L, bilirubin and creatinine µmol/L, cholesterol mmol/L, cholinesterase kU/L); category classes expanded.
- Validation: `jevtest check` passes; zero-shot prompt about 277 tokens; ICL subsample of about 351 rows at ~89.7 tokens per row.
