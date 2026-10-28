# uci_wdbc

- Observations: 569
- Predictors: 30
- Outcome: diagnosis
- Class counts: {'Benign': 357, 'Malignant': 212}
- Identifier note: record_id preserves the original WDBC case ID; each row is one fine-needle aspirate image-derived feature vector.

Raw sources and SHA256 checksums are recorded in PREPARATION.json.
No imputation, scaling or outcome-guided feature selection was applied.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Changes: mean/SE/worst structure explained; pixel units in header.
- Validation: `jevtest check` passes; zero-shot prompt about 920 tokens; ICL subsample of about 123 rows at ~254.7 tokens per row.
