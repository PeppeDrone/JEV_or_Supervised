# uci_thyroid

- Observations: 3163
- Predictors: 25
- Outcome: hypothyroid_status
- Class counts: {'No hypothyroidism': 3012, 'Hypothyroidism': 151}
- Identifier note: record_id enumerates hypothyroid.data rows; native '?' missingness is preserved as blank cells. Primary UCI Thyroid Disease hypothyroid task (binary).

Raw sources and SHA256 checksums are recorded in PREPARATION.json.
No imputation, scaling or outcome-guided feature selection was applied.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/102/thyroid+disease
- Changes: T4U ratio, FTI index and TBG labels; detection-limit zeros explained in setting; query flags relabelled.
- Validation: `jevtest check` passes; zero-shot prompt about 406 tokens; ICL subsample of about 401 rows at ~78.8 tokens per row.
