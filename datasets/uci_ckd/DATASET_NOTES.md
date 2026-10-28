# uci_ckd

- Observations: 400
- Predictors: 24
- Outcome: ckd_status
- Class counts: {'Chronic kidney disease': 250, 'No chronic kidney disease': 150}
- Identifier note: The UCI CKD release does not provide stable patient identifiers; record_id enumerates source rows. Unique patients cannot be independently verified.

Raw sources and SHA256 checksums are recorded in PREPARATION.json.
No imputation, scaling or outcome-guided feature selection was applied.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- Changes: pcv unit (%), bgr label, oedema/anaemia labels; setting names the Tamil Nadu 2015 cohort.
- Validation: `jevtest check` passes; zero-shot prompt about 406 tokens; ICL subsample of about 284 rows at ~111.0 tokens per row.
