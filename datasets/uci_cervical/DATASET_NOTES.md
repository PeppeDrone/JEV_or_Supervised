# uci_cervical

- Observations: 858
- Predictors: 29
- Outcome: biopsy
- Class counts: {'Negative biopsy': 803, 'Positive biopsy': 55}
- Note: UCI Cervical Cancer Risk Factors. Outcome = biopsy. Dx* diagnosis columns and near-empty STD timing fields excluded. Missing values preserved as blanks.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
- Changes: labels and setting rewritten; screening tests named (Hinselmann, Schiller, cytology).
- Validation: `jevtest check` passes; zero-shot prompt about 436 tokens; ICL subsample of about 383 rows at ~82.3 tokens per row.
