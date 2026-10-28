# uci_mi_lethal

- Observations: 1700
- Predictors: 111
- Outcome: lethal
- Class counts: {'Non-lethal course': 1429, 'Lethal outcome': 271}
- Note: UCI Myocardial infarction complications; predictors = admission features only; target lethal if LET_IS ≠ 0; other complication labels excluded.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications
- Changes: all 111 predictors glossed from the UCI variable table (angina history codes, infarction QRS types, dose counts, fibrinolytics); kfk_blood numeric IU/L; time_b_s categorical; classes 'Survived/Died during the hospital stay'.
- Validation: `jevtest check` passes; zero-shot prompt about 2033 tokens; ICL subsample of about 79 rows at ~391.6 tokens per row.
