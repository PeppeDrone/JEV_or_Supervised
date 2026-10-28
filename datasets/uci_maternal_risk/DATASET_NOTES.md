# uci_maternal_risk

- Observations: 1014
- Predictors: 6
- Outcome: risk_level
- Class counts: {'Low risk': 406, 'Mid risk': 336, 'High risk': 272}
- Note: UCI Maternal Health Risk; vital signs → risk level.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/863/maternal+health+risk
- Changes: heart_rate text_range 30-220 (values of 7 become out of range); blood glucose label; classes 'Medium risk'; setting names the rural Bangladesh cohort.
- Validation: `jevtest check` passes; zero-shot prompt about 172 tokens; ICL subsample of about 758 rows at ~41.7 tokens per row.
