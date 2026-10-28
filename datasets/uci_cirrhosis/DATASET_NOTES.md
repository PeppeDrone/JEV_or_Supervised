# uci_cirrhosis

- Observations: 418
- Predictors: 17
- Outcome: cirrhosis_died
- Class counts: {'Alive / censored': 257, 'Died': 161}
- Note: UCI Cirrhosis; Status D=death; N_Days excluded.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1
- Changes: 'NaNN' declared as missing code on eight predictors; stage recoded as categorical with histological stage glosses; platelets unit; follow-up period in question.
- Validation: `jevtest check` passes; zero-shot prompt about 365 tokens; ICL subsample of about 268 rows at ~117.2 tokens per row.
