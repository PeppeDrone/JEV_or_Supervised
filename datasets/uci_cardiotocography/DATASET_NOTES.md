# uci_cardiotocography

- Observations: 2126
- Predictors: 21
- Outcome: nsp
- Class counts: {'Normal': 1655, 'Suspect': 295, 'Pathologic': 176}
- Note: UCI Cardiotocography; NSP endpoint; CLASS morphologic code excluded.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/193/cardiotocography
- Changes: tendency recoded as categorical (left-asymmetric/symmetric/right-asymmetric); event-rate units; question no longer carries the NSP code.
- Validation: `jevtest check` passes; zero-shot prompt about 424 tokens; ICL subsample of about 241 rows at ~130.3 tokens per row.
