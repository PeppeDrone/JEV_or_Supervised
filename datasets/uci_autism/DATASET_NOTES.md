# uci_autism

- Observations: 704
- Predictors: 18
- Outcome: asd
- Class counts: {'No ASD': 515, 'ASD': 189}
- Note: UCI Autism Screening Adult; AQ-10 items; result sum excluded.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/426/autism+screening+adult
- Changes: AQ-10 items written out in the scored (autistic-trait) direction; demographics glossed; age text_range 17-110 (one value of 383); classes state the >= 6-items screening rule.
- Validation: `jevtest check` passes; zero-shot prompt about 427 tokens; full training fold fits the 32k ICL budget.
