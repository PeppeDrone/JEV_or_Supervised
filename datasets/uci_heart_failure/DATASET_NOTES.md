# uci_heart_failure

- Observations: 299
- Predictors: 11
- Outcome: mortality
- Class counts: {'Survived follow-up': 203, 'Died during follow-up': 96}
- Note: UCI Heart Failure Clinical Records (Chicco & Jurman). Follow-up `time` excluded as a predictor to avoid leakage with mortality.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
- Changes: setting rewritten without 'mortality'; follow-up horizon (one to ten months) in question; platelets unit per µL; CPK label.
- Validation: `jevtest check` passes; zero-shot prompt about 236 tokens; full training fold fits the 32k ICL budget.
