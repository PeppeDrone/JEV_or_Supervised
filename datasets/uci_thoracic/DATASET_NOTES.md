# uci_thoracic

- Observations: 470
- Predictors: 16
- Outcome: died_1y
- Class counts: {'Alive at 1 year': 400, 'Died within 1 year': 70}
- Note: UCI Thoracic Surgery Data; 1-year mortality.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data
- Changes: DGN groups, Zubrod performance status, T-size codes and spirometry units glossed from the UCI documentation.
- Validation: `jevtest check` passes; zero-shot prompt about 329 tokens; full training fold fits the 32k ICL budget.
