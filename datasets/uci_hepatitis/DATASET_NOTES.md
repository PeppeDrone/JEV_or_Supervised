# uci_hepatitis

- Observations: 155
- Predictors: 19
- Outcome: died
- Class counts: {'Lived': 123, 'Died': 32}
- Note: UCI Hepatitis; Class 1=die, 2=live.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/46/hepatitis
- Changes: labels and setting rewritten; follow-up horizon in classes.
- Validation: `jevtest check` passes; zero-shot prompt about 282 tokens; full training fold fits the 32k ICL budget.
