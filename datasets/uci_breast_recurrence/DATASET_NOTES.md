# uci_breast_recurrence

- Observations: 286
- Predictors: 9
- Outcome: recurrence
- Class counts: {'No recurrence': 201, 'Recurrence': 85}
- Note: UCI Breast Cancer (Ljubljana); recurrence-events endpoint.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/14/breast+cancer
- Changes: Excel-corrupted tumour-size and node-count codes decoded ('9-May' -> '5-9 mm', etc.); menopause and degree glossed; setting names the Ljubljana cohort.
- Validation: `jevtest check` passes; zero-shot prompt about 235 tokens; full training fold fits the 32k ICL budget.
