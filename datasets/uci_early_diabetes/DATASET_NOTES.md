# uci_early_diabetes

- Observations: 520
- Predictors: 16
- Outcome: diabetes
- Class counts: {'No diabetes': 200, 'Diabetes': 320}
- Note: UCI Early stage diabetes risk prediction (symptom questionnaire).

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
- Changes: symptom labels normalised with plain-language explanations; setting names Sylhet Diabetes Hospital.
- Validation: `jevtest check` passes; zero-shot prompt about 198 tokens; full training fold fits the 32k ICL budget.
