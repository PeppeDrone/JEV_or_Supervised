# uci_wbc_original

- Observations: 699
- Predictors: 9
- Outcome: malignant
- Class counts: {'Benign': 458, 'Malignant': 241}
- Note: UCI Breast Cancer Wisconsin (Original); Class 2=benign, 4=malignant.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
- Changes: 1-10 cytology scale stated in header; setting names the Wisconsin cohort.
- Validation: `jevtest check` passes; zero-shot prompt about 306 tokens; full training fold fits the 32k ICL budget.
