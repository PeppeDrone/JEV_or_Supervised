# uci_thyroid_recur

- Observations: 383
- Predictors: 16
- Outcome: recurred
- Class counts: {'No recurrence': 275, 'Recurred': 108}
- Note: UCI Differentiated Thyroid Cancer Recurrence.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence
- Changes: TNM (AJCC 8th) and N/M categories glossed; pathology, focality, adenopathy and response categories normalised; follow-up horizon in question.
- Validation: `jevtest check` passes; zero-shot prompt about 320 tokens; full training fold fits the 32k ICL budget.
