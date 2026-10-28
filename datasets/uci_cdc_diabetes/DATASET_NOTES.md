# uci_cdc_diabetes

- Observations: 1000
- Predictors: 21
- Outcome: cdc_diabetes
- Class counts: {'No diabetes': 861, 'Diabetes': 139}
- Note: UCI CDC Diabetes Health Indicators; stratified subsample n=1000.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Changes: age recoded as the 13 BRFSS age groups; general health, education and income glossed; setting names the 2015 BRFSS survey.
- Validation: `jevtest check` passes; zero-shot prompt about 403 tokens; ICL subsample of about 370 rows at ~85.2 tokens per row.
