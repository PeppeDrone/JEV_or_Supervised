# uci_diabetes130

- Observations: 1000
- Predictors: 18
- Outcome: readmit_30d
- Class counts: {'No 30-day readmission': 910, '30-day readmission': 90}
- Note: UCI Diabetes 130-US Hospitals. First encounter per patient; death/hospice discharge dispositions [11, 13, 14, 19, 20, 21] removed; stratified sample of 1000 patients (seed 20261008) for LLM-tractable evaluation. Identifiers, weight (~95% missing), diagnosis codes, and discharge disposition excluded.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Changes: admission/discharge codes glossed; HbA1c and glucose result codes; question states 30-day readmission; setting names the 130-hospital 1999-2008 cohort.
- Validation: `jevtest check` passes; zero-shot prompt about 347 tokens; ICL subsample of about 436 rows at ~72.4 tokens per row.
