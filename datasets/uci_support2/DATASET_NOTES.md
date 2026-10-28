# uci_support2

- Observations: 1000
- Predictors: 37
- Outcome: hosp_death
- Class counts: {'Discharged alive': 741, 'Died in hospital': 259}
- Note: UCI SUPPORT2; stratified subsample n=1000; survival times/costs excluded.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/880/support2
- Changes: disease groups, DNR orders, functional status and SUPPORT/APACHE scores glossed; day-3 record stated in setting.
- Validation: `jevtest check` passes; zero-shot prompt about 760 tokens; ICL subsample of about 134 rows at ~233.2 tokens per row.
