# uci_mammographic_mass

- Observations: 961
- Predictors: 4
- Outcome: malignancy
- Class counts: {'Benign': 516, 'Malignant': 445}
- Identifier note: Each row is a mammographic mass lesion, not necessarily a unique patient. record_id enumerates source lesions; unique patients cannot be independently verified.

Raw sources and SHA256 checksums are recorded in PREPARATION.json.
No imputation, scaling or outcome-guided feature selection was applied.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/161/mammographic+mass
- Changes: density glossed (iso-dense explained); setting names the Erlangen cohort and withheld BI-RADS assessment; question states biopsy outcome.
- Validation: `jevtest check` passes; zero-shot prompt about 153 tokens; full training fold fits the 32k ICL budget.
