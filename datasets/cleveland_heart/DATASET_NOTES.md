# cleveland_heart

- Observations: 303
- Predictors: 13
- Outcome: heart_disease
- Class counts: {'No heart disease': 164, 'Heart disease': 139}
- Identifier note: Original patient identifiers were removed by the UCI donors and replaced with dummy values; record_id enumerates source rows and does not independently verify unique patients.

Raw sources and SHA256 checksums are recorded in PREPARATION.json.
No imputation, scaling or outcome-guided feature selection was applied.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/45/heart+disease
- Changes: glosses for chest-pain type, resting ECG, slope, thal and ca (0-3 vessels) checked against the UCI names file; units added (thalach beats/min, oldpeak mm); classes state the >=50% narrowing criterion.
- Validation: `jevtest check` passes; zero-shot prompt about 304 tokens; full training fold fits the 32k ICL budget.
