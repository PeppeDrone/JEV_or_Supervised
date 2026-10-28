# wids2021

- Observations: 1000
- Predictors: 32
- Outcome: diabetes_mellitus
- Classes: {'No diabetes mellitus': 784, 'Diabetes mellitus': 216}
- WiDS 2021 TrainingWiDS2021.csv; stratified sample of 1000 encounters (seed 20261008) from 130157 labelled encounters. Excluded: identifiers, hospital/ICU ids, APACHE diagnosis codes, all glucose labs (near-label for existing diabetes), and unused labs/vitals outside the fixed KEEP list. Target is existing diabetes status, not incidence.

## Codebook review (2026-10-24)

- Source documentation: WiDS Datathon 2021 data dictionary (GOSSIS), https://www.kaggle.com/c/widsdatathon2021
- Changes: GCS component scale notes (eye 1-4, motor 1-6, verbal 1-5); APACHE labels; ethnicity gloss; setting states withheld glucose/HbA1c.
- Validation: `jevtest check` passes; zero-shot prompt about 583 tokens; ICL subsample of about 204 rows at ~154.1 tokens per row.
