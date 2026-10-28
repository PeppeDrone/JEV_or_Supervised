# uci_glioma

- Observations: 839
- Predictors: 23
- Outcome: gbm
- Class counts: {'Lower-grade glioma': 487, 'Glioblastoma': 352}
- Note: UCI Glioma grading clinical and mutation features; Grade 0=LGG, 1=GBM.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset
- Changes: gene names expanded from the UCI documentation; race glossed; classes name WHO grades.
- Validation: `jevtest check` passes; zero-shot prompt about 464 tokens; ICL subsample of about 379 rows at ~83.3 tokens per row.
