# uci_ilpd

- Observations: 583
- Predictors: 10
- Outcome: liver_disease
- Class counts: {'No liver disease': 167, 'Liver disease': 416}
- Note: UCI ILPD; Selector 1→liver disease, 2→no liver disease.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset
- Changes: age top-coding (90) noted; enzyme labels; classes reflect the expert 'liver-patient group' assignment.
- Validation: `jevtest check` passes; zero-shot prompt about 261 tokens; ICL subsample of about 450 rows at ~70.0 tokens per row.
