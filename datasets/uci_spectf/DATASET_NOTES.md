# uci_spectf

- Observations: 267
- Predictors: 44
- Outcome: abnormal
- Class counts: {'Normal SPECT': 55, 'Abnormal SPECT': 212}
- Note: UCI SPECTF Heart; diagnosis 0=normal, 1=abnormal.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/96/spectf+heart
- Changes: 22 ROIs listed as rest/stress pairs with the 0-100 count scale in the header.
- Validation: `jevtest check` passes; zero-shot prompt about 1097 tokens; ICL subsample of about 170 rows at ~185.1 tokens per row.
