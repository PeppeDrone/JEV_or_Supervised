# uci_parkinsons

- Observations: 195
- Predictors: 22
- Outcome: parkinsons
- Class counts: {'Healthy': 48, 'Parkinson disease': 147}
- Note: UCI Parkinson's; voice measures, status 1=PD.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/174/parkinsons
- Changes: MDVP/jitter/shimmer/nonlinear measures named and explained; setting states the sustained-phonation design.
- Validation: `jevtest check` passes; zero-shot prompt about 640 tokens; ICL subsample of about 151 rows at ~208.3 tokens per row.
