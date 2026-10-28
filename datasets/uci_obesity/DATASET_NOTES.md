# uci_obesity

- Observations: 2111
- Predictors: 16
- Outcome: obesity_class
- Class counts: {'Insufficient weight': 272, 'Normal weight': 287, 'Overweight level I': 290, 'Overweight level II': 290, 'Obesity type I': 351, 'Obesity type II': 297, 'Obesity type III': 324}
- Note: UCI Estimation of obesity levels based on eating habits and physical condition.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
- Changes: ordinal questionnaire scales (FCVC, NCP, CH2O, FAF, TUE) glossed; classes carry BMI ranges.
- Validation: `jevtest check` passes; zero-shot prompt about 458 tokens; ICL subsample of about 363 rows at ~86.5 tokens per row.
