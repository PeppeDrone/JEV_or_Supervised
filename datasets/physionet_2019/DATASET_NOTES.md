# physionet_2019

- Observations: 1000
- Predictors: 38
- Outcome: sepsis
- Class counts: {'No sepsis': 910, 'Sepsis': 90}
- Excluded: ['ICULOS', 'SepsisLabel', 'post-onset hourly rows']
- Note: PhysioNet Challenge 2019 training_setA. First 1000 patient files. Features frozen at ICULOS<=6 hours and before SepsisLabel=1. Y = ever sepsis during the stay.

## Codebook review (2026-10-24)

- Source documentation: PhysioNet/CinC Challenge 2019 training set A, https://physionet.org/content/challenge-2019/
- Changes: icu_unit SICU/MICU glossed; pH scale note and text_range 6.5-8.0; sepsis classes/question name Sepsis-3 and the ICU-stay horizon.
- Validation: `jevtest check` passes; zero-shot prompt about 659 tokens; ICL subsample of about 190 rows at ~164.8 tokens per row.
