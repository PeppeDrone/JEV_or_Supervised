# physionet_2012

- Observations: 1000
- Predictors: 76
- Outcome: icu_death
- Class counts: {'Survived': 864, 'Died': 136}
- Excluded: ['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival']
- Note: PhysioNet Challenge 2012 set A. First 1000 RecordIDs. Y = In-hospital_death. Predictors = 48h last/mean of named labs and vitals plus admission descriptors. Missing encoded as -1 in the source and converted to empty. SAPS-I/SOFA/LOS/Survival dropped.

## Codebook review (2026-10-24)

- Source documentation: PhysioNet/CinC Challenge 2012 set A, https://physionet.org/content/challenge-2012/
- Changes: pH scale note and text_range 6.5-8.0 (two ph_mean outliers 51.2 and 20.5 become 'outside documented range'); mech_vent label; setting/header/question rewritten without outcome words.
- Validation: `jevtest check` passes; zero-shot prompt about 1526 tokens; ICL subsample of about 59 rows at ~524.7 tokens per row.
