# berlin_appendicitis

- Observations: 590
- Predictors: 12
- Outcome: appendicitis
- Class counts: {'No appendicitis': 117, 'Appendicitis': 473}
- Excluded: ['Gangrän', 'Phlegmone', 'mm_App']
- Note: OSF 9wvys / Reismann et al. PLOS ONE 2019. Discovery+validation sheets concatenated (paper n=590). Y = histopathology appendicitis vs negative. Not UCI 938 Regensburg.

## Codebook review (2026-10-24)

- Source documentation: OSF 9wvys; Reismann et al., PLOS ONE 2019 (https://doi.org/10.1371/journal.pone.0222030)
- Changes: setting names the population and care point; outcome classes state histopathological confirmation.
- Validation: `jevtest check` passes; zero-shot prompt about 295 tokens; ICL subsample of about 261 rows at ~120.4 tokens per row.
