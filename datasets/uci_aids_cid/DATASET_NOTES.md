# uci_aids_cid

- Observations: 2139
- Predictors: 19
- Outcome: aids_event
- Class counts: {'No AIDS-defining event': 1618, 'AIDS-defining event': 521}
- Note: UCI AIDS Clinical Trials Group Study 175; cid endpoint; time excluded.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175
- Changes: race (0 White / 1 Non-white) and gender (0 Female / 1 Male) corrected to the UCI documentation; Karnofsky and treatment arm glossed; composite endpoint spelled out in classes/question.
- Validation: `jevtest check` passes; zero-shot prompt about 461 tokens; ICL subsample of about 275 rows at ~114.2 tokens per row.
