# uci_bone_marrow

- Observations: 187
- Predictors: 35
- Outcome: bmt_died
- Class counts: {'Alive': 102, 'Dead': 85}
- Note: UCI Bone marrow transplant in children; survival_time excluded.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/565/bone+marrow+transplant+children
- Changes: recipient gender, gender match, ABO, HLA antigen/allele mismatch counts, aGvHD/extensive cGvHD codes (0 = yes, 1 = no) and CMV score corrected to the UCI documentation; 1000000 sentinel glossed as 'never reached'.
- Validation: `jevtest check` passes; zero-shot prompt about 633 tokens; full training fold fits the 32k ICL budget.
