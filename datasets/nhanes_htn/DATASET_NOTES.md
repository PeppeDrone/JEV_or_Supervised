# nhanes_htn

- Observations: 1000
- Predictors: 12
- Outcome: hypertension
- Class counts: {'No hypertension': 459, 'Hypertension': 541}
- Excluded: ['BPXOSY1', 'BPXOSY2', 'BPXOSY3', 'BPXODI1', 'BPXODI2', 'BPXODI3', 'BPQ020', 'BPQ030', 'BPQ150', 'BPQ040A', 'BPQ050A']
- Note: NHANES 2021–2023 adults 18+ with oscillometric BP. Y = mean SBP>=130 or mean DBP>=80 or currently taking BP medication (NCHS Data Brief 511). BP values and BP-medication items are used only to build Y. First 1000 SEQN after eligibility.

## Codebook review (2026-10-24)

- Source documentation: NHANES 2021-2023 DEMO_L, BMX_L, SMQ_L, DIQ_L, BPXO_L, BPQ_L documentation; NCHS Data Brief 511
- Changes: poverty_ratio scale note (0-5, top-coded); education glossed; outcome classes state the 130/80 mmHg or medication definition; setting rewritten.
- Validation: `jevtest check` passes; zero-shot prompt about 286 tokens; ICL subsample of about 396 rows at ~79.6 tokens per row.
