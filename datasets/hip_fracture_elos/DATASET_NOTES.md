# hip_fracture_elos

- Observations: 1137
- Predictors: 18
- Outcome: extended_los
- Class counts: {'Not extended stay': 637, 'Extended stay': 500}
- Excluded: ['age_ori', 'age_sup_old', 'anemia', 'bmi', 'cost', 'death', 'drgs', 'extra_fra', 'fixation', 'gender_female', 'general_anesth', 'height', 'insu_cost', 'intra_fra', 'los', 'malnourish', 'op_time', 'po_dvt', 'po_hct_d1', 'po_hlb_d1', 'po_pneu', 'po_urin', 'pre_anal', 'time_admission_surgery', 'transfusion', 'weight']
- Note: Zenodo 10.5281/zenodo.22866907 development cohort m2_deid.csv. Predictors = S_honest_T0_deduped. Postoperative labs, complications, costs, death, LOS, and time-to-surgery excluded.

## Codebook review (2026-10-24)

- Source documentation: Zenodo https://doi.org/10.5281/zenodo.22866907 (deposit README, PROTOCOL_PaperA.md: eLOS = LOS >= 14 days, age >= 50, 2019-2025)
- Changes: eLOS definition (>= 14 days) taken from the deposit protocol and placed in classes/question; pre_pt scale note; setting names the hospital, age >= 50 and period.
- Validation: `jevtest check` passes; zero-shot prompt about 382 tokens; ICL subsample of about 260 rows at ~121.0 tokens per row.
