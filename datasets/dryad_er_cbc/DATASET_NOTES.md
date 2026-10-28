# dryad_er_cbc

- Observations: 1000
- Predictors: 16
- Outcome: er_death
- Class counts: {'Alive': 859, 'Died': 141}
- Excluded: ['ddFUp', 'deadTRUE', 'alive0_dead1', 'Dischrd0_Inward1', 'Hb_tailr', 'RDW_tailr', 'RBC_tailr', 'MCV_tailr', 'WBC_tailr', 'Neu_tailr', 'Lym_tailr', 'Mon_tailr', 'Eos_tailr', 'Bas_tailr', 'PLT_tailr', 'PCT_tailr', 'PDW_tailr', 'Tailr_score_sum', 'Tailr_BCDC_score', 'Tailr_AgeSex_InW_Wave', 'HB_bench', 'RDW_bench', 'RBC_bench', 'MCV_bench', 'WBC_bench', 'Neu_bench', 'Lym_bench', 'Mon_bench', 'Eos_bench', 'Bas_bench', 'PLT_bench', 'PCT_bench', 'PDW_bench', 'Bench_score_sum', 'Bench_BCDC_score', 'Bench_AgeSex_InW_Wave', 'Dischrd0_Inward1']
- Note: Dryad 10.5061/dryad.dncjsxm5g. First 1000 Num. Y = all-cause death. Follow-up days, inward admission, and tailored/benchmark score columns dropped.

## Codebook review (2026-10-24)

- Source documentation: Dryad https://doi.org/10.5061/dryad.dncjsxm5g; BMJ Open 2023;13:e071937
- Changes: constant covid_wave glossed ('Outside the COVID-19 infection waves'); follow-up horizon (up to about 17 months, from ddFUp) added to classes and question; setting rewritten.
- Validation: `jevtest check` passes; zero-shot prompt about 350 tokens; ICL subsample of about 199 rows at ~157.7 tokens per row.
