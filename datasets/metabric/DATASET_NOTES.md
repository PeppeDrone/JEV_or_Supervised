# metabric

- Observations: 1000
- Predictors: 13
- Outcome: pam50_subtype
- Class counts: {'LumA': 352, 'LumB': 239, 'claudin-low': 141, 'Her2': 109, 'Basal': 86, 'Normal': 73}
- Excluded: ['OS_MONTHS', 'OS_STATUS', 'RFS_MONTHS', 'RFS_STATUS', 'VITAL_STATUS', 'INTCLUST', 'CHEMOTHERAPY', 'HORMONE_THERAPY', 'RADIO_THERAPY', 'NPI', 'TMB_NONSYNONYMOUS', 'THREEGENE']
- Note: cBioPortal brca_metabric clinical TSV. Y = PAM50+claudin-low (NC dropped). Survival, INTCLUST, NPI, TMB, and post-diagnosis systemic/radiotherapy dropped. First 1000 PATIENT_ID with a valid subtype.

## Codebook review (2026-10-24)

- Source documentation: cBioPortal brca_metabric clinical table (Curtis et al., Nature 2012; Pereira et al., Nat Commun 2016)
- Changes: histological subtype, surgery, stage, grade, menopausal state glossed; PAM50 class names expanded (Luminal A/B, HER2-enriched, Basal-like, Normal-like, Claudin-low); setting names cohort and period.
- Validation: `jevtest check` passes; zero-shot prompt about 323 tokens; ICL subsample of about 515 rows at ~61.2 tokens per row.
