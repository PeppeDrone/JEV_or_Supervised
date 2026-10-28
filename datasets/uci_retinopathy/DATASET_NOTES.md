# uci_retinopathy

- Observations: 1151
- Predictors: 19
- Outcome: retinopathy
- Class counts: {'No retinopathy': 540, 'Retinopathy': 611}
- Note: UCI Diabetic Retinopathy Debrecen; image-derived features.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen
- Changes: microaneurysm detector confidence levels and exudate measures glossed; counts as scale notes.
- Validation: `jevtest check` passes; zero-shot prompt about 663 tokens; ICL subsample of about 237 rows at ~132.4 tokens per row.
