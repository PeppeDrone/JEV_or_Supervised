# uci_breast_coimbra

- Observations: 116
- Predictors: 9
- Outcome: breast_cancer
- Class counts: {'Healthy control': 52, 'Breast cancer': 64}
- Note: UCI Breast Cancer Coimbra (blood biomarkers).

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra
- Changes: HOMA-IR, MCP-1 and fasting labels clarified; setting names the Coimbra cohort.
- Validation: `jevtest check` passes; zero-shot prompt about 255 tokens; full training fold fits the 32k ICL budget.
