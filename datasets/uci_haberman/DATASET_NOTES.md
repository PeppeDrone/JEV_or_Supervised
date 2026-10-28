# uci_haberman

- Observations: 306
- Predictors: 3
- Outcome: died_5y
- Class counts: {'Survived ≥5 years': 225, 'Died within 5 years': 81}
- Note: UCI Haberman survival; status 1=survived 5y, 2=died within 5y.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/43/haberman+s+survival
- Changes: operation year scale note (two-digit, 1958-1969); classes state the five-year horizon from surgery.
- Validation: `jevtest check` passes; zero-shot prompt about 150 tokens; full training fold fits the 32k ICL budget.
