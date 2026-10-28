# uci_vertebral

- Observations: 310
- Predictors: 6
- Outcome: pathology
- Class counts: {'Normal': 100, 'Hernia': 60, 'Spondylolisthesis': 150}
- Note: UCI Vertebral Column; biomechanical predictors.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/212/vertebral+column
- Changes: six sagittal alignment measurements named with units; setting names the Lyon clinic.
- Validation: `jevtest check` passes; zero-shot prompt about 222 tokens; full training fold fits the 32k ICL budget.
