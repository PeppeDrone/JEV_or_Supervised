# uci_arrhythmia

- Observations: 429
- Predictors: 279
- Outcome: arrhythmia_class
- Class counts: {'Normal': 245, 'Ischemic changes (coronary artery disease)': 44, 'Old anterior myocardial infarction': 15, 'Old inferior myocardial infarction': 15, 'Sinus tachycardia': 13, 'Sinus bradycardia': 25, 'Right bundle branch block': 50, 'Unclassified arrhythmia': 22}
- Identifier note: record_id enumerates retained arrhythmia.data rows after dropping arrhythmia types with fewer than 10 examples (study min_class_count). Original patient identifiers were already removed upstream by UCI.

Raw sources and SHA256 checksums are recorded in PREPARATION.json.
No imputation, scaling or outcome-guided feature selection was applied.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/5/arrhythmia
- Changes: all 279 attributes glossed per lead (widths ms, deflection counts, morphology flags, amplitudes x0.1 mV, QRS/QRST areas) from the UCI names file; class names expanded; header explains units.
- Validation: `jevtest check` passes; zero-shot prompt about 5108 tokens; ICL subsample of about 18 rows at ~1649.7 tokens per row.
