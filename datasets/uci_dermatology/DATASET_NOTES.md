# uci_dermatology

- Observations: 366
- Predictors: 34
- Outcome: dermatological_diagnosis
- Class counts: {'Psoriasis': 112, 'Seborrhoeic dermatitis': 61, 'Lichen planus': 72, 'Pityriasis rosea': 49, 'Chronic dermatitis': 52, 'Pityriasis rubra pilaris': 20}
- Identifier note: Patient names and id numbers were removed from the UCI dermatology release; record_id enumerates source rows.

Raw sources and SHA256 checksums are recorded in PREPARATION.json.
No imputation, scaling or outcome-guided feature selection was applied.

## Codebook review (2026-10-24)

- Source documentation: https://archive.ics.uci.edu/dataset/33/dermatology
- Changes: 0-3 scale stated once in the header with per-line clinical/histopathology scale notes; disease class names spelled out.
- Validation: `jevtest check` passes; zero-shot prompt about 836 tokens; ICL subsample of about 275 rows at ~114.5 tokens per row.
