# mexico_covid

- Observations: 1000
- Predictors: 16
- Outcome: covid_death
- Class counts: {'Alive': 844, 'Died': 156}
- Excluded: ['INTUBADO', 'UCI', 'TIPO_PACIENTE', 'FECHA_INGRESO', 'FECHA_DEF']
- Note: Mexico SSA datos abiertos 2020 cierre. CLASIFICACION_FINAL in {1,2,3}. First 1000 confirmed rows in file order. INTUBADO/UCI/TIPO_PACIENTE dropped. Codes 97/98/99 left empty.

## Codebook review (2026-10-24)

- Source documentation: Mexico SSA Datos Abiertos, COVID-19 2020 closing file and its data dictionary
- Changes: labels clarified (pneumonia at presentation, current smoker, other chronic condition); setting states the 2020 surveillance context and withheld fields.
- Validation: `jevtest check` passes; zero-shot prompt about 211 tokens; full training fold fits the 32k ICL budget.
