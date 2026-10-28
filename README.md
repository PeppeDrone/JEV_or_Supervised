# JEV_or_Supervised

Code and original predictions for the study comparing frozen language-model class scores (AnyJev L0 readout, Qwen3.5 family) with tuned XGBoost on 45 clinical prediction tasks.

## What is included
- `src/`, `scripts/`, `tests/`, `configs/`, `pyproject.toml`: the analysis code, study configurations and tests.
- `datasets/<name>/`: dataset descriptions only (`dataset.yaml`, `DATASET_NOTES.md`, `PREPARATION.json`, and where present `readout_to_full.json`). No patient-level data tables and no raw downloads are included.
- `outputs/`: original predictions for every task and model, with the fixed hold-out assignments (`holdout/split.json`), tuned XGBoost parameters, metrics and cached language-model probabilities.
- `outputs_qwen3*/`, `outputs_sensitivity/`: original predictions for the other backbones and for the prior-sensitivity analysis.
- `reports/scientific_revision_2026-10-26_v2/`: the current versioned revision results (tables, regenerated inference scores, calibration parameters, figures). `reports/scientific_revision_2026-10-25_v1/` is the frozen earlier version.
- `manuscript/npj_digital_medicine/`: the final LaTeX project with figures and compiled PDF, and the two build scripts.
- `AGENTS.md`, `codeflow.md`, `SCIENTIFIC_REVISION_REPORT.md`: project rules, code flow and the revision report.

## What is not included
- All datasets (patient-level tables and raw source files). Obtain each from the original source named in its `DATASET_NOTES.md`, then place it in `datasets/<name>/`.
- Saved XGBoost models and per-task diagnostic plots, which can be regenerated.
- One prediction metadata file above 100 MB (`outputs/mexico_covid/covid_death__full/predictions/M2.meta.json`), which can be regenerated.
- The protected pod-run result folders.

## Reproducing
1. Install the environment from `pyproject.toml`.
2. Recreate the dataset tables with the `scripts/prepare_*.py` and `scripts/import_*.py` scripts, using the sources in each `DATASET_NOTES.md`.
3. Run `scripts/run_scientific_revision.py`, then `scripts/plot_scientific_revision.py`. Recomputation requires an explicit `--refresh`. Language-model inference needs the pinned model revisions recorded in the results and a suitable GPU.

No licence and no persistent archive identifier have been assigned yet. Data redistribution conditions of the source datasets have not been verified, which is why no data are included.
