# Scientific revision analysis flow

Scope: retain the 45 tasks, original development/hold-out assignments, frozen L0 versus tuned XGBoost primary comparison, and existing task-level inference. Reviewer points 11 and 13 are excluded. `pod_h200_run_2026-10-25_prior_none` is protected: no script in this revision traverses or writes it.

## Inputs and preserved baseline

`datasets/*/{patients.csv,dataset.yaml}` -> `jevtest.data.load/predictor_matrix`; original assignments are read from `outputs/*/*/holdout/split.json`. Six explicit `outputs_qwen3.5_*` roots contain hold-out probabilities. No new split is drawn. Baseline outputs and original manuscript/scripts are preserved in `reports/scientific_revision_2026-10-25_v1/{baseline,source_snapshot}`. Original ladder analysis is rerun there before manuscript revision.

## Execution tree

* `scripts/run_scientific_revision.py`: load fixed tasks, audit identity/class ordering and probabilities; fit reference scorers from development rows; evaluate frozen hold-out predictions; run development-prevalence interventions, available calibration budgets, ranking comparisons, multiclass diagnostics, scaling, and exploratory correlations. Outputs are CSV/JSON under the versioned revision directory, with explicit status rows for missing inference.
* `src/jevtest/revision_reference.py`: development-only CV marker selection and frozen transformations. The same selected marker supplies AUROC and balanced accuracy. Candidate CV estimates and fitted state are saved. Direction-only and additive definitions are retained from the existing script.
* `src/jevtest/revision_calibration.py`: label-budget sampling, regularised offset/temperature fitting and direct operating-point fitting. Fitting functions accept development scores/labels only. Evaluation is separate.
* `scripts/run_revision_inference.py`: explicitly selected cached backbone, fixed patients; generate development scores, semantic renderings, feature perturbations and readout-stage diagnostics using the pinned AnyJev implementation. Writes separate revision artifacts, never original prediction files.
* `src/jevtest/revision_rendering.py`: deterministic A–F renderings and development-defined perturbations; labels never enter a prompt.
* `scripts/plot_scientific_revision.py`: publication figures from CSV outputs; no model fitting or variant selection.
* `manuscript/npj_digital_medicine/build_revision_manuscript.py`: numerical claims from revision outputs -> manuscript, supplement and claim ledger; existing PDF exporter compiles the resulting LaTeX.

## Invariants and intentional corrections

Study seed 20261008. Original task-summary/scaling bootstrap streams 0/1 are retained. Original IDs/class ordering/splits and original rank-sum AUROC arithmetic are fixed. Independent patient bootstrap is retained; draws lacking any outcome class are invalid for full-class BA and macro AUROC and excluded with counts reported. No hierarchical inference redesign. Selection, imputation, direction, calibration and operating points use development data only. Best-marker share is exclusively `(AUROC_marker - .5)/(AUROC_XGB - .5)`, undefined for XGB AUROC <= .6; it can exceed one. No hold-out best-method selection or prompt selection. New locally fitted readouts are additional analyses, not zero-shot systems.

## Executed coverage

All 270 original task/backbone conditions, corrected references on all 45 tasks, and all-model prevalence budgets ran. New development inference, learned calibration, complete semantic controls, readout stages and direction/interaction probes ran only on the three 0.8B pilots. The 27B checkpoint-loading attempt failed because weights were absent from the local cache; full 27B and optional 397B intervention inference is incomplete. `experiment_status.csv` records coverage and missing-class failures. The final report distinguishes empirical coverage from executable implementation.

## Outputs, recomputation and failure

Readable CSV tables contain task/backbone/method/budget/repeat identifiers; JSON contains seeds, versions, input SHA256 hashes and fitted parameters. Predictions remain local and are not bundled for release without author review. Expensive inference checkpoints are per task and condition. Existing files are not silently replaced; explicit `--resume` verifies matching metadata before reuse. CPU work uses one numerical-library thread. Exceptions are recorded with task identity; incomplete inference is never replaced with synthetic scores. Figure/PDF logs and tests remain in the revision logs directory.
