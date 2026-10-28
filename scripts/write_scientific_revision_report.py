#!/usr/bin/env python3
"""Generate the implementation report from executed revision artifacts."""
from pathlib import Path
import sys
import json
import re
import hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_scientific_revision import OUT


# Historical (2026-10-25 v1) report generator. The v2 report (SCIENTIFIC_REVISION_REPORT.md) is written from the
# executed v2 outputs; running this would overwrite it with v1 content, so it refuses unless --allow-v1 is given.
if __name__ == "__main__" and "--allow-v1" not in sys.argv:
    raise SystemExit("v1-only generator; SCIENTIFIC_REVISION_REPORT.md is the v2 report (see reports/scientific_revision_2026-10-26_v2)")


def main():
    props=pd.read_csv(OUT/'task_properties.csv');old=pd.read_csv(OUT/'baseline/recomputed/mechanism_covariates.csv')
    primary=pd.read_csv(OUT/'primary_summary.csv');cal=pd.read_csv(OUT/'calibration_metrics.csv')
    scale=pd.read_csv(OUT/'scaling_summary.csv');status=pd.read_csv(OUT/'experiment_status.csv')
    references=pd.read_csv(OUT/'reference_comparisons.csv');direction=pd.read_csv(OUT/'direction_metrics.csv')
    claims=json.loads((OUT/'manuscript_claims.json').read_text())
    qa=json.loads((OUT/'artifact_qa.json').read_text()) if (OUT/'artifact_qa.json').exists() else None
    tests_path=OUT/'logs/tests_final.log'
    if not tests_path.exists():tests_path=OUT/'logs/tests_after_tie_fix.log'
    test_summary=[ln for ln in tests_path.read_text().splitlines() if re.search(r'\d+ passed',ln)]
    p27=primary[(primary.backbone=='27b')&(primary.metric=='balanced_accuracy')].iloc[0]
    a27=primary[(primary.backbone=='27b')&(primary.metric=='auroc')].iloc[0]
    def preval(b,kind):
        t=cal[(cal.backbone==b)&(cal.method=='prevalence')&(cal.budget=='full')&((cal.n_classes==2) if kind=='binary' else (cal.n_classes>2))]
        return t.delta_ba.median()
    old_results=pd.DataFrame([
        ['Best marker median AUROC',old.best_univariate_auc.median(),props.best_univariate_auc.median(),'Development CV selection replaces hold-out maximisation'],
        ['Best marker median BA',old.best_univariate_bacc.median(),props.best_univariate_bacc.median(),'One CV-selected predictor supplies both metrics'],
        ['27B median AUROC',a27['median'],a27['median'],'Original probabilities preserved'],
        ['27B median BA',p27['median'],p27['median'],'Original argmax preserved'],
        ['27B median paired BA gap',p27.median_delta,p27.median_delta,'Tuned XGB comparison preserved'],
    ],columns=['Quantity','Original analysis','Revised analysis','Reason'])
    formatted=old_results.copy()
    for column in ['Original analysis','Revised analysis']:formatted[column]=formatted[column].map(lambda x:f'{x:.6f}')
    rows=['| '+' | '.join(formatted.columns)+' |','|'+'---|'*len(formatted.columns)]
    rows.extend('| '+' | '.join(map(str,row))+' |' for row in formatted.itertuples(index=False,name=None))
    comparison='\n'.join(rows)
    discrepancies=[
        dict(claim='Backbones in abstract/main',reported='five',reproduced='six: 0.8B, 2B, 4B, 9B, 27B, 397B-A17B FP8',resolution='all six retained'),
        dict(claim='27B unresolved AUROC tasks in abstract',reported='29/45',reproduced=f'{int(a27.unresolved)}/45',resolution='29 belongs to 397B; revised 27B count is 25'),
        dict(claim='4B median AUROC',reported='.755',reproduced=f"{primary[(primary.backbone=='4b')&(primary.metric=='auroc')]['median'].iloc[0]:.6f}",resolution='generated manuscript uses corrected tables'),
        dict(claim='4B -> 27B shared tasks',reported='40',reproduced='45',resolution='completed six-backbone files reused'),
        dict(claim='Fig 1e share definition',reported='BA share in some versions; AUROC share in Methods',reproduced='AUROC ratio only; XGB AUROC > .6',resolution='one unbounded quantity everywhere in current outputs'),
        dict(claim='Binary/multiclass AUROC deficit',reported='ranking not a problem',reproduced='397B multiclass marginal medians 0.817 vs 0.973; paired median deficit about 0.101',resolution='substantial discrimination deficit acknowledged'),
        dict(claim='Class-collapse count',reported='BA exactly .5 used as proxy',reproduced='constant predicted class counted directly over all K',resolution='31 tasks at 0.8B, 7 at 27B, 5 at 397B'),
    ]
    pd.DataFrame(discrepancies).to_csv(OUT/'baseline_discrepancies.csv',index=False)
    current='reports/scientific_revision_2026-10-25_v1'
    text=f"""# Scientific revision implementation report

This revision implements validity corrections and executes the analyses supported by saved predictions and a local NVIDIA GB10. **It is not empirically complete:** the requested full 45-task 27B learned-calibration, semantic and readout-intervention experiments remain unexecuted. Their pipelines are implemented. No result is supplied for missing inference. Reviewer points **11 and 13 are explicitly excluded**; the 45 tasks, dataset definitions, original assignments and task-level inference framework were retained.

The protected `pod_h200_run_2026-10-25_prior_none` directory and its contents were not read, renamed, edited, moved, removed or nested by this revision. All inputs were selected through explicit dataset/output paths; no pod-run folder was traversed. The workspace has no Git metadata, so original analysis sources, manuscripts, TeX, PDFs and the existing bundle were preserved under `{current}/source_snapshot` before replacement. Original prediction files and split files were not replaced.

## Implemented and executed

1. Reproduced the original six-backbone ladder analysis and original reference analysis, writing separate baseline CSVs, figures and logs under `baseline/recomputed`. Every backbone has the same original 7,445 hold-out identities across 45 tasks, with 29,717 development patients. All 270 task-backbone original comparisons were rescored. `prediction_inventory.csv` showed zero complete development prediction sets originally available.
2. Replaced hold-out maximisation of marker AUROC **and** the independent hold-out maximisation of marker BA with one development-CV marker selection. All encodings, imputation and OVR directions are fitted inside training folds. The selected marker is refitted on all development patients; both metrics use that marker. Fold estimates, selected states and hold-out reference scores are saved for all tasks. Original direction-only and additive definitions were retained.
3. Recomputed marker share exclusively as `(marker AUROC-.5)/(XGB AUROC-.5)`, excluding XGB AUROC <= .6; values can exceed one. Recomputed reference contrasts, rankings, associations, figures and tables. The XGB-minus-additive contrast is not interpreted as a pure interaction assay.
4. Executed label-blind development-prevalence decisions at feasible budgets 10/20/50/100 and full development for all six backbones and 45 tasks, with 20 seeded repetitions at finite budgets. Smoothed prevalence, fitted parameters and exact sampled development identities are saved. Probability diagnostics retain unmodified probabilities for prevalence and binary-threshold decision interventions.
5. Generated **508 new development and 129 original hold-out readouts** at pinned 0.8B on three fixed pilots: Coimbra breast cancer, hepatitis and dermatology. Executed learned offsets, positive scalar temperature, temperature plus offsets and operating-point fitting across budgets. `{len(cal):,}` successful task/backbone/method/budget/repetition evaluations are saved in `calibration_metrics.csv`; {int((status.experiment=='calibration').sum())} missing-class learned fits and {int((status.experiment=='label_budget').sum())} infeasible backbone-budget conditions are explicitly recorded. Successful-fit curves are conditional on coverage and do not silently replace failed samples.
6. Executed all six semantic rendering conditions for those same three 0.8B pilots (18 task-condition evaluations), full readout-stage reconstruction (12 task-stage evaluations), 50 marginal feature-direction comparisons and two-feature probes where two eligible predictors existed. Saved token-answer mass, position distributions, cyclic-order flips and all four stage probabilities. Full reconstruction agrees with the pinned library's L0 output.
7. Executed direct paired reference AUROC intervals and patient-ranking correlations for all six backbones and tasks, plus development-fitted direction-link and direction-plus-L0 combinations in the three pilots. Similar performance is not called equivalence. All new ranking-test P values have separate BH corrections by diagnostic.
8. Executed paired 27B-to-397B changes, class-count-adjusted BA, class-specific recall/distributions/confusions and binary-versus-multiclass comparisons. Task-property analyses separate L0, XGB and gap associations; principal partial rank correlations account for baseline difficulty and K. Raw coefficients, uncertainty and global BH corrections are saved. Observational codebook verbosity/coverage measures are distinguished from semantic interventions.
9. Corrected the concrete absent-class bootstrap error: ordinary paired resamples lacking any required class have undefined full-class BA and macro AUROC and are counted/excluded. The ordinary patient bootstrap, 1,000 draws, 2,000 task-bootstrap summaries and original Wilcoxon framework remain. Original across-task bootstrap random streams (0 for primary summaries, 1 for scaling) and rank-sum AUROC arithmetic were retained to avoid introducing rounding-dependent rank-test differences. Calibration and new association analyses use the explicit study seed 20261008. There is no hierarchical redesign or retrospective equivalence margin.
10. Generated 3 revised main and 9 supplementary figures in editable SVG, vector PDF and 600-dpi PNG, updated Supplementary Table 1 in three readable parts, added calibration-failure coverage, and rewrote the current manuscript, supplement and cover-letter draft from executed results. Standalone main/supplement PDFs compile. The existing Overleaf bundle contains manuscript text and aggregate figures, not patient-level tables.

## Original versus corrected results

{comparison}

The original primary comparison is preserved. At 27B the BA gap is {p27.median_delta:.6f} (95% interval {p27.ci_low:.6f} to {p27.ci_high:.6f}); {int(p27.xgb_better)} tasks favour XGB and {int(p27.unresolved)} remain unresolved under the paired patient-bootstrap criterion. Unresolved intervals do not demonstrate equivalence.

Full-development prevalence adjustment changed median BA by {preval('27b','binary'):.6f} in binary and {preval('27b','multiclass'):.6f} in multiclass tasks at 27B, and by {preval('397b_a17b_fp8','binary'):.6f} and {preval('397b_a17b_fp8','multiclass'):.6f} at 397B. This empirical intervention did not close the median deficit. It does **not** rule out learned larger-model offsets, which remain untested.

Corrected 27B-minus-marker median AUROC is {references[(references.backbone=='27b')&(references.reference=='best_marker')].delta_auroc.median():.6f}. Direction agreement is {100*direction.agreement.mean():.1f}% across 50 marginal perturbations. Neither patient independence nor general clinical-direction knowledge is inferred from those repeated probes. Scaling results are in `scaling_summary.csv`: the architectures and numerical precisions differ, and marginal-median differences are distinct from median within-task changes.

`baseline_discrepancies.csv` records stale backbone/task counts, inconsistent unresolved-task claims, rounded performance discrepancies, the share-definition mismatch and the incorrect collapse proxy. Historical sources and figures are retained as provenance; current publication figures are exclusively under `{current}/figures`.

## Empirical incompleteness and resource evidence

* **Full 27B calibration and label budgets:** not executed because full development L0 probabilities are absent and the local 27B snapshot contains configuration/tokenizer material but no model-weight shards. The actual offline loading attempt failed; see `logs/inference_27b_availability.log`. A local GB10 exists; this is not a claim that no GPU was available. No remote H200 allocation/access was configured or accessed by this revision.
* **Full 27B semantic and feature-direction experiments:** the complete deterministic pipeline is implemented, but no 27B effect estimate is available. The 0.8B pilots are not substitutes for the requested primary practical configuration. Removing structured unit fields does not erase intrinsic quantity wording from feature names; this limitation is disclosed.
* **397B interventions:** no new development/semantic/protocol inference was executed. Its existing complete hold-out predictions support original performance, prevalence adjustment, scaling, ranking and multiclass analyses only.
* **Other models/tasks:** learned calibration and complementarity lack complete development inference in 267 of 270 task-backbone conditions. Remaining small-model inference was not run to full benchmark coverage. This is empirical incompleteness, not completion through softened manuscript wording.
* **Clinical direction and interactions:** within-range marginal probes and limited two-feature contrasts ran only in pilots. Joint clinical plausibility is unverified; nonlinear/context-dependent relationships were screened conservatively. These probes do not establish internal learned feature interactions or their absence.
* **Option order:** saved diagnostics quantify cyclic rotations; arbitrary non-cyclic reorderings were not executed. Protocol reconstruction uses the pinned content-free correction and log-mean, not another calibration protocol.
* **Clinical utility:** no task-specific cost function, sensitivity target or deployment threshold was invented. Class recall is available as a secondary diagnostic. No prospective validation or deployment experiment was performed.

## Claims supported, narrowed or removed

Direct evidence supports a separation between frozen discrimination and default categorical decisions, failure of the specified epidemiological-prevalence rule to close the median gap, configuration-specific effects of semantic and readout interventions in the 0.8B pilots, and heterogeneous paired changes between 27B dense and 397B-A17B FP8.

Removed claims include universal saturation at 27B, future-model futility, functional equivalence to a direction-only model, established predictor-direction knowledge, absence of interactions, irrelevance of codebook detail, ranking being unproblematic on multiclass tasks, categorical predictions being unpairable/unbootstrappable, and clinical readiness. The proposition that a few local labels repair 27B remains an untested hypothesis. Failed difference tests are described as unresolved, not equivalent.

## Ethics, release and author actions

This was a limited inspection of existing documentation, not the excluded provenance audit. `datasets/ppd_bangladesh/DATASET_NOTES.md` identifies Mendeley 4nznnrk8cg v3; hip-fracture notes identify the Zenodo development-cohort deposit; Berlin appendicitis notes identify OSF 9wvys and Reismann et al. The documents identify deposited source cohorts but do not establish this project's ethics determination or original consent permissions. The manuscript flags these items for author verification; no approval, exemption, DOI for this study, public repository URL, institutional permission or author approval was invented. Existing source-deposit identifiers are reported only as statements in the project notes.

Before submission the authors must verify governance/consent and source permissions, funding, competing interests, contributions, revised-author approval, originality/concurrent-submission declarations and a persistent study code/data location. Patient-level input, prediction and diagnostic artifacts remain local. Aggregated figures/tables can be reviewed for release; no external publication or messaging was performed.

## Journal format and compilation

Official guidance checked on 25 October 2026: [npj Digital Medicine content types](https://www.nature.com/npjdigitalmed/content-types) and [submission guidelines](https://www.nature.com/npjdigitalmed/for-authors-and-referees/submission-guidelines). It recommends a title up to 15 words, abstract up to 70, main text 1,000–1,500 excluding Methods/references/legends, no main-text subheadings, all Methods in the main file and figure legends up to 350 words. The current official page did not specify a Brief Communication figure cap; three main figures were retained. The generated document has {claims['word_counts']['title']} title words, {claims['word_counts']['abstract']} abstract words and {claims['word_counts']['main_text']} main-text words. Supplementary information is a separate PDF. Author-action statements mean the package is a review draft, not submission-certified.

PDF builds use vector figures and strict error checking. Long supplementary tables break across pages; logs are retained. Main figure references are LaTeX references with matching labels; supplementary references match the separately compiled supplement. Every principal result claim is recorded with its source, selector, value and source SHA256 in `manuscript_claims.json`.

## Validation and failures

Latest pytest result: **{test_summary[-1] if test_summary else 'not yet recorded'}** (see `{tests_path.relative_to(ROOT)}`). The initial run had one failed new test because it incorrectly assumed class permutation preserves first-index argmax tie resolution. The fixture was corrected to distinguish non-tied class-order invariance from documented exact ties; this was a test assumption error. Initial and corrected logs are retained. Existing figure-canvas warnings and a validation-only probability warning are preserved rather than suppressed.

Artifact QA: **{'PASS' if qa and qa['passed'] else 'not yet passed'}**; see `artifact_qa.json`. Checks cover original input hashes, original primary point estimates, development-CV selection replay, calibration identity separation, temperature invariance, invalid-bootstrap accounting, principal numerical traceability, word limits, the complete figure set and LaTeX references/overflow. Unit tests additionally exercise missing/unseen categories, rare classes, metric class order, probability validity, deterministic sampling/fitting, all 270 original hold-outs and outcome-blind rendering. Original-model tuning tests are retained.

An initial execution refused historical XGB column descriptions that differ from expanded current labels; explicit numeric producer-order mapping resolved this without reordering classes. The first manuscript build was 971 main-text words and was expanded with clinically relevant decision-rule limitations to meet the guidance. A PDF table-overflow check triggered a paginated landscape-table correction. All failures and final logs remain inspectable.

## Reviewer traceability matrix

The attachment supplies phases and the explicit exclusions rather than a complete verbatim numbered reviewer letter. The operational mapping below associates the requested criticisms with their supplied topics; points 11 and 13 are left excluded. “Partial” always means empirical work is still missing even when code and manuscript corrections exist.

| Point | Topic | Code / implementation | Executed evidence | Manuscript / status |
|---|---|---|---|---|
| 1 | Best-marker test selection | `revision_reference.py`, corrected `task_mechanism_covariates.py` | CV candidates, selected states, task properties; all 45 tasks | Methods references, Fig. 1d–e, Table S1; **implemented and executed** |
| 2 | Decision/calibration explanation | `revision_calibration.py`, inference/analysis entrypoints | All-model prevalence; learned 0.8B pilots and failures | Main intervention results, Figs. S2–3; **partial: learned 27B absent** |
| 3 | Clinical semantics / knowledge | `revision_rendering.py`, inference entrypoint | Six renderings on 3 pilots, direction metrics | Main semantic results, Figs. S4/S9; **partial: full 27B absent** |
| 4 | Reference resemblance / interactions | paired comparisons, rankings and combination fits | All-model reference/ranking data; pilot combinations/probes | Reference results, Fig. S8; **partial mechanistic scope**, no equivalence |
| 5 | Scaling | paired scaling summaries and distribution plots | All 45 paired 27B/397B tasks | Main scaling, Fig. S7; **executed**, architecture/precision qualified |
| 6 | Multiclass failure | chance-adjusted BA and full-class diagnostics | Original 8 multiclass tasks for every backbone; available offset pilot | Main multiclass, Fig. S6; **partial: larger-model offsets absent** |
| 7 | Coupling / exploratory associations | component and partial-rank analyses, BH correction | Raw coefficients, intervals and global q values | Fig. 3/S1 and Methods; **executed**, noncausal |
| 8 | Statistical interpretation / absent classes | bootstrap fixes in metrics, XGB and ladder scripts | Invalid counts, corrected CIs, retained task framework | Methods statistics; **executed**, no equivalence inference |
| 9 | Codebook-detail interpretation | operational verbosity and coverage measures; semantic controls | Descriptive tables; 0.8B controls | Methods/limitations; **partial: controlled 27B absent** |
| 10 | Readout validation / collapse | saved diagnostics and pinned stage reconstruction | All-model class frequencies; 3-task 0.8B token-mass/order/stage data | Fig. S5, readout Methods; **partial: 27B and arbitrary order tests absent** |
| 11 | Broad dataset design, temporal/subsampling/provenance audit | **Explicitly excluded** | No new audit or external validation | Original task definitions retained |
| 12 | Clinical decision relevance | class recall, probability and decision metrics | Per-class diagnostics and calibration measures | Main clinical framing; **executed within scope**, no invented utility |
| 13 | Comprehensive estimand / hierarchical redesign | **Explicitly excluded** | Original ordinary paired/task bootstrap framework retained | Only concrete absent-class handling corrected |
| 14 | False categorical-prediction assertion | manuscript generator | Mathematical distinction corrected; typed score diagnostics exist | Introduction and Methods; **corrected** |
| 15 | Calibration, prevalence and thresholds | separate fitted distributions/decision-only rules | Saved distinct methods/parameters/diagnostics | Methods intervention definitions; **implemented**, large-model fitting partial |
| 16 | Numerical/figure consistency | baseline comparison, claim ledger and artifact QA | Discrepancy CSV, 12 figures, regenerated tables | Entire current draft; **executed** |
| 17 | Ethics inconsistency | limited existing-document inspection | Existing notes only; no fabricated determination | Ethics and explicit author action; **unresolved governance** |
| 18 | Reproducibility and QA | seeds, hashes, versions, separate artifacts, tests | Execution manifest, dependency snapshot, logs and QA | Code/data availability and report; **implemented/executed within coverage** |
| 19 | Brief Communication format | result-driven Markdown/LaTeX build | Word counts, compiled PDFs, matching legends | **executed**, draft still needs author verification |
| 20 | Release and defensible conclusions | report, persistent-location action items, restricted-data exclusion | Local machine-readable results and aggregate figure bundle | **partial release readiness**; no public deposit or permissions invented |

## Reproduction commands

Run from the project root with the existing virtual environment. The protected pod directory is not an argument to any command.

```bash
# Additional inference in a separate revision tree (cached pilot weights).
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/run_revision_inference.py --backbone 0.8b --datasets uci_breast_coimbra,uci_hepatitis,uci_dermatology --development --semantic --direction --batch 8 --resume

# Required complete 27B inference, once its exact pinned weights/GPU resources exist.
# This is NOT a completed experiment; omit offline mode only when downloads are intended.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_revision_inference.py --backbone 27b --development --semantic --direction

# Recompute all original bootstrap results and every derived result.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_scientific_revision.py --refresh

# After inference-only additions, verified original bootstrap results can be reused.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_scientific_revision.py --refresh --reuse-bootstrap

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/plot_scientific_revision.py
.venv/bin/python manuscript/npj_digital_medicine/build_revision_manuscript.py
.venv/bin/python manuscript/npj_digital_medicine/build_pdfs.py --force
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_scientific_revision.py
.venv/bin/python scripts/write_scientific_revision_report.py
```

To replay the preserved original analysis into a new directory, run `.venv/bin/python scripts/reproduce_original_analysis.py --out reports/original_analysis_replay` with one numerical-library thread. That replay deliberately retains the original reference leakage for provenance only. The existing baseline was already executed with output paths redirected to `baseline/recomputed`; do not use original reference estimates in the revised manuscript. New authoritative publication entrypoints are listed above; older panel/table composition workflows are historical and must not regenerate publication material from the unrevised `reports/backbone_ladder` tree. Never use held-out labels to choose among methods, prompts, budgets or model variants.
"""
    (ROOT/'SCIENTIFIC_REVISION_REPORT.md').write_text(text)
    release=dict(public_study_repository=None,study_doi=None,redistribution_permissions_verified=False,
                 author_review_required=True,
                 aggregate_review_candidates=[p.name for p in sorted(OUT.glob('*.csv'))],
                 local_only_paths=['datasets/','outputs*/','inference/','references/','calibration_parameters/','combination_parameters/'],
                 manuscript_bundle_contents='LaTeX and aggregate figure PDFs only; no patient-level CSV/JSON',
                 incomplete_experiments=['full 27B learned calibration/semantic/direction/readout inference',
                                         '397B development and intervention inference','complete remaining small-model development inference'])
    (OUT/'release_manifest.json').write_text(json.dumps(release,indent=2))
    codepaths=list((ROOT/'src/jevtest').glob('revision_*.py'))
    codepaths += [ROOT/p for p in [
        'AGENTS.md','codeflow.md','src/jevtest/data.py','src/jevtest/text.py','src/jevtest/config.py',
        'src/jevtest/llm.py','src/jevtest/metrics.py','src/jevtest/xgb_optuna.py',
        'scripts/task_mechanism_covariates.py','scripts/plot_backbone_ladder.py',
        'scripts/run_scientific_revision.py','scripts/run_revision_inference.py','scripts/plot_scientific_revision.py',
        'scripts/validate_scientific_revision.py','scripts/reproduce_original_analysis.py','scripts/write_scientific_revision_report.py',
        'tests/test_scientific_revision.py','manuscript/npj_digital_medicine/build_revision_manuscript.py',
        'manuscript/npj_digital_medicine/build_pdfs.py']]
    artifacts=list(OUT.glob('*.csv'))+[p for p in OUT.glob('*.json') if p.name!='final_provenance.json']
    artifacts+=list((OUT/'figures').glob('*.pdf'))
    artifacts += [ROOT/'SCIENTIFIC_REVISION_REPORT.md']
    artifacts+=list((ROOT/'manuscript/npj_digital_medicine').glob('*.pdf'))
    artifacts+=list((ROOT/'manuscript/npj_digital_medicine/overleaf').glob('*.tex'))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    provenance=dict(code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codepaths},
                    artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in artifacts},
                    execution_manifest='execution_manifest.json',inference_manifests='inference/*/*/*/manifest.json',
                    original_inputs_preserved=True,protected_pod_directory_excluded=True)
    (OUT/'final_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(ROOT/'SCIENTIFIC_REVISION_REPORT.md')


if __name__=='__main__':main()
