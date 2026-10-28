"""Occlusion importance by predictor group: re-predict every patient with one group of predictors withheld.

System 1 (AnyJev L0 zero-shot) sees the patient text with the group's lines deleted; XGBoost sees the
group's columns as missing (native missing values). Fitted models, folds and seeds are those of the main run.
Results are cached per model and group under outputs/<dataset>/<outcome>/occlusion/, so a stopped run resumes.

Usage: PYTHONPATH=src python scripts/occlusion.py --dataset D --outcome O --model zs|xgb
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest.data import load, predictor_matrix
from jevtest.text import entries

# Ordered predictor groups; each predictor key must match exactly one group.
GROUPS = [
    ("Demographics and admission", lambda k: k in {"sex_male", "age_years", "admission_year", "care_pathway"}),
    ("Working diagnosis", lambda k: k.startswith("adm_dx_")),
    ("Presenting symptoms", lambda k: k.startswith("sx_")),
    ("Event characteristics", lambda k: k in {"symptom_duration_min", "symptoms_during_eeg", "symptom_regression_ed",
                                              "witnessed_by_staff", "witnessed_by_others"}),
    ("Epilepsy history and ASM", lambda k: k in {"known_epilepsy_idiopathic", "known_epilepsy_secondary",
                                                 "on_asm_therapy", "asm_underdosed"}),
    ("Comorbidities", lambda k: k.startswith("neuro_comorbidity_") or k in {
        "fever", "electrolyte_alterations", "chemo_radiotherapy", "chemo_radiotherapy_previous",
        "previous_intracranial_event", "poor_baseline_conditions", "nursing_home_resident", "pregnancy_postpartum",
        "cardiac_comorbidities", "hypertension", "diabetes", "dyslipidemia", "thyroid_disorder",
        "metabolic_alterations", "known_brain_lesion", "sepsis", "sepsis_source"}),
    ("Substances", lambda k: k.startswith("intoxication_") or k.startswith("substance_misuse_history_")),
    ("Medication", lambda k: k.startswith("medication_before_eeg_") or k in {"antidepressant_antipsychotic_any",
                                                                           "recent_therapy_change_any"}),
    ("Brain imaging", lambda k: k.startswith("ct_new_lesion_") or k in {"brain_ct_done", "brain_mri_done"}),
    ("Consultations and work-up", lambda k: k in {"neurology_requested", "ecg_done", "neurology_consult",
                                                  "neurosurgery_consult", "anesthesiology_consult", "other_consult",
                                                  "pharmacological_test"}),
    ("EEG timing", lambda k: k in {"admission_to_eeg_report_min", "request_to_eeg_min", "eeg_working_hours",
                                   "eeg_on_call"}),
    ("EEG report", lambda k: k.startswith("eeg_") and k not in {"eeg_working_hours", "eeg_on_call"}),
    ("Events recorded during the EEG", lambda k: k in {"motor_seizure", "non_convulsive_seizure",
                                                       "motor_status_epilepticus", "non_convulsive_status_epilepticus",
                                                       "motor_events_not_eeg_related"}),
]


def groups(run):
    """{group: [predictor keys]} for the groups present in this dataset; fails on unassigned predictors."""
    assigned, out = {}, {}
    for key in run.ds["predictors"]:
        hits = [name for name, test in GROUPS if test(key)]
        if len(hits) != 1:
            raise SystemExit(f"Predictor {key} matches {len(hits)} groups; fix GROUPS")
        assigned[key] = hits[0]
    for name, _ in GROUPS:
        members = [k for k, g in assigned.items() if g == name]
        if members:
            out[name] = members
    return out


def render_without(row, ds, removed):
    """The study's patient text with the lines of `removed` predictors deleted."""
    labels, values = entries(row, ds)
    keep = [k not in removed for k in ds["predictors"]]
    lines = "\n".join(f"{k}: {v}" for k, v, shown in zip(labels, values, keep) if shown)
    return f"{ds['setting']}\n{ds['observations_header']}\n{lines}"


def texts(run, removed):
    return [render_without(row, run.ds, removed) for _, row in run.frame.iterrows()]


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def cached(folder, model, name, compute):
    path = folder / f"{model}__{slug(name)}.npy"
    if path.exists():
        return np.load(path)
    print(f"[{model}] {name}", flush=True)
    values = compute()
    np.save(path, values)
    return values


def run_zero_shot(run, members, folder):
    from anyjev import Question
    from jevtest.llm import anyjev_runtime
    from jevtest.zeroshot import zero_shot_probs

    _, decider = anyjev_runtime(run.study)
    q = Question.choice(run.spec["question"], run.classes, name=run.outcome)
    batch = run.study["backbone"]["anyjev_batch"]

    def decide(states):
        probs = []
        for offset in range(0, len(states), batch):
            for decision in decider.decide_batch(states[offset:offset + batch], q, level="L0", require="L0"):
                probs.append(np.asarray(decision.probs, dtype=float))
            if offset % 200 == 0:
                print(f"  {offset}/{len(states)}", flush=True)
        P = np.array(probs)
        return (P / P.sum(axis=1, keepdims=True))[:, 1]

    reference = zero_shot_probs(run)
    reference = (reference / reference.sum(axis=1, keepdims=True))[:, 1]
    check = decide(texts(run, set())[:32])
    gap = float(np.abs(check - reference[:32]).max())
    print(f"[zs] full-prompt check on 32 patients: max |dp| = {gap:.2e}", flush=True)
    if gap > 1e-3:
        raise SystemExit("Zero-shot does not reproduce the main run; stopping")
    np.save(folder / "zs__full.npy", reference)
    for name, removed in members.items():
        cached(folder, "zs", name, lambda: decide(texts(run, set(removed))))


def run_xgboost(run, members, folder):
    from jevtest.classical import fit_fold

    X = predictor_matrix(run)
    columns = {k: run.ds["predictors"][k]["column"] for k in run.ds["predictors"]}
    fitted = []
    for fold, (train, _) in enumerate(run.splits):
        print(f"[xgb] refitting fold {fold + 1}", flush=True)
        fitted.append(fit_fold(X, run.y, train, "M2", run.study, fold))

    def predict(frame):
        p = np.zeros(len(run.y))
        for model, (_, test) in zip(fitted, run.splits):
            p[test] = model.predict_proba(frame.iloc[test])[:, 1]
        return p

    reference = predict(X)
    gap = float(np.abs(reference - run.predictions("M2")[:, 1]).max())
    print(f"[xgb] refitted XGBoost vs main run: max |dp| = {gap:.2e}", flush=True)
    if gap > 1e-5:
        raise SystemExit("XGBoost does not reproduce the main run; stopping")
    np.save(folder / "xgb__full.npy", reference)
    for name, removed in members.items():
        def occluded(removed=removed):
            frame = X.copy()
            frame[[columns[k] for k in removed]] = np.nan
            return predict(frame)
        cached(folder, "xgb", name, occluded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--model", required=True, choices=["zs", "xgb"])
    args = parser.parse_args(argv)
    run = load(ROOT, args.dataset, args.outcome)
    members = groups(run)
    folder = run.out / "occlusion"
    folder.mkdir(exist_ok=True)
    (folder / "groups.json").write_text(json.dumps(members, indent=2) + "\n")
    print(f"{len(members)} groups, {sum(map(len, members.values()))} predictors, {len(run.y)} patients", flush=True)
    {"zs": run_zero_shot, "xgb": run_xgboost}[args.model](run, members, folder)
    print(f"[{args.model}] done", flush=True)


if __name__ == "__main__":
    main()
