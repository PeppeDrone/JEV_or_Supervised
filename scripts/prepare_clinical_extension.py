#!/usr/bin/env python3
"""Prepare four additional clinical tabular datasets (no Kaggle login required).

  uci_heart_failure   UCI 519 — HFCR mortality (exclude follow-up `time`)
  uci_ilpd            UCI 225 — Indian Liver Patient Dataset
  uci_diabetes130     UCI 296 — 30-day readmission, first encounter / patient,
                                stratified subsample for LLM-tractable n
  uci_cervical        UCI 383 — biopsy outcome; drop Dx* leakage columns

Raw files under datasets/<name>/raw/; patients.csv + dataset.yaml beside them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
SEED = 20261008  # match configs/study.yaml

SOURCES = {
    "uci_heart_failure": "https://archive.ics.uci.edu/static/public/519/data.csv",
    "uci_ilpd": "https://archive.ics.uci.edu/static/public/225/data.csv",
    "uci_diabetes130": "https://archive.ics.uci.edu/static/public/296/data.csv",
    "uci_cervical": "https://archive.ics.uci.edu/static/public/383/data.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {"url": url, "path": str(dest.relative_to(ROOT)), "sha256": sha256(dest),
                "bytes": dest.stat().st_size, "retrieved": "cached"}
    req = urllib.request.Request(url, headers={"User-Agent": "jev_prediction_testing/clinical-ext"})
    with urllib.request.urlopen(req, timeout=180) as response:
        dest.write_bytes(response.read())
    return {"url": url, "path": str(dest.relative_to(ROOT)), "sha256": sha256(dest),
            "bytes": dest.stat().st_size, "retrieved": datetime.now(timezone.utc).isoformat()}


def blank(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "")


def dump_yaml(path: Path, obj: dict) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")


def write_bundle(name: str, frame: pd.DataFrame, spec: dict, summary: dict, retrieval: list) -> None:
    out = DATASETS / name
    out.mkdir(parents=True, exist_ok=True)
    patients = out / "patients.csv"
    frame.to_csv(patients, index=False, encoding="utf-8")
    dump_yaml(out / "dataset.yaml", spec)
    notes = {
        "dataset": name,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_retrieval": retrieval,
        "patients_csv_sha256": sha256(patients),
        "summary": summary,
    }
    (out / "PREPARATION.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    (out / "DATASET_NOTES.md").write_text(
        f"# {name}\n\n- Observations: {summary['observations']}\n"
        f"- Predictors: {summary['predictors']}\n- Outcome: {summary['outcome']}\n"
        f"- Class counts: {summary['class_counts']}\n"
        f"- Note: {summary.get('identifier_note', '')}\n",
        encoding="utf-8",
    )
    print(f"{name}: {summary['observations']} rows, {summary['predictors']} predictors, "
          f"classes={summary['class_counts']}")


# ---------------------------------------------------------------------------
# Heart failure clinical records
# ---------------------------------------------------------------------------

def prepare_heart_failure(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    # `time` is follow-up duration (days) — excluded to avoid outcome leakage.
    frame = pd.DataFrame({"record_id": [f"hfcr_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    for col, label, kind, unit, values in [
        ("age", "Age", "numeric", "years", None),
        ("anaemia", "Anaemia", "categorical", None, {"0": "No", "1": "Yes"}),
        ("creatinine_phosphokinase", "Creatinine phosphokinase", "numeric", "mcg/L", None),
        ("diabetes", "Diabetes", "categorical", None, {"0": "No", "1": "Yes"}),
        ("ejection_fraction", "Ejection fraction", "numeric", "%", None),
        ("high_blood_pressure", "High blood pressure", "categorical", None, {"0": "No", "1": "Yes"}),
        ("platelets", "Platelets", "numeric", "kiloplatelets/mL", None),
        ("serum_creatinine", "Serum creatinine", "numeric", "mg/dL", None),
        ("serum_sodium", "Serum sodium", "numeric", "mEq/L", None),
        ("sex", "Sex", "categorical", None, {"0": "Female", "1": "Male"}),
        ("smoking", "Smoking", "categorical", None, {"0": "No", "1": "Yes"}),
    ]:
        if kind == "categorical":
            frame[col] = blank(src[col].map(lambda v: str(int(v))))
            predictors[col] = {"column": col, "kind": kind, "text_label": label, "text_values": values}
        else:
            frame[col] = blank(src[col])
            predictors[col] = {"column": col, "kind": kind, "text_label": label, **({"unit": unit} if unit else {})}
    death_col = "DEATH_EVENT" if "DEATH_EVENT" in src.columns else "death_event"
    frame["death_event"] = src[death_col].map(lambda v: int(v))
    summary = {
        "observations": len(frame),
        "predictors": len(predictors),
        "outcome": "mortality",
        "excluded_predictors": ["time"],
        "class_counts": {
            "Survived follow-up": int((frame["death_event"] == 0).sum()),
            "Died during follow-up": int((frame["death_event"] == 1).sum()),
        },
        "identifier_note": (
            "UCI Heart Failure Clinical Records (Chicco & Jurman). Follow-up `time` excluded "
            "as a predictor to avoid leakage with mortality."
        ),
    }
    spec = {
        "name": "uci_heart_failure",
        "description": (
            "UCI Heart Failure Clinical Records (299 patients, 12 clinical predictors). "
            "Binary mortality during follow-up; follow-up duration excluded."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Cardiology follow-up of patients with heart failure using baseline clinical "
            "and laboratory findings to assess mortality risk."
        ),
        "observations_header": "Heart-failure clinical and laboratory findings:",
        "predictors": predictors,
        "outcomes": {
            "mortality": {
                "column": "death_event",
                "classes": ["Survived follow-up", "Died during follow-up"],
                "mapping": {"0": "Survived follow-up", "1": "Died during follow-up"},
                "question": (
                    "Based on the heart-failure clinical and laboratory findings, did the "
                    "patient die during follow-up?"
                ),
            }
        },
    }
    write_bundle("uci_heart_failure", frame, spec, summary, retrieval)


# ---------------------------------------------------------------------------
# ILPD
# ---------------------------------------------------------------------------

def prepare_ilpd(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    src.columns = [c.strip() for c in src.columns]
    frame = pd.DataFrame({"record_id": [f"ilpd_{i+1:03d}" for i in range(len(src))]})
    frame["age"] = blank(src["Age"])
    frame["gender"] = blank(src["Gender"].map(lambda v: str(v).strip()[:1].upper() if pd.notna(v) else None))
    for col, src_col, label, unit in [
        ("tb", "TB", "Total bilirubin", "mg/dL"),
        ("db", "DB", "Direct bilirubin", "mg/dL"),
        ("alkphos", "Alkphos", "Alkaline phosphatase", "IU/L"),
        ("sgpt", "Sgpt", "Alanine aminotransferase (SGPT)", "IU/L"),
        ("sgot", "Sgot", "Aspartate aminotransferase (SGOT)", "IU/L"),
        ("tp", "TP", "Total proteins", "g/dL"),
        ("alb", "ALB", "Albumin", "g/dL"),
        ("ag_ratio", "A/G Ratio", "Albumin/globulin ratio", None),
    ]:
        frame[col] = blank(src[src_col])
    # Selector: 1 = liver disease, 2 = no liver disease
    frame["liver_disease"] = src["Selector"].map(lambda v: 1 if int(v) == 1 else 0)
    predictors = {
        "age": {"column": "age", "kind": "numeric", "text_label": "Age", "unit": "years"},
        "gender": {"column": "gender", "kind": "categorical", "text_label": "Sex",
                   "text_values": {"M": "Male", "F": "Female"}},
        "tb": {"column": "tb", "kind": "numeric", "text_label": "Total bilirubin", "unit": "mg/dL"},
        "db": {"column": "db", "kind": "numeric", "text_label": "Direct bilirubin", "unit": "mg/dL"},
        "alkphos": {"column": "alkphos", "kind": "numeric", "text_label": "Alkaline phosphatase", "unit": "IU/L"},
        "sgpt": {"column": "sgpt", "kind": "numeric", "text_label": "Alanine aminotransferase (SGPT)", "unit": "IU/L"},
        "sgot": {"column": "sgot", "kind": "numeric", "text_label": "Aspartate aminotransferase (SGOT)", "unit": "IU/L"},
        "tp": {"column": "tp", "kind": "numeric", "text_label": "Total proteins", "unit": "g/dL"},
        "alb": {"column": "alb", "kind": "numeric", "text_label": "Albumin", "unit": "g/dL"},
        "ag_ratio": {"column": "ag_ratio", "kind": "numeric", "text_label": "Albumin/globulin ratio"},
    }
    summary = {
        "observations": len(frame),
        "predictors": len(predictors),
        "outcome": "liver_disease",
        "class_counts": {
            "No liver disease": int((frame["liver_disease"] == 0).sum()),
            "Liver disease": int((frame["liver_disease"] == 1).sum()),
        },
        "identifier_note": "UCI ILPD; Selector 1→liver disease, 2→no liver disease.",
    }
    spec = {
        "name": "uci_ilpd",
        "description": "UCI Indian Liver Patient Dataset (583 observations, 10 predictors).",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Hepatology evaluation using liver function tests and demographics to assess "
            "presence of liver disease."
        ),
        "observations_header": "Demographics and liver function tests:",
        "predictors": predictors,
        "outcomes": {
            "liver_disease": {
                "column": "liver_disease",
                "classes": ["No liver disease", "Liver disease"],
                "mapping": {"0": "No liver disease", "1": "Liver disease"},
                "question": (
                    "Based on the demographics and liver function tests, does the patient "
                    "have liver disease?"
                ),
            }
        },
    }
    write_bundle("uci_ilpd", frame, spec, summary, retrieval)


# ---------------------------------------------------------------------------
# Diabetes 130-US hospitals — 30-day readmission, patient-level subsample
# ---------------------------------------------------------------------------

DIABETES_N = 4000  # stratified sample after first-encounter filter (LLM-tractable); None: whole table
DEATH_DISPOSITIONS = {11, 13, 14, 19, 20, 21}


def prepare_diabetes130(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw, low_memory=False)
    src = src.sort_values("encounter_id").groupby("patient_nbr", as_index=False).first()
    src = src[~src["discharge_disposition_id"].isin(DEATH_DISPOSITIONS)].copy()
    src["y"] = (src["readmitted"] == "<30").astype(int)
    # Stratified subsample for LLM stages while keeping class balance.
    if DIABETES_N is None:
        cohort = src.reset_index(drop=True)
    else:
        rng = np.random.default_rng(SEED)
        parts = []
        for label, group in src.groupby("y"):
            take = min(len(group), max(1, int(round(DIABETES_N * (len(group) / len(src))))))
            idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
            parts.append(group.loc[idx])
        cohort = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        # Top up / trim to exactly DIABETES_N if rounding drifted
        if len(cohort) > DIABETES_N:
            cohort = cohort.sample(n=DIABETES_N, random_state=SEED).reset_index(drop=True)

    frame = pd.DataFrame({"record_id": [f"dm130_{i+1:04d}" for i in range(len(cohort))]})
    frame["race"] = blank(cohort["race"].replace("?", np.nan))
    frame["gender"] = blank(cohort["gender"])
    frame["age"] = blank(cohort["age"])
    frame["admission_type_id"] = blank(cohort["admission_type_id"].map(lambda v: str(int(v)) if pd.notna(v) else None))
    frame["time_in_hospital"] = blank(cohort["time_in_hospital"])
    frame["num_lab_procedures"] = blank(cohort["num_lab_procedures"])
    frame["num_procedures"] = blank(cohort["num_procedures"])
    frame["num_medications"] = blank(cohort["num_medications"])
    frame["number_outpatient"] = blank(cohort["number_outpatient"])
    frame["number_emergency"] = blank(cohort["number_emergency"])
    frame["number_inpatient"] = blank(cohort["number_inpatient"])
    frame["number_diagnoses"] = blank(cohort["number_diagnoses"])
    frame["max_glu_serum"] = blank(cohort["max_glu_serum"].replace("None", np.nan))
    frame["a1c"] = blank(cohort["A1Cresult"].replace("None", np.nan))
    frame["metformin"] = blank(cohort["metformin"])
    frame["insulin"] = blank(cohort["insulin"])
    frame["change"] = blank(cohort["change"])
    frame["diabetes_med"] = blank(cohort["diabetesMed"])
    frame["readmit_30d"] = cohort["y"].astype(int)

    drug = {"No": "Not prescribed", "Steady": "Steady dose", "Up": "Dose increased",
            "Down": "Dose decreased"}
    age_bands = {
        "[0-10)": "0 to 9 years", "[10-20)": "10 to 19 years", "[20-30)": "20 to 29 years",
        "[30-40)": "30 to 39 years", "[40-50)": "40 to 49 years", "[50-60)": "50 to 59 years",
        "[60-70)": "60 to 69 years", "[70-80)": "70 to 79 years", "[80-90)": "80 to 89 years",
        "[90-100)": "90 to 99 years",
    }
    admission = {
        "1": "Emergency", "2": "Urgent", "3": "Elective", "4": "Newborn",
        "5": "Not available", "6": "NULL", "7": "Trauma center", "8": "Not mapped",
    }
    predictors = {
        "race": {"column": "race", "kind": "categorical", "text_label": "Race",
                 "text_values": {
                     "Caucasian": "Caucasian", "AfricanAmerican": "African American",
                     "Hispanic": "Hispanic", "Asian": "Asian", "Other": "Other",
                 }},
        "gender": {"column": "gender", "kind": "categorical", "text_label": "Sex",
                   "text_values": {
                       "Female": "Female", "Male": "Male",
                       "Unknown/Invalid": "Unknown or invalid",
                   }},
        "age": {"column": "age", "kind": "categorical", "text_label": "Age band",
                "text_values": age_bands},
        "admission_type_id": {"column": "admission_type_id", "kind": "categorical",
                              "text_label": "Admission type", "text_values": admission},
        "time_in_hospital": {"column": "time_in_hospital", "kind": "numeric",
                             "text_label": "Time in hospital", "unit": "days"},
        "num_lab_procedures": {"column": "num_lab_procedures", "kind": "numeric",
                               "text_label": "Number of lab procedures"},
        "num_procedures": {"column": "num_procedures", "kind": "numeric",
                           "text_label": "Number of procedures"},
        "num_medications": {"column": "num_medications", "kind": "numeric",
                            "text_label": "Number of medications"},
        "number_outpatient": {"column": "number_outpatient", "kind": "numeric",
                              "text_label": "Outpatient visits (prior year)"},
        "number_emergency": {"column": "number_emergency", "kind": "numeric",
                             "text_label": "Emergency visits (prior year)"},
        "number_inpatient": {"column": "number_inpatient", "kind": "numeric",
                             "text_label": "Inpatient visits (prior year)"},
        "number_diagnoses": {"column": "number_diagnoses", "kind": "numeric",
                             "text_label": "Number of diagnoses"},
        "max_glu_serum": {"column": "max_glu_serum", "kind": "categorical",
                          "text_label": "Max serum glucose",
                          "text_values": {">200": ">200 mg/dL", ">300": ">300 mg/dL", "Norm": "Normal"}},
        "a1c": {"column": "a1c", "kind": "categorical", "text_label": "HbA1c result",
                "text_values": {">7": ">7%", ">8": ">8%", "Norm": "Normal"}},
        "metformin": {"column": "metformin", "kind": "categorical", "text_label": "Metformin",
                      "text_values": drug},
        "insulin": {"column": "insulin", "kind": "categorical", "text_label": "Insulin",
                    "text_values": drug},
        "change": {"column": "change", "kind": "categorical", "text_label": "Diabetes medication changed",
                   "text_values": {"Ch": "Changed", "No": "No change"}},
        "diabetes_med": {"column": "diabetes_med", "kind": "categorical",
                         "text_label": "Any diabetes medication",
                         "text_values": {"Yes": "Yes", "No": "No"}},
    }
    summary = {
        "observations": len(frame),
        "predictors": len(predictors),
        "outcome": "readmit_30d",
        "class_counts": {
            "No 30-day readmission": int((frame["readmit_30d"] == 0).sum()),
            "30-day readmission": int((frame["readmit_30d"] == 1).sum()),
        },
        "identifier_note": (
            f"UCI Diabetes 130-US Hospitals. First encounter per patient; death/hospice "
            f"discharge dispositions {sorted(DEATH_DISPOSITIONS)} removed; stratified sample "
            f"of {DIABETES_N} patients (seed {SEED}) for LLM-tractable evaluation. Identifiers, "
            f"weight (~95% missing), diagnosis codes, and discharge disposition excluded."
        ),
        "full_first_encounter_pool": int(len(src)),
    }
    spec = {
        "name": "uci_diabetes130",
        "description": (
            f"UCI Diabetes 130-US Hospitals 30-day readmission task "
            f"({len(frame)} patient-level encounters after cleaning and stratified subsample)."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Inpatient diabetes care: predicting 30-day hospital readmission from admission "
            "context, utilisation, and diabetes treatment features."
        ),
        "observations_header": "Inpatient encounter and diabetes treatment features:",
        "predictors": predictors,
        "outcomes": {
            "readmit_30d": {
                "column": "readmit_30d",
                "classes": ["No 30-day readmission", "30-day readmission"],
                "mapping": {"0": "No 30-day readmission", "1": "30-day readmission"},
                "question": (
                    "Based on the inpatient encounter and diabetes treatment features, was the "
                    "patient readmitted within 30 days?"
                ),
            }
        },
    }
    write_bundle("uci_diabetes130", frame, spec, summary, retrieval)


# ---------------------------------------------------------------------------
# Cervical cancer risk factors — biopsy outcome
# ---------------------------------------------------------------------------

def prepare_cervical(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw, na_values="?")
    # Drop diagnosis flags (leakage) and near-empty timing fields.
    drop = {
        "Dx", "Dx:Cancer", "Dx:CIN", "Dx:HPV",
        "STDs: Time since first diagnosis", "STDs: Time since last diagnosis",
    }
    keep_cols = [c for c in src.columns if c not in drop and c != "Biopsy"]
    frame = pd.DataFrame({"record_id": [f"cervical_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    for col in keep_cols:
        key = (
            col.lower()
            .replace(" ", "_")
            .replace(":", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("-", "_")
        )
        series = src[col]
        # binary-ish floats → categorical codes when only 0/1
        vals = series.dropna().unique()
        if len(vals) <= 3 and set(np.round(vals.astype(float), 6)).issubset({0.0, 1.0}):
            frame[key] = blank(series.map(lambda v: None if pd.isna(v) else str(int(float(v)))))
            predictors[key] = {
                "column": key, "kind": "categorical", "text_label": col,
                "text_values": {"0": "No", "1": "Yes"},
            }
        else:
            frame[key] = blank(series)
            predictors[key] = {"column": key, "kind": "numeric", "text_label": col}
    frame["biopsy"] = src["Biopsy"].map(lambda v: int(float(v)))
    summary = {
        "observations": len(frame),
        "predictors": len(predictors),
        "outcome": "biopsy",
        "excluded_predictors": sorted(drop),
        "class_counts": {
            "Negative biopsy": int((frame["biopsy"] == 0).sum()),
            "Positive biopsy": int((frame["biopsy"] == 1).sum()),
        },
        "identifier_note": (
            "UCI Cervical Cancer Risk Factors. Outcome = biopsy. Dx* diagnosis columns and "
            "near-empty STD timing fields excluded. Missing values preserved as blanks."
        ),
    }
    spec = {
        "name": "uci_cervical",
        "description": (
            f"UCI Cervical Cancer Risk Factors ({len(frame)} observations). Binary biopsy "
            f"outcome with risk-factor and screening predictors; diagnosis flags excluded."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Gynaecology clinic assessment of cervical cancer risk using sexual/reproductive "
            "history, contraception, STD history, and colposcopy screening tests before biopsy."
        ),
        "observations_header": "Risk factors and cervical screening findings:",
        "predictors": predictors,
        "outcomes": {
            "biopsy": {
                "column": "biopsy",
                "classes": ["Negative biopsy", "Positive biopsy"],
                "mapping": {"0": "Negative biopsy", "1": "Positive biopsy"},
                "question": (
                    "Based on the risk factors and cervical screening findings, is the cervical "
                    "biopsy positive?"
                ),
            }
        },
    }
    write_bundle("uci_cervical", frame, spec, summary, retrieval)


PREPARERS = {
    "uci_heart_failure": prepare_heart_failure,
    "uci_ilpd": prepare_ilpd,
    "uci_diabetes130": prepare_diabetes130,
    "uci_cervical": prepare_cervical,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=list(PREPARERS),
                        help="Prepare only named dataset (default: all four)")
    args = parser.parse_args(argv)
    names = args.dataset or list(PREPARERS)
    report = {}
    for name in names:
        raw_dir = DATASETS / name / "raw"
        meta = [download(SOURCES[name], raw_dir / "data.csv")]
        PREPARERS[name](raw_dir / "data.csv", meta)
        report[name] = json.loads((DATASETS / name / "PREPARATION.json").read_text())
    out = ROOT / "outputs" / "clinical_extension_preparation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
