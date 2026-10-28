#!/usr/bin/env python3
"""Prepare eight additional healthcare classification datasets (UCI / archive, no Kaggle).

  uci_breast_coimbra   UCI 451 — breast cancer vs control (blood biomarkers)
  uci_early_diabetes   UCI 529 — early-stage diabetes risk (symptoms)
  uci_spectf           UCI 96  — cardiac SPECT diagnosis
  uci_hepatitis        UCI 46  — hepatitis survival
  uci_obesity          UCI 544 — WHO obesity class (7-way)
  uci_bone_marrow      UCI 565 — transplant survival status
  uci_parkinsons       UCI 174 — Parkinson's vs healthy (voice)
  uci_haberman         UCI 43  — breast-cancer survival
  uci_mi_lethal        UCI 579 — MI cohort, in-hospital lethal outcome
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
SEED = 20261008


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "jev_prediction_testing/clinical-ext2"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def download(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {"url": url, "path": str(dest.relative_to(ROOT)), "sha256": sha256(dest),
                "bytes": dest.stat().st_size, "retrieved": "cached"}
    dest.write_bytes(fetch(url))
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


def numeric_pred(col: str, label: str, unit: str | None = None) -> dict:
    d = {"column": col, "kind": "numeric", "text_label": label}
    if unit:
        d["unit"] = unit
    return d


def cat_pred(col: str, label: str, values: dict | None = None) -> dict:
    d = {"column": col, "kind": "categorical", "text_label": label}
    if values:
        d["text_values"] = values
    return d


def norm_code(value) -> str | None:
    """Match jevtest.text.normalize_code so text_values keys pass the quality gate."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    try:
        number = float(text)
        if np.isfinite(number) and number == int(number):
            return str(int(number))
        if np.isfinite(number):
            return format(np.format_float_positional(number, trim="0"))
    except ValueError:
        pass
    return text


def codes_map(series: pd.Series, overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    out = {}
    for value in series:
        code = norm_code(value)
        if code is None:
            continue
        out[code] = overrides.get(code, overrides.get(value, code))
    return out


def as_code_series(series: pd.Series) -> pd.Series:
    return series.map(norm_code).fillna("")


def binary_outcome(column: str, classes: list[str], question: str) -> dict:
    return {
        column: {
            "column": column,
            "classes": classes,
            "mapping": {"0": classes[0], "1": classes[1]},
            "question": question,
        }
    }


# ---------------------------------------------------------------------------
# Individual preparers
# ---------------------------------------------------------------------------

def prepare_breast_coimbra(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"coimbra_{i+1:03d}" for i in range(len(src))]})
    cols = [
        ("Age", "age", "Age", "years"),
        ("BMI", "bmi", "Body mass index", "kg/m^2"),
        ("Glucose", "glucose", "Glucose", "mg/dL"),
        ("Insulin", "insulin", "Insulin", "µU/mL"),
        ("HOMA", "homa", "HOMA index", None),
        ("Leptin", "leptin", "Leptin", "ng/mL"),
        ("Adiponectin", "adiponectin", "Adiponectin", "µg/mL"),
        ("Resistin", "resistin", "Resistin", "ng/mL"),
        ("MCP.1", "mcp1", "MCP-1", "pg/dL"),
    ]
    predictors = {}
    for src_col, col, label, unit in cols:
        frame[col] = blank(src[src_col])
        predictors[col] = numeric_pred(col, label, unit)
    # Classification: 1 = healthy, 2 = patient
    frame["breast_cancer"] = src["Classification"].map(lambda v: 1 if int(v) == 2 else 0)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "breast_cancer",
        "class_counts": {
            "Healthy control": int((frame["breast_cancer"] == 0).sum()),
            "Breast cancer": int((frame["breast_cancer"] == 1).sum()),
        },
        "identifier_note": "UCI Breast Cancer Coimbra (blood biomarkers).",
    }
    spec = {
        "name": "uci_breast_coimbra",
        "description": "UCI Breast Cancer Coimbra (116 women, 9 blood biomarkers). Binary cancer vs control.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Outpatient oncology screening using anthropometrics and fasting blood biomarkers.",
        "observations_header": "Anthropometrics and fasting blood biomarkers:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "breast_cancer", ["Healthy control", "Breast cancer"],
            "Based on the anthropometrics and fasting blood biomarkers, does the patient have breast cancer?",
        ),
    }
    write_bundle("uci_breast_coimbra", frame, spec, summary, retrieval)


def prepare_early_diabetes(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    src.columns = [c.strip() for c in src.columns]
    frame = pd.DataFrame({"record_id": [f"ediab_{i+1:03d}" for i in range(len(src))]})
    frame["age"] = blank(src["age"])
    frame["gender"] = blank(src["gender"].astype(str).str.strip())
    predictors = {
        "age": numeric_pred("age", "Age", "years"),
        "gender": cat_pred("gender", "Sex", {"Male": "Male", "Female": "Female"}),
    }
    yes_no = {"Yes": "Yes", "No": "No"}
    for col in [c for c in src.columns if c not in ("age", "gender", "class")]:
        key = col.strip().lower().replace(" ", "_")
        frame[key] = blank(src[col].astype(str).str.strip())
        predictors[key] = cat_pred(key, col.replace("_", " ").title(), yes_no)
    frame["diabetes"] = src["class"].map(lambda v: 1 if str(v).strip().lower().startswith("pos") else 0)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "diabetes",
        "class_counts": {
            "No diabetes": int((frame["diabetes"] == 0).sum()),
            "Diabetes": int((frame["diabetes"] == 1).sum()),
        },
        "identifier_note": "UCI Early stage diabetes risk prediction (symptom questionnaire).",
    }
    spec = {
        "name": "uci_early_diabetes",
        "description": "UCI Early stage diabetes risk prediction (520 adults, symptom predictors).",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Primary-care screening for early diabetes using demographics and symptom checklist.",
        "observations_header": "Demographics and reported symptoms:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "diabetes", ["No diabetes", "Diabetes"],
            "Based on the demographics and reported symptoms, does the patient have diabetes?",
        ),
    }
    write_bundle("uci_early_diabetes", frame, spec, summary, retrieval)


def prepare_spectf(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"spectf_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    for col in [c for c in src.columns if c != "diagnosis"]:
        frame[col.lower()] = blank(src[col])
        predictors[col.lower()] = numeric_pred(col.lower(), col)
    frame["abnormal"] = src["diagnosis"].map(lambda v: int(v))
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "abnormal",
        "class_counts": {
            "Normal SPECT": int((frame["abnormal"] == 0).sum()),
            "Abnormal SPECT": int((frame["abnormal"] == 1).sum()),
        },
        "identifier_note": "UCI SPECTF Heart; diagnosis 0=normal, 1=abnormal.",
    }
    spec = {
        "name": "uci_spectf",
        "description": "UCI SPECTF Heart (267 patients, 44 SPECT features). Binary normal vs abnormal.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Nuclear cardiology using cardiac SPECT image features to detect abnormality.",
        "observations_header": "Cardiac SPECT image features:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "abnormal", ["Normal SPECT", "Abnormal SPECT"],
            "Based on the cardiac SPECT image features, is the study abnormal?",
        ),
    }
    write_bundle("uci_spectf", frame, spec, summary, retrieval)


def prepare_hepatitis(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    # Class: 1 = die, 2 = live  →  lethal binary
    # UCI binary clinical flags: 1 = no, 2 = yes; Sex: 1 = male, 2 = female.
    frame = pd.DataFrame({"record_id": [f"hep_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    yes_no = {"1": "No", "2": "Yes"}
    sex_map = {"1": "Male", "2": "Female"}
    for col in [c for c in src.columns if c != "Class"]:
        key = col.strip().lower().replace(" ", "_")
        series = src[col]
        # UCI uses ? for missing in some dumps; already NaN in data.csv
        if pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > 5:
            frame[key] = blank(series)
            predictors[key] = numeric_pred(key, col)
        else:
            coded = as_code_series(series)
            frame[key] = blank(coded.replace("", pd.NA))
            overrides = sex_map if key == "sex" else yes_no
            predictors[key] = cat_pred(key, col, codes_map(series, overrides))
    frame["died"] = src["Class"].map(lambda v: 1 if int(v) == 1 else 0)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "died",
        "class_counts": {
            "Lived": int((frame["died"] == 0).sum()),
            "Died": int((frame["died"] == 1).sum()),
        },
        "identifier_note": "UCI Hepatitis; Class 1=die, 2=live.",
    }
    spec = {
        "name": "uci_hepatitis",
        "description": "UCI Hepatitis (155 patients). Binary survival (die vs live).",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Hepatology cohort with clinical and laboratory findings predicting survival.",
        "observations_header": "Hepatitis clinical and laboratory findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "died", ["Lived", "Died"],
            "Based on the hepatitis clinical and laboratory findings, did the patient die?",
        ),
    }
    write_bundle("uci_hepatitis", frame, spec, summary, retrieval)


def prepare_obesity(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"obese_{i+1:04d}" for i in range(len(src))]})
    predictors = {}
    label_col = "NObeyesdad"
    class_order = [
        "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II",
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
    ]
    pretty = {
        "Insufficient_Weight": "Insufficient weight",
        "Normal_Weight": "Normal weight",
        "Overweight_Level_I": "Overweight level I",
        "Overweight_Level_II": "Overweight level II",
        "Obesity_Type_I": "Obesity type I",
        "Obesity_Type_II": "Obesity type II",
        "Obesity_Type_III": "Obesity type III",
    }
    cat_pretty = {
        "gender": {"Female": "Female", "Male": "Male"},
        "family_history_with_overweight": {"yes": "Yes", "no": "No"},
        "favc": {"yes": "Yes", "no": "No"},
        "caec": {"no": "No", "Sometimes": "Sometimes", "Frequently": "Frequently", "Always": "Always"},
        "smoke": {"yes": "Yes", "no": "No"},
        "scc": {"yes": "Yes", "no": "No"},
        "calc": {"no": "No", "Sometimes": "Sometimes", "Frequently": "Frequently", "Always": "Always"},
        "mtrans": {
            "Automobile": "Automobile", "Bike": "Bike", "Motorbike": "Motorbike",
            "Public_Transportation": "Public transportation", "Walking": "Walking",
        },
    }
    for col in [c for c in src.columns if c != label_col]:
        key = col.lower()
        series = src[col]
        if pd.api.types.is_numeric_dtype(series):
            frame[key] = blank(series)
            predictors[key] = numeric_pred(key, col.replace("_", " "))
        else:
            frame[key] = blank(series.astype(str))
            predictors[key] = cat_pred(
                key, col.replace("_", " "), codes_map(series, cat_pretty.get(key))
            )
    frame["obesity_class"] = src[label_col].astype(str)
    # Keep only known classes
    frame = frame[frame["obesity_class"].isin(class_order)].reset_index(drop=True)
    frame["record_id"] = [f"obese_{i+1:04d}" for i in range(len(frame))]
    classes = [pretty[c] for c in class_order]
    mapping = {c: pretty[c] for c in class_order}
    counts = {pretty[c]: int((frame["obesity_class"] == c).sum()) for c in class_order}
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "obesity_class",
        "class_counts": counts,
        "identifier_note": "UCI Estimation of obesity levels based on eating habits and physical condition.",
    }
    spec = {
        "name": "uci_obesity",
        "description": "UCI Obesity levels (2111 adults, lifestyle predictors). 7-class WHO-style categories.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Lifestyle and anthropometric assessment of obesity level.",
        "observations_header": "Demographics, eating habits and physical condition:",
        "predictors": predictors,
        "outcomes": {
            "obesity_class": {
                "column": "obesity_class",
                "classes": classes,
                "mapping": mapping,
                "question": (
                    "Based on the demographics, eating habits and physical condition, "
                    "what is the patient's obesity level?"
                ),
            }
        },
    }
    write_bundle("uci_obesity", frame, spec, summary, retrieval)


def prepare_bone_marrow(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    # Drop survival_time (leakage with status)
    frame = pd.DataFrame({"record_id": [f"bmt_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    skip = {"survival_time", "survival_status"}
    yes_no = {"0": "No", "1": "Yes"}
    overrides = {
        "recipientgender": {"0": "Male", "1": "Female"},
        "stemcellsource": {"0": "Bone marrow", "1": "Peripheral blood"},
        "donorage35": {"0": "Donor age ≤35", "1": "Donor age >35"},
        "iiiv": yes_no,
        "gendermatch": {"0": "Gender mismatched", "1": "Gender matched"},
        "donorabo": {"-1": "O", "0": "A", "1": "B", "2": "AB"},
        "recipientabo": {"-1": "O", "0": "A", "1": "B", "2": "AB"},
        "recipientrh": {"0": "Rh−", "1": "Rh+"},
        "abomatch": {"0": "ABO mismatched", "1": "ABO matched"},
        "cmvstatus": {
            "0": "Donor−/recipient−", "1": "Donor−/recipient+",
            "2": "Donor+/recipient−", "3": "Donor+/recipient+",
        },
        "donorcmv": {"0": "CMV−", "1": "CMV+"},
        "recipientcmv": {"0": "CMV−", "1": "CMV+"},
        "disease": {
            "ALL": "ALL", "AML": "AML", "chronic": "Chronic",
            "lymphoma": "Lymphoma", "nonmalignant": "Non-malignant",
        },
        "riskgroup": {"0": "Low risk", "1": "High risk"},
        "txpostrelapse": yes_no,
        "diseasegroup": {"0": "Non-malignant", "1": "Malignant"},
        "hlamatch": {"0": "10/10", "1": "9/10", "2": "8/10", "3": "7/10"},
        "hlamismatch": {"0": "Matched", "1": "Mismatched"},
        "antigen": {"-1": "None", "0": "0", "1": "1", "2": "2"},
        "allele": {"-1": "None", "0": "0", "1": "1", "2": "2", "3": "3"},
        "hlagri": {str(i): str(i) for i in range(8)},
        "recipientage10": {"0": "<10 years", "1": "≥10 years"},
        "recipientageint": {"0": "0–5 years", "1": "5–10 years", "2": "≥10 years"},
        "relapse": yes_no,
        "agvhdiiiiv": yes_no,
        "extcgvhd": yes_no,
    }
    for col in src.columns:
        if col in skip:
            continue
        key = col.lower()
        series = src[col]
        if pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > 8:
            frame[key] = blank(series)
            predictors[key] = numeric_pred(key, col)
        else:
            coded = as_code_series(series)
            # Keep non-numeric disease labels as-is
            if key == "disease":
                frame[key] = blank(series.astype(str).where(series.notna(), None))
            else:
                frame[key] = blank(coded.replace("", pd.NA))
            predictors[key] = cat_pred(key, col, codes_map(series, overrides.get(key)))
    frame["bmt_died"] = src["survival_status"].map(lambda v: int(v))
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "bmt_died",
        "excluded_predictors": ["survival_time"],
        "class_counts": {
            "Alive": int((frame["bmt_died"] == 0).sum()),
            "Dead": int((frame["bmt_died"] == 1).sum()),
        },
        "identifier_note": "UCI Bone marrow transplant in children; survival_time excluded.",
    }
    spec = {
        "name": "uci_bone_marrow",
        "description": "UCI Bone marrow transplant children (187 patients). Binary survival status.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Pediatric bone-marrow transplant cohort predicting post-transplant survival.",
        "observations_header": "Donor, recipient and transplant characteristics:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "bmt_died", ["Alive", "Dead"],
            "Based on the donor, recipient and transplant characteristics, did the patient die?",
        ),
    }
    write_bundle("uci_bone_marrow", frame, spec, summary, retrieval)


def prepare_parkinsons(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"park_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    for col in [c for c in src.columns if c not in ("name", "status")]:
        key = col.lower().replace(":", "_").replace("(", "").replace(")", "").replace("%", "pct")
        frame[key] = blank(src[col])
        predictors[key] = numeric_pred(key, col)
    frame["parkinsons"] = src["status"].map(lambda v: int(v))
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "parkinsons",
        "class_counts": {
            "Healthy": int((frame["parkinsons"] == 0).sum()),
            "Parkinson disease": int((frame["parkinsons"] == 1).sum()),
        },
        "identifier_note": "UCI Parkinson's; voice measures, status 1=PD.",
    }
    spec = {
        "name": "uci_parkinsons",
        "description": "UCI Parkinson's (195 voice recordings). Binary PD vs healthy.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Neurology voice analysis for Parkinson's disease screening.",
        "observations_header": "Biomedical voice measures:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "parkinsons", ["Healthy", "Parkinson disease"],
            "Based on the biomedical voice measures, does the subject have Parkinson disease?",
        ),
    }
    write_bundle("uci_parkinsons", frame, spec, summary, retrieval)


def prepare_haberman(raw_zip: Path, retrieval: list) -> None:
    with zipfile.ZipFile(raw_zip) as zf:
        name = [n for n in zf.namelist() if n.endswith((".data", ".dat", ".csv"))][0]
        src = pd.read_csv(zf.open(name), header=None,
                          names=["age", "op_year", "nodes", "survival"])
    frame = pd.DataFrame({"record_id": [f"hab_{i+1:03d}" for i in range(len(src))]})
    frame["age"] = blank(src["age"])
    frame["op_year"] = blank(src["op_year"])
    frame["nodes"] = blank(src["nodes"])
    # 1 = survived ≥5y, 2 = died within 5y
    frame["died_5y"] = src["survival"].map(lambda v: 1 if int(v) == 2 else 0)
    predictors = {
        "age": numeric_pred("age", "Age at operation", "years"),
        "op_year": numeric_pred("op_year", "Year of operation", "years (1900+)"),
        "nodes": numeric_pred("nodes", "Positive axillary nodes", "count"),
    }
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "died_5y",
        "class_counts": {
            "Survived ≥5 years": int((frame["died_5y"] == 0).sum()),
            "Died within 5 years": int((frame["died_5y"] == 1).sum()),
        },
        "identifier_note": "UCI Haberman survival; status 1=survived 5y, 2=died within 5y.",
    }
    spec = {
        "name": "uci_haberman",
        "description": "UCI Haberman breast-cancer survival (306 patients, 3 predictors).",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Surgical oncology follow-up after breast-cancer surgery.",
        "observations_header": "Surgical oncology findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "died_5y", ["Survived ≥5 years", "Died within 5 years"],
            "Based on the surgical oncology findings, did the patient die within five years?",
        ),
    }
    write_bundle("uci_haberman", frame, spec, summary, retrieval)


def prepare_mi_lethal(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    outcome_start = list(src.columns).index("FIBR_PREDS")
    pred_cols = [c for c in src.columns[:outcome_start] if c != "ID"]
    frame = pd.DataFrame({"record_id": [f"mi_{i+1:04d}" for i in range(len(src))]})
    predictors = {}
    yes_no = {"0": "No", "1": "Yes"}
    known = {
        "sex": {"0": "Female", "1": "Male"},
        "inf_anam": {"0": "0 prior MI", "1": "1 prior MI", "2": "2 prior MI", "3": "≥3 prior MI"},
        "stenok_an": {
            "0": "No angina", "1": "Exertional", "2": "Exertional + rest",
            "3": "Rest only", "4": "Variant", "5": "Other", "6": "Unknown form",
        },
        "fk_stenok": {
            "0": "No angina", "1": "CCS I", "2": "CCS II", "3": "CCS III", "4": "CCS IV",
        },
        "ibs_post": {"0": "None", "1": "Exertional angina", "2": "Unstable angina"},
        "gb": {"0": "No hypertension", "1": "Stage 1", "2": "Stage 2", "3": "Stage 3"},
    }
    for col in pred_cols:
        key = col.lower()
        series = src[col]
        if pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > 8:
            frame[key] = blank(series)
            predictors[key] = numeric_pred(key, col)
        else:
            coded = as_code_series(series)
            frame[key] = blank(coded.replace("", pd.NA))
            observed = codes_map(series)
            # Prefer known labels; binary 0/1 → No/Yes; otherwise keep code as readable token
            overrides = dict(known.get(key, {}))
            if not overrides and set(observed) <= {"0", "1"}:
                overrides = yes_no
            for code in observed:
                overrides.setdefault(code, code)
            predictors[key] = cat_pred(key, col, codes_map(series, overrides))
    # LET_IS == 0 → alive; any other code → lethal outcome class
    frame["lethal"] = src["LET_IS"].map(lambda v: 0 if int(v) == 0 else 1)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "lethal",
        "excluded_predictors": list(src.columns[outcome_start:]),
        "class_counts": {
            "Non-lethal course": int((frame["lethal"] == 0).sum()),
            "Lethal outcome": int((frame["lethal"] == 1).sum()),
        },
        "identifier_note": (
            "UCI Myocardial infarction complications; predictors = admission features only; "
            "target lethal if LET_IS ≠ 0; other complication labels excluded."
        ),
    }
    spec = {
        "name": "uci_mi_lethal",
        "description": (
            "UCI Myocardial infarction complications (1700 patients). Binary lethal outcome "
            "from admission-time features only."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Coronary-care admission assessment of myocardial infarction lethality risk.",
        "observations_header": "Admission clinical, ECG and laboratory findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "lethal", ["Non-lethal course", "Lethal outcome"],
            "Based on the admission clinical, ECG and laboratory findings, did the myocardial "
            "infarction follow a lethal course?",
        ),
    }
    write_bundle("uci_mi_lethal", frame, spec, summary, retrieval)


JOBS = [
    ("uci_breast_coimbra", "https://archive.ics.uci.edu/static/public/451/data.csv",
     "data.csv", prepare_breast_coimbra),
    ("uci_early_diabetes", "https://archive.ics.uci.edu/static/public/529/data.csv",
     "data.csv", prepare_early_diabetes),
    ("uci_spectf", "https://archive.ics.uci.edu/static/public/96/data.csv",
     "data.csv", prepare_spectf),
    ("uci_hepatitis", "https://archive.ics.uci.edu/static/public/46/data.csv",
     "data.csv", prepare_hepatitis),
    ("uci_obesity", "https://archive.ics.uci.edu/static/public/544/data.csv",
     "data.csv", prepare_obesity),
    ("uci_bone_marrow", "https://archive.ics.uci.edu/static/public/565/data.csv",
     "data.csv", prepare_bone_marrow),
    ("uci_parkinsons",
     "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
     "parkinsons.data", prepare_parkinsons),
    ("uci_haberman", "https://archive.ics.uci.edu/static/public/43/haberman+s+survival.zip",
     "haberman.zip", prepare_haberman),
    ("uci_mi_lethal", "https://archive.ics.uci.edu/static/public/579/data.csv",
     "data.csv", prepare_mi_lethal),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Optional subset of dataset names")
    args = parser.parse_args()
    wanted = set(args.only) if args.only else None
    for name, url, filename, prepare in JOBS:
        if wanted and name not in wanted:
            continue
        raw = DATASETS / name / "raw" / filename
        retrieval = [download(url, raw)]
        prepare(raw, retrieval)


if __name__ == "__main__":
    main()
