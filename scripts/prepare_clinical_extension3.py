#!/usr/bin/env python3
"""Prepare twelve more healthcare classification datasets (UCI, no Kaggle).

  uci_maternal_risk     UCI 863 — maternal health risk (3-class)
  uci_thyroid_recur     UCI 915 — differentiated thyroid cancer recurrence
  uci_cirrhosis         UCI 878 — cirrhosis death (exclude follow-up days)
  uci_support2          UCI 880 — SUPPORT2 in-hospital death (subsample)
  uci_aids_cid          UCI 890 — AIDS clinical trial AIDS-defining event
  uci_glioma            UCI 759 — glioma grade (LGG vs GBM)
  uci_retinopathy       UCI 329 — diabetic retinopathy presence
  uci_cardiotocography  UCI 193 — fetal NSP status (3-class)
  uci_thoracic          UCI 277 — thoracic surgery 1-year death risk
  uci_wbc_original      UCI 15  — Wisconsin breast cytology (original)
  uci_vertebral         UCI 212 — vertebral column pathology (3-class)
  uci_autism            UCI 426 — adult autism screening
  uci_breast_recurrence UCI 14  — Ljubljana breast recurrence
  uci_cdc_diabetes      UCI 891 — CDC diabetes indicators (stratified subsample)
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
SEED = 20261008


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "jev_prediction_testing/clinical-ext3"})
    with urllib.request.urlopen(req, timeout=300) as response:
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
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().strip("'\"")
    if not text or text.lower() in {"nan", "none", "nat", "?"}:
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
        out[code] = overrides.get(code, overrides.get(str(value).strip().strip("'\""), code))
    return out


def as_code_series(series: pd.Series) -> pd.Series:
    return series.map(norm_code)


def clean_str_series(series: pd.Series) -> pd.Series:
    """Strip CSV quoting artifacts so stored codes match normalize_code keys."""
    return series.map(lambda v: norm_code(v) if pd.notna(v) else None)


def binary_outcome(column: str, classes: list[str], question: str) -> dict:
    return {
        column: {
            "column": column,
            "classes": classes,
            "mapping": {"0": classes[0], "1": classes[1]},
            "question": question,
        }
    }


def multiclass_outcome(column: str, classes: list[str], mapping: dict, question: str) -> dict:
    return {column: {"column": column, "classes": classes, "mapping": mapping, "question": question}}


def add_numeric(frame, predictors, series, key, label, unit=None):
    frame[key] = blank(series)
    predictors[key] = numeric_pred(key, label, unit)


def add_cat(frame, predictors, series, key, label, overrides=None):
    coded = as_code_series(series)
    frame[key] = blank(coded)
    predictors[key] = cat_pred(key, label, codes_map(series, overrides))


# ---------------------------------------------------------------------------
# Preparers
# ---------------------------------------------------------------------------

def prepare_maternal(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"mat_{i+1:04d}" for i in range(len(src))]})
    predictors = {}
    for col, key, label, unit in [
        ("Age", "age", "Age", "years"),
        ("SystolicBP", "systolic_bp", "Systolic blood pressure", "mmHg"),
        ("DiastolicBP", "diastolic_bp", "Diastolic blood pressure", "mmHg"),
        ("BS", "blood_sugar", "Blood sugar", "mmol/L"),
        ("BodyTemp", "body_temp", "Body temperature", "°F"),
        ("HeartRate", "heart_rate", "Heart rate", "bpm"),
    ]:
        add_numeric(frame, predictors, src[col], key, label, unit)
    order = ["low risk", "mid risk", "high risk"]
    pretty = {"low risk": "Low risk", "mid risk": "Mid risk", "high risk": "High risk"}
    frame["risk_level"] = src["RiskLevel"].astype(str).str.strip()
    frame = frame[frame["risk_level"].isin(order)].reset_index(drop=True)
    frame["record_id"] = [f"mat_{i+1:04d}" for i in range(len(frame))]
    classes = [pretty[c] for c in order]
    mapping = {c: pretty[c] for c in order}
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "risk_level",
        "class_counts": {pretty[c]: int((frame["risk_level"] == c).sum()) for c in order},
        "identifier_note": "UCI Maternal Health Risk; vital signs → risk level.",
    }
    spec = {
        "name": "uci_maternal_risk",
        "description": "UCI Maternal Health Risk (1014 pregnancies, vital-sign predictors).",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Maternal health risk triage from routine vital signs.",
        "observations_header": "Maternal vital signs:",
        "predictors": predictors,
        "outcomes": multiclass_outcome(
            "risk_level", classes, mapping,
            "Based on the maternal vital signs, what is the pregnancy risk level?",
        ),
    }
    write_bundle("uci_maternal_risk", frame, spec, summary, retrieval)


def prepare_thyroid_recur(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"thyr_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    add_numeric(frame, predictors, src["Age"], "age", "Age", "years")
    yes_no = {"Yes": "Yes", "No": "No"}
    for col, key, label, ov in [
        ("Gender", "gender", "Sex", {"F": "Female", "M": "Male"}),
        ("Smoking", "smoking", "Smoking", yes_no),
        ("Hx Smoking", "hx_smoking", "History of smoking", yes_no),
        ("Hx Radiothreapy", "hx_radiotherapy", "History of radiotherapy", yes_no),
        ("Thyroid Function", "thyroid_function", "Thyroid function", None),
        ("Physical Examination", "physical_exam", "Physical examination", None),
        ("Adenopathy", "adenopathy", "Adenopathy", None),
        ("Pathology", "pathology", "Pathology", None),
        ("Focality", "focality", "Focality", None),
        ("Risk", "ata_risk", "ATA risk", None),
        ("T", "t_stage", "T stage", None),
        ("N", "n_stage", "N stage", None),
        ("M", "m_stage", "M stage", None),
        ("Stage", "stage", "AJCC stage", None),
        ("Response", "response", "Treatment response", None),
    ]:
        add_cat(frame, predictors, src[col], key, label, ov)
    frame["recurred"] = src["Recurred"].map(lambda v: 1 if str(v).strip().lower() == "yes" else 0)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "recurred",
        "class_counts": {
            "No recurrence": int((frame["recurred"] == 0).sum()),
            "Recurred": int((frame["recurred"] == 1).sum()),
        },
        "identifier_note": "UCI Differentiated Thyroid Cancer Recurrence.",
    }
    spec = {
        "name": "uci_thyroid_recur",
        "description": "UCI Differentiated thyroid cancer recurrence (383 patients).",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Thyroid oncology follow-up predicting cancer recurrence.",
        "observations_header": "Thyroid cancer clinical and staging findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "recurred", ["No recurrence", "Recurred"],
            "Based on the thyroid cancer clinical and staging findings, did the cancer recur?",
        ),
    }
    write_bundle("uci_thyroid_recur", frame, spec, summary, retrieval)


def prepare_cirrhosis(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    # N_Days is follow-up time (leakage with Status); Stage may be known at baseline — keep.
    frame = pd.DataFrame({"record_id": [f"cirr_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    add_numeric(frame, predictors, src["Age"] / 365.25, "age", "Age", "years")
    for col, key, label, ov in [
        ("Drug", "drug", "Drug", None),
        ("Sex", "sex", "Sex", {"M": "Male", "F": "Female"}),
        ("Ascites", "ascites", "Ascites", {"Y": "Yes", "N": "No"}),
        ("Hepatomegaly", "hepatomegaly", "Hepatomegaly", {"Y": "Yes", "N": "No"}),
        ("Spiders", "spiders", "Spiders", {"Y": "Yes", "N": "No"}),
        ("Edema", "edema", "Edema", {"N": "No", "S": "Present without diuretics", "Y": "Present with diuretics"}),
    ]:
        add_cat(frame, predictors, src[col], key, label, ov)
    for col, key, label, unit in [
        ("Bilirubin", "bilirubin", "Bilirubin", "mg/dL"),
        ("Cholesterol", "cholesterol", "Cholesterol", "mg/dL"),
        ("Albumin", "albumin", "Albumin", "g/dL"),
        ("Copper", "copper", "Urine copper", "µg/day"),
        ("Alk_Phos", "alk_phos", "Alkaline phosphatase", "U/L"),
        ("SGOT", "sgot", "SGOT", "U/L"),
        ("Tryglicerides", "triglycerides", "Triglycerides", "mg/dL"),
        ("Platelets", "platelets", "Platelets", "per mL / 1000"),
        ("Prothrombin", "prothrombin", "Prothrombin time", "seconds"),
        ("Stage", "stage", "Histologic stage", None),
    ]:
        add_numeric(frame, predictors, src[col], key, label, unit)
    # Binary death vs censored/alive (C or CL)
    frame["cirrhosis_died"] = src["Status"].map(lambda v: 1 if str(v).strip() == "D" else 0)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "cirrhosis_died",
        "excluded_predictors": ["N_Days"],
        "class_counts": {
            "Alive / censored": int((frame["cirrhosis_died"] == 0).sum()),
            "Died": int((frame["cirrhosis_died"] == 1).sum()),
        },
        "identifier_note": "UCI Cirrhosis; Status D=death; N_Days excluded.",
    }
    spec = {
        "name": "uci_cirrhosis",
        "description": "UCI Cirrhosis patient survival (418 patients). Binary death vs censored.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Hepatology trial cohort predicting death from baseline cirrhosis features.",
        "observations_header": "Baseline cirrhosis clinical and laboratory findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "cirrhosis_died", ["Alive / censored", "Died"],
            "Based on the baseline cirrhosis clinical and laboratory findings, did the patient die?",
        ),
    }
    write_bundle("uci_cirrhosis", frame, spec, summary, retrieval)


# Row caps for the zero-shot subsample; None keeps the whole table (scripts/prepare_full_tables.py).
SUPPORT2_N = 2500
CDC_N = 3000


def prepare_support2(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    # Drop leakage / outcome-adjacent columns
    drop = {
        "id", "death", "d.time", "slos", "charges", "totcst", "totmcst",
        "surv2m", "surv6m", "sfdm2", "hospdead",
    }
    rng = np.random.default_rng(SEED)
    # Stratified subsample for LLM-tractable size
    target_n = SUPPORT2_N
    if target_n is not None and len(src) > target_n:
        parts = []
        for label, group in src.groupby("hospdead"):
            n = int(round(target_n * len(group) / len(src)))
            n = min(max(n, 1), len(group))
            idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
            parts.append(src.loc[idx])
        src = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    frame = pd.DataFrame({"record_id": [f"sup_{i+1:04d}" for i in range(len(src))]})
    predictors = {}
    for col in src.columns:
        if col in drop:
            continue
        key = col.lower().replace(".", "_").replace(" ", "_")
        series = src[col]
        if pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > 8:
            add_numeric(frame, predictors, series, key, col)
        else:
            yes_no = {"0": "No", "1": "Yes"}
            ov = {"male": "Male", "female": "Female"} if key == "sex" else None
            if ov is None and set(codes_map(series)) <= {"0", "1"}:
                ov = yes_no
            add_cat(frame, predictors, series, key, col, ov)
    frame["hosp_death"] = src["hospdead"].map(lambda v: int(v))
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "hosp_death",
        "excluded_predictors": sorted(drop - {"hospdead", "id"}),
        "class_counts": {
            "Discharged alive": int((frame["hosp_death"] == 0).sum()),
            "Died in hospital": int((frame["hosp_death"] == 1).sum()),
        },
        "identifier_note": f"UCI SUPPORT2; stratified subsample n={len(frame)}; survival times/costs excluded.",
    }
    spec = {
        "name": "uci_support2",
        "description": "UCI SUPPORT2 critically ill adults; in-hospital death (subsampled).",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "ICU / SUPPORT study predicting in-hospital death from admission physiology.",
        "observations_header": "Admission physiology and comorbidity findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "hosp_death", ["Discharged alive", "Died in hospital"],
            "Based on the admission physiology and comorbidity findings, did the patient die in hospital?",
        ),
    }
    write_bundle("uci_support2", frame, spec, summary, retrieval)


def prepare_aids(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"aids_{i+1:04d}" for i in range(len(src))]})
    predictors = {}
    # Drop time-to-event; keep baseline labs (cd40/cd80) not follow-up cd420/cd820
    skip = {"pidnum", "cid", "time", "cd420", "cd820", "offtrt"}
    yes_no = {"0": "No", "1": "Yes"}
    for col in src.columns:
        if col in skip:
            continue
        key = col.lower()
        series = src[col]
        if pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > 8:
            add_numeric(frame, predictors, series, key, col)
        else:
            add_cat(frame, predictors, series, key, col, yes_no if set(codes_map(series)) <= {"0", "1"} else None)
    frame["aids_event"] = src["cid"].map(lambda v: int(v))
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "aids_event",
        "excluded_predictors": sorted(skip - {"pidnum", "cid"}),
        "class_counts": {
            "No AIDS-defining event": int((frame["aids_event"] == 0).sum()),
            "AIDS-defining event": int((frame["aids_event"] == 1).sum()),
        },
        "identifier_note": "UCI AIDS Clinical Trials Group Study 175; cid endpoint; time excluded.",
    }
    spec = {
        "name": "uci_aids_cid",
        "description": "UCI AIDS CTG 175 (2139 patients). Binary AIDS-defining event.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "HIV clinical trial predicting AIDS-defining illness from baseline features.",
        "observations_header": "Baseline HIV trial clinical findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "aids_event", ["No AIDS-defining event", "AIDS-defining event"],
            "Based on the baseline HIV trial clinical findings, did an AIDS-defining event occur?",
        ),
    }
    write_bundle("uci_aids_cid", frame, spec, summary, retrieval)


def prepare_glioma(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"glio_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    skip = {"Case_ID", "Grade"}
    mut_map = {"0": "Wild type", "1": "Mutant"}
    for col in src.columns:
        if col in skip:
            continue
        key = col.lower()
        series = src[col]
        if col == "Age_at_diagnosis":
            add_numeric(frame, predictors, series, "age", "Age at diagnosis", "years")
        elif col == "Gender":
            add_cat(frame, predictors, series, "gender", "Sex", {"0": "Male", "1": "Female"})
        elif col == "Race":
            add_cat(frame, predictors, series, "race", "Race", None)
        else:
            add_cat(frame, predictors, series, key, col, mut_map)
    frame["gbm"] = src["Grade"].map(lambda v: int(v))
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "gbm",
        "class_counts": {
            "Lower-grade glioma": int((frame["gbm"] == 0).sum()),
            "Glioblastoma": int((frame["gbm"] == 1).sum()),
        },
        "identifier_note": "UCI Glioma grading clinical and mutation features; Grade 0=LGG, 1=GBM.",
    }
    spec = {
        "name": "uci_glioma",
        "description": "UCI Glioma grading (839 cases). Binary LGG vs GBM from clinical+mutation features.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Neuro-oncology grading of glioma from demographics and mutation panel.",
        "observations_header": "Demographics and tumor mutation findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "gbm", ["Lower-grade glioma", "Glioblastoma"],
            "Based on the demographics and tumor mutation findings, is the glioma a glioblastoma?",
        ),
    }
    write_bundle("uci_glioma", frame, spec, summary, retrieval)


def prepare_retinopathy(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"ret_{i+1:04d}" for i in range(len(src))]})
    predictors = {}
    for col in [c for c in src.columns if c != "Class"]:
        key = col.lower().replace(".", "_")
        series = src[col]
        if series.dropna().nunique() <= 3:
            add_cat(frame, predictors, series, key, col, {"0": "No", "1": "Yes"} if set(codes_map(series)) <= {"0", "1"} else None)
        else:
            add_numeric(frame, predictors, series, key, col)
    frame["retinopathy"] = src["Class"].map(lambda v: int(v))
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "retinopathy",
        "class_counts": {
            "No retinopathy": int((frame["retinopathy"] == 0).sum()),
            "Retinopathy": int((frame["retinopathy"] == 1).sum()),
        },
        "identifier_note": "UCI Diabetic Retinopathy Debrecen; image-derived features.",
    }
    spec = {
        "name": "uci_retinopathy",
        "description": "UCI Diabetic Retinopathy Debrecen (1151 eyes). Binary retinopathy presence.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Ophthalmology screening for diabetic retinopathy from retinal image features.",
        "observations_header": "Retinal image-derived features:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "retinopathy", ["No retinopathy", "Retinopathy"],
            "Based on the retinal image-derived features, is diabetic retinopathy present?",
        ),
    }
    write_bundle("uci_retinopathy", frame, spec, summary, retrieval)


def prepare_cardiotocography(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"ctg_{i+1:04d}" for i in range(len(src))]})
    predictors = {}
    # CLASS is morphologic pattern related to NSP — exclude to avoid leakage
    for col in [c for c in src.columns if c not in ("NSP", "CLASS")]:
        key = col.lower()
        add_numeric(frame, predictors, src[col], key, col)
    order = [1, 2, 3]
    pretty = {1: "Normal", 2: "Suspect", 3: "Pathologic"}
    frame["nsp"] = src["NSP"].map(int)
    classes = [pretty[i] for i in order]
    mapping = {str(i): pretty[i] for i in order}
    # Store class labels as strings matching mapping keys used by loader — use int codes with mapping
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "nsp",
        "excluded_predictors": ["CLASS"],
        "class_counts": {pretty[i]: int((frame["nsp"] == i).sum()) for i in order},
        "identifier_note": "UCI Cardiotocography; NSP endpoint; CLASS morphologic code excluded.",
    }
    spec = {
        "name": "uci_cardiotocography",
        "description": "UCI Cardiotocography (2126 CTGs). 3-class fetal status (NSP).",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Obstetric CTG interpretation of fetal well-being.",
        "observations_header": "Cardiotocography measurements:",
        "predictors": predictors,
        "outcomes": {
            "nsp": {
                "column": "nsp",
                "classes": classes,
                "mapping": mapping,
                "question": "Based on the cardiotocography measurements, what is the fetal status (NSP)?",
            }
        },
    }
    write_bundle("uci_cardiotocography", frame, spec, summary, retrieval)


def prepare_thoracic(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"thor_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    tf = {"T": "Yes", "F": "No", "t": "Yes", "f": "No"}
    for col in [c for c in src.columns if c != "Risk1Yr"]:
        key = col.lower()
        series = src[col]
        if pd.api.types.is_numeric_dtype(series):
            add_numeric(frame, predictors, series, key, col)
        else:
            add_cat(frame, predictors, series, key, col, tf)
    frame["died_1y"] = src["Risk1Yr"].map(lambda v: 1 if str(v).strip().upper() == "T" else 0)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "died_1y",
        "class_counts": {
            "Alive at 1 year": int((frame["died_1y"] == 0).sum()),
            "Died within 1 year": int((frame["died_1y"] == 1).sum()),
        },
        "identifier_note": "UCI Thoracic Surgery Data; 1-year mortality.",
    }
    spec = {
        "name": "uci_thoracic",
        "description": "UCI Thoracic surgery (470 patients). Binary 1-year mortality.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Thoracic surgery risk prediction for 1-year death.",
        "observations_header": "Preoperative thoracic surgery findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "died_1y", ["Alive at 1 year", "Died within 1 year"],
            "Based on the preoperative thoracic surgery findings, did the patient die within one year?",
        ),
    }
    write_bundle("uci_thoracic", frame, spec, summary, retrieval)


def prepare_wbc_original(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"wbc_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    skip = {"Sample_code_number", "Class"}
    for col in src.columns:
        if col in skip:
            continue
        key = col.lower()
        # Bare nuclei may be '?' 
        series = pd.to_numeric(src[col], errors="coerce")
        add_numeric(frame, predictors, series, key, col.replace("_", " "))
    frame["malignant"] = src["Class"].map(lambda v: 1 if int(v) == 4 else 0)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "malignant",
        "class_counts": {
            "Benign": int((frame["malignant"] == 0).sum()),
            "Malignant": int((frame["malignant"] == 1).sum()),
        },
        "identifier_note": "UCI Breast Cancer Wisconsin (Original); Class 2=benign, 4=malignant.",
    }
    spec = {
        "name": "uci_wbc_original",
        "description": "UCI Wisconsin Breast Cancer Original (699 samples, cytology grades).",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Breast cytology grading for malignancy.",
        "observations_header": "Fine-needle aspirate cytology grades:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "malignant", ["Benign", "Malignant"],
            "Based on the fine-needle aspirate cytology grades, is the mass malignant?",
        ),
    }
    write_bundle("uci_wbc_original", frame, spec, summary, retrieval)


def prepare_vertebral(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"vert_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    for col in [c for c in src.columns if c != "class"]:
        add_numeric(frame, predictors, src[col], col.lower(), col.replace("_", " "))
    order = ["Normal", "Hernia", "Spondylolisthesis"]
    frame["pathology"] = src["class"].astype(str).str.strip()
    frame = frame[frame["pathology"].isin(order)].reset_index(drop=True)
    frame["record_id"] = [f"vert_{i+1:03d}" for i in range(len(frame))]
    mapping = {c: c for c in order}
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "pathology",
        "class_counts": {c: int((frame["pathology"] == c).sum()) for c in order},
        "identifier_note": "UCI Vertebral Column; biomechanical predictors.",
    }
    spec = {
        "name": "uci_vertebral",
        "description": "UCI Vertebral Column (310 patients). 3-class spinal pathology.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Orthopedic assessment of vertebral column pathology.",
        "observations_header": "Biomechanical spine measurements:",
        "predictors": predictors,
        "outcomes": multiclass_outcome(
            "pathology", order, mapping,
            "Based on the biomechanical spine measurements, what is the vertebral pathology?",
        ),
    }
    write_bundle("uci_vertebral", frame, spec, summary, retrieval)


def prepare_autism(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    # Raw dump wraps some strings in single quotes (e.g. 'Middle Eastern ').
    for col in src.select_dtypes(include=["object"]).columns:
        src[col] = clean_str_series(src[col])
    frame = pd.DataFrame({"record_id": [f"asd_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    # result is sum of A*_Score — leakage for screening score; age_desc redundant with age
    skip = {"class", "result", "age_desc"}
    yes_no = {"0": "No", "1": "Yes", "no": "No", "yes": "Yes", "NO": "No", "YES": "Yes"}
    for col in src.columns:
        if col in skip:
            continue
        key = col.lower().replace(" ", "_")
        series = src[col]
        if col.startswith("A") and col.endswith("_Score"):
            add_cat(frame, predictors, series, key, col.replace("_", " "), {"0": "No", "1": "Yes"})
        elif pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > 8:
            add_numeric(frame, predictors, series, key, col)
        else:
            pretty = None
            if key == "ethnicity":
                pretty = {c: c for c in codes_map(series)}
            add_cat(frame, predictors, series, key, col, pretty or yes_no)
    frame["asd"] = src["class"].map(lambda v: 1 if str(v).strip().upper() == "YES" else 0)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "asd",
        "excluded_predictors": ["result", "age_desc"],
        "class_counts": {
            "No ASD": int((frame["asd"] == 0).sum()),
            "ASD": int((frame["asd"] == 1).sum()),
        },
        "identifier_note": "UCI Autism Screening Adult; AQ-10 items; result sum excluded.",
    }
    spec = {
        "name": "uci_autism",
        "description": "UCI Autism Screening Adult (704 adults). Binary ASD classification.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Adult autism screening from AQ-10 questionnaire and demographics.",
        "observations_header": "AQ-10 answers and demographics:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "asd", ["No ASD", "ASD"],
            "Based on the AQ-10 answers and demographics, does the adult screen positive for ASD?",
        ),
    }
    write_bundle("uci_autism", frame, spec, summary, retrieval)


def prepare_breast_recurrence(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    frame = pd.DataFrame({"record_id": [f"brrec_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    for col in [c for c in src.columns if c != "Class"]:
        key = col.lower().replace("-", "_")
        add_cat(frame, predictors, src[col], key, col.replace("-", " "))
    frame["recurrence"] = src["Class"].map(
        lambda v: 1 if "recurrence-events" == str(v).strip() else 0
    )
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "recurrence",
        "class_counts": {
            "No recurrence": int((frame["recurrence"] == 0).sum()),
            "Recurrence": int((frame["recurrence"] == 1).sum()),
        },
        "identifier_note": "UCI Breast Cancer (Ljubljana); recurrence-events endpoint.",
    }
    spec = {
        "name": "uci_breast_recurrence",
        "description": "UCI Breast Cancer Ljubljana (286 patients). Binary recurrence.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Breast oncology predicting recurrence after primary treatment.",
        "observations_header": "Breast cancer clinical findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "recurrence", ["No recurrence", "Recurrence"],
            "Based on the breast cancer clinical findings, did recurrence events occur?",
        ),
    }
    write_bundle("uci_breast_recurrence", frame, spec, summary, retrieval)


def prepare_cdc_diabetes(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    rng = np.random.default_rng(SEED)
    target_n = CDC_N
    if target_n is not None:
        parts = []
        for label, group in src.groupby("Diabetes_binary"):
            n = int(round(target_n * len(group) / len(src)))
            n = min(max(n, 1), len(group))
            idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
            parts.append(src.loc[idx])
        src = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    frame = pd.DataFrame({"record_id": [f"cdc_{i+1:04d}" for i in range(len(src))]})
    predictors = {}
    yes_no = {"0": "No", "1": "Yes"}
    skip = {"ID", "Diabetes_binary"}
    for col in src.columns:
        if col in skip:
            continue
        key = col.lower()
        series = src[col]
        if col in {"BMI", "MentHlth", "PhysHlth"} or (pd.api.types.is_numeric_dtype(series) and series.nunique() > 8):
            add_numeric(frame, predictors, series, key, col)
        else:
            ov = {"0": "Female", "1": "Male"} if col == "Sex" else yes_no
            if set(codes_map(series)) - set(ov):
                ov = None  # fall back to identity for Age/Education/Income bands
            add_cat(frame, predictors, series, key, col, ov)
    frame["cdc_diabetes"] = src["Diabetes_binary"].map(lambda v: int(v))
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "cdc_diabetes",
        "class_counts": {
            "No diabetes": int((frame["cdc_diabetes"] == 0).sum()),
            "Diabetes": int((frame["cdc_diabetes"] == 1).sum()),
        },
        "identifier_note": f"UCI CDC Diabetes Health Indicators; stratified subsample n={len(frame)}.",
    }
    spec = {
        "name": "uci_cdc_diabetes",
        "description": "UCI CDC Diabetes Health Indicators (stratified subsample). Binary diabetes.",
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": "Population health survey predicting diabetes from lifestyle and health indicators.",
        "observations_header": "Survey health and lifestyle indicators:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "cdc_diabetes", ["No diabetes", "Diabetes"],
            "Based on the survey health and lifestyle indicators, does the respondent have diabetes?",
        ),
    }
    write_bundle("uci_cdc_diabetes", frame, spec, summary, retrieval)


JOBS = [
    ("uci_maternal_risk", "https://archive.ics.uci.edu/static/public/863/data.csv",
     "data.csv", prepare_maternal),
    ("uci_thyroid_recur", "https://archive.ics.uci.edu/static/public/915/data.csv",
     "data.csv", prepare_thyroid_recur),
    ("uci_cirrhosis", "https://archive.ics.uci.edu/static/public/878/data.csv",
     "data.csv", prepare_cirrhosis),
    ("uci_support2", "https://archive.ics.uci.edu/static/public/880/data.csv",
     "data.csv", prepare_support2),
    ("uci_aids_cid", "https://archive.ics.uci.edu/static/public/890/data.csv",
     "data.csv", prepare_aids),
    ("uci_glioma", "https://archive.ics.uci.edu/static/public/759/data.csv",
     "data.csv", prepare_glioma),
    ("uci_retinopathy", "https://archive.ics.uci.edu/static/public/329/data.csv",
     "data.csv", prepare_retinopathy),
    ("uci_cardiotocography", "https://archive.ics.uci.edu/static/public/193/data.csv",
     "data.csv", prepare_cardiotocography),
    ("uci_thoracic", "https://archive.ics.uci.edu/static/public/277/data.csv",
     "data.csv", prepare_thoracic),
    ("uci_wbc_original", "https://archive.ics.uci.edu/static/public/15/data.csv",
     "data.csv", prepare_wbc_original),
    ("uci_vertebral", "https://archive.ics.uci.edu/static/public/212/data.csv",
     "data.csv", prepare_vertebral),
    ("uci_autism", "https://archive.ics.uci.edu/static/public/426/data.csv",
     "data.csv", prepare_autism),
    ("uci_breast_recurrence", "https://archive.ics.uci.edu/static/public/14/data.csv",
     "data.csv", prepare_breast_recurrence),
    ("uci_cdc_diabetes", "https://archive.ics.uci.edu/static/public/891/data.csv",
     "data.csv", prepare_cdc_diabetes),
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
