#!/usr/bin/env python3
"""Prepare three non-UCI clinical classification tables (no Kaggle).

  ppd_bangladesh        Mendeley 4nznnrk8cg v3 — EPDS high-risk PPD (drop current screens)
  hip_fracture_elos     Zenodo 22866907 — extended LOS, honest T0 predictors only
  berlin_appendicitis   OSF 9wvys / Reismann 2019 — histopathology appendicitis

Raw files under datasets/<name>/raw/; patients.csv + dataset.yaml beside them.
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
USER_AGENT = "jev_prediction_testing/nonuci-ext"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
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
    excluded = summary.get("excluded_predictors") or []
    extra = f"- Excluded: {excluded}\n" if excluded else ""
    (out / "DATASET_NOTES.md").write_text(
        f"# {name}\n\n- Observations: {summary['observations']}\n"
        f"- Predictors: {summary['predictors']}\n- Outcome: {summary['outcome']}\n"
        f"- Class counts: {summary['class_counts']}\n"
        f"{extra}- Note: {summary.get('identifier_note', '')}\n",
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


def binary_outcome(column: str, classes: list[str], question: str) -> dict:
    return {
        column: {
            "column": column,
            "classes": classes,
            "mapping": {"0": classes[0], "1": classes[1]},
            "question": question,
        }
    }


def add_numeric(frame, predictors, series, key, label, unit=None):
    frame[key] = blank(series)
    predictors[key] = numeric_pred(key, label, unit)


def add_cat(frame, predictors, series, key, label, overrides=None):
    coded = as_code_series(series)
    frame[key] = blank(coded)
    predictors[key] = cat_pred(key, label, codes_map(series, overrides))


def tidy_title(value):
    code = norm_code(value)
    if code is None:
        return None
    return " ".join(part.capitalize() if part.islower() or part.isupper() else part for part in code.split())


# ---------------------------------------------------------------------------
# Preparers
# ---------------------------------------------------------------------------

def prepare_ppd(raw: Path, retrieval: list) -> None:
    src = pd.read_csv(raw)
    # Current EPDS/PHQ-9 items and totals are the screening construct used as Y.
    # Current mood/sleep/anger items are the same construct in other words.
    drop = {
        "sr",
        "Feeling about motherhood",
        "Worry about newborn",
        "Relax/sleep when newborn is tended ",
        "Relax/sleep when the newborn is asleep",
        "Angry after latest child birth",
        "Feeling for regular activities",
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself or that you are a failure or have let yourself or your family down",
        "rouble concentrating on things",
        "Moving or speaking or restlessness",
        "Thoughts that you would be better off dead, or of hurting yourself",
        "PHQ9 Score",
        "PHQ9 Result",
        "You have been able to laugh and see the funny side of things",
        "You have looked forward with enjoyment to things",
        "You have blamed myself unnecessarily when things went wrong",
        "You have been anxious or worried for no good reason",
        "You have felt scared or panicky for no good reason",
        "Things have been getting to you",
        "You have been so unhappy that you have had difficulty sleeping",
        "You have felt sad or miserable",
        "You have been so unhappy that you have been crying",
        "The thought of harming yourself has occurred ",
        "EPDS Score",
        "EPDS Result",
    }
    missing_drop = [c for c in drop if c not in src.columns]
    if missing_drop:
        raise SystemExit(f"ppd_bangladesh: expected source columns missing: {missing_drop}")

    frame = pd.DataFrame({"record_id": [f"ppd_{i+1:04d}" for i in range(len(src))]})
    predictors = {}

    education = src["Education Level"].map(lambda v: None if norm_code(v) is None else {
        "primary school": "Primary school",
        "high school": "High school",
        "college": "College",
        "university": "University",
    }.get(str(v).strip().lower(), tidy_title(v)))
    husband_edu = src["Husband's education level"].map(lambda v: None if norm_code(v) is None else {
        "primary school": "Primary school",
        "high school": "High school",
        "college": "College",
        "university": "University",
    }.get(str(v).strip().lower(), tidy_title(v)))
    occupation = {
        "house wife": "Housewife",
        "housewife": "Housewife",
        "student": "Student",
        "teacher": "Teacher",
        "service": "Service",
        "doctor": "Doctor",
        "business": "Business",
        "other": "Other",
    }
    occ_before = src["Occupation before latest pregnancy"].map(
        lambda v: None if norm_code(v) is None else occupation.get(str(v).strip().lower(), tidy_title(v)))
    occ_after = src["Occupation After Your Latest Childbirth"].map(
        lambda v: None if norm_code(v) is None else occupation.get(str(v).strip().lower(), tidy_title(v)))
    children = src["Total children"].map(lambda v: None if norm_code(v) is None else {
        "one": "One", "two": "Two", "more than two": "More than two",
    }.get(str(v).strip().lower(), tidy_title(v)))
    loss = src["History of pregnancy loss"].map(lambda v: None if norm_code(v) is None else {
        "miscarriage": "Miscarriage",
        "still-born delivery": "Stillbirth",
        "still-born Delivery": "Stillbirth",
    }.get(str(v).strip().lower(), tidy_title(v)))
    disease_before = src["Disease before pregnancy"].map(
        lambda v: None if norm_code(v) in {None, "1.2"} else {
            "chronic disease": "Chronic disease",
            "non-chronic disease": "Non-chronic disease",
        }.get(str(v).strip().lower(), tidy_title(v)))
    disease_during = src["Diseases during pregnancy"].map(
        lambda v: None if norm_code(v) is None else {
            "chronic disease": "Chronic disease",
            "non-chronic disease": "Non-chronic disease",
        }.get(str(v).strip().lower(), tidy_title(v)))
    inlaws = src["Relationship with the in-laws"].map(lambda v: None if norm_code(v) is None else {
        "good": "Good", "friendly": "Good", "neutral": "Neutral", "bad": "Poor", "poor": "Poor",
    }.get(str(v).strip().lower(), tidy_title(v)))
    husband = src["Relationship with husband"].map(lambda v: None if norm_code(v) is None else {
        "good": "Good", "friendly": "Good", "neutral": "Neutral", "bad": "Poor", "poor": "Poor",
    }.get(str(v).strip().lower(), tidy_title(v)))
    bond_nb = src["Relationship with the newborn"].map(lambda v: None if norm_code(v) is None else {
        "very good": "Very good", "good": "Good", "neutral": "Neutral", "bad": "Poor",
    }.get(str(v).strip().lower(), tidy_title(v)))
    bond_father = src["Relationship between father and newborn"].map(lambda v: None if norm_code(v) is None else {
        "very good": "Very good", "good": "Good", "neutral": "Neutral", "bad": "Poor",
    }.get(str(v).strip().lower(), tidy_title(v)))
    husband_income_col = [c for c in src.columns if c.startswith("Husband") and "monthly income" in c.lower()][0]

    add_numeric(frame, predictors, src["Age"], "age", "Age", "years")
    add_cat(frame, predictors, src["Residence"], "residence", "Residence")
    add_cat(frame, predictors, education, "education", "Education")
    add_cat(frame, predictors, src["Marital status"], "marital_status", "Marital status")
    add_cat(frame, predictors, occ_before, "occupation_before", "Occupation before latest pregnancy")
    add_cat(frame, predictors, src["Monthly income before latest pregnancy"],
            "income_before", "Monthly income before latest pregnancy",
            {"Less than 5000": "Less than 5000 BDT", "5000 to 10000": "5000 to 10000 BDT",
             "10000 to 20000": "10000 to 20000 BDT", "20000 to 30000": "20000 to 30000 BDT",
             "More than 30000": "More than 30000 BDT"})
    add_cat(frame, predictors, occ_after, "occupation_after", "Occupation after latest childbirth")
    add_cat(frame, predictors, src["Current monthly income"], "income_current", "Current monthly income",
            {"Less than 5000": "Less than 5000 BDT", "5000 to 10000": "5000 to 10000 BDT",
             "10000 to 20000": "10000 to 20000 BDT", "20000 to 30000": "20000 to 30000 BDT",
             "More than 30000": "More than 30000 BDT"})
    add_cat(frame, predictors, husband_edu, "husband_education", "Husband education")
    add_cat(frame, predictors, src[husband_income_col], "husband_income", "Husband monthly income",
            {"Less than 5000": "Less than 5000 BDT", "5000 to 10000": "5000 to 10000 BDT",
             "10000 to 20000": "10000 to 20000 BDT", "20000 to 30000": "20000 to 30000 BDT",
             "More than 30000": "More than 30000 BDT"})
    add_cat(frame, predictors, src["Addiction"], "addiction", "Addiction")
    add_cat(frame, predictors, children, "total_children", "Number of children")
    add_cat(frame, predictors, disease_before, "disease_before", "Disease before pregnancy")
    add_cat(frame, predictors, loss, "pregnancy_loss", "History of pregnancy loss")
    add_cat(frame, predictors, src["Family type"], "family_type", "Family type")
    add_cat(frame, predictors, src["Number of household members"], "household_members", "Household members")
    add_cat(frame, predictors, inlaws, "inlaw_relationship", "Relationship with in-laws")
    add_cat(frame, predictors, husband, "husband_relationship", "Relationship with husband")
    add_cat(frame, predictors, bond_nb, "newborn_bond", "Bond with the newborn")
    add_cat(frame, predictors, bond_father, "father_newborn_bond", "Father–newborn bond")
    add_cat(frame, predictors, src["Recieved Support"], "received_support", "Support received around childbirth")
    add_cat(frame, predictors, src["Need for Support"], "need_support", "Need for more support")
    add_cat(frame, predictors, src["Major changes or losses during pregnancy"],
            "major_changes", "Major change or loss during pregnancy")
    add_cat(frame, predictors, src["Abuse"], "abuse", "Abuse or mistreatment")
    add_cat(frame, predictors, src["Trust and share feelings"], "trust_share",
            "Can trust and share feelings with close others")
    add_numeric(frame, predictors, src["Number of the latest pregnancy"], "pregnancy_number",
                "Number of the latest pregnancy", "count")
    add_cat(frame, predictors, src["Pregnancy length"], "pregnancy_length", "Pregnancy length")
    add_cat(frame, predictors, src["Pregnancy plan"], "pregnancy_planned", "Latest pregnancy was planned")
    add_cat(frame, predictors, src["Regular checkups"], "regular_checkups", "Regular prenatal checkups")
    add_cat(frame, predictors, src["Fear of pregnancy"], "fear_of_pregnancy", "Fear of pregnancy")
    add_cat(frame, predictors, disease_during, "disease_during", "Disease during pregnancy")
    add_cat(frame, predictors, src["Age of newborn"], "newborn_age", "Age of the newborn")
    add_cat(frame, predictors, src["Age of immediate older children"], "older_child_age",
            "Age of the next older child")
    add_cat(frame, predictors, src["Mode of delivery"], "delivery_mode", "Mode of delivery")
    add_cat(frame, predictors, src["Gender of newborn"], "newborn_sex", "Sex of the newborn")
    add_cat(frame, predictors, src["Birth compliancy"], "birth_complications", "Birth complications")
    add_cat(frame, predictors, src["Breastfeed"], "breastfeed", "Currently breastfeeding")
    add_cat(frame, predictors, src["Newborn illness"], "newborn_illness", "Newborn illness")
    add_cat(frame, predictors, src["Depression before pregnancy (PHQ2)"],
            "phq2_before", "PHQ-2 depression screen before pregnancy")
    add_cat(frame, predictors, src["Depression during pregnancy (PHQ2)"],
            "phq2_during", "PHQ-2 depression screen during pregnancy")

    epds = src["EPDS Result"].astype(str).str.strip()
    if set(epds) - {"High", "Medium", "Low"}:
        raise SystemExit(f"ppd_bangladesh: unexpected EPDS Result values {sorted(set(epds))}")
    frame["ppd_high"] = (epds == "High").astype(int)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "ppd_high",
        "excluded_predictors": sorted(drop - {"sr", "EPDS Result"}),
        "class_counts": {
            "EPDS not high": int((frame["ppd_high"] == 0).sum()),
            "EPDS high risk": int((frame["ppd_high"] == 1).sum()),
        },
        "identifier_note": (
            "Mendeley 4nznnrk8cg v3 (Raisa & Kaiser). Y = EPDS Result High (score 13–30). "
            "Current EPDS/PHQ-9 items and postpartum mood/sleep items excluded."
        ),
    }
    spec = {
        "name": "ppd_bangladesh",
        "description": (
            "Bangladesh postpartum depression screening (800 women). Binary EPDS high-risk "
            "from sociodemographic, obstetric, and neonatal history."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Postpartum assessment of a woman within 24 months of a live birth in Bangladesh. "
            "The prediction uses sociodemographic, family, obstetric, and neonatal history "
            "available before Edinburgh Postnatal Depression Scale scoring."
        ),
        "observations_header": "Sociodemographic, obstetric, and neonatal history:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "ppd_high", ["EPDS not high", "EPDS high risk"],
            "Based on the sociodemographic, obstetric, and neonatal history, is the woman at high "
            "risk of postpartum depression on the Edinburgh Postnatal Depression Scale?",
        ),
    }
    write_bundle("ppd_bangladesh", frame, spec, summary, retrieval)


def prepare_hip(raw_csv: Path, retrieval: list) -> None:
    src = pd.read_csv(raw_csv)
    # Honest T0, de-duplicated (feature_sets.json S_honest_T0_deduped). tha/thx moved to T0
    # after clinical adjudication; time_admission_surgery is a mechanical LOS component.
    keep = [
        ("diagnosis", "fracture_type", "Fracture type",
         {"0": "Intracapsular", "1": "Extracapsular"}, "cat"),
        ("age", "age", "Age", None, "num_years"),
        ("gender", "sex", "Sex", {"0": "Female", "1": "Male"}, "cat"),
        ("heart_rate", "heart_rate", "Admission heart rate", None, "num_bpm"),
        ("pre_hlb", "pre_hb", "Preoperative haemoglobin", None, "num_gl"),
        ("pre_hct", "pre_hct", "Preoperative haematocrit", None, "num_pct"),
        ("pre_mchc", "pre_mchc", "Preoperative MCHC", None, "num_gl"),
        ("pre_platelet", "pre_platelet", "Preoperative platelets", None, "num_plt"),
        ("pre_alb", "pre_albumin", "Preoperative albumin", None, "num_gl"),
        ("pre_pt", "pre_pt", "Preoperative prothrombin time ratio", None, "num"),
        ("pre_aptt", "pre_aptt", "Preoperative APTT", None, "num_sec"),
        ("pre_ddimer", "pre_ddimer", "Preoperative D-dimer", None, "num_dd"),
        ("pre_ca", "pre_calcium", "Preoperative calcium", None, "num_ca"),
        ("pre_cr", "pre_creatinine", "Preoperative creatinine", None, "num_cr"),
        ("pre_liver_dis", "pre_liver", "Preoperative liver dysfunction",
         {"0": "No", "1": "Yes"}, "cat"),
        ("pre_renal_dis", "pre_renal", "Preoperative renal dysfunction",
         {"0": "No", "1": "Yes"}, "cat"),
        ("tha", "arthroplasty", "Total hip arthroplasty planned at admission",
         {"0": "No (internal fixation)", "1": "Yes"}, "cat"),
        ("thx", "tranexamic", "Protocolized tranexamic acid",
         {"0": "No", "1": "Yes"}, "cat"),
    ]
    missing = [c for c, *_ in keep if c not in src.columns]
    if missing:
        raise SystemExit(f"hip_fracture_elos: missing columns {missing}")

    frame = pd.DataFrame({"record_id": [f"hip_{i+1:04d}" for i in range(len(src))]})
    predictors = {}
    units = {
        "num_years": "years", "num_bpm": "bpm", "num_gl": "g/L", "num_pct": "%",
        "num_plt": "×10^9/L", "num_sec": "seconds", "num_dd": "mg/L",
        "num_ca": "mmol/L", "num_cr": "µmol/L",
    }
    for src_col, key, label, overrides, kind in keep:
        if kind == "cat":
            add_cat(frame, predictors, src[src_col], key, label, overrides)
        else:
            add_numeric(frame, predictors, src[src_col], key, label, units.get(kind))

    frame["extended_los"] = src["elos"].map(lambda v: 1 if int(v) == 1 else 0)
    excluded = sorted(set(src.columns) - {"seq", "elos"} - {c for c, *_ in keep})
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "extended_los",
        "excluded_predictors": excluded,
        "class_counts": {
            "Not extended stay": int((frame["extended_los"] == 0).sum()),
            "Extended stay": int((frame["extended_los"] == 1).sum()),
        },
        "identifier_note": (
            "Zenodo 10.5281/zenodo.22866907 development cohort m2_deid.csv. "
            "Predictors = S_honest_T0_deduped. Postoperative labs, complications, costs, "
            "death, LOS, and time-to-surgery excluded."
        ),
    }
    spec = {
        "name": "hip_fracture_elos",
        "description": (
            "Chongqing Hechuan hip-fracture admissions (1137). Binary extended length of stay "
            "from admission and preoperative findings only."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Orthopaedic admission after hip-fracture surgery is planned. The prediction uses "
            "fracture type, demographics, preoperative laboratories, planned arthroplasty, and "
            "protocolized tranexamic acid known at admission, before the hospital stay ends."
        ),
        "observations_header": "Admission and preoperative findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "extended_los", ["Not extended stay", "Extended stay"],
            "Based on the admission and preoperative findings, will the hospital stay be extended?",
        ),
    }
    write_bundle("hip_fracture_elos", frame, spec, summary, retrieval)


def prepare_berlin(raw_xlsx: Path, retrieval: list) -> None:
    discovery = pd.read_excel(raw_xlsx, sheet_name="discovery appendicitis yes no")
    validation = pd.read_excel(raw_xlsx, sheet_name="validation appendicitis yes no")
    discovery = discovery.rename(columns={"gender (m/f)": "gender"})
    cols = [
        "histopathology", "age (years)", "gender", "CRP (mg/L)",
        "thrombocytes (x 10 9/L)", "leukocytes  (x 10 9/L)", "neutrophils  (x 10 9/L)",
        "immature granulocytes  (x 10 9/L)", "lymphocytes  (x 10 9/L)",
        "monocytes  (x 10 9/L)", "eosinophils  (x 10 9/L)", "basophils  (x 10 9/L)",
        "appendiceal diameter (mm)",
    ]
    src = pd.concat([discovery[cols], validation[cols]], ignore_index=True)
    if len(src) != 590:
        raise SystemExit(f"berlin_appendicitis: expected 590 rows, got {len(src)}")

    frame = pd.DataFrame({"record_id": [f"app_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    add_numeric(frame, predictors, src["age (years)"], "age", "Age", "years")
    sex = src["gender"].map(lambda v: None if norm_code(v) is None else {
        "m": "Male", "f": "Female", "w": "Female",
    }.get(str(v).strip().lower()))
    add_cat(frame, predictors, sex, "sex", "Sex")
    add_numeric(frame, predictors, src["CRP (mg/L)"], "crp", "C-reactive protein", "mg/L")
    add_numeric(frame, predictors, src["thrombocytes (x 10 9/L)"], "platelets",
                "Platelets", "×10^9/L")
    add_numeric(frame, predictors, src["leukocytes  (x 10 9/L)"], "wbc",
                "Leukocytes", "×10^9/L")
    add_numeric(frame, predictors, src["neutrophils  (x 10 9/L)"], "neutrophils",
                "Neutrophils", "×10^9/L")
    add_numeric(frame, predictors, src["immature granulocytes  (x 10 9/L)"],
                "immature_granulocytes", "Immature granulocytes", "×10^9/L")
    add_numeric(frame, predictors, src["lymphocytes  (x 10 9/L)"], "lymphocytes",
                "Lymphocytes", "×10^9/L")
    add_numeric(frame, predictors, src["monocytes  (x 10 9/L)"], "monocytes",
                "Monocytes", "×10^9/L")
    add_numeric(frame, predictors, src["eosinophils  (x 10 9/L)"], "eosinophils",
                "Eosinophils", "×10^9/L")
    add_numeric(frame, predictors, src["basophils  (x 10 9/L)"], "basophils",
                "Basophils", "×10^9/L")
    add_numeric(frame, predictors, pd.to_numeric(src["appendiceal diameter (mm)"], errors="coerce"),
                "appendix_diameter", "Ultrasound appendiceal diameter", "mm")

    histo = src["histopathology"].astype(str).str.strip().str.lower()
    if set(histo) - {"complicated", "uncomplicated", "negative"}:
        raise SystemExit(f"berlin_appendicitis: unexpected histopathology {sorted(set(histo))}")
    frame["appendicitis"] = histo.ne("negative").astype(int)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "appendicitis",
        "excluded_predictors": ["Gangrän", "Phlegmone", "mm_App"],
        "class_counts": {
            "No appendicitis": int((frame["appendicitis"] == 0).sum()),
            "Appendicitis": int((frame["appendicitis"] == 1).sum()),
        },
        "histopathology_detail": histo.value_counts().to_dict(),
        "identifier_note": (
            "OSF 9wvys / Reismann et al. PLOS ONE 2019. Discovery+validation sheets concatenated "
            "(paper n=590). Y = histopathology appendicitis vs negative. Not UCI 938 Regensburg."
        ),
    }
    spec = {
        "name": "berlin_appendicitis",
        "description": (
            "Berlin paediatric suspected appendicitis (590 children). Binary histopathology "
            "from blood counts, CRP, and ultrasound diameter."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Paediatric emergency evaluation for suspected appendicitis. The prediction uses "
            "age, sex, inflammatory blood counts, C-reactive protein, and ultrasound appendiceal "
            "diameter available before histopathology."
        ),
        "observations_header": "Blood counts, CRP, and ultrasound findings:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "appendicitis", ["No appendicitis", "Appendicitis"],
            "Based on the blood counts, CRP, and ultrasound findings, is appendicitis present "
            "on histopathology?",
        ),
    }
    write_bundle("berlin_appendicitis", frame, spec, summary, retrieval)


def unpack_member(archive: Path, member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        dest.write_bytes(zf.read(member))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Optional subset of dataset names")
    args = parser.parse_args()
    wanted = set(args.only) if args.only else None

    jobs = [
        "ppd_bangladesh",
        "hip_fracture_elos",
        "berlin_appendicitis",
    ]
    for name in jobs:
        if wanted and name not in wanted:
            continue
        if name == "ppd_bangladesh":
            data_url = ("https://data.mendeley.com/public-files/datasets/4nznnrk8cg/files/"
                        "26c702d7-7cba-42f0-898f-ca7261f61955/file_downloaded")
            dict_url = ("https://data.mendeley.com/public-files/datasets/4nznnrk8cg/files/"
                        "1edfebe9-cc50-4819-8046-d13b2f49338d/file_downloaded")
            raw = DATASETS / name / "raw" / "PPD_dataset_v3.csv"
            raw_dict = DATASETS / name / "raw" / "PPD_Data_Dictionary_v3.csv"
            retrieval = [download(data_url, raw), download(dict_url, raw_dict)]
            prepare_ppd(raw, retrieval)
        elif name == "hip_fracture_elos":
            zip_url = ("https://zenodo.org/api/records/22866907/files/"
                       "deposit_JCEPI-D-26-01477_v1.0.0.zip/content")
            zpath = DATASETS / name / "raw" / "deposit_JCEPI-D-26-01477_v1.0.0.zip"
            retrieval = [download(zip_url, zpath)]
            raw = DATASETS / name / "raw" / "m2_deid.csv"
            audit = DATASETS / name / "raw" / "PREDICTION_TIME_AUDIT_adjudicated.csv"
            unpack_member(zpath, "deposit/data/m2_deid.csv", raw)
            unpack_member(zpath, "deposit/data/PREDICTION_TIME_AUDIT_adjudicated.csv", audit)
            retrieval.append({"path": str(raw.relative_to(ROOT)), "extracted_from": str(zpath.relative_to(ROOT)),
                              "sha256": sha256(raw)})
            prepare_hip(raw, retrieval)
        elif name == "berlin_appendicitis":
            zip_url = "https://osf.io/download/9wvys/"
            zpath = DATASETS / name / "raw" / "Data_project_AI_pediatric_appendicitis.zip"
            retrieval = [download(zip_url, zpath)]
            raw = DATASETS / name / "raw" / "Data project AI pediatric appendicitis.xlsx"
            unpack_member(zpath, "Data project AI pediatric appendicitis.xlsx", raw)
            retrieval.append({"path": str(raw.relative_to(ROOT)), "extracted_from": str(zpath.relative_to(ROOT)),
                              "sha256": sha256(raw)})
            prepare_berlin(raw, retrieval)


if __name__ == "__main__":
    main()
