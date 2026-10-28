#!/usr/bin/env python3
"""Prepare six public clinical tables, capped at the first 1000 patients per problem.

  physionet_2012   PhysioNet Challenge 2012 set A — ICU in-hospital death
  physionet_2019   PhysioNet Challenge 2019 set A — ever-sepsis from a t0 freeze
  nhanes_htn       NHANES 2021–2023 — measured hypertension (SBP/DBP or medication)
  metabric         cBioPortal METABRIC — PAM50+claudin-low or 3-gene subtype
  mexico_covid     Mexico SSA 2020 cierre — confirmed-case death
  dryad_er_cbc     Dryad 10.5061/dryad.dncjsxm5g — ER CBC all-cause death

Cap lives in patients.csv (not CLI --limit). First 1000 after a stable source-id
sort; stratified seed-20261008 fallback only if a class would fall below 10.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
USER_AGENT = "jev_prediction_testing/public-ext"
CAP = 1000  # None: no cap (scripts/prepare_full_tables.py)
MEXICO_ROWS = 5000  # None: every confirmed case
CH2019_FULL_N = 20643  # training_setA is p000001..p020643 with gaps (20,336 files)
SEED = 20261008
MIN_CLASS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, timeout: int = 300) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 400:
        return False
    try:
        return path.read_text(encoding="utf-8", errors="replace").startswith("version https://git-lfs")
    except OSError:
        return False


def download(url: str, dest: Path, timeout: int = 300) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not is_lfs_pointer(dest):
        return {"url": url, "path": str(dest.relative_to(ROOT)), "sha256": sha256(dest),
                "bytes": dest.stat().st_size, "retrieved": "cached"}
    dest.write_bytes(fetch(url, timeout=timeout))
    if dest.stat().st_size == 0:
        raise SystemExit(f"empty download: {url}")
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
    (out / "PREPARATION.json").write_text(json.dumps(notes, indent=2, default=str) + "\n", encoding="utf-8")
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
          f"classes={summary['class_counts']}", flush=True)


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
    if not text or text.lower() in {"nan", "none", "nat", "?", "na"}:
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
    frame[key] = blank(pd.to_numeric(series, errors="coerce"))
    predictors[key] = numeric_pred(key, label, unit)


def add_cat(frame, predictors, series, key, label, overrides=None):
    coded = as_code_series(series)
    frame[key] = blank(coded)
    predictors[key] = cat_pred(key, label, codes_map(series, overrides))


def nan_sentinel(series, sentinels=(-1, -1.0)):
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values.isin(list(sentinels)))


def cap_rows(frame: pd.DataFrame, ycol: str, n: int | None = None, min_class: int = MIN_CLASS) -> pd.DataFrame:
    """Keep the first n rows; stratified subsample only if a class would drop below min_class."""
    if ycol not in frame:
        raise SystemExit(f"cap_rows: missing {ycol}")
    n = CAP if n is None else n
    frame = frame.reset_index(drop=True)
    counts_all = frame[ycol].value_counts()
    if counts_all.min() < min_class:
        raise SystemExit(f"source classes below {min_class}: {counts_all.to_dict()}")
    if n is None or len(frame) <= n:
        return frame
    head = frame.iloc[:n]
    counts = head[ycol].value_counts()
    if counts.min() >= min_class:
        return head.reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    parts = []
    for _, group in frame.groupby(ycol, sort=False):
        take = min(len(group), max(min_class, int(round(n * (len(group) / len(frame))))))
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        parts.append(group.loc[idx])
    out = pd.concat(parts)
    if len(out) > n:
        out = out.sample(n=n, random_state=SEED)
    print(f"  cap: first {n} dropped a class below {min_class}; used stratified n={len(out)}", flush=True)
    return out.sort_index().reset_index(drop=True)


def read_cbioportal(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


# ---------------------------------------------------------------------------
# PhysioNet Challenge 2012
# ---------------------------------------------------------------------------

CH2012_TS = {
    "ALP": ("alp", "Alkaline phosphatase", "IU/L"),
    "ALT": ("alt", "Alanine aminotransferase", "IU/L"),
    "AST": ("ast", "Aspartate aminotransferase", "IU/L"),
    "Albumin": ("albumin", "Albumin", "g/dL"),
    "BUN": ("bun", "Blood urea nitrogen", "mg/dL"),
    "Bilirubin": ("bilirubin", "Bilirubin", "mg/dL"),
    "Cholesterol": ("cholesterol", "Cholesterol", "mg/dL"),
    "Creatinine": ("creatinine", "Creatinine", "mg/dL"),
    "DiasABP": ("dias_abp", "Diastolic arterial blood pressure", "mmHg"),
    "FiO2": ("fio2", "Inspired oxygen fraction", "fraction"),
    "GCS": ("gcs", "Glasgow coma scale", "points"),
    "Glucose": ("glucose", "Glucose", "mg/dL"),
    "HCO3": ("hco3", "Bicarbonate", "mmol/L"),
    "HCT": ("hct", "Hematocrit", "%"),
    "HR": ("hr", "Heart rate", "beats/min"),
    "K": ("potassium", "Potassium", "mmol/L"),
    "Lactate": ("lactate", "Lactate", "mmol/L"),
    "Mg": ("magnesium", "Magnesium", "mmol/L"),
    "MAP": ("map", "Mean arterial pressure", "mmHg"),
    "Na": ("sodium", "Sodium", "mmol/L"),
    "NIDiasABP": ("ni_dias_abp", "Noninvasive diastolic blood pressure", "mmHg"),
    "NIMAP": ("ni_map", "Noninvasive mean arterial pressure", "mmHg"),
    "NISysABP": ("ni_sys_abp", "Noninvasive systolic blood pressure", "mmHg"),
    "PaCO2": ("paco2", "Arterial PaCO2", "mmHg"),
    "PaO2": ("pao2", "Arterial PaO2", "mmHg"),
    "pH": ("ph", "Arterial pH", None),
    "Platelets": ("platelets", "Platelets", "×10^9/L"),
    "RespRate": ("resp_rate", "Respiratory rate", "breaths/min"),
    "SaO2": ("sao2", "Arterial oxygen saturation", "%"),
    "SysABP": ("sys_abp", "Systolic arterial blood pressure", "mmHg"),
    "Temp": ("temp", "Temperature", "°C"),
    "TroponinI": ("troponin_i", "Troponin I", "ng/mL"),
    "TroponinT": ("troponin_t", "Troponin T", "ng/mL"),
    "Urine": ("urine", "Urine output", "mL"),
    "WBC": ("wbc", "White blood cells", "×10^9/L"),
}


def _ch2012_file_stats(path: Path) -> dict:
    raw = pd.read_csv(path)
    raw["Value"] = nan_sentinel(raw["Value"])
    out = {"record_id": None}
    for param, series in raw.groupby("Parameter")["Value"]:
        series = series.dropna()
        if param == "RecordID" and len(series):
            out["record_id"] = int(series.iloc[0])
        elif param == "Age":
            out["age"] = series.iloc[0] if len(series) else np.nan
        elif param == "Gender":
            out["gender"] = series.iloc[0] if len(series) else np.nan
        elif param == "Height":
            out["height"] = series.iloc[0] if len(series) else np.nan
        elif param == "ICUType":
            out["icu_type"] = series.iloc[0] if len(series) else np.nan
        elif param == "Weight":
            out["weight"] = series.iloc[-1] if len(series) else np.nan
        elif param == "MechVent":
            out["mech_vent"] = series.iloc[-1] if len(series) else np.nan
        elif param in CH2012_TS:
            key = CH2012_TS[param][0]
            out[f"{key}_last"] = series.iloc[-1] if len(series) else np.nan
            out[f"{key}_mean"] = series.mean() if len(series) else np.nan
    return out


def prepare_physionet_2012(raw_dir: Path, retrieval: list) -> None:
    archive = raw_dir / "set-a.tar.gz"
    outcomes_path = raw_dir / "Outcomes-a.txt"
    extract = raw_dir / "set-a"
    if not extract.exists() or not any(extract.rglob("*.txt")):
        extract.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract)
    files = {int(p.stem): p for p in extract.rglob("*.txt") if p.stem.isdigit()}
    outcomes = pd.read_csv(outcomes_path)
    outcomes = outcomes.sort_values("RecordID").reset_index(drop=True)
    keep_ids = [rid for rid in outcomes["RecordID"].astype(int) if rid in files][:CAP]
    rows = [_ch2012_file_stats(files[rid]) for rid in keep_ids]
    src = pd.DataFrame(rows)
    needed = ["record_id", "age", "gender", "height", "icu_type", "weight", "mech_vent"]
    for src_name, (key, _, _) in CH2012_TS.items():
        needed.extend([f"{key}_last", f"{key}_mean"])
    src = src.reindex(columns=list(dict.fromkeys(needed)))
    ymap = outcomes.set_index("RecordID")["In-hospital_death"].to_dict()
    src["icu_death"] = src["record_id"].map(ymap).astype(int)
    src = cap_rows(src.sort_values("record_id"), "icu_death")
    src["record_id"] = [f"ch2012_{int(r)}" for r in src["record_id"]]

    frame = pd.DataFrame({"record_id": src["record_id"]})
    predictors = {}
    add_numeric(frame, predictors, src["age"], "age", "Age", "years")
    gender = src["gender"].map({0: "Female", 1: "Male", 0.0: "Female", 1.0: "Male"})
    add_cat(frame, predictors, gender, "sex", "Sex")
    add_numeric(frame, predictors, src["height"], "height", "Height", "cm")
    add_numeric(frame, predictors, src["weight"], "weight", "Weight", "kg")
    icu = src["icu_type"].map({
        1: "Coronary care unit", 1.0: "Coronary care unit",
        2: "Cardiac surgery recovery", 2.0: "Cardiac surgery recovery",
        3: "Medical ICU", 3.0: "Medical ICU",
        4: "Surgical ICU", 4.0: "Surgical ICU",
    })
    add_cat(frame, predictors, icu, "icu_type", "ICU type")
    vent = src["mech_vent"].map({0: "No", 1: "Yes", 0.0: "No", 1.0: "Yes"})
    add_cat(frame, predictors, vent, "mech_vent", "Mechanical ventilation")
    for src_name, (key, label, unit) in CH2012_TS.items():
        add_numeric(frame, predictors, src[f"{key}_last"], f"{key}_last", f"{label} (last 48h)", unit)
        add_numeric(frame, predictors, src[f"{key}_mean"], f"{key}_mean", f"{label} (mean 48h)", unit)
    frame["icu_death"] = src["icu_death"].astype(int)

    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "icu_death",
        "excluded_predictors": ["SAPS-I", "SOFA", "Length_of_stay", "Survival"],
        "class_counts": {
            "Survived": int((frame["icu_death"] == 0).sum()),
            "Died": int((frame["icu_death"] == 1).sum()),
        },
        "identifier_note": (
            "PhysioNet Challenge 2012 set A. First 1000 RecordIDs. Y = In-hospital_death. "
            "Predictors = 48h last/mean of named labs and vitals plus admission descriptors. "
            "Missing encoded as -1 in the source and converted to empty. SAPS-I/SOFA/LOS/Survival dropped."
        ),
        "cap": CAP,
    }
    spec = {
        "name": "physionet_2012",
        "description": (
            "PhysioNet/CinC Challenge 2012 ICU set A (first 1000 patients). Binary in-hospital death "
            "from 48-hour named vitals and laboratories."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Adult intensive-care stay. The prediction uses demographics, ICU type, and the last and "
            "mean of named vitals and laboratories recorded during the first 48 hours, before "
            "in-hospital death is known."
        ),
        "observations_header": "First 48-hour ICU vitals and laboratories:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "icu_death", ["Survived", "Died"],
            "Based on the first 48-hour ICU vitals and laboratories, will the patient die in hospital?",
        ),
    }
    write_bundle("physionet_2012", frame, spec, summary, retrieval)


# ---------------------------------------------------------------------------
# PhysioNet Challenge 2019
# ---------------------------------------------------------------------------

CH2019_LABS = {
    "HR": ("hr", "Heart rate", "beats/min"),
    "O2Sat": ("o2sat", "Oxygen saturation", "%"),
    "Temp": ("temp", "Temperature", "°C"),
    "SBP": ("sbp", "Systolic blood pressure", "mmHg"),
    "MAP": ("map", "Mean arterial pressure", "mmHg"),
    "DBP": ("dbp", "Diastolic blood pressure", "mmHg"),
    "Resp": ("resp", "Respiratory rate", "breaths/min"),
    "EtCO2": ("etco2", "End-tidal CO2", "mmHg"),
    "BaseExcess": ("base_excess", "Base excess", "mmol/L"),
    "HCO3": ("hco3", "Bicarbonate", "mmol/L"),
    "FiO2": ("fio2", "Inspired oxygen fraction", "fraction"),
    "pH": ("ph", "Arterial pH", None),
    "PaCO2": ("paco2", "Arterial PaCO2", "mmHg"),
    "SaO2": ("sao2", "Arterial oxygen saturation", "%"),
    "AST": ("ast", "Aspartate aminotransferase", "IU/L"),
    "BUN": ("bun", "Blood urea nitrogen", "mg/dL"),
    "Alkalinephos": ("alp", "Alkaline phosphatase", "IU/L"),
    "Calcium": ("calcium", "Calcium", "mg/dL"),
    "Chloride": ("chloride", "Chloride", "mmol/L"),
    "Creatinine": ("creatinine", "Creatinine", "mg/dL"),
    "Bilirubin_direct": ("bili_direct", "Direct bilirubin", "mg/dL"),
    "Glucose": ("glucose", "Glucose", "mg/dL"),
    "Lactate": ("lactate", "Lactate", "mmol/L"),
    "Magnesium": ("magnesium", "Magnesium", "mmol/L"),
    "Phosphate": ("phosphate", "Phosphate", "mg/dL"),
    "Potassium": ("potassium", "Potassium", "mmol/L"),
    "Bilirubin_total": ("bili_total", "Total bilirubin", "mg/dL"),
    "TroponinI": ("troponin_i", "Troponin I", "ng/mL"),
    "Hct": ("hct", "Hematocrit", "%"),
    "Hgb": ("hgb", "Hemoglobin", "g/dL"),
    "PTT": ("ptt", "Partial thromboplastin time", "s"),
    "WBC": ("wbc", "White blood cells", "×10^9/L"),
    "Fibrinogen": ("fibrinogen", "Fibrinogen", "mg/dL"),
    "Platelets": ("platelets", "Platelets", "×10^9/L"),
}


def _ch2019_patient(path: Path) -> dict | None:
    df = pd.read_csv(path, sep="|")
    if df.empty or "ICULOS" not in df:
        return None
    pid = path.stem
    ever = int((pd.to_numeric(df["SepsisLabel"], errors="coerce") == 1).any())
    window = df[pd.to_numeric(df["ICULOS"], errors="coerce") <= 6].copy()
    if window.empty:
        window = df.iloc[[0]].copy()
    pre = window[pd.to_numeric(window["SepsisLabel"], errors="coerce") != 1]
    freeze = pre if len(pre) else window.iloc[[0]]
    last = freeze.iloc[-1]
    row = {
        "record_id": pid,
        "sepsis": ever,
        "age": pd.to_numeric(last.get("Age"), errors="coerce"),
        "gender": last.get("Gender"),
        "hosp_adm_time": pd.to_numeric(last.get("HospAdmTime"), errors="coerce"),
        "unit1": last.get("Unit1"),
        "unit2": last.get("Unit2"),
    }
    for src_name, (key, _, _) in CH2019_LABS.items():
        row[key] = pd.to_numeric(last.get(src_name), errors="coerce")
    return row


def prepare_physionet_2019(raw_dir: Path, retrieval: list) -> None:
    files = sorted((raw_dir / "training_setA").glob("p*.psv"))
    files = [p for p in files if p.stat().st_size > 100]
    if CAP is not None and len(files) < CAP:
        raise SystemExit(f"physionet_2019: expected {CAP} psv files, found {len(files)}")
    files = files[:CAP]
    rows = [_ch2019_patient(path) for path in files]
    src = pd.DataFrame([r for r in rows if r])
    src = cap_rows(src.sort_values("record_id"), "sepsis")
    frame = pd.DataFrame({"record_id": src["record_id"]})
    predictors = {}
    add_numeric(frame, predictors, src["age"], "age", "Age", "years")
    sex = src["gender"].map({0: "Female", 1: "Male", 0.0: "Female", 1.0: "Male", "0": "Female", "1": "Male"})
    add_cat(frame, predictors, sex, "sex", "Sex")
    add_numeric(frame, predictors, src["hosp_adm_time"], "hosp_adm_time",
                "Hours from hospital to ICU admission", "hours")
    unit = np.where(pd.to_numeric(src["unit1"], errors="coerce") == 1, "MICU",
                    np.where(pd.to_numeric(src["unit2"], errors="coerce") == 1, "SICU", None))
    add_cat(frame, predictors, pd.Series(unit), "icu_unit", "ICU unit")
    for src_name, (key, label, unit) in CH2019_LABS.items():
        add_numeric(frame, predictors, src[key], key, f"{label} (early ICU)", unit)
    frame["sepsis"] = src["sepsis"].astype(int)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "sepsis",
        "excluded_predictors": ["ICULOS", "SepsisLabel", "post-onset hourly rows"],
        "class_counts": {
            "No sepsis": int((frame["sepsis"] == 0).sum()),
            "Sepsis": int((frame["sepsis"] == 1).sum()),
        },
        "identifier_note": (
            "PhysioNet Challenge 2019 training_setA. First 1000 patient files. Features frozen at "
            "ICULOS<=6 hours and before SepsisLabel=1. Y = ever sepsis during the stay."
        ),
        "cap": CAP,
    }
    spec = {
        "name": "physionet_2019",
        "description": (
            "PhysioNet/CinC Challenge 2019 ICU set A (first 1000 patients). Binary ever-sepsis from "
            "an early-stay vital and laboratory snapshot."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Adult intensive-care admission. The prediction uses age, sex, unit, and the most recent "
            "named vitals and laboratories from the first six ICU hours, before later sepsis labels."
        ),
        "observations_header": "Early ICU vitals and laboratories:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "sepsis", ["No sepsis", "Sepsis"],
            "Based on the early ICU vitals and laboratories, will the patient develop sepsis during this stay?",
        ),
    }
    write_bundle("physionet_2019", frame, spec, summary, retrieval)


# ---------------------------------------------------------------------------
# NHANES 2021–2023 measured hypertension
# ---------------------------------------------------------------------------

def _xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport", encoding="utf-8")


def prepare_nhanes_htn(raw_dir: Path, retrieval: list) -> None:
    demo = _xpt(raw_dir / "DEMO_L.xpt")
    bpx = _xpt(raw_dir / "BPXO_L.xpt")
    bmx = _xpt(raw_dir / "BMX_L.xpt")
    bpq = _xpt(raw_dir / "BPQ_L.xpt")
    diq = _xpt(raw_dir / "DIQ_L.xpt")
    smq = _xpt(raw_dir / "SMQ_L.xpt")
    src = demo.merge(bpx, on="SEQN", how="inner").merge(bmx, on="SEQN", how="left")
    src = src.merge(bpq, on="SEQN", how="left").merge(diq, on="SEQN", how="left").merge(smq, on="SEQN", how="left")
    src = src.loc[pd.to_numeric(src["RIDAGEYR"], errors="coerce") >= 18].copy()
    sbp = src[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    dbp = src[["BPXODI1", "BPXODI2", "BPXODI3"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    med_col = next((c for c in ("BPQ150", "BPQ050A", "BPQ040A") if c in src.columns), None)
    meds = pd.to_numeric(src[med_col], errors="coerce").eq(1) if med_col else pd.Series(False, index=src.index)
    measured = (sbp >= 130) | (dbp >= 80)
    src = src.loc[sbp.notna() | dbp.notna() | meds].copy()
    src["hypertension"] = (measured.loc[src.index].fillna(False) | meds.loc[src.index].fillna(False)).astype(int)
    src = src.sort_values("SEQN")
    src = cap_rows(src, "hypertension")

    frame = pd.DataFrame({"record_id": [f"nhanes_{int(s)}" for s in src["SEQN"]]})
    predictors = {}
    add_numeric(frame, predictors, src["RIDAGEYR"], "age", "Age", "years")
    sex = src["RIAGENDR"].map({1: "Male", 2: "Female", 1.0: "Male", 2.0: "Female"})
    add_cat(frame, predictors, sex, "sex", "Sex")
    race = src["RIDRETH3"].map({
        1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other or Multi-Racial",
        1.0: "Mexican American", 2.0: "Other Hispanic", 3.0: "Non-Hispanic White",
        4.0: "Non-Hispanic Black", 6.0: "Non-Hispanic Asian", 7.0: "Other or Multi-Racial",
    })
    add_cat(frame, predictors, race, "race_ethnicity", "Race and ethnicity")
    educ = src["DMDEDUC2"].map({
        1: "Less than 9th grade", 2: "9th to 11th grade", 3: "High school or GED",
        4: "Some college or AA", 5: "College graduate or above",
        1.0: "Less than 9th grade", 2.0: "9th to 11th grade", 3.0: "High school or GED",
        4.0: "Some college or AA", 5.0: "College graduate or above",
    })
    add_cat(frame, predictors, educ, "education", "Education")
    add_numeric(frame, predictors, src["INDFMPIR"], "poverty_ratio", "Family income to poverty ratio")
    add_numeric(frame, predictors, src["BMXBMI"], "bmi", "Body mass index", "kg/m^2")
    add_numeric(frame, predictors, src["BMXWAIST"], "waist", "Waist circumference", "cm")
    add_numeric(frame, predictors, src["BMXWT"], "weight", "Weight", "kg")
    add_numeric(frame, predictors, src["BMXHT"], "height", "Height", "cm")
    diabetes = src["DIQ010"].map({
        1: "Yes", 2: "No", 3: "Borderline", 1.0: "Yes", 2.0: "No", 3.0: "Borderline",
    })
    add_cat(frame, predictors, diabetes, "diabetes", "Doctor told diabetes")
    smoke100 = src["SMQ020"].map({1: "Yes", 2: "No", 1.0: "Yes", 2.0: "No"})
    add_cat(frame, predictors, smoke100, "smoked_100", "Smoked at least 100 cigarettes")
    current = src["SMQ040"].map({
        1: "Every day", 2: "Some days", 3: "Not at all",
        1.0: "Every day", 2.0: "Some days", 3.0: "Not at all",
    })
    add_cat(frame, predictors, current, "current_smoking", "Current cigarette smoking")
    frame["hypertension"] = src["hypertension"].astype(int)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "hypertension",
        "excluded_predictors": [
            "BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3",
            "BPQ020", "BPQ030", "BPQ150", "BPQ040A", "BPQ050A",
        ],
        "class_counts": {
            "No hypertension": int((frame["hypertension"] == 0).sum()),
            "Hypertension": int((frame["hypertension"] == 1).sum()),
        },
        "identifier_note": (
            "NHANES 2021–2023 adults 18+ with oscillometric BP. Y = mean SBP>=130 or mean DBP>=80 "
            "or currently taking BP medication (NCHS Data Brief 511). BP values and BP-medication "
            "items are used only to build Y. First 1000 SEQN after eligibility."
        ),
        "cap": CAP,
    }
    spec = {
        "name": "nhanes_htn",
        "description": (
            "NHANES August 2021–August 2023 (first 1000 eligible adults). Binary measured "
            "hypertension from demographics, body measures, smoking, and diabetes history."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "US National Health and Nutrition Examination Survey examination. The prediction uses "
            "demographics, body measures, smoking, and diabetes history available without using "
            "the blood-pressure readings that define the label."
        ),
        "observations_header": "Demographics, body measures, smoking, and diabetes history:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "hypertension", ["No hypertension", "Hypertension"],
            "Based on the demographics, body measures, smoking, and diabetes history, does the adult have hypertension?",
        ),
    }
    write_bundle("nhanes_htn", frame, spec, summary, retrieval)


# ---------------------------------------------------------------------------
# METABRIC
# ---------------------------------------------------------------------------

def prepare_metabric(raw_dir: Path, retrieval: list) -> None:
    patient = read_cbioportal(raw_dir / "data_clinical_patient.txt")
    sample = read_cbioportal(raw_dir / "data_clinical_sample.txt")
    src = patient.merge(sample, on="PATIENT_ID", how="inner", suffixes=("", "_sample"))
    pam50 = src["CLAUDIN_SUBTYPE"].map(lambda v: None if norm_code(v) in {None, "NC"} else str(v).strip())
    src = src.loc[pam50.notna()].copy()
    src["subtype"] = pam50.loc[src.index]
    outcome_name = "pam50_subtype"
    classes = sorted(src["subtype"].unique())
    question = (
        "Based on the clinical and immunohistochemical findings, what is the PAM50 plus "
        "claudin-low subtype of this breast tumour?"
    )
    y_note = "PAM50+claudin-low (NC dropped)"
    src = src.sort_values("PATIENT_ID")
    # Temporary numeric y for cap_rows; restore labels after.
    src["_ycode"] = src["subtype"].astype("category").cat.codes
    src = cap_rows(src, "_ycode")
    if src["subtype"].value_counts().min() < MIN_CLASS:
        raise SystemExit(f"metabric: class below {MIN_CLASS} after cap: {src['subtype'].value_counts().to_dict()}")

    frame = pd.DataFrame({"record_id": src["PATIENT_ID"].astype(str)})
    predictors = {}
    add_numeric(frame, predictors, src["AGE_AT_DIAGNOSIS"], "age", "Age at diagnosis", "years")
    add_cat(frame, predictors, src["INFERRED_MENOPAUSAL_STATE"], "menopausal", "Inferred menopausal state")
    add_cat(frame, predictors, src["LATERALITY"], "laterality", "Primary tumour laterality")
    add_cat(frame, predictors, src["HISTOLOGICAL_SUBTYPE"], "histology", "Histological subtype")
    add_cat(frame, predictors, src["BREAST_SURGERY"], "surgery", "Breast surgery")
    add_cat(frame, predictors, src["CELLULARITY"], "cellularity", "Tumour cellularity")
    add_numeric(frame, predictors, src["TUMOR_SIZE"], "tumor_size", "Tumour size", "mm")
    add_cat(frame, predictors, src["TUMOR_STAGE"], "tumor_stage", "Tumour stage")
    add_cat(frame, predictors, src["GRADE"], "grade", "Histologic grade")
    add_numeric(frame, predictors, src["LYMPH_NODES_EXAMINED_POSITIVE"],
                "positive_nodes", "Positive lymph nodes", "count")
    add_cat(frame, predictors, src["ER_STATUS"], "er_status", "ER status")
    add_cat(frame, predictors, src["PR_STATUS"], "pr_status", "PR status")
    add_cat(frame, predictors, src["HER2_STATUS"], "her2_status", "HER2 status")
    frame[outcome_name] = src["subtype"].astype(str)
    mapping = {c: c for c in classes}
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": outcome_name,
        "excluded_predictors": [
            "OS_MONTHS", "OS_STATUS", "RFS_MONTHS", "RFS_STATUS", "VITAL_STATUS",
            "INTCLUST", "CHEMOTHERAPY", "HORMONE_THERAPY", "RADIO_THERAPY", "NPI",
            "TMB_NONSYNONYMOUS", "CLAUDIN_SUBTYPE" if outcome_name == "gene3_subtype" else "THREEGENE",
        ],
        "class_counts": src["subtype"].value_counts().to_dict(),
        "identifier_note": (
            f"cBioPortal brca_metabric clinical TSV. Y = {y_note}. Survival, INTCLUST, NPI, TMB, "
            "and post-diagnosis systemic/radiotherapy dropped. First 1000 PATIENT_ID with a valid subtype."
        ),
        "cap": CAP,
    }
    spec = {
        "name": "metabric",
        "description": (
            f"METABRIC breast cancer clinical table (first 1000 tumours). {y_note} from presentation "
            "pathology and immunohistochemistry."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Primary breast tumour at diagnosis. The prediction uses age, menopausal state, "
            "pathology, and ER/PR/HER2 status available with the tumour, not survival times."
        ),
        "observations_header": "Clinical and immunohistochemical findings:",
        "predictors": predictors,
        "outcomes": {
            outcome_name: {
                "column": outcome_name,
                "classes": classes,
                "mapping": mapping,
                "question": question,
            }
        },
    }
    write_bundle("metabric", frame, spec, summary, retrieval)


# ---------------------------------------------------------------------------
# Mexico SSA COVID 2020
# ---------------------------------------------------------------------------

MX_YESNO = {1: "Yes", 2: "No", "1": "Yes", "2": "No", 1.0: "Yes", 2.0: "No"}
MX_COMORB = [
    ("NEUMONIA", "pneumonia", "Pneumonia"),
    ("DIABETES", "diabetes", "Diabetes"),
    ("EPOC", "copd", "COPD"),
    ("ASMA", "asthma", "Asthma"),
    ("INMUSUPR", "immunosuppressed", "Immunosuppression"),
    ("HIPERTENSION", "hypertension", "Hypertension"),
    ("OTRA_COM", "other_disease", "Other comorbidity"),
    ("CARDIOVASCULAR", "cardiovascular", "Cardiovascular disease"),
    ("OBESIDAD", "obesity", "Obesity"),
    ("RENAL_CRONICA", "chronic_kidney", "Chronic kidney disease"),
    ("TABAQUISMO", "smoking", "Smoking"),
]


def prepare_mexico_covid(raw_dir: Path, retrieval: list) -> None:
    zpath = raw_dir / "COVID19MEXICO2020.zip"
    wanted = []
    with zipfile.ZipFile(zpath) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as handle:
            text = io.TextIOWrapper(handle, encoding="latin-1")
            for chunk in pd.read_csv(text, chunksize=50_000, low_memory=False):
                if "CLASIFICACION_FINAL" in chunk.columns:
                    conf = chunk["CLASIFICACION_FINAL"].isin([1, 2, 3, "1", "2", "3"])
                else:
                    conf = chunk["RESULTADO_LAB"].isin([1, "1"])
                part = chunk.loc[conf]
                if part.empty:
                    continue
                wanted.append(part)
                if MEXICO_ROWS and sum(len(x) for x in wanted) >= MEXICO_ROWS:
                    break
    src = pd.concat(wanted, ignore_index=True)
    src = src.head(MEXICO_ROWS).copy() if MEXICO_ROWS else src
    def_date = src["FECHA_DEF"].astype(str).str.strip()
    src["covid_death"] = (~def_date.isin(["9999-99-99", "nan", "NaN", ""])).astype(int)
    src = cap_rows(src, "covid_death")
    frame = pd.DataFrame({"record_id": src["ID_REGISTRO"].astype(str)})
    predictors = {}
    add_numeric(frame, predictors, src["EDAD"], "age", "Age", "years")
    sex = src["SEXO"].map({1: "Female", 2: "Male", "1": "Female", "2": "Male", 1.0: "Female", 2.0: "Male"})
    add_cat(frame, predictors, sex, "sex", "Sex")
    preg = src["EMBARAZO"].map(MX_YESNO)
    add_cat(frame, predictors, preg, "pregnancy", "Pregnancy")
    indig = src["INDIGENA"].map(MX_YESNO) if "INDIGENA" in src else src.get("HABLA_LENGUA_INDIG", pd.Series(dtype=object)).map(MX_YESNO)
    add_cat(frame, predictors, indig, "indigenous", "Indigenous")
    nat = src["NACIONALIDAD"].map({1: "Mexican", 2: "Foreign", "1": "Mexican", "2": "Foreign", 1.0: "Mexican", 2.0: "Foreign"})
    add_cat(frame, predictors, nat, "nationality", "Nationality")
    for col, key, label in MX_COMORB:
        if col in src:
            add_cat(frame, predictors, src[col].map(MX_YESNO), key, label)
    frame["covid_death"] = src["covid_death"].astype(int)
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "covid_death",
        "excluded_predictors": ["INTUBADO", "UCI", "TIPO_PACIENTE", "FECHA_INGRESO", "FECHA_DEF"],
        "class_counts": {
            "Alive": int((frame["covid_death"] == 0).sum()),
            "Died": int((frame["covid_death"] == 1).sum()),
        },
        "identifier_note": (
            "Mexico SSA datos abiertos 2020 cierre. CLASIFICACION_FINAL in {1,2,3}. First 1000 "
            "confirmed rows in file order. INTUBADO/UCI/TIPO_PACIENTE dropped. Codes 97/98/99 left empty."
        ),
        "cap": CAP,
    }
    spec = {
        "name": "mexico_covid",
        "description": (
            "Mexico SSA 2020 COVID-19 line-list (first 1000 confirmed cases). Binary death from "
            "age, sex, and named comorbidities."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Confirmed SARS-CoV-2 case reported to Mexico's epidemiological surveillance system. "
            "The prediction uses age, sex, pregnancy, and comorbidities known at case notification, "
            "not intubation, ICU, or hospitalisation type."
        ),
        "observations_header": "Demographics and comorbidities:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "covid_death", ["Alive", "Died"],
            "Based on the demographics and comorbidities, will this confirmed COVID-19 case die?",
        ),
    }
    write_bundle("mexico_covid", frame, spec, summary, retrieval)


# ---------------------------------------------------------------------------
# Dryad ER CBC
# ---------------------------------------------------------------------------

ER_NUM = {
    "Age": ("age", "Age", "years"),
    "Hb": ("hemoglobin", "Hemoglobin", "g/dL"),
    "MCV": ("mcv", "Mean corpuscular volume", "fL"),
    "RDW": ("rdw", "Red cell distribution width", "%"),
    "PDW": ("pdw", "Platelet distribution width", "%"),
    "PCT": ("pct", "Plateletcrit", "%"),
    "RBC": ("rbc", "Red blood cells", "×10^12/L"),
    "WBC": ("wbc", "White blood cells", "×10^9/L"),
    "Neu": ("neutrophils", "Neutrophils", "×10^9/L"),
    "Lym": ("lymphocytes", "Lymphocytes", "×10^9/L"),
    "Mon": ("monocytes", "Monocytes", "×10^9/L"),
    "Eos": ("eosinophils", "Eosinophils", "×10^9/L"),
    "Bas": ("basophils", "Basophils", "×10^9/L"),
    "PLT": ("platelets", "Platelets", "×10^9/L"),
}


def prepare_dryad_er_cbc(raw_csv: Path, retrieval: list) -> None:
    src = pd.read_csv(raw_csv)
    if "alive0_dead1" in src.columns:
        src["er_death"] = pd.to_numeric(src["alive0_dead1"], errors="coerce").astype(int)
    elif "deadTRUE" in src.columns:
        src["er_death"] = src["deadTRUE"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0})
        src["er_death"] = src["er_death"].astype(int)
    else:
        raise SystemExit(f"dryad_er_cbc: no death column in {list(src.columns)}")
    src = src.sort_values("Num") if "Num" in src.columns else src
    src = cap_rows(src, "er_death")
    ids = src["Num"] if "Num" in src.columns else range(1, len(src) + 1)
    frame = pd.DataFrame({"record_id": [f"er_{int(i)}" for i in ids]})
    predictors = {}
    add_cat(frame, predictors, src["Sex"].map({"F": "Female", "M": "Male", "f": "Female", "m": "Male"}),
            "sex", "Sex")
    add_cat(frame, predictors, src["Wave_I_II_off"], "covid_wave", "ER period relative to COVID-19 surges")
    for col, (key, label, unit) in ER_NUM.items():
        add_numeric(frame, predictors, src[col], key, label, unit)
    frame["er_death"] = src["er_death"].astype(int)
    dropped = [c for c in src.columns if any(tok in c for tok in ("tailr", "Tailr", "bench", "Bench", "ddFUp", "Dischrd", "deadTRUE", "alive0"))]
    summary = {
        "observations": len(frame), "predictors": len(predictors), "outcome": "er_death",
        "excluded_predictors": dropped + ["Dischrd0_Inward1"],
        "class_counts": {
            "Alive": int((frame["er_death"] == 0).sum()),
            "Died": int((frame["er_death"] == 1).sum()),
        },
        "identifier_note": (
            "Dryad 10.5061/dryad.dncjsxm5g. First 1000 Num. Y = all-cause death. Follow-up days, "
            "inward admission, and tailored/benchmark score columns dropped."
        ),
        "cap": CAP,
    }
    spec = {
        "name": "dryad_er_cbc",
        "description": (
            "Northern-Italy emergency-room blood-cell differential (first 1000 adults). Binary "
            "all-cause death from age, sex, COVID-wave period, and baseline CBC indices."
        ),
        "file": "patients.csv", "separator": ",", "id_column": "record_id", "pseudonymize": False,
        "setting": (
            "Adult emergency-room presentation with a complete blood-cell differential at first draw. "
            "The prediction uses age, sex, epidemic period, and named CBC indices, not follow-up "
            "time or outcome-tuned score columns."
        ),
        "observations_header": "Emergency-room blood-cell differential:",
        "predictors": predictors,
        "outcomes": binary_outcome(
            "er_death", ["Alive", "Died"],
            "Based on the emergency-room blood-cell differential, will the patient die during follow-up?",
        ),
    }
    write_bundle("dryad_er_cbc", frame, spec, summary, retrieval)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def download_ch2019(raw_dir: Path, n: int = CAP) -> list:
    dest_dir = raw_dir / "training_setA"
    dest_dir.mkdir(parents=True, exist_ok=True)
    retrieval = []
    missing = []
    for i in range(1, n + 1):
        name = f"p{i:06d}.psv"
        dest = dest_dir / name
        # Same objects as physionet.org/files/...; the open S3 mirror is ~15x faster for 20k small files.
        url = f"https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/training_setA/{name}"
        if dest.exists() and dest.stat().st_size > 100:
            continue
        missing.append((url, dest))
    if missing:
        print(f"physionet_2019: downloading {len(missing)} psv files", flush=True)

        def one(item):
            url, dest = item
            try:
                dest.write_bytes(fetch(url, timeout=60))
            except HTTPError as error:
                if error.code != 404:  # numbering has gaps
                    raise
                return url, None
            return url, dest

        with ThreadPoolExecutor(max_workers=32) as pool:
            futs = [pool.submit(one, item) for item in missing]
            for fut in as_completed(futs):
                url, dest = fut.result()
                if dest is None:
                    continue
                retrieval.append({"url": url, "path": str(dest.relative_to(ROOT)),
                                  "bytes": dest.stat().st_size})
    have = list(dest_dir.glob("p*.psv"))
    retrieval.append({"url": "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/",
                      "n_files": len(have), "requested": n})
    return retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Optional subset of dataset names")
    args = parser.parse_args()
    wanted = set(args.only) if args.only else None
    jobs = [
        "physionet_2012", "physionet_2019", "nhanes_htn",
        "metabric", "mexico_covid", "dryad_er_cbc",
    ]
    for name in jobs:
        if wanted and name not in wanted:
            continue
        print(f"=== {name} ===", flush=True)
        if name == "physionet_2012":
            raw = DATASETS / name / "raw"
            retrieval = [
                download("https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz",
                         raw / "set-a.tar.gz", timeout=600),
                download("https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt",
                         raw / "Outcomes-a.txt"),
            ]
            prepare_physionet_2012(raw, retrieval)
        elif name == "physionet_2019":
            raw = DATASETS / name / "raw"
            retrieval = download_ch2019(raw, CAP)
            prepare_physionet_2019(raw, retrieval)
        elif name == "nhanes_htn":
            raw = DATASETS / name / "raw"
            base = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"
            retrieval = [
                download(f"{base}/DEMO_L.xpt", raw / "DEMO_L.xpt"),
                download(f"{base}/BPXO_L.xpt", raw / "BPXO_L.xpt"),
                download(f"{base}/BMX_L.xpt", raw / "BMX_L.xpt"),
                download(f"{base}/BPQ_L.xpt", raw / "BPQ_L.xpt"),
                download(f"{base}/DIQ_L.xpt", raw / "DIQ_L.xpt"),
                download(f"{base}/SMQ_L.xpt", raw / "SMQ_L.xpt"),
            ]
            prepare_nhanes_htn(raw, retrieval)
        elif name == "metabric":
            raw = DATASETS / name / "raw"
            retrieval = [
                download(
                    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_clinical_patient.txt",
                    raw / "data_clinical_patient.txt",
                ),
                download(
                    "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric/data_clinical_sample.txt",
                    raw / "data_clinical_sample.txt",
                ),
            ]
            prepare_metabric(raw, retrieval)
        elif name == "mexico_covid":
            raw = DATASETS / name / "raw"
            retrieval = [download(
                "https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/historicos/2020/COVID19MEXICO2020.zip",
                raw / "COVID19MEXICO2020.zip", timeout=600,
            )]
            prepare_mexico_covid(raw, retrieval)
        elif name == "dryad_er_cbc":
            raw = DATASETS / name / "raw"
            csv_path = raw / "DryadDB_bmjopen_2023_071937_R1_csv.csv"
            urls = [
                "https://datadryad.org/api/v2/files/2639375/download",
                "https://datadryad.org/downloads/file_stream/2639375",
            ]
            retrieval = None
            last_err = None
            for url in urls:
                try:
                    retrieval = [download(url, csv_path, timeout=120)]
                    if csv_path.stat().st_size > 10_000:
                        break
                except Exception as err:
                    last_err = err
            if retrieval is None or csv_path.stat().st_size < 10_000:
                raise SystemExit(
                    f"dryad_er_cbc: place DryadDB_bmjopen_2023_071937_R1_csv.csv in {csv_path} "
                    f"(API download failed: {last_err})"
                )
            prepare_dryad_er_cbc(csv_path, retrieval)


if __name__ == "__main__":
    main()
