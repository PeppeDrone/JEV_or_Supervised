#!/usr/bin/env python3
"""Download and prepare the five UCI clinical datasets for jev_prediction_testing.

Sources are the official UCI Machine Learning Repository static CSVs
(https://archive.ics.uci.edu/static/public/<id>/data.csv). Raw files are kept
under datasets/<name>/raw/; processed UTF-8 patients.csv and dataset.yaml are
written beside them. All transforms are deterministic and safe to re-run.

No imputation, scaling, feature selection, or outcome-guided filtering is
applied here; missing predictors are preserved as blank cells for the existing
training-fold pipelines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

UCI_CSV = {
    "cleveland_heart": (45, "https://archive.ics.uci.edu/static/public/45/data.csv"),
    "uci_ckd": (336, "https://archive.ics.uci.edu/static/public/336/data.csv"),
    "uci_dermatology": (33, "https://archive.ics.uci.edu/static/public/33/data.csv"),
    "uci_hcv": (571, "https://archive.ics.uci.edu/static/public/571/data.csv"),
    "uci_mammographic_mass": (161, "https://archive.ics.uci.edu/static/public/161/data.csv"),
    # Newer UCI static CSV (preferred when present).
    "uci_wdbc": (17, "https://archive.ics.uci.edu/static/public/17/data.csv"),
}

# Datasets without a working static/public data.csv use classic MLDB companions as primary raw.
COMPANIONS = {
    "cleveland_heart": [
        ("processed.cleveland.data",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"),
    ],
    "uci_dermatology": [
        ("dermatology.data",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"),
    ],
    "uci_mammographic_mass": [
        ("mammographic_masses.data",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"),
    ],
    "uci_wdbc": [
        ("wdbc.data",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"),
    ],
    "uci_thyroid": [
        ("hypothyroid.data",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data"),
        ("hypothyroid.names",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.names"),
    ],
    "uci_arrhythmia": [
        ("arrhythmia.data",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"),
        ("arrhythmia.names",
         "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.names"),
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {"url": url, "path": str(dest.relative_to(ROOT)), "sha256": sha256(dest),
                "bytes": dest.stat().st_size, "retrieved": "cached"}
    req = urllib.request.Request(url, headers={"User-Agent": "jev_prediction_testing/uci-prepare"})
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    dest.write_text(text, encoding="utf-8")
    return {"url": url, "path": str(dest.relative_to(ROOT)), "sha256": sha256(dest),
            "bytes": dest.stat().st_size, "retrieved": datetime.now(timezone.utc).isoformat()}


def read_uci_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def blank_missing(series: pd.Series) -> pd.Series:
    """Preserve missingness as empty CSV cells (NaN -> '')."""
    out = series.copy()
    return out.where(out.notna(), "")


def clean_token(value):
    if pd.isna(value):
        return None
    text = str(value).replace("\t", " ").strip()
    if not text or text.lower() in {"nan", "na", "?", "none"}:
        return None
    return text


def dump_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")


def write_patients(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def ensure_raw(name: str) -> tuple[Path, list[dict]]:
    raw_dir = DATASETS / name / "raw"
    meta: list[dict] = []
    primary = raw_dir / "data.csv"
    if name in UCI_CSV:
        _, url = UCI_CSV[name]
        try:
            meta.append(download(url, primary))
        except Exception as error:  # noqa: BLE001 — fall back to companions
            meta.append({"url": url, "path": str(primary.relative_to(ROOT)), "error": str(error)})
    for filename, companion in COMPANIONS.get(name, []):
        try:
            meta.append(download(companion, raw_dir / filename))
        except Exception as error:  # noqa: BLE001 — companions may be optional
            meta.append({"url": companion, "path": str((raw_dir / filename).relative_to(ROOT)),
                         "error": str(error)})
    if primary.exists() and primary.stat().st_size > 0:
        return primary, meta
    # Companion-only datasets: return the first .data companion as the working raw file.
    for filename, _ in COMPANIONS.get(name, []):
        candidate = raw_dir / filename
        if candidate.exists() and candidate.suffix == ".data" and candidate.stat().st_size > 0:
            return candidate, meta
    raise SystemExit(f"{name}: no usable raw file under {raw_dir}")

# ---------------------------------------------------------------------------
# Cleveland Heart Disease
# ---------------------------------------------------------------------------

def prepare_cleveland(raw_csv: Path) -> tuple[pd.DataFrame, dict, dict]:
    src = read_uci_csv(raw_csv)
    # Official UCI cleaned CSV for id=45 is the processed Cleveland cohort (303 rows).
    if len(src) != 303:
        raise SystemExit(f"cleveland_heart: expected 303 rows, got {len(src)}")
    frame = pd.DataFrame({
        "record_id": [f"cleveland_{i+1:03d}" for i in range(len(src))],
        "age": blank_missing(src["age"]),
        "sex": blank_missing(src["sex"].map(lambda v: None if pd.isna(v) else str(int(float(v))))),
        "cp": blank_missing(src["cp"].map(lambda v: None if pd.isna(v) else str(int(float(v))))),
        "trestbps": blank_missing(src["trestbps"]),
        "chol": blank_missing(src["chol"]),
        "fbs": blank_missing(src["fbs"].map(lambda v: None if pd.isna(v) else str(int(float(v))))),
        "restecg": blank_missing(src["restecg"].map(lambda v: None if pd.isna(v) else str(int(float(v))))),
        "thalach": blank_missing(src["thalach"]),
        "exang": blank_missing(src["exang"].map(lambda v: None if pd.isna(v) else str(int(float(v))))),
        "oldpeak": blank_missing(src["oldpeak"]),
        "slope": blank_missing(src["slope"].map(lambda v: None if pd.isna(v) else str(int(float(v))))),
        "ca": blank_missing(src["ca"].map(lambda v: None if pd.isna(v) else str(int(float(v))))),
        "thal": blank_missing(src["thal"].map(lambda v: None if pd.isna(v) else str(int(float(v))))),
        "num": src["num"].map(lambda v: int(float(v))),
    })
    summary = {
        "observations": len(frame),
        "predictors": 13,
        "outcome": "heart_disease",
        "class_counts": {
            "No heart disease": int((frame["num"] == 0).sum()),
            "Heart disease": int((frame["num"] > 0).sum()),
        },
        "identifier_note": (
            "Original patient identifiers were removed by the UCI donors and replaced with "
            "dummy values; record_id enumerates source rows and does not independently verify "
            "unique patients."
        ),
    }
    spec = {
        "name": "cleveland_heart",
        "description": (
            "UCI Heart Disease processed Cleveland cohort (303 observations, 13 clinical "
            "predictors). Binary angiographic heart-disease status from the original num field."
        ),
        "file": "patients.csv",
        "separator": ",",
        "id_column": "record_id",
        "pseudonymize": False,
        "setting": (
            "Cardiology evaluation for suspected coronary artery disease. The prediction uses "
            "baseline clinical examination and non-invasive diagnostic findings available before "
            "angiographic disease status is used as the outcome."
        ),
        "observations_header": "Clinical examination and diagnostic findings:",
        "predictors": {
            "age": {"column": "age", "kind": "numeric", "text_label": "Age", "unit": "years"},
            "sex": {"column": "sex", "kind": "categorical", "text_label": "Sex",
                    "text_values": {"0": "Female", "1": "Male"}},
            "cp": {"column": "cp", "kind": "categorical", "text_label": "Chest pain type",
                   "text_values": {
                       "1": "Typical angina",
                       "2": "Atypical angina",
                       "3": "Non-anginal pain",
                       "4": "Asymptomatic",
                   }},
            "trestbps": {"column": "trestbps", "kind": "numeric",
                         "text_label": "Resting blood pressure on admission", "unit": "mm Hg"},
            "chol": {"column": "chol", "kind": "numeric",
                     "text_label": "Serum cholesterol", "unit": "mg/dl"},
            "fbs": {"column": "fbs", "kind": "categorical",
                    "text_label": "Fasting blood sugar greater than 120 mg/dl",
                    "text_values": {"0": "No", "1": "Yes"}},
            "restecg": {"column": "restecg", "kind": "categorical",
                        "text_label": "Resting electrocardiographic results",
                        "text_values": {
                            "0": "Normal",
                            "1": "ST-T wave abnormality (T-wave inversions and/or ST elevation or depression > 0.05 mV)",
                            "2": "Probable or definite left ventricular hypertrophy by Estes' criteria",
                        }},
            "thalach": {"column": "thalach", "kind": "numeric",
                        "text_label": "Maximum heart rate achieved"},
            "exang": {"column": "exang", "kind": "categorical",
                      "text_label": "Exercise-induced angina",
                      "text_values": {"0": "No", "1": "Yes"}},
            "oldpeak": {"column": "oldpeak", "kind": "numeric",
                        "text_label": "ST depression induced by exercise relative to rest"},
            "slope": {"column": "slope", "kind": "categorical",
                      "text_label": "Slope of the peak exercise ST segment",
                      "text_values": {"1": "Upsloping", "2": "Flat", "3": "Downsloping"}},
            "ca": {"column": "ca", "kind": "numeric",
                   "text_label": "Number of major vessels coloured by fluoroscopy",
                   "scale_note": "Count of major vessels (0–3)"},
            "thal": {"column": "thal", "kind": "categorical",
                     "text_label": "Thallium stress test result",
                     "text_values": {"3": "Normal", "6": "Fixed defect", "7": "Reversible defect"}},
        },
        "outcomes": {
            "heart_disease": {
                "column": "num",
                "classes": ["No heart disease", "Heart disease"],
                "mapping": {
                    "0": "No heart disease",
                    "1": "Heart disease",
                    "2": "Heart disease",
                    "3": "Heart disease",
                    "4": "Heart disease",
                },
                "question": (
                    "Based on the available clinical examination and diagnostic findings, is "
                    "angiographically defined coronary heart disease present?"
                ),
            }
        },
    }
    return frame, spec, summary


# ---------------------------------------------------------------------------
# Chronic Kidney Disease
# ---------------------------------------------------------------------------

def prepare_ckd(raw_csv: Path) -> tuple[pd.DataFrame, dict, dict]:
    src = read_uci_csv(raw_csv)
    if len(src) != 400:
        raise SystemExit(f"uci_ckd: expected 400 rows, got {len(src)}")

    def norm_cat(series):
        return series.map(clean_token)

    # Documented formatting inconsistencies: tab characters inside dm / class labels.
    klass = src["class"].map(lambda v: clean_token(v))
    klass = klass.map(lambda v: "ckd" if v == "ckd" else ("notckd" if v == "notckd" else v))
    if set(klass.dropna()) - {"ckd", "notckd"}:
        raise SystemExit(f"uci_ckd: unexpected class labels {sorted(set(klass.dropna()))}")

    dm = norm_cat(src["dm"]).map(lambda v: "no" if v == "no" else ("yes" if v == "yes" else v))
    frame = pd.DataFrame({
        "record_id": [f"ckd_{i+1:03d}" for i in range(len(src))],
        "age": blank_missing(src["age"]),
        "bp": blank_missing(src["bp"]),
        "sg": blank_missing(src["sg"]),
        "al": blank_missing(src["al"]),
        "su": blank_missing(src["su"]),
        "rbc": blank_missing(norm_cat(src["rbc"])),
        "pc": blank_missing(norm_cat(src["pc"])),
        "pcc": blank_missing(norm_cat(src["pcc"])),
        "ba": blank_missing(norm_cat(src["ba"])),
        "bgr": blank_missing(src["bgr"]),
        "bu": blank_missing(src["bu"]),
        "sc": blank_missing(src["sc"]),
        "sod": blank_missing(src["sod"]),
        "pot": blank_missing(src["pot"]),
        "hemo": blank_missing(src["hemo"]),
        "pcv": blank_missing(src["pcv"]),
        "wc": blank_missing(src["wbcc"]),  # UCI cleaned CSV uses wbcc
        "rc": blank_missing(src["rbcc"]),
        "htn": blank_missing(norm_cat(src["htn"])),
        "dm": blank_missing(dm),
        "cad": blank_missing(norm_cat(src["cad"])),
        "appet": blank_missing(norm_cat(src["appet"])),
        "pe": blank_missing(norm_cat(src["pe"])),
        "ane": blank_missing(norm_cat(src["ane"])),
        "class": klass,
    })
    summary = {
        "observations": len(frame),
        "predictors": 24,
        "outcome": "ckd_status",
        "class_counts": {
            "Chronic kidney disease": int((frame["class"] == "ckd").sum()),
            "No chronic kidney disease": int((frame["class"] == "notckd").sum()),
        },
        "identifier_note": (
            "The UCI CKD release does not provide stable patient identifiers; record_id "
            "enumerates source rows. Unique patients cannot be independently verified."
        ),
        "formatting_notes": [
            "Stripped whitespace and tab characters from nominal fields (notably dm and class).",
            "UCI cleaned CSV columns wbcc/rbcc renamed to wc/rc to match the documented names.",
            "Missing markers preserved as blank; no imputation performed.",
        ],
    }
    albumin_scale = "Urine albumin grade 0–5 as recorded in the original study (ordinal)"
    sugar_scale = "Urine sugar grade 0–5 as recorded in the original study (ordinal)"
    spec = {
        "name": "uci_ckd",
        "description": (
            "UCI Chronic Kidney Disease cohort (400 observations, 24 predictors). Binary "
            "ckd vs notckd labels from the original donation."
        ),
        "file": "patients.csv",
        "separator": ",",
        "id_column": "record_id",
        "pseudonymize": False,
        "setting": (
            "Nephrology assessment using clinical history and laboratory measurements. The "
            "prediction is made from the recorded predictors without using the CKD class label."
        ),
        "observations_header": "Clinical history and laboratory measurements:",
        "predictors": {
            "age": {"column": "age", "kind": "numeric", "text_label": "Age", "unit": "years"},
            "bp": {"column": "bp", "kind": "numeric", "text_label": "Blood pressure", "unit": "mm Hg"},
            "sg": {"column": "sg", "kind": "numeric", "text_label": "Urine specific gravity",
                   "scale_note": "Documented discrete values 1.005, 1.010, 1.015, 1.020, 1.025"},
            "al": {"column": "al", "kind": "numeric", "text_label": "Urine albumin",
                   "scale_note": albumin_scale},
            "su": {"column": "su", "kind": "numeric", "text_label": "Urine sugar",
                   "scale_note": sugar_scale},
            "rbc": {"column": "rbc", "kind": "categorical", "text_label": "Red blood cells in urine",
                    "text_values": {"normal": "Normal", "abnormal": "Abnormal"}},
            "pc": {"column": "pc", "kind": "categorical", "text_label": "Pus cells in urine",
                   "text_values": {"normal": "Normal", "abnormal": "Abnormal"}},
            "pcc": {"column": "pcc", "kind": "categorical", "text_label": "Pus cell clumps",
                    "text_values": {"present": "Present", "notpresent": "Not present"}},
            "ba": {"column": "ba", "kind": "categorical", "text_label": "Bacteria in urine",
                   "text_values": {"present": "Present", "notpresent": "Not present"}},
            "bgr": {"column": "bgr", "kind": "numeric", "text_label": "Blood glucose (random)",
                    "unit": "mg/dl"},
            "bu": {"column": "bu", "kind": "numeric", "text_label": "Blood urea", "unit": "mg/dl"},
            "sc": {"column": "sc", "kind": "numeric", "text_label": "Serum creatinine", "unit": "mg/dl"},
            "sod": {"column": "sod", "kind": "numeric", "text_label": "Sodium", "unit": "mEq/L"},
            "pot": {"column": "pot", "kind": "numeric", "text_label": "Potassium", "unit": "mEq/L"},
            "hemo": {"column": "hemo", "kind": "numeric", "text_label": "Hemoglobin", "unit": "g"},
            "pcv": {"column": "pcv", "kind": "numeric", "text_label": "Packed cell volume"},
            "wc": {"column": "wc", "kind": "numeric", "text_label": "White blood cell count",
                   "unit": "cells/cumm"},
            "rc": {"column": "rc", "kind": "numeric", "text_label": "Red blood cell count",
                   "unit": "millions/cmm"},
            "htn": {"column": "htn", "kind": "categorical", "text_label": "Hypertension",
                    "text_values": {"yes": "Yes", "no": "No"}},
            "dm": {"column": "dm", "kind": "categorical", "text_label": "Diabetes mellitus",
                   "text_values": {"yes": "Yes", "no": "No"}},
            "cad": {"column": "cad", "kind": "categorical", "text_label": "Coronary artery disease",
                    "text_values": {"yes": "Yes", "no": "No"}},
            "appet": {"column": "appet", "kind": "categorical", "text_label": "Appetite",
                      "text_values": {"good": "Good", "poor": "Poor"}},
            "pe": {"column": "pe", "kind": "categorical", "text_label": "Pedal edema",
                   "text_values": {"yes": "Yes", "no": "No"}},
            "ane": {"column": "ane", "kind": "categorical", "text_label": "Anemia",
                    "text_values": {"yes": "Yes", "no": "No"}},
        },
        "outcomes": {
            "ckd_status": {
                "column": "class",
                "classes": ["No chronic kidney disease", "Chronic kidney disease"],
                "mapping": {
                    "notckd": "No chronic kidney disease",
                    "ckd": "Chronic kidney disease",
                },
                "question": (
                    "Based on the available clinical history and laboratory measurements, does "
                    "this patient have chronic kidney disease?"
                ),
            }
        },
    }
    return frame, spec, summary


# ---------------------------------------------------------------------------
# Dermatology
# ---------------------------------------------------------------------------

SEVERITY_NOTE = (
    "Clinical/histopathological severity score 0–3: 0 = feature not present; "
    "1–2 = relative intermediate amounts; 3 = largest amount possible"
)

DERM_FEATURES = [
    ("erythema", "erythema", "Erythema"),
    ("scaling", "scaling", "Scaling"),
    ("definite_borders", "definite-borders", "Definite borders"),
    ("itching", "itching", "Itching"),
    ("koebner_phenomenon", "koebner phenomenon", "Koebner phenomenon"),
    ("polygonal_papules", "polygonal papules", "Polygonal papules"),
    ("follicular_papules", "follicular papules", "Follicular papules"),
    ("oral_mucosal_involvement", "oral-mucosal involvement", "Oral mucosal involvement"),
    ("knee_elbow_involvement", "knee elbow involvement", "Knee and elbow involvement"),
    ("scalp_involvement", "scalp involvement", "Scalp involvement"),
    ("family_history", "family history", "Family history of these diseases"),
    ("melanin_incontinence", "melanin incontinence", "Melanin incontinence"),
    ("eosinophils_in_infiltrate", "eosinophils in the infiltrate", "Eosinophils in the infiltrate"),
    ("pnl_infiltrate", "pnl infiltrate", "Polymorphonuclear leukocyte infiltrate"),
    ("fibrosis_papillary_dermis", "fibrosis of the papillary dermis", "Fibrosis of the papillary dermis"),
    ("exocytosis", "exocytosis", "Exocytosis"),
    ("acanthosis", "acanthosis", "Acanthosis"),
    ("hyperkeratosis", "hyperkeratosis", "Hyperkeratosis"),
    ("parakeratosis", "parakeratosis", "Parakeratosis"),
    ("clubbing_rete_ridges", "clubbing of the rete ridges", "Clubbing of the rete ridges"),
    ("elongation_rete_ridges", "elongation of the rete ridges", "Elongation of the rete ridges"),
    ("thinning_suprapapillary", "thinning of the suprapapillary epidermis",
     "Thinning of the suprapapillary epidermis"),
    ("spongiform_pustule", "spongiform pustule", "Spongiform pustule"),
    ("munro_microabscess", "munro microabcess", "Munro microabscess"),
    ("focal_hypergranulosis", "focal hypergranulosis", "Focal hypergranulosis"),
    ("disappearance_granular_layer", "disappearance of the granular layer",
     "Disappearance of the granular layer"),
    ("vacuolisation_basal_layer", "vacuolisation and damage of the basal layer",
     "Vacuolisation and damage of the basal layer"),
    ("spongiosis", "spongiosis", "Spongiosis"),
    ("saw_tooth_retes", "saw-tooth appearance of retes", "Saw-tooth appearance of retes"),
    ("follicular_horn_plug", "follicular horn plug", "Follicular horn plug"),
    ("perifollicular_parakeratosis", "perifollicular parakeratosis", "Perifollicular parakeratosis"),
    ("inflammatory_mononuclear", "inflammatory monoluclear infiltrate",
     "Inflammatory mononuclear infiltrate"),
    ("band_like_infiltrate", "band-like infiltrate", "Band-like infiltrate"),
]


def prepare_dermatology(raw_csv: Path) -> tuple[pd.DataFrame, dict, dict]:
    src = read_uci_csv(raw_csv)
    if len(src) != 366:
        raise SystemExit(f"uci_dermatology: expected 366 rows, got {len(src)}")
    data = {"record_id": [f"derm_{i+1:03d}" for i in range(len(src))]}
    predictors = {}
    for out_col, src_col, label in DERM_FEATURES:
        if out_col == "family_history":
            data[out_col] = blank_missing(src[src_col].map(
                lambda v: None if pd.isna(v) else str(int(float(v)))))
            predictors[out_col] = {
                "column": out_col, "kind": "categorical", "text_label": label,
                "text_values": {
                    "0": "No family history of these diseases",
                    "1": "Family history of these diseases present",
                },
            }
        else:
            data[out_col] = blank_missing(src[src_col])
            predictors[out_col] = {
                "column": out_col, "kind": "numeric", "text_label": label,
                "scale_note": SEVERITY_NOTE,
            }
    data["age"] = blank_missing(src["age"])
    predictors["age"] = {"column": "age", "kind": "numeric", "text_label": "Age", "unit": "years"}
    data["class"] = src["class"].map(lambda v: int(float(v)))
    frame = pd.DataFrame(data)
    classes = [
        "Psoriasis",
        "Seborrhoeic dermatitis",
        "Lichen planus",
        "Pityriasis rosea",
        "Chronic dermatitis",
        "Pityriasis rubra pilaris",
    ]
    mapping = {str(i + 1): name for i, name in enumerate(classes)}
    summary = {
        "observations": len(frame),
        "predictors": 34,
        "outcome": "dermatological_diagnosis",
        "class_counts": {mapping[str(k)]: int((frame["class"] == k).sum()) for k in range(1, 7)},
        "identifier_note": (
            "Patient names and id numbers were removed from the UCI dermatology release; "
            "record_id enumerates source rows."
        ),
    }
    spec = {
        "name": "uci_dermatology",
        "description": (
            "UCI Dermatology erythemato-squamous disease cohort (366 observations, 34 predictors: "
            "12 clinical and 22 histopathological features plus age and family history)."
        ),
        "file": "patients.csv",
        "separator": ",",
        "id_column": "record_id",
        "pseudonymize": False,
        "setting": (
            "Dermatology differential diagnosis of erythemato-squamous diseases after clinical "
            "examination and histopathological evaluation of a skin biopsy."
        ),
        "observations_header": "Clinical examination and histopathological findings:",
        "predictors": predictors,
        "outcomes": {
            "dermatological_diagnosis": {
                "column": "class",
                "classes": classes,
                "mapping": mapping,
                "question": (
                    "Based on the clinical examination and histopathological findings, which of "
                    "the six erythemato-squamous diseases is the most likely diagnosis?"
                ),
            }
        },
    }
    return frame, spec, summary


# ---------------------------------------------------------------------------
# Hepatitis C
# ---------------------------------------------------------------------------

def prepare_hcv(raw_csv: Path) -> tuple[pd.DataFrame, dict, dict]:
    src = read_uci_csv(raw_csv)
    if len(src) != 615:
        raise SystemExit(f"uci_hcv: expected 615 source rows, got {len(src)}")
    category = src["Category"].map(clean_token)
    keep = category != "0s=suspect Blood Donor"
    src = src.loc[keep].reset_index(drop=True)
    category = category.loc[keep].reset_index(drop=True)
    if len(src) != 608:
        raise SystemExit(f"uci_hcv: expected 608 rows after excluding suspect donors, got {len(src)}")

    # UCI cleaned CSV names the GGT column "CGT" (documented as GGT).
    ggt_col = "CGT" if "CGT" in src.columns else "GGT"
    frame = pd.DataFrame({
        "record_id": src["ID"].map(lambda v: f"hcv_{int(v):03d}"),
        "age": blank_missing(src["Age"]),
        "sex": blank_missing(src["Sex"].map(clean_token)),
        "albumin": blank_missing(src["ALB"]),
        "alkaline_phosphatase": blank_missing(src["ALP"]),
        "alt": blank_missing(src["ALT"]),
        "ast": blank_missing(src["AST"]),
        "bilirubin": blank_missing(src["BIL"]),
        "cholinesterase": blank_missing(src["CHE"]),
        "cholesterol": blank_missing(src["CHOL"]),
        "creatinine": blank_missing(src["CREA"]),
        "ggt": blank_missing(src[ggt_col]),
        "protein": blank_missing(src["PROT"]),
        "category": category,
    })
    classes = ["Blood donor", "Hepatitis", "Fibrosis", "Cirrhosis"]
    mapping = {
        "0=Blood Donor": "Blood donor",
        "1=Hepatitis": "Hepatitis",
        "2=Fibrosis": "Fibrosis",
        "3=Cirrhosis": "Cirrhosis",
    }
    summary = {
        "observations": len(frame),
        "predictors": 12,
        "outcome": "hcv_category",
        "excluded": {"0s=suspect Blood Donor": 7},
        "class_counts": {mapping[k]: int((frame["category"] == k).sum()) for k in mapping},
        "identifier_note": (
            "Source ID values from the UCI HCV release are retained as record_id prefixes; "
            "the seven suspect blood-donor rows were excluded because that class has fewer "
            "than ten observations."
        ),
        "units_note": (
            "Original documentation does not state laboratory units for ALB/ALP/ALT/AST/BIL/"
            "CHE/CHOL/CREA/GGT/PROT; units are therefore omitted from the prompt labels."
        ),
    }
    spec = {
        "name": "uci_hcv",
        "description": (
            "UCI HCV laboratory cohort after excluding seven suspect blood-donor observations "
            "(608 remaining; four diagnostic categories)."
        ),
        "file": "patients.csv",
        "separator": ",",
        "id_column": "record_id",
        "pseudonymize": False,
        "setting": (
            "Laboratory and demographic assessment distinguishing blood donors from hepatitis C "
            "disease categories (hepatitis, fibrosis, cirrhosis)."
        ),
        "observations_header": "Demographic and laboratory findings:",
        "predictors": {
            "age": {"column": "age", "kind": "numeric", "text_label": "Age", "unit": "years"},
            "sex": {"column": "sex", "kind": "categorical", "text_label": "Sex",
                    "text_values": {"m": "Male", "f": "Female"}},
            "albumin": {"column": "albumin", "kind": "numeric", "text_label": "Albumin"},
            "alkaline_phosphatase": {"column": "alkaline_phosphatase", "kind": "numeric",
                                     "text_label": "Alkaline phosphatase"},
            "alt": {"column": "alt", "kind": "numeric", "text_label": "Alanine aminotransferase"},
            "ast": {"column": "ast", "kind": "numeric", "text_label": "Aspartate aminotransferase"},
            "bilirubin": {"column": "bilirubin", "kind": "numeric", "text_label": "Bilirubin"},
            "cholinesterase": {"column": "cholinesterase", "kind": "numeric",
                               "text_label": "Cholinesterase"},
            "cholesterol": {"column": "cholesterol", "kind": "numeric", "text_label": "Cholesterol"},
            "creatinine": {"column": "creatinine", "kind": "numeric", "text_label": "Creatinine"},
            "ggt": {"column": "ggt", "kind": "numeric",
                    "text_label": "Gamma-glutamyl transferase"},
            "protein": {"column": "protein", "kind": "numeric", "text_label": "Total protein"},
        },
        "outcomes": {
            "hcv_category": {
                "column": "category",
                "classes": classes,
                "mapping": mapping,
                "question": (
                    "Based on the demographic and laboratory findings, which diagnostic category "
                    "best describes this individual: blood donor, hepatitis, fibrosis, or cirrhosis?"
                ),
            }
        },
    }
    return frame, spec, summary


# ---------------------------------------------------------------------------
# Mammographic Mass
# ---------------------------------------------------------------------------

def prepare_mammographic(raw_csv: Path) -> tuple[pd.DataFrame, dict, dict]:
    src = read_uci_csv(raw_csv)
    if len(src) != 961:
        raise SystemExit(f"uci_mammographic_mass: expected 961 rows, got {len(src)}")

    def code(series):
        return series.map(lambda v: None if pd.isna(v) else str(int(float(v))))

    # BI-RADS is explicitly non-predictive in the UCI task definition and is excluded.
    frame = pd.DataFrame({
        "record_id": [f"mammo_{i+1:03d}" for i in range(len(src))],
        "age": blank_missing(src["Age"]),
        "shape": blank_missing(code(src["Shape"])),
        "margin": blank_missing(code(src["Margin"])),
        "density": blank_missing(code(src["Density"])),
        "severity": src["Severity"].map(lambda v: int(float(v))),
    })
    summary = {
        "observations": len(frame),
        "predictors": 4,
        "outcome": "malignancy",
        "excluded_predictors": ["BI-RADS"],
        "class_counts": {
            "Benign": int((frame["severity"] == 0).sum()),
            "Malignant": int((frame["severity"] == 1).sum()),
        },
        "identifier_note": (
            "Each row is a mammographic mass lesion, not necessarily a unique patient. "
            "record_id enumerates source lesions; unique patients cannot be independently verified."
        ),
    }
    spec = {
        "name": "uci_mammographic_mass",
        "description": (
            "UCI Mammographic Mass lesions (961 observations). Predictors are age, shape, margin "
            "and density; BI-RADS assessment is excluded as non-predictive per the UCI documentation."
        ),
        "file": "patients.csv",
        "separator": ",",
        "id_column": "record_id",
        "pseudonymize": False,
        "setting": (
            "Mammographic assessment of a breast mass lesion using age and BI-RADS attribute "
            "descriptors, before biopsy severity is known."
        ),
        "observations_header": "Patient age and mammographic mass characteristics:",
        "predictors": {
            "age": {"column": "age", "kind": "numeric", "text_label": "Age", "unit": "years"},
            "shape": {"column": "shape", "kind": "categorical", "text_label": "Mass shape",
                      "text_values": {
                          "1": "Round", "2": "Oval", "3": "Lobular", "4": "Irregular",
                      }},
            "margin": {"column": "margin", "kind": "categorical", "text_label": "Mass margin",
                       "text_values": {
                           "1": "Circumscribed",
                           "2": "Microlobulated",
                           "3": "Obscured",
                           "4": "Ill-defined",
                           "5": "Spiculated",
                       }},
            "density": {"column": "density", "kind": "categorical", "text_label": "Mass density",
                        "text_values": {
                            "1": "High",
                            "2": "Iso-dense",
                            "3": "Low",
                            "4": "Fat-containing",
                        }},
        },
        "outcomes": {
            "malignancy": {
                "column": "severity",
                "classes": ["Benign", "Malignant"],
                "mapping": {"0": "Benign", "1": "Malignant"},
                "question": (
                    "Based on the patient's age and the reported mammographic mass characteristics, "
                    "is the lesion benign or malignant?"
                ),
            }
        },
    }
    return frame, spec, summary


# ---------------------------------------------------------------------------
# Wisconsin Breast Cancer Diagnostic (WDBC)
# ---------------------------------------------------------------------------

_WDBC_STEMS = (
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
)
_WDBC_STATS = (("1", "mean"), ("2", "standard error"), ("3", "worst"))


def prepare_wdbc(raw_csv: Path) -> tuple[pd.DataFrame, dict, dict]:
    src = read_uci_csv(raw_csv)
    if "Diagnosis" not in src.columns:
        raise SystemExit("uci_wdbc: Diagnosis column missing")
    if len(src) != 569:
        raise SystemExit(f"uci_wdbc: expected 569 rows, got {len(src)}")

    frame = pd.DataFrame({"record_id": src["ID"].map(lambda v: f"wdbc_{int(v)}")})
    predictors = {}
    for stem in _WDBC_STEMS:
        for suffix, stat in _WDBC_STATS:
            col = f"{stem}{suffix}"
            if col not in src.columns:
                raise SystemExit(f"uci_wdbc: missing column {col}")
            key = f"{stem}_{stat.replace(' ', '_')}"
            frame[key] = blank_missing(src[col])
            label = stem.replace("_", " ")
            predictors[key] = {
                "column": key,
                "kind": "numeric",
                "text_label": f"Nucleus {label} ({stat})",
            }
    frame["diagnosis"] = src["Diagnosis"].map(lambda v: str(v).strip().upper())
    if set(frame["diagnosis"]) - {"B", "M"}:
        raise SystemExit(f"uci_wdbc: unexpected diagnoses {set(frame['diagnosis'])}")

    summary = {
        "observations": len(frame),
        "predictors": len(predictors),
        "outcome": "diagnosis",
        "class_counts": {
            "Benign": int((frame["diagnosis"] == "B").sum()),
            "Malignant": int((frame["diagnosis"] == "M").sum()),
        },
        "identifier_note": (
            "record_id preserves the original WDBC case ID; each row is one fine-needle "
            "aspirate image-derived feature vector."
        ),
    }
    spec = {
        "name": "uci_wdbc",
        "description": (
            "UCI Breast Cancer Wisconsin Diagnostic (WDBC) cohort (569 observations, 30 "
            "real-valued nucleus features from fine-needle aspirate images)."
        ),
        "file": "patients.csv",
        "separator": ",",
        "id_column": "record_id",
        "pseudonymize": False,
        "setting": (
            "Breast oncology assessment of a fine-needle aspirate of a breast mass using "
            "digitised cytological nucleus features, before biopsy diagnosis is confirmed."
        ),
        "observations_header": "Fine-needle aspirate nucleus morphometry:",
        "predictors": predictors,
        "outcomes": {
            "diagnosis": {
                "column": "diagnosis",
                "classes": ["Benign", "Malignant"],
                "mapping": {"B": "Benign", "M": "Malignant"},
                "question": (
                    "Based on the fine-needle aspirate nucleus morphometric features, is the "
                    "breast mass benign or malignant?"
                ),
            }
        },
    }
    return frame, spec, summary


# ---------------------------------------------------------------------------
# Thyroid Disease — hypothyroid vs negative (with native missingness)
# ---------------------------------------------------------------------------

_THYROID_COLS = [
    "class", "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "thyroid_surgery", "query_hypothyroid", "query_hyperthyroid", "pregnant", "sick", "tumor",
    "lithium", "goitre", "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",
]


def prepare_thyroid(raw_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    src = pd.read_csv(raw_path, header=None, na_values=["?"], names=_THYROID_COLS, encoding="utf-8")
    if len(src) != 3163:
        raise SystemExit(f"uci_thyroid: expected 3163 rows, got {len(src)}")

    def yn(series):
        return series.map(lambda v: None if pd.isna(v) else str(v).strip().lower())

    frame = pd.DataFrame({"record_id": [f"thyroid_{i+1:04d}" for i in range(len(src))]})
    frame["age"] = blank_missing(src["age"])
    frame["sex"] = blank_missing(yn(src["sex"]).map({"m": "M", "f": "F"}))
    bool_cols = [
        "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "thyroid_surgery",
        "query_hypothyroid", "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium", "goitre",
    ]
    for col in bool_cols:
        frame[col] = blank_missing(yn(src[col]).map({"t": "1", "f": "0", "y": "1", "n": "0"}))
    for flag, value in [
        ("TSH_measured", "TSH"), ("T3_measured", "T3"), ("TT4_measured", "TT4"),
        ("T4U_measured", "T4U"), ("FTI_measured", "FTI"), ("TBG_measured", "TBG"),
    ]:
        frame[flag] = blank_missing(yn(src[flag]).map({"t": "1", "f": "0", "y": "1", "n": "0"}))
        frame[value] = blank_missing(src[value])
    frame["thyroid_class"] = src["class"].map(lambda v: str(v).strip().lower())
    if set(frame["thyroid_class"]) - {"hypothyroid", "negative"}:
        raise SystemExit(f"uci_thyroid: unexpected classes {set(frame['thyroid_class'])}")

    tf = {"0": "No", "1": "Yes"}
    predictors = {
        "age": {"column": "age", "kind": "numeric", "text_label": "Age", "unit": "years"},
        "sex": {"column": "sex", "kind": "categorical", "text_label": "Sex",
                "text_values": {"M": "Male", "F": "Female"}},
    }
    labels = {
        "on_thyroxine": "Currently on thyroxine",
        "query_on_thyroxine": "Query on thyroxine",
        "on_antithyroid_medication": "On antithyroid medication",
        "thyroid_surgery": "History of thyroid surgery",
        "query_hypothyroid": "Query hypothyroid",
        "query_hyperthyroid": "Query hyperthyroid",
        "pregnant": "Pregnant",
        "sick": "Currently sick",
        "tumor": "Suspected tumour",
        "lithium": "On lithium",
        "goitre": "Goitre",
        "TSH_measured": "TSH measured",
        "T3_measured": "T3 measured",
        "TT4_measured": "Total T4 measured",
        "T4U_measured": "T4U measured",
        "FTI_measured": "Free thyroxine index measured",
        "TBG_measured": "TBG measured",
    }
    for col, label in labels.items():
        predictors[col] = {"column": col, "kind": "categorical", "text_label": label, "text_values": tf}
    for col, label, unit in [
        ("TSH", "TSH", "mU/L"), ("T3", "T3", "µg/dL"), ("TT4", "Total T4", "µg/dL"),
        ("T4U", "T4U", None), ("FTI", "Free thyroxine index", None), ("TBG", "TBG", None),
    ]:
        predictors[col] = {"column": col, "kind": "numeric", "text_label": label, **({"unit": unit} if unit else {})}

    summary = {
        "observations": len(frame),
        "predictors": len(predictors),
        "outcome": "hypothyroid_status",
        "class_counts": {
            "No hypothyroidism": int((frame["thyroid_class"] == "negative").sum()),
            "Hypothyroidism": int((frame["thyroid_class"] == "hypothyroid").sum()),
        },
        "identifier_note": (
            "record_id enumerates hypothyroid.data rows; native '?' missingness is preserved "
            "as blank cells. Primary UCI Thyroid Disease hypothyroid task (binary)."
        ),
    }
    spec = {
        "name": "uci_thyroid",
        "description": (
            "UCI Thyroid Disease hypothyroid cohort (3163 observations). Binary hypothyroid "
            "vs negative labels with clinical flags and thyroid function tests; missing lab "
            "values retained as blank cells."
        ),
        "file": "patients.csv",
        "separator": ",",
        "id_column": "record_id",
        "pseudonymize": False,
        "setting": (
            "Endocrinology evaluation for suspected hypothyroidism using clinical context "
            "and thyroid function tests."
        ),
        "observations_header": "Clinical context and thyroid function tests:",
        "predictors": predictors,
        "outcomes": {
            "hypothyroid_status": {
                "column": "thyroid_class",
                "classes": ["No hypothyroidism", "Hypothyroidism"],
                "mapping": {"negative": "No hypothyroidism", "hypothyroid": "Hypothyroidism"},
                "question": (
                    "Based on the clinical context and thyroid function tests, does the patient "
                    "have hypothyroidism?"
                ),
            }
        },
    }
    return frame, spec, summary


# ---------------------------------------------------------------------------
# Arrhythmia (high-dimensional ECG; rare classes with n<10 excluded)
# ---------------------------------------------------------------------------

_ARRHYTHMIA_CLASS = {
    1: "Normal",
    2: "Ischemic changes (coronary artery disease)",
    3: "Old anterior myocardial infarction",
    4: "Old inferior myocardial infarction",
    5: "Sinus tachycardia",
    6: "Sinus bradycardia",
    10: "Right bundle branch block",
    16: "Unclassified arrhythmia",
}
_ARRHYTHMIA_HEAD = [
    ("age", "Age", "years", "numeric"),
    ("sex", "Sex", None, "categorical"),
    ("height", "Height", "cm", "numeric"),
    ("weight", "Weight", "kg", "numeric"),
    ("qrs_duration", "Average QRS duration", "ms", "numeric"),
    ("pr_interval", "Average P-R interval", "ms", "numeric"),
    ("qt_interval", "Average Q-T interval", "ms", "numeric"),
    ("t_interval", "Average T-wave duration", "ms", "numeric"),
    ("p_interval", "Average P-wave duration", "ms", "numeric"),
    ("qrs_angle", "QRS angle on front plane", "degrees", "numeric"),
    ("t_angle", "T angle on front plane", "degrees", "numeric"),
    ("p_angle", "P angle on front plane", "degrees", "numeric"),
    ("qrst_angle", "QRST angle on front plane", "degrees", "numeric"),
    ("j_angle", "J angle on front plane", "degrees", "numeric"),
    ("heart_rate", "Heart rate", "beats/min", "numeric"),
]


def prepare_arrhythmia(raw_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    src = pd.read_csv(raw_path, header=None, na_values=["?"], encoding="utf-8")
    if src.shape[1] != 280:
        raise SystemExit(f"uci_arrhythmia: expected 280 columns, got {src.shape[1]}")
    y = src.iloc[:, -1].map(lambda v: int(float(v)))
    keep = y.isin(_ARRHYTHMIA_CLASS)
    dropped = int((~keep).sum())
    src = src.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    frame = pd.DataFrame({"record_id": [f"arrhythmia_{i+1:03d}" for i in range(len(src))]})
    predictors = {}
    for i, (key, label, unit, kind) in enumerate(_ARRHYTHMIA_HEAD):
        series = src.iloc[:, i]
        if key == "sex":
            frame[key] = blank_missing(series.map(lambda v: None if pd.isna(v) else str(int(float(v)))))
            predictors[key] = {
                "column": key, "kind": "categorical", "text_label": label,
                "text_values": {"0": "Male", "1": "Female"},
            }
        else:
            frame[key] = blank_missing(series)
            predictors[key] = {"column": key, "kind": "numeric", "text_label": label,
                               **({"unit": unit} if unit else {})}
    for j in range(15, 279):
        key = f"ecg_attr_{j+1:03d}"
        frame[key] = blank_missing(src.iloc[:, j])
        predictors[key] = {
            "column": key,
            "kind": "numeric",
            "text_label": f"ECG attribute {j+1}",
            "scale_note": "Channel/wave amplitude or presence feature from the UCI Arrhythmia encoding",
        }
    frame["arrhythmia_code"] = y.map(lambda v: str(int(v)))

    class_counts = {name: int((y == code).sum()) for code, name in _ARRHYTHMIA_CLASS.items()}
    summary = {
        "observations": len(frame),
        "predictors": len(predictors),
        "outcome": "arrhythmia_class",
        "class_counts": class_counts,
        "excluded_rare_classes_n_lt_10": dropped,
        "identifier_note": (
            "record_id enumerates retained arrhythmia.data rows after dropping arrhythmia "
            "types with fewer than 10 examples (study min_class_count). Original patient "
            "identifiers were already removed upstream by UCI."
        ),
    }
    spec = {
        "name": "uci_arrhythmia",
        "description": (
            f"UCI Cardiac Arrhythmia cohort ({len(frame)} observations after excluding rare "
            f"classes with n<10; {dropped} rows dropped). 279 ECG-derived predictors; "
            "8 retained rhythm classes."
        ),
        "file": "patients.csv",
        "separator": ",",
        "id_column": "record_id",
        "pseudonymize": False,
        "setting": (
            "Cardiology ECG interpretation to classify cardiac rhythm / arrhythmia type from "
            "standard electrocardiographic measurements."
        ),
        "observations_header": "Electrocardiographic measurements:",
        "predictors": predictors,
        "outcomes": {
            "arrhythmia_class": {
                "column": "arrhythmia_code",
                "classes": list(_ARRHYTHMIA_CLASS.values()),
                "mapping": {str(k): v for k, v in _ARRHYTHMIA_CLASS.items()},
                "question": (
                    "Based on the electrocardiographic measurements, which cardiac rhythm / "
                    "arrhythmia class is present?"
                ),
            }
        },
    }
    return frame, spec, summary


PREPARERS = {
    "cleveland_heart": prepare_cleveland,
    "uci_ckd": prepare_ckd,
    "uci_dermatology": prepare_dermatology,
    "uci_hcv": prepare_hcv,
    "uci_mammographic_mass": prepare_mammographic,
    "uci_wdbc": prepare_wdbc,
    "uci_thyroid": prepare_thyroid,
    "uci_arrhythmia": prepare_arrhythmia,
}


def prepare_one(name: str) -> dict:
    raw_csv, retrieval = ensure_raw(name)
    frame, spec, summary = PREPARERS[name](raw_csv)
    out_dir = DATASETS / name
    patients = out_dir / "patients.csv"
    write_patients(patients, frame)
    dump_yaml(out_dir / "dataset.yaml", spec)
    notes = {
        "dataset": name,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_retrieval": retrieval,
        "patients_csv_sha256": sha256(patients),
        "summary": summary,
    }
    (out_dir / "PREPARATION.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {name}",
        "",
        f"- Observations: {summary['observations']}",
        f"- Predictors: {summary['predictors']}",
        f"- Outcome: {summary['outcome']}",
        f"- Class counts: {summary['class_counts']}",
        f"- Identifier note: {summary['identifier_note']}",
        "",
        "Raw sources and SHA256 checksums are recorded in PREPARATION.json.",
        "No imputation, scaling or outcome-guided feature selection was applied.",
        "",
    ]
    (out_dir / "DATASET_NOTES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{name}: {summary['observations']} rows, {summary['predictors']} predictors, "
          f"classes={summary['class_counts']}")
    return notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=list(PREPARERS),
                        help="Prepare only the named dataset (default: all five)")
    args = parser.parse_args(argv)
    names = args.dataset or list(PREPARERS)
    report = {name: prepare_one(name) for name in names}
    out = ROOT / "outputs" / "uci_preparation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
