#!/usr/bin/env python3
"""Prepare WiDS Datathon 2021 diabetes_mellitus task for jevtest.

Uses TrainingWiDS2021.csv only (competition test is unlabelled).
Restrictions (per study plan):
  - encounter-level rows are unique; sample stratified cohort for LLM scale
  - drop identifiers and hospital/ICU ids
  - drop APACHE diagnosis codes (may encode diabetes)
  - drop glucose labs (near-label for *existing* diabetes status)
  - keep a fixed clinically meaningful demographic / acuity / comorbidity set

Source: Kaggle dataset mirror (official competition download requires joining
the competition / accepting rules — API returns 403 until then).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "wids2021" / "raw" / "TrainingWiDS2021.csv"
OUT = ROOT / "datasets" / "wids2021"
SEED = 20261008
N_SAMPLE = 5000  # None: whole table (scripts/prepare_full_tables.py)

# Fixed clinically meaningful subset (no glucose, no diagnosis codes, no ids).
KEEP = [
    "age", "bmi", "elective_surgery", "ethnicity", "gender", "height", "weight",
    "pre_icu_los_days",
    "bun_apache", "creatinine_apache", "sodium_apache", "wbc_apache", "hematocrit_apache",
    "heart_rate_apache", "map_apache", "resprate_apache", "temp_apache",
    "gcs_eyes_apache", "gcs_motor_apache", "gcs_verbal_apache", "gcs_unable_apache",
    "intubated_apache", "ventilated_apache", "arf_apache", "apache_post_operative",
    "aids", "cirrhosis", "hepatic_failure", "immunosuppression", "leukemia", "lymphoma",
    "solid_tumor_with_metastasis",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blank(s: pd.Series) -> pd.Series:
    return s.where(s.notna(), "")


def main():
    if not RAW.exists():
        raise SystemExit(f"Missing {RAW}; download TrainingWiDS2021.csv into datasets/wids2021/raw/")
    src = pd.read_csv(RAW, low_memory=False)
    missing = [c for c in KEEP + ["diabetes_mellitus", "encounter_id"] if c not in src.columns]
    if missing:
        raise SystemExit(f"WiDS columns missing: {missing}")

    y = src["diabetes_mellitus"].astype(int)
    if N_SAMPLE is None:
        cohort = src.reset_index(drop=True)
    else:
        rng = np.random.default_rng(SEED)
        parts = []
        for label, group in src.groupby(y):
            take = min(len(group), max(1, int(round(N_SAMPLE * (len(group) / len(src))))))
            idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
            parts.append(group.loc[idx])
        cohort = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        if len(cohort) > N_SAMPLE:
            cohort = cohort.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)

    frame = pd.DataFrame({"record_id": [f"wids_{int(eid)}" for eid in cohort["encounter_id"]]})
    predictors = {}

    cat_maps = {
        "ethnicity": None,  # filled from data
        "gender": {"M": "Male", "F": "Female"},
        "elective_surgery": {"0": "No", "1": "Yes"},
        "gcs_unable_apache": {"0": "No", "1": "Yes"},
        "intubated_apache": {"0": "No", "1": "Yes"},
        "ventilated_apache": {"0": "No", "1": "Yes"},
        "arf_apache": {"0": "No", "1": "Yes"},
        "apache_post_operative": {"0": "Non-operative", "1": "Post-operative"},
        "aids": {"0": "No", "1": "Yes"},
        "cirrhosis": {"0": "No", "1": "Yes"},
        "hepatic_failure": {"0": "No", "1": "Yes"},
        "immunosuppression": {"0": "No", "1": "Yes"},
        "leukemia": {"0": "No", "1": "Yes"},
        "lymphoma": {"0": "No", "1": "Yes"},
        "solid_tumor_with_metastasis": {"0": "No", "1": "Yes"},
    }
    labels = {
        "age": ("Age", "years", "numeric"),
        "bmi": ("Body mass index", "kg/m^2", "numeric"),
        "elective_surgery": ("Elective surgery", None, "categorical"),
        "ethnicity": ("Ethnicity", None, "categorical"),
        "gender": ("Sex", None, "categorical"),
        "height": ("Height", "cm", "numeric"),
        "weight": ("Weight", "kg", "numeric"),
        "pre_icu_los_days": ("Pre-ICU length of stay", "days", "numeric"),
        "bun_apache": ("BUN (APACHE)", "mg/dL", "numeric"),
        "creatinine_apache": ("Creatinine (APACHE)", "mg/dL", "numeric"),
        "sodium_apache": ("Sodium (APACHE)", "mEq/L", "numeric"),
        "wbc_apache": ("White blood cell count (APACHE)", "10^9/L", "numeric"),
        "hematocrit_apache": ("Hematocrit (APACHE)", "%", "numeric"),
        "heart_rate_apache": ("Heart rate (APACHE)", "beats/min", "numeric"),
        "map_apache": ("Mean arterial pressure (APACHE)", "mm Hg", "numeric"),
        "resprate_apache": ("Respiratory rate (APACHE)", "breaths/min", "numeric"),
        "temp_apache": ("Temperature (APACHE)", "°C", "numeric"),
        "gcs_eyes_apache": ("GCS eye opening (APACHE)", None, "numeric"),
        "gcs_motor_apache": ("GCS motor (APACHE)", None, "numeric"),
        "gcs_verbal_apache": ("GCS verbal (APACHE)", None, "numeric"),
        "gcs_unable_apache": ("GCS unable to assess", None, "categorical"),
        "intubated_apache": ("Intubated (APACHE ABG)", None, "categorical"),
        "ventilated_apache": ("Invasively ventilated", None, "categorical"),
        "arf_apache": ("Acute renal failure", None, "categorical"),
        "apache_post_operative": ("APACHE operative status", None, "categorical"),
        "aids": ("AIDS", None, "categorical"),
        "cirrhosis": ("Cirrhosis", None, "categorical"),
        "hepatic_failure": ("Hepatic failure", None, "categorical"),
        "immunosuppression": ("Immunosuppression", None, "categorical"),
        "leukemia": ("Leukemia / myeloma", None, "categorical"),
        "lymphoma": ("Lymphoma", None, "categorical"),
        "solid_tumor_with_metastasis": ("Solid tumour with metastasis", None, "categorical"),
    }

    eth_vals = sorted({str(v) for v in cohort["ethnicity"].dropna().unique()})
    cat_maps["ethnicity"] = {v: v for v in eth_vals}

    for col in KEEP:
        label, unit, kind = labels[col]
        series = cohort[col]
        if kind == "categorical":
            if col in {"elective_surgery", "gcs_unable_apache", "intubated_apache", "ventilated_apache",
                       "arf_apache", "apache_post_operative", "aids", "cirrhosis", "hepatic_failure",
                       "immunosuppression", "leukemia", "lymphoma", "solid_tumor_with_metastasis"}:
                frame[col] = blank(series.map(lambda v: None if pd.isna(v) else str(int(float(v)))))
            else:
                frame[col] = blank(series.map(lambda v: None if pd.isna(v) else str(v).strip()))
            predictors[col] = {"column": col, "kind": "categorical", "text_label": label,
                               "text_values": cat_maps[col]}
        else:
            frame[col] = blank(series)
            predictors[col] = {"column": col, "kind": "numeric", "text_label": label,
                               **({"unit": unit} if unit else {})}

    frame["diabetes_mellitus"] = cohort["diabetes_mellitus"].astype(int)

    summary = {
        "observations": len(frame),
        "predictors": len(predictors),
        "outcome": "diabetes_mellitus",
        "class_counts": {
            "No diabetes mellitus": int((frame["diabetes_mellitus"] == 0).sum()),
            "Diabetes mellitus": int((frame["diabetes_mellitus"] == 1).sum()),
        },
        "identifier_note": (
            f"WiDS 2021 TrainingWiDS2021.csv; stratified sample of {N_SAMPLE} encounters "
            f"(seed {SEED}) from {len(src)} labelled encounters. Excluded: identifiers, "
            f"hospital/ICU ids, APACHE diagnosis codes, all glucose labs (near-label for "
            f"existing diabetes), and unused labs/vitals outside the fixed KEEP list. "
            f"Target is existing diabetes status, not incidence."
        ),
        "full_training_n": int(len(src)),
        "excluded_near_label": ["glucose_apache", "d1_glucose_max", "d1_glucose_min",
                                "h1_glucose_max", "h1_glucose_min"],
    }
    spec = {
        "name": "wids2021",
        "description": (
            f"WiDS Datathon 2021 diabetes mellitus task ({len(frame)} ICU encounters after "
            f"restricted predictor set and stratified subsample)."
        ),
        "file": "patients.csv",
        "separator": ",",
        "id_column": "record_id",
        "pseudonymize": False,
        "setting": (
            "ICU admission: predicting whether the patient has a pre-existing diagnosis of "
            "diabetes mellitus from demographics, acuity, and non-glucose laboratory/comorbidity features."
        ),
        "observations_header": "ICU admission demographics, acuity and comorbidities:",
        "predictors": predictors,
        "outcomes": {
            "diabetes_mellitus": {
                "column": "diabetes_mellitus",
                "classes": ["No diabetes mellitus", "Diabetes mellitus"],
                "mapping": {"0": "No diabetes mellitus", "1": "Diabetes mellitus"},
                "question": (
                    "Based on the ICU admission demographics, acuity and comorbidity features "
                    "(excluding glucose measurements), does the patient have diabetes mellitus?"
                ),
            }
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    if N_SAMPLE is None:
        frame.to_csv(OUT / "patients_full.csv", index=False, encoding="utf-8")
        print(f"wids2021 full: {len(frame)} rows")
        return
    patients = OUT / "patients.csv"
    frame.to_csv(patients, index=False, encoding="utf-8")
    (OUT / "dataset.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    notes = {
        "dataset": "wids2021",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source": str(RAW.relative_to(ROOT)),
        "source_sha256": sha256(RAW),
        "patients_csv_sha256": sha256(patients),
        "summary": summary,
        "kaggle_note": (
            "Raw files obtained via kaggle datasets download -d sangeetha1213/widsdatathon2021 "
            "because competitions download -c widsdatathon2021 returned 403 until the "
            "competition is joined / rules accepted."
        ),
    }
    (OUT / "PREPARATION.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    (OUT / "DATASET_NOTES.md").write_text(
        f"# wids2021\n\n- Observations: {summary['observations']}\n"
        f"- Predictors: {summary['predictors']}\n- Outcome: diabetes_mellitus\n"
        f"- Classes: {summary['class_counts']}\n- {summary['identifier_note']}\n",
        encoding="utf-8",
    )
    print(f"wids2021: {summary['observations']} rows, {summary['predictors']} predictors, "
          f"classes={summary['class_counts']}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
