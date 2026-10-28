"""Load one dataset x outcome: labelled rows, class indices, outer folds and the model matrix."""
from pathlib import Path
import hmac, hashlib, secrets

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from jevtest.config import GateError, Run, dataset_config, dataset_dir, digest, outputs_dir, study_config
from jevtest.text import check_codes, normalize_code, numeric_value


def pseudonyms(root, ids):
    """Keyed hash of source identifiers; the key stays in private/ and is never copied."""
    path = Path(root) / "private/pseudonym.key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
    key = bytes.fromhex(path.read_text().strip())
    return [hmac.new(key, str(i).encode(), hashlib.sha256).hexdigest()[:24] for i in ids]


def folds(y, study):
    counts = pd.Series(y).value_counts()
    outer = min(study["outer_folds"], int(counts.min()))
    if outer < 2:
        raise GateError("Insufficient counts for stratified outer folds")
    splits = list(StratifiedKFold(outer, shuffle=True, random_state=study["seed"]).split(np.zeros(len(y)), y))
    for train, _ in splits:
        if pd.Series(y[train]).value_counts().min() < 2:
            raise GateError("Insufficient inner-fold counts")
    return splits


def load(root, dataset, outcome, limit=None, full=False):
    root = Path(root)
    study, ds = study_config(root), dataset_config(root, dataset)
    if outcome not in ds["outcomes"]:
        raise GateError(f"Outcome {outcome} not declared in datasets/{dataset}/dataset.yaml")
    spec = ds["outcomes"][outcome]
    path = (dataset_dir(root, dataset) / ds["file"]).resolve()
    if full:
        # Whole table for the XGBoost arm only; ``patients.csv`` stays the zero-shot subsample.
        path = path.with_name("patients_full.csv")
        if not path.exists():
            raise GateError(f"{path.name} not found; run scripts/prepare_full_tables.py")
    frame = pd.read_csv(path, sep=ds.get("separator", ","), encoding="utf-8-sig")
    columns = [info["column"] for info in ds["predictors"].values()] + [spec["column"], ds["id_column"]]
    missing = [c for c in columns if c not in frame]
    if missing:
        raise GateError(f"Columns missing from {path.name}: {missing}")

    mapping = {normalize_code(k): v for k, v in spec["mapping"].items()}
    codes = frame[spec["column"]].map(normalize_code)
    unknown = set(codes.dropna()) - set(mapping)
    if unknown:
        raise GateError(f"Outcome {outcome}: {len(unknown)} unmapped code(s); extend its mapping")
    frame = frame.loc[codes.notna()].reset_index(drop=True)
    y = np.array([spec["classes"].index(mapping[c]) for c in codes.dropna()], dtype=int)

    ids = frame.pop(ds["id_column"])
    frame.insert(0, "patient_id", pseudonyms(root, ids) if ds.get("pseudonymize", True) else ids.astype(str))
    if frame.patient_id.duplicated().any():
        raise GateError("Duplicated patient identifiers: one row per patient is required")
    if not full:
        check_codes(frame, ds)

    suffix = "__full" if full else ""
    if limit:
        # Smoke runs only: a stratified subset, written to its own output folder.
        keep, _ = train_test_split(np.arange(len(y)), train_size=limit, stratify=y, random_state=study["seed"])
        keep = np.sort(keep)
        frame, y, suffix = frame.iloc[keep].reset_index(drop=True), y[keep], suffix + f"__limit{limit}"
    else:
        counts = np.bincount(y, minlength=len(spec["classes"]))
        if counts.min() < study["min_class_count"]:
            raise GateError(f"Every class needs >= {study['min_class_count']} examples; counts {counts.tolist()}")
    data_hash = digest(digest(path).encode() + (dataset_dir(root, dataset) / "dataset.yaml").read_bytes())
    out = outputs_dir(root, study) / dataset / (outcome + suffix)
    out.mkdir(parents=True, exist_ok=True)
    return Run(root, study, ds, dataset, outcome, frame, y, folds(y, study), out, data_hash)


def predictor_matrix(run):
    """Declared kinds: categorical codes stay nominal; file column order (as neurojev.baselines)."""
    by_column = {info["column"]: info for info in run.ds["predictors"].values()}
    X = run.frame[[c for c in run.frame.columns if c in by_column]].copy()
    for column in X:
        info = by_column[column]
        # Declared "unavailable" codes are missing values for the models too.
        absent = X[column].map(normalize_code).isin(info.get("missing_codes", []))
        X[column] = X[column].mask(absent)
        if info["kind"] == "categorical":
            X[column] = X[column].map(lambda value: str(value) if pd.notna(value) else np.nan).astype(object)
        else:
            X[column] = X[column].map(numeric_value).astype(float)
            if info.get("model_transform") == "log1p":
                X[column] = np.log1p(X[column])
    return X


def m2_predictions_path(run):
    """XGBoost predictions for the manuscript: the whole-table run when one exists, else the standard run."""
    full = run.out.with_name(run.out.name + "__full") / "predictions" / "M2.csv"
    return full if full.exists() else run.out / "predictions" / "M2.csv"


def full_run(root, dataset, outcome):
    """The whole-table Run when its XGBoost predictions exist, else None."""
    if not (dataset_dir(Path(root), dataset) / "patients_full.csv").exists():
        return None
    run = load(root, dataset, outcome, full=True)
    return run if run.predictions("M2") is not None else None
