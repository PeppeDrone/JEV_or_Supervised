"""Fitted fold models saved for external validation and replication.

Layout: <run.out>/models/<model>/fold<k>.<ext> plus fold<k>.json (training patients and
chosen settings) and one MANIFEST.json per run (data, prompt, backbone, folds, versions).
Classical pipelines use joblib (scikit-learn's native format; load with the recorded
versions). Logistic calibrators and probes are stored as plain JSON coefficients.
"""
from importlib.metadata import PackageNotFoundError, version
import platform

import joblib
import numpy as np

from jevtest.config import GateError, digest, read_json, write_json

PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "scipy", "torch", "transformers", "joblib")


def model_dir(run, model):
    return run.out / "models" / model


def versions():
    found = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def write_manifest(run):
    """Everything needed to rebuild the inputs the fold models expect."""
    write_json(run.out / "models/MANIFEST.json", {
        "dataset": run.dataset, "outcome": run.outcome, "data_hash": run.data_hash,
        "dataset_spec_sha256": digest(run.root / "datasets" / run.dataset / "dataset.yaml"),
        "study": run.study, "classes": run.classes, "question": run.spec["question"],
        "setting": run.ds["setting"], "observations_header": run.ds["observations_header"],
        "predictor_columns": [info["column"] for info in run.ds["predictors"].values()],
        "folds": [{"fold": k, "train": run.ids[train].tolist(), "test": run.ids[test].tolist()}
                  for k, (train, test) in enumerate(run.splits)],
        "versions": versions(),
        "prompt": "jevtest.text.render(row, dataset_spec) followed by jevtest.llm.options_block(question, classes)",
    })


def save_pipeline(run, model, fold, fitted, train, details=None):
    folder = model_dir(run, model)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, folder / f"fold{fold}.joblib")
    write_json(folder / f"fold{fold}.json", {"fold": fold, "train": run.ids[train].tolist(),
                                              "sha256": digest(folder / f"fold{fold}.joblib"), **(details or {})})


def save_logistic(run, model, fold, pipeline, train, details=None):
    """StandardScaler + LogisticRegression as plain coefficients."""
    scale, lr = pipeline[0], pipeline[-1]
    write_json(model_dir(run, model) / f"fold{fold}.json", {
        "fold": fold, "train": run.ids[train].tolist(), **(details or {}),
        "scaler_mean": scale.mean_, "scaler_scale": scale.scale_,
        "coef": lr.coef_, "intercept": lr.intercept_, "classes": lr.classes_, "C": lr.C,
    })


def load_pipeline(out, model, fold):
    meta = read_json(out / "models" / model / f"fold{fold}.json")
    path = out / "models" / model / f"fold{fold}.joblib"
    if digest(path) != meta["sha256"]:
        raise GateError(f"{path} checksum differs from its metadata")
    return joblib.load(path)


def load_logistic(out, model, fold):
    return read_json(out / "models" / model / f"fold{fold}.json")


def apply_logistic(saved, X):
    """Class probabilities from saved scaler + multinomial/binary logistic coefficients."""
    z = (np.asarray(X, dtype=float) - np.asarray(saved["scaler_mean"])) / np.asarray(saved["scaler_scale"])
    logits = z @ np.asarray(saved["coef"]).T + np.asarray(saved["intercept"])
    if logits.shape[1] == 1:
        p = 1 / (1 + np.exp(-logits[:, 0]))
        return np.c_[1 - p, p]
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)
