"""Apply the saved Careggi fold models, unchanged, to Bologna (no fitting).

Each manuscript model is the five outer-fold models of the Careggi internal validation
(outputs/diagnosis_ICUEEG/epileptic_final_diagnosis/models). A Bologna patient's probability
is the mean of the five fold models. Before scoring Bologna, every loaded fold model is
checked to reproduce its saved Careggi out-of-fold predictions.

Usage: PYTHONPATH=src .venv/bin/python scripts/external_validate_eminence.py [external_dataset]
(default eminence_bologna_external; eminence_bologna_external_wdx adds the coded working diagnosis)
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jevtest.config import read_json, write_json
from jevtest.data import load, predictor_matrix
from jevtest.metrics import calibration
from jevtest.models import load_pipeline
from jevtest.text import normalize_code
from jevtest.zeroshot import zero_shot_probs

DEV = "diagnosis_ICUEEG"
EXT = "eminence_bologna_external"
OUTCOME = "epileptic_final_diagnosis"
MODELS = ("M2", "ZS_raw")
BOOTSTRAP = 2000
TOLERANCE = 1e-6


def metrics(y, p, include_calibration=True):
    predicted = p >= 0.5
    result = {
        "AUROC": roc_auc_score(y, p),
        "Average precision": average_precision_score(y, p),
        "Balanced accuracy": balanced_accuracy_score(y, predicted),
        "Sensitivity": float(predicted[y == 1].mean()),
        "Specificity": float((~predicted[y == 0]).mean()),
        "Brier score": float(np.mean((p - y) ** 2)),
    }
    if include_calibration:
        intercept, slope = calibration(y, np.c_[1 - p, p])
        result["Calibration intercept"] = intercept
        result["Calibration slope"] = slope
    return result


def interval(y, p, seed):
    rng = np.random.default_rng(seed)
    values = {key: [] for key in metrics(y, p, include_calibration=False)}
    for draw in rng.integers(0, len(y), size=(BOOTSTRAP, len(y))):
        if len(np.unique(y[draw])) != 2:
            continue
        for key, value in metrics(y[draw], p[draw], include_calibration=False).items():
            values[key].append(value)
    return {key: np.quantile(np.asarray(value)[np.isfinite(value)], [0.025, 0.975])
            if np.isfinite(value).any() else (np.nan, np.nan) for key, value in values.items()}


def fold_scores(development, name, dev_inputs, ext_inputs):
    """Per-fold Bologna probabilities, after checking each fold reproduces its Careggi predictions."""
    saved = development.predictions(name)
    external, gaps = [], []
    for fold, (_, test) in enumerate(development.splits):
        if name == "M2":
            model = load_pipeline(development.out, name, fold)
            replay, scored = model.predict_proba(dev_inputs.iloc[test]), model.predict_proba(ext_inputs)
        else:
            coefficients = load_logistic(development.out, name, fold)
            layer = coefficients.get("layer")
            dev_x = dev_inputs[layer] if layer else dev_inputs
            ext_x = ext_inputs[layer] if layer else ext_inputs
            replay, scored = apply_logistic(coefficients, dev_x[test]), apply_logistic(coefficients, ext_x)
        gaps.append(float(np.abs(replay[:, 1] - saved[test, 1]).max()))
        external.append(scored[:, 1])
    if max(gaps) > TOLERANCE:
        raise RuntimeError(f"{name}: saved fold models do not reproduce Careggi predictions (max gap {max(gaps):.2e})")
    return np.vstack(external), gaps


def align_categories(x_dev, x_ext):
    """Write each external categorical code as the string the development encoder saw ("1.0" vs "1").

    The fitted one-hot encoders ignore unknown strings, so a formatting difference would silently
    drop the predictor. Codes absent from development stay as they are and are ignored as unknown.
    """
    x_ext = x_ext.copy()
    for column in x_dev.columns[x_dev.dtypes == object]:
        spelling = {normalize_code(v): v for v in x_dev[column].dropna().unique()}
        x_ext[column] = x_ext[column].map(lambda v: v if pd.isna(v) else spelling.get(normalize_code(v), v))
    return x_ext


def main():
    ext = sys.argv[1] if len(sys.argv) > 1 else EXT
    development, external = load(ROOT, DEV, OUTCOME), load(ROOT, ext, OUTCOME)
    x_dev, x_ext = predictor_matrix(development), predictor_matrix(external)
    if list(x_dev) != list(x_ext):
        raise ValueError("Development and external predictor schemas differ.")
    x_ext = align_categories(x_dev, x_ext)
    for key in ("predictors", "setting", "observations_header"):
        if development.ds[key] != external.ds[key]:
            raise ValueError(f"Development and external {key} differ; prompts would not match.")
    if development.spec != external.spec:
        raise ValueError("Development and external outcome question or classes differ.")
    output = external.out / "external_validation"
    output.mkdir(parents=True, exist_ok=True)

    per_fold, checks = {}, {}
    external.progress(stage="external M2")
    per_fold["M2"], checks["M2"] = fold_scores(development, "M2", x_dev, x_ext)
    external.progress(stage="external zero-shot (Bologna inference)")
    ext_zs = zero_shot_probs(external)
    per_fold["ZS_raw"] = (ext_zs / ext_zs.sum(axis=1, keepdims=True))[:, 1][None, :]

    models = {name: per_fold[name].mean(axis=0) for name in MODELS}
    table = pd.DataFrame({"patient_id": external.ids, "observed": external.y, **models})
    for name in ("M2",):
        for fold, values in enumerate(per_fold[name]):
            table[f"{name}__fold{fold}"] = values
    table.to_csv(output / "predictions.csv", index=False)

    rows = []
    for offset, name in enumerate(MODELS):
        p = models[name]
        point, ci = metrics(external.y, p), interval(external.y, p, development.study["seed"] + offset)
        row = {"Model": name, "N": len(external.y), "Positive outcomes": int(external.y.sum())}
        for key, value in point.items():
            if key in ci:
                low, high = ci[key]
                row[key] = f"{value:.3f} ({low:.3f}–{high:.3f})" if np.isfinite(value) else "Not estimable"
            else:
                row[key] = f"{value:.3f}" if np.isfinite(value) else "Not estimable"
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "performance.csv", index=False)
    write_json(output / "manifest.json", {
        "development_dataset": DEV, "external_dataset": ext, "outcome": OUTCOME,
        "development_data_hash": development.data_hash, "external_data_hash": external.data_hash,
        "development_models": str(development.out / "models"),
        "development_model_manifest": read_json(development.out / "models/MANIFEST.json")["versions"],
        "aggregation": "mean of the five Careggi outer-fold models; ZS_raw has no fitted parameters",
        "careggi_replay_max_abs_difference": checks, "bootstrap_iterations": BOOTSTRAP,
    })
    (output / "REPORT.md").write_text(
        "# External validation: EMINENCE Bologna\n\n"
        "The five saved Careggi outer-fold models of each method were applied unchanged to Bologna and their "
        "probabilities averaged. Nothing was fitted on Bologna. Unavailable Bologna values remained missing.\n\n"
        + "| " + " | ".join(summary.columns) + " |\n|" + "---|" * len(summary.columns) + "\n"
        + "".join("| " + " | ".join(map(str, row)) + " |\n" for row in summary.itertuples(index=False)),
        encoding="utf-8")
    external.progress(stage="external validation done")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
