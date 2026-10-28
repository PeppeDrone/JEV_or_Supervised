"""Out-of-fold TreeSHAP values of XGBoost, aggregated from encoded columns back to the original predictors.

Each patient's values come from the model of the outer fold in which the patient was held out. Contributions are on
the log-odds scale; one-hot columns of a categorical predictor are summed into that predictor.
Usage: PYTHONPATH=src python scripts/xgb_shap.py --dataset D --outcome O
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import xgboost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest.classical import fit_fold
from jevtest.data import load, predictor_matrix


def encoded_owner(preprocess, X):
    """Original predictor of every encoded column, in the transformer's output order."""
    owners = []
    for name, transformer, columns in preprocess.transformers_:
        if name == "numeric":
            owners += list(columns)
        elif name == "category":
            encoder = transformer.named_steps["encode"]
            for column, categories in zip(columns, encoder.categories_):
                owners += [column] * len(categories)
    return owners


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outcome", required=True)
    args = parser.parse_args(argv)
    run = load(ROOT, args.dataset, args.outcome)
    X = predictor_matrix(run)
    columns = list(X.columns)
    shap = np.zeros((len(run.y), len(columns)))
    base = np.zeros(len(run.y))
    probability = np.zeros(len(run.y))
    for fold, (train, test) in enumerate(run.splits):
        print(f"fold {fold + 1}", flush=True)
        model = fit_fold(X, run.y, train, "M2", run.study, fold)
        preprocess, booster = model.named_steps["preprocess"], model.named_steps["model"].get_booster()
        encoded = preprocess.transform(X.iloc[test])
        owners = encoded_owner(preprocess, X)
        if len(owners) != encoded.shape[1]:
            raise SystemExit("Encoded columns could not be mapped back to predictors")
        contributions = booster.predict(xgboost.DMatrix(encoded), pred_contribs=True)
        index = {c: i for i, c in enumerate(columns)}
        for j, owner in enumerate(owners):
            shap[test, index[owner]] += contributions[:, j]
        base[test] = contributions[:, -1]
        probability[test] = model.predict_proba(X.iloc[test])[:, 1]
    logit = base + shap.sum(axis=1)
    gap = float(np.abs(1 / (1 + np.exp(-logit)) - probability).max())
    reference = float(np.abs(probability - run.predictions("M2")[:, 1]).max())
    print(f"SHAP additivity: max |dp| = {gap:.2e}; refit vs main run: max |dp| = {reference:.2e}", flush=True)
    if gap > 1e-4 or reference > 1e-5:
        raise SystemExit("SHAP values do not reproduce the XGBoost predictions")
    folder = run.out / "occlusion"
    folder.mkdir(exist_ok=True)
    np.savez(folder / "xgb_shap.npz", shap=shap, base=base, columns=np.array(columns),
             values=X.apply(lambda s: s.astype(float), axis=0).to_numpy())
    print(f"Saved {folder / 'xgb_shap.npz'}", flush=True)


if __name__ == "__main__":
    main()
