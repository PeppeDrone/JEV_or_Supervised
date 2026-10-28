#!/usr/bin/env python
"""Reference scorers on the fixed development/hold-out partitions.

The best marker is selected using development-only stratified CV AUROC; fold
transformations are fitted within training folds. One frozen marker supplies
both hold-out AUROC and balanced accuracy. Direction-only and additive models
use development data. Best-marker share is the unbounded AUROC ratio
(AUROC_marker-.5)/(AUROC_XGB-.5), undefined for XGB AUROC <= .6. These are
reference performance measures, not information fractions or interaction assays.
Outputs are versioned separately from the original study results.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevtest.benchmark import ALL_PAIRS  # noqa: E402
from jevtest.data import load, predictor_matrix  # noqa: E402
from jevtest.revision_reference import select_marker, predict_marker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "scientific_revision_2026-10-25_v1"
BASE = ROOT / "outputs"


def ovr_auc(y, score, n_classes):
    """Macro one-vs-rest AUROC; ``score`` is (n,) for one score shared by all classes or (n, k) per class."""
    score = np.asarray(score, float)
    if n_classes == 2:
        s = score if score.ndim == 1 else score[:, 1]
        return roc_auc_score(y, s)
    vals = []
    for k in range(n_classes):
        if 0 < (y == k).sum() < len(y):
            s = score if score.ndim == 1 else score[:, k]
            vals.append(roc_auc_score(y == k, s))
    return float(np.mean(vals)) if vals else np.nan


def signed_auc(y_dev, s_dev, y_hold, s_hold, n_classes):
    """Direction picked on the development set, AUROC on *all* hold-out patients; missing values take the
    development median so that an informatively missing marker is not scored on a favourable subset."""
    s_dev, s_hold = np.asarray(s_dev, float), np.asarray(s_hold, float)
    if np.isfinite(s_dev).sum() < 3:
        return np.nan
    fill = np.nanmedian(s_dev)
    s_dev, s_hold = np.where(np.isfinite(s_dev), s_dev, fill), np.where(np.isfinite(s_hold), s_hold, fill)
    if np.std(s_dev) == 0:
        return np.nan
    out = []
    for k in range(n_classes) if n_classes > 2 else [1]:
        yd, yh = (y_dev == k).astype(int), (y_hold == k).astype(int)
        if yd.min() == yd.max() or yh.min() == yh.max():
            continue
        sign = 1.0 if roc_auc_score(yd, s_dev) >= 0.5 else -1.0
        out.append(roc_auc_score(yh, sign * s_hold))
    return float(np.mean(out)) if out else np.nan


def bacc(y, labels, n_classes):
    out = [(labels[y == c] == c).mean() for c in range(n_classes) if (y == c).any()]
    return float(np.mean(out)) if out else np.nan


def univariate_rule_bacc(x_dev, y_dev, x_hold, y_hold, n_classes):
    """One-marker rule fitted on the development set: nearest class median (numeric) or level majority (categorical)."""
    if x_dev.dtype == object:
        maj = pd.Series(y_dev).groupby(x_dev.astype(object).to_numpy()).agg(lambda v: np.bincount(v, minlength=n_classes).argmax())
        default = int(np.bincount(y_dev, minlength=n_classes).argmax())
        pred = x_hold.map(maj).fillna(default).astype(int).to_numpy()
    else:
        xd, xh = x_dev.to_numpy(float), x_hold.to_numpy(float)
        if np.isfinite(xd).sum() < 3:
            return np.nan
        fill = np.nanmedian(xd)
        xd, xh = np.where(np.isfinite(xd), xd, fill), np.where(np.isfinite(xh), xh, fill)
        med = np.array([np.median(xd[y_dev == c]) if (y_dev == c).any() else np.nan for c in range(n_classes)])
        if np.isnan(med).any() or np.std(xd) == 0:
            return np.nan
        pred = np.abs(xh[:, None] - med[None, :]).argmin(axis=1)
    return bacc(y_hold, pred, n_classes)


def direction_only_score(X_dev, y_dev, X_hold, n_classes):
    """Equal-weight additive score with development-set directions; one column per class (or one for binary)."""
    classes = range(n_classes) if n_classes > 2 else [1]
    scores = np.zeros((len(X_hold), len(list(classes))))
    for j, k in enumerate(classes):
        yk = pd.Series((y_dev == k).astype(float), index=X_dev.index)
        parts = []
        for col in X_dev:
            d, h = X_dev[col], X_hold[col]
            if d.dtype == object:
                rates = yk.groupby(d.astype(object)).mean()
                centred = (rates - yk.mean())
                sgn = np.sign(centred)
                parts.append(h.map(sgn).astype(float).fillna(0.0).to_numpy())
            else:
                dv = d.astype(float)
                if dv.notna().sum() < 3 or dv.std(skipna=True) == 0:
                    continue
                mu, sd = dv.mean(skipna=True), dv.std(skipna=True)
                z_dev = (dv - mu) / sd
                corr = np.corrcoef(z_dev.fillna(0.0).to_numpy(), yk.to_numpy())[0, 1]
                sgn = 1.0 if (np.isnan(corr) or corr >= 0) else -1.0
                parts.append(sgn * ((h.astype(float) - mu) / sd).fillna(0.0).to_numpy())
        scores[:, j] = np.sum(parts, axis=0) if parts else 0.0
    return scores[:, 0] if n_classes == 2 else scores


def additive_model(X_dev, y_dev, X_hold):
    num = [c for c in X_dev if X_dev[c].dtype != object]
    cat = [c for c in X_dev if X_dev[c].dtype == object]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat),
    ])
    clf = make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(X_dev, y_dev)
    return clf.predict_proba(X_hold)


def one_task(dataset, outcome):
    hold_dir = BASE / dataset / outcome / "holdout"
    split = json.loads((hold_dir / "split.json").read_text())
    metrics = json.loads((hold_dir / "metrics.json").read_text())
    run = load(ROOT, dataset, outcome)
    X = predictor_matrix(run)
    ids = run.frame["patient_id"].astype(str)
    dev = ids.isin(split["development"]).to_numpy()
    hold = ids.isin(split["holdout"]).to_numpy()
    if dev.sum() + hold.sum() != len(ids):
        raise RuntimeError(f"{dataset}/{outcome}: split does not cover the table")
    y, k = run.y, int(metrics["n_classes"])
    X_dev, X_hold, y_dev, y_hold = X[dev], X[hold], y[dev], y[hold]
    xgb = pd.read_csv(hold_dir / "XGB_optuna.csv", dtype={"patient_id": str})
    auc_xgb = ovr_auc(xgb["observed"].to_numpy(), xgb[[c for c in xgb.columns if c.startswith("p__")]].to_numpy(), k)

    # Select exactly once using inner development CV. The hold-out labels enter only evaluation.
    marker, candidates = select_marker(X_dev, y_dev, k)
    uni_scores, uni_labels = predict_marker(marker, X_hold[marker["predictor"]])
    best_uni = ovr_auc(y_hold, uni_scores, k)

    dscore = direction_only_score(X_dev, y_dev, X_hold, k)
    direction = ovr_auc(y_hold, dscore, k)
    # operating points without local labels: sign (binary) or argmax (multiclass) of the direction score; LR argmax
    direction_b = bacc(y_hold, (dscore > 0).astype(int) if k == 2 else dscore.argmax(axis=1), k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P_add = additive_model(X_dev, y_dev, X_hold)
    additive = ovr_auc(y_hold, P_add, k)
    additive_b = bacc(y_hold, P_add.argmax(axis=1), k)
    bacc_xgb = bacc(xgb["observed"].to_numpy(), xgb["predicted"].to_numpy(), k)
    best_uni_b = bacc(y_hold, uni_labels, k)
    head_b = bacc_xgb - 0.5 if k == 2 else bacc_xgb - 1.0 / k

    head = auc_xgb - 0.5
    return {
        "dataset": dataset, "outcome": outcome,
        "n_development": int(metrics["n_development"]), "n_holdout": int(metrics["n_holdout"]), "n_classes": k,
        "n_predictors": int(X.shape[1]), "auc_XGB": auc_xgb,
        "selected_predictor": marker["predictor"], "marker_development_cv_auc": marker["cv_auroc"],
        "best_univariate_auc": best_uni, "univariate_share": (best_uni - 0.5) / head if head > 0.1 else np.nan,
        "direction_only_auc": direction, "additive_auc": additive,
        "additive_share": (additive - 0.5) / head if head > 0.1 else np.nan, "interaction_gain": auc_xgb - additive,
        "bacc_XGB": bacc_xgb, "best_univariate_bacc": best_uni_b, "direction_only_bacc": direction_b, "additive_bacc": additive_b,
        "interaction_gain_bacc": bacc_xgb - additive_b,
    }


def main():
    rows = []
    for dataset, outcome in ALL_PAIRS:
        if not (BASE / dataset / outcome / "holdout" / "XGB_optuna.csv").exists():
            continue
        try:
            rows.append(one_task(dataset, outcome))
            r = rows[-1]
            print(f"{dataset:28s} {outcome:18s} dev={r['n_development']:5d} p={r['n_predictors']:3d} "
                  f"AUC xgb={r['auc_XGB']:.3f} uni={r['best_univariate_auc']:.3f} dir={r['direction_only_auc']:.3f} "
                  f"add={r['additive_auc']:.3f} | BA xgb={r['bacc_XGB']:.3f} uni={r['best_univariate_bacc']:.3f} "
                  f"dir={r['direction_only_bacc']:.3f} add={r['additive_bacc']:.3f}")
        except Exception as exc:  # keep going; report the task
            print(f"{dataset}/{outcome}: {exc}", file=sys.stderr)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "mechanism_covariates.csv", index=False)
    print(f"wrote {OUT / 'mechanism_covariates.csv'} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
