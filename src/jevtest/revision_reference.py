"""Development-only single-marker selection; fitted objects contain no hold-out data."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def score_auc(y, scores):
    """Binary class 1 or full-class macro OVR; missing classes are undefined."""
    y, scores = np.asarray(y), np.asarray(scores)
    k = scores.shape[1]
    if set(np.unique(y)) != set(range(k)):
        return np.nan
    targets = [1] if k == 2 else range(k)
    return float(np.mean([roc_auc_score(y == c, scores[:, c]) for c in targets]))


def fit_marker(x, y, k):
    y = np.asarray(y)
    if set(np.unique(y)) != set(range(k)):
        raise ValueError("marker training partition lacks a class")
    prior = np.bincount(y, minlength=k) / len(y)
    if x.dtype == object:
        values = x.fillna("__MISSING__").astype(str).to_numpy()
        rates, majority = {}, {}
        for level in sorted(set(values)):
            counts = np.bincount(y[values == level], minlength=k)
            rates[level] = ((counts + prior) / (counts.sum() + 1)).tolist()
            majority[level] = int(counts.argmax())
        return dict(kind="categorical", rates=rates, majority=majority,
                    prior=prior.tolist(), default=int(prior.argmax()), classes=k)
    values = x.to_numpy(float)
    finite = np.isfinite(values)
    fill = float(np.median(values[finite])) if finite.any() else 0.0
    values = np.where(finite, values, fill)
    signs = []
    for c in range(k):
        signs.append(1.0 if roc_auc_score(y == c, values) >= .5 else -1.0)
    medians = [float(np.median(values[y == c])) for c in range(k)]
    return dict(kind="numeric", fill=fill, signs=signs, medians=medians, classes=k)


def predict_marker(state, x):
    if state["kind"] == "categorical":
        values = x.fillna("__MISSING__").astype(str)
        scores = np.array([state["rates"].get(v, state["prior"]) for v in values])
        labels = np.array([state["majority"].get(v, state["default"]) for v in values])
    else:
        values = x.to_numpy(float)
        values = np.where(np.isfinite(values), values, state["fill"])
        scores = values[:, None] * np.array(state["signs"])[None, :]
        labels = np.abs(values[:, None] - np.array(state["medians"])[None, :]).argmin(axis=1)
    return scores, labels


def select_marker(X_dev, y_dev, k, seed=20261008, folds=5):
    """Select on mean validation AUROC; freeze transformations on all development data.

    Encodings, imputation and directions are fitted inside each training fold.
    Fold count reduces to the rarest development class. Ties use column name.
    """
    y_dev = np.asarray(y_dev)
    count = np.bincount(y_dev, minlength=k)
    nfold = min(folds, int(count.min()))
    if nfold < 2:
        raise ValueError("fewer than two development cases in a class: CV unavailable")
    splits = list(StratifiedKFold(nfold, shuffle=True, random_state=seed).split(X_dev, y_dev))
    rows = []
    for column in sorted(X_dev.columns):
        for fold, (train, valid) in enumerate(splits):
            state = fit_marker(X_dev[column].iloc[train], y_dev[train], k)
            scores, _ = predict_marker(state, X_dev[column].iloc[valid])
            auc = score_auc(y_dev[valid], scores)
            rows.append(dict(predictor=column, fold=fold, cv_auroc=auc,
                             valid=bool(np.isfinite(auc))))
    estimates = pd.DataFrame(rows)
    means = estimates.groupby("predictor").cv_auroc.agg(["mean", "count"])
    means = means[means["count"] == nfold]
    if means.empty:
        raise ValueError("no candidate has valid AUROC in every development fold")
    chosen = means.sort_values("mean", ascending=False, kind="stable").index[0]
    state = fit_marker(X_dev[chosen], y_dev, k)
    state.update(predictor=chosen, cv_auroc=float(means.loc[chosen, "mean"]), folds=nfold, seed=seed)
    return state, estimates
