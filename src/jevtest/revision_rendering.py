"""Fixed semantic interventions, without access to outcome values."""
from copy import deepcopy
import numpy as np
from scipy.stats import spearmanr
from jevtest.text import render, normalize_code, render_value, entries

CONDITIONS = ("original", "no_setting", "no_units", "anonymous", "no_category_gloss", "concise")


def render_condition(row, ds, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == "original":
        return render(row, ds)
    spec = deepcopy(ds)
    if condition == "no_units":
        for info in spec["predictors"].values():
            info.pop("unit", None)
            # Units embedded in labels/scale notes cannot reliably be stripped automatically.
        return render(row, spec)
    if condition in ("anonymous", "no_category_gloss"):
        for i, info in enumerate(spec["predictors"].values()):
            if info["kind"] == "categorical":
                info["text_values"] = {str(c): str(c) for c in info["text_values"]}
            if condition == "anonymous":
                info["text_label"] = f"Variable {i + 1}"
                for field in ("unit", "scale_note", "text_values"):
                    if field != "text_values" or info["kind"] != "categorical":
                        info.pop(field, None)
                # Conditional prose can reveal the source variable/category; retain missingness only.
                if "text_requires" in info:
                    info["text_requires"]["absent_text"] = "not observed"
                    info["text_requires"]["unknown_text"] = "not observed"
        if condition == "anonymous":
            spec["setting"] = "Observations for one patient."
            spec["observations_header"] = "Recorded values:"
        return render(row, spec)
    labels, values = entries(row, spec)
    lines = "\n".join(f"{a}: {b}" for a, b in zip(labels, values))
    if condition == "no_setting":
        return f"{spec['observations_header']}\n{lines}"
    # Remove header and setting boilerplate, retaining labels, units, scales and category interpretations.
    return lines


def direction_plan(X_dev, y_dev, ds, k, max_features=2):
    """Predefine class K-1 versus rest; reject non-monotone empirical quartile trends.

    Quantile bounds are within documented ranges. Clinical joint plausibility is
    unverified and explicitly flagged; no internal-knowledge inference is licensed.
    """
    contrast = (np.asarray(y_dev) == k - 1).astype(int)
    candidates = []
    for name, info in ds["predictors"].items():
        col = info["column"]
        if info["kind"] == "categorical" or info.get("text_requires") or info.get("model_transform"):
            continue
        x = X_dev[col].to_numpy(float)
        valid = np.isfinite(x)
        if valid.sum() < 20 or len(np.unique(x[valid])) < 5 or np.ptp(contrast[valid]) == 0:
            continue
        rho, p = spearmanr(x[valid], contrast[valid])
        low, high = np.quantile(x[valid], [.25, .75])
        limits = info.get("text_range", info.get("range"))
        if limits:
            low, high = max(low, limits[0]), min(high, limits[1])
        if low >= high or not np.isfinite(rho) or abs(rho) < .1:
            continue
        edges = np.unique(np.quantile(x[valid], [0, .25, .5, .75, 1]))
        bins = np.digitize(x[valid], edges[1:-1])
        rates = np.array([contrast[valid][bins == b].mean() for b in np.unique(bins)])
        # No direction claim for an observed reversal larger than 0.10.
        monotone = bool(np.all(np.diff(rates) * np.sign(rho) >= -.1))
        candidates.append(dict(predictor=name, column=col, low=float(low), high=float(high),
                               development_rho=float(rho), development_p=float(p), contrast=k - 1,
                               monotone=monotone, joint_plausibility="unverified; within marginal range only"))
    return sorted([c for c in candidates if c["monotone"]], key=lambda c: -abs(c["development_rho"]))[:max_features]
