"""Additional locally supervised readouts; selection and fitting accept development data only."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


METHODS = ("prevalence", "offset", "temperature", "temperature_offset", "operating_point")


def probabilities(P):
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] < 2 or not np.isfinite(P).all() or (P < 0).any():
        raise ValueError("invalid class probabilities")
    if not np.allclose(P.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError("class probabilities are not normalised")
    return P


def log_scores(P):
    return np.log(np.maximum(probabilities(P), np.finfo(float).tiny))


def calibration_sample(n_dev, budget, seed):
    """Uniform sampling without replacement, blind to labels; no class balancing."""
    if budget == "full":
        return np.arange(n_dev)
    budget = int(budget)
    if budget > n_dev:
        raise ValueError("budget exceeds development population")
    return np.sort(np.random.default_rng(seed).choice(n_dev, budget, replace=False))


def fit_readout(P_dev, y_dev, method, k=None):
    """Fixed Gaussian shrinkage (quadratic penalty divided by n); no test input."""
    y_dev = np.asarray(y_dev, int)
    k = int(k if k is not None else P_dev.shape[1])
    counts = np.bincount(y_dev, minlength=k)
    if len(y_dev) == 0 or len(counts) != k:
        raise ValueError("invalid development labels")
    prior = (counts + .5) / (len(y_dev) + .5 * k)
    if method == "prevalence":
        return dict(method=method, prior=prior.tolist(), n=len(y_dev), missing_classes=int((counts == 0).sum()))
    z = log_scores(P_dev)
    if (counts == 0).any():
        raise ValueError("calibration sample lacks a class")
    if method == "operating_point" and k == 2:
        # Exhaustive empirical BA over distinct score boundaries; tie -> closest to 0.5, then lower.
        p = np.asarray(P_dev)[:, 1]
        levels = np.unique(p)
        thresholds = np.unique(np.r_[0., (levels[:-1] + levels[1:]) / 2, 1., .5])
        pred = p[:, None] >= thresholds[None, :]
        ba = .5 * (pred[y_dev == 1].mean(0) + (~pred[y_dev == 0]).mean(0))
        # Shrink score shifts for small samples; penalty fixed before evaluation.
        criterion = ba - (thresholds - .5) ** 2 / np.sqrt(len(y_dev))
        rank = np.lexsort((thresholds, np.abs(thresholds - .5), -criterion))
        return dict(method=method, threshold=float(thresholds[rank[0]]), n=len(y_dev), penalty="(t-.5)^2/sqrt(n)")
    temperature = method in ("temperature", "temperature_offset")
    offsets = method in ("offset", "temperature_offset", "operating_point")
    if not (temperature or offsets):
        raise ValueError(method)
    # K-1 free offsets with last fixed at zero, then centre: unique identifiable representation.
    def unpack(theta):
        invT = float(np.exp(theta[0])) if temperature else 1.
        raw = np.r_[theta[int(temperature):], 0.] if offsets else np.zeros(k)
        return invT, raw - raw.mean()
    def objective(theta):
        invT, bias = unpack(theta)
        scores = z * invT + bias
        loss = np.mean(logsumexp(scores, axis=1) - scores[np.arange(len(y_dev)), y_dev])
        if method == "operating_point":
            # Smooth BA surrogate: class-balanced cross entropy, retaining only K-1 biases.
            weights = len(y_dev) / (k * counts[y_dev])
            loss = np.mean(weights * (logsumexp(scores, axis=1) - scores[np.arange(len(y_dev)), y_dev]))
        penalty = .5 * np.sum(bias ** 2) / len(y_dev)
        if temperature:
            penalty += .5 * theta[0] ** 2 / len(y_dev)
        return loss + penalty
    theta0 = np.zeros(int(temperature) + (k - 1 if offsets else 0))
    bounds = ([(-3., 3.)] if temperature else []) + ([(-8., 8.)] * (k - 1) if offsets else [])
    fit = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=500, ftol=1e-12))
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"calibration optimisation failed: {fit.message}")
    invT, bias = unpack(fit.x)
    return dict(method=method, inverse_temperature=invT, bias=bias.tolist(), n=len(y_dev),
                objective=float(fit.fun), regularisation="0.5*(sum(centred_bias^2)+log_inverse_temperature^2)/n; omit absent terms", success=True)


def apply_readout(P, fitted):
    P = probabilities(P)
    method = fitted["method"]
    if method == "prevalence":
        # Decision intervention only: keep original P for probability diagnostics.
        labels = (P / np.array(fitted["prior"])).argmax(1)
        return P.copy(), labels
    if "threshold" in fitted:
        return P.copy(), (P[:, 1] >= fitted["threshold"]).astype(int)
    transformed = softmax(log_scores(P) * fitted["inverse_temperature"] + np.array(fitted["bias"]), axis=1)
    labels = transformed.argmax(1)
    if method == "temperature" and not np.array_equal(labels, P.argmax(1)):
        raise AssertionError("positive temperature changed argmax")
    return transformed, labels
