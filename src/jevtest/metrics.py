"""Out-of-fold metrics with a patient bootstrap; paired deltas use the same resamples for every model."""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score

KEYS = ("auroc", "brier", "log_loss", "balanced_accuracy", "macro_f1")


def auroc(y, P):
    k = P.shape[1]
    if k == 2:
        return float(roc_auc_score(y, P[:, 1])) if len(np.unique(y)) == 2 else np.nan
    # Macro one-vs-rest; undefined when a resample lacks a class.
    if len(np.unique(y)) < k:
        return np.nan
    return float(np.mean([roc_auc_score(y == c, P[:, c]) for c in range(k)]))


def brier(y, P):
    k = P.shape[1]
    if k == 2:
        return float(np.mean((P[:, 1] - y) ** 2))
    return float(np.mean(np.sum((P - np.eye(k)[y]) ** 2, axis=1)))


def calibration(y, P):
    """Calibration intercept and slope of the logit of P(class 1) (binary only)."""
    if P.shape[1] != 2 or np.ptp(P[:, 1]) < 1e-12:
        return np.nan, np.nan
    z = np.log(np.clip(P[:, 1], 1e-7, 1 - 1e-7) / np.clip(P[:, 0], 1e-7, 1))
    fit = minimize(lambda b: np.sum(np.logaddexp(0, b[0] + b[1] * z) - y * (b[0] + b[1] * z)), [0.0, 1.0], method="BFGS")
    return (float(fit.x[0]), float(fit.x[1])) if fit.success and np.all(np.abs(fit.x) < 50) else (np.nan, np.nan)


def point(y, P, labels=None, probabilistic=True):
    """`labels` overrides argmax (Mode tie-breaks); vote shares are not probabilities, so no log loss/calibration."""
    k = P.shape[1]
    labels = P.argmax(axis=1) if labels is None else labels
    result = {
        "auroc": auroc(y, P),
        "brier": brier(y, P),
        "log_loss": float(log_loss(y, np.clip(P, 1e-15, 1), labels=range(k))) if probabilistic else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
        "macro_f1": float(f1_score(y, labels, average="macro", labels=range(k), zero_division=0)),
    }
    result["calibration_intercept"], result["calibration_slope"] = calibration(y, P) if probabilistic else (np.nan, np.nan)
    return result


def resamples(n, iterations, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(iterations)]


def bootstrap(y, P, draws, labels=None, probabilistic=True):
    """Metric values on each resample (calibration skipped for speed)."""
    out = {k: [] for k in KEYS}
    for i in draws:
        yy, PP = y[i], P[i]
        ll = labels[i] if labels is not None else PP.argmax(axis=1)
        out["auroc"].append(auroc(yy, PP))
        out["brier"].append(brier(yy, PP))
        out["log_loss"].append(float(log_loss(yy, np.clip(PP, 1e-15, 1), labels=range(P.shape[1]))) if probabilistic else np.nan)
        # A missing class changes the estimand from full-class BA. Mark that draw undefined.
        out["balanced_accuracy"].append(float(balanced_accuracy_score(yy, ll))
                                        if len(np.unique(yy)) == P.shape[1] else np.nan)
        out["macro_f1"].append(float(f1_score(yy, ll, average="macro", labels=range(P.shape[1]), zero_division=0)))
    return {k: np.array(v) for k, v in out.items()}


def interval(values):
    values = values[np.isfinite(values)]
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))) if len(values) else (np.nan, np.nan)
