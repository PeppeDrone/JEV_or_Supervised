"""Nested-CV operating-point study for binary zero-shot probabilities."""
import argparse
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest.config import GateError
from jevtest.data import load
from jevtest import figure_style as fs


METHODS = ("ZS_raw",)
ALPHA = 0.05


def _positive_threshold(scores, labels):
    positive_scores = scores[labels == 1]
    if not len(positive_scores):
        raise ValueError("Threshold-selection data contain no positive patients.")
    return float(np.min(positive_scores))


def _exact_binomial_interval(successes, trials, alpha=ALPHA):
    if not trials:
        return np.nan, np.nan
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def _bootstrap_interval(values, iterations, seed):
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(iterations, len(values)))
    estimates = values[draws].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _nested_predictions(run, raw_probabilities, positive_class, bootstrap_iterations):
    positive_labels = (run.y == positive_class).astype(int)
    raw_scores = raw_probabilities[:, positive_class]
    rows = []
    oof = {
        method: {
            "scores": np.full(len(run.y), np.nan),
            "thresholds": np.full(len(run.y), np.nan),
            "predicted": np.full(len(run.y), -1, dtype=int),
        }
        for method in METHODS
    }
    folds = []
    inner_folds = run.study["inner_folds"]

    for fold, (train, test) in enumerate(run.splits):
        train_labels = positive_labels[train]
        train_min_class = int(np.bincount(train_labels, minlength=2).min())
        if train_min_class < 2:
            raise GateError(f"Fold {fold}: insufficient class counts for nested threshold selection.")

        raw_threshold = _positive_threshold(raw_scores[train], train_labels)
        fold_predictions = {
            "ZS_raw": (raw_threshold, raw_scores[test]),
        }

        for method, (threshold, scores) in fold_predictions.items():
            predicted = (scores >= threshold).astype(int)
            oof[method]["scores"][test] = scores
            oof[method]["thresholds"][test] = threshold
            oof[method]["predicted"][test] = predicted
            tp = int(np.sum((positive_labels[test] == 1) & (predicted == 1)))
            fn = int(np.sum((positive_labels[test] == 1) & (predicted == 0)))
            tn = int(np.sum((positive_labels[test] == 0) & (predicted == 0)))
            fp = int(np.sum((positive_labels[test] == 0) & (predicted == 1)))
            folds.append({
                "method": method,
                "fold": fold,
                "threshold": threshold,
                "n_test": len(test),
                "tp": tp,
                "fn": fn,
                "tn": tn,
                "fp": fp,
                "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "alert_rate": float(predicted.mean()),
            })

    if any(not np.isfinite(result["scores"]).all() for result in oof.values()):
        raise RuntimeError("Nested CV did not produce a held-out prediction for every patient.")

    summaries = []
    rng_seed = run.study["seed"] + 97
    for method in METHODS:
        result = oof[method]
        predicted = result["predicted"]
        tp = int(np.sum((positive_labels == 1) & (predicted == 1)))
        fn = int(np.sum((positive_labels == 1) & (predicted == 0)))
        tn = int(np.sum((positive_labels == 0) & (predicted == 0)))
        fp = int(np.sum((positive_labels == 0) & (predicted == 1)))
        sensitivity_ci = _exact_binomial_interval(tp, tp + fn)
        negative_predictions = (predicted[positive_labels == 0] == 0).astype(float)
        specificity_ci = _bootstrap_interval(
            negative_predictions, bootstrap_iterations, rng_seed
        )
        alert_ci = _bootstrap_interval(
            (predicted == 1).astype(float), bootstrap_iterations, rng_seed + 1
        )
        summaries.append({
            "method": method,
            "patients": len(run.y),
            "positive_class": run.classes[positive_class],
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "sensitivity_exact_95_lower": sensitivity_ci[0],
            "sensitivity_exact_95_upper": sensitivity_ci[1],
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "specificity_bootstrap_95_lower": specificity_ci[0],
            "specificity_bootstrap_95_upper": specificity_ci[1],
            "alert_rate": float((predicted == 1).mean()),
            "alert_rate_bootstrap_95_lower": alert_ci[0],
            "alert_rate_bootstrap_95_upper": alert_ci[1],
            "threshold_min": float(result["thresholds"].min()),
            "threshold_median": float(np.median(result["thresholds"])),
            "threshold_max": float(result["thresholds"].max()),
            "threshold_selection": "largest threshold retaining 100% training sensitivity",
        })
    return oof, pd.DataFrame(folds), pd.DataFrame(summaries)


def _plot_summary(summary, output_dir):
    methods = summary["method"].tolist()
    positions = np.arange(len(methods))
    with plt.rc_context(fs.RC):
        figure, axes = plt.subplots(1, 2, figsize=(fs.WIDTH_DOUBLE, 63 * fs.MM), sharey=True)
        metrics = (
            ("sensitivity", "sensitivity_exact_95_lower", "sensitivity_exact_95_upper",
             "Sensitivity", "Exact two-sided 95% binomial CI"),
            ("specificity", "specificity_bootstrap_95_lower", "specificity_bootstrap_95_upper",
             "Specificity", "Patient-bootstrap 95% CI"),
        )
        for index, (axis, (value, lower, upper, label, interval)) in enumerate(zip(axes, metrics)):
            estimates = summary[value].to_numpy()
            low = summary[lower].to_numpy()
            high = summary[upper].to_numpy()
            for position, method, estimate, lo, hi in zip(positions, methods, estimates, low, high):
                axis.errorbar(estimate, position, xerr=[[estimate - lo], [hi - estimate]],
                              fmt="none", capsize=2.2, zorder=2, **fs.line(method))
                axis.plot(estimate, position, zorder=3, **fs.line(method, markers=True))
            axis.axvline(1, color="#8c8c8c", linestyle=(0, (2, 2)), linewidth=0.6, zorder=0)
            axis.set(xlabel=label, xlim=(0, 1.02), yticks=positions)
            axis.set_title(interval, loc="left")
            fs.panel_letter(axis, chr(ord("a") + index), x=-0.08, y=1.0)
        axes[0].set_yticklabels([fs.MODELS[m]["label"] for m in methods])
        axes[0].set_ylim(len(methods) - 0.5, -0.5)
        figure.text(0.01, 0.01, "Points: pooled held-out patients; intervals quantify uncertainty in the displayed estimate.",
                    ha="left", va="bottom", fontsize=6.5, color="#4d4d4d")
        figure.subplots_adjust(left=0.30, right=0.985, bottom=0.20, top=0.88, wspace=0.18)
        fs.save(figure, output_dir, "sensitivity_specificity")


def _plot_thresholds(folds, output_dir):
    with plt.rc_context(fs.RC):
        figure, axis = plt.subplots(figsize=(fs.WIDTH_SINGLE, 62 * fs.MM))
        for method, group in folds.groupby("method", sort=False):
            group = group.sort_values("fold")
            axis.plot(group["fold"] + 1, group["threshold"], label=fs.MODELS[method]["label"],
                      **fs.line(method, markers=True))
        axis.set(xlabel="Outer held-out fold", ylabel="Positive-class probability threshold",
                 xticks=sorted(folds["fold"].unique() + 1), ylim=(0, 1))
        axis.legend(loc="best")
        fs.panel_letter(axis, "a", x=-0.08, y=1.0)
        note = "Each point is a threshold selected from the corresponding outer-training patients only."
        figure.text(0.12, 0.05, "\n".join(textwrap.wrap(note, width=75)),
                    ha="left", va="bottom", fontsize=6.5, color="#4d4d4d")
        figure.subplots_adjust(left=0.20, right=0.98, bottom=0.24, top=0.93)
        fs.save(figure, output_dir, "thresholds_by_fold")


def _plot_fold_performance(folds, output_dir):
    with plt.rc_context(fs.RC):
        figure, axes = plt.subplots(1, 2, figsize=(fs.WIDTH_DOUBLE, 61 * fs.MM), sharey=True)
        for method, group in folds.groupby("method", sort=False):
            group = group.sort_values("fold")
            for axis, metric in zip(axes, ("sensitivity", "specificity")):
                axis.plot(group["fold"] + 1, group[metric], label=fs.MODELS[method]["label"],
                          **fs.line(method, markers=True))
        for index, (axis, ylabel) in enumerate(zip(axes, ("Sensitivity", "Specificity"))):
            axis.set(xlabel="Outer held-out fold", ylabel=ylabel,
                     xticks=sorted(folds["fold"].unique() + 1), ylim=(0, 1.02),
                     yticks=[0, 0.25, 0.5, 0.75, 1])
            fs.panel_letter(axis, chr(ord("a") + index), x=-0.08, y=1.0)
        axes[0].legend(loc="lower left")
        figure.text(0.01, 0.01, "One point = performance in one held-out outer fold; lines connect folds for readability.",
                    ha="left", va="bottom", fontsize=6.5, color="#4d4d4d")
        figure.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.93, wspace=0.20)
        fs.save(figure, output_dir, "fold_performance")


def _write_report(run, summary, folds, output_dir, bootstrap_iterations):
    lines = [
        f"# Zero-shot threshold study: {run.dataset} / {run.outcome}",
        "",
        f"Positive class: **{summary.iloc[0]['positive_class']}**.",
        "",
        "Thresholds are selected inside each outer-training fold as the highest threshold that "
        "retains 100% sensitivity on that fold's threshold-selection data. Results below are "
        "computed only on the corresponding held-out outer folds.",
        "",
        "This is an exploratory retrospective analysis, not a guarantee of zero missed seizures "
        "in future patients. A final operational threshold requires independent external validation.",
        "",
        "Sensitivity intervals are exact two-sided 95% binomial intervals. Specificity and alert-rate "
        f"intervals use {bootstrap_iterations} patient-bootstrap resamples of the held-out predictions.",
        "",
        "## Held-out interpretation",
        "",
    ]
    for row in summary.itertuples(index=False):
        if row.fn:
            lines.append(
                f"- **{row.method} did not achieve zero false negatives:** "
                f"{row.fn} of {row.tp + row.fn} seizure/status patients were missed "
                f"({row.sensitivity:.1%} held-out sensitivity)."
            )
        else:
            lines.append(
                f"- **{row.method} had zero false negatives** among {row.tp + row.fn} "
                "held-out seizure/status patients; this finite-sample result is not a future guarantee."
            )
    lines += [
        "",
        "| Method | TP | FN | TN | FP | Sensitivity (exact 95% CI) | Specificity (bootstrap 95% CI) | Alert rate (bootstrap 95% CI) | Threshold min / median / max |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.tp} | {row.fn} | {row.tn} | {row.fp} | "
            f"{row.sensitivity:.3f} ({row.sensitivity_exact_95_lower:.3f}–{row.sensitivity_exact_95_upper:.3f}) | "
            f"{row.specificity:.3f} ({row.specificity_bootstrap_95_lower:.3f}–{row.specificity_bootstrap_95_upper:.3f}) | "
            f"{row.alert_rate:.3f} ({row.alert_rate_bootstrap_95_lower:.3f}–{row.alert_rate_bootstrap_95_upper:.3f}) | "
            f"{row.threshold_min:.4g} / {row.threshold_median:.4g} / {row.threshold_max:.4g} |"
        )
    lines += [
        "",
        "## Outer-fold results",
        "",
        "| Method | Fold | Threshold | Sensitivity | Specificity | TP | FN | TN | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folds.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.fold + 1} | {row.threshold:.4g} | "
            f"{row.sensitivity:.3f} | {row.specificity:.3f} | "
            f"{row.tp} | {row.fn} | {row.tn} | {row.fp} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_study(root, dataset, outcome, positive_class=None, output_dir=None,
              bootstrap_iterations=2000):
    if bootstrap_iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")
    run = load(root, dataset, outcome)
    if len(run.classes) != 2:
        raise GateError(f"Threshold study requires a binary outcome; {outcome} has {len(run.classes)} classes.")
    if positive_class is None:
        positive_class = run.classes[-1]
    if positive_class not in run.classes:
        raise GateError(f"Positive class {positive_class!r} is not in {run.classes!r}.")
    positive_index = run.classes.index(positive_class)
    raw = run.predictions("ZS_raw")
    if raw is None:
        raise GateError("ZS_raw predictions are unavailable; run the `zeroshot` stage first.")

    destination = Path(output_dir) if output_dir else run.out / "threshold_study"
    destination.mkdir(parents=True, exist_ok=True)
    oof, folds, summary = _nested_predictions(
        run, raw, positive_index, bootstrap_iterations
    )
    summary.to_csv(destination / "summary.csv", index=False)
    folds.to_csv(destination / "fold_results.csv", index=False)

    prediction_rows = []
    for method, result in oof.items():
        prediction_rows.append(pd.DataFrame({
            "method": method,
            "fold": run.fid,
            "patient_id": run.ids,
            "observed": run.y,
            "observed_positive": (run.y == positive_index).astype(int),
            "positive_score": result["scores"],
            "threshold": result["thresholds"],
            "predicted_positive": result["predicted"],
        }))
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        destination / "held_out_predictions.csv", index=False
    )
    _plot_summary(summary, destination)
    _plot_thresholds(folds, destination)
    _plot_fold_performance(folds, destination)
    _write_report(run, summary, folds, destination, bootstrap_iterations)
    print(f"Threshold study complete: {destination}")
    print(summary[[
        "method", "sensitivity", "sensitivity_exact_95_lower",
        "specificity", "specificity_bootstrap_95_lower", "alert_rate",
        "threshold_min", "threshold_median", "threshold_max",
    ]].to_string(index=False))
    return destination, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset directory name under datasets/")
    parser.add_argument("--outcome", required=True, help="Binary outcome key declared in dataset.yaml")
    parser.add_argument("--positive-class", help="Positive class label (default: last class in dataset.yaml)")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root (default: script's parent)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: outcome/threshold_study)")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args(argv)
    run_study(
        args.root.resolve(), args.dataset, args.outcome, args.positive_class,
        args.output_dir, args.bootstrap_iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
