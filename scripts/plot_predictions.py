"""Create diagnostic figures from provenance-validated out-of-fold predictions."""
import argparse
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest.benchmark import ALL_PAIRS, OUTCOME_LABEL
from jevtest.config import outputs_root
from jevtest.data import full_run, load
from jevtest.metrics import auroc, bootstrap, interval, point, resamples
from jevtest import figure_style as fs

LABEL = {"ZS_raw": "System 1", "ICL_full": "ICL full"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Raw-count heatmaps: neutral ink wash (not a percentage / ratio scale).
COUNTS = LinearSegmentedColormap.from_list(
    "counts", ["#f4f3ef", "#d5d3cc", "#8a8882", "#3f3e3a", "#141413"]
)
BENCHMARK_PAIRS = ALL_PAIRS
METRICS = (
    ("auroc", "AUROC", "higher"),
    ("balanced_accuracy", "Balanced accuracy", "higher"),
    ("macro_f1", "Macro F1", "higher"),
    ("brier", "Brier score", "lower"),
    ("log_loss", "Log loss", "lower"),
)
RC = fs.RC


def load_predictions(root, dataset, outcome, whole_table_m2=False):
    """``whole_table_m2``: XGBoost from the whole-table run, scored on its own patients (``y``)."""
    run = load(root, dataset, outcome)
    whole = full_run(root, dataset, outcome) if whole_table_m2 else None
    predictions = {}
    for model in fs.MODELS:
        probabilities = run.predictions(model)
        if probabilities is not None:
            labels = run.prediction_labels(model)
            predictions[model] = {
                "probabilities": probabilities,
                "labels": labels if labels is not None else probabilities.argmax(axis=1),
                "folds": run.fid.copy(),
                "y": run.y,
            }
    if whole is not None:
        probabilities = whole.predictions("M2")
        labels = whole.prediction_labels("M2")
        predictions["M2"] = {
            "probabilities": probabilities,
            "labels": labels if labels is not None else probabilities.argmax(axis=1),
            "folds": whole.fid.copy(),
            "y": whole.y,
        }
    if not predictions:
        raise RuntimeError(
            f"No validated predictions found for {dataset}/{outcome}; run model stages first."
        )
    return run, predictions


def _name(model):
    return fs.MODELS.get(model, {}).get("label", LABEL.get(model, model))


def _line(model, **extra):
    return fs.line(model, **extra)


def _wrap(text, width):
    return "\n".join(textwrap.wrap(str(text), width)) or str(text)


def _probabilistic(model):
    return model != "Mode"


def _heading(figure, title, subtitle=None, legend=None):
    """Short title only — captions belong in the paper, not on the canvas."""
    del subtitle  # accepted for call-site compatibility; never drawn
    figure.suptitle(title, x=0.08, y=0.98, ha="left", va="top",
                    fontsize=fs.LETTER_PT, fontweight="bold", color=INK)
    if legend is not None:
        handles, labels, options = legend
        figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012),
                      fontsize=fs.CELL_PT, frameon=False, **options)


def _save(figure, output_dir, stem):
    fs.save(figure, output_dir, stem)
    plt.close(figure)


def _page_stem(stem, page):
    return stem if page == 1 else f"{stem}_p{page:02d}"


def _pages(items, size):
    for start in range(0, len(items), size):
        yield start // size + 1, items[start:start + size]


def _panel(axis, index):
    fs.panel_letter(axis, chr(ord("a") + index), x=0.0, y=1.01)


def _class_indices(run):
    return [1] if len(run.classes) == 2 else list(range(len(run.classes)))


def _summary(run):
    n = len(run.y)
    if len(run.classes) == 2:
        return f"n = {n} patients · positive class: {run.classes[1]} ({(run.y == 1).mean():.0%})"
    return f"n = {n} patients · {len(run.classes)} classes · pooled out-of-fold predictions"


def compute_metrics(run, predictions):
    metrics = {}
    for model, result in predictions.items():
        truth = result.get("y", run.y)
        draws = resamples(len(truth), run.study["bootstrap_iterations"], run.study["seed"])
        probabilistic = _probabilistic(model)
        values = point(truth, result["probabilities"], result["labels"], probabilistic)
        samples = bootstrap(
            truth, result["probabilities"], draws, result["labels"], probabilistic
        )
        metrics[model] = {
            key: (values[key], *interval(samples[key])) for key, _, _ in METRICS
        }
        metrics[model]["calibration_slope"] = values["calibration_slope"]
        metrics[model]["calibration_intercept"] = values["calibration_intercept"]
    return metrics


def _ranked(metrics):
    """Models ordered best-first by AUROC; colour is unaffected by this order."""
    return sorted(metrics, key=lambda m: -np.nan_to_num(metrics[m]["auroc"][0], nan=-1))


def _ci(value, lower, upper):
    if not np.isfinite(lower):
        return f"{value:.3f}"
    return f"{value:.3f} ({lower:.3f}–{upper:.3f})"


def _square(axis):
    axis.set(xlim=(0, 1), ylim=(0, 1.005), xticks=np.linspace(0, 1, 6),
             yticks=np.linspace(0, 1, 6))
    axis.set_aspect("equal")


def _curve_grid(count, panel=2.15, columns=3):
    columns = min(columns, count)
    rows = -(-count // columns)
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(fs.WIDTH_DOUBLE, min(fs.MAX_HEIGHT, panel * rows + 1.45)),
        squeeze=False, sharex=True, sharey=True,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.87,
                            wspace=0.20, hspace=0.30)
    for axis in axes.ravel()[count:]:
        axis.remove()
    return figure, axes, columns


def _label_outer(axes, count, columns, xlabel, ylabel):
    flat = axes.ravel()
    for index in range(count):
        axis = flat[index]
        has_below = index + columns < count
        axis.tick_params(labelbottom=not has_below, labelleft=index % columns == 0)
        if not has_below:
            axis.set_xlabel(xlabel, labelpad=2)
        if index % columns == 0:
            axis.set_ylabel(ylabel, labelpad=2)


def plot_roc(run, predictions, metrics, output_dir):
    order = _ranked(metrics)
    if len(run.classes) == 2:
        figure, axis = plt.subplots(figsize=(fs.WIDTH_SINGLE, fs.WIDTH_SINGLE))
        figure.subplots_adjust(left=0.18, right=0.97, bottom=0.29, top=0.82)
        inset = axis.inset_axes([0.42, 0.10, 0.54, 0.50])
        for model in order:
            truth = run.y == 1
            fpr, tpr, _ = roc_curve(truth, predictions[model]["probabilities"][:, 1])
            label = f"{_name(model):<14} {_ci(*metrics[model]['auroc'])}"
            axis.plot(fpr, tpr, label=label, **_line(model))
            inset.plot(fpr, tpr, **_line(model))
        axis.plot([0, 1], [0, 1], color="#b9b8b2", linestyle="--", linewidth=1)
        _square(axis)
        axis.set(xlabel="False-positive rate (1 − specificity)",
                 ylabel="True-positive rate (sensitivity)")
        inset.set(xlim=(0, 0.3), ylim=(0.6, 1.0), xticks=[0, 0.1, 0.2, 0.3],
                  yticks=[0.6, 0.8, 1.0])
        inset.tick_params(labelsize=7.5)
        inset.set_title("Zoom: FPR 0–0.3", fontsize=8, fontweight="normal", color=MUTED)
        for side in ("top", "right"):
            inset.spines[side].set_visible(True)
        axis.add_patch(Rectangle((0, 0.6), 0.3, 0.4, fill=False, edgecolor="#9a9994",
                                 linewidth=0.9, zorder=0))
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1,
                    title="AUROC (95% bootstrap CI)", title_fontsize=8.5,
                    fontsize=fs.CELL_PT)
        _panel(axis, 0)
        _heading(figure, "ROC curves")
        _save(figure, output_dir, "roc_curves")
    else:
        for page, class_ids in _pages(list(range(len(run.classes))), 6):
            figure, axes, columns = _curve_grid(len(class_ids))
            for panel_index, (class_index, axis) in enumerate(zip(class_ids, axes.ravel())):
                truth = run.y == class_index
                for model in order:
                    scores = predictions[model]["probabilities"][:, class_index]
                    fpr, tpr, _ = roc_curve(truth, scores)
                    axis.plot(fpr, tpr, **_line(model))
                axis.plot([0, 1], [0, 1], color="#9a9994", linestyle="--", linewidth=0.8)
                _square(axis)
                axis.set_title(_wrap(f"{run.classes[class_index]} (n = {truth.sum()})", 24),
                               fontsize=fs.LABEL_PT)
                _panel(axis, panel_index)
            _label_outer(axes, len(class_ids), columns, "False-positive rate", "True-positive rate")
            _heading(figure, "ROC curves (one-vs-rest)",
                     legend=([fs.handle(m) for m in order], [_name(m) for m in order],
                             {"ncol": min(5, len(order))}))
            _save(figure, output_dir, _page_stem("roc_curves", page))


def plot_precision_recall(run, predictions, metrics, output_dir):
    order = _ranked(metrics)

    def draw(axis, class_index, small):
        truth = (run.y == class_index).astype(int)
        scored = []
        for model in order:
            scores = predictions[model]["probabilities"][:, class_index]
            scored.append((average_precision_score(truth, scores), model, scores))
        for ap, model, scores in sorted(scored, key=lambda s: -s[0]):
            precision, recall, _ = precision_recall_curve(truth, scores)
            label = f"{_name(model)} {ap:.3f}" if small else f"{_name(model):<14} AP {ap:.3f}"
            axis.plot(recall, precision, label=label, drawstyle="steps-post", **_line(model))
        prevalence = truth.mean()
        axis.axhline(prevalence, color="#9a9994", linestyle=":", linewidth=1)
        if not small:
            axis.text(0.02, prevalence + 0.015, f"no skill = prevalence {prevalence:.2f}",
                      fontsize=8.5, color=MUTED, va="bottom")
        _square(axis)

    if len(run.classes) == 2:
        figure, axis = plt.subplots(figsize=(fs.WIDTH_SINGLE, fs.WIDTH_SINGLE))
        figure.subplots_adjust(left=0.18, right=0.97, bottom=0.29, top=0.82)
        draw(axis, 1, small=False)
        axis.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), title="Average precision",
                    title_fontsize=fs.CELL_PT, fontsize=fs.CELL_PT, ncol=2)
        _panel(axis, 0)
        _heading(figure, "Precision–recall curves")
        _save(figure, output_dir, "precision_recall_curves")
    else:
        for page, class_ids in _pages(list(range(len(run.classes))), 6):
            figure, axes, columns = _curve_grid(len(class_ids))
            for panel_index, (class_index, axis) in enumerate(zip(class_ids, axes.ravel())):
                draw(axis, class_index, small=True)
                truth = run.y == class_index
                axis.set_title(_wrap(f"{run.classes[class_index]} (n = {truth.sum()}, {truth.mean():.0%})", 24),
                               fontsize=fs.LABEL_PT)
                _panel(axis, panel_index)
            _label_outer(axes, len(class_ids), columns, "Recall", "Precision")
            _heading(figure, "Precision–recall curves (one-vs-rest)",
                     legend=([fs.handle(m) for m in order], [_name(m) for m in order],
                             {"ncol": min(5, len(order))}))
            _save(figure, output_dir, _page_stem("precision_recall_curves", page))


def _wilson(successes, n, z=1.96):
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def _reliability(scores, outcome, bins=10):
    """Equal-count bins so every point rests on a similar number of patients."""
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([scores.min(), scores.max() + 1e-9])
    ids = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        selected = ids == b
        if selected.any():
            n = selected.sum()
            low, high = _wilson(outcome[selected].sum(), n)
            rows.append((scores[selected].mean(), outcome[selected].mean(), low, high, n))
    return np.array(rows)


def plot_calibration(run, predictions, metrics, output_dir):
    models = [m for m in _ranked(metrics) if _probabilistic(m)]
    binary = len(run.classes) == 2
    for page, model_page in _pages(models, 6):
        columns = min(3, len(model_page))
        rows = -(-len(model_page) // columns)
        figure = plt.figure(figsize=(fs.WIDTH_DOUBLE, min(fs.MAX_HEIGHT, 2.55 * rows + 1.45)))
        outer = figure.add_gridspec(rows, columns, hspace=0.48, wspace=0.30,
                                    left=0.09, right=0.98, bottom=0.12, top=0.86)
        for index, model in enumerate(model_page):
            row, column = divmod(index, columns)
            inner = outer[row, column].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
            axis = figure.add_subplot(inner[0])
            histogram = figure.add_subplot(inner[1], sharex=axis)
            probabilities = predictions[model]["probabilities"]
            if binary:
                scores, outcome = probabilities[:, 1], (run.y == 1).astype(float)
            else:
                scores = probabilities.max(axis=1)
                outcome = (probabilities.argmax(axis=1) == run.y).astype(float)
            table = _reliability(scores, outcome)
            style = _line(model)
            axis.plot([0, 1], [0, 1], color="#9a9994", linestyle="--", linewidth=0.8)
            axis.fill_between(table[:, 0], table[:, 2], table[:, 3], color=style["color"],
                              alpha=0.16, linewidth=0)
            axis.plot(table[:, 0], table[:, 1], marker="o", markersize=3.5,
                      markeredgecolor="white", markeredgewidth=0.7, **style)
            _square(axis)
            axis.set_aspect("auto")
            axis.set_title(_name(model), fontsize=fs.LABEL_PT)
            axis.tick_params(labelbottom=False)
            brier = metrics[model]["brier"][0]
            if binary and np.isfinite(metrics[model]["calibration_slope"]):
                note = (f"Brier {brier:.3f}\nslope {metrics[model]['calibration_slope']:.2f}"
                        f" · intercept {metrics[model]['calibration_intercept']:+.2f}")
            else:
                ece = np.sum(table[:, 4] * np.abs(table[:, 0] - table[:, 1])) / table[:, 4].sum()
                note = f"Brier {brier:.3f}\nECE {ece:.3f}"
            axis.text(0.03, 0.97, note, transform=axis.transAxes, va="top", fontsize=fs.CELL_PT,
                      color=INK, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
            histogram.hist(scores, bins=np.linspace(0, 1, 41), color=style["color"], alpha=0.75,
                           linewidth=0)
            histogram.set_yscale("log")
            histogram.set_yticks([])
            histogram.minorticks_off()
            histogram.grid(False)
            histogram.spines["left"].set_visible(False)
            if index % columns == 0:
                axis.set_ylabel("Observed frequency" if binary else "Observed accuracy")
            histogram.set_xlabel("Predicted probability" if binary else "Top-class confidence")
            _panel(axis, index)
        _heading(figure, "Calibration curves" if binary else "Top-label calibration")
        _save(figure, output_dir, _page_stem("calibration_curves", page))


def plot_confusion_matrices(run, predictions, metrics, output_dir):
    """Raw-count matrices paginated by model; one shared scale per page."""
    k = len(run.classes)
    names = [_wrap(c, 16 if k == 2 else 20) for c in run.classes]
    order = _ranked(metrics)
    page_size = 6 if k <= 4 else 4
    for page, model_page in _pages(order, page_size):
        columns = min(3 if k <= 2 else 2, len(model_page))
        rows = -(-len(model_page) // columns)
        figure, axes = plt.subplots(rows, columns, squeeze=False,
                                    figsize=(fs.WIDTH_DOUBLE, min(fs.MAX_HEIGHT, 2.25 * rows + 1.45)))
        figure.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.86,
                                wspace=0.34, hspace=0.48)
        matrices = {m: confusion_matrix(run.y, predictions[m]["labels"], labels=np.arange(k))
                    for m in model_page}
        vmax = max(int(matrix.max()) for matrix in matrices.values()) or 1
        for index, (axis, model) in enumerate(zip(axes.ravel(), model_page)):
            counts = matrices[model]
            axis.imshow(counts, vmin=0, vmax=vmax, cmap=COUNTS)
            axis.set(xticks=np.arange(k), yticks=np.arange(k))
            axis.set_xticks(np.arange(k + 1) - 0.5, minor=True)
            axis.set_yticks(np.arange(k + 1) - 0.5, minor=True)
            axis.grid(False)
            axis.grid(which="minor", color="white", linewidth=1.2)
            axis.tick_params(which="both", length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
            for row in range(k):
                for col in range(k):
                    value = int(counts[row, col])
                    axis.text(col, row, str(value), ha="center", va="center", fontsize=fs.CELL_PT,
                              fontweight="bold" if row == col else "normal",
                              color="white" if value > 0.55 * vmax else INK)
            axis.set_title(f"{_name(model)}\nBalanced accuracy {metrics[model]['balanced_accuracy'][0]:.3f}",
                           fontsize=fs.LABEL_PT)
            if index + columns >= len(model_page):
                axis.set_xticklabels(names, rotation=35 if k > 2 else 0, ha="right" if k > 2 else "center",
                                     rotation_mode="anchor", fontsize=fs.CELL_PT)
                axis.set_xlabel("Predicted class")
            else:
                axis.set_xticklabels([])
            if index % columns == 0:
                axis.set_yticklabels(names, fontsize=fs.CELL_PT)
                axis.set_ylabel("Observed class")
            else:
                axis.set_yticklabels([])
            _panel(axis, index)
        for axis in axes.ravel()[len(model_page):]:
            axis.remove()
        _heading(figure, "Confusion matrices")
        _save(figure, output_dir, _page_stem("confusion_matrices", page))


def plot_metric_comparison(run, predictions, metrics, output_dir):
    """Forest plots retain the pooled estimate and bootstrap interval, never bars."""
    order = _ranked(metrics)
    shown = [item for item in METRICS if any(np.isfinite(metrics[m][item[0]][0]) for m in order)]
    for page, metric_page in _pages(shown, 3):
        figure, axes = plt.subplots(1, len(metric_page), sharey=True,
                                    figsize=(fs.WIDTH_DOUBLE, min(fs.MAX_HEIGHT, 0.38 * len(order) + 1.7)))
        axes = np.atleast_1d(axes)
        figure.subplots_adjust(left=0.21, right=0.90, bottom=0.14, top=0.84, wspace=0.35)
        y_positions = np.arange(len(order))
        for panel_index, (axis, (key, title, better)) in enumerate(zip(axes, metric_page)):
            values = np.array([metrics[m][key] for m in order], dtype=float)
            finite = np.isfinite(values[:, 0])
            if not finite.any():
                continue
            best = np.nanmax(values[:, 0]) if better == "higher" else np.nanmin(values[:, 0])
            axis.axvline(best, color="#9a9994", linestyle=":", linewidth=0.8)
            for y, model, (value, lower, upper) in zip(y_positions, order, values):
                if not np.isfinite(value):
                    axis.text(0.5, y, "not available", transform=axis.get_yaxis_transform(),
                              ha="center", va="center", fontsize=fs.CELL_PT, color=MUTED)
                    continue
                style = _line(model)
                if np.isfinite(lower):
                    axis.hlines(y, lower, upper, color=style["color"], linewidth=1.3)
                axis.plot(value, y, marker="o", markersize=4.2, markeredgecolor="white",
                          markeredgewidth=0.7, **style)
            bounds = values[finite, 1:3]
            low, high = np.nanmin(bounds), np.nanmax(bounds)
            span = max(high - low, 0.02)
            axis.set_xlim(low - 0.10 * span, high + 0.12 * span)
            axis.set_title(f"{title} {'↑' if better == 'higher' else '↓'}", fontsize=fs.LABEL_PT)
            axis.grid(axis="y", visible=False)
            axis.tick_params(axis="y", length=0)
            _panel(axis, panel_index)
        axes[0].set(yticks=y_positions, yticklabels=[_name(m) for m in order],
                    ylim=(len(order) - 0.5, -0.5))
        _heading(figure, "Model comparison")
        _save(figure, output_dir, _page_stem("metric_comparison", page))


def plot_fold_auroc(run, predictions, metrics, output_dir):
    """Show outer-fold AUROC as individual observations (Nature Methods: avoid bars of means alone)."""
    order = _ranked(metrics)
    folds = len(run.splits)
    scores = np.full((len(order), folds), np.nan)
    for i, model in enumerate(order):
        result = predictions[model]
        for fold in range(folds):
            selected = result["folds"] == fold
            truth = run.y[selected]
            if len(np.unique(truth)) == len(run.classes):
                scores[i, fold] = auroc(truth, result["probabilities"][selected])
    height = min(fs.MAX_HEIGHT, max(70 * fs.MM, 12 * fs.MM * len(order) + 40 * fs.MM))
    figure, axis = plt.subplots(figsize=(fs.WIDTH_DOUBLE, height))
    x = np.arange(len(order))
    jitter = np.linspace(-0.12, 0.12, folds) if folds > 1 else np.zeros(1)
    for fold in range(folds):
        axis.plot(x + jitter[fold], scores[:, fold], color="#c9c8c2", linewidth=0.75, zorder=1)
    for i, model in enumerate(order):
        style = _line(model)
        axis.scatter(x[i] + jitter, scores[i], s=22, color=style["color"], edgecolor="white",
                     linewidth=0.6, marker=fs.MODELS[model]["marker"], zorder=3)
        if np.isfinite(scores[i]).any():
            mean = np.nanmean(scores[i])
            axis.hlines(mean, x[i] - 0.22, x[i] + 0.22, color=INK, linewidth=1.1, zorder=4)
    finite = scores[np.isfinite(scores)]
    if finite.size:
        span = max(finite.max() - finite.min(), 0.02)
        axis.set_ylim(max(0, finite.min() - 0.15 * span), min(1, finite.max() + 0.15 * span))
    means = [f"{np.nanmean(row):.3f}" if np.isfinite(row).any() else "n/a" for row in scores]
    axis.set_xticks(x, [f"{_name(m)}\nmean {v}" for m, v in zip(order, means)])
    axis.set_xlim(-0.5, len(order) - 0.5)
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("AUROC" if len(run.classes) == 2 else "Macro one-vs-rest AUROC")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.84)
    _heading(figure, "AUROC by outer fold")
    _save(figure, output_dir, "fold_auroc")


def _outcome_short(outcome):
    return OUTCOME_LABEL.get(outcome, outcome.replace("_", " "))


def load_benchmark_bundle(root, pairs=None):
    """Load every benchmark outcome that already has at least one validated prediction CSV."""
    bundle = []
    for dataset, outcome in (pairs or BENCHMARK_PAIRS):
        pred_dir = outputs_root(root) / dataset / outcome / "predictions"
        if not pred_dir.is_dir() or not any(pred_dir.glob("*.csv")):
            continue
        try:
            run, predictions = load_predictions(root, dataset, outcome, whole_table_m2=True)
        except Exception as error:
            print(f"skip {dataset}/{outcome}: {error}", flush=True)
            continue
        metrics = compute_metrics(run, predictions)
        bundle.append((run, predictions, metrics))
    return bundle


# Combined overview: prefer a readable subset of models (heatmap still shows all present).
COMBINED_FOCUS = ["M2", "ZS_raw"]
COMBINED_RC = {
    **RC,
    "font.size": 8.0, "axes.labelsize": 9.0, "axes.titlesize": 9.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6, "lines.markersize": 5.5,
}
TITLE_PT, LEG_PT = 10.0, 8.0


def _combined_height(n, row_mm=10.5, header_mm=48, floor_mm=95):
    """Fit every outcome on one Nature canvas (max 170 mm)."""
    usable = fs.MAX_HEIGHT - header_mm * fs.MM
    row = min(row_mm * fs.MM, usable / max(n, 1))
    return min(fs.MAX_HEIGHT, max(floor_mm * fs.MM, row * n + header_mm * fs.MM))


def _combined_ytick(n):
    return 6.0 if n > 24 else 8.0


def _combined_models(bundle):
    return [m for m in fs.MODELS if any(m in metrics for _, _, metrics in bundle)]


def _combined_focus(bundle):
    present = _combined_models(bundle)
    focus = [m for m in COMBINED_FOCUS if m in present]
    return focus or present[:4]


def _heading_big(figure, title, subtitle=None, legend=None):
    del subtitle  # accepted for call-site compatibility; never drawn
    figure.suptitle(title, x=0.06, y=0.975, ha="left", va="top",
                    fontsize=TITLE_PT, fontweight="bold", color=INK)
    if legend is not None:
        handles, labels, options = legend
        figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                      fontsize=LEG_PT, frameon=False, handlelength=2.8, **options)


def _legend_handles(models):
    from matplotlib.lines import Line2D
    handles = []
    for model in models:
        style = fs.MODELS[model]
        handles.append(Line2D([], [], color=style["color"], linestyle=style["linestyle"],
                              marker=style["marker"], markersize=6.5, linewidth=1.8,
                              markeredgecolor="white", markeredgewidth=0.5))
    return handles


def _ink_on(color_rgb):
    """White text on dark cells, black on light (perceptual luminance)."""
    r, g, b = color_rgb[:3]
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.45 else INK


# Wins over XGBoost only: white → light red → deep red. Ties / losses stay white.
WIN_CMAP = LinearSegmentedColormap.from_list(
    "win_vs_xgb",
    ["#FFFFFF", "#FDD4C2", "#FC8D6B", "#E34A33", "#B30000"],
)
REFERENCE_FILL = "#D0D0D0"  # fixed grey XGBoost column
EQUAL_OR_WORSE_FILL = "#FFFFFF"


def _heatmap_vs_reference(bundle, output_dir, stem, title, value_fn, cbar_label,
                          reference="M2"):
    """Heatmap: grey XGBoost column; white if ≤ XGBoost; red shades if better.

    Cell text shows the absolute metric. Colour encodes only positive
    metric(model) − metric(XGBoost).
    """
    from matplotlib.colors import Normalize

    outcomes = [_outcome_short(run.outcome) for run, _, _ in bundle]
    models = _combined_focus(bundle)
    if not models or not outcomes:
        return
    if reference not in models:
        return _heatmap_absolute(bundle, output_dir, stem, title, value_fn, cbar_label)

    values = np.full((len(outcomes), len(models)), np.nan)
    for i, item in enumerate(bundle):
        for j, model in enumerate(models):
            values[i, j] = value_fn(item, model)

    ref_j = models.index(reference)
    wins = np.full_like(values, np.nan)  # positive Δ only
    for i in range(len(outcomes)):
        ref = values[i, ref_j]
        if not np.isfinite(ref):
            continue
        for j, model in enumerate(models):
            if j == ref_j:
                continue
            if np.isfinite(values[i, j]):
                delta = values[i, j] - ref
                if delta > 0:
                    wins[i, j] = delta

    finite = wins[np.isfinite(wins)]
    vmax = float(max(finite.max(), 0.02)) if finite.size else 0.1
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = WIN_CMAP.copy()

    height = _combined_height(len(outcomes), row_mm=9.5)
    figure, axis = plt.subplots(figsize=(fs.WIDTH_DOUBLE, height))
    rgba = np.zeros((len(outcomes), len(models), 4))
    ref_rgb = (*to_rgb(REFERENCE_FILL), 1.0)
    white_rgb = (*to_rgb(EQUAL_OR_WORSE_FILL), 1.0)
    miss_rgb = (*to_rgb(fs.MISSING), 1.0)
    for i in range(len(outcomes)):
        for j in range(len(models)):
            if j == ref_j and np.isfinite(values[i, j]):
                rgba[i, j] = ref_rgb
            elif np.isfinite(wins[i, j]):
                rgba[i, j] = cmap(norm(wins[i, j]))
            elif np.isfinite(values[i, j]):
                rgba[i, j] = white_rgb  # equal or worse than XGBoost
            else:
                rgba[i, j] = miss_rgb

    axis.imshow(rgba, aspect="auto", interpolation="nearest")
    axis.set_xticks(np.arange(len(models)), [_name(m) for m in models], rotation=25,
                    ha="right", rotation_mode="anchor", fontsize=8)
    axis.set_yticks(np.arange(len(outcomes)), outcomes, fontsize=_combined_ytick(len(outcomes)))
    axis.tick_params(length=0, pad=3)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.add_patch(Rectangle((ref_j - 0.5, -0.5), 1.0, len(outcomes),
                             fill=False, edgecolor=INK, linewidth=1.2, zorder=5))

    for i in range(len(outcomes)):
        for j, model in enumerate(models):
            value = values[i, j]
            if not np.isfinite(value):
                axis.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                      fontweight="bold", color=_ink_on(rgba[i, j]))
            if model == "ZS_raw" and np.isfinite(wins[i, j]):
                axis.plot(
                    j - 0.37, i - 0.25,
                    marker="^", markersize=5.2,
                    markerfacecolor=fs.IMPROVED_COLOR, markeredgecolor="white",
                    markeredgewidth=0.6, linestyle="none", zorder=6,
                )

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = figure.colorbar(mappable, ax=axis, fraction=0.04, pad=0.025)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.8)
    figure.subplots_adjust(left=0.23, right=0.88, bottom=0.18, top=0.90)
    _heading_big(figure, title)
    _save(figure, output_dir, stem)


def _heatmap_absolute(bundle, output_dir, stem, title, value_fn, cbar_label, vmin_floor=0.5):
    """Absolute-value heatmap fallback when the XGBoost reference column is absent."""
    outcomes = [_outcome_short(run.outcome) for run, _, _ in bundle]
    models = _combined_focus(bundle)
    if not models or not outcomes:
        return
    grid = np.full((len(outcomes), len(models)), np.nan)
    for i, item in enumerate(bundle):
        for j, model in enumerate(models):
            grid[i, j] = value_fn(item, model)
    height = _combined_height(len(outcomes), row_mm=9.5)
    figure, axis = plt.subplots(figsize=(fs.WIDTH_DOUBLE, height))
    finite = grid[np.isfinite(grid)]
    vmin = max(vmin_floor, float(np.nanmin(finite))) if finite.size else vmin_floor
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-3
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(fs.MISSING)
    image = axis.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(models)), [_name(m) for m in models], rotation=25,
                    ha="right", rotation_mode="anchor", fontsize=8)
    axis.set_yticks(np.arange(len(outcomes)), outcomes, fontsize=_combined_ytick(len(outcomes)))
    axis.tick_params(length=0, pad=3)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for i in range(len(outcomes)):
        for j in range(len(models)):
            value = grid[i, j]
            if not np.isfinite(value):
                axis.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=8)
                continue
            rgba = cmap((value - vmin) / (vmax - vmin))
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                      fontweight="bold", color=_ink_on(rgba))
    cbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.025)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=2.5, width=0.8)
    figure.subplots_adjust(left=0.23, right=0.88, bottom=0.18, top=0.90)
    _heading_big(figure, title)
    _save(figure, output_dir, stem)


def _outcome_bands(axis, centers, half=0.48):
    """Alternating bands + separators so each outcome reads as one group."""
    for i, yc in enumerate(centers):
        if i % 2 == 0:
            axis.axhspan(yc - half, yc + half, facecolor="#F2F2F2", edgecolor="none", zorder=0)
        if i > 0:
            mid = 0.5 * (centers[i - 1] + yc)
            axis.axhline(mid, color="#D0D0D0", linewidth=0.7, zorder=1)


def _model_dodge(n_models, half=0.34):
    if n_models <= 1:
        return [0.0]
    return list(np.linspace(-half, half, n_models))


def plot_combined_roc_binary(bundle, output_dir):
    """Binary ROC grid — 3 columns, thick curves, readable panel titles."""
    binary = [(run, predictions, metrics) for run, predictions, metrics in bundle
              if len(run.classes) == 2]
    if not binary:
        return
    focus = _combined_focus(bundle)
    columns = 3
    rows = -(-len(binary) // columns)
    panel = min(1.85, (fs.MAX_HEIGHT - 1.5) / max(rows, 1))
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(fs.WIDTH_DOUBLE, min(fs.MAX_HEIGHT, panel * rows + 1.5)),
        squeeze=False, sharex=True, sharey=True,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.91, wspace=0.22, hspace=0.38)
    for axis in axes.ravel()[len(binary):]:
        axis.remove()
    legend_models = []
    for index, (axis, (run, predictions, metrics)) in enumerate(zip(axes.ravel(), binary)):
        order = [m for m in focus if m in metrics] or _ranked(metrics)[:4]
        for model in order:
            fpr, tpr, _ = roc_curve(predictions[model].get("y", run.y), predictions[model]["probabilities"][:, 1])
            axis.plot(fpr, tpr, **_line(model, linewidth=1.7))
            if model not in legend_models:
                legend_models.append(model)
        axis.plot([0, 1], [0, 1], color="#9a9994", linestyle="--", linewidth=1.0)
        _square(axis)
        axis.set_title(_outcome_short(run.outcome), fontsize=8.5, pad=3)
        axis.tick_params(labelsize=7.5)
        _panel(axis, index)
    _label_outer(axes, len(binary), columns, "False-positive rate", "True-positive rate")
    _heading_big(figure, "ROC curves",
                 legend=(_legend_handles(legend_models), [_name(m) for m in legend_models],
                         {"ncol": min(4, len(legend_models))}))
    _save(figure, output_dir, "roc_curves_binary_all")


def _ba_value(item, model):
    _, _, metrics = item
    return metrics[model]["balanced_accuracy"][0] if model in metrics else np.nan


def _ba_delta(metrics, model):
    if model not in metrics or "M2" not in metrics:
        return np.nan
    return metrics[model]["balanced_accuracy"][0] - metrics["M2"]["balanced_accuracy"][0]


PANEL_LETTERS = ("a", "b")
SHOW_HEADING = True


def plot_combined_balanced_accuracy(bundle, output_dir):
    """One figure: (a) XGBoost / System 1 heatmap, (b) XGBoost forest and System 1 Δ."""
    from matplotlib.colors import Normalize

    models = [m for m in ("M2", "ZS_raw") if any(m in metrics for _, _, metrics in bundle)]
    if "M2" not in models or not bundle:
        return
    n = len(bundle)
    y = np.arange(n)
    labels = [_outcome_short(run.outcome) for run, _, _ in bundle]
    values = np.array([[_ba_value(item, model) for model in models] for item in bundle], dtype=float)
    ref_j = models.index("M2")
    wins = np.full_like(values, np.nan)
    for i in range(n):
        ref = values[i, ref_j]
        if not np.isfinite(ref):
            continue
        for j, model in enumerate(models):
            if j == ref_j or not np.isfinite(values[i, j]):
                continue
            delta = values[i, j] - ref
            if delta > 0:
                wins[i, j] = delta
    finite = wins[np.isfinite(wins)]
    vmax = float(max(finite.max(), 0.02)) if finite.size else 0.1
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = WIN_CMAP.copy()

    deltas_all = [_ba_delta(metrics, "ZS_raw") for _, _, metrics in bundle]
    deltas_all = [d for d in deltas_all if np.isfinite(d)]
    if deltas_all:
        lo, hi = min(deltas_all), max(deltas_all)
        pad = 0.04
        delta_xlim = (min(lo, 0) - pad, max(hi, 0) + pad)
    else:
        delta_xlim = (-0.1, 0.1)

    height = _combined_height(n, row_mm=9.5)
    figure = plt.figure(figsize=(fs.WIDTH_DOUBLE, height))
    outer = figure.add_gridspec(
        2, 2, width_ratios=[1.00, 2.20], height_ratios=[1.0, 0.055],
        wspace=0.55, hspace=0.08,
        left=0.155, right=0.985, bottom=0.08, top=0.84,
    )
    heat = figure.add_subplot(outer[0, 0])
    cbar_ax = figure.add_subplot(outer[1, 0])
    forests = outer[0, 1].subgridspec(1, 2, wspace=0.48, width_ratios=[1.0, 1.0])
    xgb_ax = figure.add_subplot(forests[0, 0])
    zs_ax = figure.add_subplot(forests[0, 1])

    rgba = np.zeros((n, len(models), 4))
    ref_rgb = (*to_rgb(REFERENCE_FILL), 1.0)
    white_rgb = (*to_rgb(EQUAL_OR_WORSE_FILL), 1.0)
    miss_rgb = (*to_rgb(fs.MISSING), 1.0)
    for i in range(n):
        for j in range(len(models)):
            if j == ref_j and np.isfinite(values[i, j]):
                rgba[i, j] = ref_rgb
            elif np.isfinite(wins[i, j]):
                rgba[i, j] = cmap(norm(wins[i, j]))
            elif np.isfinite(values[i, j]):
                rgba[i, j] = white_rgb
            else:
                rgba[i, j] = miss_rgb
    heat.imshow(rgba, aspect="auto", interpolation="nearest")
    heat.set_xticks(np.arange(len(models)))
    heat.set_xticklabels([_name(m) for m in models], fontsize=8.5)
    for tick, model in zip(heat.get_xticklabels(), models):
        tick.set_color(fs.MODELS[model]["color"])
    heat.tick_params(axis="x", top=False, bottom=False, labeltop=True, labelbottom=False,
                     length=0, pad=4)
    heat.set_yticks(y, labels, fontsize=_combined_ytick(n))
    heat.tick_params(axis="y", length=0, pad=3)
    for spine in heat.spines.values():
        spine.set_visible(False)
    heat.add_patch(Rectangle((ref_j - 0.5, -0.5), 1.0, n,
                             fill=False, edgecolor=INK, linewidth=1.2, zorder=5))
    heat.set_ylim(n - 0.55, -0.55)
    for i in range(n):
        for j, model in enumerate(models):
            value = values[i, j]
            if not np.isfinite(value):
                heat.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=7.5)
                continue
            heat.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                      fontweight="bold", color=_ink_on(rgba[i, j]))
            if model == "ZS_raw" and np.isfinite(wins[i, j]):
                heat.plot(
                    j - 0.37, i - 0.25,
                    marker="^", markersize=4.6,
                    markerfacecolor=fs.IMPROVED_COLOR, markeredgecolor="white",
                    markeredgewidth=0.5, linestyle="none", zorder=6,
                )
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = figure.colorbar(mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Δ vs XGBoost when better", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.5, length=2.0, width=0.7)

    for axis, model, xlabel, xlim, xticks in (
        (xgb_ax, "M2", "Balanced accuracy", (0.30, 0.98), [0.5, 0.75, 0.9]),
        (zs_ax, "ZS_raw", "Δ balanced accuracy", delta_xlim,
         [delta_xlim[0], 0.0, delta_xlim[1]] if delta_xlim[0] < 0 < delta_xlim[1] else list(delta_xlim)),
    ):
        style = _line(model, linewidth=1.8)
        marker = fs.MODELS[model]["marker"]
        _outcome_bands(axis, y, half=0.45)
        if model == "M2":
            for yi, (_, _, metrics) in enumerate(bundle):
                if model not in metrics:
                    continue
                value, low, high = metrics[model]["balanced_accuracy"]
                if not np.isfinite(value):
                    continue
                axis.hlines(yi, low, high, color=style["color"], linewidth=2.4, zorder=3)
                axis.plot(value, yi, marker=marker, markersize=5.8, linestyle="none",
                          color=style["color"], markeredgecolor="white", markeredgewidth=0.6, zorder=4)
            axis.axvline(0.5, color="#9a9994", linestyle=":", linewidth=1.0, zorder=2)
        else:
            axis.axvline(0, color="#9a9994", linestyle=":", linewidth=1.0, zorder=2)
            for yi, (_, _, metrics) in enumerate(bundle):
                delta = _ba_delta(metrics, model)
                if not np.isfinite(delta):
                    continue
                axis.plot([0, delta], [yi, yi], color=style["color"], linewidth=2.2,
                          solid_capstyle="round", zorder=3)
                axis.plot(delta, yi, marker=marker, markersize=5.8, linestyle="none",
                          color=style["color"], markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        axis.set(xlim=xlim, ylim=(n - 0.55, -0.55), xticks=xticks, xlabel=xlabel)
        if model != "M2":
            axis.set_xticklabels([f"{t:.2f}" for t in xticks])
        axis.set_title(_name(model), fontsize=8.5, color=style["color"], pad=4)
        axis.set_yticks(y, labels)
        axis.tick_params(labelsize=7.0, labelleft=True, left=False,
                         axis="y", pad=2)
        axis.tick_params(axis="x", labelsize=7.0)
        plt.setp(axis.get_yticklabels(), fontsize=_combined_ytick(n))

    figure.canvas.draw()
    letter_y = max(heat.get_position().y1, xgb_ax.get_position().y1) + 0.018
    figure.text(0.012, letter_y, PANEL_LETTERS[0],
                fontsize=fs.LETTER_PT, fontweight="bold", color=INK, va="bottom", ha="left")
    figure.text(xgb_ax.get_position().x0, letter_y, PANEL_LETTERS[1],
                fontsize=fs.LETTER_PT, fontweight="bold", color=INK, va="bottom", ha="left")
    if SHOW_HEADING:
        _heading_big(figure, "Balanced accuracy by outcome")
    _save(figure, output_dir, "balanced_accuracy_all")


def create_combined_figures(root, output_dir=None, pairs=None):
    """Summary figures covering every available benchmark outcome (no page splits)."""
    root = Path(root)
    destination = Path(output_dir) if output_dir else outputs_root(root) / "combined" / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    obsolete = {
        "auroc_bars_all",
        "auroc_forest_all",
        "confusion_matrices_all",
        "auroc_heatmap_all",
        "auroc_delta_all",
        "roc_curves_binary_all",
        "balanced_accuracy_cmat",
        "balanced_accuracy_forest_all",
        "balanced_accuracy_delta_all",
    }
    for path in list(destination.glob("*")):
        if path.suffix.lower() not in {".png", ".pdf", ".svg"}:
            continue
        stem = path.stem
        if "_p0" in stem or stem in obsolete:
            path.unlink(missing_ok=True)
    for stem in obsolete:
        for ext in (".png", ".pdf", ".svg"):
            (destination / f"{stem}{ext}").unlink(missing_ok=True)
    with plt.rc_context(COMBINED_RC):
        bundle = load_benchmark_bundle(root, pairs)
        if not bundle:
            print("No predictions available for combined figures yet.")
            return destination
        plot_combined_balanced_accuracy(bundle, destination)
    print(f"Created combined figures from {len(bundle)} outcome(s): {destination}")
    return destination


def create_figures(root, dataset, outcome, output_dir=None, combined=True):
    run, predictions = load_predictions(root, dataset, outcome)
    destination = Path(output_dir) if output_dir else run.out / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        metrics = compute_metrics(run, predictions)
        plot_roc(run, predictions, metrics, destination)
        plot_precision_recall(run, predictions, metrics, destination)
        plot_calibration(run, predictions, metrics, destination)
        plot_confusion_matrices(run, predictions, metrics, destination)
        plot_metric_comparison(run, predictions, metrics, destination)
        plot_fold_auroc(run, predictions, metrics, destination)
    print(f"Created six figures from {len(predictions)} validated model(s): {destination}")
    if combined:
        create_combined_figures(root)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Dataset directory name under datasets/")
    parser.add_argument("--outcome", help="Outcome key declared in dataset.yaml")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root (default: script's parent)")
    parser.add_argument("--output-dir", type=Path, help="Figure directory (default: outcome/figures)")
    parser.add_argument("--combined-only", action="store_true",
                        help="Only rebuild cross-outcome combined figures under outputs/combined/")
    parser.add_argument("--no-combined", action="store_true",
                        help="Skip automatic combined figures after a single-outcome plot")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.combined_only:
        create_combined_figures(root, args.output_dir)
        return 0
    if not args.dataset or not args.outcome:
        parser.error("--dataset and --outcome are required unless --combined-only is set")
    create_figures(root, args.dataset, args.outcome, args.output_dir,
                   combined=not args.no_combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
