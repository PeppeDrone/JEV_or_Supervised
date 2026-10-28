#!/usr/bin/env python
"""Publication figures for the Qwen3.5 backbone ladder on the shared XGB hold-out patients.

Reads predictions/ZS_raw_holdout.csv from every outputs_qwen3.5_*/ dir (or the dirs given on the command line)
and outputs/<dataset>/<outcome>/holdout/XGB.csv; computes AUROC (macro one-vs-rest for multiclass) and
balanced accuracy of the argmax per task, a patient-level paired bootstrap of AUROC(zero-shot) - AUROC(XGB) per
task, across-task tests per size, and associations with table complexity and codebook detail.

Writes to reports/backbone_ladder/, each figure as SVG + PDF (editable text) + 300-dpi PNG:

  fig_ladder_scaling      a median AUROC vs size (IQR bands) | b median balanced accuracy vs size |
                          c median paired gap with CI and Wilcoxon p | d per-task CI verdicts per size
  fig_ladder_per_task     a AUROC per task, every size + XGB | b balanced accuracy per task
  fig_ladder_paired_ci    forest plots of the per-task paired gap for the two largest backbones
  fig_ladder_vs_xgb       per-task AUROC against XGB, one panel per size
  fig_ladder_covariates   paired gap against inverted F1, N1, C2, T2, missingness, codebook detail (two largest sizes)
  fig_ladder_scaling_slope  per-task AUROC gain per decade of parameters against the same covariates
  fig_ladder_summary      one unifying figure / graphical abstract: a both metrics vs size | b paired gap per size |
                          c per-task AUROC vs XGB at 27B | d zero-shot minus reference scorers | e gap vs best-marker
                          share | f balanced-accuracy gap, binary vs multiclass
  fig_ladder_mechanisms   zero-shot AUROC minus reference scorers (best single marker, direction-only score, additive
                          logistic regression, XGB) and the paired gap against best-marker share, interaction gain,
                          XGB training n, predictor count, class count (two largest sizes)
  ladder_table.csv, stats_by_size.csv, scaling_steps.csv (paired gain of each size over the previous one on shared
  tasks), covariate_spearman.csv, scaling_slopes.csv, mechanism_spearman.csv, mechanism_rank_regression.csv

Covariates come from outputs/combined/figures/complexity_and_gaps.csv and
outputs/combined/codebook/codebook_quality_and_gaps.csv (scripts/analyse_uci_complexity_gaps.py,
scripts/analyse_codebook_quality.py); they describe the tables, not a backbone. Mechanism covariates come from
scripts/task_mechanism_covariates.py (run it first; same development/hold-out split). Partial backbones (a run still
in progress) are included with the tasks they have; titles say "n/45 tasks, run in progress".

Every figure and table is produced twice: on AUROC (plain names) and on balanced accuracy of the argmax at the
model's own operating point (suffix _bacc), the quantity a single-patient zero-shot user gets without any local
labels; XGB keeps its tuned decision rule. --metric auc|bacc|both (default both).

  .venv/bin/python scripts/plot_backbone_ladder.py
  .venv/bin/python scripts/plot_backbone_ladder.py --metric bacc outputs_qwen3.5_27b outputs_qwen3.5_397b_a17b_fp8
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jevtest.benchmark import ALL_PAIRS  # noqa: E402
from task_names import task_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "scientific_revision_2026-10-25_v1" / "ladder"
BASE = ROOT / "outputs"
SIZE_B = {"0.8b": 0.8, "2b": 2, "4b": 4, "9b": 9, "27b": 27, "35b_a3b": 35, "122b_a10b": 122, "397b_a17b": 397,
          "397b_a17b_fp8": 397, "35b_a3b_fp8": 35, "122b_a10b_fp8": 122}

# ---- Nature Portfolio style -------------------------------------------------------------------------------------
MM = 1 / 25.4
W1, W2 = 89 * MM, 183 * MM                       # single / double column widths
NAVY, TEAL, TURQ, ORANGE, GREY, INK = "#0B2545", "#1B7F79", "#7FC8C3", "#F2A65A", "#8C8C8C", "#222222"
SIZE_RAMP = ["#BFE3E0", "#7FC8C3", "#3FA8A3", "#1B7F79", "#13606A", "#0F4C5C", "#0B2545"]
VERDICT = {"loss": ORANGE, "parity": "#BDBDBD", "win": TEAL}
COVARIATE_LABELS = {"complexity_F1": "Inverted F1, class overlap", "N1": "N1, border fraction",
                    "C2": "C2, imbalance ratio", "T2": "T2, features per sample",                     "missingness": "Missingness", "explanation_rate": "Codebook explanation rate"}
LOG_COV = {"C2", "T2"}


def style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "legend.title_fontsize": 6.5, "legend.frameon": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "lines.linewidth": 0.9, "patch.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.grid": False, "figure.dpi": 110, "savefig.dpi": 300, "savefig.facecolor": "white",
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        # superscripts in P values use the same face as the body text, not the mathtext default (DejaVu)
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.default": "regular",
    })


ZERO = dict(color=GREY, ls="--", lw=0.7)   # every reference line through zero / identity


def regress(ax, x, y, color=NAVY):
    """OLS line with a 95% confidence band for the mean, drawn behind the points."""
    from scipy.stats import t as tdist

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 8 or np.std(x) == 0:
        return
    b1, b0 = np.polyfit(x, y, 1)
    res = y - (b0 + b1 * x)
    n = len(x)
    se = np.sqrt((res @ res) / (n - 2))
    xx = np.linspace(x.min(), x.max(), 60)
    half = tdist.ppf(0.975, n - 2) * se * np.sqrt(1 / n + (xx - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum())
    yy = b0 + b1 * xx
    ax.fill_between(xx, yy - half, yy + half, color=color, alpha=0.10, lw=0, zorder=1)
    ax.plot(xx, yy, color=color, lw=0.9, zorder=2, solid_capstyle="round")


def count_label(n, total):
    """'45 tasks' or '32/45 tasks, run in progress' for a backbone that has not finished the ladder."""
    return f"{n} tasks" if n >= total else f"{n}/{total} tasks, run in progress"


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    """Editable vector (SVG with live text, PDF with Type 42 fonts) plus a 300-dpi RGB raster export."""
    name = name + msuffix()
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.close(fig)
    png = OUT / f"{name}.png"
    with Image.open(png) as im:
        im.convert("RGB").save(png, dpi=(300, 300))


# ---- data -------------------------------------------------------------------------------------------------------
def label_of(path: Path) -> str:
    tag = path.name.replace("outputs_qwen3.5_", "")
    size = re.sub(r"_fp8$", "", tag)
    return size.upper().replace("_", "-") + (" FP8" if tag.endswith("_fp8") else "")


def short(lab: str) -> str:
    """Tick label: '397B-A17B FP8' -> '397B\nMoE A17B' (precision is a Methods detail); dense sizes unchanged."""
    size = lab.split("-")[0]
    active = lab.split("-")[1].split()[0] if "-A" in lab else ""
    return size + (f"\nMoE {active}" if active else "")


def disp(lab: str) -> str:
    """One-line display name: '397B MoE A17B'."""
    return short(lab).replace("\n", " ")


def annotate_corner(ax, xv, yv, text, ylow):
    """Put the ρ/P box in the lower corner with fewer low-lying points."""
    xv, yv = np.asarray(xv, float), np.asarray(yv, float)
    mid = np.nanmedian(xv)
    left = np.sum((yv < ylow) & (xv <= mid))
    right = np.sum((yv < ylow) & (xv > mid))
    x, ha = (0.03, "left") if left <= right else (0.97, "right")
    ax.text(x, 0.04, text, transform=ax.transAxes, ha=ha, va="bottom", fontsize=5.8, color=INK)


# ---- metric switch ------------------------------------------------------------------------------------------------
# "auc": rank AUROC (does the readout carry the information). "bacc": balanced accuracy of the argmax at the model's
# own operating point (what a single-patient zero-shot user actually gets; XGB uses its tuned decision rule).
METRICS = {"auc": {"name": "AUROC", "delta": "ΔAUROC", "suffix": ""},
           "bacc": {"name": "BA", "delta": "ΔBA", "suffix": "_bacc"}}
METRIC = "auc"


def mname():
    return METRICS[METRIC]["name"]


def mdelta():
    return METRICS[METRIC]["delta"]


def msuffix():
    return METRICS[METRIC]["suffix"]


def fast_bacc(y: np.ndarray, labels: np.ndarray, K: int) -> float:
    """Full-class balanced accuracy; undefined if a bootstrap draw omits a class."""
    if len(np.unique(y)) != K:
        return np.nan
    out = []
    for c in range(K):
        pos = y == c
        if pos.any():
            out.append((labels[pos] == c).mean())
    return float(np.mean(out)) if out else np.nan


def fast_auc(y: np.ndarray, P: np.ndarray) -> float:
    """Rank AUROC; macro one-vs-rest for K > 2 (vectorised so the paired bootstrap stays cheap)."""
    from scipy.stats import rankdata

    K = P.shape[1]
    if len(np.unique(y)) != K:
        return np.nan
    out = []
    for c in (range(K) if K > 2 else [1]):
        pos = y == c
        n1, n0 = int(pos.sum()), int((~pos).sum())
        if n1 and n0:
            r = rankdata(P[:, c])
            out.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out)) if out else np.nan


def metrics(y, P, predicted=None):
    from sklearn.metrics import balanced_accuracy_score

    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    labels = P.argmax(axis=1) if predicted is None else predicted
    return fast_auc(y, P), balanced_accuracy_score(y, labels)


def paired_bootstrap(y, P_zs, P_xgb, lab_zs, lab_xgb, iterations=1000, seed=20261008):
    """Patient-level paired bootstrap of ΔAUROC and Δbalanced accuracy (same resamples for both)."""
    rng = np.random.default_rng(seed)
    n, K = len(y), P_zs.shape[1]
    d_auc, d_bacc = np.empty(iterations), np.empty(iterations)
    for b in range(iterations):
        idx = rng.integers(0, n, n)
        d_auc[b] = fast_auc(y[idx], P_zs[idx]) - fast_auc(y[idx], P_xgb[idx])
        d_bacc[b] = fast_bacc(y[idx], lab_zs[idx], K) - fast_bacc(y[idx], lab_xgb[idx], K)
    ci = lambda d: (float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)))
    return ci(d_auc), ci(d_bacc)


def collect(dirs):
    rows = []
    for dataset, outcome in ALL_PAIRS:
        xgb_path = BASE / dataset / outcome / "holdout" / "XGB_optuna.csv"
        if not xgb_path.exists():
            continue
        xgb = pd.read_csv(xgb_path, dtype={"patient_id": str}).set_index("patient_id")
        xcols = [c for c in xgb.columns if c.startswith("p__")]
        y, P_x = xgb["observed"].to_numpy(), xgb[xcols].to_numpy(float)
        row = {"dataset": dataset, "outcome": outcome, "task": dataset.replace("uci_", "").replace("_", " "),
               "n": len(xgb), "n_classes": len(xcols)}
        row["auc_XGB"], row["bacc_XGB"] = metrics(y, P_x, xgb["predicted"].to_numpy())
        for d in dirs:
            f = d / dataset / outcome / "predictions" / "ZS_raw_holdout.csv"
            if not f.exists():
                continue
            z = pd.read_csv(f, dtype={"patient_id": str}).set_index("patient_id")
            if set(z.index) != set(xgb.index):
                print(f"warning: {f} does not list the XGB hold-out patients; skipped", file=sys.stderr)
                continue
            z = z.loc[xgb.index]
            P_z = z[[c for c in z.columns if c.startswith("p__")]].to_numpy(float)
            lab = label_of(d)
            row[f"auc_{lab}"], row[f"bacc_{lab}"] = metrics(y, P_z)
            if pd.notna(row[f"auc_{lab}"]):
                row[f"dauc_{lab}"] = row[f"auc_{lab}"] - row["auc_XGB"]
                row[f"dbacc_{lab}"] = row[f"bacc_{lab}"] - row["bacc_XGB"]
                (row[f"dauc_lo_{lab}"], row[f"dauc_hi_{lab}"]), (row[f"dbacc_lo_{lab}"], row[f"dbacc_hi_{lab}"]) = \
                    paired_bootstrap(y, P_z, P_x, P_z.argmax(axis=1), xgb["predicted"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def as_metric(table, metric):
    """View of the ladder table in which auc_* / dauc_* hold the chosen metric (the figure code reads those keys).
    For "bacc" the two metric families are swapped so that bacc_* then holds AUROC (used only by the two-metric panels)."""
    if metric == "auc":
        return table.copy()
    ren = {}
    for c in table.columns:
        if c.startswith("dbacc_"):
            ren[c] = "dauc_" + c[len("dbacc_"):]
        elif c.startswith("dauc_"):
            ren[c] = "dbacc_" + c[len("dauc_"):]
        elif c.startswith("bacc_"):
            ren[c] = "auc_" + c[len("bacc_"):]
        elif c.startswith("auc_"):
            ren[c] = "bacc_" + c[len("auc_"):]
    return table.rename(columns=ren)


def task_covariates():
    cov = None
    comp = BASE / "combined" / "figures" / "complexity_and_gaps.csv"
    if comp.exists():
        cov = pd.read_csv(comp)[["dataset", "outcome", "complexity_F1", "N1", "C2", "T2", "missingness"]]
    code = BASE / "combined" / "codebook" / "codebook_quality_and_gaps.csv"
    if code.exists():
        k = pd.read_csv(code)[["dataset", "outcome", "explanation_rate"]]
        cov = k if cov is None else cov.merge(k, on=["dataset", "outcome"], how="outer")
    return cov


def verdicts(table, lab):
    lo, hi = table[f"dauc_lo_{lab}"], table[f"dauc_hi_{lab}"]
    return pd.Series(np.where(lo > 0, "win", np.where(hi < 0, "loss", "parity")), index=table.index).where(lo.notna())


# ---- figures ----------------------------------------------------------------------------------------------------
def fig_scaling(table, labels, sizes, colors, stats, plt):
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.0), gridspec_kw={"wspace": 0.6})
    complete = table.dropna(subset=[f"auc_{lab}" for lab in labels])
    other = "AUROC" if METRIC == "bacc" else "BA"
    for ax, m, name, letter in zip(axes[:2], ["auc", "bacc"], [mname(), other], "ab"):
        xgb = complete[f"{m}_XGB"]
        ax.axhspan(xgb.quantile(0.25), xgb.quantile(0.75), color=ORANGE, alpha=0.18, lw=0)
        ax.axhline(xgb.median(), color=ORANGE, ls="--", lw=0.9)
        xs = [s for _, s in sizes]
        med = [complete[f"{m}_{lab}"].median() for lab, _ in sizes]
        q1 = [complete[f"{m}_{lab}"].quantile(0.25) for lab, _ in sizes]
        q3 = [complete[f"{m}_{lab}"].quantile(0.75) for lab, _ in sizes]
        ax.fill_between(xs, q1, q3, color=TURQ, alpha=0.35, lw=0)
        ax.plot(xs, med, "-", color=NAVY, lw=1.0)
        ax.scatter(xs, med, s=14, color=[colors[lab] for lab, _ in sizes], edgecolor=NAVY, linewidth=0.5, zorder=3)
        ax.axhline(0.5, color=GREY, lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_xlabel("Parameters (billions)")
        ax.set_ylabel(f"Median {name if name == 'AUROC' else name.lower()} over tasks")
        ax.set_ylim(0.45, 1.0)
        ax.text(0.98, 0.04, f"n = {len(complete)} tasks", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=INK)
        panel(ax, letter)
    axes[0].text(0.03, 0.965, "XGB median, IQR", color="#C77A2E", fontsize=6, ha="left", va="top", transform=axes[0].transAxes)
    axes[0].text(0.03, 0.885, "Zero-shot median, IQR", color=NAVY, fontsize=6, ha="left", va="top", transform=axes[0].transAxes)

    ax = axes[2]
    x = np.arange(len(stats))
    ax.axhline(0, **ZERO)
    ax.axhspan(-0.05, 0.0, color=GREY, alpha=0.12, lw=0)
    ax.errorbar(x, stats.median_delta, yerr=[stats.median_delta - stats.median_ci_lo, stats.median_ci_hi - stats.median_delta],
                fmt="none", ecolor=NAVY, elinewidth=0.8, capsize=2.5, capthick=0.8)
    ax.scatter(x, stats.median_delta, s=16, color=[colors[b] for b in stats.backbone], edgecolor=NAVY, linewidth=0.5, zorder=3)
    ylo = min(-0.33, stats.median_ci_lo.min() - 0.09)
    for xi, p in enumerate(stats.wilcoxon_p):
        exp = int(np.floor(np.log10(p))) if p > 0 else 0
        txt = f"{p / 10 ** exp:.1f}×10$^{{{exp}}}$" if p < 1e-3 else f"{p:.3f}"
        ax.text(xi, ylo + 0.012, txt, ha="center", va="bottom", fontsize=5.2, color=INK, rotation=90)
    ax.text(-0.45, ylo + 0.012, "P", ha="right", va="bottom", fontsize=5.5, color=INK, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([short(b) for b in stats.backbone], fontsize=5.6)
    for xi, n in enumerate(stats.tasks):
        ax.text(xi, 0.065, f"{n}", ha="center", va="bottom", fontsize=5.2, color=INK)
    ax.text(-0.45, 0.065, "n", ha="right", va="bottom", fontsize=5.5, color=INK, style="italic")
    ax.set_ylabel(f"Median paired {mdelta()}\n(zero-shot − XGB)")
    ax.set_ylim(ylo, 0.11)
    ax.set_yticks([t for t in np.arange(-0.4, 0.06, 0.05) if t >= ylo])
    panel(ax, "c")

    ax = axes[3]
    bottom = np.zeros(len(stats))
    for key, name in [("ci_below_0", "XGB better"), ("ci_covers_0", "Undecided"), ("ci_above_0", "Zero-shot better")]:
        vals = stats[key].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=VERDICT[{"ci_below_0": "loss", "ci_covers_0": "parity", "ci_above_0": "win"}[key]],
               width=0.7, edgecolor="white", linewidth=0.5, label=name)
        for xi, (b, v) in enumerate(zip(bottom, vals)):
            if v:
                ax.text(xi, b + v / 2, str(int(v)), ha="center", va="center", fontsize=6, color="white" if key != "ci_covers_0" else INK)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([short(b) for b in stats.backbone], fontsize=5.6)
    ax.set_ylabel("Tasks")
    ax.set_ylim(0, stats.tasks.max() * 1.5)
    ax.legend(loc="upper left", handlelength=0.9, handleheight=0.9, labelspacing=0.25, borderaxespad=0.1)
    panel(ax, "d")
    save(fig, "fig_ladder_scaling")


def fig_per_task(table, labels, colors, plt):
    d = table.sort_values("auc_XGB").reset_index(drop=True)
    yy = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 0.115 * len(d) + 0.7), gridspec_kw={"wspace": 0.05})
    other = "AUROC" if METRIC == "bacc" else "BA"
    for ax, m, name, letter in zip(axes, ["auc", "bacc"], [mname(), other], "ab"):
        for i in yy:
            vals = [d.loc[i, f"{m}_{lab}"] for lab in labels if pd.notna(d.loc[i, f"{m}_{lab}"])]
            if len(vals) > 1:
                ax.plot([min(vals), max(vals)], [i, i], color="#D9D9D9", lw=0.7, zorder=1)
        for lab in labels:
            ax.scatter(d[f"{m}_{lab}"], yy, s=22, color=colors[lab], edgecolor=NAVY, linewidth=0.35, alpha=0.8, zorder=3, label=disp(lab))
        ax.scatter(d[f"{m}_XGB"], yy, s=30, marker="D", color=ORANGE, edgecolor=INK, linewidth=0.35, alpha=0.8, zorder=4, label="XGB")
        ax.axvline(0.5, color=GREY, lw=0.5, ls=":")
        ax.set_xlim(0.1, 1.02)
        ax.set_ylim(-0.8, len(d) - 0.2)
        ax.set_xlabel(name, fontsize=8)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_yticks(yy)
        ax.tick_params(axis="y", length=0, labelsize=6.8)
        panel(ax, letter, x=-0.3 if letter == "a" else -0.08, y=1.012)
    names_n = [f"{task_name(ds, t)}, {int(n)}" for ds, t, n in zip(d.dataset, d.task, d.n)]
    axes[0].set_yticklabels(names_n, fontsize=6.8)
    axes[0].text(0, 1.006, "Dataset, N", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=6.8, clip_on=False, color=INK)
    axes[1].set_yticklabels([])
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.58, 1.012), ncol=len(names), markerscale=1.1,
               fontsize=7.5, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.0, frameon=False)
    fig.subplots_adjust(top=0.97)
    save(fig, "fig_ladder_per_task")


def fig_paired_ci(table, labels, plt):
    big = [lab for lab in labels if table[f"dauc_{lab}"].notna().sum() >= 10][-2:]
    if not big:
        return
    fig, axes = plt.subplots(1, len(big), figsize=(W2 if len(big) == 2 else W1, 0.115 * len(table) + 0.7), squeeze=False,
                             gridspec_kw={"wspace": 0.5})
    for ax, lab, letter in zip(axes[0], big, "ab"):
        s = table.dropna(subset=[f"dauc_{lab}"]).sort_values(f"dauc_{lab}").reset_index(drop=True)
        v = verdicts(s, lab)
        for i in range(len(s)):
            ax.plot([s[f"dauc_lo_{lab}"][i], s[f"dauc_hi_{lab}"][i]], [i, i], color=VERDICT[v[i]], lw=1.0, solid_capstyle="butt")
            ax.plot(s[f"dauc_{lab}"][i], i, "o", color=VERDICT[v[i]], ms=2.6, mec=INK, mew=0.3)
        ax.axvline(0, **ZERO)
        ax.axvspan(-0.05, 0.05, color=GREY, alpha=0.1, lw=0)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([f"{task_name(ds, t)}, {n}" for ds, t, n in zip(s.dataset, s.task, s.n)], fontsize=5.8)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.8, len(s) - 0.2)
        c = v.value_counts()
        ax.set_title(f"{disp(lab)} ({count_label(len(s), len(table))}): {c.get('loss', 0)} XGB better, {c.get('parity', 0)} not significant, "
                     f"{c.get('win', 0)} zero-shot better", fontsize=6.5, color=INK, pad=3)
        ax.set_xlabel(f"{mdelta()}, zero-shot − XGB (95% paired bootstrap CI)")
        panel(ax, letter, x=-0.42)
    save(fig, "fig_ladder_paired_ci")


def fig_vs_xgb(table, labels, colors, plt):
    n = len(labels)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(W2, 1.55 * rows + 0.45), squeeze=False, gridspec_kw={"wspace": 0.3, "hspace": 0.5})
    fig.subplots_adjust(bottom=0.26 / rows, top=0.93)
    for k, (ax, lab) in enumerate(zip(axes.flat, labels)):
        s = table.dropna(subset=[f"auc_{lab}"])
        v = verdicts(s, lab)
        ax.plot([0.1, 1.02], [0.1, 1.02], **ZERO)
        ax.fill_between([0.1, 1], [0.05, 0.95], [0.15, 1.05], color=GREY, alpha=0.08, lw=0)
        ax.scatter(s["auc_XGB"], s[f"auc_{lab}"], s=np.clip(s["n"] / 8, 5, 40), color=[VERDICT[x] for x in v],
                   edgecolor=INK, linewidth=0.3, alpha=0.9)
        ax.set_xlim(0.1, 1.02)
        ax.set_ylim(0.1, 1.02)
        ax.set_xticks([0.2, 0.5, 0.8, 1.0])
        ax.set_yticks([0.2, 0.5, 0.8, 1.0])
        ax.set_title(f"{disp(lab)} ({len(s)} tasks)" if len(s) >= len(table) else f"{disp(lab)} ({len(s)}/{len(table)} tasks)",
                     fontsize=7, pad=3, color=colors[lab] if lab != labels[0] else NAVY)
        if k % cols == 0:
            ax.set_ylabel(f"{mname()} zero-shot")
        if k // cols == rows - 1:
            ax.set_xlabel(f"{mname()} XGB")
        panel(ax, "abcdefghij"[k], x=-0.22)
    for ax in axes.flat[n:]:
        ax.axis("off")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", ls="", color=VERDICT[k], mec=INK, mew=0.3, ms=4, label=t)
               for k, t in [("loss", "XGB better (CI < 0)"), ("parity", "Undecided (CI covers 0)"), ("win", "Zero-shot better (CI > 0)")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, frameon=False,
               columnspacing=1.5)
    save(fig, "fig_ladder_vs_xgb")


def fig_covariates(table, labels, colors, plt):
    from scipy.stats import spearmanr

    cov = task_covariates()
    if cov is None:
        return None, None
    merged = table.merge(cov, on=["dataset", "outcome"], how="left")
    measures = [m for m in COVARIATE_LABELS if m in merged]
    big = [lab for lab in labels if merged[f"dauc_{lab}"].notna().sum() >= 10][-2:]
    assoc = []
    if big:
        ncol = 4
        per = int(np.ceil(len(measures) / ncol))  # rows per backbone
        fig, axes = plt.subplots(per * len(big), ncol, figsize=(W2, 1.25 * per * len(big) + 0.3), squeeze=False,
                                 gridspec_kw={"wspace": 0.1, "hspace": 0.6})
        for b, lab in enumerate(big):
            s = merged.dropna(subset=[f"dauc_{lab}"])
            v = verdicts(s, lab)
            for k, m in enumerate(measures):
                ax = axes[b * per + k // ncol, k % ncol]
                ok = s[m].notna()
                xv, yv = s.loc[ok, m], s.loc[ok, f"dauc_{lab}"]
                ax.axhline(0, **ZERO)
                xplot = np.log10(xv) if m in LOG_COV else xv
                regress(ax, xplot, yv)
                ax.scatter(xplot, yv, s=np.clip(s.loc[ok, "n"] / 10, 4, 30),
                           color=[VERDICT[x] for x in v[ok]], edgecolor=INK, linewidth=0.25, alpha=0.9, zorder=3)
                if ok.sum() >= 8:
                    rho, p = spearmanr(xv, yv)
                    assoc.append({"backbone": lab, "measure": m, "tasks": int(ok.sum()), "spearman_rho": rho, "p": p})
                    ax.text(0.03, 0.96, f"ρ = {rho:+.2f}, P = {p:.2f}", transform=ax.transAxes, ha="left", va="top",
                            fontsize=5.8, color=INK)
                ax.set_ylim(-0.75, 0.3)
                ax.set_yticks([-0.6, -0.3, 0.0])
                ax.set_xlabel(("log10 " if m in LOG_COV else "") + COVARIATE_LABELS[m], fontsize=6)
                if k % ncol == 0:
                    ax.set_ylabel(mdelta() if METRIC == "auc" else "ΔBalanced acc.", fontsize=6.5)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=5.8)
            for k in range(len(measures), per * ncol):
                axes[b * per + k // ncol, k % ncol].axis("off")
            panel(axes[b * per, 0], "ab"[b], x=-0.42)
            axes[b * per, 0].set_title(f"{disp(lab)} ({count_label(len(s), len(table))})", loc="left", fontsize=7, pad=3, color=INK, x=0.0)
        save(fig, "fig_ladder_covariates")
    pd.DataFrame(assoc).to_csv(OUT / f"covariate_spearman{msuffix()}.csv", index=False)
    return merged, measures


def fig_scaling_slope(merged, measures, labels, sizes, plt):
    from scipy.stats import spearmanr

    if merged is None or len(sizes) < 3:
        return
    rows = []
    for _, row in merged.iterrows():
        avail = [(lab, s) for lab, s in sizes if pd.notna(row.get(f"auc_{lab}"))]
        if len(avail) >= 3:
            xs = [np.log10(s) for _, s in avail]
            ys = [row[f"auc_{lab}"] for lab, _ in avail]
            last = avail[-1][0]
            lo, hi = row[f"dauc_lo_{last}"], row[f"dauc_hi_{last}"]
            verdict = "win" if lo > 0 else ("loss" if hi < 0 else "parity")
            rows.append({"dataset": row.dataset, "outcome": row.outcome, "task": row.task, "n": row.n, "sizes": len(avail),
                         "slope_auc_per_decade": np.polyfit(xs, ys, 1)[0], "largest_size": last, "auc_largest": ys[-1],
                         "auc_XGB": row.auc_XGB, "verdict_largest": verdict, **{m: row[m] for m in measures}})
    slopes = pd.DataFrame(rows)
    slopes.to_csv(OUT / f"scaling_slopes{msuffix()}.csv", index=False)
    from matplotlib.lines import Line2D

    ncol = 4
    nrow = int(np.ceil(len(measures) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(W2, 1.25 * nrow + 0.5), squeeze=False, gridspec_kw={"wspace": 0.1, "hspace": 0.6})
    for k, m in enumerate(measures):
        ax = axes[k // ncol, k % ncol]
        ok = slopes[m].notna()
        xv = slopes.loc[ok, m]
        xplot = np.log10(xv) if m in LOG_COV else xv
        regress(ax, xplot, slopes.loc[ok, "slope_auc_per_decade"])
        ax.scatter(xplot, slopes.loc[ok, "slope_auc_per_decade"],
                   color=[VERDICT[x] for x in slopes.loc[ok, "verdict_largest"]], s=np.clip(slopes.loc[ok, "n"] / 10, 4, 30),
                   edgecolor=INK, linewidth=0.25, alpha=0.9, zorder=3)
        if ok.sum() >= 8:
            rho, p = spearmanr(xv, slopes.loc[ok, "slope_auc_per_decade"])
            annotate_corner(ax, np.log10(xv) if m in LOG_COV else xv, slopes.loc[ok, "slope_auc_per_decade"],
                            f"ρ = {rho:+.2f}\nP = {p:.2f}", -0.25)
        ax.axhline(0, **ZERO)
        ax.set_ylim(-0.62, 0.3)
        ax.set_yticks([-0.4, -0.2, 0.0, 0.2])
        ax.set_xlabel(("log10 " if m in LOG_COV else "") + COVARIATE_LABELS[m], fontsize=6)
        ax.tick_params(labelsize=5.8)
        if k % ncol == 0:
            ax.set_ylabel(f"{mname() if METRIC == 'auc' else 'Balanced acc.'} gain per\ndecade of parameters", fontsize=6.5)
        else:
            ax.set_yticklabels([])
    for k in range(len(measures), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    for r in range(nrow):
        panel(axes[r, 0], "ab"[r], x=-0.42)
    handles = [Line2D([], [], marker="o", ls="", color=VERDICT[k], mec=INK, mew=0.3, ms=4, label=t)
               for k, t in [("loss", "XGB better at largest size (CI < 0)"), ("parity", "Undecided (CI covers 0)"),
                            ("win", "Zero-shot better (CI > 0)")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, frameon=False,
               columnspacing=1.5)
    fig.subplots_adjust(top=0.95, bottom=0.17)
    save(fig, "fig_ladder_scaling_slope")


def across_task_stats(table, labels):
    from scipy.stats import wilcoxon

    rows, rng = [], np.random.default_rng(0)
    for lab in labels:
        dl = table[f"dauc_{lab}"].dropna().to_numpy()
        if len(dl) < 5:
            continue
        boots = [np.median(rng.choice(dl, len(dl))) for _ in range(2000)]
        try:
            p = wilcoxon(dl).pvalue
        except ValueError:
            p = np.nan
        v = verdicts(table, lab).value_counts()
        auc_col, ba_col = ("bacc", "auc") if METRIC == "bacc" else ("auc", "bacc")  # as_metric() swaps the families
        rows.append({"backbone": lab, "tasks": len(dl), "metric": METRIC, "median_delta": np.median(dl),
                     "median_ci_lo": np.percentile(boots, 2.5), "median_ci_hi": np.percentile(boots, 97.5), "wilcoxon_p": p,
                     "zs_better_point": int((dl > 0).sum()), "within_0.05": int((dl >= -0.05).sum()),
                     "ci_above_0": int(v.get("win", 0)), "ci_covers_0": int(v.get("parity", 0)), "ci_below_0": int(v.get("loss", 0)),
                     "median_auc": table[f"{auc_col}_{lab}"].median(), "median_bacc": table[f"{ba_col}_{lab}"].median(),
                     "collapsed_bacc_0.5": int((table[f"{ba_col}_{lab}"] == 0.5).sum())})
    return pd.DataFrame(rows)


def scaling_steps(table, labels):
    """Paired gain of each size over the previous one, on the tasks both have scored (plus largest vs 27B)."""
    from scipy.stats import wilcoxon

    pairs = list(zip(labels[:-1], labels[1:]))
    if len(labels) >= 3 and (labels[-2], labels[-1]) not in pairs:
        pairs.append((labels[-2], labels[-1]))
    rows, rng = [], np.random.default_rng(1)
    for small, big in pairs:
        d = (table[f"auc_{big}"] - table[f"auc_{small}"]).dropna().to_numpy()
        if len(d) < 5:
            continue
        boots = [np.median(rng.choice(d, len(d))) for _ in range(2000)]
        try:
            p = wilcoxon(d).pvalue
        except ValueError:
            p = np.nan
        rows.append({"from": small, "to": big, "shared_tasks": len(d), "metric": METRIC, "median_gain": np.median(d),
                     "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5), "wilcoxon_p": p,
                     "to_better": int((d > 0).sum()), "from_better": int((d < 0).sum())})
    return pd.DataFrame(rows)


# ---- mechanism covariates (scripts/task_mechanism_covariates.py) --------------------------------------------------
MECH_LABELS = {"univariate_share": "Best-marker share",
               "interaction_gain": "Interaction gain",
               "n_development": "XGB training n",
               "n_predictors": "Predictors",
               "n_classes": "Classes"}
MECH_LOG = {"n_development", "n_predictors"}
REFERENCES_BY_METRIC = {
    "auc": [("best_univariate_auc", "best\nmarker"), ("direction_only_auc", "direction\nonly"), ("additive_auc", "additive\nLR"),
            ("auc_XGB", "XGB")],
    "bacc": [("best_univariate_bacc", "best\nmarker"), ("direction_only_bacc", "direction\nonly"), ("additive_bacc", "additive\nLR"),
             ("auc_XGB", "XGB")]}  # auc_XGB holds the chosen metric after as_metric()
MECH_BY_METRIC = {"auc": {"univariate_share": "univariate_share", "interaction_gain": "interaction_gain"},
                  "bacc": {"univariate_share": "univariate_share", "interaction_gain": "interaction_gain_bacc"}}


def mechanism_covariates():
    f = OUT.parent / "task_properties.csv"
    if not f.exists():
        f = OUT.parent / "mechanism_covariates.csv"
    if not f.exists():
        print("mechanism_covariates.csv missing; run scripts/task_mechanism_covariates.py first", file=sys.stderr)
        return None
    m = pd.read_csv(f)
    if "interaction_gain_bacc" not in m:
        m["interaction_gain_bacc"] = m["bacc_XGB"] - m["additive_bacc"]
    m = m.drop(columns=["auc_XGB", "bacc_XGB", "n_holdout", "n_classes"], errors="ignore")
    # expose the chosen metric's share / gain under the generic names the figure code uses
    for generic, col in MECH_BY_METRIC[METRIC].items():
        if col in m:
            m[generic] = m[col]
    return m


def fig_mechanisms(table, labels, plt):
    from scipy.stats import rankdata, spearmanr

    mech = mechanism_covariates()
    if mech is None:
        return
    merged = table.merge(mech, on=["dataset", "outcome"], how="left")
    big = [lab for lab in labels if merged[f"dauc_{lab}"].notna().sum() >= 10][-2:]
    if not big:
        return
    measures = list(MECH_LABELS)
    ncol = 1 + len(measures)
    fig, axes = plt.subplots(len(big), ncol, figsize=(W2, 1.65 * len(big) + 0.4), squeeze=False,
                             gridspec_kw={"wspace": 0.12, "hspace": 0.5, "width_ratios": [1.6] + [1] * len(measures)})
    rows = []
    for r, lab in enumerate(big):
        s = merged.dropna(subset=[f"dauc_{lab}"])
        v = verdicts(s, lab)
        n_tasks = len(s)
        # column 0: zero-shot AUROC minus each reference scorer, one strip per reference
        ax = axes[r, 0]
        rng = np.random.default_rng(0)
        REFERENCES = REFERENCES_BY_METRIC[METRIC]
        for j, (col, name) in enumerate(REFERENCES):
            d = (s[f"auc_{lab}"] - s[col]).to_numpy()
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(d)), d, s=6, color=TURQ if col != "auc_XGB" else ORANGE,
                       edgecolor=INK, linewidth=0.2, alpha=0.8, zorder=2)
            q1, q2, q3 = np.percentile(d, [25, 50, 75])
            ax.plot([j - 0.3, j + 0.3], [q2, q2], color=INK, lw=1.1, zorder=3)
            ax.plot([j, j], [q1, q3], color=INK, lw=0.6, zorder=3)
            rows.append({"backbone": lab, "measure": f"zs_minus_{col}", "tasks": n_tasks, "median": q2, "q1": q1, "q3": q3})
        ax.axhline(0, **ZERO)
        ax.set_xticks(range(len(REFERENCES)))
        ax.set_xticklabels([name for _, name in REFERENCES], fontsize=5.5)
        ax.set_ylabel(f"{disp(lab)}\nzero-shot {mname()} − scorer", fontsize=6.5)
        ax.set_ylim(-0.75, 0.3)
        ax.set_yticks([-0.6, -0.3, 0.0, 0.3])
        ax.tick_params(labelsize=5.8)
        panel(ax, "ab"[r], x=-0.5)
        ax.set_title(f"{disp(lab)}: {count_label(n_tasks, len(table))}", loc="left",
                     fontsize=7, pad=3, color=INK)
        # columns 1..: paired gap against the mechanism covariates
        for c, m in enumerate(measures, start=1):
            ax = axes[r, c]
            ok = s[m].notna()
            xv, yv = s.loc[ok, m].astype(float), s.loc[ok, f"dauc_{lab}"]
            xplot = np.log10(xv) if m in MECH_LOG else xv
            if m == "n_classes":
                xplot = xplot + rng.uniform(-0.15, 0.15, len(xplot))
            ax.axhline(0, **ZERO)
            regress(ax, np.log10(xv) if m in MECH_LOG else xv, yv)
            ax.scatter(xplot, yv, s=np.clip(s.loc[ok, "n"] / 10, 4, 30), color=[VERDICT[x] for x in v[ok]], edgecolor=INK,
                       linewidth=0.25, alpha=0.9, zorder=3)
            if ok.sum() >= 8:
                rho, p = spearmanr(xv, yv)
                # partial Spearman given the XGB ceiling (rank residuals), so a share metric cannot ride on auc_XGB alone
                rx, ry, rz = rankdata(xv), rankdata(yv), rankdata(s.loc[ok, "auc_XGB"])
                ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
                ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
                prho = float(np.corrcoef(ex, ey)[0, 1])
                rows.append({"backbone": lab, "measure": m, "tasks": int(ok.sum()), "spearman_rho": rho, "p": p,
                             "partial_rho_given_auc_XGB": prho})
                ax.text(0.03, 0.96, f"ρ = {rho:+.2f}, P = {p:.3f}" if p >= 0.001 else f"ρ = {rho:+.2f}, P < 0.001",
                        transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=INK)
            ax.set_ylim(-0.75, 0.3)
            ax.set_yticks([-0.6, -0.3, 0.0, 0.3])
            ax.set_yticklabels([])
            ax.set_xlabel(("log10 " if m in MECH_LOG else "") + MECH_LABELS[m], fontsize=6.2)
            if m == "n_classes":
                ax.set_xticks(sorted(xv.unique()))
            if m == "univariate_share":
                ax.set_xlim(-0.02, 1.3 if METRIC == "auc" else 1.6)
            ax.tick_params(labelsize=5.8)
    fig.subplots_adjust(top=0.94)
    save(fig, "fig_ladder_mechanisms")
    pd.DataFrame(rows).to_csv(OUT / f"mechanism_spearman{msuffix()}.csv", index=False)
    return merged


def mechanism_rank_regression(merged, labels):
    """Rank (Spearman-type) multiple regression of the paired gap on the mechanism covariates, standardised betas."""
    from scipy.stats import rankdata, t as tdist

    if merged is None:
        return
    cols = ["univariate_share", "interaction_gain", "n_development", "n_predictors", "n_classes"]
    rows = []
    for lab in labels:
        d = merged.dropna(subset=[f"dauc_{lab}"] + cols)
        if len(d) < 15:
            continue
        X = d[cols].apply(rankdata)
        X = ((X - X.mean()) / X.std()).to_numpy()
        y = rankdata(d[f"dauc_{lab}"])
        y = (y - y.mean()) / y.std()
        X1 = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        res = y - X1 @ beta
        n, p = X1.shape
        se = np.sqrt(np.diag(res @ res / (n - p) * np.linalg.inv(X1.T @ X1)))
        pv = 2 * (1 - tdist.cdf(np.abs(beta / se), n - p))
        r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
        for c, b, s_, q in zip(cols, beta[1:], se[1:], pv[1:]):
            rows.append({"backbone": lab, "tasks": len(d), "covariate": c, "std_beta": b, "se": s_, "p": q, "model_r2": r2})
    pd.DataFrame(rows).to_csv(OUT / f"mechanism_rank_regression{msuffix()}.csv", index=False)
    return pd.DataFrame(rows)


def fig_summary(table, labels, sizes, colors, plt):
    """One unifying figure (graphical abstract): knows -> stops scaling -> why -> fails at the decision.
    a scaling of both metrics | b paired gap vs XGB per size, both metrics | c per-task AUROC vs XGB at the plateau model |
    d zero-shot minus reference scorers | e gap vs best-marker share | f balanced-accuracy gap by number of classes."""
    from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

    global METRIC
    saved_metric = METRIC
    METRIC = "auc"
    ref = "27B" if "27B" in labels else labels[-1]           # plateau model: everything above it adds nothing
    big = labels[-1]
    mech = mechanism_covariates()
    d = table.merge(mech, on=["dataset", "outcome"], how="left") if mech is not None else table.copy()
    complete = d.dropna(subset=[f"auc_{lab}" for lab in labels])

    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.3), gridspec_kw={"wspace": 0.42, "hspace": 0.55})
    (a, b, c), (p_d, e, f) = axes

    # a: median AUROC and balanced accuracy versus size, XGB references
    xs = [s for _, s in sizes]
    for m, col, name in [("auc", NAVY, "AUROC"), ("bacc", TEAL, "BA")]:
        med = [complete[f"{m}_{lab}"].median() for lab, _ in sizes]
        q1 = [complete[f"{m}_{lab}"].quantile(0.25) for lab, _ in sizes]
        q3 = [complete[f"{m}_{lab}"].quantile(0.75) for lab, _ in sizes]
        a.fill_between(xs, q1, q3, color=col, alpha=0.12, lw=0)
        a.plot(xs, med, "-o", color=col, lw=1.0, ms=3.5, mec=col)
        a.axhline(complete[f"{m}_XGB"].median(), color=col, ls="--", lw=0.9)
        a.text(0.72, complete[f"{m}_XGB"].median() + (0.012 if m == "auc" else -0.012),
               "XGB AUROC" if m == "auc" else "XGB BA", color=col, fontsize=5.6,
               va="bottom" if m == "auc" else "top", ha="left")
        a.text(xs[-1] * 1.45, med[-1], f"zero-shot\n{name}", color=col, fontsize=5.6,
               va="center", ha="left")
    a.axhline(0.5, **ZERO)
    a.set_xscale("log")
    a.set_xticks([1, 10, 100])
    a.set_xticklabels(["1", "10", "100"])
    a.set_xlim(0.6, 4000)
    a.set_ylim(0.45, 1.0)
    a.set_xlabel("Parameters (billions)")
    a.set_ylabel("Median over tasks")
    a.text(0.03, 0.04, f"n = {len(complete)} tasks; median, IQR", transform=a.transAxes, va="bottom", fontsize=5.8)
    panel(a, "a")

    # b: paired balanced-accuracy gap vs XGB per size, median with bootstrap CI over tasks (AUROC is descriptive only)
    rng = np.random.default_rng(0)
    x = np.arange(len(labels))
    meds, lo, hi = [], [], []
    for lab in labels:
        dl = (d[f"bacc_{lab}"] - d["bacc_XGB"]).dropna().to_numpy()
        boots = [np.median(rng.choice(dl, len(dl))) for _ in range(2000)]
        meds.append(np.median(dl)), lo.append(np.percentile(boots, 2.5)), hi.append(np.percentile(boots, 97.5))
    meds, lo, hi = map(np.array, (meds, lo, hi))
    b.errorbar(x, meds, yerr=[meds - lo, hi - meds], fmt="o", color=TEAL, ms=3.5, elinewidth=0.8, capsize=2, capthick=0.8)
    for xi, m_ in zip(x, meds):
        b.text(xi, 0.045, f"{m_:+.3f}", ha="center", va="top", fontsize=5.4, color=INK)
    b.axhline(0, **ZERO)
    b.axhspan(-0.05, 0.0, color=GREY, alpha=0.12, lw=0)
    b.set_xticks(x)
    b.set_xticklabels([short(lab) for lab in labels], fontsize=5.6)
    b.set_ylabel("ΔBA, zero-shot − XGB")
    b.set_ylim(min(-0.4, b.get_ylim()[0]), 0.06)
    panel(b, "b")

    # c: per-task balanced accuracy (the decision each arm actually makes) against XGB at the largest model,
    #    coloured by the paired-CI verdict on balanced accuracy
    s = d.dropna(subset=[f"auc_{ref}"])
    sb = d.dropna(subset=[f"bacc_{big}"])
    lo, hi = sb[f"dbacc_lo_{big}"], sb[f"dbacc_hi_{big}"]
    vb = pd.Series(np.where(lo > 0, "win", np.where(hi < 0, "loss", "parity")), index=sb.index)
    c.plot([0.1, 1.02], [0.1, 1.02], **ZERO)
    c.fill_between([0.1, 1.02], [0.05, 0.97], [0.15, 1.07], color=GREY, alpha=0.08, lw=0)
    c.scatter(sb["bacc_XGB"], sb[f"bacc_{big}"], s=np.clip(sb["n"] / 8, 5, 40), color=[VERDICT[t] for t in vb], edgecolor=INK,
              linewidth=0.3, alpha=0.9, zorder=3)
    cnt = vb.value_counts()
    c.set_xlim(0.1, 1.02)
    c.set_ylim(0.1, 1.02)
    c.set_xticks([0.2, 0.5, 0.8, 1.0])
    c.set_yticks([0.2, 0.5, 0.8, 1.0])
    c.set_xlabel("BA, XGB")
    c.set_ylabel(f"BA, zero-shot {short(big).replace(chr(10), ' ')}")
    from matplotlib.lines import Line2D

    items = [("loss", f"XGB better, n = {cnt.get('loss', 0)}"), ("parity", f"not significant, n = {cnt.get('parity', 0)}"),
             ("win", f"zero-shot better ({cnt.get('win', 0)})")]
    handles = [Line2D([], [], marker="o", ls="", color=VERDICT[k], mec=INK, mew=0.3, ms=4, label=t)
               for k, t in items if not (k == "win" and cnt.get("win", 0) == 0)]
    c.legend(handles=handles, loc="upper left", fontsize=5.6, handletextpad=0.3, borderaxespad=0.2)
    panel(c, "c")

    # d: zero-shot balanced accuracy minus reference scorers at their own label-free operating points (largest model)
    if mech is not None:
        refs = REFERENCES_BY_METRIC["bacc"]
        for j, (col, name) in enumerate(refs):
            col = "bacc_XGB" if col == "auc_XGB" else col
            dd = (sb[f"bacc_{big}"] - sb[col]).dropna().to_numpy()
            p_d.scatter(j + rng.uniform(-0.18, 0.18, len(dd)), dd, s=6, color=TURQ if col != "bacc_XGB" else ORANGE, edgecolor=INK,
                        linewidth=0.2, alpha=0.8, zorder=2)
            q1, q2, q3 = np.percentile(dd, [25, 50, 75])
            p_d.plot([j - 0.3, j + 0.3], [q2, q2], color=INK, lw=1.1, zorder=3)
            p_d.plot([j, j], [q1, q3], color=INK, lw=0.6, zorder=3)
            p_d.text(j, 0.27, f"{q2:+.3f}", ha="center", va="top", fontsize=5.6, color=INK)
        p_d.axhline(0, **ZERO)
        p_d.set_xticks(range(len(refs)))
        p_d.set_xticklabels([n for _, n in refs], fontsize=5.8)
        p_d.set_ylabel(f"BA, zero-shot\n{short(big).replace(chr(10), ' ')} − scorer")
        p_d.set_ylim(-0.75, 0.3)
        p_d.set_yticks([-0.6, -0.3, 0.0, 0.3])
        p_d.set_xlabel("")
        panel(p_d, "d")

        # e: balanced-accuracy gap against best-marker share (largest model), regression + Spearman
        share = "univariate_share"
        ok = d[f"dbacc_{big}"].notna() & d[share].notna()
        xv, yv = d.loc[ok, share], d.loc[ok, f"dbacc_{big}"]
        vv = vb.reindex(d.loc[ok].index).fillna("parity")
        e.axhline(0, **ZERO)
        regress(e, xv, yv)
        e.scatter(xv, yv, s=np.clip(d.loc[ok, "n"] / 10, 4, 30), color=[VERDICT[t] for t in vv], edgecolor=INK, linewidth=0.25,
                  alpha=0.9, zorder=3)
        rho, p = spearmanr(xv, yv)
        e.text(0.97, 0.04, f"ρ = {rho:+.2f}, P {'< 0.001' if p < 0.001 else f'= {p:.3f}'}\n{disp(big)}, n = {int(ok.sum())} tasks",
               transform=e.transAxes, ha="right", va="bottom", fontsize=5.8)
        e.set_xlim(-0.02, 1.3)
        e.set_ylim(-0.75, 0.3)
        e.set_yticks([-0.6, -0.3, 0.0, 0.3])
        e.set_xlabel("Best-marker AUROC ratio S")
        e.set_ylabel("ΔBA, zero-shot − XGB")
        panel(e, "e")

    # f: balanced-accuracy gap, binary versus multiclass outcomes (largest model)
    s2 = d.dropna(subset=[f"dbacc_{big}"])
    groups = [("2 classes", s2[s2.n_classes == 2]), ("3–8 classes", s2[s2.n_classes > 2])]
    for j, (name, g) in enumerate(groups):
        dd = g[f"dbacc_{big}"].to_numpy()
        f.scatter(j + rng.uniform(-0.18, 0.18, len(dd)), dd, s=np.clip(g["n"] / 10, 4, 30), color=[VERDICT[t] for t in verdicts(g, big)],
                  edgecolor=INK, linewidth=0.25, alpha=0.9, zorder=2)
        q1, q2, q3 = np.percentile(dd, [25, 50, 75])
        f.plot([j - 0.3, j + 0.3], [q2, q2], color=INK, lw=1.1, zorder=3)
        f.plot([j, j], [q1, q3], color=INK, lw=0.6, zorder=3)
        f.text(j, 0.27, f"{q2:+.2f}\nn = {len(dd)}", ha="center", va="top", fontsize=5.6)
    pu = mannwhitneyu(groups[0][1][f"dbacc_{big}"], groups[1][1][f"dbacc_{big}"]).pvalue
    f.axhline(0, **ZERO)
    f.set_xticks([0, 1])
    f.set_xticklabels([n for n, _ in groups])
    f.set_xlim(-0.6, 1.6)
    f.set_ylim(-0.75, 0.3)
    f.set_yticks([-0.6, -0.3, 0.0, 0.3])
    f.set_ylabel(f"ΔBA, zero-shot {short(big).replace(chr(10), ' ')} − XGB")
    f.set_xlabel("Outcome classes")
    f.text(0.97, 0.04, f"Mann–Whitney\nP {'< 0.001' if pu < 0.001 else f'= {pu:.3f}'}", transform=f.transAxes, ha="right", va="bottom",
           fontsize=5.8)
    panel(f, "f")

    save(fig, "fig_ladder_summary")
    METRIC = saved_metric


def covariates_by_size(table, labels):
    """Spearman ρ of the paired gap with every task property, per backbone size and metric.

    Balanced accuracy first: it is the metric a single-patient zero-shot user actually gets (argmax at the
    model's own operating point). P values are unadjusted; q is Benjamini–Hochberg within each metric × size family.
    Writes covariates_by_size.csv and draws fig_ladder_covariate_heatmap (a: ΔBA, b: ΔAUROC).
    """
    from scipy.stats import false_discovery_control, spearmanr
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mech = pd.read_csv(OUT.parent / "task_properties.csv") if (OUT.parent / "task_properties.csv").exists() else None
    if mech is None:
        return None
    if "interaction_gain_bacc" not in mech:
        mech["interaction_gain_bacc"] = mech["bacc_XGB"] - mech["additive_bacc"]
    d = table.merge(mech.drop(columns=["n_classes", "auc_XGB", "bacc_XGB", "n_holdout"], errors="ignore"),
                    on=["dataset", "outcome"], how="left")
    cov = task_covariates()
    if cov is not None:
        d = d.merge(cov, on=["dataset", "outcome"], how="left")
    classic = [c for c in COVARIATE_LABELS if c in d]
    plan = {"bacc": ("dbacc_", ["univariate_share", "interaction_gain_bacc", "n_development", "n_predictors", "n_classes"]),
            "auc": ("dauc_", ["univariate_share", "interaction_gain", "n_development", "n_predictors", "n_classes"])}
    rows = []
    for metric, (pref, mv) in plan.items():
        for lab in labels:
            y = d[pref + lab]
            fam = []
            for v in mv + classic:
                ok = y.notna() & d[v].notna()
                if ok.sum() < 10:
                    continue
                rho, p = spearmanr(y[ok], d.loc[ok, v])
                fam.append(dict(metric=METRICS[metric]["delta"], backbone=lab, variable=v, tasks=int(ok.sum()), rho=rho, p=p))
            q = false_discovery_control([r["p"] for r in fam]) if fam else []
            for r, qq in zip(fam, q):
                r["q_bh"] = qq
            rows += fam
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "covariates_by_size.csv", index=False)

    def row_label(v):
        base = v.replace("_bacc", "")
        return MECH_LABELS.get(base, COVARIATE_LABELS.get(base, base))

    fig, ax = plt.subplots(figsize=(W1 * 1.25, 2.9))
    axes = [ax]
    norm = TwoSlopeNorm(vmin=-0.75, vcenter=0, vmax=0.75)
    cmap = plt.cm.RdBu_r
    for metric in ["bacc"]:                      # balanced accuracy is the analysis metric; AUROC is descriptive only
        sub = t[t.metric == METRICS[metric]["delta"]]
        order = plan[metric][1] + classic
        piv = sub.pivot(index="variable", columns="backbone", values="rho").reindex(index=order, columns=labels)
        pp = sub.pivot(index="variable", columns="backbone", values="p").reindex(index=order, columns=labels)
        qq = sub.pivot(index="variable", columns="backbone", values="q_bh").reindex(index=order, columns=labels)
        ax.imshow(piv.values, cmap=cmap, norm=norm, aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                r, p, q = piv.values[i, j], pp.values[i, j], qq.values[i, j]
                if np.isnan(r) or not (p < 0.05 or q < 0.10):
                    continue
                mark = "†" if q < 0.10 else "*"
                ax.text(j, i, f"{r:+.2f}{mark}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(r) > 0.45 else INK)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([short(l) for l in labels], fontsize=6)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([row_label(v) for v in order], fontsize=6)
        ax.axhline(len(plan[metric][1]) - 0.5, color=INK, lw=0.6)
        ax.set_title("ΔBA (zero-shot − XGB)", fontsize=7, color=INK, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
    cb.set_label("Spearman ρ", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    saved = METRIC
    globals()["METRIC"] = "auc"           # metric-independent figure: no suffix
    save(fig, "fig_ladder_covariate_heatmap")
    globals()["METRIC"] = saved
    return t


KEY_COVARIATES = [("univariate_share", "Best-marker share", False),
                  ("n_classes", "Outcome classes", True),
                  ("complexity_F1", "Inverted F1, class overlap", False),
                  ("N1", "N1, border fraction", False)]


def fig_key_covariates(table, labels, bysize, plt):
    """Supplementary figure: the four task properties of the heat map that the text discusses, ΔBA against each at
    every size. Rows are backbones, columns are properties (panel letters a-d), so x axes are shared down a column and
    drawn once at the bottom. ρ and its mark come from covariates_by_size.csv so the panels agree with Fig. 3."""
    from matplotlib.lines import Line2D

    mech = pd.read_csv(OUT.parent / "task_properties.csv") if (OUT.parent / "task_properties.csv").exists() else None
    if mech is None or bysize is None:
        return
    d = table.merge(mech.drop(columns=["n_classes", "auc_XGB", "bacc_XGB", "n_holdout"], errors="ignore"),
                    on=["dataset", "outcome"], how="left")
    cov = task_covariates()
    if cov is not None:
        d = d.merge(cov, on=["dataset", "outcome"], how="left")
    keys = [k for k in KEY_COVARIATES if k[0] in d]
    sub = bysize[bysize.metric == "ΔBA"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(labels), len(keys), figsize=(W1 * 1.7, 1.05 * len(labels) + 0.4), squeeze=False,
                             sharex="col", gridspec_kw={"wspace": 0.12, "hspace": 0.18})
    for c, (v, label, jitter) in enumerate(keys):
        for r, lab in enumerate(labels):
            ax = axes[r, c]
            ok = d[f"dbacc_{lab}"].notna() & d[v].notna()
            xv, yv = d.loc[ok, v].astype(float), d.loc[ok, f"dbacc_{lab}"]
            lo, hi = d.loc[ok, f"dbacc_lo_{lab}"], d.loc[ok, f"dbacc_hi_{lab}"]
            verd = np.where(lo > 0, "win", np.where(hi < 0, "loss", "parity"))
            xplot = xv + rng.uniform(-0.15, 0.15, len(xv)) if jitter else xv
            ax.axhline(0, **ZERO)
            regress(ax, xv, yv)
            ax.scatter(xplot, yv, s=np.clip(d.loc[ok, "n"] / 10, 4, 30), color=[VERDICT[t] for t in verd], edgecolor=INK,
                       linewidth=0.25, alpha=0.9, zorder=3)
            row = sub[(sub.variable == v) & (sub.backbone == lab)]
            if len(row):
                rho, p, q = row.rho.iloc[0], row.p.iloc[0], row.q_bh.iloc[0]
                mark = "†" if q < 0.10 else ("*" if p < 0.05 else "")
                ax.text(0.97, 0.93, f"ρ = {rho:+.2f}{mark}", transform=ax.transAxes, ha="right", va="top",
                        fontsize=5.8, color=INK, fontweight="bold" if mark else "normal")
            ax.set_ylim(-0.75, 0.3)
            ax.set_yticks([-0.6, -0.3, 0.0, 0.3])
            if v == "n_classes":
                ax.set_xticks([2, 4, 6, 8])
            ax.tick_params(labelsize=5.8)
            if c == 0:
                ax.set_ylabel(disp(lab).replace(" MoE", "\nMoE"), fontsize=7, color=INK)
            else:
                ax.set_yticklabels([])
            if r == len(labels) - 1:
                ax.set_xlabel(label, fontsize=6.5)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if r == 0:
                panel(ax, "abcdef"[c], x=-0.02 if c else -0.45, y=1.03)
    fig.supylabel("ΔBA, zero-shot − XGB", fontsize=7, x=0.01)
    items = [("loss", "XGB better"), ("parity", "Not significant")]
    handles = [Line2D([], [], marker="o", ls="", color=VERDICT[k], mec=INK, mew=0.3, ms=4, label=t) for k, t in items]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0), frameon=False,
               handletextpad=0.3, columnspacing=1.5, fontsize=6.5)
    fig.subplots_adjust(bottom=0.09)
    saved = METRIC
    globals()["METRIC"] = "auc"           # BA-only, metric-independent file name
    save(fig, "fig_ladder_key_covariates")
    globals()["METRIC"] = saved


def main(argv):
    style()
    import matplotlib.pyplot as plt

    metrics_to_run = ["auc", "bacc"]
    if "--metric" in argv:
        i = argv.index("--metric")
        metrics_to_run = ["auc", "bacc"] if argv[i + 1] == "both" else [argv[i + 1]]
        argv = argv[:i] + argv[i + 2:]
    dirs = [ROOT / a for a in argv] if argv else sorted(ROOT.glob("outputs_qwen3.5_*"))
    dirs = [d for d in dirs if any(d.glob("*/*/predictions/ZS_raw_holdout.csv"))]
    dirs.sort(key=lambda d: (SIZE_B.get(d.name.replace("outputs_qwen3.5_", ""), 1e9), d.name))
    OUT.mkdir(parents=True, exist_ok=True)
    table = collect(dirs)
    table.to_csv(OUT / "ladder_table.csv", index=False)
    labels = [label_of(d) for d in dirs if f"auc_{label_of(d)}" in table and table[f"auc_{label_of(d)}"].notna().any()]
    dirs = [d for d in dirs if label_of(d) in labels]
    ramp = SIZE_RAMP if len(labels) <= len(SIZE_RAMP) else [plt.cm.viridis(x) for x in np.linspace(0.9, 0.1, len(labels))]
    colors = {lab: ramp[len(ramp) - len(labels) + i] for i, lab in enumerate(labels)}
    sizes = [(lab, SIZE_B.get(d.name.replace("outputs_qwen3.5_", ""))) for lab, d in zip(labels, dirs)]
    sizes = [(lab, s) for lab, s in sizes if s]

    global METRIC
    fig_summary(table, labels, sizes, colors, plt)
    bysize = covariates_by_size(table, labels)
    fig_key_covariates(table, labels, bysize, plt)
    if bysize is not None:
        pd.set_option("display.width", 220)
        print("\nSpearman ρ of the paired gap with task properties, by size (ΔBA first; * P<0.05, † BH q<0.10):")
        for mt, g in bysize.groupby("metric", sort=False):
            piv = g.pivot(index="variable", columns="backbone", values="rho").reindex(columns=labels)
            flag = g.assign(f=np.where(g.q_bh < 0.10, "†", np.where(g.p < 0.05, "*", " "))).pivot(index="variable", columns="backbone", values="f").reindex(columns=labels)
            print(f"\n-- {mt}")
            print((piv.round(2).astype(str) + flag).to_string())
    for metric in metrics_to_run:
        METRIC = metric
        view = as_metric(table, metric)
        stats = across_task_stats(view, labels)
        stats.to_csv(OUT / f"stats_by_size{msuffix()}.csv", index=False)
        if len(stats):
            fig_scaling(view, labels, sizes, colors, stats, plt)
        fig_per_task(view, labels, colors, plt)
        fig_paired_ci(view, labels, plt)
        fig_vs_xgb(view, labels, colors, plt)
        merged, measures = fig_covariates(view, labels, colors, plt)
        fig_scaling_slope(merged, measures, labels, sizes, plt)
        mech = fig_mechanisms(view, labels, plt)
        reg = mechanism_rank_regression(mech, [lab for lab in labels if view[f"dauc_{lab}"].notna().sum() >= 15])
        steps = scaling_steps(view, labels)
        steps.to_csv(OUT / f"scaling_steps{msuffix()}.csv", index=False)

        pd.set_option("display.width", 200)
        print(f"\n===== {mname()} =====")
        print(stats.round(3).to_string(index=False))
        print("\nStep gains on shared tasks (larger − smaller, paired per task):")
        print(steps.round(3).to_string(index=False))
        if mech is not None:
            print("\nMechanism covariates vs paired gap (Spearman; partial given the XGB ceiling):")
            print(pd.read_csv(OUT / f"mechanism_spearman{msuffix()}.csv").dropna(subset=["spearman_rho"]).round(3).to_string(index=False))
        if reg is not None and len(reg):
            print("\nRank regression of the gap on all five (standardised betas):")
            print(reg.round(3).to_string(index=False))
    print(f"\nXGB: {len(table)} tasks, median AUROC {table.auc_XGB.median():.3f}, median balanced accuracy {table.bacc_XGB.median():.3f}")
    print(f"wrote {OUT}/fig_ladder_*[_bacc].{{svg,pdf,png}} and the csv tables")


if __name__ == "__main__":
    main(sys.argv[1:])
