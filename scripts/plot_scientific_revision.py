#!/usr/bin/env python3
"""Publication figures from executed revision tables; saves PDF/SVG/600-dpi PNG.

Every figure is drawn at its final print size (170 mm wide, within the 183 mm
double-column limit and the 170 mm height limit), so all text is 6 to 7 pt after
typesetting. Panel letters are placed from the rendered layout. Only presentation
is handled here; every statistic comes unchanged from the executed tables.
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from jevtest.benchmark import OUTCOME_LABEL
from run_scientific_revision import BACKBONES, OUT, summary_interval

FIG = OUT / "figures"
N, T, O, G = "#0B2545", "#1B7F79", "#D88035", "#8C8C8C"
INK, GRID = "#1A1A1A", "#E3E3E3"
MM = 1 / 25.4
W = 170 * MM
FS, FS_SMALL, FS_CELL = 9.0, 8.0, 7.5
LABEL = dict(zip(BACKBONES, ["0.8B", "2B", "4B", "9B", "27B", "397B\nA17B"]))
FLAT = {b: LABEL[b].replace("\n", "-") for b in BACKBONES}
FLAT["397b_a17b_fp8"] = "397B-A17B FP8"
MINUS = "−"

RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 5, "axes.labelpad": 2.5,
    "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "legend.fontsize": 8.5,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": .4,
    "legend.columnspacing": 1.2, "axes.linewidth": 1.0, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0, "xtick.major.size": 3.2,
    "ytick.major.size": 3.2, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": .7, "lines.linewidth": 1.9, "lines.markersize": 5.5,
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.unicode_minus": True,
    "figure.facecolor": "white", "savefig.facecolor": "white",
}


def read(name):
    try:
        return pd.read_csv(OUT / f"{name}.csv")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def new(height_mm, *args, **kw):
    fig, axes = plt.subplots(*args, figsize=(W, height_mm * MM), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=3 / 72, h_pad=3 / 72, wspace=.05, hspace=.05)
    return fig, axes


def tidy(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, zorder=0)
    ax.tick_params(length=2.5)


def zero(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0, color=G, lw=1.12, zorder=1)


def signed(v, nd=2):
    s = f"{abs(v):.{nd}f}"
    return (MINUS if v < 0 and float(s) != 0 else "") + s


def finish(fig, name, panels=None):
    """Place bold panel letters from the rendered layout, then save all formats."""
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>" and a.get_visible()] if panels is None else panels
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for letter, ax in zip("abcdefghijkl", panels):
        bb = ax.get_tightbbox(renderer).transformed(inv)
        room = (ax.get_position().x0 - bb.x0) * fig.get_figwidth() * 72
        if room < 10 and ax.get_title("left"):  # no y-label to the left: make room for the letter
            ax._left_title.set_transform(ax._left_title.get_transform() + ScaledTranslation(10 / 72, 0, fig.dpi_scale_trans))
        fig.text(max(bb.x0, 0.004), min(bb.y1, .996), letter, fontsize=9, fontweight="bold", va="top", ha="left")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600)
    plt.close(fig)


def heat(ax, values, rows, cols, cmap, norm=None, vmin=None, vmax=None, fmt=None, text_switch=None, fs=FS_CELL):
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.28)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if fmt:
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = values[i][j]
                if np.isfinite(v):
                    ax.text(j, i, fmt(i, j, v), ha="center", va="center", fontsize=fs,
                            color="white" if text_switch(v) else INK)
    return im


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update(RC)
    original = read("original_metrics"); props = read("task_properties"); refs = read("reference_comparisons")
    assoc = read("exploratory_associations"); cal = read("calibration_summary")
    xt6 = [LABEL[b] for b in BACKBONES]

    # ---------------- Fig. 1 ----------------
    fig, axes = new(112, 2, 3)
    ax = axes[0, 0]
    ax.grid(axis="y", zorder=0)
    for metric, color, name, pos in (("auroc", N, "L0 AUROC", (2.1, .715, "top")),
                                     ("balanced_accuracy", T, "L0 BA", (.9, .56, "bottom"))):
        vals = [original.loc[original.backbone == b, metric].median() for b in BACKBONES]
        ax.plot(range(6), vals, "o-", color=color, ms=5.2, zorder=3)
        ax.text(pos[0], pos[1], name, color=color, fontsize=FS_SMALL, va=pos[2], ha="left")
    for val, color, name, y in ((original.xgb_auroc.median(), N, "XGBoost AUROC", .0), (original.xgb_ba.median(), T, "XGBoost BA", .0)):
        ax.axhline(val, ls=(0, (4, 2)), color=color, lw=1.44, zorder=2)
        ax.text(-.35, val + .008, name, color=color, fontsize=FS_SMALL, va="bottom", ha="left")
    ax.set(xticks=range(6), xticklabels=xt6, ylabel="Median over 45 tasks", xlim=(-.45, 5.45), ylim=(.46, .95),
           title="Frozen L0 versus XGBoost")
    ax = axes[0, 1]
    primary = read("primary_summary"); tidy(ax)
    zero(ax)
    for metric, color, name, offset in (("auroc", N, "AUROC", -.12), ("balanced_accuracy", T, "Balanced accuracy", .12)):
        vals = np.array([primary[(primary.backbone == b) & (primary.metric == metric)][["median_delta", "ci_low", "ci_high"]].iloc[0].to_numpy() for b in BACKBONES])
        ax.errorbar(np.arange(6) + offset, vals[:, 0], yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], fmt="o", ms=5.2,
                    color=color, capsize=2.4, elinewidth=1.6, label=name, zorder=3)
    ax.set(xticks=range(6), xticklabels=xt6, ylabel="Median paired L0 " + MINUS + " XGBoost", xlim=(-.5, 5.5), ylim=(-.41, .02),
           title="Ranking and decision deficits")
    ax.legend(loc="lower right", handlelength=.8, borderaxespad=.2)
    ax = axes[0, 2]; sub = original[original.backbone == "27b"]; tidy(ax, "both")
    colors = np.where(sub.auroc_hi < 0, O, np.where(sub.auroc_lo > 0, T, G))
    ax.plot([.1, 1.02], [.1, 1.02], ls=(0, (4, 2)), c=G, lw=1.28, zorder=1)
    ax.scatter(sub.xgb_auroc, sub.auroc, c=colors, s=24, alpha=.9, linewidths=.3, edgecolors="white", zorder=3)
    nl, ni, nh = int((sub.auroc_hi < 0).sum()), int(((sub.auroc_lo <= 0) & (sub.auroc_hi >= 0)).sum()), int((sub.auroc_lo > 0).sum())
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, ms=5.0, label=l) for c, l in ((O, f"Lower (n={nl})"), (G, f"Zero in CI (n={ni})"), (T, f"Higher (n={nh})"))],
              loc="upper left", handletextpad=.1, borderaxespad=.1, labelspacing=.25)
    ax.set(xlabel="XGBoost AUROC", ylabel="27B L0 AUROC", xlim=(.15, 1.03), ylim=(.15, 1.03), title="Paired discrimination")
    ax.set_aspect("equal", adjustable="box")
    ax = axes[1, 0]; tidy(ax); zero(ax)
    rng = np.random.default_rng(1)
    for i, ref in enumerate(("best_marker", "direction_only", "additive")):
        values = refs[(refs.backbone == "27b") & (refs.reference == ref)].delta_auroc
        ax.scatter(np.full(len(values), i) + rng.uniform(-.17, .17, len(values)), values, s=14, color=N, alpha=.45, linewidths=0, zorder=2)
        m, l, h = summary_interval(values)
        ax.errorbar(i, m, yerr=[[m - l], [h - m]], fmt="D", ms=5.6, color=O, mec="white", mew=.5, capsize=2.5, elinewidth=2.24, zorder=4)
    ax.set(xticks=range(3), xticklabels=["CV best\nmarker", "Direction\nonly", "Additive\nlogistic"], xlim=(-.6, 2.6),
           ylabel="27B " + MINUS + " reference AUROC", title="Selected references")
    ax = axes[1, 1]; tidy(ax, "both"); zero(ax)
    sub = original[original.backbone == "27b"].merge(props[["dataset", "outcome", "univariate_share"]], on=["dataset", "outcome"])
    ax.scatter(sub.univariate_share, sub.delta_auroc, c=N, s=24, linewidths=.3, edgecolors="white", zorder=3)
    ax.set(xlabel="Marker share S", ylabel="27B " + MINUS + " XGBoost AUROC", title="Marker share can exceed one")
    ax = axes[1, 2]; tidy(ax); zero(ax)
    rng = np.random.default_rng(2)
    for i, backbone in enumerate(("27b", "397b_a17b_fp8")):
        for j, kind in enumerate(("binary", "multiclass")):
            sub = original[(original.backbone == backbone) & ((original.n_classes == 2) if kind == "binary" else (original.n_classes > 2))]
            x = i * 3 + j; col = N if kind == "binary" else T
            ax.scatter(np.full(len(sub), x) + rng.uniform(-.2, .2, len(sub)), sub.delta_adjusted_ba, color=col, s=15, alpha=.6, linewidths=0, zorder=2)
            ax.hlines(sub.delta_adjusted_ba.median(), x - .38, x + .38, color=O, lw=2.6, zorder=4)
    ax.set(xticks=[0, 1, 3, 4], xticklabels=["27B\nbinary", "27B\nmulti-\nclass", "397B\nbinary", "397B\nmulti-\nclass"],
           ylabel="Chance-adjusted BA gap", title="Class-count adjustment", xlim=(-.6, 4.6))
    finish(fig, "fig1_summary")

    # ---------------- Fig. 2 ----------------
    fig, ax = new(168)
    pv = original.pivot(index="outcome", columns="backbone", values="delta_ba").reindex(columns=BACKBONES)
    pv = pv.loc[pv["27b"].sort_values().index]
    im = heat(ax, pv.values, [OUTCOME_LABEL.get(o, o) for o in pv.index], xt6, "RdBu_r", norm=TwoSlopeNorm(vmin=-.7, vcenter=0, vmax=.7),
              fmt=lambda i, j, v: signed(v), text_switch=lambda v: abs(v) > .38)
    ax.tick_params(axis="x", labelsize=FS_SMALL)
    ax.set_title("Default argmax balanced accuracy minus tuned XGBoost, by task", pad=6)
    cb = fig.colorbar(im, ax=ax, location="bottom", shrink=.55, aspect=40, pad=.015)
    cb.set_label("Paired task BA difference (L0 " + MINUS + " XGBoost)")
    cb.outline.set_visible(False); cb.ax.tick_params(length=2)
    finish(fig, "fig2_per_task")

    # ---------------- Supplementary Fig. 1 ----------------
    variables = [v for v in ("univariate_share", "interaction_gain", "n_development", "n_predictors", "n_classes", "complexity_F1", "N1", "C2", "T2", "missingness", "explanation_rate", "prompt_words", "unit_coverage", "category_gloss_coverage") if v in set(assoc.variable)]
    names = {"univariate_share": "Marker share", "interaction_gain": "Interaction gain", "n_development": "Development n", "n_predictors": "Predictors",
             "n_classes": "Classes", "complexity_F1": "Inverted F1", "N1": "N1", "C2": "C2", "T2": "T2", "missingness": "Missingness",
             "explanation_rate": "Explanation rate", "prompt_words": "Prompt words", "unit_coverage": "Unit coverage", "category_gloss_coverage": "Category gloss coverage"}
    backs = list(BACKBONES)
    fig, axes = new(98, 1, 3)
    for ax, target, title in zip(axes, ("balanced_accuracy", "xgb_ba", "delta_ba"), ("L0 balanced accuracy", "XGBoost balanced accuracy", "L0 " + MINUS + " XGBoost")):
        s = assoc[assoc.target == target]
        pvt = s.pivot(index="variable", columns="backbone", values="rho").reindex(index=variables, columns=backs)
        qq = s.pivot(index="variable", columns="backbone", values="q_bh_global").reindex(index=variables, columns=backs)
        im = heat(ax, pvt.values, [names[v] for v in variables], xt6, "RdBu_r", vmin=-1, vmax=1,
                  fmt=lambda i, j, v, qq=qq: signed(v).replace('0.', '.') + ("*" if qq.iloc[i, j] < .05 else ""), text_switch=lambda v: abs(v) > .6, fs=7.2)
        ax.tick_params(axis="x", labelsize=FS_SMALL)
        ax.set_title(title)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=.4, aspect=40, pad=.02)
    cb.set_label("Spearman ρ (* global Benjamini–Hochberg q < 0.05)")
    cb.outline.set_visible(False); cb.ax.tick_params(length=2)
    finish(fig, "suppfig1_associations", list(axes))

    # ---------------- Supplementary Fig. 2 ----------------
    fig, axes = new(66, 1, 2, sharey=True)
    for ax, kind in zip(axes, ("binary", "multiclass")):
        tidy(ax); zero(ax)
        for b, color in (("27b", N), ("397b_a17b_fp8", T)):
            sub = cal[(cal.backbone == b) & (cal.kind == kind) & (cal.method == "prevalence") & (cal.metric == "delta_ba")]
            sub = sub.set_index("budget").reindex(["10", "20", "50", "100", "full"])
            ax.plot(range(5), sub["median"], "o-", c=color, label=FLAT[b], ms=5.0)
        ax.set(xticks=range(5), xticklabels=["10", "20", "50", "100", "Full dev"], title=f"{kind.title()} tasks", xlabel="Development labels used", xlim=(-.3, 4.3))
    axes[0].set_ylabel("Median BA change from original L0")
    axes[0].legend(loc="lower left", borderaxespad=.2)
    finish(fig, "suppfig2_prevalence_budget", list(axes))

    # ---------------- Fig. 3 and Supplementary Fig. 3 ----------------
    rawcal = read("calibration_metrics")
    METHOD_COLOR = dict(prevalence=G, offset=N, temperature=T, temperature_offset=O, operating_point="#7A4EAB")
    METHOD_NAME = dict(prevalence="Prevalence", offset="Offset", temperature="Temperature", temperature_offset="Temperature + offset", operating_point="Operating point")
    BUDGETS = ["10", "20", "50", "100", "pool", "full"]
    learned = [b for b in BACKBONES if b != "0.8b" and len(rawcal) and ((rawcal.backbone == b) & (rawcal.method == "offset")).any()]
    focus = "27b" if "27b" in learned else (learned[0] if learned else None)
    if focus:
        sub = rawcal[rawcal.backbone == focus].assign(kind=lambda d: np.where(d.n_classes == 2, "binary", "multiclass"), budget=lambda d: d.budget.astype(str))
        scored = set(map(tuple, sub.loc[sub.method == "offset", ["dataset", "outcome"]].drop_duplicates().to_numpy()))
        sub = sub[[t in scored for t in zip(sub.dataset, sub.outcome)]]
        sub.loc[(sub.budget == "full") & (sub.method != "prevalence"), "budget"] = "pool"
        per = sub.groupby(["kind", "method", "budget", "dataset", "outcome"])[["delta_ba", "remaining_gap", "recovered_gap_ratio", "balanced_accuracy"]].mean().reset_index()
        fig = plt.figure(figsize=(W, 146 * MM), layout="constrained")
        fig.get_layout_engine().set(w_pad=3 / 72, h_pad=3 / 72, wspace=.05, hspace=.05)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, .17, 1])
        axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])], [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]])
        legend_ax = fig.add_subplot(gs[1, :]); legend_ax.axis("off")
        for ax, kind in zip(axes[0], ("binary", "multiclass")):
            tidy(ax); zero(ax)
            for method, color in METHOD_COLOR.items():
                xs, ms, ls, hs = [], [], [], []
                for i, b in enumerate(BUDGETS):
                    v = per[(per.kind == kind) & (per.method == method) & (per.budget == b)].delta_ba
                    if len(v):
                        m, l, h = summary_interval(v)
                        xs.append(i + (list(METHOD_COLOR).index(method) - 2) * .09); ms.append(m); ls.append(m - l); hs.append(h - m)
                if xs:
                    ax.errorbar(xs, ms, yerr=[ls, hs], fmt="o-", ms=4.5, lw=1.6, color=color, capsize=2.2, elinewidth=1.28, zorder=3)
            ax.set(xticks=range(len(BUDGETS)), xticklabels=["10", "20", "50", "100", "Pool", "Full dev"], xlabel="Development labels used",
                   title=f"{FLAT[focus]}, {kind} tasks (n={per[per.kind == kind].dataset.nunique()})")
        axes[0][0].set_ylabel("Median BA change\nfrom original L0")
        whole = sub[sub.vs_xgb_ba_lo.notna()] if "vs_xgb_ba_lo" in sub else sub[sub.budget.isin(["pool", "full"])]
        whole = whole.groupby(["kind", "method", "dataset", "outcome"])[["remaining_gap"]].mean().reset_index()
        rng = np.random.default_rng(0)
        for ax, kind in zip(axes[1], ("binary", "multiclass")):
            tidy(ax); zero(ax)
            g = whole[whole.kind == kind]
            base = original[(original.backbone == focus) & ((original.n_classes == 2) if kind == "binary" else (original.n_classes > 2))]
            base = base[[t in scored for t in zip(base.dataset, base.outcome)]]
            methods = [m for m in METHOD_COLOR if (g.method == m).any()]
            data = [-base.delta_ba.to_numpy()] + [g[g.method == m].remaining_gap.to_numpy() for m in methods]
            ax.boxplot(data, tick_labels=["Original"] + [METHOD_NAME[m] for m in methods],
                       showfliers=False, widths=.55, medianprops=dict(color=INK, lw=1.92), boxprops=dict(lw=1.12), whiskerprops=dict(lw=1.12), capprops=dict(lw=1.12))
            for i, d in enumerate(data):
                ax.scatter(np.full(len(d), i + 1) + rng.uniform(-.16, .16, len(d)), d, s=14, color=INK, alpha=.5, linewidths=0, zorder=3)
            ax.tick_params(axis="x", rotation=30)
            for t in ax.get_xticklabels():
                t.set_ha("right"); t.set_rotation_mode("anchor")
            ax.set(title=f"{kind.title()}: gap after full-pool fitting")
        axes[1][0].set_ylabel("Remaining XGBoost " + MINUS + " L0 BA")
        legend_ax.legend(handles=[Line2D([], [], marker="o", color=METHOD_COLOR[m], ms=6, lw=2.2, label=METHOD_NAME[m]) for m in METHOD_COLOR],
                         loc="center", ncol=5, title="Adjustment method (panels a and b)", title_fontsize=8.5, columnspacing=1.4)
        finish(fig, "fig3_decision_repair", list(axes.flat))

        status = read("experiment_status")
        fig, axes = new(66, 1, len(learned) + 1, squeeze=False)
        axes = axes[0]
        for ax, b in zip(axes, learned):
            tidy(ax)
            s2 = rawcal[rawcal.backbone == b].assign(budget=lambda d: d.budget.astype(str))
            for method, color in METHOD_COLOR.items():
                v = s2[s2.method == method].groupby(["budget", "dataset", "outcome"]).balanced_accuracy.mean().groupby("budget").median().reindex(BUDGETS)
                ax.plot(range(len(BUDGETS)), v.values, "o-", ms=4.5, color=color)
            ax.set(xticks=range(len(BUDGETS)), xticklabels=["10", "20", "50", "100", "Pool", "Full"], title=f"{FLAT[b]}: tasks with scores", xlabel="Development labels used")
        axes[0].set_ylabel("Median hold-out BA")
        ax = axes[-1]; tidy(ax)
        if len(status) and "budget" in status:
            fails = status[(status.experiment == "calibration") & (status.status == "failed") & (status.backbone != "0.8b")]
            counts = fails.groupby(["backbone", "budget"]).size().unstack(fill_value=0) if len(fails) else pd.DataFrame()
            if len(counts):
                counts.T.rename(index=lambda v: str(int(float(v))) if str(v).replace(".","",1).isdigit() else v).plot.bar(ax=ax, rot=0, width=.8, color=[N, O, T, G][:counts.shape[0]], legend=False)
                ax.legend([FLAT.get(c, c) for c in counts.index], loc="upper right", borderaxespad=.2)
        ax.set(title="Learned fits lacking a class", ylabel="Failed fits", xlabel="Development labels used")
        fig.legend(handles=[Line2D([], [], marker="o", color=METHOD_COLOR[m], ms=5.0, lw=1.6, label=METHOD_NAME[m]) for m in METHOD_COLOR],
                   loc="outside lower center", ncol=5)
        finish(fig, "suppfig3_learned_calibration", list(axes))

    # ---------------- Supplementary Figs. 4 and 5 ----------------
    def condition_panels(table, key, order, stem, names):
        fig, axes = new(78, 1, 3)
        bk = [b for b in BACKBONES if b != "0.8b" and (table.backbone == b).any()]
        cols = [N, O]
        for ax, metric, ylabel in zip(axes, ("delta_auroc", "delta_ba", "constant_class"),
                                      ("Paired AUROC change\nvs regenerated original", "Paired BA change\nvs regenerated original", "Tasks with one\npredicted class")):
            tidy(ax)
            if metric != "constant_class":
                zero(ax)
            for j, b in enumerate(bk):
                t = table[table.backbone == b]
                for i, c in enumerate(order):
                    x = i + (j - (len(bk) - 1) / 2) * .36
                    if metric == "constant_class":
                        ax.bar(x, t[t[key] == c].constant_class.sum(), width=.34, color=cols[j % 2], zorder=3)
                        continue
                    if metric not in t:
                        continue
                    v = t[t[key] == c][metric].dropna()
                    if not len(v):
                        continue
                    ax.scatter(np.full(len(v), x) + np.random.default_rng(i).uniform(-.08, .08, len(v)), v, s=14, alpha=.5, color=cols[j % 2], linewidths=0, zorder=2)
                    if len(v) > 2:
                        m, l, h = summary_interval(v)
                        ax.errorbar(x, m, yerr=[[m - l], [h - m]], fmt="D", ms=5.0, color=INK, mec="white", mew=.4, capsize=2, elinewidth=1.76, zorder=4)
            ax.set(xticks=range(len(order)), xticklabels=[names[c] for c in order], ylabel=ylabel, xlim=(-.6, len(order) - .4))
            ax.tick_params(axis="x", labelsize=FS_SMALL, rotation=35)
            for t in ax.get_xticklabels():
                t.set_ha("right"); t.set_rotation_mode("anchor")
        handles = [Line2D([], [], marker="o", ls="", color=cols[j % 2], ms=5.0, label=f"{FLAT[b]} ({table[table.backbone == b].dataset.nunique()} tasks)") for j, b in enumerate(bk)]
        handles.append(Line2D([], [], marker="D", ls="", color=INK, ms=5.0, label="Median and 95% task-bootstrap interval"))
        fig.legend(handles=handles, loc="outside lower center", ncol=3)
        finish(fig, stem, list(axes))

    sem = read("semantic_metrics")
    if len(sem):
        condition_panels(sem, "condition", ["no_setting", "no_units", "anonymous", "no_category_gloss", "concise"], "suppfig4_semantic",
                         {"no_setting": "No setting", "no_units": "No units", "anonymous": "Anonymous", "no_category_gloss": "No category gloss", "concise": "Concise"})
    prot = read("readout_metrics")
    if len(prot):
        condition_panels(prot, "stage", ["raw", "rotation_only", "prior_only", "full"], "suppfig5_readout",
                         {"raw": "Raw", "rotation_only": "Rotation only", "prior_only": "Prior only", "full": "Full"})

    # ---------------- Supplementary Fig. 6 ----------------
    cm = read("multiclass_confusion")
    after = [m for m in cm.method.unique() if m.endswith("_development_offset") and ((cm.backbone == "27b") & (cm.method == m)).any()] if len(cm) else []
    rows = [("original", "27b")] + [(after[0], "27b")] if after else [("original", "27b")]
    tasks = cm[(cm.backbone == "27b") & (cm.method == "original")].outcome.unique()
    fig, axes = new(168, 2 * len(rows), 4)
    for r, (method, b) in enumerate(rows):
        for ax, task in zip(axes[2 * r:2 * r + 2].flat, tasks):
            g = cm[(cm.backbone == b) & (cm.method == method) & (cm.outcome == task)]
            pv = g.pivot(index="observed_class", columns="predicted_class", values="count"); norm = pv.div(pv.sum(1), axis=0)
            k = len(pv)
            im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
            for i in range(k):
                for j in range(k):
                    v = norm.iloc[i, j]
                    if np.isfinite(v) and v >= .005:
                        ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.8 if k > 5 else FS_CELL, color="white" if v > .55 else INK)
            ax.set_xticks(range(k), pv.columns); ax.set_yticks(range(k), pv.index)
            ax.set_xticks(np.arange(-.5, k), minor=True); ax.set_yticks(np.arange(-.5, k), minor=True)
            ax.grid(which="minor", color="white", lw=0.96); ax.tick_params(which="both", length=0, labelsize=FS_SMALL)
            ax.set_title(f"{OUTCOME_LABEL.get(task, task)}\n{'Original L0' if method == 'original' else 'Pool-fitted offsets'}", fontsize=FS_SMALL, pad=3)
            for s in ax.spines.values():
                s.set_linewidth(.5)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Predicted class")
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("Observed class")
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=.35, aspect=40, pad=.01)
    cb.set_label("Percentage of observed class (row-normalised)"); cb.set_ticks([0, .25, .5, .75, 1], labels=["0", "25", "50", "75", "100"])
    cb.outline.set_visible(False); cb.ax.tick_params(length=2)
    finish(fig, "suppfig6_multiclass", list(axes.flat))

    # ---------------- Supplementary Fig. 7 ----------------
    paired = read("paired_27b_397b")
    fig, axes = new(60, 1, 3)
    for ax, metric, nice in zip(axes[:2], ("auroc", "balanced_accuracy"), ("AUROC", "balanced accuracy")):
        tidy(ax)
        d = paired[f"delta_{metric}"]
        ax.hist(d, bins=15, color=N, edgecolor="white", linewidth=.5, zorder=3)
        zero(ax, False); ax.axvline(d.median(), c=O, ls=(0, (4, 2)), lw=2.08, zorder=4)
        ax.text(d.median(), .98, " median", color=O, fontsize=FS_SMALL, transform=ax.get_xaxis_transform(), va="top", ha="left", bbox=dict(fc="white", ec="none", pad=.6))
        ax.set(xlabel=f"397B-A17B FP8 {MINUS} 27B {nice}", ylabel="Tasks", title="Change in AUROC" if metric == "auroc" else "Change in BA")
    ax = axes[2]; tidy(ax, "both")
    ax.plot([.1, 1.02], [.1, 1.02], ls=(0, (4, 2)), c=G, lw=1.28, zorder=1)
    ax.scatter(paired.auroc_27b, paired.auroc_397b, s=22, c=N, linewidths=.3, edgecolors="white", zorder=3)
    ax.set(xlabel="27B dense AUROC", ylabel="397B-A17B FP8 AUROC", title="Paired AUROC", xlim=(.15, 1.03), ylim=(.15, 1.03))
    finish(fig, "suppfig7_scaling", list(axes))

    # ---------------- Supplementary Fig. 8 ----------------
    ranks = read("ranking_agreement"); sub = ranks[ranks.backbone.isin(["27b", "397b_a17b_fp8"])]
    sub = sub.groupby(["dataset", "outcome", "backbone", "reference"])[["spearman", "kendall_tau_b"]].mean().reset_index()
    comp = read("complementarity")
    fig, axes = new(76, 1, 3)
    refn = {"best_marker": "Marker", "direction_only": "Direction", "additive": "Additive"}
    for ax, metric, nice in zip(axes[:2], ("spearman", "kendall_tau_b"), ("Spearman correlation", "Kendall τ-b")):
        tidy(ax)
        pos, values, cols = [], [], []
        for gi, b in enumerate(("27b", "397b_a17b_fp8")):
            for ri, ref in enumerate(refn):
                pos.append(gi * 4 + ri + 1); cols.append(N if gi == 0 else O)
                values.append(sub.loc[(sub.backbone == b) & (sub.reference == ref), metric].dropna().to_numpy())
        bp = ax.boxplot(values, positions=pos, widths=.62, patch_artist=True, medianprops=dict(color=INK, lw=1.92), boxprops=dict(lw=1.12),
                        whiskerprops=dict(lw=1.12), capprops=dict(lw=1.12), flierprops=dict(marker="o", ms=3.1, mfc="none", mec=G, mew=.5))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_alpha(.35); patch.set_edgecolor(c)
        ax.set_xticks(pos, [refn[r] for r in refn] * 2, rotation=40, ha="right", rotation_mode="anchor")
        ax.set(ylabel=nice, title="Spearman agreement" if metric == "spearman" else "Kendall agreement")
    axes[0].legend(handles=[Patch(fc=N, alpha=.35, ec=N, label="27B"), Patch(fc=O, alpha=.35, ec=O, label="397B-A17B FP8")], loc="upper center", ncol=2, borderaxespad=.1, columnspacing=1.0)
    axes[0].set_ylim(top=1.32)
    ax = axes[2]; tidy(ax, "both")
    ax.plot([.1, 1.02], [.1, 1.02], ls=(0, (4, 2)), c=G, lw=1.28, zorder=1)
    if len(comp):
        for j, b in enumerate([b for b in BACKBONES if b != "0.8b" and (comp.backbone == b).any()]):
            pv = comp[comp.backbone == b].pivot_table(index=["dataset", "outcome"], columns="method", values="auroc")
            if {"direction_link", "direction_plus_L0"} <= set(pv.columns):
                ax.scatter(pv.direction_link, pv.direction_plus_L0, s=22, color=[N, O][j % 2], linewidths=.3, edgecolors="white", zorder=3, label=f"{FLAT[b]} (n={len(pv)})")
        ax.legend(loc="lower right", handletextpad=.1, borderaxespad=.1)
    ax.set(xlabel="Direction-only link AUROC", ylabel="Direction + L0 AUROC", title="Complementarity", xlim=(.3, 1.03), ylim=(.3, 1.03))
    finish(fig, "suppfig8_rankings", list(axes))

    # ---------------- Supplementary Fig. 9 ----------------
    direction = read("direction_metrics"); inter = read("interaction_metrics")
    if len(direction):
        bk = [b for b in BACKBONES if (direction.backbone == b).any()]
        b = bk[-1]; d = direction[direction.backbone == b].assign(aligned=lambda x: x.delta_log_odds * np.sign(x.development_rho))
        order = d.groupby("outcome").aligned.median().sort_values().index.tolist()
        fig, axes = new(150, 1, 2, gridspec_kw=dict(width_ratios=[2.6, 1]))
        ax = axes[0]; tidy(ax, "x"); zero(ax, False)
        rng = np.random.default_rng(3)
        for i, task in enumerate(order):
            g = d[d.outcome == task]
            ax.scatter(g.aligned, np.full(len(g), i) + rng.uniform(-.22, .22, len(g)), s=16, c=np.where(g.aligned > 0, T, O), alpha=.6, linewidths=0, zorder=2)
            ax.vlines(g.aligned.median(), i - .38, i + .38, color=INK, lw=2.08, zorder=4)
        ax.set(yticks=range(len(order)), yticklabels=[OUTCOME_LABEL.get(t, t) for t in order], ylim=(-.7, len(order) - .3),
               xlabel="Log-odds change aligned to development association", title=f"{FLAT[b]} one-feature perturbations (joint plausibility unverified)")
        ax.tick_params(axis="y", labelsize=FS_SMALL, length=0)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=T, ms=4.4, label="Concordant"), Line2D([], [], marker="o", ls="", color=O, ms=4.4, label="Discordant"),
                           Line2D([], [], marker="|", ls="", color=INK, ms=8.8, mew=1.3, label="Task median")], loc="lower right", borderaxespad=.2)
        ax = axes[1]; tidy(ax); zero(ax)
        if len(inter):
            ib = [b2 for b2 in BACKBONES if b2 != "0.8b" and (inter.backbone == b2).any()]
            for j, b2 in enumerate(ib):
                v = inter[inter.backbone == b2].interaction_contrast
                ax.scatter(np.full(len(v), j) + rng.uniform(-.15, .15, len(v)), v, s=16, color=[N, O][j % 2], alpha=.6, linewidths=0, zorder=2)
                ax.hlines(v.median(), j - .3, j + .3, color=INK, lw=2.08, zorder=4)
            ax.set(xticks=range(len(ib)), xticklabels=[FLAT[b2] for b2 in ib], xlim=(-.6, len(ib) - .4))
        ax.set(ylabel="Two-feature difference-in-differences (log odds)", title="Paired contrasts")
        finish(fig, "suppfig9_directions", list(axes))
    print(f"Saved {len(list(FIG.glob('*.pdf')))} figure PDFs in {FIG}")


if __name__ == "__main__":
    main()
