#!/usr/bin/env python3
"""Publication figure: codebook detail vs System 1 and vs the XGBoost gap.

Reads outputs/combined/codebook/codebook_quality_and_gaps.csv and writes
PDF / SVG / PNG into outputs/combined/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest import figure_style as fs

CSV = ROOT / "outputs" / "combined" / "codebook" / "codebook_quality_and_gaps.csv"
OUT = ROOT / "outputs" / "combined" / "figures"

COLUMNS = (
    ("added_per_predictor", "Unique words per predictor"),
    ("words_per_predictor", "Prompt words per predictor"),
    ("context_words", "Setting + question length"),
)
XLIM = {
    "added_per_predictor": (-0.2, 6.55),
    "words_per_predictor": (1.6, 25.4),
    "context_words": (16, 61),
}
XTICKS = {
    "context_words": [20, 30, 40, 50, 60],
}


def fmt_p(p: float, name: str = "p") -> str:
    if p < 0.001:
        return f"{name} < 0.001"
    return f"{name} = {p:.3f}"


def bh_adjust(pvalues):
    """Benjamini–Hochberg FDR over the six displayed Spearman tests."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.clip(p[order], 0.0, 1.0)
    adj = np.empty(n)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adj[i] = running
    out = np.empty(n)
    out[order] = adj
    return out


def fit_band(ax, x, y, color, rng, emphasize: bool):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 5 or float(np.std(x)) < 1e-12:
        return
    fit = linregress(x, y)
    xline = np.linspace(float(x.min()), float(x.max()), 80)
    yline = fit.intercept + fit.slope * xline
    if emphasize:
        yhat = fit.intercept + fit.slope * x
        resid = y - yhat
        draws = np.empty((400, len(xline)))
        for i in range(400):
            yb = yhat + rng.choice(resid, size=len(resid), replace=True)
            slope, intercept = np.polyfit(x, yb, 1)
            draws[i] = intercept + slope * xline
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        ax.fill_between(xline, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(xline, yline, color=color, linewidth=1.2, zorder=2, solid_capstyle="round")
    else:
        ax.plot(xline, yline, color=color, linewidth=0.9, linestyle=(0, (3, 2)),
                zorder=2, solid_capstyle="round")


def panel(ax, frame, x_key, y_key, letter, xlabel, ylabel, show_ylabel,
          show_xlabel, rng, p_adj):
    usable = frame.dropna(subset=[x_key, y_key])
    x = usable[x_key].to_numpy(dtype=float)
    y = usable[y_key].to_numpy(dtype=float)
    win = usable["delta_zs_raw"].to_numpy(dtype=float) < 0.0
    win_c, lose_c = fs.IMPROVED_COLOR, fs.MODELS["M2"]["color"]
    colours = np.where(win, win_c, lose_c)
    rho, p = spearmanr(x, y)
    significant = p_adj < 0.05
    line_color = fs.MODELS["ZS_raw"]["color"] if significant else fs.REFERENCE
    fit_band(ax, x, y, line_color, rng, emphasize=significant)
    if y_key == "delta_zs_raw":
        ax.axhline(0.0, color=fs.REFERENCE, linestyle=(0, (3, 2)),
                   linewidth=fs.REFERENCE_LW, zorder=0)

    ax.scatter(x, y, s=28, marker="o", c=colours,
               edgecolor="white", linewidth=0.45, zorder=3)

    fs.panel_letter(ax, letter, x=-0.16, y=1.06)
    ax.set_title(
        f"ρ = {rho:.2f}, {fmt_p(p)}, {fmt_p(p_adj, '$p_{\\mathrm{adj}}$')}",
        loc="left", fontsize=6.5,
    )
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=6.5)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False, length=2.5)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=7.0)
    if y_key == "bacc_ZS_raw":
        ax.set_ylim(0.10, 1.02)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        ax.set_ylim(-0.22, 0.90)
        ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlim(*XLIM[x_key])
    if x_key in XTICKS:
        ax.set_xticks(XTICKS[x_key])
    ax.tick_params(axis="both", labelsize=6.5)


def main():
    frame = pd.read_csv(CSV)
    rng = np.random.default_rng(0)
    win_c, lose_c = fs.IMPROVED_COLOR, fs.MODELS["M2"]["color"]
    rows = (
        ("bacc_ZS_raw", "System 1 balanced accuracy"),
        ("delta_zs_raw", "XGBoost − System 1"),
    )

    specs = [(y_key, x_key) for y_key, _ in rows for x_key, _ in COLUMNS]
    p_raw = []
    for y_key, x_key in specs:
        usable = frame.dropna(subset=[x_key, y_key])
        p_raw.append(float(spearmanr(usable[x_key], usable[y_key]).pvalue))
    p_adj = bh_adjust(p_raw)
    p_adj_map = dict(zip(specs, p_adj))

    with plt.rc_context(fs.RC):
        fig = plt.figure(figsize=(fs.WIDTH_DOUBLE, 118 * fs.MM))
        gs = fig.add_gridspec(2, 3)
        axes = [[None, None, None], [None, None, None]]
        for c, (x_key, _xlabel) in enumerate(COLUMNS):
            axes[1][c] = fig.add_subplot(gs[1, c])
            axes[0][c] = fig.add_subplot(gs[0, c], sharex=axes[1][c])

        letter = 0
        for r, (y_key, ylabel) in enumerate(rows):
            for c, (x_key, xlabel) in enumerate(COLUMNS):
                panel(
                    axes[r][c], frame, x_key, y_key, chr(ord("a") + letter),
                    xlabel, ylabel, show_ylabel=(c == 0), show_xlabel=(r == 1),
                    rng=rng, p_adj=float(p_adj_map[(y_key, x_key)]),
                )
                letter += 1

        handles = [
            Patch(facecolor=fs.blend(win_c, 0.85), edgecolor=win_c, linewidth=0.6,
                  label="System 1 > XGBoost"),
            Patch(facecolor=fs.blend(lose_c, 0.85), edgecolor=lose_c, linewidth=0.6,
                  label="System 1 ≤ XGBoost"),
            Line2D([0], [0], color=fs.MODELS["ZS_raw"]["color"], linewidth=1.2,
                   label=r"OLS + 95% CI, $p_{\mathrm{adj}}$ < 0.05"),
            Line2D([0], [0], color=fs.REFERENCE, linewidth=0.9, linestyle=(0, (3, 2)),
                   label=r"OLS, $p_{\mathrm{adj}}$ ≥ 0.05"),
        ]
        fig.legend(
            handles, [h.get_label() for h in handles],
            loc="lower center", bbox_to_anchor=(0.5, 0.015),
            ncol=4, fontsize=6.5, frameon=False,
            handletextpad=0.4, columnspacing=1.3, borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.93,
                            wspace=0.28, hspace=0.28)
        OUT.mkdir(parents=True, exist_ok=True)
        fs.save(fig, OUT, "codebook_quality")


if __name__ == "__main__":
    main()
    print(f"Wrote {OUT / 'codebook_quality.pdf'}")
    print(f"Wrote {OUT / 'codebook_quality.svg'}")
    print(f"Wrote {OUT / 'codebook_quality.png'}")
