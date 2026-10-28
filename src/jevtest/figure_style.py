"""Publication figure defaults shared by diagnostics and manuscript panels.

Sizes follow the supplied Nature research-figure specification. Use ``rc_context(RC)``
when constructing figures; export preserves the canvas size, never a tight crop.
Model identities are fixed across datasets, independent of performance rank.
"""
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

MM = 1 / 25.4
WIDTH_SINGLE, WIDTH_DOUBLE, MAX_HEIGHT = 89 * MM, 183 * MM, 170 * MM
INK, SECONDARY, REFERENCE = "#1A1A1A", "#595959", "#777777"
MISSING = "#E6E6E6"
LABEL_PT, TICK_PT, CELL_PT, LETTER_PT = 7.0, 7.0, 6.5, 8.0
AXIS_LW, CURVE_LW, REFERENCE_LW = 0.6, 1.2, 0.7

# Paper pair: XGBoost (dark grey) and System 1 (dark green; label-free AnyJev L0
# zero-shot). IMPROVED_COLOR is the dark red used when System 1 beats XGBoost.
MODELS = OrderedDict([
    ("M2", {"label": "XGBoost", "color": "#3C3C3C", "linestyle": "-", "marker": "o"}),
    ("ZS_raw", {"label": "System 1", "color": "#0D5C3A", "linestyle": (0, (5, 1.4, 1, 1.4)), "marker": "^"}),
    ("ICL_full", {"label": "In-context learning", "color": "#882255", "linestyle": (0, (6, 2)), "marker": "v"}),
    ("Mode", {"label": "Majority vote", "color": "#777777", "linestyle": (0, (1, 1.5)), "marker": "<"}),
    ("Mean", {"label": "Mean ensemble", "color": "#333333", "linestyle": (0, (5, 1.5, 1, 1.5)), "marker": ">"}),
    ("Suspicion", {"label": "Initial suspicion", "color": "#882255", "linestyle": (0, (9, 2.2)), "marker": "v"}),
])
REFERENCE_MODEL = "M2"
FIGURE_MODELS = ["M2", "ZS_raw"]
FIGURE_MODELS_CAL = ["M2", "ZS_raw"]
IMPROVED_COLOR = "#8B1A1A"

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": TICK_PT, "axes.labelsize": LABEL_PT, "axes.titlesize": LABEL_PT,
    "axes.titleweight": "normal", "axes.titlepad": 4, "axes.labelpad": 3,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
    "legend.fontsize": TICK_PT, "legend.frameon": False,
    "legend.handlelength": 2.3, "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
    "axes.linewidth": AXIS_LW, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "axes.axisbelow": True, "grid.color": "#D9D9D9", "grid.linewidth": 0.4,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": AXIS_LW, "ytick.major.width": AXIS_LW,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": CURVE_LW, "lines.markersize": 3.5,
    "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 450, "figure.dpi": 120,
    "savefig.bbox": None, "savefig.transparent": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.unicode_minus": True,
}


def apply():
    plt.rcParams.update(RC)


def line(model, markers=False, **extra):
    style = MODELS[model]
    kwargs = {"color": style["color"], "linestyle": style["linestyle"], "linewidth": CURVE_LW}
    if markers:
        kwargs.update(marker=style["marker"], markersize=3.5,
                      markeredgecolor="white", markeredgewidth=0.4)
    kwargs.update(extra)
    return kwargs


def reference_line(axis, xs, ys):
    axis.plot(xs, ys, color=REFERENCE, linestyle=(0, (3, 2)), linewidth=REFERENCE_LW, zorder=0)


def handle(model, markers=True):
    return Line2D([], [], **line(model, markers=markers))


def model_legend(figure, models, anchor_y):
    figure.legend([handle(m) for m in models], [MODELS[m]["label"] for m in models],
                  loc="lower center", bbox_to_anchor=(0.5, anchor_y), ncol=min(4, len(models)),
                  handlelength=2.6, columnspacing=1.2, borderaxespad=0)


def panel_letter(axis, letter, x=-0.2, y=1.04):
    return axis.text(x, y, letter, transform=axis.transAxes, fontsize=LETTER_PT,
                     fontweight="bold", fontstyle="normal", va="bottom", ha="left", color=INK)


def blend(color, share):
    """Colour at `share` of the way from white to `color`."""
    r, g, b = to_rgb(color)
    return (1 - share * (1 - r), 1 - share * (1 - g), 1 - share * (1 - b))


def unit_square(axis):
    axis.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    axis.set_xticklabels(["0", "0.5", "1"])
    axis.set_yticklabels(["0", "0.5", "1"])
    axis.set_aspect("equal")


def validate_figure(figure):
    """Return export diagnostics; geometric bounds do not certify absence of overlaps."""
    width, height = figure.get_size_inches() / MM
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Figure dimensions must be finite and positive.")
    if width > 183.05 or height > 170.05:
        raise ValueError(f"Figure is {width:.1f} × {height:.1f} mm; split into pages within 183 × 170 mm.")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = figure.bbox
    issues, fonts = [], set()
    for text in figure.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        # Explicitly clipped data annotations may legitimately lie outside the view.
        if text.get_clip_on() and text.get_clip_path() is not None:
            continue
        fonts.add(round(text.get_fontsize(), 2))
        if text.get_fontsize() < 5:
            issues.append(f"text below 5 pt: {text.get_text()[:60]!r}")
        box = text.get_window_extent(renderer)
        if (np.isfinite(box.extents).all() and
            (box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or
             box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1)):
            issues.append(f"text outside canvas: {text.get_text()[:60]!r}")
    return {"width_mm": float(width), "height_mm": float(height),
            "font_sizes_pt": sorted(fonts), "issues": list(dict.fromkeys(issues))}


def save(figure, folder, stem):
    """Save fixed-size editable PDF/SVG and 450-dpi RGB PNG, then close the figure.

    Render all formats before replacing existing files, so a rendering failure does
    not replace previously valid exports. Layout concerns are emitted as warnings.
    """
    folder = Path(folder)
    try:
        check = validate_figure(figure)
        if check["issues"]:
            warnings.warn(f"{stem}: " + "; ".join(check["issues"][:6]), UserWarning, stacklevel=2)
        folder.mkdir(parents=True, exist_ok=True)
        # Set export options explicitly even if a caller has altered global rcParams.
        with TemporaryDirectory(prefix=".figure-", dir=folder) as temp, plt.rc_context(RC):
            for extension in ("pdf", "svg", "png"):
                figure.savefig(Path(temp) / f"{stem}.{extension}", dpi=450, bbox_inches=None,
                               facecolor="white", transparent=False)
            for extension in ("pdf", "svg", "png"):
                (Path(temp) / f"{stem}.{extension}").replace(folder / f"{stem}.{extension}")
    finally:
        plt.close(figure)
