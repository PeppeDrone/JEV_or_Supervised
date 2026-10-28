#!/usr/bin/env python3
"""Problem-complexity vs M2−ZS_raw Balanced Accuracy gaps on the clinical benchmark.

Complexity measures (Ho & Basu / Lorena taxonomy + one tabular covariate):

  F1  feature overlapping — mean Fisher discriminant ratio (higher = easier);
      we plot complexity_F1 = 1 / (1 + F1) so higher = harder.
  F2  feature overlapping — geometric mean of per-feature class-range overlap
      ratios (Lorena product form stabilized as (∏ r_j)^{1/p}; higher = harder).
  N1  neighbourhood — fraction of points whose NN has another label (higher = harder).
  N2  neighbourhood — mean intra-class NN distance / mean inter-class NN distance
      (higher = harder).
  C2  class balance — imbalance ratio max_c n_c / min_c n_c (higher = more imbalanced).
  T2  dimensionality — n_features / n_samples (higher = harder).
  missingness — fraction of NA cells in the declared predictor matrix (higher = more missing).

Delta (pooled out-of-fold Balanced Accuracy):
  delta_zs_raw  = BalAcc(M2) − BalAcc(ZS_raw)

Primary association plots compare complexity distributions via boxplots of
  improved = 1{BalAcc(ZS_raw) > BalAcc(M2)}  ≡  1{delta < 0}
annotated with univariate logistic regression OR(+1SD) and likelihood-ratio test p-value.
Spearman on the continuous delta is retained as a secondary table.
The Improved colour is the dark red previously used for the JEV/probe plot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest.benchmark import ALL_PAIRS as PAIRS, OUTCOME_LABEL as LABEL
from jevtest.config import outputs_root
from jevtest.data import load, m2_predictions_path, predictor_matrix
from jevtest import figure_style as fs

MEASURES = (
    ("complexity_F1", "F1 inverse (feature overlap)", "feature"),
    ("F2", "F2 overlapping volume", "feature"),
    ("N1", "N1 borderline fraction", "neighbourhood"),
    ("N2", "N2 intra/inter NN ratio", "neighbourhood"),
    ("C2", "C2 imbalance ratio", "class balance"),
    ("T2", "T2 features / samples", "dimensionality"),
    ("missingness", "Missingness rate", "data quality"),
)
PLOT_DEFINITIONS = {
    "complexity_F1": "1 / (1 + mean Fisher ratio)",
    "F2": "geom. mean of class-range overlap / span",
    "N1": "nearest-neighbour label mismatch fraction",
    "N2": "intra-class / inter-class NN distance",
    # Existing C2: max/min class-count ratio (Lorena-style imbalance).
    "C2": "computed as max/min class count",
    "T2": "n_numeric_features / n_samples",
    "missingness": "fraction of NA cells in predictors",
}


def auroc_from_csv(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    pcols = [c for c in df.columns if c.startswith("p__")]
    y = df["observed"].to_numpy()
    P = df[pcols].to_numpy(dtype=float)
    if P.shape[1] == 2:
        return float(roc_auc_score(y, P[:, 1]))
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro"))


def bacc_from_csv(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    pcols = [c for c in df.columns if c.startswith("p__")]
    y = df["observed"].to_numpy()
    if "predicted" in df.columns:
        labels = df["predicted"].to_numpy(dtype=int)
    else:
        P = df[pcols].to_numpy(dtype=float)
        labels = P.argmax(axis=1)
    return float(balanced_accuracy_score(y, labels))


def numeric_matrix(X: pd.DataFrame) -> np.ndarray:
    """Encode all declared predictors (numeric + categorical), impute, and standardize."""
    cols = []
    for name in X.columns:
        s = X[name]
        if np.issubdtype(s.dtype, np.number):
            cols.append(s.to_numpy(dtype=float).reshape(-1, 1))
            continue
        non_na = s.dropna()
        if len(non_na) == 0:
            cols.append(np.zeros((len(s), 1), dtype=float))
            continue
        parsed = pd.to_numeric(non_na, errors="coerce")
        if parsed.notna().all():
            cols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float).reshape(-1, 1))
        else:
            cats = pd.Categorical(s)
            codes = cats.codes.astype(float)
            codes[codes < 0] = np.nan
            if len(cats.categories) <= 2:
                cols.append(codes.reshape(-1, 1))
            else:
                dummies = pd.get_dummies(s, dummy_na=False, dtype=float).to_numpy(dtype=float)
                na_mask = s.isna().to_numpy()
                if na_mask.any():
                    dummies[na_mask, :] = np.nan
                cols.append(dummies)
    mat = np.hstack(cols) if cols else np.zeros((len(X), 1), dtype=float)
    arr = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(mat)
    return StandardScaler().fit_transform(arr)


def missingness_rate(X: pd.DataFrame) -> float:
    """Fraction of missing cells in the declared predictor matrix (before imputation)."""
    if X.size == 0:
        return float("nan")
    return float(X.isna().to_numpy().mean())


def fisher_f1(X: np.ndarray, y: np.ndarray) -> float:
    """Mean Fisher discriminant ratio over features (Lorena/Ho F1 family; higher = easier)."""
    ratios = []
    classes = np.unique(y)
    for j in range(X.shape[1]):
        col = X[:, j]
        means = []
        vars_ = []
        for c in classes:
            z = col[y == c]
            means.append(z.mean())
            vars_.append(z.var(ddof=1) if len(z) > 1 else 0.0)
        means = np.asarray(means)
        # multi-class: average pairwise squared mean gap over average variance
        if len(classes) == 2:
            num = (means[0] - means[1]) ** 2
        else:
            num = float(np.mean([(means[a] - means[b]) ** 2
                                 for a in range(len(means)) for b in range(a + 1, len(means))]))
        den = float(np.mean(vars_)) + 1e-12
        ratios.append(num / den)
    return float(np.mean(ratios))


def overlapping_f2(X: np.ndarray, y: np.ndarray) -> float:
    """Geometric-mean form of Lorena F2 (higher = harder).

    For each feature j and each class pair (a, b):
      overlap = max(0, min(max_a, max_b) - max(min_a, min_b))
      span    = max(max_a, max_b) - min(min_a, min_b)
    Feature contribution is the mean pairwise overlap/span. Classic F2 is ∏ r_j;
    we report (∏ r_j)^{1/p} so wide tables do not underflow to ~0.
    """
    if X.size == 0:
        return float("nan")
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    ratios = []
    for j in range(X.shape[1]):
        col = X[:, j]
        mins, maxs = [], []
        for c in classes:
            z = col[y == c]
            if len(z) == 0:
                continue
            mins.append(float(z.min()))
            maxs.append(float(z.max()))
        if len(mins) < 2:
            continue
        pair_ratios = []
        for a in range(len(mins)):
            for b in range(a + 1, len(mins)):
                span = max(maxs[a], maxs[b]) - min(mins[a], mins[b])
                if span <= 0:
                    pair_ratios.append(1.0)
                    continue
                overlap = max(0.0, min(maxs[a], maxs[b]) - max(mins[a], mins[b]))
                pair_ratios.append(overlap / span)
        ratios.append(float(np.mean(pair_ratios)))
    if not ratios:
        return float("nan")
    ratios = np.clip(np.asarray(ratios, dtype=float), 1e-12, 1.0)
    return float(np.exp(np.mean(np.log(ratios))))


def neighbourhood_n1(X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of instances whose nearest neighbour has a different label (Ho & Basu N1)."""
    if len(y) < 3:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X)
    idx = nn.kneighbors(return_distance=False)[:, 1]
    return float(np.mean(y[idx] != y))


def neighbourhood_n2(X: np.ndarray, y: np.ndarray) -> float:
    """Mean intra-class NN distance / mean inter-class NN distance (Lorena N2; higher = harder)."""
    n = len(y)
    if n < 3:
        return float("nan")
    # Extra neighbours so we can pick the nearest same-label and different-label neighbour.
    k = min(n, max(8, int(np.sqrt(n)) + 2))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X)
    dists, idxs = nn.kneighbors()
    intra, inter = [], []
    for i in range(n):
        # Skip self at column 0.
        same = None
        other = None
        for d, j in zip(dists[i, 1:], idxs[i, 1:]):
            if same is None and y[j] == y[i]:
                same = float(d)
            elif other is None and y[j] != y[i]:
                other = float(d)
            if same is not None and other is not None:
                break
        if same is not None:
            intra.append(same)
        if other is not None:
            inter.append(other)
    if not intra or not inter:
        return float("nan")
    return float(np.mean(intra) / (np.mean(inter) + 1e-12))


def imbalance_c2(y: np.ndarray) -> float:
    """Imbalance ratio max/min class count (Lorena C2; 1 = balanced)."""
    counts = np.bincount(y)
    counts = counts[counts > 0]
    return float(counts.max() / counts.min())


def dimensionality_t2(n_features: int, n_samples: int) -> float:
    """Features-per-sample ratio (Lorena T2; higher = harder)."""
    if n_samples <= 0:
        return float("nan")
    return float(n_features) / float(n_samples)


def complexity_row(run) -> dict:
    raw = predictor_matrix(run)
    X = numeric_matrix(raw)
    y = run.y
    f1 = fisher_f1(X, y)
    n_feat = int(raw.shape[1])
    n = int(len(y))
    return {
        "dataset": run.dataset,
        "outcome": run.outcome,
        "label": LABEL.get(run.outcome, run.outcome),
        "n": n,
        "n_classes": int(len(run.classes)),
        "n_features_numeric": n_feat,
        "n_features_encoded": int(X.shape[1]),
        "F1": f1,
        "complexity_F1": 1.0 / (1.0 + f1),
        "F2": overlapping_f2(X, y),
        "N1": neighbourhood_n1(X, y),
        "N2": neighbourhood_n2(X, y),
        "C2": imbalance_c2(y),
        "T2": dimensionality_t2(n_feat, n),
        "missingness": missingness_rate(raw),
    }


def spearman_table(frame: pd.DataFrame, delta_col: str) -> pd.DataFrame:
    rows = []
    sub = frame.dropna(subset=[delta_col])
    for key, title, domain in MEASURES:
        usable = sub.dropna(subset=[key, delta_col])
        n = len(usable)
        if n < 3:
            rows.append({"delta": delta_col, "measure": key, "title": title, "domain": domain,
                         "n": n, "spearman_rho": np.nan, "pvalue": np.nan})
            continue
        rho, p = spearmanr(usable[key], usable[delta_col])
        rows.append({"delta": delta_col, "measure": key, "title": title, "domain": domain,
                     "n": n, "spearman_rho": float(rho), "pvalue": float(p)})
    return pd.DataFrame(rows)


def _bernoulli_loglik(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_logistic_improved(x: np.ndarray, improved: np.ndarray) -> dict:
    """Univariate logistic: P(improved | complexity). LRT vs intercept-only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(improved, dtype=float)
    n = int(len(y))
    n_win = int(y.sum())
    out = {
        "n": n,
        "n_win": n_win,
        "coef": np.nan,
        "intercept": np.nan,
        "odds_ratio": np.nan,
        "lrt_pvalue": np.nan,
        "fitted": False,
    }
    if n < 3 or n_win == 0 or n_win == n:
        return out

    # Standardise x so the slope is numerically stable across F1 / N1 / C2 scales.
    x_mean = float(x.mean())
    x_std = float(x.std(ddof=0))
    if x_std < 1e-12:
        return out
    z = ((x - x_mean) / x_std).reshape(-1, 1)

    full = LogisticRegression(penalty=None, solver="lbfgs", max_iter=4000)
    full.fit(z, y)
    p_full = full.predict_proba(z)[:, 1]
    ll_full = _bernoulli_loglik(y, p_full)

    p_null = np.full(n, y.mean(), dtype=float)
    ll_null = _bernoulli_loglik(y, p_null)
    lrt = max(0.0, -2.0 * (ll_null - ll_full))
    p_lrt = float(chi2.sf(lrt, df=1))

    # Map standardised coef back to original x units: logit = a + b*((x-m)/s)
    #        = (a - b*m/s) + (b/s)*x
    b_std = float(full.coef_.ravel()[0])
    a_std = float(full.intercept_.ravel()[0])
    coef = b_std / x_std
    intercept = a_std - b_std * x_mean / x_std
    out.update({
        "coef": coef,
        "intercept": intercept,
        "odds_ratio": float(np.exp(coef)),
        "lrt_pvalue": p_lrt,
        "fitted": True,
        "x_mean": x_mean,
        "x_std": x_std,
        "model": full,
    })
    return out


def logistic_table(frame: pd.DataFrame, delta_col: str) -> pd.DataFrame:
    rows = []
    sub = frame.dropna(subset=[delta_col])
    for key, title, domain in MEASURES:
        usable = sub.dropna(subset=[key, delta_col])
        x = usable[key].to_numpy(dtype=float)
        improved = (usable[delta_col].to_numpy(dtype=float) < 0.0).astype(float)
        fit = fit_logistic_improved(x, improved)
        or_sd = float(np.exp(fit["coef"] * fit["x_std"])) if fit["fitted"] else np.nan
        rows.append({
            "delta": delta_col,
            "measure": key,
            "title": title,
            "domain": domain,
            "n": fit["n"],
            "n_win": fit["n_win"],
            "coef_per_unit": fit["coef"],
            "odds_ratio_per_unit": fit["odds_ratio"],
            "odds_ratio_per_sd": or_sd,
            "lrt_pvalue": fit["lrt_pvalue"],
        })
    return pd.DataFrame(rows)


def plot_scatter(frame: pd.DataFrame, delta_col: str, model: str, out: Path,
                 measures=None, first_letter: str = "a", n_rows: int = 3, ylabels=None):
    """Boxplots + numbered jittered points comparing complexity in Improved vs Not improved,
    with univariate logistic regression stats (OR per +1 SD and LRT p-value) in panel titles.
    """
    usable_all = frame.dropna(subset=[delta_col]).copy()
    method_label = fs.MODELS[model]["label"]
    win_color = fs.IMPROVED_COLOR
    lose_color = fs.MODELS["M2"]["color"]
    labels = sorted(usable_all["label"].unique())
    outcome_numbers = {label: number for number, label in enumerate(labels, start=1)}
    rng = np.random.default_rng(0)
    measures = MEASURES if measures is None else measures
    n_measures = len(measures)
    # Grid of measure panels; the three-column legend spans the last two cells.
    n_cols = 3

    with plt.rc_context(fs.RC):
        fig = plt.figure(figsize=(fs.WIDTH_DOUBLE, min(fs.MAX_HEIGHT, 56 * fs.MM * n_rows)))
        gs = fig.add_gridspec(n_rows, n_cols)
        box_axes = [fig.add_subplot(gs[i // n_cols, i % n_cols]) for i in range(n_measures)]
        legend_ax = fig.add_subplot(gs[-1, 1:])

        for index, (ax, (key, title, _domain)) in enumerate(zip(box_axes, measures)):
            usable = frame.dropna(subset=[key, delta_col]).sort_values("label")
            ax.set_ylabel(ylabels[key] if ylabels else f"{title}\n({PLOT_DEFINITIONS[key]})", fontsize=6.5)
            fs.panel_letter(ax, chr(ord(first_letter) + index), x=-0.16, y=1.08)
            if usable.empty:
                ax.set_title("No paired AUROC estimates", loc="left", fontsize=6.5)
                continue

            x = usable[key].to_numpy(dtype=float)
            delta = usable[delta_col].to_numpy(dtype=float)
            improved = (delta < 0.0).astype(float)

            g_not = x[improved <= 0.5]
            g_imp = x[improved > 0.5]
            n_not = len(g_not)
            n_win = len(g_imp)
            n = len(x)

            # Draw boxplots for groups with at least 1 observation
            positions, data_groups, box_colors = [], [], []
            if n_not > 0:
                positions.append(0)
                data_groups.append(g_not)
                box_colors.append(lose_color)
            if n_win > 0:
                positions.append(1)
                data_groups.append(g_imp)
                box_colors.append(win_color)

            if data_groups:
                bp = ax.boxplot(
                    data_groups,
                    positions=positions,
                    widths=0.46,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#1A1A1A", "linewidth": 1.3},
                    whiskerprops={"color": "#595959", "linewidth": 0.8},
                    capprops={"color": "#595959", "linewidth": 0.8},
                    boxprops={"linewidth": 0.8},
                    zorder=1,
                )
                for patch, c in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(fs.blend(c, 0.28))
                    patch.set_edgecolor(c)

            # Overlay individual numbered dataset points
            jitter = rng.uniform(-0.14, 0.14, size=len(improved))
            x_pos = improved + jitter
            colours = np.where(improved > 0.5, win_color, lose_color)
            ax.scatter(x_pos, x, s=42, marker="o", c=colours,
                       edgecolor="white", linewidth=0.6, zorder=3)
            for row, xp in zip(usable.itertuples(index=False), x_pos):
                number = outcome_numbers[row.label]
                ax.text(xp, getattr(row, key), str(number),
                        ha="center", va="center", fontsize=5.0, color="white",
                        fontweight="bold", zorder=4)

            ax.set_xlim(-0.52, 1.52)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"No\n(n={n_not})", f"Yes\n(n={n_win})"], fontsize=6.5)
            ax.margins(y=0.14)

            fit = fit_logistic_improved(x, improved)
            if fit["fitted"]:
                # Report standardized OR (+1 SD) so extreme unit scales (T2, missingness) remain readable
                or_sd = float(np.exp(fit["coef"] * fit["x_std"]))
                or_txt = f"{or_sd:.2f}" if 0.01 <= or_sd < 100 else f"{or_sd:.1e}"
                ax.set_title(
                    f"OR(+1SD)={or_txt}, LRT p{' < 0.001' if fit['lrt_pvalue'] < 0.001 else '=' + format(fit['lrt_pvalue'], '.3f')}",
                    loc="left", fontsize=6.5,
                )
            else:
                ax.set_title(
                    f"wins {n_win}/{n}; logreg n/a",
                    loc="left", fontsize=6.5,
                )

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=fs.blend(win_color, 0.28), edgecolor=win_color, linewidth=0.8,
                  label=f"Yes: {method_label} > XGB"),
            Patch(facecolor=fs.blend(lose_color, 0.28), edgecolor=lose_color, linewidth=0.8,
                  label=f"No: {method_label} ≤ XGB"),
        ]
        handles.extend([
            Line2D([], [], marker="o", color="none",
                   markerfacecolor=win_color if (usable_all.loc[usable_all["label"] == label, delta_col].iloc[0] < 0) else lose_color,
                   markeredgecolor="white", markeredgewidth=0.5, markersize=6.0,
                   label=f"{number}  {label}")
            for label, number in outcome_numbers.items()
        ])
        legend_ax.set_axis_off()
        legend_ax.legend(
            handles, [h.get_label() for h in handles],
            loc="center", bbox_to_anchor=(0.50, 0.50),
            fontsize=5.8, frameon=False, handletextpad=0.35,
            labelspacing=0.42, columnspacing=1.15, borderaxespad=0.0, ncol=3,
            title=f"{method_label} vs XGB (Bal. Acc.)", title_fontsize=6.5,
        )
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.07, top=0.93,
                            wspace=0.38, hspace=0.55)
        out.parent.mkdir(parents=True, exist_ok=True)
        stem = out.stem if out.suffix.lower() in {".png", ".pdf", ".svg"} else out.name
        fs.save(fig, out.parent, stem)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out_dir = outputs_root(root) / "combined" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    old_dir = outputs_root(root) / "combined" / "complexity"
    if old_dir.is_dir() and old_dir.resolve() != out_dir.resolve():
        for leftover in old_dir.iterdir():
            if leftover.is_file():
                leftover.unlink()
        try:
            old_dir.rmdir()
        except OSError:
            pass

    rows = []
    for dataset, outcome in PAIRS:
        run = load(root, dataset, outcome)
        row = complexity_row(run)
        pred = run.out / "predictions"
        row["bacc_M2"] = bacc_from_csv(m2_predictions_path(run))
        row["bacc_ZS_raw"] = bacc_from_csv(pred / "ZS_raw.csv")
        row["auroc_M2"] = auroc_from_csv(m2_predictions_path(run))
        row["auroc_ZS_raw"] = auroc_from_csv(pred / "ZS_raw.csv")
        m2 = row["bacc_M2"]
        row["delta_zs_raw"] = (m2 - row["bacc_ZS_raw"]) if m2 is not None and row["bacc_ZS_raw"] is not None else np.nan
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "complexity_and_gaps.csv", index=False)

    corr = spearman_table(frame, "delta_zs_raw")
    corr.to_csv(out_dir / "complexity_gap_spearman.csv", index=False)

    logit = logistic_table(frame, "delta_zs_raw")
    logit.to_csv(out_dir / "complexity_gap_logistic.csv", index=False)

    for leftover in out_dir.glob("complexity_vs_delta_probe_hybrid.*"):
        leftover.unlink()

    plot_scatter(frame, "delta_zs_raw", "ZS_raw", out_dir / "complexity_vs_delta_zs_raw")

    summary = {
        "metric": "balanced_accuracy",
        "measures": [
            {"id": "complexity_F1", "source": "Ho & Basu / Lorena F1 (inverted)", "domain": "feature overlapping",
             "definition": "1 / (1 + mean Fisher discriminant ratio over encoded features); higher = harder"},
            {"id": "F2", "source": "Lorena F2 (geometric-mean form)", "domain": "feature overlapping",
             "definition": "(∏_j mean pairwise class-range overlap/span)^{1/p}; higher = harder"},
            {"id": "N1", "source": "Ho & Basu N1", "domain": "neighbourhood",
             "definition": "fraction of points whose nearest neighbour has a different label; higher = harder"},
            {"id": "N2", "source": "Lorena N2", "domain": "neighbourhood",
             "definition": "mean intra-class NN distance / mean inter-class NN distance; higher = harder"},
            {"id": "C2", "source": "Lorena C2", "domain": "class balance",
             "definition": "max class count / min class count; higher = more imbalanced"},
            {"id": "T2", "source": "Lorena T2", "domain": "dimensionality",
             "definition": "n_features / n_samples; higher = harder"},
            {"id": "missingness", "source": "tabular covariate", "domain": "data quality",
             "definition": "fraction of NA cells in the declared predictor matrix; higher = more missing"},
        ],
        "analyses": [
            {"name": "delta_zs_raw", "delta": "BalAcc(M2) - BalAcc(ZS_raw)",
             "binary": "improved = 1{BalAcc(ZS_raw) > BalAcc(M2)}"},
        ],
        "n_outcomes": int(len(frame)),
        "n_with_zs_raw_delta": int(frame["delta_zs_raw"].notna().sum()),
        "note": (
            "Primary plots: boxplots of complexity measures by improved = 1{BalAcc(ZS_raw) > BalAcc(M2)}, "
            "annotated with univariate logistic regression OR(+1SD) and LRT p-value. "
            "Improved uses the dark red from the retired JEV/probe complexity plot."
        ),
    }
    (out_dir / "complexity_gap_summary.json").write_text(json.dumps(summary, indent=2))

    print(frame.to_string(index=False))
    print()
    print("Logistic (improved = System 1 Balanced Accuracy > XGB):")
    print(logit.to_string(index=False))
    print()
    print("Spearman (continuous Balanced Accuracy delta):")
    print(corr.to_string(index=False))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
