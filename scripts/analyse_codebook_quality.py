#!/usr/bin/env python3
"""Codebook explanation quality vs LLM / XGBoost Balanced Accuracy.

The frozen Qwen3-4B reader never sees the CSV schema. It sees only the English
serialisation from ``datasets/<name>/dataset.yaml``. We still cannot score
medical correctness against an external gold codebook. The v1 metrics treated
that as three global rates plus raw word counts. Those measures had four
failures, which this script repairs.

v1 problems
  1. Opaque-code translation saturates: 0/1 → Yes/No is almost universal, and
     the rule misses abbreviations (ge40, DGN3), CamelCase dumps (HighBP), and
     Excel-corrupted bins (14-Oct). The unit of analysis was codes, so one
     multi-level variable dominated the rate.
  2. Unit coverage assumes every numeric needs an SI unit. Dermatology 0–3
     scores already have a scale_note (ignored). SPECT / Parkinson codes have
     no physical unit; penalising them measures feature type, not codebook work.
  3. Label specificity counted underscore expansion as a pass (serum_creatinine
     → Serum creatinine) and let short cryptic names through (scoma, edu).
  4. Detail was bulk token count. Dermatology copies the same scale_note onto
     33 variables; Arrhythmia is 279 lines. The z-scored composite was those
     two points. Prompt length is also ~ n_predictors × words per line.

v2 (primary)
  Accuracy is a **per-predictor explanation rate** in [0, 1]:
    explanation_rate = mean_j 0.5 * label_score_j + 0.5 * quantity_score_j
    label_score      1 if the text_label is ordinary English (Title case word
                     or a multi-word phrase); 0 if it is a storage surface
                     (identifier, CamelCase concat, indexed code F1R/DGN3,
                     Excel date); 0.4 if it is a dumped lowercase word.
    quantity_score   numeric: 1 if unit or scale_note; 0.8 if the label already
                     carries a parenthetical unit; else 0.
                     categorical: fraction of text_values that are not storage
                     surfaces (City→City counts; 3→3 and ge40→ge40 do not).
  Every task has a defined score; variables are equally weighted.

  Detail is three length measures (no bulk-prompt total):
    added_per_predictor   unique codebook word types minus schema tokens,
                          divided by n_predictors (repeated scale_notes count
                          once).
    words_per_predictor   mean ZS_raw prompt tokens / n_predictors.
    context_words         tokens in setting + observations_header + question.

v1 columns stay in the CSV as v1_* so the two definitions can be compared.
Targets: bacc_ZS_raw and delta_zs_raw = BalAcc(M2) − BalAcc(ZS_raw).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import chi2, pearsonr, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest.benchmark import ALL_PAIRS as PAIRS, OUTCOME_LABEL as LABEL
from jevtest.config import outputs_root
from jevtest.data import load, m2_predictions_path
from jevtest.llm import options_block
from jevtest.text import render
from jevtest import figure_style as fs

PROMPT_SAMPLE = 1000
PROMPT_SEED = 20261008
WORD = re.compile(r"[A-Za-z]{2,}")
EXCEL_DATE = re.compile(
    r"^\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$", re.I
)
INDEXED_CODE = re.compile(r"^[A-Za-z]{1,8}\d+[A-Za-z0-9]*$")
PAREN_UNIT = re.compile(
    r"\((?!(?:mean|worst|standard error|sd|se|iqr|min|max)\b)[^)]+\)", re.I
)
TITLE_WORD = re.compile(r"^[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?$")
COMMON_WORD = {
    "age", "sex", "race", "bmi", "yes", "no", "male", "female", "weight",
    "height", "smoker", "stroke", "income", "diabetes",
}

MEASURES = (
    ("explanation_rate", "Explanation rate", "fidelity"),
    ("label_readability", "Label readability", "fidelity"),
    ("quantity_explained", "Quantity explained", "fidelity"),
    ("words_per_predictor", "Prompt words per predictor", "detail"),
    ("added_per_predictor", "Unique words per predictor", "detail"),
    ("context_words", "Setting + question length", "detail"),
    ("v1_fidelity_score", "v1 fidelity (comparison)", "v1"),
    ("v1_detail_score", "v1 detail (comparison)", "v1"),
    ("v1_translation_rate", "v1 opaque-code translation", "v1"),
    ("v1_unit_coverage", "v1 unit coverage", "v1"),
)
PLOT_MEASURES = (
    "explanation_rate",
    "label_readability",
    "quantity_explained",
    "words_per_predictor",
    "added_per_predictor",
    "context_words",
)
PLOT_DEFINITIONS = {
    "explanation_rate": "mean ½ label + ½ quantity per predictor",
    "label_readability": "P(text_label is ordinary English)",
    "quantity_explained": "units / scale notes / non-code values",
    "words_per_predictor": "ZS_raw prompt tokens / n predictors",
    "added_per_predictor": "unique codebook words − schema tokens",
    "context_words": "setting + header + question tokens",
}
UNIT_INTERVAL = {
    "explanation_rate", "label_readability", "quantity_explained",
    "v1_fidelity_score", "v1_translation_rate", "v1_unit_coverage",
}


def tokens(text) -> int:
    if text is None:
        return 0
    return len(str(text).split())


def slug(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def is_storage_surface(text) -> bool:
    """Raw stored token: number, identifier, CamelCase dump, indexed code, Excel date."""
    s = str(text).strip()
    if not s:
        return True
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        return True
    if EXCEL_DATE.fullmatch(s):
        return True
    if "_" in s:
        return True
    if " " not in s and re.search(r"[a-z][A-Z]", s):
        return True
    if INDEXED_CODE.fullmatch(s):
        return True
    if " " not in s and ":" in s:
        return True
    if re.fullmatch(r"[A-Za-z]", s):
        return True
    compact = re.sub(r"[^A-Za-z]", "", s)
    if compact.isupper() and 2 <= len(compact) <= 5 and " " not in s and re.search(r"\d", s):
        return True
    return False


def is_storage_key(column: str) -> bool:
    name = str(column)
    if "_" in name or any(ch.isdigit() for ch in name):
        return True
    if re.search(r"[a-z][A-Z]", name):
        return True
    if name.isupper() and len(name) > 3:
        return True
    if name.isalpha() and name.islower() and len(name) > 8:
        return True
    return False


def is_identifier_dump(column: str, text_label: str) -> bool:
    label = str(text_label).strip()
    if slug(label) != slug(column):
        return False
    if " " in label or "-" in label:
        return False
    return is_storage_key(column)


def is_opaque_code(code: str) -> bool:
    return bool(re.fullmatch(r"^(?:[+-]?\d+(?:\.\d+)?|[A-Za-z])$", str(code).strip()))


def label_score(column: str, text_label: str) -> float:
    label = str(text_label).strip()
    if is_storage_surface(label):
        return 0.0
    if " " in label:
        return 1.0
    word = re.sub(r"[^A-Za-z]", "", label)
    if TITLE_WORD.fullmatch(word) and len(word) >= 3:
        return 1.0
    if word.lower() in COMMON_WORD:
        return 0.75
    if re.fullmatch(r"[a-z]+", label) and len(label) >= 3:
        return 0.4
    if is_identifier_dump(column, label):
        return 0.0
    return 0.5


def quantity_score(info: dict) -> float:
    if info["kind"] == "numeric":
        if str(info.get("unit") or "").strip():
            return 1.0
        if str(info.get("scale_note") or "").strip():
            return 1.0
        if PAREN_UNIT.search(str(info.get("text_label") or "")):
            return 0.8
        return 0.0
    texts = list((info.get("text_values") or {}).values())
    if not texts:
        return 0.0
    return float(np.mean([0.0 if is_storage_surface(text) else 1.0 for text in texts]))


def schema_tokens(column: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(column))
    text = re.sub(r"[_.:()]+", " ", text)
    return {m.group(0).lower() for m in WORD.finditer(text)}


def codebook_tokens(info: dict) -> set[str]:
    parts = [info.get("text_label"), info.get("scale_note"), info.get("unit")]
    parts.extend((info.get("text_values") or {}).values())
    found = set()
    for part in parts:
        if part:
            found.update(m.group(0).lower() for m in WORD.finditer(str(part)))
    return found


def bacc_from_csv(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    pcols = [c for c in df.columns if c.startswith("p__")]
    y = df["observed"].to_numpy()
    if "predicted" in df.columns:
        labels = df["predicted"].to_numpy(dtype=int)
    else:
        labels = df[pcols].to_numpy(dtype=float).argmax(axis=1)
    return float(balanced_accuracy_score(y, labels))


def codebook_row(run) -> dict:
    ds = run.ds
    predictors = list(ds["predictors"].values())
    n_pred = max(len(predictors), 1)

    label_scores, quantity_scores = [], []
    v1_label_words, v1_value_words = [], []
    n_numeric = n_with_unit = 0
    n_opaque = n_translated = 0
    n_specific = 0
    schema, codebook = set(), set()

    for info in predictors:
        note = info.get("scale_note")
        label_scores.append(label_score(info["column"], info["text_label"]))
        quantity_scores.append(quantity_score(info))
        schema |= schema_tokens(info["column"])
        codebook |= codebook_tokens(info)

        v1_label_words.append(tokens(info["text_label"]) + (tokens(note) if note else 0))
        if not is_identifier_dump(info["column"], info["text_label"]):
            n_specific += 1
        if info["kind"] == "numeric":
            n_numeric += 1
            if str(info.get("unit") or "").strip():
                n_with_unit += 1
        for code, text in (info.get("text_values") or {}).items():
            v1_value_words.append(tokens(text))
            if is_opaque_code(code):
                n_opaque += 1
                if str(text).strip().lower() != str(code).strip().lower():
                    n_translated += 1

    context = " ".join([
        str(ds.get("setting") or ""),
        str(ds.get("observations_header") or ""),
        str(run.spec.get("question") or ""),
    ])
    rng = np.random.default_rng(PROMPT_SEED)
    n = len(run.frame)
    take = np.sort(rng.choice(n, size=min(PROMPT_SAMPLE, n), replace=False))
    prompt_words = []
    options = options_block(run.spec["question"], run.classes, list(range(len(run.classes))))
    for idx in take:
        body = render(run.frame.iloc[int(idx)], ds)
        prompt_words.append(tokens(f"{body}\n\n{options}"))
    mean_prompt = float(np.mean(prompt_words)) if prompt_words else np.nan
    added = codebook - schema

    pred_explain = 0.5 * np.asarray(label_scores) + 0.5 * np.asarray(quantity_scores)
    return {
        "dataset": run.dataset,
        "outcome": run.outcome,
        "label": LABEL.get(run.outcome, run.outcome),
        "n": n,
        "n_predictors": n_pred,
        "n_numeric": n_numeric,
        "n_opaque_codes": n_opaque,
        "explanation_rate": float(pred_explain.mean()),
        "label_readability": float(np.mean(label_scores)),
        "quantity_explained": float(np.mean(quantity_scores)),
        "added_per_predictor": float(len(added) / n_pred),
        "words_per_predictor": float(mean_prompt / n_pred) if np.isfinite(mean_prompt) else np.nan,
        "context_words": float(tokens(context)),
        "mean_prompt_words": mean_prompt,
        "unique_codebook_words": int(len(codebook)),
        "unique_added_words": int(len(added)),
        "v1_mean_label_words": float(np.mean(v1_label_words)) if v1_label_words else np.nan,
        "v1_mean_value_words": float(np.mean(v1_value_words)) if v1_value_words else np.nan,
        "v1_translation_rate": (n_translated / n_opaque) if n_opaque else np.nan,
        "v1_unit_coverage": (n_with_unit / n_numeric) if n_numeric else np.nan,
        "v1_label_specificity": (n_specific / n_pred) if n_pred else np.nan,
    }


def add_composites(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    detail_parts = ["added_per_predictor", "words_per_predictor", "context_words"]
    ranks = []
    for col in detail_parts:
        x = out[col].astype(float)
        ranks.append(pd.Series(rankdata(x.fillna(x.median())), index=out.index))
    out["detail_score"] = np.mean(np.column_stack(ranks), axis=1)

    z_cols = []
    for col in ("v1_mean_label_words", "v1_mean_value_words", "context_words", "mean_prompt_words"):
        x = out[col].astype(float)
        std = float(x.std(ddof=0))
        z_cols.append((x - x.mean()) / std if std > 1e-12 else x * 0.0)
    out["v1_detail_score"] = np.nanmean(np.column_stack(z_cols), axis=1)
    out["v1_fidelity_score"] = np.nanmean(
        out[["v1_translation_rate", "v1_unit_coverage", "v1_label_specificity"]].to_numpy(dtype=float),
        axis=1,
    )
    return out


def partial_spearman(x, y, z):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = np.asarray(x)[mask], np.asarray(y)[mask], np.asarray(z)[mask]
    if len(x) < 5:
        return np.nan, np.nan, int(len(x))
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    design = np.column_stack([np.ones(len(rz)), rz])

    def resid(a):
        beta, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ beta

    rho, p = pearsonr(resid(rx), resid(ry))
    return float(rho), float(p), int(len(x))


def spearman_table(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    sub = frame.dropna(subset=[target])
    for key, title, domain in MEASURES:
        usable = sub.dropna(subset=[key, target, "n_predictors"])
        n = len(usable)
        if n < 3:
            rows.append({
                "target": target, "measure": key, "title": title, "domain": domain,
                "n": n, "spearman_rho": np.nan, "pvalue": np.nan,
                "partial_rho_n_predictors": np.nan, "partial_pvalue": np.nan,
                "iqr": np.nan,
            })
            continue
        rho, p = spearmanr(usable[key], usable[target])
        pr, pp, _ = partial_spearman(
            usable[key].to_numpy(dtype=float),
            usable[target].to_numpy(dtype=float),
            usable["n_predictors"].to_numpy(dtype=float),
        )
        x = usable[key].to_numpy(dtype=float)
        rows.append({
            "target": target, "measure": key, "title": title, "domain": domain,
            "n": n, "spearman_rho": float(rho), "pvalue": float(p),
            "partial_rho_n_predictors": pr, "partial_pvalue": pp,
            "iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
        })
    return pd.DataFrame(rows)


def _bernoulli_loglik(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_logistic_improved(x: np.ndarray, improved: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(improved, dtype=float)
    n = int(len(y))
    n_win = int(y.sum())
    out = {"n": n, "n_win": n_win, "coef": np.nan, "odds_ratio": np.nan,
           "lrt_pvalue": np.nan, "fitted": False, "x_std": np.nan}
    if n < 3 or n_win == 0 or n_win == n:
        return out
    x_mean = float(x.mean())
    x_std = float(x.std(ddof=0))
    if x_std < 1e-12:
        return out
    z = ((x - x_mean) / x_std).reshape(-1, 1)
    full = LogisticRegression(penalty=None, solver="lbfgs", max_iter=4000)
    full.fit(z, y)
    p_full = full.predict_proba(z)[:, 1]
    ll_full = _bernoulli_loglik(y, p_full)
    ll_null = _bernoulli_loglik(y, np.full(n, y.mean()))
    lrt = max(0.0, -2.0 * (ll_null - ll_full))
    b_std = float(full.coef_.ravel()[0])
    out.update({
        "coef": b_std / x_std,
        "odds_ratio": float(np.exp(b_std / x_std)),
        "lrt_pvalue": float(chi2.sf(lrt, df=1)),
        "fitted": True,
        "x_std": x_std,
    })
    return out


def logistic_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = frame.dropna(subset=["delta_zs_raw"])
    for key, title, domain in MEASURES:
        usable = sub.dropna(subset=[key, "delta_zs_raw"])
        x = usable[key].to_numpy(dtype=float)
        improved = (usable["delta_zs_raw"].to_numpy(dtype=float) < 0.0).astype(float)
        fit = fit_logistic_improved(x, improved)
        or_sd = float(np.exp(fit["coef"] * fit["x_std"])) if fit["fitted"] else np.nan
        rows.append({
            "measure": key, "title": title, "domain": domain, "n": fit["n"],
            "n_win": fit["n_win"], "odds_ratio_per_sd": or_sd,
            "lrt_pvalue": fit["lrt_pvalue"],
        })
    return pd.DataFrame(rows)


def plot_scatter(frame: pd.DataFrame, y_col: str, ylabel: str, out: Path):
    usable_all = frame.dropna(subset=[y_col]).copy()
    labels = sorted(usable_all["label"].unique())
    numbers = {label: i for i, label in enumerate(labels, start=1)}
    llm_color = fs.MODELS["ZS_raw"]["color"]
    xgb_color = fs.MODELS["M2"]["color"]
    n_cols, n_rows = 3, 3

    with plt.rc_context(fs.RC):
        fig = plt.figure(figsize=(fs.WIDTH_DOUBLE, min(fs.MAX_HEIGHT, 50 * fs.MM * n_rows)))
        gs = fig.add_gridspec(n_rows, n_cols)
        axes = [fig.add_subplot(gs[i // n_cols, i % n_cols]) for i in range(len(PLOT_MEASURES))]
        legend_ax = fig.add_subplot(gs[-1, :])

        for index, (ax, key) in enumerate(zip(axes, PLOT_MEASURES)):
            title = dict((k, t) for k, t, _ in MEASURES)[key]
            usable = frame.dropna(subset=[key, y_col]).sort_values("label")
            fs.panel_letter(ax, chr(ord("a") + index), x=-0.18, y=1.08)
            ax.set_xlabel(f"{title}\n({PLOT_DEFINITIONS[key]})", fontsize=6.5)
            if index % n_cols == 0:
                ax.set_ylabel(ylabel, fontsize=6.5)
            if usable.empty:
                ax.set_title("No paired estimates", loc="left", fontsize=6.5)
                continue
            x = usable[key].to_numpy(dtype=float)
            y = usable[y_col].to_numpy(dtype=float)
            if y_col == "delta_zs_raw":
                improved = y < 0.0
                colours = np.where(improved, fs.IMPROVED_COLOR, xgb_color)
            else:
                colours = np.full(len(y), llm_color)
            ax.scatter(x, y, s=42, marker="o", c=colours,
                       edgecolor="white", linewidth=0.6, zorder=3)
            for row in usable.itertuples(index=False):
                ax.text(getattr(row, key), getattr(row, y_col), str(numbers[row.label]),
                        ha="center", va="center", fontsize=5.0, color="white",
                        fontweight="bold", zorder=4)
            if y_col == "delta_zs_raw":
                ax.axhline(0.0, color=fs.REFERENCE, linestyle=(0, (3, 2)),
                           linewidth=fs.REFERENCE_LW, zorder=0)
                ax.set_ylim(-0.22, 0.90)
            else:
                ax.set_ylim(0.08, 1.0)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            if key in UNIT_INTERVAL:
                ax.set_xlim(-0.05, 1.05)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            rho, p = spearmanr(x, y)
            ax.set_title(f"ρ={rho:.2f}, p={p:.3f}, n={len(x)}", loc="left", fontsize=6.5)

        handles = [
            Line2D([], [], marker="o", color="none",
                   markerfacecolor=(
                       fs.IMPROVED_COLOR
                       if y_col == "delta_zs_raw" and float(usable_all.loc[
                           usable_all["label"] == label, y_col].iloc[0]) < 0
                       else (xgb_color if y_col == "delta_zs_raw" else llm_color)
                   ),
                   markeredgecolor="white", markeredgewidth=0.5, markersize=6.0,
                   label=f"{number}  {label}")
            for label, number in numbers.items()
        ]
        legend_ax.set_axis_off()
        legend_ax.legend(
            handles, [h.get_label() for h in handles],
            loc="center", bbox_to_anchor=(0.5, 0.5),
            fontsize=5.2, frameon=False, ncol=5, handletextpad=0.3,
            columnspacing=0.9, labelspacing=0.38, borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.04, top=0.92,
                            wspace=0.34, hspace=0.62)
        out.parent.mkdir(parents=True, exist_ok=True)
        fs.save(fig, out.parent, out.stem)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out_dir = outputs_root(root) / "combined" / "codebook"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset, outcome in PAIRS:
        run = load(root, dataset, outcome)
        row = codebook_row(run)
        pred = run.out / "predictions"
        row["bacc_M2"] = bacc_from_csv(m2_predictions_path(run))
        row["bacc_ZS_raw"] = bacc_from_csv(pred / "ZS_raw.csv")
        m2, zs = row["bacc_M2"], row["bacc_ZS_raw"]
        row["delta_zs_raw"] = (m2 - zs) if m2 is not None and zs is not None else np.nan
        rows.append(row)

    frame = add_composites(pd.DataFrame(rows))
    frame.to_csv(out_dir / "codebook_quality_and_gaps.csv", index=False)

    corr_bacc = spearman_table(frame, "bacc_ZS_raw")
    corr_delta = spearman_table(frame, "delta_zs_raw")
    corr = pd.concat([corr_bacc, corr_delta], ignore_index=True)
    corr.to_csv(out_dir / "codebook_quality_spearman.csv", index=False)

    logit = logistic_table(frame)
    logit.to_csv(out_dir / "codebook_quality_logistic.csv", index=False)

    outliers = {"uci_dermatology", "uci_arrhythmia"}
    sensitivity = pd.concat([
        spearman_table(frame.loc[~frame["dataset"].isin(outliers)].copy(), "bacc_ZS_raw"),
        spearman_table(frame.loc[~frame["dataset"].isin(outliers)].copy(), "delta_zs_raw"),
    ], ignore_index=True)
    sensitivity.to_csv(out_dir / "codebook_quality_spearman_excl_long_prompts.csv", index=False)

    plot_scatter(frame, "bacc_ZS_raw", "System 1 balanced accuracy",
                 out_dir / "codebook_vs_llm_bacc")
    plot_scatter(frame, "delta_zs_raw", "XGB − System 1 balanced accuracy",
                 out_dir / "codebook_vs_delta")

    summary = {
        "metric": "balanced_accuracy",
        "version": 2,
        "fidelity": (
            "per-predictor explanation_rate = mean of 0.5*label_score + 0.5*quantity_score; "
            "not gold-standard medical correctness"
        ),
        "detail": (
            "prompt words per predictor; unique codebook word types minus schema "
            "tokens per predictor; setting + observations_header + question length"
        ),
        "delta": "BalAcc(M2) - BalAcc(ZS_raw)",
        "n_outcomes": int(len(frame)),
        "n_with_zs_raw": int(frame["bacc_ZS_raw"].notna().sum()),
        "prompt_sample": PROMPT_SAMPLE,
        "prompt_seed": PROMPT_SEED,
    }
    (out_dir / "codebook_quality_summary.json").write_text(json.dumps(summary, indent=2))

    show = [
        "label", "n_predictors", "explanation_rate", "label_readability",
        "quantity_explained", "words_per_predictor", "added_per_predictor",
        "context_words", "bacc_ZS_raw", "delta_zs_raw",
    ]
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 180)
    print(frame[show].to_string(index=False))
    print()
    print("Spearman (v2 primary, v1 retained for comparison):")
    print(corr.to_string(index=False))
    print()
    print("Logistic P(System 1 BalAcc > XGB):")
    print(logit.to_string(index=False))
    print()
    print("Spearman excluding Dermatology and Arrhythmia:")
    print(sensitivity.to_string(index=False))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
