#!/usr/bin/env python3
"""Collect the XGB_optuna hold-out arm across tasks: summary table, paired-Δ figure, tuning diagnostics.

Reads ``outputs/<dataset>/<outcome>/holdout/metrics.json`` (primary split) and, when present,
``holdout/sensitivity/seed*/metrics.json`` (extra split seeds). Writes to ``outputs/combined/holdout/``:

  summary.csv                 one row per task: both arms on the identical hold-out patients
                              (balanced accuracy at each arm's rule, AUROC, Brier, log loss), paired
                              Δ with 95% bootstrap CIs, tuning diagnostics (inner-CV best, CV−hold-out
                              gap, trials, consensus-config score), and the previous M2 / ZS_raw figures.
  sensitivity.csv             per task x seed hold-out scores and win flags.
  paired_delta.{pdf,svg,png}  tasks sorted by Δ balanced accuracy with CIs; argmax and prior-matched
                              zero-shot rules side by side; Δ AUROC in the second panel.
  tuning_gap.{pdf,svg,png}    inner-CV best vs hold-out (winner's curse) and best vs consensus config.
  supplementary_table2.md     markdown block for the manuscript.
  summary.json                headline counts.
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
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest.benchmark import ALL_PAIRS, OUTCOME_LABEL
from jevtest.config import outputs_root
from jevtest import figure_style as fs

ARMS = ("XGB_optuna", "XGB_optuna_argmax", "XGB_optuna_consensus", "ZS_raw", "ZS_raw_prior_matched")
METRICS = ("balanced_accuracy", "auroc", "brier", "log_loss")


def exact_binomial(successes, trials, alpha=0.05):
    if trials == 0:
        return np.nan, np.nan
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lo, hi


def previous_scores(root):
    """Old M2 (full table) and ZS_raw out-of-fold balanced accuracy, if the complexity table exists."""
    path = outputs_root(root) / "combined/figures/complexity_and_gaps.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {(r.dataset, r.outcome): (r.bacc_M2, r.bacc_ZS_raw, r.auroc_M2, r.auroc_ZS_raw)
            for r in frame.itertuples(index=False)}


def flatten(metrics, dataset, outcome, previous):
    row = {"dataset": dataset, "outcome": outcome, "label": OUTCOME_LABEL.get(outcome, outcome),
           "seed": metrics["seed"], "n_development": metrics["n_development"], "n_holdout": metrics["n_holdout"],
           "n_classes": metrics["n_classes"], "inner_folds": metrics["inner_folds"],
           "trials_completed": metrics["trials_completed"], "trials_pruned": metrics["trials_pruned"],
           "tuning_minutes": metrics["tuning_seconds"] / 60.0,
           "cv_best_bacc": metrics["best_cv_balanced_accuracy"], "cv_best_sd": metrics["best_cv_sd"],
           "cv_minus_holdout_bacc": metrics["cv_minus_holdout_balanced_accuracy"]}
    for arm in ARMS:
        for metric in METRICS:
            row[f"{arm}__{metric}"] = metrics["holdout"][arm][metric]
    for name, short in (("XGB_optuna_minus_ZS_raw", "delta"), ("XGB_optuna_minus_ZS_raw_prior_matched", "delta_prior")):
        pair = metrics["paired"][name]
        row[f"{short}_bacc"] = pair["delta_bacc"]
        row[f"{short}_bacc_lo"], row[f"{short}_bacc_hi"] = pair["delta_bacc_ci"]
        row[f"{short}_auroc"] = pair["delta_auroc"]
        row[f"{short}_auroc_lo"], row[f"{short}_auroc_hi"] = pair["delta_auroc_ci"]
    best = metrics["best_params"]
    row["class_weight_exponent"] = best.get("class_weight_exponent")
    row["threshold"] = best.get("threshold")
    row["prior_exponent"] = best.get("prior_exponent")
    row["max_depth"] = best.get("max_depth")
    row["n_estimators"] = best.get("n_estimators")
    row["learning_rate"] = best.get("learning_rate")
    row["zs_win"] = row["delta_bacc"] < 0
    row["zs_win_ci"] = row["delta_bacc_hi"] < 0
    row["zs_win_prior_matched"] = row["delta_prior_bacc"] < 0
    row["zs_win_auroc"] = row["delta_auroc"] < 0
    row["zs_noninferior_005"] = row["delta_bacc_hi"] < 0.05
    prev = previous.get((dataset, outcome))
    if prev:
        row["prev_M2_bacc"], row["prev_ZS_bacc"], row["prev_M2_auroc"], row["prev_ZS_auroc"] = prev
        row["prev_delta_bacc"] = prev[0] - prev[1]
    return row


def flatten_full(metrics):
    model = "XGB_optuna_full"
    row = {"n_development_full": metrics["n_development"], "n_full_table": metrics["n_full_table"],
           "n_full_rows_excluded_as_holdout": metrics["n_full_rows_excluded_as_holdout"],
           "full_map_method": metrics["map_method"], "full_trials_completed": metrics["trials_completed"],
           "full_tuning_minutes": metrics["tuning_seconds"] / 60.0, "full_refit_minutes": metrics["refit_seconds"] / 60.0,
           "full_cv_best_bacc": metrics["best_cv_balanced_accuracy"],
           "full_cv_minus_holdout_bacc": metrics["cv_minus_holdout_balanced_accuracy"]}
    for metric in METRICS:
        row[f"{model}__{metric}"] = metrics["holdout"][model][metric]
        row[f"{model}_argmax__{metric}"] = metrics["holdout"][model + "_argmax"][metric]
    pair = metrics["paired"][f"{model}_minus_ZS_raw"]
    row["delta_full_bacc"], (row["delta_full_bacc_lo"], row["delta_full_bacc_hi"]) = pair["delta_bacc"], pair["delta_bacc_ci"]
    row["delta_full_auroc"], (row["delta_full_auroc_lo"], row["delta_full_auroc_hi"]) = pair["delta_auroc"], pair["delta_auroc_ci"]
    gain = metrics["paired"].get(f"{model}_minus_XGB_optuna")
    if gain:
        row["full_minus_subsample_bacc"], (row["full_minus_subsample_bacc_lo"], row["full_minus_subsample_bacc_hi"]) = \
            gain["delta_bacc"], gain["delta_bacc_ci"]
        row["full_minus_subsample_auroc"] = gain["delta_auroc"]
    row["zs_win_full"] = row["delta_full_bacc"] < 0
    row["zs_win_full_auroc"] = row["delta_full_auroc"] < 0
    return row


def collect(root):
    previous = previous_scores(root)
    rows, sens, missing = [], [], []
    for dataset, outcome in ALL_PAIRS:
        folder = outputs_root(root) / dataset / outcome / "holdout"
        path = folder / "metrics.json"
        if not path.exists():
            missing.append(f"{dataset}/{outcome}")
            continue
        row = flatten(json.loads(path.read_text()), dataset, outcome, previous)
        full_path = folder / "full" / "metrics.json"
        if full_path.exists():
            row.update(flatten_full(json.loads(full_path.read_text())))
        rows.append(row)
        for extra in sorted(folder.glob("sensitivity/seed*/metrics.json")):
            sens.append(flatten(json.loads(extra.read_text()), dataset, outcome, previous))
    return pd.DataFrame(rows), pd.DataFrame(sens), missing


def plot_paired_delta(frame, out_dir):
    frame = frame.sort_values("delta_bacc").reset_index(drop=True)
    n = len(frame)
    ys = np.arange(n)
    xgb_c, zs_c, win_c = fs.MODELS["M2"]["color"], fs.MODELS["ZS_raw"]["color"], fs.IMPROVED_COLOR
    height = min(fs.MAX_HEIGHT, (28 + 3.4 * n) * fs.MM)
    with plt.rc_context(fs.RC):
        fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(fs.WIDTH_DOUBLE, height), sharey=True,
                                         gridspec_kw={"width_ratios": [1.35, 1.0]})
        for ax in (ax_b, ax_a):
            for i in range(n):
                if i % 2 == 0:
                    ax.axhspan(i - 0.5, i + 0.5, color="#F2F2F2", zorder=0, linewidth=0)
            ax.axvline(0, color=fs.REFERENCE, linestyle=(0, (3, 2)), linewidth=fs.REFERENCE_LW, zorder=1)
        def intervals(ax, x, lo, hi, colours):
            for xi, l, h, yi, c in zip(x, lo, hi, ys, colours):
                ax.plot([l, h], [yi, yi], color=c, linewidth=0.8, solid_capstyle="butt", zorder=2)
                ax.plot([l, l], [yi - 0.18, yi + 0.18], color=c, linewidth=0.6, zorder=2)
                ax.plot([h, h], [yi - 0.18, yi + 0.18], color=c, linewidth=0.6, zorder=2)

        # panel a: Δ balanced accuracy; zero-shot at its argmax (filled) and prior-matched (open)
        colours = np.where(frame.delta_bacc < 0, win_c, xgb_c)
        intervals(ax_b, frame.delta_bacc, frame.delta_bacc_lo, frame.delta_bacc_hi, colours)
        ax_b.scatter(frame.delta_bacc, ys, s=14, c=colours, edgecolor="white", linewidth=0.4, zorder=3)
        ax_b.scatter(frame.delta_prior_bacc, ys + 0.28, s=12, facecolor="white", edgecolor=zs_c, linewidth=0.7,
                     marker="D", zorder=3)
        ax_b.set_yticks(ys)
        ax_b.set_yticklabels(frame.label, fontsize=6.0)
        ax_b.set_ylim(-0.6, n - 0.4)
        ax_b.set_xlabel("Δ balanced accuracy on the hold-out (XGB_optuna − zero-shot)")
        ax_b.axvline(0.05, color="#BBBBBB", linestyle=(0, (1, 1.5)), linewidth=0.6, zorder=1)
        for ax, lo_col, hi_col in ((ax_b, "delta_bacc_lo", "delta_bacc_hi"), (ax_a, "delta_auroc_lo", "delta_auroc_hi")):
            lo, hi = float(frame[lo_col].min()), float(frame[hi_col].max())
            ax.set_xlim(lo - 0.03, hi + 0.03)
            ax.set_xticks([t for t in np.arange(-0.6, 1.01, 0.2) if lo - 0.03 <= t <= hi + 0.03])
        # panel b: Δ AUROC
        colours_a = np.where(frame.delta_auroc < 0, win_c, xgb_c)
        intervals(ax_a, frame.delta_auroc, frame.delta_auroc_lo, frame.delta_auroc_hi, colours_a)
        ax_a.scatter(frame.delta_auroc, ys, s=14, c=colours_a, edgecolor="white", linewidth=0.4, zorder=3)
        ax_a.set_xlabel("Δ AUROC on the hold-out")
        from matplotlib.lines import Line2D
        handles = [
            Line2D([], [], marker="o", color="none", markerfacecolor=win_c, markeredgecolor="white", markersize=4.5,
                   label="zero-shot better"),
            Line2D([], [], marker="o", color="none", markerfacecolor=xgb_c, markeredgecolor="white", markersize=4.5,
                   label="XGB_optuna better"),
            Line2D([], [], marker="D", color="none", markerfacecolor="white", markeredgecolor=zs_c, markersize=4.0,
                   label="Δ vs zero-shot at a prior-matched rule"),
            Line2D([], [], color="#BBBBBB", linestyle=(0, (1, 1.5)), linewidth=0.8, label="non-inferiority margin 0.05"),
        ]
        fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=2, fontsize=6.2,
                   bbox_to_anchor=(0.6, 1.0), frameon=False, columnspacing=1.2, handletextpad=0.4)
        bottom = 0.11 if n < 20 else 0.06
        top = 0.90 if n < 20 else 0.94
        fig.subplots_adjust(left=0.22, right=0.97, bottom=bottom, top=top, wspace=0.08)
        fig.text(0.005, top + 0.005, "a", fontsize=fs.LETTER_PT, fontweight="bold", va="bottom", ha="left")
        fig.text(ax_a.get_position().x0 - 0.02, top + 0.005, "b", fontsize=fs.LETTER_PT, fontweight="bold",
                 va="bottom", ha="left")
        fs.save(fig, out_dir, "paired_delta")


def plot_tuning_gap(frame, out_dir):
    with plt.rc_context(fs.RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.WIDTH_DOUBLE, 70 * fs.MM))
        c = fs.MODELS["M2"]["color"]
        lo, hi = 0.3, 1.02
        for ax, x, y, xl, yl in (
            (ax1, frame.cv_best_bacc, frame.XGB_optuna__balanced_accuracy,
             "Best inner-CV balanced accuracy", "Hold-out balanced accuracy (best trial)"),
            (ax2, frame.XGB_optuna__balanced_accuracy, frame.XGB_optuna_consensus__balanced_accuracy,
             "Hold-out balanced accuracy (best trial)", "Hold-out balanced accuracy (top-5 consensus)"),
        ):
            fs.reference_line(ax, [lo, hi], [lo, hi])
            ax.scatter(x, y, s=16, c=c, edgecolor="white", linewidth=0.4, zorder=3)
            ticks = [0.4, 0.6, 0.8, 1.0]
            ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=xl, ylabel=yl, xticks=ticks, yticks=ticks)
            ax.set_aspect("equal")
        gap = frame.cv_minus_holdout_bacc
        ax1.set_title(f"CV − hold-out: median {gap.median():+.3f} (IQR {gap.quantile(.25):+.3f} to {gap.quantile(.75):+.3f})",
                      loc="left", fontsize=6.5)
        d = frame.XGB_optuna__balanced_accuracy - frame.XGB_optuna_consensus__balanced_accuracy
        ax2.set_title(f"best − consensus: median {d.median():+.3f}", loc="left", fontsize=6.5)
        fs.panel_letter(ax1, "a", x=-0.18, y=1.02)
        fs.panel_letter(ax2, "b", x=-0.18, y=1.02)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.9, wspace=0.35)
        fs.save(fig, out_dir, "tuning_gap")


def supplementary_table(frame, sens):
    lines = [
        "**Supplementary Table 2. Maximally tuned XGBoost against the zero-shot readout on an identical hold-out.** "
        "For every task a stratified 20% hold-out was drawn once (seed 20261008) from the patients the readout scored. "
        "On the remaining 80%, Optuna (TPE, {t} trials, median pruning) maximised 5-fold inner-CV balanced accuracy over "
        "tree hyperparameters, a class-weight exponent and the decision rule (binary threshold; multiclass prior exponent). "
        "The best trial was refit on the 80% and applied once to the hold-out. Zero-shot columns are AnyJev L0 probabilities "
        "for the same hold-out patients, labelled by argmax (ZS) and by argmax after dividing by the development-set class "
        "prior (ZS-prior). Δ = XGBoost − zero-shot with a 1,000-draw paired patient bootstrap; asterisks mark zero-shot wins "
        "on balanced accuracy. CV−HO is the best inner-CV score minus the hold-out score (optimism).".format(
            t=int(frame.trials_completed.max() + frame.trials_pruned.max())),
        "",
        "| Task | *n* hold-out | Classes | XGB BA | ZS BA | ZS-prior BA | Δ BA (95% CI) | XGB AUROC | ZS AUROC | Δ AUROC (95% CI) | CV−HO | *w* | rule |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in frame.itertuples(index=False):
        star = "*" if r.zs_win else ""
        rule = f"τ={r.threshold:.2f}" if np.isfinite(r.threshold or np.nan) else f"α={r.prior_exponent:.2f}"
        lines.append(
            f"| {r.label}{star} | {r.n_holdout:,} | {r.n_classes} | {r.XGB_optuna__balanced_accuracy:.3f} | "
            f"{r.ZS_raw__balanced_accuracy:.3f} | {r.ZS_raw_prior_matched__balanced_accuracy:.3f} | "
            f"{r.delta_bacc:+.3f} ({r.delta_bacc_lo:+.3f}, {r.delta_bacc_hi:+.3f}) | "
            f"{r.XGB_optuna__auroc:.3f} | {r.ZS_raw__auroc:.3f} | "
            f"{r.delta_auroc:+.3f} ({r.delta_auroc_lo:+.3f}, {r.delta_auroc_hi:+.3f}) | "
            f"{r.cv_minus_holdout_bacc:+.3f} | {r.class_weight_exponent:.2f} | {rule} |"
        )
    if len(sens):
        per_task = sens.groupby("label").agg(seeds=("seed", "count"), zs_wins=("zs_win", "sum"),
                                             median_delta=("delta_bacc", "median")).reset_index()
        lines += ["", f"Sensitivity splits ({int(per_task.seeds.max())} extra seeds, reduced budget): zero-shot won on "
                      f"{int(per_task.zs_wins.sum())} of {int(per_task.seeds.sum())} task-seed pairs; tasks with a zero-shot "
                      f"win on a majority of seeds: "
                      + (", ".join(per_task.loc[per_task.zs_wins > per_task.seeds / 2, "label"]) or "none") + "."]
    if "delta_full_bacc" in frame and frame.delta_full_bacc.notna().any():
        sub = frame.loc[frame.delta_full_bacc.notna()].sort_values("delta_full_bacc")
        lines += [
            "",
            "**Supplementary Table 2b. Whole-table training on the same hold-out.** For the tables whose readout "
            "arm scored a 1,000-patient subsample, XGB_optuna_full was tuned (Optuna on a stratified cap of 50,000 "
            "development rows) and refit on every row of the whole table except the hold-out patients and any "
            "duplicate copies of them (`readout_to_full.json`). Columns as above; the last column is the paired gain "
            "of whole-table over 80%-of-subsample training on the identical hold-out.",
            "",
            "| Task | *n* trained | XGB-full BA | ZS BA | Δ BA (95% CI) | XGB-full AUROC | ZS AUROC | Δ AUROC (95% CI) | full − subsample BA (95% CI) |",
            "|---|---:|---:|---:|---|---:|---:|---|---|",
        ]
        for r in sub.itertuples(index=False):
            star = "*" if r.zs_win_full else ""
            gain = (f"{r.full_minus_subsample_bacc:+.3f} ({r.full_minus_subsample_bacc_lo:+.3f}, {r.full_minus_subsample_bacc_hi:+.3f})"
                    if "full_minus_subsample_bacc" in frame and np.isfinite(r.full_minus_subsample_bacc) else "—")
            lines.append(
                f"| {r.label}{star} | {int(r.n_development_full):,} | {r.XGB_optuna_full__balanced_accuracy:.3f} | "
                f"{r.ZS_raw__balanced_accuracy:.3f} | {r.delta_full_bacc:+.3f} ({r.delta_full_bacc_lo:+.3f}, {r.delta_full_bacc_hi:+.3f}) | "
                f"{r.XGB_optuna_full__auroc:.3f} | {r.ZS_raw__auroc:.3f} | "
                f"{r.delta_full_auroc:+.3f} ({r.delta_full_auroc_lo:+.3f}, {r.delta_full_auroc_hi:+.3f}) | {gain} |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out_dir = outputs_root(root) / "combined" / "holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, sens, missing = collect(root)
    if frame.empty:
        print("No hold-out metrics found; run scripts/queue_xgb_holdout.sh first.")
        return 1
    frame.to_csv(out_dir / "summary.csv", index=False)
    if len(sens):
        sens.to_csv(out_dir / "sensitivity.csv", index=False)
    plot_paired_delta(frame, out_dir)
    plot_tuning_gap(frame, out_dir)
    (out_dir / "supplementary_table2.md").write_text(supplementary_table(frame, sens))

    n = len(frame)
    wins = int(frame.zs_win.sum())
    summary = {
        "n_tasks": n, "missing": missing,
        "zs_wins_bacc": wins, "zs_wins_bacc_exact_ci": exact_binomial(wins, n),
        "zs_wins_bacc_ci_excludes_zero": int(frame.zs_win_ci.sum()),
        "zs_wins_bacc_prior_matched": int(frame.zs_win_prior_matched.sum()),
        "zs_wins_auroc": int(frame.zs_win_auroc.sum()),
        "zs_noninferior_margin_005": int(frame.zs_noninferior_005.sum()),
        "median_bacc": {"XGB_optuna": float(frame.XGB_optuna__balanced_accuracy.median()),
                        "XGB_optuna_argmax": float(frame.XGB_optuna_argmax__balanced_accuracy.median()),
                        "ZS_raw": float(frame.ZS_raw__balanced_accuracy.median()),
                        "ZS_raw_prior_matched": float(frame.ZS_raw_prior_matched__balanced_accuracy.median())},
        "median_auroc": {"XGB_optuna": float(frame.XGB_optuna__auroc.median()), "ZS_raw": float(frame.ZS_raw__auroc.median())},
        "median_delta_bacc": float(frame.delta_bacc.median()),
        "median_cv_minus_holdout_bacc": float(frame.cv_minus_holdout_bacc.median()),
        "median_best_minus_consensus_bacc": float((frame.XGB_optuna__balanced_accuracy - frame.XGB_optuna_consensus__balanced_accuracy).median()),
        "median_tuning_minutes": float(frame.tuning_minutes.median()),
        "zs_win_tasks": frame.loc[frame.zs_win, "label"].tolist(),
        "zs_win_tasks_prior_matched": frame.loc[frame.zs_win_prior_matched, "label"].tolist(),
    }
    if "prev_delta_bacc" in frame:
        summary["previous_protocol"] = {"zs_wins_bacc": int((frame.prev_delta_bacc < 0).sum()),
                                        "median_M2_bacc": float(frame.prev_M2_bacc.median()),
                                        "median_ZS_bacc": float(frame.prev_ZS_bacc.median())}
    if len(sens):
        summary["sensitivity"] = {"task_seed_pairs": int(len(sens)), "zs_wins": int(sens.zs_win.sum()),
                                  "tasks_with_majority_wins": sens.groupby("label").zs_win.mean().loc[lambda s: s > 0.5].index.tolist()}
    if "delta_full_bacc" in frame and frame.delta_full_bacc.notna().any():
        sub = frame.loc[frame.delta_full_bacc.notna()]
        summary["full_table_arm"] = {
            "n_tasks": int(len(sub)), "zs_wins_bacc": int(sub.zs_win_full.sum()), "zs_wins_auroc": int(sub.zs_win_full_auroc.sum()),
            "zs_win_tasks": sub.loc[sub.zs_win_full, "label"].tolist(),
            "median_full_minus_subsample_bacc": float(sub.full_minus_subsample_bacc.median()) if "full_minus_subsample_bacc" in sub else None,
            "median_full_minus_subsample_auroc": float(sub.full_minus_subsample_auroc.median()) if "full_minus_subsample_auroc" in sub else None,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 250)
    show = ["label", "n_holdout", "XGB_optuna__balanced_accuracy", "XGB_optuna_argmax__balanced_accuracy",
            "ZS_raw__balanced_accuracy", "ZS_raw_prior_matched__balanced_accuracy", "delta_bacc", "delta_bacc_lo",
            "delta_bacc_hi", "XGB_optuna__auroc", "ZS_raw__auroc", "cv_minus_holdout_bacc", "class_weight_exponent"]
    print(frame.sort_values("delta_bacc")[show].round(3).to_string(index=False))
    print()
    print(json.dumps(summary, indent=2, default=float))
    if missing:
        print(f"\nMissing hold-out runs: {len(missing)}: {missing}")
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
