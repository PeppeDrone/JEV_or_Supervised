#!/usr/bin/env python3
"""Sensitivity of the zero-shot readout to the AnyJev L0 label prior, task by task on the shared hold-out.

Compares two readouts of the same backbone (default: the reported Qwen3.5-27B run with the content-free prior
divided out, against the run with the prior switched off) on every task finished in both, plus tuned XGB.
Safe to run while the alternative is still being computed: only tasks with predictions/ZS_raw_holdout.csv count,
and the in-flight task is reported from progress.json.

Writes <out>/prior_sensitivity.csv and <out>/prior_sensitivity.png. Nothing here feeds the manuscript pipeline
(reports/backbone_ladder and outputs_qwen3.5_* are untouched).

  .venv/bin/python scripts/compare_prior_sensitivity.py
  .venv/bin/python scripts/compare_prior_sensitivity.py --alt outputs_sensitivity/qwen3.5_27b_prior_batch --alt-label "batch prior"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from jevtest.benchmark import ALL_PAIRS  # noqa: E402
from plot_backbone_ladder import fast_auc, fast_bacc, metrics  # noqa: E402
from task_names import task_name  # noqa: E402


def read_pred(path: Path):
    t = pd.read_csv(path, dtype={"patient_id": str}).set_index("patient_id")
    cols = [c for c in t.columns if c.startswith("p__")]
    return t, cols


def paired_dbacc_ci(y, lab_a, lab_b, K, iterations=1000, seed=20261008):
    rng = np.random.default_rng(seed)
    d = np.empty(iterations)
    n = len(y)
    for b in range(iterations):
        idx = rng.integers(0, n, n)
        d[b] = fast_bacc(y[idx], lab_a[idx], K) - fast_bacc(y[idx], lab_b[idx], K)
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def collect(ref: Path, alt: Path, xgb_base: Path):
    rows, pending = [], []
    for dataset, outcome in ALL_PAIRS:
        xgb_path = xgb_base / dataset / outcome / "holdout" / "XGB_optuna.csv"
        f_ref = ref / dataset / outcome / "predictions" / "ZS_raw_holdout.csv"
        f_alt = alt / dataset / outcome / "predictions" / "ZS_raw_holdout.csv"
        prog = alt / dataset / outcome / "progress.json"
        if not f_alt.exists():
            if prog.exists():
                try:
                    p = json.loads(prog.read_text())
                    pending.append((dataset, outcome, p.get("completed"), p.get("patients")))
                except (OSError, ValueError):
                    pending.append((dataset, outcome, None, None))
            continue
        if not (xgb_path.exists() and f_ref.exists()):
            continue
        xgb, xcols = read_pred(xgb_path)
        zr, _ = read_pred(f_ref)
        za, _ = read_pred(f_alt)
        if set(za.index) != set(xgb.index) or set(zr.index) != set(xgb.index):
            print(f"warning: {dataset}/{outcome} hold-out patients differ between arms; skipped", file=sys.stderr)
            continue
        zr, za = zr.loc[xgb.index], za.loc[xgb.index]
        y = xgb["observed"].to_numpy()
        K = len(xcols)
        P_x = xgb[xcols].to_numpy(float)
        P_r = zr[[c for c in zr.columns if c.startswith("p__")]].to_numpy(float)
        P_a = za[[c for c in za.columns if c.startswith("p__")]].to_numpy(float)
        lab_r, lab_a = P_r.argmax(axis=1), P_a.argmax(axis=1)
        row = {"dataset": dataset, "outcome": outcome, "task": task_name(dataset), "n": len(y), "n_classes": K}
        row["auc_XGB"], row["bacc_XGB"] = metrics(y, P_x, xgb["predicted"].to_numpy())
        row["auc_ref"], row["bacc_ref"] = metrics(y, P_r)
        row["auc_alt"], row["bacc_alt"] = metrics(y, P_a)
        row["dbacc_alt_ref"] = row["bacc_alt"] - row["bacc_ref"]
        row["dbacc_lo"], row["dbacc_hi"] = paired_dbacc_ci(y, lab_a, lab_r, K)
        row["dauc_alt_ref"] = row["auc_alt"] - row["auc_ref"]
        row["argmax_changed"] = float((lab_a != lab_r).mean())
        row["dbacc_ref_xgb"] = row["bacc_ref"] - row["bacc_XGB"]
        row["dbacc_alt_xgb"] = row["bacc_alt"] - row["bacc_XGB"]
        rows.append(row)
    return pd.DataFrame(rows), pending


def summarise(t: pd.DataFrame, ref_label: str, alt_label: str):
    from scipy.stats import wilcoxon

    def block(sub, name):
        if sub.empty:
            return
        d = sub["dbacc_alt_ref"].to_numpy()
        p = wilcoxon(d).pvalue if (len(d) >= 6 and np.any(d != 0)) else np.nan
        print(f"  {name:<11} n={len(sub):>2}  median BA  {ref_label}: {sub.bacc_ref.median():.3f}   "
              f"{alt_label}: {sub.bacc_alt.median():.3f}   XGB: {sub.bacc_XGB.median():.3f}   "
              f"median ΔBA({alt_label} − {ref_label}) {np.median(d):+.3f}  "
              f"[IQR {np.percentile(d, 25):+.3f}, {np.percentile(d, 75):+.3f}]  Wilcoxon P={p:.3f}  "
              f"better {int((d > 0).sum())} / worse {int((d < 0).sum())}   "
              f"median AUROC {ref_label} {sub.auc_ref.median():.3f} vs {alt_label} {sub.auc_alt.median():.3f}")

    block(t, "all")
    block(t[t.n_classes == 2], "binary")
    block(t[t.n_classes > 2], "multiclass")


def figure(t: pd.DataFrame, out: Path, ref_label: str, alt_label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    binary = t.n_classes == 2
    c_bin, c_multi = "#1F5F8B", "#F2A65A"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    for m, c, lab in [(binary, c_bin, "binary"), (~binary, c_multi, "multiclass")]:
        ax.scatter(t.bacc_ref[m], t.bacc_alt[m], s=18 + 60 * t.n[m] / t.n.max(), c=c, alpha=0.75, label=lab, edgecolor="white", lw=0.4)
    lo = min(t.bacc_ref.min(), t.bacc_alt.min()) - 0.03
    ax.set_xlim(lo, 1.0); ax.set_ylim(lo, 1.0)
    ax.set_xlabel(f"BA, {ref_label}"); ax.set_ylabel(f"BA, {alt_label}")
    ax.set_title(f"a  Balanced accuracy, {len(t)} tasks", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    for m, c in [(binary, c_bin), (~binary, c_multi)]:
        ax.scatter(t.auc_ref[m], t.auc_alt[m], s=18 + 60 * t.n[m] / t.n.max(), c=c, alpha=0.75, edgecolor="white", lw=0.4)
    lo = min(t.auc_ref.min(), t.auc_alt.min()) - 0.03
    ax.set_xlim(lo, 1.0); ax.set_ylim(lo, 1.0)
    ax.set_xlabel(f"AUROC, {ref_label}"); ax.set_ylabel(f"AUROC, {alt_label}")
    ax.set_title("b  Ranking", loc="left", fontsize=10)

    ax = axes[2]
    s = t.sort_values("dbacc_alt_ref").reset_index(drop=True)
    ypos = np.arange(len(s))
    cols = np.where(s.n_classes == 2, c_bin, c_multi)
    ax.axvline(0, ls="--", c="grey", lw=0.8)
    ax.hlines(ypos, s.dbacc_lo, s.dbacc_hi, color=cols, lw=1.0, alpha=0.6)
    ax.scatter(s.dbacc_alt_ref, ypos, c=cols, s=16, zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels([f"{n}, {k}" for n, k in zip(s.task, s.n)], fontsize=6)
    ax.set_xlabel(f"ΔBA, {alt_label} − {ref_label} (95% paired bootstrap)")
    med = float(np.median(s.dbacc_alt_ref))
    ax.set_title(f"c  Per task; median {med:+.3f}", loc="left", fontsize=10)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right"):
        for a in axes:
            a.spines[sp].set_visible(False)
    fig.suptitle(f"AnyJev L0 label-prior sensitivity, Qwen3.5-27B, shared hold-out (interim: {len(t)}/45 tasks)", fontsize=10, y=1.0)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "prior_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="outputs_qwen3.5_27b", help="reference readout dir (reported run)")
    ap.add_argument("--alt", default="outputs_sensitivity/qwen3.5_27b_prior_none", help="alternative readout dir")
    ap.add_argument("--ref-label", default="content-free prior")
    ap.add_argument("--alt-label", default="prior off")
    ap.add_argument("--xgb", default="outputs", help="base outputs dir holding <task>/holdout/XGB_optuna.csv")
    ap.add_argument("--out", default="reports/sensitivity_prior_27b")
    a = ap.parse_args()

    t, pending = collect(ROOT / a.ref, ROOT / a.alt, ROOT / a.xgb)
    out = ROOT / a.out
    print(f"{a.alt}: {len(t)} / {len(ALL_PAIRS)} tasks finished and comparable")
    for dataset, outcome, done, total in pending:
        print(f"  in flight: {dataset}/{outcome}  {done}/{total} patients")
    if t.empty:
        return
    summarise(t, a.ref_label, a.alt_label)
    out.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(out / "prior_sensitivity.csv", index=False)
    figure(t, out, a.ref_label, a.alt_label)
    print(f"wrote {out}/prior_sensitivity.{{csv,png}}")


if __name__ == "__main__":
    main()
