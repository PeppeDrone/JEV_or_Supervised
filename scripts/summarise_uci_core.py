#!/usr/bin/env python3
"""Build outputs/uci_core_benchmark_summary.{md,csv} from provenance-validated predictions.

Uses the repository metric functions only (no new evaluation methodology). Reports the
five requested models when available; does not invent missing GPU predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest.benchmark import ALL_PAIRS as PAIRS
from jevtest.data import full_run, load
from jevtest.metrics import KEYS, bootstrap, interval, point, resamples
MODELS = ["M2", "ZS_raw"]
REFERENCE = "M2"


def auprc(y, P):
    if P.shape[1] != 2 or len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, P[:, 1]))


def evaluate(run, model, draws):
    P = run.predictions(model)
    if P is None:
        return None
    values = point(run.y, P)
    samples = bootstrap(run.y, P, draws)
    row = {
        "dataset": run.dataset,
        "outcome": run.outcome,
        "n_observations": len(run.y),
        "n_predictors": len(run.ds["predictors"]),
        "n_classes": len(run.classes),
        "model": model,
        "accuracy": float(accuracy_score(run.y, P.argmax(axis=1))),
        "auprc": auprc(run.y, P),
    }
    for key, value in values.items():
        row[key] = value
        if key in KEYS:
            row[f"{key}_lower"], row[f"{key}_upper"] = interval(samples[key])
    return row, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root)

    rows, by_pair = [], {}
    for dataset, outcome in PAIRS:
        run = load(root, dataset, outcome)
        found = {}
        for model in MODELS:
            try:
                # XGBoost is scored on the whole table where one was prepared; zero-shot on the subsample.
                scored = (full_run(root, dataset, outcome) if model == "M2" else None) or run
                draws = resamples(len(scored.y), scored.study["bootstrap_iterations"], scored.study["seed"])
                result = evaluate(scored, model, draws)
            except Exception as error:  # provenance / load errors must surface
                found[model] = {"error": str(error)}
                continue
            if result is None:
                found[model] = None
                continue
            row, samples = result
            rows.append(row)
            found[model] = {"row": row, "samples": samples, "P": scored.predictions(model)}
        by_pair[(dataset, outcome)] = {"run": run, "found": found}

    table = pd.DataFrame(rows)
    out_csv = root / "outputs/uci_core_benchmark_summary.csv"
    out_md = root / "outputs/uci_core_benchmark_summary.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    lines = [
        "# UCI core benchmark summary",
        "",
        "Models: M2 (XGBoost), ZS_raw (System 1).",
        "Same stratified 5-fold outer CV (seed 20261008) as the existing NeuroJev benchmark.",
        "Metrics use the repository `jevtest.metrics` definitions; multiclass AUROC is macro one-vs-rest.",
        "Bootstrap CIs are patient-level resamples of out-of-fold predictions (conditional on fitted folds).",
        "Differences vs M2 (XGBoost) use paired resamples when both arms score the same patients and independent resamples when XGBoost was scored on the whole table; intervals that include zero are not treated as supported improvements.",
        "",
    ]

    if len(table):
        show = table[[
            "dataset", "outcome", "n_observations", "n_predictors", "n_classes", "model",
            "auroc", "auroc_lower", "auroc_upper", "auprc", "log_loss", "brier", "accuracy",
            "balanced_accuracy", "macro_f1",
        ]].copy()
        lines += ["## Metrics", "", show.to_markdown(index=False), ""]

    lines += ["## Per-dataset notes", ""]
    for (dataset, outcome), info in by_pair.items():
        run, found = info["run"], info["found"]
        lines.append(f"### {dataset} / {outcome}")
        lines.append(
            f"{len(run.y)} observations, {len(run.ds['predictors'])} predictors, "
            f"{len(run.classes)} classes ({', '.join(run.classes)})."
        )
        available = [m for m in MODELS if isinstance(found.get(m), dict) and "row" in found[m]]
        missing = [m for m in MODELS if m not in available]
        if missing:
            details = []
            for m in missing:
                item = found.get(m)
                if item is None:
                    details.append(f"{m} (predictions absent)")
                elif isinstance(item, dict) and "error" in item:
                    details.append(f"{m} ({item['error']})")
                else:
                    details.append(m)
            lines.append("Unavailable: " + "; ".join(details) + ".")
        if not available:
            lines += ["", ""]
            continue

        # Best by AUROC among available models
        aurocs = {m: found[m]["row"]["auroc"] for m in available}
        best = max(aurocs, key=aurocs.get)
        lines.append(f"Highest observed AUROC: **{best}** ({aurocs[best]:.3f}).")

        def paired(a, b, metric="auroc"):
            if a not in available or b not in available:
                return None
            d = found[a]["samples"][metric] - found[b]["samples"][metric]
            d = d[np.isfinite(d)]
            if not len(d):
                return None
            lo, hi = interval(d)
            point_delta = found[a]["row"][metric] - found[b]["row"][metric]
            return point_delta, lo, hi

        if REFERENCE in available:
            for m in available:
                if m == REFERENCE:
                    continue
                cmp = paired(m, REFERENCE)
                if cmp is None:
                    continue
                d, lo, hi = cmp
                supported = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"
                lines.append(f"{m} − M2 AUROC = {d:+.3f} ({lo:+.3f} to {hi:+.3f}); {supported}.")
        lines += ["", ""]

    lines += [
        "## Binary vs multiclass",
        "",
        "- Binary tasks (Cleveland, CKD, mammographic mass): AUROC is the positive-class ROC; AUPRC is reported.",
        "- Multiclass tasks (dermatology, HCV): AUROC is macro one-vs-rest; AUPRC is left blank.",
        "- Do not pool AUROC across unrelated datasets.",
        "",
        f"CSV: `{out_csv.relative_to(root)}`",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    print(f"{len(table)} model rows across {len(by_pair)} datasets")


if __name__ == "__main__":
    main()
