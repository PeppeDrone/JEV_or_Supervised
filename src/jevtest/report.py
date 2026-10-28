"""performance.csv, paired_vs_M2.csv and REPORT.md for one dataset x outcome."""
import numpy as np
import pandas as pd

from jevtest.config import read_json, save_table
from jevtest.metrics import KEYS, bootstrap, interval, point, resamples

MODELS = ["M2", "ZS_raw", "ICL_full", "Mode", "Mean"]
DESCRIPTION = {
    "M2": "XGBoost (native missing values)",
    "ZS_raw": "System 1 (zero-shot AnyJev L0, content-free prior)",
    "ICL_full": "Training patients as labelled rows in the prompt",
    "Mode": "Majority vote of ensemble members (vote shares)",
    "Mean": "Equal-weight mean of ensemble members' posteriors",
}
REFERENCE = "M2"


def fmt(v, digits=3):
    return "" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def summarise(run):
    found = {m: P for m in MODELS if (P := run.predictions(m)) is not None}
    if not found:
        raise SystemExit("No predictions yet; run the model stages first")
    y = run.y
    labels = {"Mode": run.prediction_labels("Mode")} if "Mode" in found else {}
    draws = resamples(len(y), run.study["bootstrap_iterations"], run.study["seed"])
    perf, samples = [], {}
    for m, P in found.items():
        probabilistic = m != "Mode"
        values = point(y, P, labels.get(m), probabilistic)
        samples[m] = bootstrap(y, P, draws, labels.get(m), probabilistic)
        row = {"model": m}
        for key, v in values.items():
            row[key] = v
            if key in KEYS:
                row[f"{key}_lower"], row[f"{key}_upper"] = interval(samples[m][key])
        perf.append(row)
    perf = pd.DataFrame(perf)
    paired = []
    if REFERENCE in found:
        for m in found:
            if m == REFERENCE:
                continue
            for key in KEYS:
                d = samples[m][key] - samples[REFERENCE][key]
                d = d[np.isfinite(d)]
                if not len(d):
                    continue
                lower, upper = interval(d)
                paired.append({
                    "model": m, "metric": key,
                    "delta_vs_M2": (
                        perf.set_index("model").loc[m, key]
                        - perf.set_index("model").loc[REFERENCE, key]
                    ),
                    "ci_lower": lower, "ci_upper": upper,
                    "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
                })
    paired = pd.DataFrame(paired)
    save_table(perf, run.out / "performance.csv")
    save_table(paired, run.out / "paired_vs_M2.csv")
    write_report(run, perf, paired)
    return perf, paired


def write_report(run, perf, paired):
    counts = np.bincount(run.y, minlength=len(run.classes))
    lines = [
        f"# {run.dataset} / {run.outcome}", "",
        f"{len(run.y)} patients; classes " + ", ".join(f"{c} {n}" for c, n in zip(run.classes, counts))
        + f"; {len(run.splits)} stratified outer folds (seed {run.study['seed']}). Every fitted step uses outer-training "
        f"patients only. Backbone {run.study['backbone']['model_id']} @ {run.study['backbone']['revision'][:10]}.", "",
        "| Model | What | AUROC (95% CI) | Brier | Log loss | Bal. acc. | Macro F1 | Cal. slope | ΔAUROC vs XGBoost (95% CI) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    deltas = paired[paired.metric == "auroc"].set_index("model") if len(paired) else pd.DataFrame()
    for _, r in perf.iterrows():
        d = deltas.loc[r.model] if r.model in deltas.index else None
        delta = "" if d is None else f"{d.delta_vs_M2:+.3f} ({d.ci_lower:+.3f} to {d.ci_upper:+.3f})"
        lines.append(f"| {r.model} | {DESCRIPTION[r.model]} | {fmt(r.auroc)} ({fmt(r.auroc_lower)}–{fmt(r.auroc_upper)}) | "
                     f"{fmt(r.brier)} | {fmt(r.log_loss)} | {fmt(r.balanced_accuracy)} | {fmt(r.macro_f1)} | "
                     f"{fmt(r.calibration_slope, 2)} | {delta} |")
    missing = [m for m in MODELS if m not in set(perf.model)]
    if missing:
        lines += ["", f"Not yet available: {', '.join(missing)}."]
    lines += ["", f"Ensemble members (equal weights): {', '.join(run.study['ensemble']['members'])}. "
              "Mode AUROC uses vote shares; ties go to the class with the highest mean posterior. "
              + ("AUROC is macro one-vs-rest. " if len(run.classes) > 2 else "")
              + "CIs: patient bootstrap of the out-of-fold predictions "
              f"({run.study['bootstrap_iterations']} resamples, conditional on the fitted folds)."]
    if (run.out / "context.json").exists():
        c = read_json(run.out / "context.json")
        lines += ["", "## ICL-full context", "",
                  f"Limit {c['context_limit']} tokens, budget {c['budget']}, longest query {c['max_query_tokens']} tokens. "
                  + ("Every training patient fits in every fold." if c["mode"] == "all" else
                     f"Representative class-stratified subsample of n = {c['n']} training patients per fold "
                     "(patients nearest k-means centres within each class)."), "",
                  "| Fold | Training patients | In prompt | Per class | Max prompt tokens | Max abs SMD | Max missingness diff |",
                  "|---|---|---|---|---|---|---|"]
        for f in c["folds"]:
            lines.append(f"| {f['fold']} | {f['n_available']} | {f['n_selected']} | {f['class_counts']} | "
                         f"{f['max_prompt_tokens']} | {fmt(f['max_abs_smd'], 2)} | {fmt(f['max_missingness_difference'], 2)} |")
    (run.out / "REPORT.md").write_text("\n".join(lines) + "\n")
