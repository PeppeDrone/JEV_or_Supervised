"""XGB_optuna: a maximally tuned XGBoost arm on a fixed hold-out shared with the zero-shot readout.

Protocol (identical for every task):
  1. One stratified hold-out (``holdout.fraction``) is drawn from the patients the zero-shot
     arm scored, with the study seed, and frozen in ``holdout/split.json``.
  2. On the remaining development patients, Optuna (TPE, median pruner) maximises the mean
     inner stratified k-fold balanced accuracy. Each trial owns its tree hyperparameters,
     a class-weight exponent (sample weight of class c = (n / (K n_c)) ** w) and its decision
     rule (binary: threshold on P(class 1); multiclass: argmax P / prior ** alpha, prior from
     the fold's training part), so the CV score already includes the operating point.
  3. The best trial is refit on all development patients and applied once to the hold-out.
     The zero-shot L0 probabilities of the same hold-out patients are read from the validated
     ``ZS_raw`` artefact; both arms are scored with the same patient bootstrap.

Nothing from the hold-out is read during tuning. ``holdout/`` holds the split, every trial,
the chosen settings, both arms' hold-out predictions, metrics and the refit pipeline.
"""
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from jevtest.classical import preprocessing
from jevtest.config import GateError, digest, read_json, write_json
from jevtest.data import predictor_matrix
from jevtest.metrics import auroc, brier, point
from jevtest.models import versions

MODEL = "XGB_optuna"
LOG_UNIFORM = {"learning_rate", "min_child_weight", "reg_alpha", "reg_lambda"}
INT_PARAMS = {"max_depth", "n_estimators"}
CATEGORICAL = {"max_bin", "grow_policy"}
TREE_PARAMS = ("max_depth", "learning_rate", "n_estimators", "min_child_weight", "subsample",
               "colsample_bytree", "colsample_bylevel", "gamma", "reg_alpha", "reg_lambda",
               "max_bin", "grow_policy")


# ----------------------------------------------------------------------------- split and folds

def holdout_split(y, fraction, seed):
    """Sorted development / hold-out row indices; stratified on the outcome."""
    dev, hold = train_test_split(np.arange(len(y)), test_size=fraction, stratify=y, random_state=seed)
    return np.sort(dev), np.sort(hold)


def inner_folds(y, k, seed):
    k = min(int(k), int(np.bincount(y).min()))
    if k < 2:
        raise GateError("Insufficient class counts for inner stratified folds on the development set")
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


# ----------------------------------------------------------------------------- weights and decisions

def class_weights(y, exponent, n_classes):
    """Per-sample weight (n / (K n_c)) ** exponent; exponent 0 = unweighted, 1 = fully balanced."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    per_class = (len(y) / (n_classes * counts)) ** float(exponent)
    return per_class[y]


def class_prior(y, n_classes):
    prior = np.bincount(y, minlength=n_classes).astype(float)
    prior = np.clip(prior, 1.0, None)
    return prior / prior.sum()


def decide(P, prior, threshold=None, prior_exponent=None):
    """Labels from probabilities. Binary: P(class 1) >= threshold. Multiclass: argmax P / prior ** alpha."""
    P = np.asarray(P, dtype=float)
    if P.shape[1] == 2:
        tau = 0.5 if threshold is None else float(threshold)
        return (P[:, 1] >= tau).astype(int)
    alpha = 0.0 if prior_exponent is None else float(prior_exponent)
    return np.argmax(P / np.asarray(prior, dtype=float) ** alpha, axis=1)


def rule_from(params, n_classes):
    if n_classes == 2:
        return {"threshold": float(params["threshold"])}
    return {"prior_exponent": float(params["prior_exponent"])}


# ----------------------------------------------------------------------------- model

def make_pipeline(X, params, seed, threads):
    tree = {k: params[k] for k in TREE_PARAMS}
    model = XGBClassifier(random_state=seed, n_jobs=threads, tree_method="hist", eval_metric="mlogloss",
                          verbosity=0, **tree)
    return Pipeline([("preprocess", preprocessing(X, "M2")), ("model", model)])


def fit(X, y, params, n_classes, seed, threads):
    pipe = make_pipeline(X, params, seed, threads)
    pipe.fit(X, y, model__sample_weight=class_weights(y, params["class_weight_exponent"], n_classes))
    return pipe


def suggest(trial, space, n_classes):
    params = {}
    for name in TREE_PARAMS + ("class_weight_exponent",):
        bounds = space[name]
        if name in CATEGORICAL:
            params[name] = trial.suggest_categorical(name, list(bounds))
        elif name in INT_PARAMS:
            params[name] = trial.suggest_int(name, int(bounds[0]), int(bounds[1]))
        else:
            params[name] = trial.suggest_float(name, float(bounds[0]), float(bounds[1]), log=name in LOG_UNIFORM)
    if n_classes == 2:
        params["threshold"] = trial.suggest_float("threshold", *map(float, space["threshold"]))
    else:
        params["prior_exponent"] = trial.suggest_float("prior_exponent", *map(float, space["prior_exponent"]))
    return params


def cv_balanced_accuracy(X, y, folds, params, n_classes, seed, threads, trial=None):
    """Mean inner-fold balanced accuracy under the trial's weights and decision rule."""
    import optuna

    scores = []
    for step, (train, valid) in enumerate(folds):
        pipe = fit(X.iloc[train], y[train], params, n_classes, seed, threads)
        P = pipe.predict_proba(X.iloc[valid])
        labels = decide(P, class_prior(y[train], n_classes), **rule_from(params, n_classes))
        scores.append(float(balanced_accuracy_score(y[valid], labels)))
        if trial is not None:
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores)), float(np.std(scores))


def tune(X, y, folds, cfg, n_classes, seed, threads, n_trials, timeout_s, progress=None):
    import warnings
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=min(20, n_trials // 4 or 1))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    space = cfg["search_space"]

    def objective(trial):
        params = suggest(trial, space, n_classes)
        mean, sd = cv_balanced_accuracy(X, y, folds, params, n_classes, seed, threads, trial)
        trial.set_user_attr("cv_sd", sd)
        if progress is not None and trial.number % 10 == 0:
            done = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            progress(stage=MODEL, trial=trial.number, best=float(max(done)) if done else None)
        return mean

    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, gc_after_trial=True)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise GateError("Optuna produced no completed trial within the budget")
    return study


def trials_frame(study):
    rows = []
    for t in study.trials:
        rows.append({"number": t.number, "state": t.state.name, "value": t.value,
                     "cv_sd": t.user_attrs.get("cv_sd"), "duration_s": (t.duration.total_seconds() if t.duration else None),
                     **{f"param_{k}": v for k, v in t.params.items()}})
    return pd.DataFrame(rows)


def consensus_params(study, top=5):
    """Median (numeric) / mode (categorical) of the top-k completed trials: a robustness check."""
    import optuna

    done = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                  key=lambda t: t.value, reverse=True)[:top]
    params = {}
    for name in done[0].params:
        values = [t.params[name] for t in done]
        if name in CATEGORICAL:
            params[name] = max(set(values), key=values.count)
        elif name in INT_PARAMS:
            params[name] = int(round(float(np.median(values))))
        else:
            params[name] = float(np.median(values))
    return params


# ----------------------------------------------------------------------------- scoring

def metrics_block(y, P, labels):
    out = point(y, P, labels=labels)
    return {k: out[k] for k in ("balanced_accuracy", "auroc", "brier", "log_loss", "macro_f1")}


def paired_bootstrap(y, P_a, labels_a, P_b, labels_b, iterations, seed):
    """Δ = arm A − arm B on the same resamples; 95% percentile intervals."""
    rng = np.random.default_rng(seed)
    d_bacc, d_auroc = [], []
    n = len(y)
    for _ in range(iterations):
        i = rng.integers(0, n, n)
        d_bacc.append(balanced_accuracy_score(y[i], labels_a[i]) - balanced_accuracy_score(y[i], labels_b[i])
                      if len(np.unique(y[i])) == P_a.shape[1] else np.nan)
        a, b = auroc(y[i], P_a[i]), auroc(y[i], P_b[i])
        d_auroc.append(a - b if np.isfinite(a) and np.isfinite(b) else np.nan)
    d_bacc, d_auroc = np.asarray(d_bacc), np.asarray(d_auroc)
    invalid_bacc = int((~np.isfinite(d_bacc)).sum())
    d_bacc = d_bacc[np.isfinite(d_bacc)]
    d_auroc = d_auroc[np.isfinite(d_auroc)]
    return {
        "delta_bacc": float(balanced_accuracy_score(y, labels_a) - balanced_accuracy_score(y, labels_b)),
        "delta_bacc_ci": [float(v) for v in np.quantile(d_bacc, [0.025, 0.975])] if len(d_bacc) else [np.nan, np.nan],
        "delta_auroc": float(auroc(y, P_a) - auroc(y, P_b)),
        "delta_auroc_ci": [float(v) for v in np.quantile(d_auroc, [0.025, 0.975])] if len(d_auroc) else [np.nan, np.nan],
        "iterations": int(iterations),
        "invalid_missing_class_draws": invalid_bacc,
    }


def prediction_table(model, ids, y, P, labels, classes):
    table = pd.DataFrame({"model": model, "patient_id": ids, "observed": y})
    for c, name in enumerate(classes):
        table[f"p__{name}"] = P[:, c]
    table["predicted"] = np.asarray(labels, dtype=int)
    return table


# ----------------------------------------------------------------------------- entry point

def run_holdout(run, seed=None, out=None, n_trials=None, max_minutes=None):
    """Tune, refit, score. ``seed`` other than the study seed is a sensitivity split."""
    cfg = run.study["holdout"]
    seed = run.study["seed"] if seed is None else int(seed)
    out = Path(out) if out is not None else run.out / "holdout"
    out.mkdir(parents=True, exist_ok=True)
    n_trials = int(cfg["optuna_trials"] if n_trials is None else n_trials)
    timeout_s = 60.0 * float(cfg["max_minutes"] if max_minutes is None else max_minutes)
    threads = int(cfg.get("threads", run.study["threads"]))
    K = len(run.classes)

    P_zs_all = run.predictions("ZS_raw")
    if P_zs_all is None:
        raise GateError("ZS_raw predictions are unavailable; run the `zeroshot` stage first")

    X = predictor_matrix(run)
    y = run.y
    dev, hold = holdout_split(y, float(cfg["fraction"]), seed)
    if np.bincount(y[hold], minlength=K).min() < 1 or np.bincount(y[dev], minlength=K).min() < 2:
        raise GateError("Hold-out split leaves a class without patients")
    write_json(out / "split.json", {
        "seed": seed, "fraction": float(cfg["fraction"]), "data_hash": run.data_hash,
        "development": run.ids[dev].tolist(), "holdout": run.ids[hold].tolist(),
        "class_counts_development": np.bincount(y[dev], minlength=K).tolist(),
        "class_counts_holdout": np.bincount(y[hold], minlength=K).tolist(),
    })

    folds = inner_folds(y[dev], cfg["inner_folds"], seed)
    started = time.time()
    study = tune(X.iloc[dev].reset_index(drop=True), y[dev], folds, cfg, K, seed, threads, n_trials, timeout_s,
                 progress=run.progress)
    tuning_s = time.time() - started
    trials = trials_frame(study)
    trials.to_csv(out / "trials.csv", index=False)
    best = dict(study.best_trial.params)
    best_cv = float(study.best_value)
    best_cv_sd = float(study.best_trial.user_attrs.get("cv_sd", np.nan))

    # Refit on all development patients; one pass over the hold-out.
    X_dev, X_hold = X.iloc[dev].reset_index(drop=True), X.iloc[hold].reset_index(drop=True)
    prior_dev = class_prior(y[dev], K)
    pipe = fit(X_dev, y[dev], best, K, seed, threads)
    P_xgb = pipe.predict_proba(X_hold)
    P_xgb = P_xgb / P_xgb.sum(axis=1, keepdims=True)
    labels_xgb = decide(P_xgb, prior_dev, **rule_from(best, K))
    labels_xgb_argmax = P_xgb.argmax(axis=1)

    # Robustness: consensus of the top-5 trials, refit and scored once.
    consensus = consensus_params(study)
    pipe_c = fit(X_dev, y[dev], consensus, K, seed, threads)
    P_c = pipe_c.predict_proba(X_hold)
    labels_c = decide(P_c, prior_dev, **rule_from(consensus, K))

    # Zero-shot on the identical hold-out patients: its own argmax, and a prior-matched rule.
    P_zs = P_zs_all[hold]
    labels_zs = P_zs.argmax(axis=1)
    labels_zs_prior = np.argmax(P_zs / prior_dev, axis=1)

    y_hold = y[hold]
    boot_seed = seed + 7
    iterations = int(cfg.get("bootstrap_iterations", run.study["bootstrap_iterations"]))
    metrics = {
        "model": MODEL, "dataset": run.dataset, "outcome": run.outcome, "seed": seed,
        "n_development": int(len(dev)), "n_holdout": int(len(hold)), "n_classes": K,
        "inner_folds": len(folds), "trials_completed": int((trials.state == "COMPLETE").sum()),
        "trials_pruned": int((trials.state == "PRUNED").sum()), "tuning_seconds": tuning_s,
        "best_params": best, "best_cv_balanced_accuracy": best_cv, "best_cv_sd": best_cv_sd,
        "consensus_params": consensus,
        "holdout": {
            "XGB_optuna": metrics_block(y_hold, P_xgb, labels_xgb),
            "XGB_optuna_argmax": metrics_block(y_hold, P_xgb, labels_xgb_argmax),
            "XGB_optuna_consensus": metrics_block(y_hold, P_c, labels_c),
            "ZS_raw": metrics_block(y_hold, P_zs, labels_zs),
            "ZS_raw_prior_matched": metrics_block(y_hold, P_zs, labels_zs_prior),
        },
        "paired": {
            "XGB_optuna_minus_ZS_raw": paired_bootstrap(y_hold, P_xgb, labels_xgb, P_zs, labels_zs, iterations, boot_seed),
            "XGB_optuna_minus_ZS_raw_prior_matched": paired_bootstrap(y_hold, P_xgb, labels_xgb, P_zs, labels_zs_prior,
                                                                      iterations, boot_seed),
        },
    }
    metrics["cv_minus_holdout_balanced_accuracy"] = best_cv - metrics["holdout"]["XGB_optuna"]["balanced_accuracy"]

    classes = run.classes
    prediction_table(MODEL, run.ids[hold], y_hold, P_xgb, labels_xgb, classes).to_csv(out / f"{MODEL}.csv", index=False)
    prediction_table("ZS_raw", run.ids[hold], y_hold, P_zs, labels_zs, classes).to_csv(out / "ZS_raw.csv", index=False)
    joblib.dump(pipe, out / "model.joblib")
    write_json(out / "best_params.json", {"best": best, "consensus": consensus, "rule": rule_from(best, K),
                                          "prior_development": prior_dev.tolist()})
    write_json(out / "metrics.json", metrics)
    write_json(out / "manifest.json", {
        "model": MODEL, "dataset": run.dataset, "outcome": run.outcome, "data_hash": run.data_hash,
        "dataset_spec_sha256": digest(run.root / "datasets" / run.dataset / "dataset.yaml"),
        "holdout_config": cfg, "seed": seed, "n_trials_requested": n_trials, "timeout_seconds": timeout_s,
        "classes": classes, "predictor_columns": list(X.columns),
        "zs_raw_csv_sha256": digest(run.out / "predictions/ZS_raw.csv"),
        "files": {name: digest(out / name) for name in ("split.json", "trials.csv", f"{MODEL}.csv", "ZS_raw.csv",
                                                        "model.joblib", "best_params.json", "metrics.json")},
        "versions": {**versions(), "optuna": _optuna_version()},
    })
    run.progress(stage=MODEL, done=True, holdout_bacc=metrics["holdout"]["XGB_optuna"]["balanced_accuracy"],
                 zs_bacc=metrics["holdout"]["ZS_raw"]["balanced_accuracy"])
    return metrics


def rescore_holdout(run, source):
    """Reuse the tuned hold-out arm of another study (same data, seed and split) and re-score only the readout.

    XGBoost tuning does not depend on the backbone, so a second backbone needs no second Optuna run: the
    `holdout/` folder under ``source`` (that study's outputs dir) is copied, and every ZS_raw block of
    metrics.json, the paired bootstraps and holdout/ZS_raw.csv are recomputed from this study's ZS_raw.
    """
    import shutil

    cfg = run.study["holdout"]
    src = Path(source) / run.dataset / run.outcome / "holdout"
    if not (src / "metrics.json").exists():
        raise GateError(f"No tuned hold-out arm to reuse at {src}; run `xgb_holdout` there first")
    split = read_json(src / "split.json")
    if split["seed"] != run.study["seed"]:
        raise GateError("Source hold-out split was made with a different seed; cannot reuse it")
    # The split is a function of labels and seed, so a codebook (dataset.yaml) edit that changes data_hash but not
    # the patients keeps the arm comparable; the ids are checked below and the source hash is recorded in the manifest.
    out = run.out / "holdout"
    if out.resolve() != src.resolve():
        out.mkdir(parents=True, exist_ok=True)
        for name in ("split.json", "trials.csv", f"{MODEL}.csv", "model.joblib", "best_params.json"):
            shutil.copy2(src / name, out / name)
    metrics = read_json(src / "metrics.json")
    K = len(run.classes)
    index = {pid: i for i, pid in enumerate(run.ids)}
    if set(split["holdout"]) | set(split["development"]) != set(index) or len(split["holdout"]) + len(split["development"]) != len(index):
        raise GateError("Source hold-out split does not cover the same patients; cannot reuse it")
    hold = np.array([index[pid] for pid in split["holdout"]])
    dev = np.array([index[pid] for pid in split["development"]])
    y_hold, prior_dev = run.y[hold], class_prior(run.y[dev], K)
    from jevtest.zeroshot import holdout_zero_shot_probs
    P_zs = holdout_zero_shot_probs(run, hold)  # full ZS_raw, or the hold-out-only artifact of a large backbone
    xgb = pd.read_csv(out / f"{MODEL}.csv", dtype={"patient_id": str})
    if not np.array_equal(xgb.patient_id.to_numpy(), run.ids[hold]):
        raise GateError("Source XGB_optuna hold-out predictions do not list the same patients")
    # Class labels may have been reworded in the codebook since the arm was tuned; the class order (codes) is fixed.
    prob_columns = [c for c in xgb.columns if c.startswith("p__")]
    if len(prob_columns) != K:
        raise GateError("Source XGB_optuna hold-out predictions do not have one probability column per class")
    P_xgb = xgb[prob_columns].to_numpy(dtype=float)
    labels_xgb = xgb["predicted"].to_numpy(dtype=int)
    labels_zs = P_zs.argmax(axis=1)
    labels_zs_prior = np.argmax(P_zs / prior_dev, axis=1)
    seed = int(split["seed"])
    iterations = int(cfg.get("bootstrap_iterations", run.study["bootstrap_iterations"]))
    metrics["holdout"]["ZS_raw"] = metrics_block(y_hold, P_zs, labels_zs)
    metrics["holdout"]["ZS_raw_prior_matched"] = metrics_block(y_hold, P_zs, labels_zs_prior)
    metrics["paired"] = {
        "XGB_optuna_minus_ZS_raw": paired_bootstrap(y_hold, P_xgb, labels_xgb, P_zs, labels_zs, iterations, seed + 7),
        "XGB_optuna_minus_ZS_raw_prior_matched": paired_bootstrap(y_hold, P_xgb, labels_xgb, P_zs, labels_zs_prior,
                                                                  iterations, seed + 7),
    }
    metrics["rescored_from"] = str(src)
    prediction_table("ZS_raw", run.ids[hold], y_hold, P_zs, labels_zs, run.classes).to_csv(out / "ZS_raw.csv", index=False)
    write_json(out / "metrics.json", metrics)
    manifest = read_json(src / "manifest.json")
    zs_source = next(p for p in ("predictions/ZS_raw.csv", "predictions/ZS_raw_holdout.csv") if (run.out / p).exists())
    manifest["zs_raw_csv_sha256"] = digest(run.out / zs_source)
    manifest["zs_raw_source"] = zs_source
    manifest["files"]["ZS_raw.csv"] = digest(out / "ZS_raw.csv")
    manifest["files"]["metrics.json"] = digest(out / "metrics.json")
    manifest["rescored_from"] = str(src)
    manifest["source_data_hash"], manifest["data_hash"] = split["data_hash"], run.data_hash
    write_json(out / "manifest.json", manifest)
    run.progress(stage=MODEL, done=True, rescored=True,
                 holdout_bacc=metrics["holdout"]["XGB_optuna"]["balanced_accuracy"],
                 zs_bacc=metrics["holdout"]["ZS_raw"]["balanced_accuracy"])
    return metrics


def run_sensitivity(run):
    """Extra split seeds with the reduced budget; written under holdout/sensitivity/seed<k>/."""
    cfg = run.study["holdout"]
    base = run.study["seed"]
    results = []
    for k in range(1, int(cfg.get("sensitivity_seeds", 0)) + 1):
        seed = base + k
        out = run.out / "holdout" / "sensitivity" / f"seed{seed}"
        if (out / "metrics.json").exists():
            results.append(read_json(out / "metrics.json"))
            continue
        results.append(run_holdout(run, seed=seed, out=out, n_trials=cfg.get("sensitivity_trials"),
                                   max_minutes=cfg.get("sensitivity_max_minutes")))
    return results


def run_holdout_full(run, out=None):
    """XGB_optuna_full: the same hold-out, but the development set is every other row of the whole table.

    Requires the primary ``holdout/split.json`` and ``datasets/<name>/readout_to_full.json`` (from
    ``scripts/map_readout_to_full.py``). Every full-table row that is a copy of a hold-out patient is
    excluded from training. Optuna runs on a stratified cap of development rows; the best trial is
    refit on all of them and applied once to the hold-out patients (predictor values from the readout
    table, identical by construction of the map).
    """
    from jevtest.data import load
    from jevtest.config import dataset_dir

    cfg = run.study["holdout"]
    fcfg = cfg.get("full", {})
    seed = run.study["seed"]
    threads = int(cfg.get("threads", run.study["threads"]))
    K = len(run.classes)
    primary_dir = run.out / "holdout"
    out = Path(out) if out is not None else primary_dir / "full"
    split_path = primary_dir / "split.json"
    map_path = dataset_dir(run.root, run.dataset) / "readout_to_full.json"
    if not split_path.exists():
        raise GateError("primary holdout/split.json is missing; run the `xgb_holdout` stage first")
    if not map_path.exists():
        raise GateError(f"{map_path} is missing; run scripts/map_readout_to_full.py")
    split = read_json(split_path)
    if split["data_hash"] != run.data_hash:
        raise GateError("holdout/split.json was drawn from different data (hash mismatch)")
    mapping = read_json(map_path)
    rows = _mapped_rows(run, mapping["rows"])
    unmatched = [rid for rid in split["holdout"] if rid not in rows]
    if unmatched:
        raise GateError(f"{len(unmatched)} hold-out patients have no row in the full table (e.g. {unmatched[:3]})")
    primary_metrics_path = primary_dir / "metrics.json"
    primary = read_json(primary_metrics_path) if primary_metrics_path.exists() else None

    full = load(run.root, run.dataset, run.outcome, full=True)
    if full.classes != run.classes:
        raise GateError("full table declares different classes")
    hold_full_ids = {fid for rid in split["holdout"] for fid in rows[rid]}
    dev_mask = ~np.isin(full.ids, list(hold_full_ids))
    if int((~dev_mask).sum()) != len(hold_full_ids):
        raise GateError(f"only {int((~dev_mask).sum())} of {len(hold_full_ids)} mapped hold-out rows exist in the full table")
    X_full = predictor_matrix(full)
    X_dev, y_dev = X_full.loc[dev_mask].reset_index(drop=True), full.y[dev_mask]
    if np.bincount(y_dev, minlength=K).min() < 2:
        raise GateError("full-table development set leaves a class with < 2 patients")

    out.mkdir(parents=True, exist_ok=True)
    cap = int(fcfg.get("tuning_cap", 50000))
    if len(y_dev) > cap:
        tune_idx, _ = train_test_split(np.arange(len(y_dev)), train_size=cap, stratify=y_dev, random_state=seed)
        tune_idx = np.sort(tune_idx)
    else:
        tune_idx = np.arange(len(y_dev))
    n_trials = int(fcfg.get("optuna_trials", cfg["optuna_trials"]))
    timeout_s = 60.0 * float(fcfg.get("max_minutes", cfg["max_minutes"]))
    folds = inner_folds(y_dev[tune_idx], cfg["inner_folds"], seed)
    started = time.time()
    study = tune(X_dev.iloc[tune_idx].reset_index(drop=True), y_dev[tune_idx], folds, cfg, K, seed, threads,
                 n_trials, timeout_s, progress=run.progress)
    tuning_s = time.time() - started
    trials = trials_frame(study)
    trials.to_csv(out / "trials.csv", index=False)
    best = dict(study.best_trial.params)
    best_cv = float(study.best_value)

    # Hold-out rows in the primary split's order, values from the readout table.
    pos = {pid: i for i, pid in enumerate(run.ids)}
    hold = np.array([pos[rid] for rid in split["holdout"]])
    X_hold = predictor_matrix(run).iloc[hold].reset_index(drop=True)
    y_hold = run.y[hold]
    prior_dev = class_prior(y_dev, K)

    refit_started = time.time()
    pipe = fit(X_dev, y_dev, best, K, seed, threads)
    refit_s = time.time() - refit_started
    P_xgb = pipe.predict_proba(X_hold)
    P_xgb = P_xgb / P_xgb.sum(axis=1, keepdims=True)
    labels_xgb = decide(P_xgb, prior_dev, **rule_from(best, K))
    labels_xgb_argmax = P_xgb.argmax(axis=1)

    P_zs = run.predictions("ZS_raw")
    if P_zs is None:
        raise GateError("ZS_raw predictions are unavailable")
    P_zs = P_zs[hold]
    labels_zs = P_zs.argmax(axis=1)
    labels_zs_prior = np.argmax(P_zs / prior_dev, axis=1)

    iterations = int(cfg.get("bootstrap_iterations", run.study["bootstrap_iterations"]))
    boot_seed = seed + 7
    model = MODEL + "_full"
    metrics = {
        "model": model, "dataset": run.dataset, "outcome": run.outcome, "seed": seed,
        "n_development": int(len(y_dev)), "n_development_tuning": int(len(tune_idx)), "n_full_table": int(len(full.y)),
        "n_full_rows_excluded_as_holdout": int(len(hold_full_ids)), "n_holdout": int(len(hold)), "n_classes": K,
        "map_method": mapping["method"], "inner_folds": len(folds),
        "trials_completed": int((trials.state == "COMPLETE").sum()), "trials_pruned": int((trials.state == "PRUNED").sum()),
        "tuning_seconds": tuning_s, "refit_seconds": refit_s,
        "best_params": best, "best_cv_balanced_accuracy": best_cv,
        "best_cv_sd": float(study.best_trial.user_attrs.get("cv_sd", np.nan)),
        "holdout": {
            model: metrics_block(y_hold, P_xgb, labels_xgb),
            model + "_argmax": metrics_block(y_hold, P_xgb, labels_xgb_argmax),
            "ZS_raw": metrics_block(y_hold, P_zs, labels_zs),
            "ZS_raw_prior_matched": metrics_block(y_hold, P_zs, labels_zs_prior),
        },
        "paired": {
            f"{model}_minus_ZS_raw": paired_bootstrap(y_hold, P_xgb, labels_xgb, P_zs, labels_zs, iterations, boot_seed),
            f"{model}_minus_ZS_raw_prior_matched": paired_bootstrap(y_hold, P_xgb, labels_xgb, P_zs, labels_zs_prior,
                                                                    iterations, boot_seed),
        },
    }
    metrics["cv_minus_holdout_balanced_accuracy"] = best_cv - metrics["holdout"][model]["balanced_accuracy"]
    if primary is not None:
        prim = pd.read_csv(primary_dir / f"{MODEL}.csv", dtype={"patient_id": str})
        if prim.patient_id.tolist() == split["holdout"]:
            P_prim = prim[[f"p__{c}" for c in run.classes]].to_numpy(dtype=float)
            labels_prim = prim["predicted"].to_numpy(dtype=int)
            metrics["holdout"][MODEL] = metrics_block(y_hold, P_prim, labels_prim)
            metrics["paired"][f"{model}_minus_{MODEL}"] = paired_bootstrap(y_hold, P_xgb, labels_xgb, P_prim, labels_prim,
                                                                           iterations, boot_seed)

    prediction_table(model, run.ids[hold], y_hold, P_xgb, labels_xgb, run.classes).to_csv(out / f"{model}.csv", index=False)
    joblib.dump(pipe, out / "model.joblib")
    write_json(out / "best_params.json", {"best": best, "rule": rule_from(best, K), "prior_development": prior_dev.tolist()})
    write_json(out / "metrics.json", metrics)
    write_json(out / "manifest.json", {
        "model": model, "dataset": run.dataset, "outcome": run.outcome, "data_hash": run.data_hash,
        "full_data_hash": full.data_hash, "readout_to_full_sha256": digest(map_path),
        "split_sha256": digest(split_path), "holdout_config": cfg, "seed": seed, "n_trials_requested": n_trials,
        "timeout_seconds": timeout_s, "tuning_cap": cap, "classes": run.classes, "predictor_columns": list(X_full.columns),
        "files": {name: digest(out / name) for name in ("trials.csv", f"{model}.csv", "model.joblib", "best_params.json",
                                                        "metrics.json")},
        "versions": {**versions(), "optuna": _optuna_version()},
    })
    run.progress(stage=model, done=True, holdout_bacc=metrics["holdout"][model]["balanced_accuracy"],
                 zs_bacc=metrics["holdout"]["ZS_raw"]["balanced_accuracy"])
    return metrics


def _mapped_rows(run, rows):
    """readout_to_full.json is keyed by source ids; translate both sides when the run pseudonymises."""
    if not run.ds.get("pseudonymize", True):
        return rows
    from jevtest.data import pseudonyms

    keys = list(rows)
    values = sorted({v for vs in rows.values() for v in vs})
    pk = dict(zip(keys, pseudonyms(run.root, keys)))
    pv = dict(zip(values, pseudonyms(run.root, values)))
    return {pk[k]: [pv[v] for v in vs] for k, vs in rows.items()}


def run_stage(run):
    run_holdout(run)


def run_sensitivity_stage(run):
    run_sensitivity(run)


def _optuna_version():
    try:
        import optuna
        return optuna.__version__
    except ImportError:
        return None
