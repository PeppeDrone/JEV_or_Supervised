#!/usr/bin/env python3
"""Execute the 45-task scientific correction without changing any original artifact.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_scientific_revision.py
Repeat after additional inference with --refresh (derived outputs only).
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from scipy.stats import spearmanr, kendalltau, rankdata, wilcoxon, mannwhitneyu, false_discovery_control, t as student_t
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from jevtest.benchmark import ALL_PAIRS
from jevtest.config import digest
from jevtest.data import load, predictor_matrix
from jevtest.revision_reference import select_marker, predict_marker, score_auc
from jevtest.revision_calibration import METHODS, fit_readout, apply_readout, calibration_sample, probabilities, log_scores
from jevtest.revision_rendering import render_condition
from jevtest.revision_paths import CURRENT, INFERENCE_ROOTS
from task_mechanism_covariates import direction_only_score, additive_model
import plot_backbone_ladder as ladder

OUT = CURRENT
BACKBONES = ("0.8b", "2b", "4b", "9b", "27b", "397b_a17b_fp8")
SEED = 20261008


def clean(x):
    if isinstance(x, (np.ndarray, list, tuple)): return [clean(v) for v in x]
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    return x


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, allow_nan=False))


def ba(y, labels, k):
    return float(np.mean([(labels[y == c] == c).mean() for c in range(k)])) if len(np.unique(y)) == k else np.nan


def evaluate(y, P, labels=None, probability_metrics=None):
    P = probabilities(P)
    labels = P.argmax(1) if labels is None else np.asarray(labels)
    k = P.shape[1]
    recall = [float((labels[y == c] == c).mean()) if (y == c).any() else np.nan for c in range(k)]
    frequency = np.bincount(labels, minlength=k) / len(y)
    result = dict(balanced_accuracy=ba(y, labels, k), chance_adjusted_ba=(ba(y, labels, k)-1/k)/(1-1/k),
                  class_recall=json.dumps(clean(recall)), predicted_frequency=json.dumps(frequency.tolist()),
                  constant_class=bool(len(np.unique(labels)) == 1))
    if probability_metrics is None:
        target = np.eye(k)[y]
        brier = np.mean((P[:, 1] - y)**2) if k == 2 else np.mean(np.sum((P-target)**2, axis=1))
        ece = []
        for c in range(k):
            bins = np.minimum((P[:, c] * 10).astype(int), 9)
            ece.append(sum(np.mean(bins == b)*abs(P[bins == b, c].mean()-(y[bins == b] == c).mean())
                           for b in range(10) if (bins == b).any()))
        # Retain the original rank-sum AUROC arithmetic, including exact numerical ties.
        result.update(auroc=ladder.fast_auc(y, P), brier=float(brier), log_loss=float(-np.mean(log_scores(P)[np.arange(len(y)), y])),
                      ece_macro_10bins=float(np.mean(ece)), mean_entropy=float(-np.mean(np.sum(P*log_scores(P), axis=1))))
    else:
        result.update({v: probability_metrics[v] for v in ("auroc", "brier", "log_loss", "ece_macro_10bins", "mean_entropy")})
    return result


def summary_interval(values, rng=None):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values): return [np.nan, np.nan, np.nan]
    rng = np.random.default_rng(SEED) if rng is None else rng
    draws = np.median(rng.choice(values, (2000, len(values)), replace=True), axis=1)
    return [float(np.median(values)), *np.quantile(draws, [.025, .975]).tolist()]


def paired_interval(y, A, B, labA, labB, iterations=1000):
    rng = np.random.default_rng(SEED)
    k, n = A.shape[1], len(y)
    da, db, invalid = [], [], 0
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) != k:
            invalid += 1
            continue
        da.append(ladder.fast_auc(y[idx], A[idx])-ladder.fast_auc(y[idx], B[idx]))
        db.append(ba(y[idx], labA[idx], k)-ba(y[idx], labB[idx], k))
    return dict(auroc_lo=float(np.quantile(da, .025)), auroc_hi=float(np.quantile(da, .975)),
                ba_lo=float(np.quantile(db, .025)), ba_hi=float(np.quantile(db, .975)),
                invalid_missing_class_draws=invalid, valid_draws=len(da), requested_draws=iterations)


def partial_rank(x, y, controls):
    ranks = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in controls])
    rx, ry = rankdata(x), rankdata(y)
    rx -= ranks @ np.linalg.lstsq(ranks, rx, rcond=None)[0]
    ry -= ranks @ np.linalg.lstsq(ranks, ry, rcond=None)[0]
    if min(np.std(rx), np.std(ry)) < 1e-10: return np.nan, np.nan
    r = float(np.corrcoef(rx, ry)[0, 1]); df = len(x)-len(controls)-2
    p = 2*student_t.sf(abs(r)*np.sqrt(df/max(1-r*r, 1e-15)), df)
    return r, float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--reuse-bootstrap", action="store_true", help="reuse verified bootstrap CIs after new inference; point metrics are recomputed")
    args = ap.parse_args()
    old_manifest = json.loads((OUT/"execution_manifest.json").read_text()) if (OUT/"execution_manifest.json").exists() else None
    cached_original, cached_refs = {}, {}
    if args.reuse_bootstrap:
        if not old_manifest or old_manifest["bootstrap"] != args.bootstrap:
            raise ValueError("bootstrap cache requires a completed run with the same resample count")
        for path, sha in old_manifest["input_sha256"].items():
            if path.startswith(("outputs/", "outputs_qwen3.5_", "datasets/")) and digest(ROOT/path) != sha:
                raise ValueError(f"bootstrap input changed: {path}")
        if digest(ROOT/"src/jevtest/revision_reference.py") != old_manifest["input_sha256"]["src/jevtest/revision_reference.py"]:
            raise ValueError("reference-fitting code changed")
        cached_original = {(r["dataset"],r["outcome"],r["backbone"]):r for r in pd.read_csv(OUT/"original_metrics.csv").to_dict("records")}
        cached_refs = {(r["dataset"],r["outcome"],r["backbone"],r["reference"]):r for r in pd.read_csv(OUT/"reference_comparisons.csv").to_dict("records")}
    if (OUT / "execution_manifest.json").exists() and not args.refresh:
        raise RuntimeError("derived results already exist; use --refresh after additional inference")
    manifest = dict(seed=SEED, repeats=args.repeats, bootstrap=args.bootstrap, tasks=ALL_PAIRS,
                    backbones=BACKBONES, command=sys.argv, versions={}, input_sha256={})
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "anyjev", "matplotlib"):
        manifest["versions"][package] = importlib.metadata.version(package)
    task_rows, originals, comparisons, interventions, failures, ranking, class_rows = [], [], [], [], [], [], []
    selected_rows, candidate_rows, semantic, protocol, direction, interactions, complementarity = [], [], [], [], [], [], []
    reliability, class_mapping, reference_pairs = [], [], []
    for dataset, outcome in ALL_PAIRS:
        key = dict(dataset=dataset, outcome=outcome)
        run = load(ROOT, dataset, outcome)
        split_path = ROOT / "outputs" / dataset / outcome / "holdout/split.json"
        split = json.loads(split_path.read_text())
        ix = {p: i for i, p in enumerate(run.ids)}
        if set(split["development"]) & set(split["holdout"]): raise AssertionError("overlapping split")
        if set(split["development"]) | set(split["holdout"]) != set(ix): raise AssertionError("split coverage")
        dev, hold = np.array([ix[p] for p in split["development"]]), np.array([ix[p] for p in split["holdout"]])
        ydev, y = run.y[dev], run.y[hold]
        k = len(run.classes)
        X = predictor_matrix(run)
        Xd, Xh = X.iloc[dev], X.iloc[hold]
        for path in (split_path, ROOT / "datasets" / dataset / "dataset.yaml", ROOT / "datasets" / dataset / run.ds["file"]):
            manifest["input_sha256"][str(path.relative_to(ROOT))] = digest(path)
        xgb_path = split_path.parent / "XGB_optuna.csv"
        xgb = pd.read_csv(xgb_path, dtype={"patient_id": str}).set_index("patient_id").loc[run.ids[hold]]
        cols = ["p__" + c for c in run.classes]
        xcols = list(xgb.filter(regex="^p__").columns)
        if len(xcols) != k: raise AssertionError("XGB class count")
        # The original producer enumerates numeric class indices in prediction_table.
        # Clinical class descriptions were subsequently expanded in the codebooks;
        # retain the producer's numeric order, record every historical name explicitly.
        for c in range(k):
            class_mapping.append(dict(**key, class_index=c, xgb_column=xcols[c], current_column=cols[c],
                                      evidence="xgb_optuna.prediction_table enumerates class indices; observed numeric labels agree"))
        if not np.array_equal(y, xgb.observed): raise AssertionError("XGB labels")
        Px, lx = probabilities(xgb[xcols].to_numpy()), xgb.predicted.to_numpy()
        mx = evaluate(y, Px, lx)
        manifest["input_sha256"][str(xgb_path.relative_to(ROOT))] = digest(xgb_path)

        # DEVELOPMENT-ONLY REFERENCE FITTING =====================================
        state, candidates = select_marker(Xd, ydev, k, SEED)
        ms, ml = predict_marker(state, Xh[state["predictor"]])
        ds = direction_only_score(Xd, ydev, Xh, k)
        dd = direction_only_score(Xd, ydev, Xd, k)
        # Store both class contrasts for binary rank comparisons, without claiming probabilities.
        dmat = np.column_stack([-ds, ds]) if k == 2 else ds
        dl = (ds > 0).astype(int) if k == 2 else ds.argmax(1)
        with warnings.catch_warnings(record=True) as caught:
            Pa = additive_model(Xd, ydev, Xh)
        for w in caught:
            failures.append(dict(**key, experiment="additive_fit_warning", status="warning", reason=str(w.message)))
        selected_rows.append(dict(**key, **{a: state[a] for a in ("predictor", "cv_auroc", "folds", "seed")}))
        candidates = candidates.assign(**key); candidate_rows.append(candidates)
        refdir = OUT / "references" / dataset / outcome
        save_json(refdir / "selected_marker.json", state)
        pred = pd.DataFrame(dict(patient_id=run.ids[hold], observed=y, marker_label=ml, direction_label=dl))
        for c in range(k):
            pred[f"marker_score__{c}"] = ms[:, c]; pred[f"direction_score__{c}"] = dmat[:, c]; pred[f"additive_p__{c}"] = Pa[:, c]
        pred.to_csv(refdir / "holdout.csv", index=False)
        reference_pairs.append(dict(**key, n_classes=k, comparison="best_marker_minus_direction_only",
                                    delta_auroc=score_auc(y,ms)-score_auc(y,dmat),delta_ba=ba(y,ml,k)-ba(y,dl,k),
                                    **paired_interval(y,ms,dmat,ml,dl,args.bootstrap)))
        head = mx["auroc"] - .5
        infos = list(run.ds["predictors"].values())
        texts = [render_condition(run.frame.iloc[i], run.ds, "original") for i in dev]
        task_rows.append(dict(**key, n_development=len(dev), n_holdout=len(hold), n_predictors=X.shape[1], n_classes=k,
                              auc_XGB=mx["auroc"], bacc_XGB=mx["balanced_accuracy"], adjusted_ba_XGB=mx["chance_adjusted_ba"],
                              best_univariate_auc=score_auc(y, ms), best_univariate_bacc=ba(y, ml, k),
                              selected_predictor=state["predictor"], marker_cv_auc=state["cv_auroc"],
                              univariate_share=(score_auc(y, ms)-.5)/head if head > .1 else np.nan,
                              direction_only_auc=score_auc(y, dmat), direction_only_bacc=ba(y, dl, k),
                              additive_auc=score_auc(y, Pa), additive_bacc=ba(y, Pa.argmax(1), k),
                              interaction_gain=mx["auroc"]-score_auc(y, Pa),
                              prompt_words=np.mean([len(t.split()) for t in texts]),
                              prompt_characters=np.mean([len(t) for t in texts]),
                              unit_coverage=np.mean([bool(i.get("unit")) for i in infos]),
                              category_gloss_coverage=np.mean([bool(i.get("text_values")) for i in infos])))

        # FROZEN HOLD-OUT EVALUATION AND ADDITIONAL INTERVENTIONS =================
        for backbone in BACKBONES:
            basekey = dict(**key, backbone=backbone, n_classes=k, n_holdout=len(y))
            zp = ROOT / f"outputs_qwen3.5_{backbone}" / dataset / outcome / "predictions/ZS_raw_holdout.csv"
            z = pd.read_csv(zp, dtype={"patient_id": str}).set_index("patient_id")
            if set(z.index) != set(run.ids[hold]) or z.index.has_duplicates: raise AssertionError("hold-out identity mismatch")
            z = z.loc[run.ids[hold]]
            if list(z.filter(regex="^p__").columns) != cols or not np.array_equal(z.observed, y): raise AssertionError("L0 class/label mismatch")
            P = probabilities(z[cols].to_numpy()); l = P.argmax(1); original = evaluate(y, P)
            manifest["input_sha256"][str(zp.relative_to(ROOT))] = digest(zp)
            ci_keys=("auroc_lo","auroc_hi","ba_lo","ba_hi","invalid_missing_class_draws","valid_draws","requested_draws")
            cached=cached_original.get((dataset,outcome,backbone))
            ci = {a:cached[a] for a in ci_keys} if cached else paired_interval(y, P, Px, l, lx, args.bootstrap)
            originals.append(dict(**basekey, **original, xgb_ba=mx["balanced_accuracy"], xgb_auroc=mx["auroc"],
                                  delta_ba=original["balanced_accuracy"]-mx["balanced_accuracy"],
                                  delta_auroc=original["auroc"]-mx["auroc"],
                                  delta_adjusted_ba=original["chance_adjusted_ba"]-mx["chance_adjusted_ba"], **ci))
            if k > 2:
                for source, labels in (("original", l), ("XGB", lx)):
                    for c in range(k):
                        for p in range(k):
                            class_rows.append(dict(**basekey, method=source, observed_class=c, predicted_class=p,
                                                   count=int(np.sum((y == c) & (labels == p))), class_name=run.classes[c]))
            for c in range(k):
                bins = np.minimum((P[:, c]*10).astype(int), 9)
                for b in range(10):
                    mask = bins == b
                    if mask.any(): reliability.append(dict(**basekey, class_index=c, bin=b, n=int(mask.sum()),
                                                            mean_prediction=float(P[mask, c].mean()), observed_fraction=float((y[mask] == c).mean())))
            for reference, scores, labels in (("best_marker", ms, ml), ("direction_only", dmat, dl), ("additive", Pa, Pa.argmax(1))):
                cached=cached_refs.get((dataset,outcome,backbone,reference))
                cires = {a:cached[a] for a in ci_keys} if cached else paired_interval(y, P, scores, l, labels, args.bootstrap)
                comparisons.append(dict(**basekey, reference=reference, delta_auroc=score_auc(y, P)-score_auc(y, scores),
                                        delta_ba=ba(y,l,k)-ba(y,labels,k), **cires))
                for c in ([1] if k == 2 else range(k)):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rho, rp = spearmanr(P[:, c], scores[:, c]); tau, tp = kendalltau(P[:, c], scores[:, c])
                    ranking.append(dict(**basekey, reference=reference, contrast=c, spearman=rho, spearman_p=rp,
                                        kendall_tau_b=tau, kendall_p=tp, tie_note="tau-b accounts for ties; not an equivalence test"))

            found = [r / backbone / dataset / outcome for r in INFERENCE_ROOTS if (r / backbone / dataset / outcome / "manifest.json").exists()]
            if len(found) > 1: raise AssertionError(f"duplicate additional inference for {backbone} {dataset}/{outcome}")
            infdir = found[0] if found else OUT / "inference" / backbone / dataset / outcome
            imeta = {}
            if (infdir/"manifest.json").exists():
                imeta=json.loads((infdir/"manifest.json").read_text())
                if imeta["classes"]!=run.classes or imeta["split_sha256"]!=digest(split_path) or imeta["data_hash"]!=run.data_hash:
                    raise AssertionError("additional inference provenance differs from fixed task")
                original_meta=json.loads(zp.with_suffix(".meta.json").read_text())
                original_backbone=original_meta["key"]["study"]["backbone"]
                if imeta["backbone"]["model_id"]!=original_backbone["model_id"] or imeta["backbone"]["revision"]!=original_backbone["revision"]:
                    raise AssertionError("additional inference backbone revision mismatch")
            devfile = infdir / "development.json"
            Pdev = None
            # Calibration population: every development patient (v1 design) or the prespecified label-blind pool.
            pool = np.arange(len(dev))
            if imeta.get("development_ids") is not None:
                where = {p: j for j, p in enumerate(run.ids[dev])}
                pool = np.array([where[p] for p in imeta["development_ids"]])
            pool_name = "full" if len(pool) == len(dev) else "pool"
            if devfile.exists():
                saved = json.loads(devfile.read_text())
                if set(saved) == set(run.ids[dev[pool]]):
                    Pdev = probabilities(np.array([saved[p]["probs"] for p in run.ids[dev[pool]]]))
                    manifest["input_sha256"][str(devfile.relative_to(ROOT))] = digest(devfile)
                else:
                    failures.append(dict(**basekey, experiment="development_inference", status="incomplete", reason=f"{len(saved)}/{len(pool)} calibration-population patients saved"))
            if Pdev is None:
                failures.append(dict(**basekey, experiment="learned_calibration_and_complementarity", status="incomplete", reason="complete development L0 scores unavailable"))
            fitted_states = []
            # Prevalence-only rules need labels but no development scores, so they keep the full partition.
            for budget in (10, 20, 50, 100, "full", "pool"):
                if budget == "pool" and (Pdev is None or pool_name == "full"):
                    continue
                population = len(pool) if (Pdev is not None and budget != "full") else len(dev)
                if budget not in ("full", "pool") and budget > population:
                    failures.append(dict(**basekey, experiment="label_budget", budget=budget, status="infeasible", reason="budget exceeds calibration population")); continue
                for repeat in range(1 if budget in ("full", "pool") else args.repeats):
                    idx = calibration_sample(population, "full" if budget == "pool" else budget, SEED + repeat)
                    scored = Pdev is not None and (budget != "full" or pool_name == "full")
                    methods = METHODS if scored else ("prevalence",)
                    rows_dev = pool[idx] if scored else idx
                    for method in methods:
                        calkey = dict(**basekey, method=method, budget=str(budget), repeat=repeat, calibration_n=len(idx),
                                      calibration_population=pool_name if scored else "full")
                        try:
                            fitted = fit_readout(Pdev[idx] if scored else np.empty((len(idx), k)), ydev[rows_dev], method, k)
                            Q, labels = apply_readout(P, fitted)
                            unchanged = method == "prevalence" or "threshold" in fitted
                            met = evaluate(y, Q, labels, original if unchanged else None)
                            gap = mx["balanced_accuracy"]-original["balanced_accuracy"]
                            versus_xgb = {}
                            if budget in ("full", "pool") and scored:
                                # Paired patient bootstrap of the adjusted decision against the saved XGB decision.
                                pi = paired_interval(y, Q, Px, labels, lx, args.bootstrap)
                                versus_xgb = dict(vs_xgb_ba_lo=pi["ba_lo"], vs_xgb_ba_hi=pi["ba_hi"],
                                                  vs_xgb_invalid_draws=pi["invalid_missing_class_draws"])
                            interventions.append(dict(**calkey, **met, delta_ba=met["balanced_accuracy"]-original["balanced_accuracy"],
                                                      remaining_gap=mx["balanced_accuracy"]-met["balanced_accuracy"],
                                                      recovered_gap_ratio=(met["balanced_accuracy"]-original["balanced_accuracy"])/gap if gap > .01 else np.nan,
                                                      missing_calibration_classes=fitted.get("missing_classes", 0), **versus_xgb))
                            fitted_states.append(dict(method=method,budget=str(budget),repeat=repeat,calibration_ids=run.ids[dev[rows_dev]].tolist(),parameters=fitted))
                            if k > 2 and budget in ("full", "pool") and method == "offset":
                                for c in range(k):
                                    for predclass in range(k):
                                        class_rows.append(dict(**basekey, method=f"{budget}_development_offset", observed_class=c,predicted_class=predclass,
                                                               count=int(np.sum((y == c)&(labels == predclass))),class_name=run.classes[c]))
                        except (ValueError, RuntimeError) as error:
                            failures.append(dict(**calkey, experiment="calibration", status="failed", reason=str(error)))
            save_json(OUT / "calibration_parameters" / backbone / dataset / f"{outcome}.json", fitted_states)
            if Pdev is not None:
                ddev = (np.column_stack([-dd,dd]) if k == 2 else dd)[pool]
                # Both combination arms learn the same regularised logistic link on development only.
                for name, Fd, Fh in (("direction_link", ddev, dmat), ("direction_plus_L0",np.column_stack([ddev,log_scores(Pdev)]), np.column_stack([dmat,log_scores(P)]))):
                    from sklearn.pipeline import make_pipeline
                    from sklearn.preprocessing import StandardScaler
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.,max_iter=2000,random_state=SEED))
                    model.fit(Fd,ydev[pool])
                    complementarity.append(dict(**basekey,method=name,fit_population=pool_name,fit_n=len(pool),**evaluate(y,model.predict_proba(Fh))))
                    save_json(OUT/"combination_parameters"/backbone/dataset/f"{outcome}_{name}.json",
                              dict(coefficients=model[-1].coef_, intercept=model[-1].intercept_,mean=model[0].mean_,scale=model[0].scale_,classes=model[-1].classes_))

            # SEMANTIC AND READOUT ABLATIONS: execute only for complete matching patient sets.
            # Regenerated original is the paired control for interventions on rendering/protocol.
            # The frozen saved L0 remains the benchmark and calibration target throughout.
            # Rendering/protocol reads cover all hold-out patients (v1) or a prespecified label-blind subset (v2).
            sub = np.arange(len(hold))
            if imeta.get("holdout_ids") is not None:
                where = {p: j for j, p in enumerate(run.ids[hold])}
                sub = np.array([where[p] for p in imeta["holdout_ids"]])
            ys, Ps, sub_ids = y[sub], P[sub], run.ids[hold[sub]]
            subkey = dict(basekey, n_holdout=len(sub), holdout_population="full" if len(sub) == len(hold) else "subset",
                          subset_classes_complete=bool(len(np.unique(ys)) == k))
            Pcontrol=Ps
            if (infdir/"original.json").exists():
                control=json.loads((infdir/"original.json").read_text())
                if set(control)==set(sub_ids) and len(np.unique(ys))==k:
                    Pcontrol=probabilities(np.array([control[p]["probs"] for p in sub_ids]))
                    comparisons.append(dict(**subkey,reference="regenerated_original_control",delta_auroc=score_auc(ys,Pcontrol)-score_auc(ys,Ps),
                                            delta_ba=ba(ys,Pcontrol.argmax(1),k)-ba(ys,Ps.argmax(1),k),
                                            argmax_agreement=float(np.mean(Pcontrol.argmax(1)==Ps.argmax(1))),
                                            max_probability_drift=float(np.abs(Pcontrol-Ps).max()),mean_probability_drift=float(np.abs(Pcontrol-Ps).mean())))
            for condition in ("original", "no_setting", "no_units", "anonymous", "no_category_gloss", "concise"):
                path = infdir / f"{condition}.json"
                if not path.exists(): continue
                saved = json.loads(path.read_text())
                if set(saved) != set(sub_ids):
                    failures.append(dict(**subkey,experiment=condition,status="incomplete",reason="hold-out inference incomplete"));continue
                if len(np.unique(ys)) != k:
                    failures.append(dict(**subkey,experiment=condition,status="undefined",reason="label-blind hold-out subset lacks a class; full-class BA/AUROC undefined"));continue
                Q = probabilities(np.array([saved[p]["probs"] for p in sub_ids]))
                semantic.append(dict(**subkey,condition=condition,**evaluate(ys,Q),delta_ba=ba(ys,Q.argmax(1),k)-ba(ys,Pcontrol.argmax(1),k),
                                     delta_auroc=score_auc(ys,Q)-score_auc(ys,Pcontrol),mean_absolute_score_change=float(np.abs(Q-Pcontrol).mean()),
                                     **paired_interval(ys,Q,Pcontrol,Q.argmax(1),Pcontrol.argmax(1),args.bootstrap)))
                if condition != "original": continue
                from anyjev.calibrate.permute import marginalize
                from anyjev.calibrate.contextual import apply_contextual
                stage_arrays={a:[] for a in ("raw", "rotation_only", "prior_only", "full")}
                for pid in sub_ids:
                    dg=saved[pid]["diagnostics"]; raw=np.array(dg["p_pos_raw"]); perms=dg["perms"]
                    corrected=apply_contextual(raw,np.array(dg["prior"])**dg["prior_strength"])
                    stage_arrays["raw"].append(marginalize(raw[:1],perms[:1]))
                    stage_arrays["rotation_only"].append(marginalize(raw,perms))
                    stage_arrays["prior_only"].append(marginalize(corrected[:1],perms[:1]))
                    stage_arrays["full"].append(marginalize(corrected,perms))
                for stage, values in stage_arrays.items():
                    Q=np.array(values)
                    protocol.append(dict(**subkey,stage=stage,**evaluate(ys,Q),delta_ba=ba(ys,Q.argmax(1),k)-ba(ys,Pcontrol.argmax(1),k),
                                         delta_auroc=score_auc(ys,Q)-score_auc(ys,Pcontrol),
                                         mean_answer_mass=float(np.mean([s["diagnostics"]["answer_mass"] for s in saved.values()])),
                                         mean_order_flip_raw=float(np.mean([s["diagnostics"]["order_flip_raw"] for s in saved.values()])),
                                         mean_entropy_l0=float(np.mean([-np.sum(np.array(s["probs"])*np.log(np.maximum(s["probs"],1e-300))) for s in saved.values()])),
                                         **paired_interval(ys,Q,Pcontrol,Q.argmax(1),Pcontrol.argmax(1),args.bootstrap)))
                if not np.allclose(stage_arrays["full"],np.array([saved[p]["probs"] for p in sub_ids]),atol=1e-12):raise AssertionError("readout reconstruction")
            for experiment, target in (("direction",direction),("interaction",interactions)):
                path=infdir/f"{experiment}.json"
                if not path.exists():continue
                saved=json.loads(path.read_text())
                groups={}
                for item in saved.values():
                    ident=(item["patient_id"],item.get("predictor","pair"))
                    groups.setdefault(ident,{})[item.get("level",item.get("first","")+"_"+item.get("second",""))]=item
                for (pid,predictor), group in groups.items():
                    required=("low","high") if experiment=="direction" else ("low_low","low_high","high_low","high_high")
                    if any(a not in group for a in required):continue
                    def contrast(item):
                        pp=np.array(item["probs"]);c=item["contrast"]
                        return float(np.log(max(pp[c],1e-15)/max(1-pp[c],1e-15)))
                    if experiment=="direction":
                        delta=contrast(group["high"])-contrast(group["low"])
                        target.append(dict(**basekey,patient_id=pid,predictor=predictor,delta_log_odds=delta,
                                           development_rho=group["low"]["development_rho"],agreement=bool(np.sign(delta)==np.sign(group["low"]["development_rho"])),
                                           joint_plausibility="unverified"))
                    else:
                        delta=contrast(group["high_high"])-contrast(group["high_low"])-contrast(group["low_high"])+contrast(group["low_low"])
                        target.append(dict(**basekey,patient_id=pid,interaction_contrast=delta,joint_plausibility="unverified"))
        print(f"completed {dataset}/{outcome}", flush=True)

    # CANONICAL TABLES ==========================================================
    tables=dict(task_properties=pd.DataFrame(task_rows),original_metrics=pd.DataFrame(originals),reference_comparisons=pd.DataFrame(comparisons),
                calibration_metrics=pd.DataFrame(interventions),experiment_status=pd.DataFrame(failures),ranking_agreement=pd.DataFrame(ranking),
                multiclass_confusion=pd.DataFrame(class_rows),selected_markers=pd.DataFrame(selected_rows),
                marker_cv_candidates=pd.concat(candidate_rows,ignore_index=True),semantic_metrics=pd.DataFrame(semantic),
                readout_metrics=pd.DataFrame(protocol),direction_metrics=pd.DataFrame(direction),interaction_metrics=pd.DataFrame(interactions),
                complementarity=pd.DataFrame(complementarity),reliability_bins=pd.DataFrame(reliability),
                class_column_mapping=pd.DataFrame(class_mapping),reference_pair_comparisons=pd.DataFrame(reference_pairs))
    for metric in ("spearman","kendall"):
        frame=tables["ranking_agreement"]
        frame[f"{metric}_q_bh"]=np.nan;ok=frame[f"{metric}_p"].notna()
        frame.loc[ok,f"{metric}_q_bh"]=false_discovery_control(frame.loc[ok,f"{metric}_p"])
    for name, table in tables.items():table.to_csv(OUT/f"{name}.csv",index=False)
    original=tables["original_metrics"]; properties=tables["task_properties"]
    summaries=[]
    # Preserve the original across-task random streams, separately per metric.
    primary_rng={m:np.random.default_rng(0) for m in ("balanced_accuracy","auroc")}
    for backbone,g in original.groupby("backbone",sort=False):
        for metric,delta,lo,hi in (("balanced_accuracy","delta_ba","ba_lo","ba_hi"),("auroc","delta_auroc","auroc_lo","auroc_hi")):
            med,low,high=summary_interval(g[delta],primary_rng[metric]); vals=g[delta].to_numpy()
            summaries.append(dict(backbone=backbone,metric=metric,tasks=len(g),median=float(g[metric].median()),median_delta=med,
                                  ci_low=low,ci_high=high,wilcoxon_p=float(wilcoxon(vals).pvalue),
                                  xgb_better=int((g[hi]<0).sum()),unresolved=int(((g[lo]<=0)&(g[hi]>=0)).sum()),l0_better=int((g[lo]>0).sum()),
                                  constant_tasks=int(g.constant_class.sum())))
    pd.DataFrame(summaries).to_csv(OUT/"primary_summary.csv",index=False)
    scaling=[]
    paired=original[original.backbone=="27b"].merge(original[original.backbone=="397b_a17b_fp8"],on=["dataset","outcome"],suffixes=("_27b","_397b"))
    for metric in ("auroc","balanced_accuracy"):
        paired[f"delta_{metric}"]=paired[f"{metric}_397b"]-paired[f"{metric}_27b"]
        d=paired[f"delta_{metric}"]
        scaling_rng=np.random.default_rng(1)
        # The original consecutive ladder consumed four bootstrap blocks before 27B -> 397B.
        for small,big in zip(BACKBONES[:-2],BACKBONES[1:-1]):
            earlier=original[original.backbone==small].merge(original[original.backbone==big],on=["dataset","outcome"],suffixes=("_small","_big"))
            summary_interval(earlier[f"{metric}_big"]-earlier[f"{metric}_small"],scaling_rng)
        m,l,h=summary_interval(d,scaling_rng)
        scaling.append(dict(metric=metric,median_paired_delta=m,ci_low=l,ci_high=h,wilcoxon_p=float(wilcoxon(d).pvalue),
                            improved=int((d>1e-12).sum()),deteriorated=int((d < -1e-12).sum()),unchanged=int((abs(d)<=1e-12).sum()),
                            difference_marginal_medians=float(paired[f"{metric}_397b"].median()-paired[f"{metric}_27b"].median()),
                            descriptive_change_above_0_01=int((d>.01).sum()),descriptive_change_below_minus_0_01=int((d<-.01).sum())))
    paired.to_csv(OUT/"paired_27b_397b.csv",index=False);pd.DataFrame(scaling).to_csv(OUT/"scaling_summary.csv",index=False)
    group=[]
    for b,g in original.groupby("backbone",sort=False):
        for metric in ("balanced_accuracy","auroc","delta_ba","delta_adjusted_ba","delta_auroc"):
            binvals=g.loc[g.n_classes==2,metric];multivals=g.loc[g.n_classes>2,metric]
            group.append(dict(backbone=b,metric=metric,binary_median=binvals.median(),multiclass_median=multivals.median(),
                              mann_whitney_p=mannwhitneyu(binvals,multivals,alternative="two-sided").pvalue,binary_n=len(binvals),multiclass_n=len(multivals)))
    group=pd.DataFrame(group)
    group["q_bh_exploratory"]=false_discovery_control(group.mann_whitney_p)
    group.to_csv(OUT/"binary_multiclass.csv",index=False)

    # EXPLORATORY ASSOCIATIONS: explicit components and coupling controls =========
    cov=ladder.task_covariates()
    if cov is not None:properties=properties.merge(cov,on=["dataset","outcome"],how="left")
    variables=[c for c in ("univariate_share","interaction_gain","n_development","n_predictors","n_classes","complexity_F1","N1","C2","T2","missingness","explanation_rate","prompt_words","prompt_characters","unit_coverage","category_gloss_coverage") if c in properties]
    assoc=[]
    for b,g in original.groupby("backbone",sort=False):
        data=g.merge(properties.drop(columns=["n_classes","n_holdout"]),on=["dataset","outcome"])
        for outcome_var in ("balanced_accuracy","xgb_ba","delta_ba","auroc","xgb_auroc","delta_auroc"):
            for var in variables:
                sub=data[[var,outcome_var,"n_classes","xgb_ba","xgb_auroc"]].loc[:,lambda a:~a.columns.duplicated()].dropna()
                if len(sub)<10 or sub[var].nunique()<2:continue
                r,p=spearmanr(sub[var],sub[outcome_var])
                ar,ap=np.nan,np.nan
                if outcome_var.startswith("delta") and var!="n_classes":
                    ar,ap=partial_rank(sub[var].to_numpy(),sub[outcome_var].to_numpy(),[sub["n_classes"].to_numpy(),sub["xgb_ba" if outcome_var=="delta_ba" else "xgb_auroc"].to_numpy()])
                rlow,rhigh=np.nan,np.nan
                if b in ("27b","397b_a17b_fp8") and outcome_var.startswith("delta"):
                    rng=np.random.default_rng(SEED);rs=[]
                    xx,yy=sub[var].to_numpy(),sub[outcome_var].to_numpy()
                    for _ in range(1000):
                        ii=rng.integers(0,len(sub),len(sub))
                        if np.ptp(xx[ii])>0 and np.ptp(yy[ii])>0:rs.append(spearmanr(xx[ii],yy[ii]).statistic)
                    rlow,rhigh=np.quantile(rs,[.025,.975]) if rs else (np.nan,np.nan)
                assoc.append(dict(backbone=b,target=outcome_var,variable=var,tasks=len(sub),rho=r,p=p,rho_ci_low=rlow,rho_ci_high=rhigh,partial_rho=ar,partial_p=ap))
    assoc=pd.DataFrame(assoc)
    # One deliberately broad correction family across all new task-property cells.
    assoc["q_bh_global"]=np.nan;valid=assoc.p.notna();assoc.loc[valid,"q_bh_global"]=false_discovery_control(assoc.loc[valid,"p"])
    assoc["partial_q_bh_global"]=np.nan;valid=assoc.partial_p.notna();assoc.loc[valid,"partial_q_bh_global"]=false_discovery_control(assoc.loc[valid,"partial_p"])
    assoc.to_csv(OUT/"exploratory_associations.csv",index=False)
    properties.to_csv(OUT/"task_properties.csv",index=False)
    # Label-budget summaries average repetitions within task before task-level inference.
    cal=tables["calibration_metrics"]
    rows=[]
    for (b,method,budget,kind),g in cal.assign(kind=np.where(cal.n_classes==2,"binary","multiclass")).groupby(["backbone","method","budget","kind"],sort=False):
        means=g.groupby(["dataset","outcome"])[["balanced_accuracy","delta_ba","remaining_gap","recovered_gap_ratio"]].mean()
        for metric in means:
            m,l,h=summary_interval(means[metric]);rows.append(dict(backbone=b,method=method,budget=budget,kind=kind,metric=metric,
                                                                 tasks=len(means),median=m,ci_low=l,ci_high=h))
    pd.DataFrame(rows).to_csv(OUT/"calibration_summary.csv",index=False)
    for path in list((ROOT/"src/jevtest").glob("revision_*.py"))+[Path(__file__),ROOT/"scripts/run_revision_inference.py"]:
        manifest["input_sha256"][str(path.relative_to(ROOT))]=digest(path)
    save_json(OUT/"execution_manifest.json",manifest)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
