"""XGB_optuna hold-out arm: split, weights, decision rule, leakage, and the imbalance case."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from jevtest.data import load
from jevtest import xgb_optuna as xo


def test_holdout_split_is_deterministic_stratified_and_disjoint():
    y = np.array([0] * 80 + [1] * 20)
    dev, hold = xo.holdout_split(y, 0.2, 7)
    dev2, hold2 = xo.holdout_split(y, 0.2, 7)
    assert dev.tolist() == dev2.tolist() and hold.tolist() == hold2.tolist()
    assert not set(dev) & set(hold) and len(dev) + len(hold) == 100
    assert len(hold) == 20 and np.bincount(y[hold]).tolist() == [16, 4]
    assert np.all(np.diff(dev) > 0) and np.all(np.diff(hold) > 0)
    _, hold3 = xo.holdout_split(y, 0.2, 8)
    assert hold3.tolist() != hold.tolist()


def test_inner_folds_never_see_the_holdout():
    y = np.array([0] * 60 + [1] * 15)
    dev, hold = xo.holdout_split(y, 0.2, 1)
    folds = xo.inner_folds(y[dev], 5, 1)
    assert len(folds) == 5
    for train, valid in folds:
        # indices are positions within the development set; map back and check against hold-out
        assert not set(dev[train]) & set(hold) and not set(dev[valid]) & set(hold)
        assert not set(train) & set(valid)
    assert len(xo.inner_folds(np.array([0] * 20 + [1] * 3), 5, 0)) == 3  # capped by the smallest class


def test_class_weights_exponent_endpoints():
    y = np.array([0] * 90 + [1] * 10)
    assert np.allclose(xo.class_weights(y, 0.0, 2), 1.0)
    w = xo.class_weights(y, 1.0, 2)
    assert np.isclose(w[y == 0][0], 100 / (2 * 90)) and np.isclose(w[y == 1][0], 100 / (2 * 10))
    assert np.isclose(w[y == 0].sum(), w[y == 1].sum())  # fully balanced total mass per class
    half = xo.class_weights(y, 0.5, 2)
    assert w[y == 1][0] > half[y == 1][0] > 1.0


def test_decision_rule_reduces_to_argmax_at_defaults():
    rng = np.random.default_rng(0)
    P2 = rng.dirichlet([1, 1], 50)
    assert xo.decide(P2, None, threshold=0.5).tolist() == P2.argmax(axis=1).tolist()
    assert xo.decide(P2, None, threshold=0.1).sum() >= xo.decide(P2, None, threshold=0.9).sum()
    P3 = rng.dirichlet([1, 1, 1], 50)
    prior = np.array([0.7, 0.2, 0.1])
    assert xo.decide(P3, prior, prior_exponent=0.0).tolist() == P3.argmax(axis=1).tolist()
    # dividing by the prior shifts decisions toward the rarer classes
    shifted = xo.decide(P3, prior, prior_exponent=1.0)
    assert (shifted == 0).sum() <= (P3.argmax(axis=1) == 0).sum()


def test_tuned_rule_beats_argmax_on_imbalanced_synthetic_task(root):
    """Argmax on a prevalence-calibrated learner collapses to the majority class; the tuned rule does not."""
    frame = pd.read_csv(root / "datasets/synthetic/data.csv")
    rng = np.random.default_rng(3)
    # weak signal, 12% positives, so the untuned argmax predicts almost only the majority class
    latent = 0.08 * (frame["score"].fillna(10) - 10) + rng.normal(0, 1, len(frame))
    frame["outcome_bin"] = (latent > np.quantile(latent, 0.88)).astype(int)
    frame.to_csv(root / "datasets/synthetic/data.csv", index=False)
    path = root / "configs/study.yaml"
    study = yaml.safe_load(path.read_text())
    study["holdout"].update(optuna_trials=15, max_minutes=2, threads=1, bootstrap_iterations=50)
    path.write_text(yaml.safe_dump(study))

    run = load(root, "synthetic", "bin")
    p = np.clip(0.3 + 0.4 * rng.random(len(run.y)), 0.01, 0.99)
    run.save_predictions("ZS_raw", np.c_[1 - p, p])
    metrics = xo.run_holdout(run)
    hold = metrics["holdout"]
    assert metrics["n_holdout"] == 30 and metrics["n_development"] == 120
    assert hold["XGB_optuna"]["balanced_accuracy"] >= hold["XGB_optuna_argmax"]["balanced_accuracy"] - 0.15
    assert 0.05 <= metrics["best_params"]["threshold"] <= 0.95
    assert 0.0 <= metrics["best_params"]["class_weight_exponent"] <= 1.0
    for arm in hold.values():
        assert 0 <= arm["balanced_accuracy"] <= 1 and 0 <= arm["auroc"] <= 1

    out = run.out / "holdout"
    split = json.loads((out / "split.json").read_text())
    assert len(split["holdout"]) == 30 and not set(split["holdout"]) & set(split["development"])
    xgb = pd.read_csv(out / "XGB_optuna.csv", dtype={"patient_id": str})
    zs = pd.read_csv(out / "ZS_raw.csv", dtype={"patient_id": str})
    assert xgb.patient_id.tolist() == split["holdout"] == zs.patient_id.tolist()
    assert np.array_equal(xgb.observed, zs.observed)
    trials = pd.read_csv(out / "trials.csv")
    assert (trials.state == "COMPLETE").sum() >= 1 and len(trials) <= 15
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["files"]["XGB_optuna.csv"] and manifest["versions"]["optuna"]


def test_holdout_config_does_not_invalidate_other_arms(root):
    run = load(root, "synthetic", "bin")
    p = np.full(len(run.y), 0.5)
    run.save_predictions("ZS_raw", np.c_[1 - p, p])
    path = root / "configs/study.yaml"
    study = yaml.safe_load(path.read_text())
    study["holdout"]["optuna_trials"] = 999
    path.write_text(yaml.safe_dump(study))
    assert load(root, "synthetic", "bin").predictions("ZS_raw") is not None


def test_full_arm_excludes_every_copy_of_a_holdout_patient(root):
    """Whole-table training must drop all rows identical to a hold-out patient, and reuse the primary split."""
    from conftest import synthetic_frame

    folder = root / "datasets/synthetic"
    sub = pd.read_csv(folder / "data.csv")
    # whole table: fresh rows + the readout rows under different ids + a duplicate copy of 10 readout rows
    extra = synthetic_frame(n=400, seed=9)
    extra["record_id"] = [f"FULL-{i:04d}" for i in range(len(extra))]
    copies = sub.iloc[:10].copy()
    copies["record_id"] = [f"DUP-{i:04d}" for i in range(10)]
    renamed = sub.copy()
    renamed["record_id"] = [f"REN-{i:04d}" for i in range(len(sub))]
    full = pd.concat([extra, renamed, copies], ignore_index=True)
    full.to_csv(folder / "patients_full.csv", index=False)

    path = root / "configs/study.yaml"
    study = yaml.safe_load(path.read_text())
    study["holdout"].update(optuna_trials=6, max_minutes=1, threads=1, bootstrap_iterations=30)
    study["holdout"]["full"] = {"tuning_cap": 300, "optuna_trials": 6, "max_minutes": 1}
    path.write_text(yaml.safe_dump(study))

    run = load(root, "synthetic", "bin")
    p = np.clip(0.3 + 0.4 * np.random.default_rng(1).random(len(run.y)), 0.01, 0.99)
    run.save_predictions("ZS_raw", np.c_[1 - p, p])
    with pytest.raises(Exception, match="split.json"):
        xo.run_holdout_full(run)
    xo.run_holdout(run)
    with pytest.raises(Exception, match="readout_to_full"):
        xo.run_holdout_full(run)

    # Build the value map the way scripts/map_readout_to_full.py does (ids are pseudonymised, so by value).
    import importlib.util
    spec = importlib.util.spec_from_file_location("map_readout_to_full",
                                                  xo.Path(__file__).resolve().parents[1] / "scripts/map_readout_to_full.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapping = module.build_map(folder)
    assert mapping["method"] == "value" and not mapping["unmatched"]
    assert mapping["n_readout_with_duplicates_in_full"] == 10
    (folder / "readout_to_full.json").write_text(json.dumps(mapping))
    # The map is keyed by source ids; the run uses pseudonyms, and the arm must translate both sides.
    from jevtest.data import pseudonyms
    keyed = {pseud: mapping["rows"][src] for src, pseud in zip(sub.record_id.astype(str), pseudonyms(root, sub.record_id))}

    metrics = xo.run_holdout_full(run)
    split = json.loads((run.out / "holdout/split.json").read_text())
    n_hold = len(split["holdout"])
    dup_in_hold = sum(1 for rid in split["holdout"] if len(keyed[rid]) > 1)
    assert metrics["n_holdout"] == n_hold
    assert metrics["n_full_rows_excluded_as_holdout"] == n_hold + dup_in_hold
    assert metrics["n_development"] == len(full) - n_hold - dup_in_hold
    assert "XGB_optuna_full" in metrics["holdout"] and "XGB_optuna" in metrics["holdout"]
    assert "XGB_optuna_full_minus_XGB_optuna" in metrics["paired"]
    table = pd.read_csv(run.out / "holdout/full/XGB_optuna_full.csv", dtype={"patient_id": str})
    assert table.patient_id.tolist() == split["holdout"]


def test_consensus_params_median_and_mode():
    import optuna

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))

    def objective(trial):
        d = trial.suggest_int("max_depth", 2, 10)
        trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        trial.suggest_float("threshold", 0.05, 0.95)
        return float(d)

    study.optimize(objective, n_trials=8)
    consensus = xo.consensus_params(study, top=5)
    assert isinstance(consensus["max_depth"], int) and consensus["grow_policy"] in {"depthwise", "lossguide"}
    assert 0.05 <= consensus["threshold"] <= 0.95
