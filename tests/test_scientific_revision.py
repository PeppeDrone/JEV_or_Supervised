"""Scientific safeguards for the additional analyses, including real-data provenance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from jevtest.revision_reference import select_marker, predict_marker, score_auc
from jevtest.revision_calibration import fit_readout, apply_readout, calibration_sample, probabilities
from jevtest.revision_rendering import render_condition
from jevtest.metrics import bootstrap

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def real_development():
    from jevtest.data import load, predictor_matrix
    run = load(ROOT, "uci_breast_coimbra", "breast_cancer")
    split = json.loads((ROOT / "outputs/uci_breast_coimbra/breast_cancer/holdout/split.json").read_text())
    positions = {p:i for i,p in enumerate(run.ids)}
    dev = [positions[p] for p in split["development"]]
    hold = [positions[p] for p in split["holdout"]]
    return run, predictor_matrix(run), dev, hold


def test_holdout_label_changes_cannot_change_marker(real_development):
    run,X,dev,hold = real_development
    a,ca = select_marker(X.iloc[dev],run.y[dev],2)
    altered = run.y.copy();altered[hold] = 1-altered[hold]
    b,cb = select_marker(X.iloc[dev],altered[dev],2)
    assert a == b
    pd.testing.assert_frame_equal(ca,cb)
    sa,la = predict_marker(a,X.iloc[hold][a["predictor"]])
    sb,lb = predict_marker(b,X.iloc[hold][b["predictor"]])
    np.testing.assert_array_equal(sa,sb);np.testing.assert_array_equal(la,lb)


@pytest.mark.parametrize("k",[2,5])
@pytest.mark.parametrize("method",["prevalence","offset","temperature","temperature_offset","operating_point"])
def test_calibration_separation_and_determinism(k,method):
    rng=np.random.default_rng(55)
    # Validation-only probabilities; never used in scientific outputs.
    P=rng.dirichlet(np.ones(k),80);y=np.arange(80)%k
    Q=rng.dirichlet(np.ones(k),21);test_y=np.arange(21)%k
    a=fit_readout(P,y,method,k)
    test_y[:]=0
    b=fit_readout(P,y,method,k)
    assert a==b
    transformed,labels=apply_readout(Q,a)
    probabilities(transformed)
    assert len(labels)==len(Q)
    if method=="temperature":np.testing.assert_array_equal(labels,Q.argmax(1))


def test_temperature_extreme_probabilities_and_ties():
    P=np.array([[1e-250,1e-200,1.],[.4,.4,.2],[0.,.1,.9]])
    for invT in (.05,1.,20.):
        transformed,labels=apply_readout(P,dict(method="temperature",inverse_temperature=invT,bias=[0,0,0]))
        np.testing.assert_array_equal(labels,P.argmax(1));probabilities(transformed)


def test_rare_and_unseen_categories():
    X=pd.DataFrame({"categorical":pd.Series(["a","a","b","b","c","c","d","d"],dtype=object),"numeric":[1,np.nan,2,3,4,5,6,7]})
    y=np.array([0,1,0,1,0,1,0,1])
    fitted,candidates=select_marker(X,y,2,folds=5)
    assert fitted["folds"]==4 and candidates.valid.all()
    s,p=predict_marker(fitted,pd.Series(["unseen"],dtype=object) if fitted["kind"]=="categorical" else pd.Series([np.nan]))
    assert np.isfinite(s).all()
    with pytest.raises(ValueError,match="two development"):
        select_marker(X.iloc[:3],np.array([0,0,1]),2)


def test_missing_class_fails_explicitly():
    P=np.full((10,3),1/3)
    with pytest.raises(ValueError,match="lacks a class"):fit_readout(P,np.zeros(10,int),"offset",3)
    assert np.isnan(score_auc(np.zeros(10,int),P))
    assert fit_readout(P,np.zeros(10,int),"prevalence",3)["missing_classes"]==2
    values=bootstrap(np.array([0,1,2]),np.eye(3),[np.array([0,0,1])])
    assert np.isnan(values["balanced_accuracy"][0]) and np.isnan(values["auroc"][0])


def test_budget_sampling_is_label_blind():
    a=calibration_sample(100,20,42);b=calibration_sample(100,20,42)
    np.testing.assert_array_equal(a,b)
    assert len(set(a))==20
    with pytest.raises(ValueError):calibration_sample(10,20,42)


def test_class_order_and_metrics():
    # Exclude ties here: argmax resolves exact ties by the first declared class.
    y=np.array([0,0,1,1,2,2]);P=np.array([[.8,.1,.1],[.6,.2,.2],[.1,.8,.1],[.2,.45,.35],[.1,.2,.7],[.4,.1,.5]])
    expected=np.mean([roc_auc_score(y==c,P[:,c]) for c in range(3)])
    assert score_auc(y,P)==pytest.approx(expected)
    perm=np.array([2,0,1]);inverse=np.argsort(perm)
    assert score_auc(inverse[y],P[:,perm])==pytest.approx(expected)
    assert balanced_accuracy_score(inverse[y],P[:,perm].argmax(1))==balanced_accuracy_score(y,P.argmax(1))


def test_anonymous_rendering_contains_no_predictor_metadata(real_development):
    run,X,dev,hold=real_development
    row=run.frame.iloc[hold[0]].copy()
    text=render_condition(row,run.ds,"anonymous")
    for info in run.ds["predictors"].values():
        assert info["text_label"] not in text
    changed=row.copy();changed[run.spec["column"]]="NOT A PROMPT INPUT"
    for condition in ("original","no_setting","no_units","anonymous","no_category_gloss","concise"):
        assert render_condition(row,run.ds,condition)==render_condition(changed,run.ds,condition)


def test_all_six_original_holdouts_and_probability_vectors():
    from jevtest.benchmark import ALL_PAIRS
    for dataset,outcome in ALL_PAIRS:
        split=json.loads((ROOT/"outputs"/dataset/outcome/"holdout/split.json").read_text())
        expected=set(split["holdout"])
        original_classes=None
        for backbone in ("0.8b","2b","4b","9b","27b","397b_a17b_fp8"):
            path=ROOT/f"outputs_qwen3.5_{backbone}"/dataset/outcome/"predictions/ZS_raw_holdout.csv"
            table=pd.read_csv(path,dtype={"patient_id":str})
            assert set(table.patient_id)==expected and not table.patient_id.duplicated().any()
            cols=[c for c in table if c.startswith("p__")]
            if original_classes is None:original_classes=cols
            assert cols==original_classes
            probabilities(table[cols].to_numpy())


def test_capped_design_subsets_are_outcome_blind_and_nested(real_development):
    """v2 27B pools/subsets depend on split identities and the seed only, never on outcomes."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference", ROOT / "scripts/run_revision_inference.py")
    inference = importlib.util.module_from_spec(spec); spec.loader.exec_module(inference)
    split = json.loads((ROOT / "outputs/uci_breast_coimbra/breast_cancer/holdout/split.json").read_text())
    a = inference.label_blind_subset(split["development"], 50, 20261008)
    b = inference.label_blind_subset(split["development"], 50, 20261008)
    assert a == b and len(set(a)) == 50 and set(a) <= set(split["development"])
    assert not set(a) & set(split["holdout"])
    assert inference.label_blind_subset(split["holdout"], 0, 1) == split["holdout"]
    assert inference.label_blind_subset(split["holdout"], 10**6, 1) == split["holdout"]
    # The function receives no labels at all: relabelling cannot change the sample.
    assert "y" not in inference.label_blind_subset.__code__.co_varnames


def test_pool_budget_samples_stay_inside_pool():
    pool = np.sort(np.random.default_rng(0).choice(500, 200, replace=False))
    for repeat in range(5):
        idx = calibration_sample(len(pool), 50, 20261008 + repeat)
        assert set(pool[idx]) <= set(pool) and len(set(idx)) == 50
