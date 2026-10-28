import numpy as np
import pandas as pd
import pytest

from conftest import FakeTokenizer, set_study
from jevtest.config import GateError
from jevtest.data import load
from jevtest.ensembles import mean, mode
from jevtest.icl import ICLPrompt, plan_context, quotas, run_icl, table_rows
from jevtest.llm import answer_instruction, options_block, orders
from jevtest.metrics import auroc, point
from jevtest.text import render


def test_prompt_text_has_no_identifiers_or_outcomes(root):
    run = load(root, "synthetic", "bin")
    raw = pd.read_csv(root / "datasets/synthetic/data.csv")
    texts = [render(row, run.ds) for _, row in run.frame.iterrows()]
    assert all("SRC-" not in t and "outcome" not in t and "POOR" not in t and "GOOD" not in t for t in texts)
    assert not set(run.ids) & set(raw.record_id)  # pseudonymised
    assert any("Severity score (0–20; higher is better): not observed" in t for t in texts)
    assert any("EEG result: not applicable (EEG not performed)" in t for t in texts)


def test_untranslated_code_is_rejected(root):
    path = root / "datasets/synthetic/data.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "sex"] = 9
    frame.to_csv(path, index=False)
    with pytest.raises(GateError, match="Untranslated"):
        load(root, "synthetic", "bin")


def test_option_orders_match_neurojev_binary_prompt():
    assert orders(2) == [[0, 1], [1, 0]]
    assert answer_instruction(2) == "Answer with A or B."
    assert answer_instruction(3) == "Answer with A, B or C."
    assert options_block("Q?", ["X", "Y"], [1, 0]) == "Question: Q?\nA. Y\nB. X\nAnswer with A or B."
    assert sorted(map(tuple, orders(3))) == [(0, 1, 2), (1, 2, 0), (2, 0, 1)]


@pytest.mark.parametrize("outcome", ["bin", "tri"])
def test_classical_models_give_valid_posteriors(root, outcome):
    from jevtest.classical import run_classical

    run = load(root, "synthetic", outcome)
    run_classical(run)
    P = run.predictions("M2")
    assert P is not None
    assert P.shape == (len(run.y), len(run.classes)) and np.allclose(P.sum(axis=1), 1)
    assert run.predictions("M1") is None


def test_quotas_proportional_minimum_and_balanced():
    cfg = {"balanced": False, "min_per_class": 10}
    assert quotas(np.array([80, 20]), 50, cfg).tolist() == [40, 10]
    assert quotas(np.array([95, 5]), 40, cfg).tolist() == [35, 5]  # capped by the class size
    assert quotas(np.array([90, 10, 50]), 60, {"balanced": True, "min_per_class": 5}).tolist() == [25, 10, 25]


@pytest.mark.parametrize("outcome", ["bin", "tri"])
def test_context_subsample_respects_budget_and_folds(root, outcome):
    set_study(root, context_limit=1500, native_context=1500, margin_tokens=50, margin_fraction=0.0, min_per_class=5)
    run = load(root, "synthetic", outcome)
    context = plan_context(run, FakeTokenizer())
    assert context["mode"] == "subsample"
    assert {f["n_selected"] for f in context["folds"]} == {context["n"]}
    for fold, (train, test) in zip(context["folds"], run.splits):
        assert fold["max_prompt_tokens"] <= context["budget"]
        assert not set(fold["patient_ids"]) & set(run.ids[test])
        assert set(fold["patient_ids"]) <= set(run.ids[train])
        assert min(fold["class_counts"]) >= 5
    # One more row per class would not fit in at least one fold (largest common n).
    assert plan_context(run, FakeTokenizer()) == context  # deterministic


def test_context_uses_everyone_when_it_fits(root):
    run = load(root, "synthetic", "bin")
    context = plan_context(run, FakeTokenizer())
    assert context["mode"] == "all"
    assert all(f["n_selected"] == f["n_available"] for f in context["folds"])


def test_balanced_subsample(root):
    set_study(root, context_limit=1500, native_context=1500, margin_tokens=50, margin_fraction=0.0, min_per_class=5,
              balanced=True)
    run = load(root, "synthetic", "bin")
    counts = [f["class_counts"] for f in plan_context(run, FakeTokenizer())["folds"]]
    assert all(abs(a - b) <= 1 for a, b in counts)


def test_context_too_small_is_a_gate(root):
    set_study(root, context_limit=300, native_context=300, margin_tokens=10, margin_fraction=0.0)
    with pytest.raises(GateError):
        plan_context(load(root, "synthetic", "bin"), FakeTokenizer())


def test_icl_refuses_stale_context(root):
    run = load(root, "synthetic", "bin")
    plan_context(run, FakeTokenizer())
    set_study(root, min_per_class=11)
    with pytest.raises(GateError, match="stale"):
        run_icl(load(root, "synthetic", "bin"))


def test_icl_prefix_holds_no_test_patient(root):
    run = load(root, "synthetic", "bin")
    labels, rows, names = table_rows(run)
    prompt = ICLPrompt(FakeTokenizer(), run, labels)
    train, test = run.splits[0]
    prefix, suffix = prompt.parts([rows[i] for i in train], [names[i] for i in train], rows[test[0]], [0, 1])
    assert prefix.endswith(ICLPrompt.MARKER) and rows[test[0]] in suffix and "Answer with A or B." in suffix
    assert prefix.count(" -> ") == len(train)


def test_mode_tie_break_and_mean():
    a = np.array([[0.6, 0.4], [0.2, 0.8]])
    b = np.array([[0.1, 0.9], [0.3, 0.7]])
    shares, winner = mode(np.stack([a, b]))
    assert shares.tolist() == [[0.5, 0.5], [0.0, 1.0]]
    assert winner.tolist() == [1, 1]  # tie in row 0: mean posterior favours class 1
    assert np.allclose(mean(np.stack([a, b])).sum(axis=1), 1)


def test_multiclass_metrics():
    y = np.array([0, 1, 2, 0, 1, 2])
    P = np.eye(3)[y] * 0.8 + 0.2 / 3
    assert auroc(y, P) == 1.0
    assert point(y, P)["balanced_accuracy"] == 1.0


def test_ensembles_and_report_end_to_end(root):
    from jevtest.ensembles import run_ensembles
    from jevtest.report import summarise

    run = load(root, "synthetic", "bin")
    rng = np.random.default_rng(1)
    for model in run.study["ensemble"]["members"] + ["ZS_raw"]:
        p = np.clip(0.5 * run.y + rng.random(len(run.y)) * 0.5, 0.01, 0.99)
        run.save_predictions(model, np.c_[1 - p, p])
    assert run_ensembles(run)
    perf, paired = summarise(run)
    assert set(perf.model) == {"M2", "ZS_raw", "ICL_full", "Mode", "Mean"}
    assert np.isnan(perf.set_index("model").loc["Mode", "log_loss"])
    assert "| Mean |" in (run.out / "REPORT.md").read_text()
