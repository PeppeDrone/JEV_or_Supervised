"""Regression checks for reusing results across changed experiments."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from jevtest.config import GateError, digest
from jevtest.data import load
from jevtest.ensembles import run_ensembles
from jevtest.report import summarise


def probabilities(run):
    return np.eye(len(run.classes))[run.y] * 0.6 + 0.4 / len(run.classes)


@pytest.mark.parametrize("outcome", ["bin", "tri"])
def test_predictions_reusable_in_unchanged_experiment(root, outcome):
    run = load(root, "synthetic", outcome)
    P = probabilities(run)
    run.save_predictions("M2", P)
    assert np.allclose(load(root, "synthetic", outcome).predictions("M2"), P)


@pytest.mark.parametrize("change,reason,model", [
    ("seed", "study", "M2"), ("predictors", "data_hash", "M2"), ("spec", "dataset_spec", "M2"),
    ("backbone", "study", "ZS_raw"), ("search", "study", "M2"),
])
def test_changed_experiment_rejects_old_predictions(root, change, reason, model):
    run = load(root, "synthetic", "bin")
    run.save_predictions(model, probabilities(run))
    if change == "predictors":
        path = root / "datasets/synthetic/data.csv"
        frame = pd.read_csv(path)
        frame["age"] += 1
        frame.to_csv(path, index=False)
    elif change == "spec":
        path = root / "datasets/synthetic/dataset.yaml"
        spec = yaml.safe_load(path.read_text())
        spec["predictors"]["age"]["text_label"] = "Changed age description"
        path.write_text(yaml.safe_dump(spec))
    else:
        path = root / "configs/study.yaml"
        study = yaml.safe_load(path.read_text())
        if change == "seed":
            study["seed"] += 1
        elif change == "backbone":
            study["backbone"]["revision"] = "different-revision"
        elif change == "search":
            study["search_iterations"] += 1
        path.write_text(yaml.safe_dump(study))
    changed = load(root, "synthetic", "bin")
    if change == "seed":
        assert not np.array_equal(run.fid, changed.fid)
    with pytest.raises(GateError, match=reason):
        changed.predictions(model)
    # Regenerating the model resolves the mismatch.
    changed.save_predictions(model, probabilities(changed))
    assert np.allclose(changed.predictions(model), probabilities(changed))


def test_backbone_change_keeps_classical_predictions(root):
    """M2 never touches the language model, so one classical fit is shared across backbone studies."""
    run = load(root, "synthetic", "bin")
    run.save_predictions("M2", probabilities(run))
    path = root / "configs/study.yaml"
    study = yaml.safe_load(path.read_text())
    study["backbone"]["model_id"] = "Qwen/Qwen3-32B"
    study["backbone"]["revision"] = "different-revision"
    study["backbone"]["device_map"] = "auto"
    study["outputs_dir"] = "outputs"
    path.write_text(yaml.safe_dump(study))
    assert np.allclose(load(root, "synthetic", "bin").predictions("M2"), probabilities(run))


def test_ensemble_membership_does_not_invalidate_classical_predictions(root):
    """Ensemble list is Mean/Mode-only; changing it must not force classical re-runs."""
    run = load(root, "synthetic", "bin")
    run.save_predictions("M2", probabilities(run))
    path = root / "configs/study.yaml"
    study = yaml.safe_load(path.read_text())
    study["ensemble"]["members"] = ["M2", "ZS_raw"]
    path.write_text(yaml.safe_dump(study))
    assert np.allclose(load(root, "synthetic", "bin").predictions("M2"), probabilities(run))


def test_actual_training_split_is_checked_even_when_seed_is_unchanged(root):
    run = load(root, "synthetic", "bin")
    run.save_predictions("M2", probabilities(run))
    # Same test assignments but a different training set must also be rejected.
    train, test = run.splits[0]
    run.splits[0] = (train[1:], test)
    with pytest.raises(GateError, match="folds"):
        run.predictions("M2")


def test_legacy_predictions_are_not_silently_adopted(root):
    run = load(root, "synthetic", "bin")
    run.save_predictions("M2", probabilities(run))
    (run.out / "predictions/M2.meta.json").unlink()
    with pytest.raises(GateError, match="metadata is missing.*classical"):
        run.predictions("M2")


@pytest.mark.parametrize("command", ["report", "ensemble"])
def test_cli_returns_quality_gate_for_legacy_predictions(root, capsys, command):
    from jevtest.cli import main

    run = load(root, "synthetic", "bin")
    run.save_predictions("M2", probabilities(run))
    (run.out / "predictions/M2.meta.json").unlink()
    code = main([command, "--dataset", "synthetic", "--outcome", "bin", "--root", str(root)])
    assert code == 2
    assert "QUALITY GATE: M2 predictions rejected" in capsys.readouterr().err


def test_replaced_csv_is_rejected(root):
    run = load(root, "synthetic", "bin")
    run.save_predictions("M2", probabilities(run))
    path = run.out / "predictions/M2.csv"
    table = pd.read_csv(path)
    table.loc[0, "p__GOOD"] = 0.99
    table.to_csv(path, index=False)
    with pytest.raises(GateError, match="checksum"):
        run.predictions("M2")


def test_csv_fold_column_is_checked_against_current_assignments(root):
    run = load(root, "synthetic", "bin")
    run.save_predictions("M2", probabilities(run))
    path = run.out / "predictions/M2.csv"
    table = pd.read_csv(path)
    table["fold"] = (table["fold"] + 1) % len(run.splits)
    table.to_csv(path, index=False)
    # Even if the file checksum is updated, inconsistent fold columns are invalid.
    meta = path.with_suffix(".meta.json")
    stored = json.loads(meta.read_text())
    stored["csv_sha256"] = digest(path)
    meta.write_text(json.dumps(stored))
    with pytest.raises(GateError, match="CSV.*folds"):
        run.predictions("M2")


@pytest.mark.parametrize("contents", ["{", "[]", '{"key": null}', '{"key": {"version": 999}}'])
def test_invalid_provenance_is_a_quality_gate(root, contents):
    run = load(root, "synthetic", "bin")
    run.save_predictions("M2", probabilities(run))
    (run.out / "predictions/M2.meta.json").write_text(contents)
    with pytest.raises(GateError, match="M2 predictions rejected"):
        run.predictions("M2")


def test_report_and_ensemble_refuse_mixed_experiments(root):
    run = load(root, "synthetic", "bin")
    for model in run.study["ensemble"]["members"]:
        run.save_predictions(model, probabilities(run))
    path = root / "configs/study.yaml"
    study = yaml.safe_load(path.read_text())
    study["seed"] += 1
    path.write_text(yaml.safe_dump(study))
    changed = load(root, "synthetic", "bin")
    # All members still carry the old seed; first ensemble member is M2.
    with pytest.raises(GateError, match="M2 predictions rejected"):
        run_ensembles(changed)
    with pytest.raises(GateError, match="M2 predictions rejected"):
        summarise(changed)
    assert not (run.out / "predictions/Mean.csv").exists()
    assert not (run.out / "performance.csv").exists()


def test_ensemble_rejects_changed_member_outputs_and_keeps_mode_labels_together(root):
    run = load(root, "synthetic", "bin")
    for model in run.study["ensemble"]["members"]:
        run.save_predictions(model, probabilities(run))
    assert run_ensembles(run)
    shares = run.predictions("Mode")
    labels = run.prediction_labels("Mode")
    assert np.array_equal(labels, run.y)
    assert len(shares) == len(run.y)
    # An obsolete separate label file cannot override labels in the checked CSV.
    np.save(run.out / "predictions/Mode_labels.npy", 1 - labels)
    assert np.array_equal(run.prediction_labels("Mode"), labels)
    run.save_predictions("M2", probabilities(run)[:, ::-1])
    for model in ("Mean", "Mode"):
        with pytest.raises(GateError, match="members"):
            run.predictions(model)
    assert run_ensembles(run)
    assert run.predictions("Mean") is not None


@pytest.mark.parametrize("row", [[-0.2, 1.2], [0.0, 0.0], [np.nan, 1.0]])
def test_invalid_probabilities_cannot_be_signed_with_provenance(root, row):
    run = load(root, "synthetic", "bin")
    with pytest.raises(GateError, match="invalid class probabilities"):
        run.save_predictions("M2", np.tile(row, (len(run.y), 1)))
