"""Second-backbone studies: JEVTEST_STUDY override, per-study outputs dir, shared M2, re-scored hold-out arm."""
import numpy as np
import pytest
import yaml

from jevtest import xgb_optuna as xo
from jevtest.config import GateError, outputs_root, read_json, study_path
from jevtest.data import load


def second_study(root, monkeypatch, **backbone):
    study = yaml.safe_load((root / "configs/study.yaml").read_text())
    study["backbone"].update(model_id="Qwen/Qwen3-32B", revision="other", **backbone)
    study["outputs_dir"] = "outputs_second"
    path = root / "configs/study_second.yaml"
    path.write_text(yaml.safe_dump(study))
    monkeypatch.setenv("JEVTEST_STUDY", "configs/study_second.yaml")
    return path


def test_study_override_selects_file_and_outputs_dir(root, monkeypatch):
    assert study_path(root) == root / "configs/study.yaml"
    second_study(root, monkeypatch)
    assert study_path(root) == root / "configs/study_second.yaml"
    assert outputs_root(root) == root / "outputs_second"
    run = load(root, "synthetic", "bin")
    assert run.out == root / "outputs_second/synthetic/bin"
    assert run.study["backbone"]["model_id"] == "Qwen/Qwen3-32B"
    monkeypatch.setenv("JEVTEST_STUDY", "configs/missing.yaml")
    with pytest.raises(GateError, match="JEVTEST_STUDY"):
        load(root, "synthetic", "bin")


def test_placement_kwargs_follow_the_backbone_block(root, monkeypatch):
    from jevtest.llm import placement_kwargs

    base = load(root, "synthetic", "bin")
    assert placement_kwargs(base.study) == {}
    second_study(root, monkeypatch, device_map="auto", max_memory={"0": "70GiB", "cpu": "100GiB"})
    sharded = load(root, "synthetic", "bin")
    kwargs = placement_kwargs(sharded.study)
    assert kwargs["device_map"] == "auto" and kwargs["max_memory"] == {0: "70GiB", "cpu": "100GiB"}
    # Placement knobs are not part of the readout's provenance.
    assert "device_map" not in base.prediction_key("ZS_raw")["study"]["backbone"]
    assert "device_map" not in sharded.prediction_key("ZS_raw")["study"]["backbone"]
    assert "outputs_dir" not in sharded.prediction_key("ZS_raw")["study"]


def test_rescored_holdout_reuses_tuning_and_rescores_readout(root, monkeypatch):
    path = root / "configs/study.yaml"
    study = yaml.safe_load(path.read_text())
    study["holdout"].update(optuna_trials=4, max_minutes=1, threads=1, bootstrap_iterations=20)
    path.write_text(yaml.safe_dump(study))
    rng = np.random.default_rng(0)

    base = load(root, "synthetic", "bin")
    p = np.clip(0.3 + 0.4 * rng.random(len(base.y)), 0.01, 0.99)
    base.save_predictions("ZS_raw", np.c_[1 - p, p])
    base.save_predictions("M2", np.eye(2)[base.y] * 0.6 + 0.2)
    tuned = xo.run_holdout(base)

    second_study(root, monkeypatch)
    second = load(root, "synthetic", "bin")
    assert second.out != base.out
    # The shared classical fit validates under the second study without refitting.
    for name in ("M2.csv", "M2.meta.json"):
        (second.out / "predictions").mkdir(parents=True, exist_ok=True)
        (second.out / "predictions" / name).write_bytes((base.out / "predictions" / name).read_bytes())
    assert np.allclose(second.predictions("M2"), base.predictions("M2"))

    with pytest.raises(GateError, match="ZS_raw"):
        xo.rescore_holdout(second, root / "outputs")
    q = np.clip(0.2 + 0.6 * rng.random(len(second.y)), 0.01, 0.99)
    second.save_predictions("ZS_raw", np.c_[1 - q, q])
    rescored = xo.rescore_holdout(second, root / "outputs")

    assert rescored["best_params"] == tuned["best_params"]
    assert rescored["holdout"]["XGB_optuna"] == tuned["holdout"]["XGB_optuna"]
    assert rescored["holdout"]["ZS_raw"] != tuned["holdout"]["ZS_raw"]
    assert rescored["rescored_from"].endswith("outputs/synthetic/bin/holdout")
    split = read_json(second.out / "holdout/split.json")
    assert split["holdout"] == read_json(base.out / "holdout/split.json")["holdout"]
    manifest = read_json(second.out / "holdout/manifest.json")
    assert manifest["zs_raw_csv_sha256"] != read_json(base.out / "holdout/manifest.json")["zs_raw_csv_sha256"]
    assert (second.out / "holdout/model.joblib").exists()


def test_holdout_only_zero_shot_artifact_feeds_the_rescored_arm(root, monkeypatch):
    from jevtest import zeroshot as zs

    path = root / "configs/study.yaml"
    study = yaml.safe_load(path.read_text())
    study["holdout"].update(optuna_trials=4, max_minutes=1, threads=1, bootstrap_iterations=20)
    path.write_text(yaml.safe_dump(study))
    rng = np.random.default_rng(1)
    base = load(root, "synthetic", "bin")
    p = np.clip(0.3 + 0.4 * rng.random(len(base.y)), 0.01, 0.99)
    base.save_predictions("ZS_raw", np.c_[1 - p, p])
    xo.run_holdout(base)

    second_study(root, monkeypatch)
    second = load(root, "synthetic", "bin")
    hold = zs.holdout_rows(second)
    from jevtest.config import read_json as rj
    assert second.ids[hold].tolist() == rj(base.out / "holdout/split.json")["holdout"]
    q = np.clip(0.2 + 0.6 * rng.random(len(hold)), 0.01, 0.99)
    zs.save_holdout_predictions(second, np.c_[1 - q, q], hold)
    assert (second.out / "predictions/ZS_raw_holdout.csv").exists()
    assert not (second.out / "predictions/ZS_raw.csv").exists()
    P = zs.holdout_zero_shot_probs(second, hold)
    assert np.allclose(P[:, 1], q, atol=1e-9)
    rescored = xo.rescore_holdout(second, root / "outputs")
    assert rescored["holdout"]["ZS_raw"]["n"] == len(hold) if "n" in rescored["holdout"]["ZS_raw"] else True
    manifest = read_json(second.out / "holdout/manifest.json")
    assert manifest["zs_raw_source"] == "predictions/ZS_raw_holdout.csv"
    # A hold-out artifact made for other patients is refused.
    (second.out / "predictions/ZS_raw_holdout.csv").write_text(
        (second.out / "predictions/ZS_raw_holdout.csv").read_text().replace(second.ids[hold][0], "SRC-9999"))
    with pytest.raises(GateError, match="ZS_raw_holdout"):
        zs.holdout_zero_shot_probs(second, hold)
