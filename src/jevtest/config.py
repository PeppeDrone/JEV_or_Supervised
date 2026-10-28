"""Configuration, local I/O and the per-run context shared by every stage."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib, json, os, time

import numpy as np
import pandas as pd
import yaml


class GateError(RuntimeError):
    """A check refused the run; fix the input instead of working around it."""


def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path, obj):
    """Atomic replace so a kill mid-write cannot leave a truncated JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=json_default, allow_nan=False))
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else Path(data).read_bytes()).hexdigest()


def study_path(root):
    """configs/study.yaml unless JEVTEST_STUDY names another study file (absolute or relative to root)."""
    override = os.environ.get("JEVTEST_STUDY")
    if not override:
        return Path(root) / "configs/study.yaml"
    path = Path(override)
    if not path.is_absolute():
        path = Path(root) / path
    if not path.exists():
        raise GateError(f"JEVTEST_STUDY points to a missing file: {path}")
    return path


def study_config(root):
    return read_yaml(study_path(root))


def outputs_dir(root, study):
    """Per-study output root: study['outputs_dir'] (default 'outputs'), relative to root unless absolute."""
    path = Path(study.get("outputs_dir") or "outputs")
    return path if path.is_absolute() else Path(root) / path


def outputs_root(root):
    """Output root of the active study (configs/study.yaml or JEVTEST_STUDY); for scripts that read outputs."""
    return outputs_dir(root, study_config(root))


def dataset_dir(root, name):
    return Path(root) / "datasets" / name


def dataset_config(root, name):
    path = dataset_dir(root, name) / "dataset.yaml"
    if not path.exists():
        raise GateError(f"No dataset spec at {path}")
    return read_yaml(path)


@dataclass
class Run:
    """One dataset x outcome: labelled patients, class indices and the fixed outer folds."""
    root: Path
    study: dict
    ds: dict
    dataset: str
    outcome: str
    frame: pd.DataFrame
    y: np.ndarray
    splits: list
    out: Path
    data_hash: str
    fid: np.ndarray = field(init=False)

    def __post_init__(self):
        self.fid = np.zeros(len(self.y), dtype=int)
        for fold, (_, test) in enumerate(self.splits):
            self.fid[test] = fold

    @property
    def spec(self):
        return self.ds["outcomes"][self.outcome]

    @property
    def classes(self):
        return list(self.spec["classes"])

    @property
    def ids(self):
        return self.frame.patient_id.to_numpy(dtype=str)

    def _prediction_error(self, model, reason):
        stage = {"M2": "classical", "ZS_raw": "zeroshot",
                 "ICL_full": "icl", "Mean": "ensemble", "Mode": "ensemble"}.get(model, "the producing")
        return GateError(f"{model} predictions rejected: {reason}; rerun the `{stage}` stage "
                         f"for {self.dataset}/{self.outcome}")

    @staticmethod
    def _strip_noncausal_study(study, model):
        """Drop study knobs that do not change a model's predictions (legacy metas included)."""
        probe = study.get("probe")
        if isinstance(probe, dict):
            probe.pop("hybrid_heads", None)
            probe.pop("hybrid_pca", None)
        # Throughput / checkpoint knobs do not change AnyJev L0 probabilities for a fixed backbone.
        backbone = study.get("backbone")
        if isinstance(backbone, dict):
            backbone.pop("anyjev_checkpoint_every", None)
            backbone.pop("anyjev_batch", None)
            # Placement and determinism-warning knobs for sharded backbones do not change the readout.
            backbone.pop("device_map", None)
            backbone.pop("max_memory", None)
            backbone.pop("deterministic_warn_only", None)
            # Extra from_pretrained kwargs (kernel selection, loader options) do not change the readout.
            backbone.pop("model_kwargs", None)
        # Where outputs are written is not part of the experiment identity.
        study.pop("outputs_dir", None)
        # Ensemble membership only affects Mean/Mode (and is also bound via member file digests).
        if model not in ("Mean", "Mode"):
            study.pop("ensemble", None)
        # The classical arm never touches the language model: its provenance is the same under every
        # backbone study, so one M2 fit can be shared across outputs dirs (scripts/queue_pod.sh).
        if model == "M2":
            for key in ("backbone", "icl", "probe", "zero_shot_calibration_C"):
                study.pop(key, None)
        # The hold-out XGB_optuna arm has its own provenance (holdout/manifest.json).
        study.pop("holdout", None)
        return study

    def _study_provenance(self, model):
        """Study settings that affect this model. Unused probe knobs must not invalidate others."""
        study = json.loads(json.dumps(self.study, default=json_default, allow_nan=False))
        return self._strip_noncausal_study(study, model)

    def prediction_key(self, model):
        """Identity of the loaded experiment, including the actual train/test patients."""
        key = {"version": 1, "model": model, "dataset": self.dataset, "outcome": self.outcome,
               "data_hash": self.data_hash, "dataset_spec": self.ds, "study": self._study_provenance(model),
               "classes": self.classes, "patient_ids": self.ids.tolist(), "observed": self.y.tolist(),
               "folds": [{"train": self.ids[train].tolist(), "test": self.ids[test].tolist()}
                         for train, test in self.splits]}
        if model in ("Mean", "Mode"):
            # A member can be regenerated under identical settings. Bind the ensemble
            # to its exact input files as well as to the experiment settings.
            try:
                key["members"] = {
                    name: {"csv": digest(self.out / f"predictions/{name}.csv"),
                           "metadata": digest(self.out / f"predictions/{name}.meta.json")}
                    for name in self.study["ensemble"]["members"]
                }
            except OSError as error:
                raise self._prediction_error(model, "ensemble input files or metadata are missing") from error
        # Freeze nested settings and use the same JSON representation on save/load.
        return json.loads(json.dumps(key, default=json_default, allow_nan=False))

    def save_predictions(self, model, P, labels=None):
        """Save out-of-fold probabilities and provenance; Mode labels share the checked CSV."""
        P = np.asarray(P, dtype=float)
        if (P.shape != (len(self.y), len(self.classes)) or not np.isfinite(P).all()
                or (P < 0).any() or (P.sum(axis=1) <= 0).any()):
            raise self._prediction_error(model, "invalid class probabilities")
        key = self.prediction_key(model)
        P = P / P.sum(axis=1, keepdims=True)  # float32 learners (XGBoost) sum to 1 only within ~1e-7
        table = pd.DataFrame({"model": model, "fold": self.fid, "patient_id": self.ids, "observed": self.y})
        for c, name in enumerate(self.classes):
            table[f"p__{name}"] = P[:, c]
        if labels is not None:
            labels = np.asarray(labels)
            if labels.shape != self.y.shape or not np.isin(labels, np.arange(len(self.classes))).all():
                raise self._prediction_error(model, "invalid predicted labels")
            table["predicted"] = labels.astype(int)
        elif model == "Mode":
            raise self._prediction_error(model, "Mode requires its tie-broken labels")
        path = self.out / f"predictions/{model}.csv"
        save_table(table, path)
        write_json(path.with_suffix(".meta.json"), {"key": key, "csv_sha256": digest(path)})

    def _prediction_table(self, model):
        """Reject legacy, stale or replaced artifacts before any consumer can use them."""
        path = self.out / f"predictions/{model}.csv"
        if not path.exists():
            return None
        meta = path.with_suffix(".meta.json")
        if not meta.exists():
            raise self._prediction_error(model, "provenance metadata is missing (legacy CSV)")
        try:
            stored = read_json(meta)
            if not isinstance(stored, dict) or not isinstance(stored.get("key"), dict):
                raise ValueError("invalid metadata structure")
        except (OSError, ValueError) as error:
            raise self._prediction_error(model, "provenance metadata is unreadable or invalid") from error
        expected = self.prediction_key(model)
        stored_key = json.loads(json.dumps(stored["key"], default=json_default, allow_nan=False))
        if isinstance(stored_key.get("study"), dict):
            self._strip_noncausal_study(stored_key["study"], model)
        differences = [name for name, value in expected.items() if stored_key.get(name) != value]
        if differences or stored_key.keys() != expected.keys():
            raise self._prediction_error(model, "stale provenance (" + ", ".join(differences or ["schema"]) + ")")
        try:
            if stored.get("csv_sha256") != digest(path):
                raise self._prediction_error(model, "CSV checksum differs from its provenance metadata")
            table = pd.read_csv(path, dtype={"patient_id": str})
        except (OSError, ValueError, pd.errors.ParserError) as error:
            raise self._prediction_error(model, "prediction CSV is unreadable or invalid") from error
        columns = [f"p__{c}" for c in self.classes]
        required = {"model", "fold", "patient_id", "observed", *columns}
        if model == "Mode":
            required.add("predicted")
        if (not required <= set(table.columns) or len(table) != len(self.y)
                or not (table["model"] == model).all()
                or not np.array_equal(table.patient_id, self.ids)
                or not np.array_equal(table.observed, self.y)
                or not np.array_equal(table.fold, self.fid)):
            raise self._prediction_error(model, "CSV patients, labels, model or folds do not match")
        try:
            P = table[columns].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise self._prediction_error(model, "invalid class probabilities") from error
        if (not np.isfinite(P).all() or (P < 0).any() or (P > 1).any()
                or not np.allclose(P.sum(axis=1), 1, rtol=0, atol=1e-6)):
            raise self._prediction_error(model, "invalid class probabilities")
        if model == "Mode" and not table.predicted.isin(range(len(self.classes))).all():
            raise self._prediction_error(model, "invalid predicted labels")
        return table

    def predictions(self, model):
        table = self._prediction_table(model)
        if table is None:
            return None
        return table[[f"p__{c}" for c in self.classes]].to_numpy()

    def prediction_labels(self, model):
        """Read optional tie-broken labels from the same validated prediction artifact."""
        table = self._prediction_table(model)
        return table["predicted"].to_numpy(dtype=int) if table is not None and "predicted" in table else None

    def progress(self, **values):
        write_json(self.out / "progress.json", dict(values, time=time.strftime("%Y-%m-%dT%H:%M:%S")))
        print(json.dumps(values), flush=True)
