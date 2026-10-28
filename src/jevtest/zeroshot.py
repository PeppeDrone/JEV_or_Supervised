"""ZS_raw: frozen zero-shot AnyJev L0 readout."""
import json

import numpy as np

from jevtest.config import GateError, read_json, write_json
from jevtest.text import render

# Flush partial probs to disk this often so a kill does not throw away a long GPU run.
DEFAULT_CHECKPOINT_EVERY = 10


def cache_key(run):
    b = run.study["backbone"]
    key = {"model_id": b["model_id"], "revision": b["revision"], "anyjev_commit": b["anyjev_commit"],
           "question": run.spec["question"], "classes": run.classes, "data_hash": run.data_hash}
    # Only non-default L0 prior settings enter the key, so existing content-free caches stay valid.
    if b.get("anyjev_prior", "content_free") != "content_free":
        key["prior"] = b["anyjev_prior"]
    if b.get("anyjev_prior_strength") is not None:
        key["prior_strength"] = float(b["anyjev_prior_strength"])
    return key


def _load_partial(path, key):
    if not path.exists():
        return {}
    try:
        partial = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if partial.get("key") != key or not isinstance(partial.get("probs"), dict):
        return {}
    return {str(pid): probs for pid, probs in partial["probs"].items()}


def _save_partial(path, key, probs_map):
    write_json(path, {"key": key, "probs": probs_map})


def zero_shot_probs(run, rows=None):
    """Cached P(class) per patient; recomputed only if the backbone, question, classes or data change.

    Long runs also checkpoint into ``cache/zs_partial.json`` every N patients and resume from it
    if the process dies before the final ``cache/zs.json`` is written. With ``rows`` (row indices),
    only those patients are scored; they stay in the partial cache so a later full run reuses them,
    and the returned array is aligned with ``rows``.
    """
    path = run.out / "cache/zs.json"
    partial_path = run.out / "cache/zs_partial.json"
    key = cache_key(run)
    wanted = list(range(len(run.ids))) if rows is None else [int(i) for i in rows]
    if path.exists():
        cached = read_json(path)
        if cached["key"] == key:
            return np.array([cached["probs"][str(run.ids[i])] for i in wanted])

    probs_map = _load_partial(partial_path, key)
    n = len(wanted)
    pending = [i for i in wanted if str(run.ids[i]) not in probs_map]
    if pending:
        from anyjev import Question
        from jevtest.llm import anyjev_runtime

        backend, decider = anyjev_runtime(run.study)
        q = Question.choice(run.spec["question"], run.classes, name=run.outcome)
        states = {i: render(run.frame.iloc[i], run.ds) for i in pending}
        batch = max(1, int(run.study["backbone"]["anyjev_batch"]))
        every = int(run.study["backbone"].get("anyjev_checkpoint_every", DEFAULT_CHECKPOINT_EVERY))
        every = max(1, every)
        since_save = 0
        done = lambda: sum(str(run.ids[i]) in probs_map for i in wanted)
        run.progress(stage="zero-shot", completed=done(), patients=n)
        for offset in range(0, len(pending), batch):
            chunk = pending[offset:offset + batch]
            for i, decision in zip(chunk, decider.decide_batch(
                    [states[i] for i in chunk], q, level="L0", require="L0")):
                probs_map[str(run.ids[i])] = np.asarray(decision.probs, dtype=float).tolist()
            since_save += len(chunk)
            run.progress(stage="zero-shot", completed=done(), patients=n)
            if since_save >= every or offset + batch >= len(pending):
                _save_partial(partial_path, key, probs_map)
                since_save = 0
        del backend, decider

    missing = [str(run.ids[i]) for i in wanted if str(run.ids[i]) not in probs_map]
    if missing:
        raise RuntimeError(f"zero-shot incomplete after run; missing {len(missing)} patients (e.g. {missing[:3]})")

    ordered = [probs_map[str(run.ids[i])] for i in wanted]
    if rows is None:
        write_json(path, {"key": key, "probs": dict(zip((str(p) for p in run.ids), ordered))})
        if partial_path.exists():
            partial_path.unlink()
    else:
        _save_partial(partial_path, key, probs_map)
    return np.array(ordered)


HOLDOUT_MODEL = "ZS_raw_holdout"


def holdout_rows(run):
    """Row indices of the XGB_optuna hold-out patients: a deterministic function of the labels and seed,
    so every backbone (and the tuned XGBoost arm) is scored on exactly the same patients."""
    from jevtest.xgb_optuna import holdout_split

    _, hold = holdout_split(run.y, float(run.study["holdout"]["fraction"]), run.study["seed"])
    return np.asarray(hold)


def save_holdout_predictions(run, P, hold):
    """ZS_raw on the hold-out patients only (large backbones): own artifact, same provenance key as ZS_raw."""
    import pandas as pd
    from jevtest.config import digest, save_table

    P = np.asarray(P, dtype=float)
    if P.shape != (len(hold), len(run.classes)) or not np.isfinite(P).all() or (P < 0).any():
        raise GateError(f"{HOLDOUT_MODEL} predictions rejected: invalid class probabilities")
    P = P / P.sum(axis=1, keepdims=True)
    table = pd.DataFrame({"model": HOLDOUT_MODEL, "fold": run.fid[hold], "patient_id": run.ids[hold],
                          "observed": run.y[hold]})
    for c, name in enumerate(run.classes):
        table[f"p__{name}"] = P[:, c]
    path = run.out / f"predictions/{HOLDOUT_MODEL}.csv"
    save_table(table, path)
    key = dict(run.prediction_key("ZS_raw"), model=HOLDOUT_MODEL, holdout_ids=run.ids[hold].tolist())
    write_json(path.with_suffix(".meta.json"), {"key": key, "csv_sha256": digest(path)})


def holdout_zero_shot_probs(run, hold):
    """P(class) for the hold-out rows from the full ZS_raw artifact or, failing that, from ZS_raw_holdout."""
    import pandas as pd
    from jevtest.config import digest

    P = run.predictions("ZS_raw") if (run.out / "predictions/ZS_raw.csv").exists() else None
    if P is not None:
        return P[hold]
    path = run.out / f"predictions/{HOLDOUT_MODEL}.csv"
    meta = path.with_suffix(".meta.json")
    if not path.exists() or not meta.exists():
        raise GateError("ZS_raw predictions are unavailable; run the `zeroshot` stage (optionally --subset holdout) first")
    stored = read_json(meta)
    expected = dict(run.prediction_key("ZS_raw"), model=HOLDOUT_MODEL, holdout_ids=run.ids[hold].tolist())
    if stored.get("key") != expected or stored.get("csv_sha256") != digest(path):
        raise GateError(f"{HOLDOUT_MODEL} predictions rejected: stale provenance; rerun `zeroshot --subset holdout`")
    table = pd.read_csv(path, dtype={"patient_id": str})
    if not np.array_equal(table.patient_id.to_numpy(), run.ids[hold]):
        raise GateError(f"{HOLDOUT_MODEL} predictions rejected: patients differ from the hold-out split")
    return table[[f"p__{c}" for c in run.classes]].to_numpy(dtype=float)


def run_zero_shot(run, subset="all"):
    if subset == "holdout":
        hold = holdout_rows(run)
        P = zero_shot_probs(run, rows=hold)
        save_holdout_predictions(run, P, hold)
        return P
    P = zero_shot_probs(run)
    run.save_predictions("ZS_raw", P / P.sum(axis=1, keepdims=True))
    return P
