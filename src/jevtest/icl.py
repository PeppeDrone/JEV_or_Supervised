"""ICL_full: training patients as labelled table rows in one prompt, then the test patient.

Ported from neurojev.evaluation.m3_learning (ICLPrompt, run_icl_full). The context step decides,
before any GPU use, whether every training patient fits in the backbone's context. If not, it picks
one subsample size n for all folds and, per fold, a representative class-stratified subsample: in each
class, the patients nearest the centres of k-means clusters (k = class quota) of the training fold's
standardised predictors. Only training patients of the fold are ever used; test patients never enter
a prefix. The readout is the answer-letter softmax averaged over the cyclic option orders.
"""
import math

import numpy as np
from sklearn.cluster import KMeans

from jevtest.classical import preprocessing
from jevtest.config import GateError, digest, read_json, write_json
from jevtest.data import predictor_matrix
from jevtest.llm import chat, options_block, orders
from jevtest.text import entries


class ICLPrompt:
    """Header, training rows, then the query; the prefix up to MARKER is shared within a fold."""

    MARKER = "Patient to predict (same columns):\n"

    def __init__(self, tok, run, labels):
        self.tok, self.run = tok, run
        caption = run.ds.get("columns_caption", "baseline observations")
        self.header = f"{run.ds['setting']}\n\nColumns ({caption}): " + " | ".join(labels)

    def content(self, rows, outcomes, query, order):
        body = self.header + "\n\n"
        if rows:
            name = self.run.spec.get("icl_outcome_name", "outcome")
            body += (f"Training patients from this unit, one per line: values in column order, then the observed {name}.\n"
                     + "\n".join(f"{r} -> {o}" for r, o in zip(rows, outcomes)) + "\n\n")
        return body + self.MARKER + f"{query}\n\n{options_block(self.run.spec['question'], self.run.classes, order)}"

    def parts(self, rows, outcomes, query, order):
        full = chat(self.tok, self.content(rows, outcomes, query, order))
        cut = full.rindex(self.MARKER) + len(self.MARKER)
        return full[:cut], full[cut:]


def table_rows(run):
    parsed = [entries(row, run.ds) for _, row in run.frame.iterrows()]
    labels = parsed[0][0]
    return labels, [" | ".join(values) for _, values in parsed], [run.classes[c] for c in run.y]


def icl_order(run, fold, chosen):
    return np.random.default_rng(run.study["seed"] + 1000 * fold).permutation(np.sort(chosen))


def quotas(counts, n, cfg):
    """Rows per class: proportional to the training fold (largest remainder) or equal; >= min_per_class."""
    k = len(counts)
    share = np.full(k, n / k) if cfg["balanced"] else n * counts / counts.sum()
    q = np.floor(share).astype(int)
    for c in np.argsort(-(share - q), kind="stable")[: n - q.sum()]:
        q[c] += 1
    q = np.minimum(np.maximum(q, cfg["min_per_class"]), counts)
    while q.sum() > n:  # the minimum raised some classes: take back from the largest
        q[np.argmax(q - cfg["min_per_class"])] -= 1
    while q.sum() < n and (q < counts).any():  # a capped class left room: give it to the others
        q[np.argmax(np.where(q < counts, share - q, -np.inf))] += 1
    if (q < np.minimum(cfg["min_per_class"], counts)).any():
        raise GateError(f"Context budget leaves fewer than {cfg['min_per_class']} rows per class")
    return q


def representative(Z, pool, quota, seed):
    """Patients nearest the k-means centres (k = quota) of one class's standardised predictors."""
    if quota >= len(pool):
        return list(pool)
    centres = KMeans(quota, n_init=3, random_state=seed).fit(Z[pool]).cluster_centers_
    chosen, used = [], set()
    for centre in centres:
        for j in np.argsort(((Z[pool] - centre) ** 2).sum(axis=1), kind="stable"):
            if pool[j] not in used:
                used.add(pool[j]); chosen.append(pool[j])
                break
    return chosen


def balance(Z, full, chosen, X, columns):
    """Standardised mean differences (subsample vs training fold) and missingness differences."""
    sd = Z[full].std(axis=0)
    smd = np.where(sd > 0, (Z[chosen].mean(axis=0) - Z[full].mean(axis=0)) / np.where(sd > 0, sd, 1), 0.0)
    top = np.argsort(-np.abs(smd))[:5]
    missing = np.abs(X.iloc[chosen].isna().mean() - X.iloc[full].isna().mean())
    return {"max_abs_smd": float(np.abs(smd).max()), "top_smd": {columns[j]: round(float(smd[j]), 3) for j in top},
            "max_missingness_difference": float(missing.max())}


def context_key(run, tok_revision):
    b = run.study["backbone"]
    return {"data_hash": run.data_hash, "icl": run.study["icl"], "model_id": b["model_id"], "revision": tok_revision,
            "question": run.spec["question"], "classes": run.classes, "seed": run.study["seed"],
            "folds": digest(str([t.tolist() for _, t in run.splits]).encode())}


def plan_context(run, tok=None):
    """Write context.json: per fold, the ordered training rows of the ICL prefix and their token counts."""
    cfg = run.study["icl"]
    if tok is None:
        from jevtest.llm import tokenizer
        tok = tokenizer(run.study)
    if cfg["context_limit"] > cfg["native_context"] * cfg["yarn_factor"]:
        raise GateError("context_limit exceeds native_context x yarn_factor")
    cap = cfg["context_limit"] - max(cfg["margin_tokens"], math.ceil(cfg["margin_fraction"] * cfg["context_limit"]))
    labels, rows, names = table_rows(run)
    prompt = ICLPrompt(tok, run, labels)
    count = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
    k = len(run.classes)
    query = max(count(prompt.parts([], [], rows[i], o)[1]) for i in range(len(rows)) for o in orders(k))

    def prefix_tokens(ordered):
        return count(prompt.parts([rows[i] for i in ordered], [names[i] for i in ordered], "QUERY", orders(k)[0])[0])

    X = predictor_matrix(run)
    design = {}
    for fold, (train, _) in enumerate(run.splits):
        pre = preprocessing(X, "scaled").fit(X.iloc[train])
        design[fold] = (pre.transform(X), list(pre.get_feature_names_out()))

    def select(fold, n):
        train = run.splits[fold][0]
        counts = np.bincount(run.y[train], minlength=k)
        chosen = []
        for c, q in enumerate(quotas(counts, n, cfg)):
            chosen += representative(design[fold][0], train[run.y[train] == c], q, run.study["seed"] + 1000 * fold + c)
        return icl_order(run, fold, np.array(chosen))

    full = {fold: icl_order(run, fold, train) for fold, (train, _) in enumerate(run.splits)}
    if all(prefix_tokens(full[f]) + query <= cap for f in full):
        n, chosen = None, full
    else:
        # Largest common n (bisection) at which every fold's representative subsample fits.
        lo, hi = cfg["min_per_class"] * k, min(len(t) for t, _ in run.splits)
        fits = lambda n: (lambda s: s if all(prefix_tokens(s[f]) + query <= cap for f in s) else None)(
            {f: select(f, n) for f in full})
        chosen = fits(lo)
        if chosen is None:
            raise GateError(f"Even {lo} training rows ({cfg['min_per_class']} per class) exceed the context budget")
        while lo < hi:
            mid = (lo + hi + 1) // 2
            run.progress(stage="context", trying_n=mid)
            trial = fits(mid)
            lo, hi, chosen = (mid, hi, trial) if trial is not None else (lo, mid - 1, chosen)
        n = lo
    folds = []
    for fold, (train, _) in enumerate(run.splits):
        ordered = chosen[fold]
        prefix = prefix_tokens(ordered)
        folds.append({"fold": fold, "n_available": len(train), "n_selected": len(ordered),
                      "class_counts": np.bincount(run.y[ordered], minlength=k).tolist(),
                      "prefix_tokens": prefix, "max_prompt_tokens": prefix + query,
                      **balance(design[fold][0], train, ordered, X, design[fold][1]),
                      "patient_ids": run.ids[ordered].tolist()})
    result = {"key": context_key(run, run.study["backbone"]["revision"]), "mode": "all" if n is None else "subsample",
              "n": n, "context_limit": cfg["context_limit"], "budget": cap, "max_query_tokens": query, "folds": folds}
    write_json(run.out / "context.json", result)
    return result


def run_icl(run):
    path = run.out / "context.json"
    if not path.exists():
        raise GateError("Run the `context` stage first")
    context = read_json(path)
    if context["key"] != context_key(run, run.study["backbone"]["revision"]):
        raise GateError("context.json is stale (data, study config, backbone, question or folds changed); rerun `context`")
    cfg, k = run.study["icl"], len(run.classes)
    labels, rows, names = table_rows(run)
    position = {p: i for i, p in enumerate(run.ids)}
    P = np.zeros((len(run.y), k))
    llm = None
    for fold, (_, test) in enumerate(run.splits):
        cache = run.out / f"cache/icl/fold{fold}.json"
        if cache.exists() and (stored := read_json(cache))["key"] == context["key"] and stored["test"] == run.ids[test].tolist():
            P[test] = stored["probs"]
            continue
        if llm is None:
            from jevtest.llm import LLM
            rope = ({"rope_type": "yarn", "factor": float(cfg["yarn_factor"]),
                     "original_max_position_embeddings": cfg["native_context"]} if cfg["yarn_factor"] > 1 else None)
            llm = LLM(run.study, rope_scaling=rope)
            prompt = ICLPrompt(llm.tokenizer, run, labels)
        run.progress(stage="icl", fold=fold, rows=context["folds"][fold]["n_selected"])
        ordered = [position[p] for p in context["folds"][fold]["patient_ids"]]
        assert not set(ordered) & set(test.tolist())
        shown = ([rows[i] for i in ordered], [names[i] for i in ordered])
        prefix = prompt.parts(*shown, "QUERY", orders(k)[0])[0]
        # The prefix does not depend on the option order, so it is encoded once for every order.
        suffixes = [prompt.parts(*shown, rows[i], o)[1] for i in test for o in orders(k)]
        def progress(completed, total):
            run.progress(stage="icl", fold=fold, rows=len(ordered), completed_queries=completed, total_queries=total)

        scores, tokens = llm.prefix_scores(prefix, suffixes, k, cfg["reuse"], progress=progress)
        for j, i in enumerate(test):
            p = np.zeros(k)
            for o, letters in zip(orders(k), scores[j * k:(j + 1) * k]):
                p[o] += letters  # letter position s carries class o[s]
            P[i] = p / k
        write_json(cache, {"key": context["key"], "test": run.ids[test].tolist(), "probs": P[test], "prefix_tokens": tokens})
    run.save_predictions("ICL_full", P)
