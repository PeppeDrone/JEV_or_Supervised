#!/usr/bin/env python3
"""Reduce the zero-shot readout sample of the large tables to one common size.

The four large tables whose prepare scripts keep a class-stratified sample (WiDS 2021, Diabetes 130-US,
CDC/BRFSS, SUPPORT2) are redrawn to ``--n`` patients, class-stratified within that sample with the
study seed. Two stratified draws in a row are still a stratified random sample of the full table. The
saved zero-shot probabilities of the patients that remain are carried into the new cache, so the
readout is not run again. ``patients_full.csv`` (the XGBoost arm) is not touched.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"wids2021": "diabetes_mellitus", "uci_diabetes130": "readmit_30d",
         "uci_support2": "hosp_death", "uci_cdc_diabetes": "cdc_diabetes"}


def stratified(frame, column, n, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in frame.groupby(column, sort=False):
        take = min(len(group), max(1, int(round(n * len(group) / len(frame)))))
        parts.append(group.loc[rng.choice(group.index.to_numpy(), size=take, replace=False)])
    cohort = pd.concat(parts).sample(frac=1.0, random_state=seed)
    if len(cohort) > n:
        cohort = cohort.sample(n=n, random_state=seed)
    return cohort.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--only", nargs="*", choices=list(TASKS))
    args = parser.parse_args()
    seed = yaml.safe_load((ROOT / "configs/study.yaml").read_text())["seed"]
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from jevtest import data
    from jevtest.config import read_json, write_json
    from jevtest.zeroshot import cache_key

    for dataset in args.only or TASKS:
        outcome = TASKS[dataset]
        folder = ROOT / "datasets" / dataset
        spec = yaml.safe_load((folder / "dataset.yaml").read_text())
        column = spec["outcomes"][outcome]["column"]
        out = ROOT / "outputs" / dataset / outcome
        old_cache = read_json(out / "cache/zs.json")
        frame = pd.read_csv(folder / "patients.csv", dtype=str, keep_default_na=False)
        before = len(frame)
        if before > args.n:
            frame = stratified(frame, column, args.n, seed)
            frame.to_csv(folder / "patients.csv", index=False, encoding="utf-8")
            prep = json.loads((folder / "PREPARATION.json").read_text())
            prep["readout_subsample"] = {"n": len(frame), "from_n": before, "seed": seed,
                                         "method": "class-stratified, within the earlier stratified sample"}
            (folder / "PREPARATION.json").write_text(json.dumps(prep, indent=2) + "\n")
        run = data.load(ROOT, dataset, outcome)
        ids = [str(p) for p in run.ids]
        missing = [p for p in ids if p not in old_cache["probs"]]
        if missing:
            raise SystemExit(f"{dataset}: {len(missing)} patients without a saved readout probability")
        write_json(out / "cache/zs.json", {"key": cache_key(run), "probs": {p: old_cache["probs"][p] for p in ids}})
        (out / "cache/zs_partial.json").unlink(missing_ok=True)
        print(f"{dataset}: {before} -> {len(frame)} patients; {len(ids)} saved probabilities kept")


if __name__ == "__main__":
    main()
