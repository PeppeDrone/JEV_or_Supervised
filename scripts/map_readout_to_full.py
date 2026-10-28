#!/usr/bin/env python3
"""Map every readout patient (``patients.csv``) to its row(s) in the whole table (``patients_full.csv``).

``record_id`` is the row position assigned *after* shuffling and subsampling, so on the tables whose
subsample was redrawn (Diabetes 130-US, CDC/BRFSS, SUPPORT2) the same id names different patients in
the two files. This script does not rewrite either file (that would change data hashes and invalidate
every prediction artefact). It writes ``datasets/<name>/readout_to_full.json``:

  method   "id"     every readout id exists in the full table with identical predictor and outcome values
           "value"  rows matched on all predictor columns plus the outcome; a readout row maps to every
                    full-table row with identical values (duplicates included), so that excluding the
                    mapped rows from training removes all copies of a hold-out patient
  rows     {readout_record_id: [full_record_id, ...]}
  unmatched readout ids with no identical full-table row (reported, never silently dropped downstream)

The XGB_optuna_full arm (jevtest.xgb_optuna.run_holdout_full) refuses to run when the map is missing
or when a hold-out patient is unmatched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TASKS = ["wids2021", "uci_diabetes130", "uci_support2", "uci_cdc_diabetes", "physionet_2012", "physionet_2019",
         "nhanes_htn", "metabric", "mexico_covid", "dryad_er_cbc"]


def columns(spec):
    predictors = [info["column"] for info in spec["predictors"].values()]
    outcomes = [o["column"] for o in spec["outcomes"].values()]
    return predictors + outcomes


def build_map(folder: Path) -> dict:
    spec = yaml.safe_load((folder / "dataset.yaml").read_text())
    idc = spec["id_column"]
    cols = columns(spec)
    sep = spec.get("separator", ",")
    sub = pd.read_csv(folder / spec["file"], sep=sep, dtype=str, keep_default_na=False)
    full = pd.read_csv(folder / "patients_full.csv", sep=sep, dtype=str, keep_default_na=False)
    missing = [c for c in cols + [idc] if c not in sub or c not in full]
    if missing:
        raise SystemExit(f"{folder.name}: columns missing from one of the files: {missing}")

    # Fast path: identical ids carry identical values.
    merged = sub.merge(full, on=idc, how="left", suffixes=("", "__full"), indicator=True)
    same = (merged["_merge"] == "both")
    for c in cols:
        same &= merged[c] == merged[f"{c}__full"]
    if bool(same.all()):
        rows = {rid: [rid] for rid in sub[idc]}
        return {"dataset": folder.name, "method": "id", "n_readout": len(sub), "n_full": len(full),
                "n_matched": len(sub), "n_full_rows_mapped": len(sub), "unmatched": [], "rows": rows}

    key = lambda frame: frame[cols].astype(str).agg("\x1f".join, axis=1)
    full_groups = full.assign(_k=key(full)).groupby("_k")[idc].apply(list).to_dict()
    rows, unmatched = {}, []
    for rid, k in zip(sub[idc], key(sub)):
        hit = full_groups.get(k)
        if hit:
            rows[rid] = hit
        else:
            unmatched.append(rid)
    mapped = sum(len(v) for v in rows.values())
    return {"dataset": folder.name, "method": "value", "n_readout": len(sub), "n_full": len(full),
            "n_matched": len(rows), "n_full_rows_mapped": mapped, "n_readout_with_duplicates_in_full":
            int(sum(len(v) > 1 for v in rows.values())), "unmatched": unmatched, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=TASKS)
    args = parser.parse_args(argv)
    for name in args.only or TASKS:
        folder = ROOT / "datasets" / name
        if not (folder / "patients_full.csv").exists():
            print(f"{name}: no patients_full.csv; skipped")
            continue
        result = build_map(folder)
        (folder / "readout_to_full.json").write_text(json.dumps(result, indent=1))
        print(f"{name}: method={result['method']} matched {result['n_matched']}/{result['n_readout']} "
              f"-> {result['n_full_rows_mapped']} full rows; unmatched {len(result['unmatched'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
