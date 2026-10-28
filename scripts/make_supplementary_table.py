"""Supplementary Table 1 (markdown rows) from the analysis outputs; patient counts are per arm."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevtest.benchmark import ALL_PAIRS
from jevtest.config import outputs_root


def n_rows(path):
    return len(pd.read_csv(path, usecols=["patient_id"]))


def main():
    gaps = pd.read_csv(outputs_root(ROOT) / "combined/figures/complexity_and_gaps.csv").set_index("dataset")
    cb = pd.read_csv(outputs_root(ROOT) / "combined/codebook/codebook_quality_and_gaps.csv").set_index("dataset")
    print("| Task | *n* LLM | *n* XGBoost | Predictors | Classes | LLM | XGBoost | Δ |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, outcome in ALL_PAIRS:
        out = outputs_root(ROOT) / dataset / outcome
        full = outputs_root(ROOT) / dataset / f"{outcome}__full/predictions/M2.csv"
        n_llm = n_rows(out / "predictions/ZS_raw.csv")
        n_xgb = n_rows(full if full.exists() else out / "predictions/M2.csv")
        g = gaps.loc[dataset]
        win = "*" if g.delta_zs_raw < 0 else ""
        print(f"| {g.label}{win} | {n_llm:,} | {n_xgb:,} | {int(cb.loc[dataset, 'n_predictors'])} | {int(g.n_classes)} "
              f"| {g.bacc_ZS_raw:.3f} | {g.bacc_M2:.3f} | {g.delta_zs_raw:+.3f} |".replace("-", "−"))


if __name__ == "__main__":
    main()
