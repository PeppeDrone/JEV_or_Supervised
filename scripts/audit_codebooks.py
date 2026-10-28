#!/usr/bin/env python3
"""Read-only lint of every ``datasets/*/dataset.yaml`` codebook and its CSV.

The frozen reader sees only what ``jevtest.text.render`` emits (setting, header,
``text_label (scale_note): value unit`` lines, question and classes). This script
lists, per dataset, every place where that text is still a storage surface rather
than clinical English, plus CSV problems the codebook does not declare:

  raw-values        categorical text_values that echo the stored code
  label-is-column   text_label equal to the column name (modulo case / punctuation)
  no-quantity       numeric predictor without unit or scale_note
  sentinel-text     non-numeric strings in a numeric column not declared in missing_codes
  sentinel-value    a numeric value (-1, -9, 99, 999 ...) carried by >= 2 % of rows
  excel-date        text_values such as 8-Jun / 14-Oct (Excel-corrupted ranges)
  binary-numeric    numeric predictor whose observed values are only {0, 1}
  constant          predictor with a single observed value
  repeated-note     the same scale_note on >= 5 predictors (say it once in the header)
  short-setting     setting shorter than 15 words
  question          no horizon / population cue, or a raw code in parentheses
  outcome-leak      an outcome word appearing in setting / header / labels
  unused-feature    no predictor declares range or missing_codes

Run before and after a codebook pass; the counts should go to zero except for
findings that are explicitly accepted in DATASET_NOTES.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtest.text import normalize_code  # noqa: E402

EXCEL_DATE = re.compile(r"^\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$", re.I)
PLAIN_ENGLISH = re.compile(r"^[A-Z][a-z]+(?:[ -][A-Za-z][a-z]*)*$")
SENTINELS = (-1, -9, -99, -999, -9999, 99, 999, 9999, 99999)
HORIZON = re.compile(
    r"\b(within|during|at|after|before|in[- ]hospital|year|month|day|hour|week|follow-up|"
    r"present|current|baseline|admission|screen|positive|most likely|which|what|"
    # cross-sectional questions name their subject instead of a horizon
    r"this (patient|person|woman|man|adult|child|case|respondent|participant|recording|"
    r"glioma|tumou?r|mass|lesion|pregnancy|sample|record|speaker|image))\b", re.I)
RAW_PAREN = re.compile(r"\(([A-Z]{2,6}|[A-Za-z]+_[A-Za-z_]+)\)")
OUTCOME_WORDS = ("died", "death", "mortality", "deceased", "lethal", "recurr", "readmi", "survived")
# legitimate predictor terms that contain an outcome word
OUTCOME_WHITELIST = ("recurrence period density", "risk of recurrence category", "recurrent laryngeal")
# columns whose plausible values coincide with a sentinel: 99 is a real saturation/percentage,
# -1 is a real base excess, and so on
SENTINEL_SKIP = {
    99: re.compile(r"(sat|spo2|sao2|o2|hct|hematocrit|haematocrit|pcv|hb|hemoglobin|haemoglobin|"
                   r"glucose|bgr|systolic|diastolic|bp|pressure|map|heart|hr|weight|height|age|pulse|"
                   r"chloride|alp|alk|creatinine|bun|urea|sodium|potassium)", re.I),
    -1: re.compile(r"(base_?excess|be|baseexcess|temp|tendency)", re.I),
}


def slug(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def looks_raw(code, text) -> bool:
    """A text_value that is still the stored code, unless the code itself is plain English."""
    c, t = str(code).strip(), str(text).strip()
    if c.lower() != t.lower():
        return False
    if PLAIN_ENGLISH.fullmatch(t):
        return False
    # multi-word phrases without code punctuation ("6 to 8", "Less than 9th grade", "D-penicillamine")
    return not (re.search(r"[A-Za-z]", t) and "_" not in t and (" " in t or "-" in t))


def audit_dataset(folder: Path) -> dict:
    ds = yaml.safe_load((folder / "dataset.yaml").read_text())
    frame = pd.read_csv(folder / ds["file"], sep=ds.get("separator", ","), encoding="utf-8-sig", low_memory=False)
    findings: dict[str, list[str]] = collections.defaultdict(list)
    notes = collections.Counter()
    any_range = any_missing = False

    for name, info in ds["predictors"].items():
        col = frame[info["column"]]
        text_label = str(info["text_label"])
        same_as_column = slug(text_label) in (slug(info["column"]), slug(name))
        if same_as_column and not PLAIN_ENGLISH.fullmatch(text_label.strip()):
            findings["label-is-column"].append(f"{name}: '{text_label}'")
        if info.get("range") or info.get("text_range"):
            any_range = True
        if info.get("missing_codes"):
            any_missing = True
        note = info.get("scale_note")
        if note:
            notes[str(note)] += 1

        observed = col.dropna()
        codes = {normalize_code(v) for v in observed} - {None}
        declared_missing = set(info.get("missing_codes", []))
        if len(codes - declared_missing) == 1 and len(observed) > 20:
            findings["constant"].append(f"{name}: only {sorted(codes - declared_missing)}")

        if info["kind"] == "numeric":
            if not str(info.get("unit") or "").strip() and not str(note or "").strip():
                findings["no-quantity"].append(name)
            text = observed.astype(str).str.strip()
            numeric = pd.to_numeric(text.str.replace(r"^([+-]?\d+),(\d+)$", r"\1.\2", regex=True), errors="coerce")
            bad = text[numeric.isna()]
            bad = bad[~bad.map(normalize_code).isin(declared_missing)]
            if len(bad):
                findings["sentinel-text"].append(
                    f"{name}: {len(bad)} rows e.g. {sorted(bad.unique().tolist())[:3]}")
            values = numeric.dropna()
            if len(values) > 20:
                for s in SENTINELS:
                    skip = SENTINEL_SKIP.get(s)
                    if skip and (skip.search(name) or skip.search(str(info["column"]))):
                        continue
                    if s < 0 and ((values < 0) & (values != s)).any():
                        continue  # other negatives exist, so -1/-9 are real values
                    share = float((values == s).mean())
                    if share >= 0.02 and str(s) not in declared_missing and values.nunique() > 3:
                        findings["sentinel-value"].append(f"{name}: {share:.0%} of rows equal {s}")
                distinct = set(values.unique().tolist())
                if distinct <= {0.0, 1.0} and len(distinct) == 2:
                    findings["binary-numeric"].append(name)
        else:
            text_values = info.get("text_values") or {}
            raw = [str(k) for k, v in text_values.items() if looks_raw(k, v)]
            if raw:
                findings["raw-values"].append(f"{name}: {raw[:6]}{' ...' if len(raw) > 6 else ''}")
            excel = [str(v) for v in text_values.values() if EXCEL_DATE.fullmatch(str(v))]
            if excel:
                findings["excel-date"].append(f"{name}: {excel}")

    for note, count in notes.items():
        if count >= 5:
            findings["repeated-note"].append(f"'{note[:60]}...' on {count} predictors")

    setting = str(ds.get("setting") or "")
    if len(setting.split()) < 15:
        findings["short-setting"].append(f"{len(setting.split())} words: '{setting}'")
    header = str(ds.get("observations_header") or "")
    labels = " ".join(str(i["text_label"]) for i in ds["predictors"].values())
    for word in OUTCOME_WORDS:
        for where, text in (("setting", setting), ("header", header), ("labels", labels)):
            lowered = text.lower()
            for term in OUTCOME_WHITELIST:
                lowered = lowered.replace(term, "")
            if word in lowered:
                findings["outcome-leak"].append(f"'{word}' in {where}")

    for outcome, spec in ds["outcomes"].items():
        q = str(spec["question"])
        if not HORIZON.search(q):
            findings["question"].append(f"{outcome}: no horizon/population cue: '{q}'")
        if RAW_PAREN.search(q):
            findings["question"].append(f"{outcome}: raw code in question: {RAW_PAREN.findall(q)}")
        if len(set(spec["classes"])) != len(spec["classes"]):
            findings["question"].append(f"{outcome}: duplicate classes")
    if not any_range and not any_missing:
        findings["unused-feature"].append("no predictor declares range or missing_codes")

    return {"dataset": ds["name"], "n_predictors": len(ds["predictors"]),
            "findings": {k: v for k, v in sorted(findings.items())}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dataset", action="append", help="limit to these datasets (repeatable)")
    parser.add_argument("--json", type=Path, help="also write the full report to this path")
    parser.add_argument("--quiet", action="store_true", help="only the per-dataset totals")
    args = parser.parse_args(argv)

    folders = sorted(p.parent for p in (args.root / "datasets").glob("*/dataset.yaml")
                     if not p.parent.name.startswith("EXAMPLE"))
    if args.dataset:
        folders = [f for f in folders if f.name in set(args.dataset)]
    reports = [audit_dataset(folder) for folder in folders]

    totals = collections.Counter()
    for report in reports:
        n = sum(len(v) for v in report["findings"].values())
        for k, v in report["findings"].items():
            totals[k] += len(v)
        print(f"== {report['dataset']} ({report['n_predictors']} predictors): {n} finding(s)")
        if not args.quiet:
            for kind, items in report["findings"].items():
                for item in items:
                    print(f"   [{kind}] {item}")
    print("\nTotals:", dict(sorted(totals.items())))
    print(f"Datasets with findings: {sum(1 for r in reports if r['findings'])}/{len(reports)}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"totals": dict(totals), "datasets": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
