"""Deterministic patient text from the dataset spec (ported from neurojev.models.clinical_text).

Only declared predictors are read; outcomes and identifiers never are. Missing values
render as "not observed" and nothing is imputed or interpreted.
"""
from decimal import Decimal, InvalidOperation
import re

import numpy as np
import pandas as pd

from jevtest.config import GateError


def normalize_code(value):
    """Normalize CSV strings and numeric codes without assigning ordinal meaning."""
    if pd.isna(value) or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        number = Decimal(text)
        if number.is_finite():
            return format(number.normalize(), "f")
    except InvalidOperation:
        pass
    return text


def numeric_value(value):
    """Numeric fields, including unambiguous decimal commas."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+,\d+", text):
        text = text.replace(",", ".")
    result = pd.to_numeric(text, errors="coerce")
    return float(result) if pd.notna(result) and np.isfinite(result) else np.nan


def render_value(value, info):
    code = normalize_code(value)
    if code is None:
        return "not observed"
    if code in info.get("missing_codes", []):
        return "not observed (recorded as unavailable)"
    if info["kind"] == "categorical":
        # Unknown codes are rejected when the dataset is loaded, so they never reach a prompt.
        return info["text_values"][code]
    number = numeric_value(value)
    if not np.isfinite(number):
        return "not observed (invalid numeric value)"
    limits = info.get("text_range", info.get("range"))
    if limits and not limits[0] <= number <= limits[1]:
        return "not observed (outside documented range)"
    rendered = f"{number:.6g}"
    if info.get("unit"):
        rendered += f" {info['unit']}"
    meaning = info.get("text_values", {}).get(code)
    return f"{rendered} ({meaning})" if meaning else rendered


def label(info):
    return f"{info['text_label']} ({info['scale_note']})" if info.get("scale_note") else info["text_label"]


def entries(row, ds):
    """(labels, rendered values) in spec order."""
    labels, values = [], []
    predictors = ds["predictors"]
    for info in predictors.values():
        rendered = render_value(row.get(info["column"]), info)
        condition = info.get("text_requires")
        if condition:
            parent = normalize_code(row.get(predictors[condition["predictor"]]["column"]))
            if parent not in condition["codes"]:
                rendered = condition["absent_text"] if parent in condition.get("absent_codes", []) else condition["unknown_text"]
        labels.append(label(info))
        values.append(rendered)
    return labels, values


def render(row, ds):
    """Setting sentence, observations header, then one 'label: value' line per predictor."""
    labels, values = entries(row, ds)
    lines = "\n".join(f"{k}: {v}" for k, v in zip(labels, values))
    return f"{ds['setting']}\n{ds['observations_header']}\n{lines}"


def check_codes(frame, ds):
    """Every observed categorical code must have a text translation (or be a declared missing code)."""
    problems = []
    for name, info in ds["predictors"].items():
        if info["kind"] != "categorical":
            continue
        known = set(info.get("text_values", {})) | set(info.get("missing_codes", []))
        codes = {normalize_code(v) for v in frame[info["column"]]} - {None}
        if codes - known:
            problems.append(f"{name}: {len(codes - known)} untranslated code(s)")
    if problems:
        raise GateError("Untranslated categorical codes may not enter prompts: " + "; ".join(problems))
