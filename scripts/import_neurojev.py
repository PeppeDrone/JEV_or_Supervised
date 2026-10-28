"""Register the NeuroJev GOSE cohort as dataset `neurojev_gose` (the reproduction check).

Reads only neurojev configs (approved predictors, codebook text, the jev_improvement prompt
constants and the GOSE question); the data file stays where it is and is referenced by path.
Run from jev_prediction_testing/ with the neurojev environment:
    python scripts/import_neurojev.py [--neurojev ../neurojev]
"""
from pathlib import Path
import argparse, sys

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--neurojev", default="../neurojev")
parser.add_argument("--root", default=".")
args = parser.parse_args()
neurojev = Path(args.neurojev).resolve()
sys.path.insert(0, str(neurojev / "src"))

from neurojev.common import config  # noqa: E402
from neurojev.evaluation.jev_improvement import SCALE_NOTES, SETTING, jev_question  # noqa: E402

cfg = config(neurojev)
predictors = {}
for key, info in cfg["clinical_mapping"]["predictors"].items():
    if not info.get("approved"):
        continue
    entry = {"column": f"x__{key}", "kind": info["kind"], "text_label": info.get("text_label", info["label"])}
    if key in SCALE_NOTES:
        entry["scale_note"] = SCALE_NOTES[key]
    for field in ("unit", "range", "text_range", "missing_codes", "model_transform"):
        if info.get(field):
            entry[field] = info[field]
    values = info.get("text_values", info.get("values", {})) if info["kind"] == "categorical" else info.get("text_values", {})
    if values:
        entry["text_values"] = {str(k): v for k, v in values.items()}
    if info.get("text_requires"):
        entry["text_requires"] = {"predictor": info["text_requires"]["column"], "codes": info["text_requires"]["codes"],
                                  "absent_codes": ["0"], "absent_text": "not applicable (EEG not performed)",
                                  "unknown_text": "not observed (EEG performance not established)"}
    predictors[key] = entry

spec = cfg["outcomes"]["outcomes"]["gose"]
dataset = {
    "name": "neurojev_gose",
    "description": "NeuroJev single-centre cohort: 32 approved t0 predictors, binary discharge GOSE (reproduction dataset).",
    "file": str(Path("../..") / ".." / neurojev.name / "data/processed/patients.csv"),
    "separator": ",",
    "id_column": "patient_id",
    "pseudonymize": False,  # already HMAC pseudonyms from neurojev
    "setting": SETTING,
    "observations_header": "Patient observations recorded in the baseline event (t0):",
    "columns_caption": "baseline t0 observations",
    "predictors": predictors,
    "outcomes": {"gose": {
        "column": "y__gose", "classes": list(spec["classes"]),
        "mapping": {"0": spec["classes"][0], "1": spec["classes"][1]},
        "question": jev_question(cfg).text,
        "icl_outcome_name": "discharge outcome",
    }},
}
target = Path(args.root) / "datasets/neurojev_gose/dataset.yaml"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True, width=120))
print(f"wrote {target}: {len(predictors)} predictors; missing_codes on "
      f"{[k for k, v in predictors.items() if v.get('missing_codes')]}")
