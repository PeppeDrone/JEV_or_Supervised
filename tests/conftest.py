"""Synthetic fixtures only: a generated dataset with numeric, categorical and conditional predictors."""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    """One token per whitespace-separated word; a fixed chat wrapper."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<user>\n" + messages[0]["content"] + "\n<assistant>\n"

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": text.split()}


def synthetic_frame(n=150, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.normal(10, 3, n).round(1)
    eeg = rng.integers(0, 2, n)
    frame = pd.DataFrame({
        "record_id": [f"SRC-{i:04d}" for i in range(n)],
        "age": rng.integers(18, 90, n),
        "score": np.where(rng.random(n) < 0.15, np.nan, score),
        "sex": rng.integers(1, 3, n),
        "eeg_done": eeg,
        "eeg_result": np.where(eeg == 1, rng.integers(1, 3, n), np.nan),
    })
    frame["outcome_bin"] = (score + rng.normal(0, 2, n) > 10.5).astype(int)
    frame["outcome_tri"] = np.digitize(score + rng.normal(0, 2, n), [8.5, 11.5])
    return frame


def spec():
    question = "Which synthetic outcome is most likely? Missing measurements are unknown."
    return {
        "name": "synthetic", "file": "data.csv", "separator": ",", "id_column": "record_id", "pseudonymize": True,
        "setting": "Setting: synthetic test cohort.", "observations_header": "Patient observations at baseline:",
        "predictors": {
            "age": {"column": "age", "kind": "numeric", "text_label": "Age", "unit": "years", "range": [0, 120]},
            "score": {"column": "score", "kind": "numeric", "text_label": "Severity score", "scale_note": "0–20; higher is better"},
            "sex": {"column": "sex", "kind": "categorical", "text_label": "Sex", "text_values": {"1": "Female", "2": "Male"}},
            "eeg_done": {"column": "eeg_done", "kind": "categorical", "text_label": "EEG performed",
                         "text_values": {"0": "No", "1": "Yes"}},
            "eeg_result": {"column": "eeg_result", "kind": "categorical", "text_label": "EEG result",
                           "text_values": {"1": "Reactive", "2": "Unreactive"},
                           "text_requires": {"predictor": "eeg_done", "codes": ["1"], "absent_codes": ["0"],
                                             "absent_text": "not applicable (EEG not performed)",
                                             "unknown_text": "not observed (EEG performance not established)"}},
        },
        "outcomes": {
            "bin": {"column": "outcome_bin", "classes": ["POOR", "GOOD"], "mapping": {"0": "POOR", "1": "GOOD"},
                    "question": question},
            "tri": {"column": "outcome_tri", "classes": ["LOW", "MID", "HIGH"],
                    "mapping": {"0": "LOW", "1": "MID", "2": "HIGH"}, "question": question},
        },
    }


@pytest.fixture
def root(tmp_path):
    (tmp_path / "configs").mkdir()
    study = yaml.safe_load((ROOT / "configs/study.yaml").read_text())
    study.update(search_iterations=2, bootstrap_iterations=50)
    (tmp_path / "configs/study.yaml").write_text(yaml.safe_dump(study))
    folder = tmp_path / "datasets/synthetic"
    folder.mkdir(parents=True)
    synthetic_frame().to_csv(folder / "data.csv", index=False)
    (folder / "dataset.yaml").write_text(yaml.safe_dump(spec(), allow_unicode=True))
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


def set_study(root, **icl):
    path = root / "configs/study.yaml"
    study = yaml.safe_load(path.read_text())
    study["icl"].update(icl)
    path.write_text(yaml.safe_dump(study))
