import numpy as np

from jevtest.data import load
from scripts.study_zero_shot_threshold import run_study


def test_zero_shot_threshold_study_creates_nested_cv_results(root):
    run = load(root, "synthetic", "bin")
    rng = np.random.default_rng(12)
    scores = np.clip(0.15 + 0.7 * run.y + rng.normal(0, 0.2, len(run.y)), 0.01, 0.99)
    run.save_predictions("ZS_raw", np.column_stack((1 - scores, scores)))

    output, summary = run_study(
        root,
        "synthetic",
        "bin",
        positive_class="GOOD",
        bootstrap_iterations=100,
    )

    assert set(summary["method"]) == {"ZS_raw"}
    assert (summary["patients"] == len(run.y)).all()
    assert (summary["fn"] >= 0).all()
    assert (summary["sensitivity_exact_95_lower"] <= summary["sensitivity"]).all()
    assert (summary["sensitivity_exact_95_upper"] >= summary["sensitivity"]).all()
    assert (output / "REPORT.md").is_file()
    assert (output / "summary.csv").is_file()
    assert len(list(output.glob("*.png"))) == 3


def test_zero_shot_threshold_study_requires_binary_outcome(root):
    import pytest

    from jevtest.config import GateError

    with pytest.raises(GateError, match="requires a binary outcome"):
        run_study(root, "synthetic", "tri", bootstrap_iterations=100)
