import numpy as np

from jevtest.data import load
from scripts.plot_predictions import create_figures


def test_plot_predictions_creates_publication_figures_for_available_models(root):
    run = load(root, "synthetic", "bin")
    for model, weight in (("M2", 0.8), ("ZS_raw", 0.65)):
        probabilities = (
            np.eye(len(run.classes))[run.y] * weight
            + (1 - weight) / len(run.classes)
        )
        run.save_predictions(model, probabilities)

    output = create_figures(root, "synthetic", "bin")

    expected = {
        "roc_curves.png",
        "precision_recall_curves.png",
        "calibration_curves.png",
        "confusion_matrices.png",
        "metric_comparison.png",
        "fold_auroc.png",
    }
    created = {path.name for path in output.glob("*.png")}
    assert expected <= created
    assert "metric_comparison_p02.png" in created
    for suffix in (".png", ".pdf", ".svg"):
        assert all(path.stat().st_size > 0 for path in output.glob(f"*{suffix}"))


def test_plot_predictions_handles_multiclass_outputs(root):
    run = load(root, "synthetic", "tri")
    probabilities = (
        np.eye(len(run.classes))[run.y] * 0.8
        + (1 - 0.8) / len(run.classes)
    )
    run.save_predictions("M2", probabilities)

    output = create_figures(root, "synthetic", "tri")

    assert {"roc_curves.png", "precision_recall_curves.png", "calibration_curves.png",
            "confusion_matrices.png", "fold_auroc.png"} <= {path.name for path in output.glob("*.png")}
