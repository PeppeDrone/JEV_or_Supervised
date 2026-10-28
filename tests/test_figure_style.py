"""Export failures must not damage prior figures or silently change physical size."""
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from jevtest import figure_style as fs


def test_fixed_canvas_and_editable_text_in_exports(tmp_path):
    with plt.rc_context(fs.RC):
        figure, axis = plt.subplots(figsize=(fs.WIDTH_SINGLE, 60 * fs.MM))
        figure.subplots_adjust(left=0.20, bottom=0.25)
        axis.plot([0, 1], [0, 1], **fs.line("M2", markers=True))
        axis.set(xlabel="Predicted probability", ylabel="Observed frequency")
        fs.save(figure, tmp_path, "calibration")
    root = ET.parse(tmp_path / "calibration.svg").getroot()
    assert float(root.attrib["width"].removesuffix("pt")) == pytest.approx(89 / 25.4 * 72)
    assert float(root.attrib["height"].removesuffix("pt")) == pytest.approx(60 / 25.4 * 72)
    assert root.findall(".//{http://www.w3.org/2000/svg}text")
    pdf = (tmp_path / "calibration.pdf").read_bytes()
    media_box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
    assert media_box is not None
    assert float(media_box[1]) == pytest.approx(89 / 25.4 * 72)
    with Image.open(tmp_path / "calibration.png") as raster:
        assert raster.size == (int(89 / 25.4 * 450), int(60 / 25.4 * 450))
        assert raster.info["dpi"][0] == pytest.approx(450, abs=0.02)
    assert not plt.fignum_exists(figure.number)


def test_oversized_figure_is_rejected_before_overwriting(tmp_path):
    previous = tmp_path / "figure.pdf"
    previous.write_bytes(b"previous figure")
    figure = plt.figure(figsize=(fs.WIDTH_DOUBLE, 200 * fs.MM))
    with pytest.raises(ValueError, match="split into pages"):
        fs.save(figure, tmp_path, "figure")
    assert previous.read_bytes() == b"previous figure"
    assert not plt.fignum_exists(figure.number)


def test_render_failure_preserves_all_previous_formats(tmp_path, monkeypatch):
    for extension in ("pdf", "svg", "png"):
        (tmp_path / f"figure.{extension}").write_bytes(b"previous figure")
    with plt.rc_context(fs.RC):
        figure = plt.figure(figsize=(fs.WIDTH_SINGLE, 60 * fs.MM))
    original_save = figure.savefig

    def fail_on_svg(path, **kwargs):
        if path.suffix == ".svg":
            raise RuntimeError("simulated renderer failure")
        original_save(path, **kwargs)

    monkeypatch.setattr(figure, "savefig", fail_on_svg)
    with pytest.raises(RuntimeError, match="simulated renderer"):
        fs.save(figure, tmp_path, "figure")
    for extension in ("pdf", "svg", "png"):
        assert (tmp_path / f"figure.{extension}").read_bytes() == b"previous figure"
    assert not list(tmp_path.glob(".figure-*"))
    assert not plt.fignum_exists(figure.number)


def test_layout_diagnostic_detects_clipped_text():
    with plt.rc_context(fs.RC):
        figure = plt.figure(figsize=(fs.WIDTH_SINGLE, 60 * fs.MM))
        figure.text(1.1, 0.5, "Outside the canvas")
        assert any("outside canvas" in issue for issue in fs.validate_figure(figure)["issues"])
        plt.close(figure)
