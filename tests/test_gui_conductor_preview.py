"""Conductor schematics report supplied physical dimensions without invention."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication

from coilgen.gui.conductor_preview import ConductorPreview


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["conductor-preview-test"])


@pytest.fixture
def preview(application):
    widget = ConductorPreview()
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    widget.resize(360, 250)
    widget.show()
    application.processEvents()
    yield widget
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_cross_section_shows_copper_area_and_insulated_outside_dimensions(preview, application):
    preview.set_dimensions(2.0, 1.0, 0.1)
    application.processEvents()
    assert preview.dimensions == (2.0, 1.0, 0.1)
    assert "Cu 2 × 1 mm · area 2 mm²" in preview.summary_label.text()
    assert "Outside 2.2 × 1.2 mm" in preview.summary_label.text()
    assert "insulation 0.1 mm per side" in preview.summary_label.text()
    # Native rendering contains the copper rectangle, rather than merely text.
    image = preview.drawing.grab().toImage()
    assert any(
        image.pixelColor(x, y).getRgb()[:3] == (204, 119, 50)
        for x in range(30, image.width() - 30, 10)
        for y in range(30, image.height() - 40, 10)
    )


def test_updates_do_not_round_the_supplied_physical_values(preview):
    values = (1.234567891234567, 2.345678912345678, 0.012345678912345)
    preview.set_dimensions(*values)
    assert preview.dimensions == values
    preview.set_dimensions(3, 2, 0)
    assert preview.dimensions == (3, 2, 0)
    assert "Outside 3 × 2 mm" in preview.summary_label.text()


@pytest.mark.parametrize(
    "values",
    [
        (None, 1, 0),
        ("1e", 1, 0),
        (0, 1, 0),
        (-1, 1, 0),
        (1, 0, 0),
        (1, 1, -0.1),
        (float("nan"), 1, 0),
        (1, float("inf"), 0),
        (True, 1, 0),
    ],
)
def test_incomplete_or_invalid_draft_clears_previous_geometry(preview, values):
    preview.set_dimensions(2, 1, 0.1)
    preview.set_dimensions(*values)
    assert preview.dimensions is None
    assert preview.drawing._dimensions is None
    assert "Complete valid conductor dimensions" in preview.summary_label.text()
    assert preview.drawing._message == "Dimensions incomplete"


def test_custom_profile_does_not_draw_or_report_a_rectangular_cross_section(preview):
    preview.set_dimensions(2, 1, 0.1)
    preview.set_dimensions(2, 1, 0.1, custom=True)
    assert preview.custom and preview.dimensions is None
    assert preview.drawing._dimensions is None
    assert "Custom profile" in preview.summary_label.text()
    assert "area" not in preview.summary_label.text()
    assert "Outside" not in preview.summary_label.text()
    assert "profile viewer" in preview.drawing._message


def test_supplied_polygon_draws_actual_offset_coordinates_and_area(preview, application):
    points = np.array(((0.001, 0), (0.003, 0), (0.0025, 0.002), (0.0015, 0.002)))
    original = points.copy()
    preview.set_profile(points, 0.1)
    application.processEvents()
    assert preview.custom and preview.dimensions is None
    assert "4 points · Cu area 3 mm²" in preview.summary_label.text()
    assert "u 1…3 mm · n 0…2 mm" in preview.summary_label.text()
    assert "insulation 0.1 mm" in preview.summary_label.text()
    np.testing.assert_array_equal(preview.drawing._profile, points * 1000)
    np.testing.assert_array_equal(points, original)
    image = preview.drawing.grab().toImage()
    assert any(
        image.pixelColor(x, y).getRgb()[:3] == (204, 119, 50)
        for x in range(30, image.width() - 30, 10)
        for y in range(30, image.height() - 40, 10)
    )


@pytest.mark.parametrize("insulation", (-0.1, float("inf"), True))
def test_invalid_polygon_insulation_clears_last_valid_shape(preview, insulation):
    points = ((-0.001, -0.001), (0.001, -0.001), (0.001, 0.001), (-0.001, 0.001))
    preview.set_profile(points, 0)
    assert preview.drawing._profile is not None
    preview.set_profile(points, insulation)
    assert preview.drawing._profile is None and preview.dimensions is None
    assert "Invalid profile draft" in preview.summary_label.text()


def test_switching_back_to_rectangle_clears_polygon_preview(preview):
    preview.set_profile(((0, 0), (0.002, 0), (0.001, 0.002)), 0)
    preview.set_dimensions(2, 1, 0.1)
    assert preview.drawing._profile is None and not preview.custom
    assert preview.dimensions == (2, 1, 0.1)
