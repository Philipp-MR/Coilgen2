"""Fixes from the 12 Nov walkthrough re-run: no repeated values, full unwrapped length, labels."""

import numpy as np
import pytest

from coilgen.application.cad import GeometryOption, geometry_option_label
from coilgen.application.scene_views import VIEW_PRESETS
from coilgen.web.result_display import result_overview
from test_gradient_headline import _metadata, _record
from test_view_clarity import pair


def test_results_board_does_not_repeat_what_the_gradient_row_states(tmp_path):
    metadata = _metadata()
    metadata["wire_fields"] = {
        "fabricated": {
            "available": True,
            "current_refitted": False,
            "current_A": 19.026,
            "rms_error_T": 8.7e-6,
            "peak_error_T": 24.8e-6,
            "relative_rms_error": 0.0496,
            "target_rms_T": 1e-4,
            "sample_count": 257,
            "rms_convention": "weighted_component_rms",
        }
    }
    board = result_overview(_record(tmp_path, metadata))
    assert board["gradient_cards"]
    keys = [card["key"] for card in board["field_cards"] + board["coil_cards"]]
    assert keys == ["rms", "peak", "mass", "volume"]
    assert board["field_context"].splitlines()[0].endswith("· errors at +19.03 A")
    metadata["summary"].pop("target_gradient_t_per_m")
    board = result_overview(_record(tmp_path, metadata))
    assert not board["gradient_cards"]
    assert {"relative", "current"} <= {card["key"] for card in board["field_cards"]}
    assert "errors at" not in board["field_context"]


def test_unwrapped_view_spans_the_whole_cylinder_without_a_stream_function():
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from coilgen.application.unwrapped_view import draw_unwrapped

    figure = Figure()
    arrays = pair()
    draw_unwrapped(figure, arrays)
    low, high = figure.axes[0].get_ylim()
    z = arrays["vertices_m"][:, 2] * 1e3
    assert low <= z.min() and high >= z.max()
    assert (high - low) < 1.2 * np.ptp(z)


def test_presets_look_from_both_sides_of_x_and_y():
    assert {"From +x (YZ)", "From −x (YZ)", "From +y (XZ)", "From −y (XZ)"} <= set(VIEW_PRESETS)


def test_grouped_export_rows_leave_the_role_to_the_group_heading():
    option = GeometryOption(
        "part001_body0001.stl",
        "part001_body0001",
        "stl",
        "fabrication_review_geometry",
        "passed",
        "m",
        ((0, 0, 0), (1, 1, 1)),
        False,
        "0" * 64,
        "primary",
    )
    assert geometry_option_label(option) == "Primary · part001_body0001 · STL solid"
    assert geometry_option_label(option, with_role=False) == "part001_body0001 · STL solid"


def test_desktop_board_hides_the_repeated_cards_under_a_gradient_row(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from coilgen.gui.results_overview import ResultsOverviewPanel

    app = QApplication.instance() or QApplication([])
    panel = ResultsOverviewPanel()
    try:
        panel.set_record(_record(tmp_path))
        for key in ("relative", "current"):
            assert panel.metric_values[key].parentWidget().isHidden()
        for key in ("resistance", "inductance"):
            assert panel.coil_metric_values[key].parentWidget().isHidden()
        assert not panel.metric_values["rms"].parentWidget().isHidden()
    finally:
        panel.deleteLater()
        app.processEvents()
