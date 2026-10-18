"""The result board exposes saved error/current scope without field-array reads."""

from copy import deepcopy
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics, QPalette, QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from coilgen.application.service import RunRecord
from coilgen.gui.coil_statistics import build_coil_statistics
from coilgen.gui.results_overview import ResultsOverviewPanel, summarize_field_result


class ScalarOnlyRecord(RunRecord):
    @property
    def metadata(self):
        raise AssertionError("The board must borrow immutable metadata, not thaw it.")

    @property
    def arrays(self):
        raise AssertionError("The board must not copy field arrays.")

    def array(self, name, *, copy=True):
        raise AssertionError("The board must not evaluate or load field arrays.")


def _metadata():
    return {
        "name": "Conical spherical harmonic Z2",
        "project_revision": 7,
        "config": {"target_field_component": "Bz"},
        "summary": {"sheet_accepted": True, "accepted": False, "scope": "fabrication_review"},
        "fabrication": {"ready": False},
        "pipeline_sweeps": {"checks_passed": False},
        "wire_fields": {
            "routed": {
                "available": True,
                "current_refitted": True,
                "current_A": 7.5,
                "rms_error_T": 0.01e-6,
                "scope": "design_grid_wire_field",
                "field_model": "thin_wire_biot_savart",
                "at_stream_current": {
                    "available": True,
                    "current_refitted": False,
                    "current_A": -3.125,
                    "rms_error_T": 2.4e-6,
                    "peak_error_T": 8.2e-6,
                    "relative_rms_error": 0.24,
                    "target_rms_T": 10e-6,
                    "sample_count": 241,
                    "rms_convention": "weighted_component_rms",
                },
            }
        },
        # A large immutable diagnostics branch must not be traversed by this board.
        "diagnostics": {"witnesses": tuple({"index": i, "position": (i, 0, 0)} for i in range(3000))},
    }


def _record(metadata=None, names=None):
    names = names or ("target_points_m", "target_field_T", "routed_field_at_stream_current_T", "routed_field_T")
    return ScalarOnlyRecord(
        Path("outputs/results_overview_validation_20261011/saved"),
        metadata or _metadata(),
        {name: np.zeros((3, 3)) for name in names},
    )


def _coil_metadata():
    metadata = _metadata()
    metadata["summary"].update(resistance_ohm=0.035, inductance_h=80e-6, current_a=-3.125)
    metadata["fabrication"].update(
        electrical_metrics={
            "bodies": [
                {
                    "body_id": "wire",
                    "circuit_complete": True,
                    "current_A": -3.125,
                    "resistance_ohm": 0.035,
                    "inductance_H": 80e-6,
                }
            ]
        },
        paths=[{"body_id": "wire", "diagnostics": {"absolute_volume_m3": 1.2e-6}}],
    )
    metadata["pipeline_sweeps"]["bodies"] = [{"body_id": "wire", "diagnostics": {"absolute_volume_m3": 9e-6}}]
    return metadata


def test_original_signed_current_and_component_error_are_saved_scalars_not_the_better_refit():
    record = _record()
    immutable = record.metadata_view
    summary = summarize_field_result(record)
    assert summary.visible and summary.source_key == "routed_at_stream_current"
    assert summary.component == "Bz" and summary.current_A == -3.125
    assert summary.rms_uT == pytest.approx(2.4) and summary.peak_uT == pytest.approx(8.2)
    assert summary.relative_percent == pytest.approx(24.0)
    assert summary.badge == "Checks failed" and summary.tone == "error"
    assert "revision 7" in summary.title
    assert "original contour current" in summary.context and "Design samples · n=241" in summary.context
    assert "wire acceptance is separate" in summary.context
    assert "pointwise division" in summary.tooltip
    assert record.metadata_view is immutable


@pytest.mark.parametrize("component, expected", (("x", "Bx"), ("vector", "vector"), ("BdotB0", "BdotB0")))
def test_component_comes_from_saved_config_without_unconditional_bz(component, expected):
    metadata = _metadata()
    metadata["config"]["target_field_component"] = component
    summary = summarize_field_result(_record(metadata))
    assert summary.component == expected
    assert (
        "full vector B error" in summary.context
        if expected == "vector"
        else expected.replace("BdotB0", "B dot B0") in summary.context
    )


@pytest.mark.parametrize("bad", (None, 0.0, -1.0, float("nan"), float("inf"), True))
def test_relative_error_is_unavailable_when_recorded_target_rms_is_missing_zero_or_invalid(bad):
    metadata = _metadata()
    evidence = metadata["wire_fields"]["routed"]["at_stream_current"]
    evidence["target_rms_T"] = bad
    evidence["relative_rms_error"] = 0.0
    assert summarize_field_result(_record(metadata)).relative_percent is None


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -1.0, True, "1e-6"))
def test_invalid_error_scalars_are_not_presented_as_zero_or_computed_from_arrays(bad):
    metadata = _metadata()
    evidence = metadata["wire_fields"]["routed"]["at_stream_current"]
    evidence.update(rms_error_T=bad, peak_error_T=bad, relative_rms_error=bad)
    summary = summarize_field_result(_record(metadata))
    assert summary.rms_uT is summary.peak_uT is summary.relative_percent is None


def test_verified_final_conductor_metrics_take_precedence_without_laundering_sheet_acceptance():
    metadata = _metadata()
    metadata["summary"]["current_a"] = -3.125
    metadata["fabricated_physics"] = {
        "current_source": "original_contour_current_no_refit",
        "checks": [
            {
                "name": "Fabricated path field error (original current)",
                "verified": True,
                "passed": True,
                "rms_error_T": 3.1e-6,
                "peak_error_T": 9.4e-6,
                "relative_field_error": 0.31,
                "target_rms_T": 10e-6,
            },
        ],
    }
    names = (*_record().array_names, "fabricated_field_T")
    summary = summarize_field_result(_record(metadata, names))
    assert summary.source_key == "fabricated" and summary.rms_uT == pytest.approx(3.1)
    assert summary.current_A == -3.125 and summary.relative_percent == pytest.approx(31.0)
    assert summary.badge == "Checks failed"  # The saved fabrication geometry still fails.
    metadata["summary"]["scope"] = "continuous_sheet"
    assert summarize_field_result(_record(metadata, names)).current_A is None
    metadata["fabricated_physics"]["checks"][0]["verified"] = False
    assert summarize_field_result(_record(metadata, names)).source_key == "routed_at_stream_current"


def test_independent_winding_samples_preserve_per_body_currents_without_an_invented_sum():
    metadata = _metadata()
    metadata["winding_validation"] = {
        "available": True,
        "verified": True,
        "current_refitted": False,
        "current_source": "original_contour_current_no_refit",
        "component": "By",
        "point_count": 95,
        "rms_error_T": 4e-6,
        "peak_error_T": 12e-6,
        "target_rms_T": 20e-6,
        "relative_field_error": 0.2,
        "currents_A": {"body-a": -2.0, "body-b": 3.0},
    }
    names = (*_record().array_names, "winding_validation_points_m", "winding_validation_field_T")
    summary = summarize_field_result(_record(metadata, names))
    assert summary.source_key == "winding_validation" and summary.component == "By"
    assert summary.current_A is None and summary.current_display == "Per body"
    assert "Independent samples · n=95" in summary.context
    assert "Independent thin-wire winding check" in summary.context
    assert "quadrature" not in summary.context
    assert "no aggregate circuit current" in summary.tooltip


def test_sheet_evidence_uses_explicit_error_scalars_and_does_not_borrow_solver_or_field_rms():
    metadata = {
        "name": "Sheet run",
        "config": {"target_field_component": "Bx"},
        "summary": {"rms": 10e-6, "relative_field_error": 0.1, "scope": "continuous_sheet", "accepted": True},
    }
    summary = summarize_field_result(_record(metadata, ("target_points_m", "target_field_T", "sheet_field_T")))
    assert summary.source_key == "sheet" and summary.component == "Bx"
    assert summary.rms_uT is summary.peak_uT is summary.relative_percent is summary.current_A is None
    assert summary.badge == "Sheet passed" and summary.current_display == "Not applicable"
    metadata["validation"] = {
        "available": True,
        "rms_error_T": 1e-6,
        "peak_error_T": 3e-6,
        "relative_field_error": 0.1,
        "target_rms_T": 10e-6,
        "point_count": 80,
        "independent_quadrature_check": False,
    }
    summary = summarize_field_result(_record(metadata, ("validation_points_m", "validation_field_T")))
    assert summary.source_key == "validation" and summary.rms_uT == pytest.approx(1.0)
    assert "Independent quadrature unverified" in summary.context


def test_preview_only_has_no_computed_result_board_and_legacy_refit_is_explicit():
    assert not summarize_field_result(_record(names=("target_points_m", "target_field_T"))).visible
    metadata = _metadata()
    summary = summarize_field_result(_record(metadata, ("target_points_m", "routed_field_T")))
    assert summary.source_key == "routed" and summary.current_A == 7.5
    assert summary.current_label == "Refitted current" and "refitted recorded current" in summary.context


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def native_font(app):
    font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else ()
    yield QFont(families[0], 9) if families else app.font()
    if font_id >= 0:
        QFontDatabase.removeApplicationFont(font_id)


@pytest.fixture
def widgets(app, native_font):
    root = QWidget()
    root.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    root.setFont(native_font)
    layout = QVBoxLayout(root)
    layout.setContentsMargins(0, 0, 0, 0)
    panel = ResultsOverviewPanel(root)
    layout.addWidget(panel)
    layout.addWidget(QLabel("Saved residual field viewport", root), 1)
    yield root, panel
    root.close()
    root.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _show(root, app, width=800):
    root.resize(width, 400)
    root.show()
    for _ in range(3):
        app.processEvents()


def test_verification_failure_and_stale_snapshot_remain_visible_but_integrity_gates_analysis(widgets, app):
    root, panel = widgets
    record = _record()
    panel.set_record(record, source_error="Checksum mismatch: result arrays")
    _show(root, app)
    assert panel.isVisible() and panel.badge.text() == "Verification failed"
    assert panel.metric_values["rms"].text() == "2.4 µT"
    assert not panel.buttons["checks"].isEnabled()
    panel.set_analysis_available(True)
    assert all(not button.isEnabled() for button in panel.buttons.values())
    assert "Checksum mismatch" in panel.toolTip()
    panel.set_record(record, stale=True)
    assert panel.badge.text() == "Earlier model" and "Checks failed" in panel.context.text()
    assert panel.buttons["field"].isEnabled()
    panel.set_analysis_available(False)
    assert not panel.buttons["field"].isEnabled() and panel.buttons["checks"].isEnabled()
    assert panel.buttons["coil"].isEnabled()


def test_identity_cache_avoids_repeated_metadata_work_and_invalidates_on_health_or_revision_state(widgets):
    _, panel = widgets
    record = _record()
    with (
        patch("coilgen.gui.results_overview.summarize_field_result", wraps=summarize_field_result) as summarize,
        patch("coilgen.gui.results_overview.build_coil_statistics", wraps=build_coil_statistics) as properties,
    ):
        panel.set_record(record)
        first = panel.summary
        panel.set_record(record)
        assert summarize.call_count == properties.call_count == 1 and panel.summary is first
        panel.set_record(record, stale=True)
        panel.set_record(record, source_error="Unavailable")
        assert summarize.call_count == properties.call_count == 3
    panel.set_record(None)
    assert panel.isHidden() and not panel.source_key


@pytest.mark.parametrize("width", (660, 800))
def test_native_font_render_and_six_actions_fit_narrow_result_board(widgets, app, tmp_path, width):
    root, panel = widgets
    panel.set_record(_record(_coil_metadata()))
    _show(root, app, width)
    assert panel.width() == width
    assert panel.height() <= 260  # Eight metric cards retain more space for the field viewport.
    for key, value in panel.metric_values.items():
        assert QFontMetrics(value.font()).horizontalAdvance(value.text()) <= value.width(), key
    for key, value in panel.coil_metric_values.items():
        assert QFontMetrics(value.font()).horizontalAdvance(value.text()) <= value.width(), key
    for button in panel.buttons.values():
        assert button.height() >= 32 and button.geometry().right() < panel.width()
        assert not button.icon().isNull()
    actions = []
    panel.action_requested.connect(actions.append)
    for button in panel.buttons.values():
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert actions == list(panel.buttons)
    image_path = tmp_path / f"results_overview_{width}.png"
    assert root.grab().save(str(image_path))
    evidence = Path("outputs/results_overview_validation_20261011")
    evidence.mkdir(parents=True, exist_ok=True)
    assert root.grab().save(str(evidence / image_path.name))


def test_theme_change_keeps_badge_semantics_and_saved_values_readable(widgets, app):
    root, panel = widgets
    panel.set_record(_record())
    palette = QPalette(root.palette())
    palette.setColor(QPalette.ColorRole.Window, QColor("#20252b"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#292f36"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#f5f5f5"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#f5f5f5"))
    root.setPalette(palette)
    _show(root, app)
    assert "#292f36" in panel.styleSheet()
    assert panel.badge.property("semanticTone") == "error"
    assert panel.metric_values["current"].text() == "-3.125 A"
    assert panel.metric_values["current"].palette().color(QPalette.ColorRole.WindowText) == QColor("#f5f5f5")
    assert panel.context.palette().color(QPalette.ColorRole.WindowText) == QColor("#f5f5f5")
    evidence = Path("outputs/results_overview_validation_20261011")
    evidence.mkdir(parents=True, exist_ok=True)
    assert root.grab().save(str(evidence / "results_overview_dark.png"))


def test_scalar_summary_leaves_original_metadata_detached_and_unchanged():
    metadata = _metadata()
    original = deepcopy(metadata)
    summarize_field_result(_record(metadata))
    assert metadata == original


def test_saved_coil_properties_keep_exact_units_material_and_failed_geometry_scope(widgets, app):
    root, panel = widgets
    panel.set_record(_record(_coil_metadata()))
    _show(root, app)
    assert panel.has_coil_statistics
    assert panel.coil_metric_values["resistance"].text() == "0.035 Ω"
    assert panel.coil_metric_values["inductance"].text() == "80 µH"
    assert panel.coil_metric_values["volume"].text() == "1.2 cm³"  # The pipeline's 9 cm³ is an alternative.
    assert panel.coil_metric_values["mass"].text() == "Needs density"
    assert "one complete modeled electrical path" in panel.coil_metric_values["resistance"].toolTip().lower()
    assert "self inductance excludes mutual coupling" in panel.coil_metric_values["inductance"].toolTip()
    assert "excludes insulation" in panel.coil_metric_values["volume"].toolTip()
    assert "Geometry checks failed" in panel.coil_metric_values["volume"].toolTip()
    assert "No conductor density recorded" in panel.coil_metric_values["mass"].toolTip()
    assert "reviewed conductor mesh" in panel.coil_scope.text() and "geometry failed" in panel.coil_scope.text()
    assert "mass needs density" in panel.coil_scope.text()
    assert panel.badge.text() == "Checks failed"


def test_coil_only_saved_record_has_properties_and_checks_without_a_computed_field(widgets, app):
    root, panel = widgets
    record = _record(_coil_metadata(), ("support_vertices_m",))
    panel.set_record(record)
    _show(root, app)
    assert panel.isVisible() and panel.has_coil_statistics
    assert panel.field_cards.isHidden() and "Field not calculated" in panel.context.text()
    assert panel.buttons["coil"].isEnabled() and panel.buttons["checks"].isEnabled()
    assert all(not panel.buttons[key].isEnabled() for key in ("field", "slices", "errors", "harmonics"))
    assert panel.source_key == "" and panel.heading.full_text.startswith("Results ·")
    panel.set_record(record, source_error="Saved geometry hash mismatch")
    assert panel.isVisible() and panel.badge.text() == "Verification failed"
    assert all(not button.isEnabled() for button in panel.buttons.values())


def test_multiple_body_statistics_do_not_invent_a_whole_circuit_or_add_volumes(widgets):
    _, panel = widgets
    metadata = _coil_metadata()
    metadata["fabrication"]["electrical_metrics"]["bodies"].append({"body_id": "other", "circuit_complete": True})
    metadata["fabrication"]["paths"].append({"body_id": "other", "diagnostics": {"absolute_volume_m3": 3e-6}})
    panel.set_record(_record(metadata))
    assert panel.coil_metric_values["resistance"].text() == "Per body"
    assert panel.coil_metric_values["inductance"].text() == "Per body"
    assert panel.coil_metric_values["volume"].text() == "Per body"
    assert panel.coil_report.row("coil:resistance").value is None
    assert panel.coil_report.row("coil:inductance").value is None
    assert panel.coil_report.row("coil:volume").value is None
    assert "Saved values for 1 body; open Coil stats" in panel.coil_metric_values["resistance"].toolTip()
    assert (
        "self inductances cannot be added without mutual coupling" in panel.coil_metric_values["inductance"].toolTip()
    )
    assert "per-body values are listed separately" in panel.coil_metric_values["volume"].toolTip()


@pytest.mark.parametrize("current", (None, float("nan"), True))
def test_per_body_tiles_require_finite_saved_values(widgets, current):
    _, panel = widgets
    metadata = _coil_metadata()
    metadata["fabrication"]["electrical_metrics"]["bodies"] = [
        {"body_id": "a", "current_A": current},
        {"body_id": "b", "current_A": current},
    ]
    metadata["fabrication"]["paths"] = [{"body_id": "a"}, {"body_id": "b"}]
    panel.set_record(_record(metadata))
    assert all(panel.coil_metric_values[key].text() == "Not recorded" for key in ("resistance", "inductance", "volume"))


@pytest.mark.parametrize("legacy", (False, True))
def test_fabricated_field_keeps_original_per_body_currents_and_saved_sample_count(legacy):
    metadata = _coil_metadata()
    metadata["summary"]["current_a"] = None
    metadata["fabricated_physics"] = {
        "current_source": "original_contour_current_no_refit",
        "currents_A": {} if legacy else {"a": -2.0, "b": 3.0},
        "checks": [
            {
                "name": "Fabricated path field error (original current)",
                "verified": True,
                "rms_error_T": 1e-6,
                "sample_count": 257,
                "component": "By",
            }
        ],
    }
    summary = summarize_field_result(_record(metadata, (*_record().array_names, "fabricated_field_T")))
    assert summary.current_A is None and summary.current_display == "Per body"
    assert summary.current_label == "Original body currents"
    assert "By error" in summary.context and "n=257" in summary.context
    assert "no aggregate circuit current" in summary.tooltip


def test_explicit_recorded_mass_has_no_implicit_density_estimate(widgets):
    _, panel = widgets
    metadata = _coil_metadata()
    metadata["summary"]["conductor_mass_kg"] = 0.125
    panel.set_record(_record(metadata))
    assert panel.coil_metric_values["mass"].text() == "125 g"
    assert "no material or density inferred" in panel.coil_metric_values["mass"].toolTip()


def test_recorded_independent_field_failure_cannot_be_hidden_by_passed_geometry():
    metadata = _metadata()
    metadata["summary"]["accepted"] = True
    metadata["fabrication"]["ready"] = metadata["pipeline_sweeps"]["checks_passed"] = True
    metadata["validation"] = {"available": True, "passed": False}
    assert summarize_field_result(_record(metadata)).badge == "Checks failed"
