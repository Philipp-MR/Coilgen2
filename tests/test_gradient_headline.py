"""Gradient headline on the Results card: efficiency, current for the target, series R/L, error, fringe."""

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from coilgen.gui.result_summary import (
    failed_check_text,
    gradient_context,
    gradient_metric_cards,
    shield_fringe_reading,
)
from coilgen.research.application import _gradient_headline_summary
from coilgen.web.result_display import result_overview

from _coilgen_test_support.gradient_headline import (
    BODIES as BODIES,
    FIELD_ARRAYS as FIELD_ARRAYS,
    _metadata as _metadata,
    _record as _record,
)


def _cards(record):
    return {card["key"]: card for card in gradient_metric_cards(record)}


def test_gradient_headline_states_efficiency_current_series_values_error_and_fringe(tmp_path):
    cards = _cards(_record(tmp_path))
    drive = 1.1 * 19.026
    assert list(cards) == ["efficiency", "current", "resistance", "inductance", "error", "fringe"]
    assert cards["efficiency"]["value"] == f"{20 / drive:.4g} mT/m/A"
    assert cards["current"]["name"] == "Current for 20 mT/m"
    assert cards["current"]["value"] == f"{drive:.4g} A"
    # Both reviewed bodies carry one current, so they are in series: R is their sum.
    assert cards["resistance"]["value"] == "84.5 mΩ"
    assert "2 bodies in series" in cards["resistance"]["tooltip"]
    # No complete single-body inductance: the layout 1ᵀL1 with mutual coupling.
    assert cards["inductance"]["value"] == "28.02 µH"
    assert "mutual" in cards["inductance"]["tooltip"]
    assert (cards["error"]["name"], cards["error"]["value"]) == ("Field shape error", "3.57 %")
    assert cards["fringe"]["name"] == "Shield fringe RMS"
    assert cards["fringe"]["value"] == "29.54 µT / 20 µT ✕"
    assert cards["fringe"]["status"] == "failed"
    assert "Reviewed conductor field" in cards["fringe"]["tooltip"]
    assert gradient_context(list(cards.values())) == (
        f"Reviewed conductor at {drive:.4g} A for 20 mT/m · R and L with all parts driven in series"
    )


def test_layout_values_are_used_when_no_wire_field_or_common_current_is_recorded(tmp_path):
    metadata = _metadata()
    metadata["fabrication"]["electrical_metrics"]["bodies"][1]["current_A"] = -19.026
    cards = _cards(_record(tmp_path, metadata, names=("target_points_m",)))
    assert cards["current"]["value"] == "19.59 A"
    assert cards["efficiency"]["value"] == f"{20 / 19.594:.4g} mT/m/A"
    assert cards["efficiency"]["source"] == "Layout winding"
    assert cards["resistance"]["value"] == "84.52 mΩ"
    assert "layout winding" in cards["resistance"]["tooltip"]


@pytest.mark.parametrize(
    "change",
    (
        lambda m: m["summary"].update(design_context="mr_shim") or m["config"].update(design_context="mr_shim"),
        lambda m: m["summary"].pop("target_gradient_t_per_m"),
        lambda m: m["summary"].update(target_gradient_t_per_m=float("nan")),
    ),
)
def test_no_gradient_headline_for_other_designs_or_runs_without_a_recorded_target(tmp_path, change):
    metadata = _metadata()
    change(metadata)
    assert gradient_metric_cards(_record(tmp_path, metadata)) == []


def test_fringe_reading_is_unverified_without_a_verdict_and_absent_without_samples():
    metadata = _metadata()
    metadata["fabricated_physics"]["sampled_suppression"]["fabricated"]["rms_passed"] = None
    reading = shield_fringe_reading(metadata)
    assert reading["passed"] is None and reading["source"] == "Reviewed conductor"
    metadata["fabricated_physics"]["sampled_suppression"]["fabricated"]["available"] = False
    assert shield_fringe_reading(metadata)["source"] == "Stream function model"
    del metadata["fabricated_physics"]["sampled_suppression"]
    assert shield_fringe_reading(metadata) is None


def test_failed_reasons_put_the_shield_first_and_count_each_body_once():
    issues = [
        SimpleNamespace(severity="error", title="Conductor bend radius is too small", body_id=body)
        for body in (*BODIES, *("pipeline_" + body for body in BODIES))
    ]
    issues += [
        SimpleNamespace(severity="error", title="Bend is too tight for the conductor width", body_id=BODIES[0]),
        SimpleNamespace(severity="error", title="Conductor penetrates its support", body_id=BODIES[0]),
        SimpleNamespace(severity="error", title="Conductor surface triangles are poorly shaped", body_id=BODIES[1]),
        SimpleNamespace(severity="warning", title="Feed breaks are not designed", body_id=""),
    ]
    assert failed_check_text(_metadata(), issues) == (
        "Shield fringe field over limit · Conductor bend radius is too small (2 bodies) · "
        "Bend is too tight for the conductor width · +2 more"
    )
    passed = _metadata()
    passed["fabricated_physics"]["sampled_suppression"]["fabricated"]["rms_passed"] = True
    assert failed_check_text(passed, issues[-1:]) == ""


def test_saved_run_records_the_gradient_values_that_used_to_reach_only_the_report():
    report = {
        "layout_gradient_efficiency_mT_per_m_per_A": 1.02072,
        "layout_needed_current_A": -19.594,
        "coil_resistance_total_ohm": 0.08452,
        "coil_inductance_same_current_H": 2.80248e-05,
    }
    summary = _gradient_headline_summary(report)
    assert summary["layout_gradient_efficiency_mt_per_m_per_a"] == pytest.approx(1.02072)
    assert summary["layout_current_for_target_a"] == pytest.approx(19.594)
    assert summary["target_gradient_t_per_m"] == pytest.approx(1.02072 * 19.594 / 1000)
    assert summary["layout_series_resistance_ohm"] == pytest.approx(0.08452)
    assert summary["layout_series_inductance_h"] == pytest.approx(2.80248e-05)
    report.update(layout_gradient_efficiency_mT_per_m_per_A=float("nan"), coil_resistance_total_ohm=float("nan"))
    summary = _gradient_headline_summary(report)
    assert summary["target_gradient_t_per_m"] is None and summary["layout_current_for_target_a"] is None
    assert summary["layout_series_resistance_ohm"] is None


def test_browser_board_shows_the_gradient_row_instead_of_the_general_drive_row(tmp_path):
    board = result_overview(_record(tmp_path))
    assert [card["key"] for card in board["gradient_cards"]][:2] == ["efficiency", "current"]
    assert board["drive_cards"] == []
    assert board["gradient_context"].startswith("Reviewed conductor at ")
    metadata = _metadata()
    metadata["summary"].pop("target_gradient_t_per_m")
    board = result_overview(_record(tmp_path, metadata))
    assert board["gradient_cards"] == [] and board["drive_cards"]


def test_desktop_board_shows_the_gradient_row_and_why_checks_failed(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from coilgen.gui.results_overview import ResultsOverviewPanel

    app = QApplication.instance() or QApplication([])
    panel = ResultsOverviewPanel()
    try:
        panel.set_record(_record(tmp_path))
        assert not panel.gradient_cards.isHidden() and panel.drive_cards.isHidden()
        assert panel.gradient_labels["current"].text() == "Current for 20 mT/m"
        assert panel.gradient_values["fringe"].text() == "29.54 µT / 20 µT ✕"
        assert panel.gradient_frames["fringe"].property("status") == "failed"
        assert not panel.failed_reasons.isHidden()
        assert panel.failed_reasons.text().startswith(
            "Failed: Shield fringe field over limit · Conductor bend radius is too small (2 bodies)"
        )
        metadata = deepcopy(_metadata())
        metadata["summary"]["design_context"] = metadata["config"]["design_context"] = "mr_shim"
        panel.set_record(_record(Path(tmp_path) / "shim", metadata))
        assert panel.gradient_cards.isHidden() and not panel.drive_cards.isHidden()
    finally:
        panel.deleteLater()
        app.processEvents()
