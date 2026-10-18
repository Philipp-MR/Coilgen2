"""Offscreen acceptance navigation preserves evidence and witness ownership."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from coilgen.application.diagnostics import ProblemIssue
from coilgen.gui.acceptance_panel import AcceptancePanel
from coilgen.gui.colors import _contrast, item_foreground


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["acceptance-panel-test"])


@pytest.fixture
def panel(application):
    widget = AcceptancePanel()
    yield widget
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _record(name="first", *, failed=True):
    return SimpleNamespace(
        directory=Path(name),
        metadata={
            "project_id": "owner",
            "project_revision": 4,
            "artifacts": [{"name": "request.json", "sha256": "0" * 64}],
            "pipeline_stage": "evaluated",
            "summary": {"sheet_accepted": True, "scope": "fabrication_review"},
            "constraints": [],
            "wire_fields": {},
            "pipeline_sweeps": {"checks_passed": not failed, "body_count": 1},
            "fabrication": {
                "ready": not failed,
                "paths": [
                    {
                        "body_id": "body1",
                        "diagnostics": {
                            "sweep_capsule_min_centerline_distance_m": 0.001,
                            "sweep_required_centerline_clearance_m": 0.003,
                            "sweep_capsule_clearance_violation": failed,
                        },
                    }
                ],
            },
        },
    )


def _issue(*, locatable=True, body="body1"):
    return ProblemIssue(
        f"{body}:sweep_clearance_violation",
        "Conductor envelopes are too close",
        "Saved capsule screen fails.",
        body_id=body,
        points_m=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) if locatable else (),
        scope="fabrication_review",
        settings_node="winding",
    )


def _select(panel, key):
    panel.table.selectRow(next(index for index, row in enumerate(panel.report.checks) if row.key == key))


def test_empty_panel_disables_evidence_and_navigation_actions(panel):
    assert panel.table.rowCount() == 0
    assert not panel.locate.isEnabled() and not panel.settings.isEnabled()
    assert not panel.problems.isEnabled()
    assert not panel.edit_settings.isEnabled()


def test_recorded_geometry_witness_routes_the_exact_issue_without_new_location(panel):
    issue = _issue()
    received, actions, inspections = [], [], []
    panel.witness_requested.connect(received.append)
    panel.action_requested.connect(actions.append)
    panel.saved_settings_requested.connect(lambda *args: inspections.append(args))
    record = _record()
    panel.set_record(record, project=SimpleNamespace(id="owner", fingerprint="current", revision=8), issues=(issue,))
    _select(panel, "body:Strict review:body1:clearance")
    assert panel.locate.isEnabled()
    panel.locate.click()
    panel.settings.click()
    assert inspections == [(record, "winding")]
    assert actions == []
    panel.edit_settings.click()
    assert received == [issue] and actions == ["winding"]
    assert "Saved capsule screen" in panel.details.text()
    assert "− minimum" in panel.table.item(panel.table.currentRow(), 3).toolTip()
    assert "Saved source revision 4; applied channel revision 8" in panel.details.text()


@pytest.mark.parametrize("locatable,body", ((False, "body1"), (True, "otherbody")))
def test_missing_or_wrong_body_witness_never_enables_location(panel, locatable, body):
    panel.set_record(_record(), issues=(_issue(locatable=locatable, body=body),))
    _select(panel, "body:Strict review:body1:clearance")
    assert not panel.locate.isEnabled()


def test_passed_geometry_cannot_inherit_an_old_failed_witness(panel):
    panel.set_record(_record(failed=False), issues=(_issue(),))
    _select(panel, "body:Strict review:body1:clearance")
    assert panel._check().status == "Passed"
    assert not panel.locate.isEnabled()


def test_refresh_preserves_selection_but_new_record_focuses_its_failure(panel):
    record = _record()
    panel.set_record(record)
    _select(panel, "body:Strict review:body1:clearance")
    panel.set_record(record, pending=True)
    assert panel._check().key == "body:Strict review:body1:clearance"
    panel.set_record(_record("second"))
    assert panel._check().key == panel.report.focus_key


def test_stage_button_moves_to_the_requested_stage_without_mutating_result(panel):
    record = _record()
    before = repr(record.metadata)
    panel.set_record(record)
    panel.stage_buttons["contours"].click()
    assert panel._check().stage == "contours"
    assert repr(record.metadata) == before


def test_pending_revision_identity_does_not_rewrite_numeric_margins(panel):
    record = _record()
    record.metadata.update(project_id="owner", project_fingerprint="saved")
    panel.set_record(record, project=SimpleNamespace(id="owner", fingerprint="saved"), pending=True)
    assert "Pending edits" in panel.identity.text()
    _select(panel, "body:Strict review:body1:clearance")
    assert panel._check().margin == pytest.approx(-0.002)
    assert "-2 mm" == panel.table.item(panel.table.currentRow(), 3).text()


def test_dark_theme_uses_readable_semantic_foregrounds(panel, application):
    palette = QPalette(panel.palette())
    palette.setColor(QPalette.ColorRole.Window, QColor("#202020"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#202020"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#f5f5f5"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#f5f5f5"))
    panel.setPalette(palette)
    panel.set_record(_record())
    application.processEvents()
    _select(panel, "body:Strict review:body1:clearance")
    row = panel.table.currentRow()
    assert panel.table.item(row, 4).foreground() == item_foreground("error", panel.table.palette())
    assert panel.stage_buttons["physical"].property("semanticTone") == "error"
    palette = panel.stage_buttons["physical"].palette()
    assert _contrast(palette.color(QPalette.ColorRole.ButtonText), palette.color(QPalette.ColorRole.Button)) >= 4.5


@pytest.mark.parametrize("owner", (None, "another-channel"))
def test_saved_settings_inspection_does_not_enable_editing_another_channel(panel, owner):
    record = _record()
    record.metadata["project_id"] = owner
    actions, inspections = [], []
    panel.action_requested.connect(actions.append)
    panel.saved_settings_requested.connect(lambda *args: inspections.append(args))
    panel.set_record(record, project=SimpleNamespace(id="owner", fingerprint="current", revision=8))
    _select(panel, "body:Strict review:body1:clearance")
    assert panel.settings.isEnabled()
    assert not panel.edit_settings.isEnabled()
    panel.settings.click()
    panel.edit_settings.click()
    assert inspections == [(record, "winding")] and actions == []


@pytest.mark.parametrize("artifact", (None, {"name": "request.json", "sha256": "bad"}))
def test_saved_settings_requires_a_recorded_request_artifact(panel, artifact):
    record = _record()
    record.metadata["artifacts"] = [artifact] if artifact else []
    panel.set_record(record)
    assert not panel.settings.isEnabled()
    assert "request.json" in panel.settings.toolTip()


def test_integrity_diagnostic_disables_saved_settings_and_source_edit_actions(panel):
    panel.set_record(
        _record(),
        project=SimpleNamespace(id="owner", fingerprint="current", revision=8),
        diagnostic="Saved request checksum changed.",
    )
    assert not panel.settings.isEnabled() and not panel.edit_settings.isEnabled()
    assert "checksum" in panel.settings.toolTip()


def test_unavailable_routed_evidence_cannot_hide_available_contour_current(panel):
    record = _record()
    record.metadata["wire_fields"] = {
        "routed": {"available": False, "current_A": 25, "current_refitted": True, "reason": "No connected winding."},
        "contour": {"available": True, "current_A": 12, "current_refitted": False},
    }
    panel.set_record(record)
    assert "Original contours: original wire current 12 A" in panel.metrics.text()
    assert "25 A" not in panel.metrics.text()
    panel.stage_buttons["routed"].click()
    row = next(index for index, check in enumerate(panel.report.checks) if check.key == "routed:rms")
    assert panel.table.item(row, 1).text() == "—"
    assert panel.table.item(row, 1).foreground().color() == panel.table.palette().color(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
    )
    assert "No connected winding" in panel.table.item(row, 1).toolTip()


def test_unavailable_original_current_is_not_promoted_by_refitted_field(panel):
    record = _record()
    record.metadata["wire_fields"] = {
        "routed": {
            "available": True,
            "current_A": 7,
            "current_refitted": True,
            "at_stream_current": {"available": False, "current_A": 99},
        },
    }
    panel.set_record(record)
    assert "Connected conductor: saved refitted current 7 A" in panel.metrics.text()
    assert "99 A" not in panel.metrics.text()


def test_narrow_acceptance_window_retains_evaluation_scope_and_saved_margin(panel, application):
    panel.set_record(_record())
    panel.resize(760, 600)
    panel.show()
    application.processEvents()
    assert not panel.table.isColumnHidden(5)
    assert not panel.table.isColumnHidden(3)
    assert panel.table.columnWidth(5) >= 170
    assert panel.table.columnWidth(0) < 340


def test_integrity_failure_or_different_run_disables_locate_without_erasing_saved_check(panel):
    received = []
    panel.witness_requested.connect(received.append)
    record = _record()
    panel.set_record(record, issues=(_issue(),), diagnostic="Saved evidence checksum changed.")
    _select(panel, "body:Strict review:body1:clearance")
    assert not panel.locate.isEnabled()

    panel._locate()
    assert not received
    assert panel._check().status == "Failed"
    from dataclasses import replace

    wrong = replace(_issue(), run_directory=str(Path("other-run").resolve()))
    panel.set_record(record, issues=(wrong,))
    _select(panel, "body:Strict review:body1:clearance")
    assert not panel.locate.isEnabled()


def test_archived_execution_timestamp_is_presented_separately_from_live_job_state(panel):
    record = _record()
    record.execution_times = {"created_at": "2026-10-11T03:27:53.159402+00:00"}
    before = repr(record.metadata)
    panel.set_record(record)
    assert "date unrecorded" not in panel.identity.text()
    assert "UTC: 2026-10-11T03:27:53" in panel.identity.toolTip()
    assert "job" not in record.metadata and repr(record.metadata) == before
