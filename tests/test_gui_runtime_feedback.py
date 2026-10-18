"""Opaque solves show real operations and liveness without invented advancement."""

from datetime import datetime, timedelta, timezone
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QMainWindow, QVBoxLayout

from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.gui.activity_feedback import ComputationFeedback
from coilgen.gui.design_plan import build_design_plan
from coilgen.gui.design_plan_panel import DesignPlanPanel
from coilgen.gui.design_progress import ComputationIndicator, build_design_progress
from coilgen.gui.runtime_feedback import runtime_feedback, work_detail


NOW = datetime(2026, 10, 11, 12, 3, tzinfo=timezone.utc)


def _state(**changes):
    return {
        "id": "shim-worker",
        "name": "Conical Z2",
        "status": "running",
        "active_stage": "sheet",
        "pipeline_stage": "evaluated",
        "progress": 0.64,
        "progress_kind": "allocated_work",
        "progress_operation": "Optimize stream function",
        "stage": "CVXPY quadratic solve",
        "progress_indeterminate": True,
        "phase_progress": None,
        "created_at": "2026-10-11T11:59:00Z",
        "started_at": "2026-10-11T12:00:00Z",
        "operation_started_at": "2026-10-11T12:00:00Z",
        "last_progress_at": "2026-10-11T12:00:30Z",
        "worker_alive": True,
        "worker_observed_at": "2026-10-11T12:02:59Z",
        **changes,
    }


def test_recorded_process_liveness_is_separate_from_numerical_progress_and_operation():
    state = _state()
    before = dict(state)
    report = runtime_feedback(state, now=NOW)
    assert report.operation == "Optimize stream function · CVXPY quadratic solve"
    assert "Elapsed 3m 00s" in report.line
    assert "Worker observed alive 1s ago" in report.line
    assert "Last progress 2m 30s ago" in report.line
    assert "not measure CPU activity, numerical advancement or completion" in report.tooltip
    assert report.phase_percent is None and report.allocated
    assert state == before


@pytest.mark.parametrize("status", ("running", "completed"))
def test_durable_state_warning_retains_numerical_scope_and_explains_restart_recovery(status):
    state = _state(status=status, persistence_error="PermissionError: owned job.json is locked")
    before = dict(state)
    report = runtime_feedback(state, now=NOW)
    assert "Job state not saved" in report.line and "retrying" in report.line
    assert "Worker supervision continues" in report.tooltip
    assert "restart recovery lacks the latest state" in report.tooltip
    assert "job.json is locked" in report.tooltip
    assert report.phase_percent is None and report.allocated
    assert state == before


@pytest.mark.parametrize("timestamp", (None, "invalid", "2026-10-11", "2027-01-01T00:00:00Z"))
def test_invalid_or_future_observations_cannot_claim_current_worker_liveness(timestamp):
    report = runtime_feedback(_state(worker_observed_at=timestamp), now=NOW)
    assert "Worker observed alive" not in report.line
    assert "Worker liveness unreported" in report.line


def test_legacy_status_update_does_not_become_numerical_progress_or_worker_observation():
    report = runtime_feedback(
        _state(worker_alive=None, worker_observed_at=None, last_progress_at=None, updated_at="2026-10-11T12:02:50Z"),
        now=NOW,
    )
    assert "Status updated 10s ago" in report.line and "Last progress" not in report.line
    assert "not a numerical progress event" in report.tooltip
    queued = runtime_feedback(_state(status="queued", started_at=None), now=NOW)
    assert "Since submission 4m 00s" in queued.line and "Waiting for a worker" in queued.line
    assert "Worker observed" not in queued.line


def test_opaque_or_invalid_counters_cannot_reuse_a_previous_operation_fraction():
    assert work_detail(_state(phase_progress=0.8, progress_completed=8, progress_total=10)) == (
        "current operation has no measurable work total"
    )
    assert "work units" not in work_detail(
        _state(progress_indeterminate=False, progress_completed=True, progress_total=10)
    )


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widgets(app):
    font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else ()
    font = QFont(families[0], 9) if families else app.font()
    root = QMainWindow()
    root.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    root.setFont(font)
    panel = DesignPlanPanel(root)
    root.setCentralWidget(panel)
    indicator = ComputationIndicator(root)
    root.statusBar().addPermanentWidget(indicator, 1)
    monitor = ComputationFeedback(root, indicator)
    yield root, panel, indicator, monitor
    monitor.close()
    root.close()
    root.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    if font_id >= 0:
        QFontDatabase.removeApplicationFont(font_id)


@pytest.mark.parametrize("kind", ("opaque", "counted", "cancel_pending"))
def test_real_widgets_render_operation_scope_and_keep_passive_editor_drafts(widgets, app, tmp_path, kind):
    root, panel, indicator, monitor = widgets
    project = ProjectDocument.from_config(preset_config("conical_harmonic"), name="Conical Z2")
    assembly = ProjectAssembly.new("Normal shim example", (project,))
    channel = assembly.channels[0]
    state = _state(project_id=project.id, project_fingerprint=project.fingerprint, project_revision=project.revision)
    now = datetime.now(timezone.utc)
    state.update(
        created_at=(now - timedelta(minutes=4)).isoformat(),
        started_at=(now - timedelta(minutes=3)).isoformat(),
        operation_started_at=(now - timedelta(minutes=3)).isoformat(),
        last_progress_at=(now - timedelta(minutes=2, seconds=30)).isoformat(),
        worker_observed_at=(now - timedelta(seconds=1)).isoformat(),
    )
    if kind == "counted":
        state.update(progress_indeterminate=False, phase_progress=0.3, progress_completed=3, progress_total=10)
    elif kind == "cancel_pending":
        state.update(status="cancelling", progress_indeterminate=False, phase_progress=0.8)
    panel.set_plan(build_design_plan(assembly, states=(state,), now=now))
    monitor.set_activity((state,))
    root.resize(340, 740)
    root.show()
    editor = QDialog(root)
    editor.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    layout = QVBoxLayout(editor)
    draft = QLineEdit("retained conductor spacing draft", editor)
    layout.addWidget(draft)
    editor.show()
    for _ in range(3):
        app.processEvents()
    footer = editor.findChild(ComputationIndicator, "window_computation_indicator")
    assert footer is not None and footer.isVisible()
    assert footer.runtime_label.isVisible() and "Worker observed alive" in footer.runtime_label.text()
    assert "Optimize stream function" in panel.activity_label.text()
    assert "CVXPY quadratic solve" in panel.activity_label.text()
    assert not panel.runtime_label.isHidden() and "Last progress" in panel.runtime_label.text()
    assert not any(row.text(1) == "Completed" for row in panel.stage_items.values())
    if kind == "counted":
        assert panel.channel_items[channel.id].text(1) == "Computing"
        assert panel.progress_bar.maximum() == 100 and panel.progress_bar.value() == 30
        assert panel.progress_bar.format() == "Operation 30%"
        assert indicator.label.text() == footer.label.text() == "Operation 30%"
        assert indicator.progress.value() == 30
    else:
        assert panel.progress_bar.maximum() == indicator.progress.maximum() == footer.progress.maximum() == 0
        assert "%" not in indicator.label.text() and "%" not in panel.progress_bar.format()
        if kind == "cancel_pending":
            assert "Cancellation requested" in panel.runtime_label.text()
            assert panel.stage_items[(channel.id, "sheet")].text(1) == "Stopping"
        else:
            assert "no measurable work total" in panel.activity_label.text()
    assert panel.runtime_label.geometry().bottom() < panel.height()
    assert root.grab().save(str(tmp_path / f"feedback_{kind}.png"))
    requested = []
    panel.job_requested.connect(requested.append)
    QTest.mouseClick(panel.job_button, Qt.MouseButton.LeftButton)
    assert requested == ["shim-worker"] and draft.text() == "retained conductor spacing draft"
    monitor.set_activity((state,), unavailable="Cannot refresh worker status")
    assert indicator.runtime_label.isHidden() and footer.runtime_label.isHidden()
    snapshot = build_design_progress("evaluated", state, unavailable="Cannot refresh worker status")
    assert not snapshot.runtime.line and not snapshot.busy
    editor.close()
