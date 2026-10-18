"""Execution feedback remains distinct from numerical and manufacturing verdicts."""

from datetime import datetime, timezone
import time
from unittest.mock import Mock

import pytest

import test_workbench_usability as usability

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QHeaderView

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.application.service import RunRecord
from coilgen.gui.workbench import CoilGenWorkbench


application = usability.application
windows = usability.windows


def _settle_completion(window):
    deadline = time.monotonic() + 10
    while window._run_load_tasks and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)
    assert not window._run_load_tasks


def _record(window, directory):
    return RunRecord(
        directory,
        {
            "name": window.project.name,
            "project_id": window.project.id,
            "project_revision": window.project.revision,
            "project_fingerprint": window.project.fingerprint,
            "project_model_fingerprint": window.project.model_fingerprint,
            "summary": {"scope": "continuous_sheet", "accepted": False, "sheet_accepted": False},
            "constraints": [],
            "wire_fields": {},
            "fabrication": None,
            "warnings": [],
            "artifacts": [],
        },
        {},
    )


def _complete(window):
    id_ = window.jobs.states[0]["id"]
    record = _record(window, window.jobs_directory / id_)
    window.service.load_run = Mock(return_value=record)
    window.jobs.states[0].update(status="completed", progress=1.0, stage="Complete")
    window.poll_jobs()
    _settle_completion(window)
    return id_, record


def test_stale_checkpoint_explains_reuse_scope_without_hiding_recorded_results(windows):
    from coilgen.application.reuse import ReusePlan

    window = windows()
    window._reuse_plan = ReusePlan((), ("sheet",), ())
    window._refresh_stage_status(refresh_reuse=False)
    item = window.stage_items["sheet"]
    assert item.text(1) == "Stale"
    assert "checkpoint fingerprint differs" in item.toolTip(1)
    assert "current source, inputs, requirements" in item.toolTip(1)
    assert "saved result remains available for inspection" in item.toolTip(1)


def test_state_persistence_warning_is_visible_without_disabling_worker_cancellation(windows):
    window = windows()
    window.compute()
    state = window.jobs.states[0]
    state.update(status="running", stage="Solving field", persistence_error="PermissionError: job.json locked")
    window.poll_jobs()
    assert "job state not saved; retrying" in window.execution_label.text()
    assert "job.json locked" in window.execution_label.toolTip()
    assert window.cancel_action.isEnabled() and window.cancel_action.isVisible()
    assert not window._last_runtime_error
    state.pop("persistence_error")
    window.poll_jobs()
    assert "not saved" not in window.execution_label.text()
    assert window.cancel_action.isEnabled()


def test_persistent_progress_keeps_execution_completion_separate_from_acceptance(windows):
    window = windows()
    assert window.execution_label.text() == "Ready to compute"
    assert not window.view_results_button.isEnabled()
    assert not window.cancel_action.isEnabled()
    assert not window.resume_action.isEnabled()

    state = window.compute()
    assert "Queued" in window.execution_label.text()
    assert window.cancel_action.isEnabled()
    assert not window.view_results_button.isEnabled()
    window.jobs.states[0].update(status="running", stage="Connecting contour groups", progress=0.63)
    window.poll_jobs()
    assert "Connecting contour groups" in window.execution_label.text()
    assert window.execution_progress.value() == 63
    assert window.jobs_table.item(0, 6).text() == "63%"

    id_, _ = _complete(window)
    assert id_ == state["id"]
    assert "Execution completed" in window.execution_label.text()
    assert "review engineering checks" in window.execution_label.text()
    assert "Passed" not in window.execution_label.text()
    assert window.execution_progress.value() == 100
    assert window.view_results_button.isEnabled()
    assert not window.cancel_action.isEnabled()
    # This synthetic result contains no checkpoint artifacts.
    assert not window.resume_action.isEnabled()
    assert window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "geometry"
    assert window.runs[id_].metadata["summary"]["accepted"] is False


def test_verification_error_is_persistent_and_disables_results_action(windows):
    window = windows()
    window.compute()
    window.service.load_run = Mock(side_effect=ValueError("Artifact checksum changed"))
    window.jobs.states[0].update(status="completed", stage="Complete", progress=1.0)
    window.poll_jobs()
    _settle_completion(window)
    assert "evidence verification failed" in window.execution_label.text()
    assert "checksum" in window.execution_label.toolTip()
    assert not window.view_results_button.isEnabled()
    assert not window.cancel_action.isEnabled()
    calls = window.service.load_run.call_count
    window.poll_jobs()
    assert window.service.load_run.call_count == calls
    assert "evidence verification failed" in window.execution_label.text()


def test_completion_does_not_navigate_or_consume_pending_scientific_draft(windows):
    window = windows()
    window.compute()
    original = window.project
    window._draft_controls["radius_mm"].setValue(172)
    _complete(window)
    assert window.project is original
    assert window._draft_dirty
    assert window._draft_controls["radius_mm"].value() == pytest.approx(172)
    assert window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "geometry"
    assert window.view_results_button.isEnabled()

    window.view_results_button.click()
    assert window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "results"
    assert window.project.revision == original.revision + 1
    assert window.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.172)
    assert not window._draft_dirty
    assert "STALE" in window.result_state.text()


def test_view_results_retains_invalid_pending_draft_in_its_settings_page(windows):
    config = preset_config()
    config.coil_mesh_file = "create conical frustum mesh"
    config.surface_is_cylinder_flag = False
    window = windows(ProjectDocument.from_config(config))
    window.compute()
    _complete(window)
    original = window.project
    window._draft_controls["rotation_axis_x"].setValue(0)
    window._draft_controls["rotation_angle_deg"].setValue(45)
    calls = window.service.load_run.call_count

    assert not window.view_results()
    assert window.project is original
    assert window._draft_dirty
    assert window._draft_controls["rotation_angle_deg"].value() == 45
    assert window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "geometry"
    assert window.service.load_run.call_count == calls
    assert not window.readiness_panel.isHidden()


def test_view_results_reverifies_saved_evidence_before_review(windows):
    window = windows()
    window.compute()
    _complete(window)
    window.service.load_run.side_effect = ValueError("Saved mesh no longer matches manifest")
    assert not window.view_results()
    assert not window.view_results_button.isEnabled()
    assert "verification failed" in window.execution_label.text()
    assert window.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) == "geometry"


@pytest.mark.parametrize("status", ["failed", "cancelled", "interrupted", "cancelling"])
def test_terminal_and_cancelling_jobs_have_logical_action_states(windows, status):
    window = windows()
    window.compute()
    window.jobs.states[0].update(status=status, stage=status.capitalize(), error="Solver stopped")
    window.jobs.cancel = Mock()
    window.poll_jobs()
    assert status.capitalize() in window.execution_label.text() or "Execution failed" in window.execution_label.text()
    assert not window.view_results_button.isEnabled()
    assert not window.cancel_action.isEnabled()
    window.cancel_selected_job()
    window.jobs.cancel.assert_not_called()


def test_job_selection_updates_cancel_action_and_readable_activity(windows, application):
    window = windows()
    first = window.compute()
    window.jobs.states[0].update(status="failed", error="No convergence")
    window.poll_jobs()
    second = window.compute()
    window.jobs.states[-1].update(status="running", stage="Clearance check " + "long activity " * 60)
    window.poll_jobs()
    window._focus_job(second["id"])
    assert window.cancel_action.isEnabled()
    window._focus_job(first["id"])
    assert not window.cancel_action.isEnabled()
    assert "Execution failed" in window.execution_label.text()
    window.jobs_table.clearSelection()
    assert not window.cancel_action.isEnabled()
    header = window.jobs_table.horizontalHeader()
    assert header.sectionResizeMode(4) == QHeaderView.ResizeMode.Stretch
    row = next(
        row
        for row in range(window.jobs_table.rowCount())
        if window.jobs_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == second["id"]
    )
    assert "long activity" in window.jobs_table.item(row, 4).toolTip()


def test_elapsed_uses_submission_to_now_for_active_jobs_and_freezes_terminal_time():
    state = {
        "status": "running",
        "created_at": "2026-10-03T15:00:00+00:00",
        "updated_at": "2026-10-03T15:00:10+00:00",
    }
    now = datetime(2026, 10, 3, 16, 2, 3, tzinfo=timezone.utc)
    assert CoilGenWorkbench._job_elapsed(state, now=now) == "1:02:03"
    state["status"] = "failed"
    assert CoilGenWorkbench._job_elapsed(state, now=now) == "0:10"
    state["created_at"] = "bad timestamp"
    assert CoilGenWorkbench._job_elapsed(state, now=now) == "—"


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "unknown", True])
def test_missing_or_invalid_worker_progress_remains_unknown(value):
    assert CoilGenWorkbench._job_progress({"status": "running", "progress": value}) is None
