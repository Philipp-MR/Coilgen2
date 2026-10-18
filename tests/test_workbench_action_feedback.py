"""User actions acknowledge drafts, applied revisions and the correct job."""

from copy import deepcopy
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest

from coilgen.gui import workbench
from coilgen.gui.readiness import ReadinessIssue, ReadinessReport
from coilgen.research.storage import result_metadata_digest
from test_workbench_usability import application as application, windows as windows


def wait_until(predicate, timeout_ms=3000):
    """Deliver queued Qt callbacks until a bounded asynchronous condition holds."""
    if predicate():
        return
    loop = QEventLoop()
    poll, deadline = QTimer(), QTimer()
    poll.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    poll.start(5)
    deadline.start(timeout_ms)
    try:
        loop.exec()
    finally:
        poll.stop()
        deadline.stop()
    assert predicate(), "Asynchronous workbench condition did not settle before the deadline"


@pytest.fixture(autouse=True)
def ready_environment(monkeypatch):
    # Exercise action feedback without solver/file-discovery dependencies;
    # native model validators still decide whether edits can be committed.
    monkeypatch.setattr(workbench, "check_readiness", lambda *_args, **_kwargs: ReadinessReport())


def test_native_numeric_draft_marks_clean_document_without_changing_model(windows):
    window = windows()
    window.undo_stack.setClean()
    original = window.project
    assert not window.isWindowModified()
    radius = window._draft_controls["radius_mm"]
    radius.lineEdit().selectAll()
    QTest.keyClicks(radius.lineEdit(), "174")
    assert window._draft_dirty and window.isWindowModified()
    assert not window.pending_panel.isHidden()
    assert "Pending settings" in window.statusBar().currentMessage()
    assert window.project is original and window.undo_stack.count() == 0


def test_returning_field_to_original_clears_pending_title_and_status(windows):
    window = windows()
    window.undo_stack.setClean()
    original = window.project
    radius = window._draft_controls["radius_mm"]
    initial = radius.value()
    radius.setValue(initial + 10)
    assert window.isWindowModified() and window._draft_dirty
    radius.setValue(initial)
    assert not window._draft_dirty and not window.isWindowModified()
    assert window.pending_panel.isHidden()
    assert "Pending settings" not in window.statusBar().currentMessage()
    assert window.project is original and window.undo_stack.count() == 0


def test_apply_undo_redo_acknowledge_actual_revision_without_stale_pending(windows):
    window = windows()
    window.undo_stack.setClean()
    original = window.project
    radius_m = original.config.cylinder_mesh_parameter_list[1] + 0.01
    window._draft_controls["radius_mm"].setValue(radius_m * 1000)
    assert window.apply_settings()
    applied = window.project
    assert applied.config.cylinder_mesh_parameter_list[1] == pytest.approx(radius_m)
    assert applied.revision == original.revision + 1
    assert not window._draft_dirty and window.pending_panel.isHidden()
    assert f"Applied revision {applied.revision}" in window.statusBar().currentMessage()
    assert "Pending settings" not in window.statusBar().currentMessage()
    assert window.isWindowModified()

    window.undo_stack.undo()
    assert window.project.fingerprint == original.fingerprint
    assert "Undid" in window.statusBar().currentMessage()
    assert f"applied revision {original.revision}" in window.statusBar().currentMessage()
    assert not window.isWindowModified()

    window.undo_stack.redo()
    assert window.project.fingerprint == applied.fingerprint
    assert "Redid" in window.statusBar().currentMessage()
    assert f"applied revision {applied.revision}" in window.statusBar().currentMessage()
    assert window.isWindowModified()


def test_reopen_saved_project_replaces_welcome_status_with_opened_document(windows, tmp_path):
    window = windows()
    original = window.project
    path = window.save_project(tmp_path / "feedback.coilgen.json")
    assert window.close_project()
    assert window.project is None and window.statusBar().currentMessage() == "No project open"
    window.open_project(path)
    assert window.project.fingerprint == original.fingerprint
    assert window.project.revision == original.revision
    assert "Opened" in window.statusBar().currentMessage()
    assert window.project.name in window.statusBar().currentMessage()
    assert "No project open" not in window.computation_indicator.message.text()
    assert not window.isWindowModified()


def test_modeling_warnings_remain_accessible_without_disabling_compute(windows, monkeypatch, application):
    note = ReadinessIssue(
        "The requested study skips inductance estimation.",
        "winding",
        "skip_inductance_calculation",
        "Review the electrical-estimate settings before interpreting voltage requirements.",
        "warning",
        "Inductance not evaluated",
    )
    monkeypatch.setattr(workbench, "check_readiness", lambda *_args, **_kwargs: ReadinessReport((note,)))
    window = windows()
    before = window.project
    assert not window.modeling_notes_button.isHidden()
    assert "1 modeling note" in window.modeling_notes_button.text()
    assert note.message in window.modeling_notes_button.toolTip()
    assert window.compute_button.isEnabled() and window.compute_action.isEnabled()
    assert window.modeling_notes_panel.issues == (note,)
    window.modeling_notes_button.click()
    application.processEvents()
    assert window.modeling_notes_window.isVisible()
    assert "Compute is available" in window.modeling_notes_panel.heading.text()
    assert window.project is before and window.undo_stack.count() == 0


def test_incomplete_numeric_edit_remains_inline_and_does_not_open_messages(windows):
    window = windows()
    window.evidence.close_all()
    original = window.project
    radius = window._draft_controls["radius_mm"]
    radius.lineEdit().selectAll()
    QTest.keyClick(radius.lineEdit(), Qt.Key.Key_Backspace)
    assert radius.lineEdit().text().strip() in {"", radius.suffix().strip()}
    assert not window.apply_settings()
    assert window.project is original and window._draft_dirty
    assert not window.draft_error_label.isHidden()
    assert "complete numeric value" in window.draft_error_label.text()
    assert not window._field_errors["radius_mm"].isHidden()
    assert not window.evidence.isTabVisible(window.evidence.indexOf(window.messages))
    assert not window.evidence.any_open()


def test_rejected_physical_edit_reports_inline_without_opening_messages(windows):
    window = windows()
    original = window.project
    window.evidence.close_all()
    assert not window.apply_config({"target_region_radius": -1})
    assert window.project is original and window.undo_stack.count() == 0
    assert not window.draft_error_label.isHidden()
    assert "Input edit rejected" in window.statusBar().currentMessage()
    assert "Input edit rejected" in window.messages.toPlainText()
    assert not window.evidence.isTabVisible(window.evidence.indexOf(window.messages))
    assert not window.evidence.any_open()


@pytest.mark.parametrize("status,tone", [("failed", "error"), ("cancelled", "warning"), ("interrupted", "warning")])
def test_terminal_missing_progress_never_animates_and_preserves_state_tone(windows, status, tone):
    window = windows()
    state = {
        "id": "stopped",
        "project_id": window.project.id,
        "name": "Stopped gradient design",
        "status": status,
        "progress": None,
    }
    window.jobs.states = [state]
    window.poll_jobs()
    window._focus_job(state["id"], open_window=False)
    assert window.execution_progress.maximum() == 100
    assert window.execution_progress.value() == 0
    assert window.execution_progress.format() == "Unreported"
    assert window.execution_label.property("semanticTone") == tone
    # Execution has stopped; saved-checkpoint verification may still be active.
    wait_until(lambda: not window._stopped_reuse_tasks and window.computation_indicator.isHidden())
    assert window.computation_indicator.isHidden()
    assert window.execution_progress.maximum() == 100 and window.execution_progress.format() == "Unreported"
    assert window.execution_label.property("semanticTone") == tone
    assert window.states[state["id"]]["status"] == status


def test_compact_guidance_reveals_and_selects_its_exact_hidden_active_job(windows, application):
    window = windows()
    before = window.project
    old = {
        "id": "earlier-stop",
        "name": "Earlier calculation",
        "project_id": before.id,
        "status": "interrupted",
    }
    target = {
        "id": "current-computation",
        "name": "Current gradient calculation",
        "project_id": before.id,
        "status": "running",
        "progress": 0.2,
        "project_revision": before.revision,
        "project_fingerprint": before.fingerprint,
    }
    window._runtime_snapshot_taken = True
    window._previous_job_ids.add(old["id"])
    window._session_job_ids.add(target["id"])
    window.jobs.states = [old, target]
    window.jobs_filter.setCurrentIndex(window.jobs_filter.findData("previous"))
    window.poll_jobs()
    assert window.jobs_table.rowCount() == 1
    assert window.jobs_table.item(0, 0).data(Qt.ItemDataRole.UserRole) == old["id"]
    assert window._guidance_state.job_id == target["id"]
    assert window._guidance_state.action_key == "show_job"
    assert not window.guidance_action_button.isHidden()
    window.guidance_action_button.click()
    application.processEvents()
    assert window.jobs_filter.currentData() == "all"
    assert window._selected_job_id() == target["id"]
    assert window.evidence.window_for(window.jobs_table).isVisible()
    assert window.project is before and window.undo_stack.count() == 0


def test_queued_without_reported_progress_waits_without_animation_or_false_percent(windows):
    window = windows()
    window.jobs.states = [{"id": "waiting", "project_id": window.project.id, "status": "queued"}]
    window.poll_jobs()
    window._focus_job("waiting", open_window=False)
    assert window.execution_progress.maximum() == 100
    assert window.execution_progress.value() == 0
    assert window.execution_progress.format() == "Queued"
    assert window.execution_label.property("semanticTone") == "info"
    assert window.computation_indicator.label.text() == "Queued"
    assert window.computation_indicator.progress.isHidden()


def test_observed_terminal_outcomes_persist_without_repeats_or_startup_historical_noise(windows, monkeypatch):
    window = windows()
    before = window.project
    states = [
        {"id": "old-failure", "status": "failed", "project_id": before.id},
        {"id": "new-failure", "status": "running", "project_id": before.id, "name": "Gx"},
        {"id": "new-cancel", "status": "running", "project_id": before.id, "name": "Gy"},
    ]
    window.jobs.states = states
    window.evidence.close_all()
    publish = Mock(wraps=window.job_outcomes.publish)
    monkeypatch.setattr(window.job_outcomes, "publish", publish)
    window.poll_jobs()
    assert window.job_outcomes.outcomes == () and window.job_outcomes.isHidden()
    states[1].update(status="failed", error="Optimization did not converge")
    states[2].update(status="cancelled", completed_stages=["prepared"])
    window.poll_jobs()
    assert {outcome.job_id for outcome in window.job_outcomes.outcomes} == {"new-failure", "new-cancel"}
    assert window.job_outcomes.selected.job_id == "new-failure"
    assert publish.call_count == 2
    logged = window.messages.toPlainText()
    window.poll_jobs()
    assert publish.call_count == 2 and window.messages.toPlainText() == logged
    assert not window.evidence.any_open()
    window.job_outcomes.dismiss_button.click()
    window.poll_jobs()
    assert {outcome.job_id for outcome in window.job_outcomes.outcomes} == {"new-cancel"}
    assert publish.call_count == 2
    assert window.states["new-failure"]["status"] == "failed"
    assert window.states["new-cancel"]["status"] == "cancelled"
    assert window.project is before and window.undo_stack.count() == 0


def test_notice_action_reveals_exact_filtered_failure_without_automatic_windows(windows, application):
    window = windows()
    before = window.project
    target = {
        "id": "hidden-failure",
        "name": "Gz solver",
        "project_id": before.id,
        "status": "running",
    }
    window._runtime_snapshot_taken = True
    window._session_job_ids.add(target["id"])
    window.jobs.states = [{"id": "earlier", "project_id": before.id, "status": "interrupted"}, target]
    window._previous_job_ids.add("earlier")
    window.jobs_filter.setCurrentIndex(window.jobs_filter.findData("previous"))
    window.poll_jobs()
    window.evidence.close_all()
    target.update(status="failed", error="Recorded solver breakdown", log=["Conditioning check failed"])
    window.poll_jobs()
    assert not window.evidence.any_open()
    assert window._selected_job_id() is None
    assert window.job_outcomes.selected.job_id == target["id"]
    window.job_outcomes.action_button.click()
    application.processEvents()
    assert window.jobs_filter.currentData() == "all"
    assert window._selected_job_id() == target["id"]
    assert window.evidence.window_for(window.messages).isVisible()
    assert "Recorded solver breakdown" in window.messages.toPlainText()
    assert "Conditioning check failed" in window.messages.toPlainText()
    assert window.project is before and window.undo_stack.count() == 0


def test_completed_notice_preserves_user_selected_result_and_plot_until_action(
    windows, tmp_path, application, monkeypatch
):
    window = windows()
    before = window.project
    metadata = {
        "project_id": before.id,
        "project_fingerprint": before.fingerprint,
        "project_revision": before.revision,
        "pipeline_stage": "evaluated",
        "summary": {"scope": "continuous_sheet", "sheet_accepted": False},
    }
    old = SimpleNamespace(
        directory=tmp_path / "earlier", metadata={**metadata, "name": "Earlier inspected trial"}, arrays={}
    )
    finished = SimpleNamespace(
        directory=tmp_path / "finished", metadata={**metadata, "name": "New completed trial"}, arrays={}
    )
    window.runs["earlier"] = old
    window.show_run("earlier")
    window.viewport.state = {"dataset": "target", "units": "µT"}
    window._pause_live_follow()
    window.evidence.close_all()
    target = {"id": "finished", "project_id": before.id, "status": "running", "name": "New completed trial"}
    window.jobs.states = [target]
    window.poll_jobs()
    window.runs[target["id"]] = finished
    window._loaded_ids.add(target["id"])
    target["status"] = "completed"
    window.poll_jobs()
    assert window._selected_run == "earlier" and window.viewport.displayed is old.metadata
    assert window.viewport.view_state() == {"dataset": "target", "units": "µT"}
    assert not window.evidence.any_open()
    assert "engineering checks failed" in window.job_outcomes.selected.title
    assert window.job_outcomes.action_button.text() == "Review checks"
    loader = Mock(return_value=finished)
    monkeypatch.setattr(window.service, "load_run", loader)
    window.job_outcomes.action_button.click()
    application.processEvents()
    loader.assert_called_once_with(finished.directory)
    assert window._selected_run == "finished" and window.viewport.displayed is finished.metadata
    assert window.evidence.window_for(window.acceptance_panel).isVisible()
    assert window.project is before and window.undo_stack.count() == 0


def test_completed_source_verification_failure_has_persistent_notice_without_auto_popup(windows, monkeypatch):
    window = windows()
    state = {"id": "bad-evidence", "project_id": window.project.id, "status": "running"}
    window.jobs.states = [state]
    window.poll_jobs()
    window.evidence.close_all()
    entered, release = Event(), Event()

    def changed_evidence(_directory):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("Test did not release background evidence verification")
        raise ValueError("Scientific artifact checksum changed")

    loader = Mock(side_effect=changed_evidence)
    monkeypatch.setattr(window.service, "load_run", loader)
    state["status"] = "completed"
    try:
        window.poll_jobs()
        wait_until(entered.is_set)
        assert state["id"] in window._run_load_tasks
        assert window.job_outcomes.selected is None
        assert not window.evidence.any_open()
        assert "Verifying completed results" in window.computation_indicator.toolTip()
        assert window.computation_indicator.label.text() == "Working…"
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        wait_until(lambda: bool(heartbeat))
        assert not release.is_set()  # Qt remains responsive while the file worker is held.
    finally:
        release.set()
        wait_until(lambda: not window._run_load_tasks)
    wait_until(lambda: window.job_outcomes.selected is not None)
    assert not window.evidence.any_open()
    notice = window.job_outcomes.selected
    assert notice.job_id == state["id"] and notice.tone == "error"
    assert "evidence verification failed" in notice.title
    assert "Scientific artifact checksum changed" in notice.detail
    assert not window.job_outcomes.isHidden()
    window.poll_jobs()
    assert loader.call_count == 1 and len(window.job_outcomes.outcomes) == 1
    assert window.job_outcomes.selected == notice


def test_background_outcome_survives_project_close_and_arrives_on_welcome_screen(windows, application):
    window = windows()
    state = {"id": "background-job", "project_id": window.project.id, "status": "running", "name": "Closed-channel Gx"}
    window.jobs.states = [state]
    window.poll_jobs()
    assert window.execution_progress.maximum() == 0
    window._clear_project()
    assert window.project is None
    state.update(status="failed", error="Worker stopped unexpectedly")
    window.poll_jobs()
    assert window.project is None and not window.evidence.any_open()
    assert not window.job_outcomes.isHidden()
    assert window.job_outcomes.selected.job_id == state["id"]
    assert "Worker stopped unexpectedly" in window.job_outcomes.selected.detail
    assert window.computation_indicator.isHidden()
    window.job_outcomes.action_button.click()
    application.processEvents()
    assert window.project is None and window._selected_job_id() == state["id"]
    assert window.evidence.window_for(window.jobs_table).isVisible()
    assert "Execution failed" in window.execution_label.text()
    assert "Worker stopped unexpectedly" in window.execution_label.text()
    assert window.execution_label.property("semanticTone") == "error"
    assert window.execution_progress.maximum() == 100 and window.execution_progress.value() == 0
    assert window.execution_progress.format() == "Unreported"
    assert not window.compute_button.isEnabled() and not window.compute_action.isEnabled()
    assert not window.view_results_button.isEnabled() and not window.resume_action.isVisible()


def test_verified_engineering_failure_notice_keeps_its_exact_scope_after_project_close(windows, tmp_path):
    window = windows()
    state = {
        "id": "checked-failure",
        "project_id": window.project.id,
        "project_revision": window.project.revision,
        "name": "Gradient winding trial",
        "status": "running",
    }
    window.jobs.states = [state]
    window.poll_jobs()
    record = SimpleNamespace(
        directory=tmp_path / state["id"],
        arrays={},
        metadata={
            "project_id": window.project.id,
            "pipeline_stage": "evaluated",
            "summary": {"scope": "continuous_sheet", "sheet_accepted": False},
            "constraints": [{"name": "Independent field error", "passed": False}],
        },
    )
    # The controller consumes the verified record cache; artifact verification
    # itself has dedicated source-health and completed-load regressions.
    window.runs[state["id"]] = record
    window._loaded_ids.add(state["id"])
    state["status"] = "completed"
    window.poll_jobs()
    notice = window.job_outcomes.selected
    assert notice.tone == "error" and "engineering checks failed" in notice.title
    assert "1 failed" in notice.detail and "Saved sheet checks: Failed" in notice.detail
    window._clear_project()
    assert window.runs == {} and window.project is None
    assert window.job_outcomes.action_button.text() == "View job"
    assert window.job_outcomes.selected.action_label == "Review checks"
    window.poll_jobs()
    assert window.job_outcomes.selected == notice
    assert window.job_outcomes.label.property("semanticTone") == "error"
    assert notice.detail in window.job_outcomes.label.toolTip()
    assert "evidence unverified" not in window.job_outcomes.selected.title


def test_welcome_notice_action_caches_job_state_for_later_log_selection(windows, application):
    window = windows()
    state = {
        "id": "welcome-failure",
        "project_id": window.project.id,
        "name": "Background winding computation",
        "status": "running",
    }
    window.jobs.states = [state]
    window.poll_jobs()
    state.update(status="failed", error="Recorded optimization breakdown", log=["Solver iteration 47"])
    window.poll_jobs()
    window._clear_project()
    assert window.project is None and window.states == {}
    assert window.job_outcomes.selected.job_id == state["id"]
    window.job_outcomes.action_button.click()
    application.processEvents()
    assert window.states[state["id"]] == state
    assert window._selected_job_id() == state["id"]
    assert "Solver iteration 47" in window.messages.toPlainText()
    assert "Recorded optimization breakdown" in window.messages.toPlainText()
    window.messages.clear()
    window._job_selected(window.jobs_table.currentRow())
    assert "Solver iteration 47" in window.messages.toPlainText()
    assert "Recorded optimization breakdown" in window.messages.toPlainText()
    assert window.project is None


@pytest.mark.parametrize("progress", [None, 0.45])
def test_runtime_failure_stops_animation_identifies_last_known_state_and_recovers(windows, monkeypatch, progress):
    window = windows()
    state = {
        "id": "runtime-job",
        "name": "Running gradient design",
        "project_id": window.project.id,
        "status": "running",
        "progress": progress,
    }
    window.jobs.states = [state]
    window.poll_jobs()
    window._focus_job(state["id"], open_window=False)
    loader = window.jobs.list

    def disconnected():
        raise OSError("Job state connection lost")

    monkeypatch.setattr(window.jobs, "list", disconnected)
    window.poll_jobs()
    assert window.execution_progress.maximum() == 100
    assert window.execution_progress.format() == ("Unavailable" if progress is None else "Last reported: %p%")
    assert window.execution_progress.value() == (0 if progress is None else 45)
    assert window.execution_label.property("semanticTone") == "warning"
    assert "runtime unavailable" in window.execution_label.text().lower()
    assert window.computation_indicator.label.text() == "Activity unavailable"
    assert window._guidance_state.title == "Computation activity unavailable"
    assert window._guidance_state.action_label == "View last known job"
    assert window._guidance_state.tone == "warning" and window._guidance_state.job_id == state["id"]
    assert "Last recorded job state" in window._guidance_state.detail
    assert not window.cancel_action.isEnabled() and not window.resume_action.isEnabled()
    assert window.states[state["id"]]["status"] == "running"

    monkeypatch.setattr(window.jobs, "list", loader)
    window.poll_jobs()
    assert window._last_runtime_error is None
    assert window.execution_progress.maximum() == (0 if progress is None else 100)
    assert window.execution_label.property("semanticTone") == "info"
    assert window._guidance_state.title != "Computation activity unavailable"
    assert window._guidance_state.job_id == state["id"]
    assert window.cancel_action.isEnabled()
    assert window.states[state["id"]]["progress"] == progress


def test_runtime_failure_keeps_pending_model_guidance_priority(windows, monkeypatch):
    window = windows()
    before = window.project
    window.jobs.states = [{"id": "running", "project_id": before.id, "status": "running"}]
    window.poll_jobs()
    window._draft_controls["radius_mm"].setValue(before.config.cylinder_mesh_parameter_list[1] * 1000 + 10)
    guidance = window._guidance_state
    assert "Pending" in guidance.title

    def disconnected():
        raise OSError("Job state connection lost")

    monkeypatch.setattr(window.jobs, "list", disconnected)
    window.poll_jobs()
    assert window._guidance_state.title == guidance.title
    assert window._guidance_state.action_key == guidance.action_key
    assert window.project is before and window._draft_dirty


@pytest.mark.parametrize("verified", [True, False])
def test_completed_notice_explicitly_reloads_lost_record_after_reopen_without_applying_draft(
    windows, tmp_path, monkeypatch, application, verified
):
    window = windows()
    path = window.save_project(tmp_path / "outcome-owner.coilgen.json")
    state = {
        "id": "reopen-completed",
        "name": "Completed coil trial",
        "project_id": window.project.id,
        "project_revision": window.project.revision,
        "status": "running",
    }
    window.jobs.states = [state]
    window.poll_jobs()
    record = SimpleNamespace(
        directory=window.jobs.directory / state["id"],
        arrays={},
        metadata={
            "name": state["name"],
            "project_id": window.project.id,
            "project_fingerprint": window.project.fingerprint,
            "project_revision": window.project.revision,
            "pipeline_stage": "evaluated",
            "summary": {"scope": "continuous_sheet", "sheet_accepted": False},
        },
    )
    window.runs[state["id"]] = record
    window._loaded_ids.add(state["id"])
    state["status"] = "completed"
    window.poll_jobs()
    window._clear_project()
    assert window.job_outcomes.action_button.text() == "View job"
    window.open_project(path)
    assert state["id"] not in window.runs
    before = window.project
    radius = window._draft_controls["radius_mm"]
    pending_radius = radius.value() + 10
    radius.setValue(pending_radius)
    loader = Mock(return_value=record) if verified else Mock(side_effect=ValueError("Saved evidence checksum changed"))
    monkeypatch.setattr(window.service, "load_run", loader)
    window.job_outcomes.action_button.click()
    application.processEvents()
    loader.assert_called_once_with(window.jobs.directory / state["id"])
    assert window.project is before and window._draft_dirty
    assert window._draft_controls["radius_mm"].value() == pytest.approx(pending_radius)
    if verified:
        assert window.runs[state["id"]] is record and window._selected_run == state["id"]
        assert window.evidence.window_for(window.acceptance_panel).isVisible()
        assert "engineering checks failed" in window.job_outcomes.selected.title
    else:
        assert "Saved evidence checksum changed" in window._run_verification_error(state["id"])
        assert "evidence verification failed" in window.job_outcomes.selected.title
        assert window.job_outcomes.selected.tone == "error"
        assert window.evidence.window_for(window.messages).isVisible()


@pytest.mark.parametrize("failure", ["checksum", "replaced_metadata"])
def test_cached_completed_notice_reverifies_source_and_preserves_prior_plot_and_draft_on_failure(
    windows, tmp_path, monkeypatch, application, failure
):
    window = windows()
    before = window.project
    metadata = {
        "project_id": before.id,
        "project_fingerprint": before.fingerprint,
        "project_revision": before.revision,
        "pipeline_stage": "evaluated",
        "summary": {"scope": "continuous_sheet", "sheet_accepted": False},
    }
    old = SimpleNamespace(
        directory=tmp_path / "earlier",
        metadata={**deepcopy(metadata), "name": "Previously inspected result"},
        arrays={},
    )
    cached = SimpleNamespace(
        directory=tmp_path / "cached", metadata={**deepcopy(metadata), "name": "Cached completed result"}, arrays={}
    )
    window.runs["earlier"] = old
    window.show_run("earlier")
    window.viewport.state = {"dataset": "target", "units": "µT"}
    window._pause_live_follow()
    state = {"id": "cached", "project_id": before.id, "status": "running", "name": cached.metadata["name"]}
    window.jobs.states = [state]
    window.poll_jobs()
    window.runs[state["id"]] = cached
    window._loaded_ids.add(state["id"])
    state["status"] = "completed"
    window.poll_jobs()
    assert window._selected_run == "earlier" and window.job_outcomes.selected.job_id == state["id"]
    radius = window._draft_controls["radius_mm"]
    pending_radius = radius.value() + 10
    radius.setValue(pending_radius)
    if failure == "checksum":
        loader = Mock(side_effect=ValueError("Cached scientific artifact checksum changed"))
        diagnostic = "Cached scientific artifact checksum changed"
    else:
        replaced = deepcopy(cached.metadata)
        replaced["summary"]["sheet_accepted"] = True
        replaced["metadata_sha256"] = result_metadata_digest(replaced)
        fresh = SimpleNamespace(directory=cached.directory, metadata=replaced, arrays={})
        loader = Mock(return_value=fresh)
        diagnostic = "Saved run metadata changed after selection"
    monkeypatch.setattr(window.service, "load_run", loader)
    window.job_outcomes.action_button.click()
    application.processEvents()
    loader.assert_called_once_with(cached.directory)
    assert diagnostic in window._run_verification_error(state["id"])
    assert window.runs[state["id"]] is cached
    assert window._selected_run == "earlier" and window.viewport.displayed is old.metadata
    assert window.viewport.view_state() == {"dataset": "target", "units": "µT"}
    assert window.project is before and window._draft_dirty
    assert window._draft_controls["radius_mm"].value() == pytest.approx(pending_radius)
    assert "evidence verification failed" in window.job_outcomes.selected.title
    assert window.job_outcomes.selected.tone == "error"
    assert diagnostic in window.job_outcomes.selected.detail
    assert window.evidence.window_for(window.messages).isVisible()


def test_welcome_selected_active_job_can_be_cancelled_without_reopening_project(windows, monkeypatch, application):
    window = windows()
    state = {
        "id": "closed-project-worker",
        "name": "Background Gy calculation",
        "project_id": window.project.id,
        "status": "running",
        "progress": None,
    }
    window.jobs.states = [state]
    window.poll_jobs()
    window._clear_project()
    window.poll_jobs()
    window._focus_job(state["id"])
    application.processEvents()
    assert window.project is None and window._selected_job_id() == state["id"]
    assert "Computing" in window.execution_label.text()
    assert window.execution_progress.maximum() == 0
    assert window.cancel_action.isEnabled() and window.cancel_action.isVisible()
    assert not window.compute_button.isEnabled() and not window.view_results_button.isEnabled()
    assert not window.resume_action.isVisible()

    def cancel(id_):
        assert id_ == state["id"]
        state["status"] = "cancelled"

    cancellation = Mock(side_effect=cancel)
    monkeypatch.setattr(window.jobs, "cancel", cancellation, raising=False)
    window.cancel_action.trigger()
    application.processEvents()
    cancellation.assert_called_once_with(state["id"])
    assert window.project is None and window.states[state["id"]]["status"] == "cancelled"
    assert "Cancelled" in window.execution_label.text()
    assert window.execution_label.property("semanticTone") == "warning"
    assert window.execution_progress.maximum() == 100 and window.execution_progress.format() == "Unreported"
    assert not window.cancel_action.isEnabled() and not window.resume_action.isVisible()
    assert not window.compute_button.isEnabled() and not window.view_results_button.isEnabled()
