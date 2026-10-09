"""The bottom-left activity indicator follows workers, not viewed evidence."""

import pytest

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QDockWidget

from test_workbench_usability import application as application, windows as windows


def publish(window, *, status="running", progress=0.45, **fields):
    state = {
        "id": "worker",
        "name": "Gradient study",
        "status": status,
        "progress": progress,
        "project_id": window.project.id if window.project else "closed-project",
        **fields,
    }
    window.jobs.states = [state]
    window._loaded_ids.add(state["id"])
    window.poll_jobs()
    return state


def test_running_indicator_stays_in_lower_left_when_windows_and_status_messages_change(windows, application):
    window = windows()
    window.show()
    publish(window)
    indicator = window.computation_indicator
    window.findChild(QDockWidget, "settings_dock").hide()
    window.evidence.close_all()
    window.statusBar().showMessage("Saved current project")
    application.processEvents()
    assert indicator.isVisible() and indicator.label.text() == "Computing (45%)"
    assert indicator.progress.value() == 45
    assert indicator.message.text() == "Saved current project"
    location = indicator.label.mapTo(window, QPoint(0, 0))
    assert location.x() <= 10
    assert location.y() >= window.height() - window.statusBar().height()
    # Long notifications yield their width to the persistent computation label.
    window.resize(800, 600)
    window.statusBar().showMessage("A long saved-project notification " * 30)
    application.processEvents()
    assert indicator.label.width() >= indicator.label.sizeHint().width()
    assert indicator.label.mapTo(window.statusBar(), QPoint(0, 0)).x() <= 10
    window.statusBar().clearMessage()
    assert indicator.label.text() == "Computing (45%)"
    assert indicator.message.text() == ""


def test_historical_selection_paused_follow_and_new_channel_do_not_hide_running_worker(windows, monkeypatch):
    window = windows()
    state = publish(window)
    window.states["historical"] = {"id": "historical", "status": "completed", "progress": 1.0}
    monkeypatch.setattr(window, "_selected_job_id", lambda: "historical")
    window._pause_live_follow()
    window._update_execution_feedback()
    assert window.computation_indicator.label.text() == "Computing (45%)"
    window.add_channel("Another coil")
    window.poll_jobs()
    assert window.project.id != state["project_id"]
    assert window.computation_indicator.label.text() == "Computing (45%)"


def test_welcome_screen_still_tracks_worker_until_it_stops(windows):
    window = windows()
    publish(window)
    window._clear_project()
    assert window.project is None
    window.jobs.states[0]["progress"] = 0.63
    window.poll_jobs()
    assert window.computation_indicator.label.text() == "Computing (63%)"
    window.jobs.states[0]["status"] = "completed"
    window.poll_jobs()
    assert window.computation_indicator.isHidden()


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "interrupted"])
def test_terminal_worker_state_clears_computing_indicator(windows, status):
    window = windows()
    publish(window)
    publish(window, status=status, progress=1.0)
    assert window.computation_indicator.isHidden()
    assert window.computation_indicator.label.text() == ""


@pytest.mark.parametrize("progress", [None, float("nan"), float("inf"), True, "0.45", -0.2, 1.2])
def test_unreported_or_invalid_progress_keeps_activity_visible_without_invented_percent(windows, progress):
    window = windows()
    publish(window, progress=progress)
    indicator = window.computation_indicator
    assert not indicator.isHidden() and indicator.label.text() == "Computing…"
    assert indicator.progress.minimum() == indicator.progress.maximum() == 0


def test_queue_and_cancellation_have_distinct_labels(windows):
    window = windows()
    publish(window, status="queued", progress=0)
    assert window.computation_indicator.label.text() == "Queued"
    assert window.computation_indicator.progress.isHidden()
    publish(window, status="cancelling")
    assert window.computation_indicator.label.text() == "Cancelling…"
    assert window.computation_indicator.progress.maximum() == 0


def test_multiple_workers_show_count_and_individual_progress_without_false_total(windows):
    window = windows()
    first = publish(window)
    window.jobs.states.append(
        {**first, "id": "another", "name": "Another channel", "project_id": "other", "progress": 0.8}
    )
    window.poll_jobs()
    indicator = window.computation_indicator
    assert indicator.label.text() == "Computing (2 jobs)"
    assert "Gradient study: running (45%)" in indicator.toolTip()
    assert "Another channel: running (80%)" in indicator.toolTip()
    assert indicator.progress.maximum() == 0


def test_background_work_is_visible_even_without_a_design_job(windows):
    window = windows()
    task = object()
    window._gradient_tasks = {task: (task, None, "Compare measured winding field")}
    window.statusBar().showMessage("Comparing independent measurements")
    indicator = window.computation_indicator
    assert indicator.label.text() == "Working…" and not indicator.isHidden()
    assert "Compare measured winding field" in indicator.toolTip()
    window._gradient_tasks.clear()
    window._refresh_computation_indicator()
    assert indicator.isHidden()


def test_local_field_work_reports_percent_phase_and_unknown_factorization(windows):
    window = windows()
    label = "Field analysis"
    window._research_busy(label, True)
    window._research_progress(
        label, {"percent": 37, "message": "slice point integration", "completed": 37, "total": 100}
    )
    indicator = window.computation_indicator
    assert indicator.label.text() == "Computing (37%)" and indicator.progress.value() == 37
    assert "37/100 work units" in indicator.toolTip()
    window._research_progress(label, {"percent": None, "message": "harmonic factorization"})
    assert indicator.label.text() == "Computing…" and indicator.progress.maximum() == 0
    assert "harmonic factorization" in indicator.toolTip()
    window._research_busy(label, False)
    assert indicator.isHidden() and label not in window._local_progress
    # Late queued callbacks cannot resurrect a completed/closed task.
    window._research_progress(label, {"percent": 98, "message": "late callback"})
    assert indicator.isHidden()


def test_opaque_worker_keeps_completed_work_floor_and_shows_busy_activity(windows):
    window = windows()
    publish(window, progress=0.45, progress_indeterminate=True, phase_progress=None, stage="Sparse factorization")
    indicator = window.computation_indicator
    assert indicator.label.text() == "Computing (45%)" and indicator.progress.maximum() == 0
    assert "current operation has no measurable work total" in indicator.toolTip()
    publish(
        window, progress=0.46, progress_indeterminate=False, phase_progress=0.3, progress_completed=3, progress_total=10
    )
    assert indicator.label.text() == "Computing (46%)" and indicator.progress.value() == 46
    assert "current operation 30% (3/10 work units)" in indicator.toolTip()


def test_nearly_finished_local_work_cannot_round_to_100_percent(windows):
    window = windows()
    window._research_busy("Field analysis", True)
    window._research_progress("Field analysis", {"percent": 99.95, "message": "Final samples"})
    assert window.computation_indicator.label.text() == "Computing (99%)"
    window._research_progress("Field analysis", {"percent": 100, "message": "Complete"})
    assert window.computation_indicator.label.text() == "Computing (100%)"
    window._research_busy("Field analysis", False)


def test_two_copied_result_analyses_with_same_label_keep_independent_activity(windows):
    window = windows()
    label = "Field analysis · identical saved run"
    first, second = ("first directory", "digest"), ("second directory", "digest")
    window._research_busy(label, True, key=first)
    window._research_busy(label, True, key=second)
    assert window.computation_indicator.label.text() == "Working (2 tasks)"
    window._research_busy(label, False, key=first)
    assert window._research_active_tasks == {second}
    window._research_progress(label, {"percent": 24, "message": "Samples"}, key=second)
    assert window.computation_indicator.label.text() == "Computing (24%)"
    window._research_busy(label, False, key=second)
    assert window.computation_indicator.isHidden()


def test_engineering_task_progress_signal_updates_indicator_and_clears_on_finish(windows):
    from coilgen.gui.design_tasks import DesignStudyTask

    window = windows()
    task = DesignStudyTask(lambda project: None, window.project)
    window._gradient_tasks = {task.signals: (task, lambda value: None, "Measurement comparison")}
    task.signals.progress.connect(window._gradient_action_progress)
    task.signals.finished.connect(window._gradient_action_finished)
    task.signals.progress.emit({"percent": 61, "message": "Compared samples"})
    assert window.computation_indicator.label.text() == "Computing (61%)"
    task.signals.finished.emit(None, "")
    assert window.computation_indicator.isHidden() and task.signals not in window._local_progress


def test_failed_runtime_refresh_does_not_present_stale_percent_as_current_activity(windows, monkeypatch):
    window = windows()
    publish(window)
    original = window.jobs.list

    def unavailable():
        raise OSError("worker state unavailable")

    monkeypatch.setattr(window.jobs, "list", unavailable)
    window.poll_jobs()
    assert window.computation_indicator.label.text() == "Activity unavailable"
    assert "worker state unavailable" in window.computation_indicator.toolTip()
    monkeypatch.setattr(window.jobs, "list", original)
    window.poll_jobs()
    assert window.computation_indicator.label.text() == "Computing (45%)"


def test_local_export_stays_active_while_job_runtime_is_unavailable_and_recovers(windows, monkeypatch):
    window = windows()
    publish(window)
    original = window.jobs.list
    window._artifact_export_task = object()

    def unavailable():
        raise OSError("worker state unavailable")

    monkeypatch.setattr(window.jobs, "list", unavailable)
    window.poll_jobs()
    indicator = window.computation_indicator
    assert indicator.label.text() == "Working… · job activity unavailable"
    assert indicator.label.property("semanticTone") == "warning"
    assert not indicator.progress.isHidden() and indicator.progress.maximum() == 0
    assert "worker state unavailable" in indicator.toolTip()
    assert "Exporting saved artifacts" in indicator.toolTip()
    assert "45%" not in indicator.label.text() and "45%" not in indicator.toolTip()
    window._artifact_export_task = None
    window._refresh_computation_indicator()
    assert indicator.label.text() == "Activity unavailable" and indicator.progress.isHidden()
    window._artifact_export_task = object()
    monkeypatch.setattr(window.jobs, "list", original)
    window.poll_jobs()
    assert indicator.label.text() == "Computing (2 tasks)" and indicator.progress.maximum() == 0
    assert "Exporting saved artifacts" in indicator.toolTip()
    window._artifact_export_task = None
    publish(window, status="completed", progress=1)
    assert indicator.isHidden()


def test_concurrent_field_analysis_and_job_have_individual_percentages_without_ambiguous_total(windows):
    window = windows()
    publish(window, progress=0.53)
    window._research_busy("Field analysis", True)
    window._research_progress("Field analysis", {"percent": 19, "message": "Slice integration"})
    indicator = window.computation_indicator
    assert indicator.label.text() == "Computing (2 tasks)" and indicator.progress.maximum() == 0
    assert "Gradient study: running (53%)" in indicator.toolTip()
    assert "Field analysis (19%)" in indicator.toolTip()
    window._research_busy("Field analysis", False)
    assert indicator.label.text() == "Computing (53%)"


def test_repeated_poll_processing_errors_are_reported_once_and_recover(windows, monkeypatch):
    window = windows()
    publish(window)
    original = window._fill_jobs
    errors = []
    monkeypatch.setattr(window, "_error", errors.append)

    def unavailable(_states):
        raise ValueError("cannot display worker state")

    monkeypatch.setattr(window, "_fill_jobs", unavailable)
    window.poll_jobs()
    window.poll_jobs()
    assert errors == ["Job runtime unavailable: cannot display worker state"]
    assert window.computation_indicator.label.text() == "Activity unavailable"
    monkeypatch.setattr(window, "_fill_jobs", original)
    window.poll_jobs()
    assert window._last_runtime_error is None
    assert window.computation_indicator.label.text() == "Computing (45%)"
