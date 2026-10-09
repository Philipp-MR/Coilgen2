"""Background research work supplies bounded, final-preserving GUI feedback."""

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from coilgen.gui.array_studies import ArrayStudyDialog, _ArrayTask
from coilgen.gui.design_tasks import DesignStudyTask, ProgressSignalRelay
from coilgen.progress import ProgressReporter, progress_scope


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_signal_bursts_preserve_latest_update_scientific_boundaries_and_final_completion(monkeypatch):
    import coilgen.progress as progress

    now = [0.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
    updates = []
    relay = ProgressSignalRelay(updates.append)
    relay.report({"label": "Field", "overall_percent": 0})
    for index in range(1, 101):
        relay.report({"label": "Field", "overall_percent": index / 2})
    assert len(updates) == 1
    relay.flush()
    assert updates[-1]["overall_percent"] == 50
    relay.report({"label": "CoilGen", "completed_stage": "sheet", "percent": None})
    assert updates[-1]["completed_stage"] == "sheet"
    relay.report({"label": "Field", "overall_percent": 75})
    relay.report({"label": "CoilGen", "overall_percent": 100})
    assert updates[-1]["overall_percent"] == 100
    count = len(updates)
    relay.flush()
    assert len(updates) == count
    now[0] = 0.11
    relay.report({"label": "Field", "overall_percent": 20})
    assert updates[-1]["overall_percent"] == 20


def test_general_study_worker_forwards_progress_and_failed_work_never_completes(application, monkeypatch):
    import coilgen.progress as progress

    monkeypatch.setattr(progress.time, "monotonic", lambda: 0)
    project = object()

    def operation(captured):
        assert captured is project
        with progress_scope(0, 100, overall=True):
            ProgressReporter(label="Study").report(40, "Completed first physical operator")
            raise ValueError("Fixture failure after real progress")

    task = DesignStudyTask(operation, project)
    events, finished = [], []
    task.signals.progress.connect(events.append)
    task.signals.finished.connect(lambda result, error: finished.append((result, error)))
    task.run()
    assert events[-1]["overall_percent"] == 40
    assert not any(event.get("overall_percent") == 100 for event in events)
    assert finished[0][0] is None and "ValueError" in finished[0][1]


def test_opaque_entry_is_visible_before_blocking_work_and_repeated_entries_are_bounded(monkeypatch):
    import coilgen.progress as progress

    monkeypatch.setattr(progress.time, "monotonic", lambda: 0)
    updates = []
    relay = ProgressSignalRelay(updates.append)
    relay.report({"label": "Field", "overall_percent": 40, "progress_indeterminate": False})
    relay.report({"label": "Solve", "overall_percent": 40, "progress_indeterminate": True, "message": "Factorization"})
    assert updates[-1]["message"] == "Factorization"
    for index in range(1000):
        relay.report({"label": "Solve", "overall_percent": 40, "progress_indeterminate": True, "message": str(index)})
    assert len(updates) == 2
    relay.flush()
    assert updates[-1]["message"] == "999"
    relay.report({"label": "Solve", "overall_percent": 45, "progress_indeterminate": False})
    relay.flush()
    relay.report({"label": "Solve", "percent": None, "message": "Legacy opaque work"})
    assert updates[-1]["message"] == "Legacy opaque work" and updates[-1]["progress_indeterminate"]


def test_array_cancellation_reaches_nested_kernel_context_and_never_reports_complete(application):
    events, finished = [], []

    def operation(_text_progress):
        with progress_scope(0, 100, overall=True):
            reporter = ProgressReporter(label="Wire kernel")
            reporter.report(0, "Starting first sample chunk")
            task.cancel.set()
            reporter.report(50, "This cancelled chunk must not publish")
        return {}

    task = _ArrayTask(operation)
    task.signals.details.connect(events.append)
    task.signals.finished.connect(lambda result, error: finished.append((result, error)))
    task.run()
    assert finished[0][0] is None and "DesignCancelled" in finished[0][1]
    assert not any(event.get("overall_percent", 0) > 0 for event in events)


def test_array_progress_bar_distinguishes_opaque_work_from_counted_completion(application):
    dialog = ArrayStudyDialog(None)
    dialog._progress_details({"overall_percent": 45, "progress_indeterminate": False})
    assert dialog.progress.maximum() == 1000 and dialog.progress.value() == 450
    dialog._progress_details({"overall_percent": 45, "progress_indeterminate": True})
    assert dialog.progress.maximum() == 0
    dialog._progress_details({"overall_percent": 99.99, "progress_indeterminate": False})
    assert dialog.progress.value() == 999
    dialog._progress_details({"overall_percent": 100, "progress_indeterminate": False})
    assert dialog.progress.value() == 1000
    dialog.close()
