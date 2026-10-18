"""Project opening verifies evidence in a worker and preserves the current owner."""

from contextlib import contextmanager
import json
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

from coilgen.application import DesignService, ProjectAssembly, ProjectDocument, RunArchive
from coilgen.application.presets import preset_config
from coilgen.gui import project_loading, workbench
from test_workbench_project_lifecycle import _owned_run
from test_workbench_usability import application as application, windows as windows


def wait_until(predicate, timeout_ms=10000):
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
    assert predicate(), "Background project opening did not settle before the deadline"


@contextmanager
def held_document(monkeypatch, path):
    """Hold real parsing, then drain every started task before releasing patches."""
    held = SimpleNamespace(entered=Event(), release=Event(), threads=[], started=set(), done=set())
    original_load, original_run = project_loading.load_document, project_loading.ProjectOpenTask.run

    def load(source, **kwargs):
        if Path(source).resolve() == path.resolve():
            held.threads.append(get_ident())
            held.entered.set()
            assert held.release.wait(10), "Test did not release the project archive worker"
        return original_load(source, **kwargs)

    def run(task):
        held.started.add(task.token)
        try:
            return original_run(task)
        finally:
            held.done.add(task.token)

    monkeypatch.setattr(project_loading, "load_document", load)
    monkeypatch.setattr(project_loading.ProjectOpenTask, "run", run)
    try:
        yield held
    finally:
        held.release.set()
        wait_until(lambda: held.started <= held.done)
        QCoreApplication.processEvents()  # Deliver the finished signal queued before run() returned.


@pytest.fixture
def incoming(tmp_path):
    project = ProjectDocument.from_config(preset_config(), name="Incoming verified project")
    record = _owned_run(project, tmp_path / "incoming-run")
    archive = RunArchive.capture(record, alias="Original field evidence")
    view = {"source": "run", "dataset": "target", "units": "µT", "model_node": "results"}
    project = project.with_results((archive,), session={"selected_run_id": archive.id, "view_state": view})
    return SimpleNamespace(
        project=project,
        record=record,
        archive=archive,
        view=view,
        path=project.save(tmp_path / "incoming.coilgen.json"),
    )


def retain_path(window, tmp_path):
    window.project_path = window.assembly.save(tmp_path / "current.coilgen.json")
    window._can_discard = lambda: True
    return window.project_path


def test_file_open_is_responsive_and_commits_verified_history_without_gui_thread_io(
    windows, incoming, tmp_path, monkeypatch
):
    window = windows()
    previous, assembly = window.project, window.assembly
    previous_path = retain_path(window, tmp_path)
    main_thread, calls = get_ident(), []
    pages, set_widget = [], window.settings.setWidget
    materialize, restore, load_run = RunArchive.materialize, window.history.restore, DesignService.load_run
    readiness, reuse = project_loading.check_readiness, project_loading.execution_reuse

    def archive_io(*args, **kwargs):
        calls.append(("materialize", get_ident()))
        return materialize(*args, **kwargs)

    def history_io(*args, **kwargs):
        calls.append(("history.restore", get_ident()))
        return restore(*args, **kwargs)

    def result_io(*args, **kwargs):
        calls.append(("load_run", get_ident()))
        return load_run(*args, **kwargs)

    def readiness_io(*args, **kwargs):
        calls.append(("readiness", get_ident()))
        return readiness(*args, **kwargs)

    def reuse_io(*args, **kwargs):
        calls.append(("reuse", get_ident()))
        return reuse(*args, **kwargs)

    def build_page(widget):
        pages.append(window._workflow_page())
        return set_widget(widget)

    monkeypatch.setattr(RunArchive, "materialize", archive_io)
    monkeypatch.setattr(window.history, "restore", history_io)
    monkeypatch.setattr(DesignService, "load_run", staticmethod(result_io))
    monkeypatch.setattr(window.service, "load_run", result_io)
    monkeypatch.setattr(project_loading, "check_readiness", readiness_io)
    monkeypatch.setattr(workbench, "check_readiness", readiness_io)
    monkeypatch.setattr(project_loading, "execution_reuse", reuse_io)
    monkeypatch.setattr(workbench, "execution_reuse", reuse_io)
    monkeypatch.setattr(window.settings, "setWidget", build_page)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(incoming.path), ""))
    monkeypatch.setattr(window, "open_project", lambda *_args: pytest.fail("File Open must use its worker"))
    ticks, timer = [], QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(True))
    with held_document(monkeypatch, incoming.path) as held:
        timer.start()
        try:
            window._open_dialog()
            wait_until(held.entered.is_set)
            wait_until(lambda: len(ticks) >= 3, timeout_ms=1000)
            assert len(held.threads) == 1 and held.threads[0] != main_thread
            assert window.project is previous and window.assembly is assembly
            assert window.project_path == previous_path and not window.runs
            assert window.undo_stack.count() == 0 and window._project_open_task is not None
            assert not window.computation_indicator.isHidden()
            assert window.computation_indicator.progress.maximum() == 0
            assert not held.release.is_set()
            held.release.set()
            wait_until(lambda: window._project_open_task is None)
        finally:
            timer.stop()
    assert window.project.id == incoming.project.id and window.project_path == incoming.path
    assert window.project.fingerprint == incoming.project.fingerprint
    assert window._prepared_project_history is None and window.undo_stack.isClean()
    assert window._selected_run in window.runs and window._history_entries[0].verified
    selected = window.runs[window._selected_run]
    for name, values in incoming.record.arrays.items():
        np.testing.assert_array_equal(selected.arrays[name], values)
    assert window.viewport.view_state() == incoming.view
    assert {name for name, _thread in calls} == {"materialize", "history.restore", "load_run", "readiness", "reuse"}
    assert all(thread != main_thread for _name, thread in calls), calls
    assert window.computation_indicator.isHidden() and window._project_open_cancel.isHidden()
    assert pages == ["results"]  # Intermediate restore choices must not rebuild expensive pages.
    restored = window.project
    window.select_node("geometry")
    assert pages == ["results", "geometry"] and window.settings.widget() is not None
    radius = window._draft_controls["radius_mm"]
    radius.setValue(radius.value() + 2)
    assert window._draft_dirty and window.project is restored  # Final controls still edit an unapplied draft.


def test_async_collection_open_restores_saved_active_channel_and_history_scope(windows, incoming, tmp_path):
    other = ProjectDocument.from_config(preset_config(), name="Other incoming channel")
    assembly = ProjectAssembly.new("Incoming collection", (other, incoming.project))
    selected = assembly.channels[1]
    project = selected.project.with_results(
        selected.project.saved_runs,
        session={
            **selected.project.session,
            "active_channel_id": selected.id,
        },
    )
    assembly = assembly.update_channel(selected.id, project=project)
    path = assembly.save(tmp_path / "incoming-collection.coilgen.json")
    window = windows()
    window._start_project_open(path)
    wait_until(lambda: window._project_open_task is None)
    assert window._collection_mode and window.assembly.id == assembly.id
    assert window._active_channel_id == selected.id and window.project.id == incoming.project.id
    assert window._history_key == (incoming.project.id, selected.id)
    assert window._selected_run in window.runs and window._history_entries
    assert all(
        entry.project_id == incoming.project.id and entry.channel_id == selected.id for entry in window._history_entries
    )
    assert window.viewport.view_state() == incoming.view
    assert window.project_path == path and window._prepared_project_history is None


def test_newer_source_integrity_failure_cannot_publish_older_positive_reuse(windows, tmp_path, monkeypatch):
    from test_application_project_bundle import _prepared_document

    project, _record, archive = _prepared_document(tmp_path)
    project = project.with_results((archive,), session={"selected_run_id": archive.id})
    path = project.save(tmp_path / "checkpoint-project.coilgen.json")
    window = windows()
    entered, release, done = Event(), Event(), Event()
    original_reuse, original_run = project_loading.execution_reuse, project_loading.ProjectOpenTask.run
    captured = []

    def held_reuse(*args, **kwargs):
        plan = original_reuse(*args, **kwargs)
        captured.append(plan)
        entered.set()
        assert release.wait(10), "Test did not release verified reuse preparation"
        return plan

    def run(task):
        try:
            return original_run(task)
        finally:
            done.set()

    monkeypatch.setattr(project_loading, "execution_reuse", held_reuse)
    monkeypatch.setattr(project_loading.ProjectOpenTask, "run", run)
    try:
        task = window._start_project_open(path)
        wait_until(entered.is_set)
        assert captured[0].available  # An actual prepared checkpoint passed verification.
        directory = captured[0].available[0].run_directory
        window._mark_run_verification_error(
            "newer-integrity-check", "Checkpoint changed after verification", directory=directory
        )
        assert window._project_open_task is task
        release.set()
        wait_until(lambda: window._project_open_task is None)
    finally:
        release.set()
        wait_until(done.is_set)
        QCoreApplication.processEvents()
    assert window.project.id == project.id and window.project_path == path
    assert not any(entry.run_directory == directory for entry in window._reuse_plan.available)
    assert "Checkpoint changed after verification" in window._source_health.known_error(directory)


@pytest.mark.parametrize(
    "mutation",
    [
        "revision",
        "native",
        "native_already_dirty",
        "incomplete_native",
        "expert",
        "expert_already_dirty",
        "hypothesis",
        "hypothesis_already_dirty",
        "rationale",
        "rationale_already_dirty",
        "channel",
        "alias",
        "new",
        "template",
        "close_project",
        "close_window",
    ],
)
def test_late_open_cannot_replace_changes_made_while_loading(windows, incoming, tmp_path, monkeypatch, mutation):
    project = ProjectDocument.from_config(preset_config(), name="Current owner")
    assembly = ProjectAssembly.new("Current channels", (project, project.duplicate("Other channel")))
    window = windows(assembly)
    previous_path = retain_path(window, tmp_path)
    errors = []
    monkeypatch.setattr(window, "_error", errors.append)
    editor = None
    if mutation.startswith("expert"):
        window._advanced_dialog()
        editor = window.expert_window.fields["sweep_min_bend_radius"].control
        if mutation.endswith("already_dirty"):
            editor.setText("4")
    elif mutation.startswith("rationale") or mutation == "alias":
        record = _owned_run(window.project, tmp_path / "prior-run")
        window.load_run(record.directory)
        entry = next(entry for entry in window._history_entries if entry.directory == record.directory)
        assert window.history_panel.select_record(entry.id)
        if mutation.startswith("rationale"):
            editor = window.history_panel.rationale
            assert editor.isEnabled()
    elif mutation.startswith("hypothesis"):
        editor = window.history_panel.hypothesis
        assert editor.isEnabled()
    if mutation in {"hypothesis_already_dirty", "rationale_already_dirty"}:
        editor.setPlainText("Prior unapplied note")
    if mutation == "native_already_dirty":
        radius = window._draft_controls["radius_mm"]
        radius.setValue(radius.value() + 5)
        assert window._draft_dirty
    with held_document(monkeypatch, incoming.path) as held:
        task = window._start_project_open(incoming.path)
        wait_until(held.entered.is_set)
        if mutation == "revision":
            assert window.apply_config({"target_gradient_strength": 0.031})
        elif mutation.startswith("native"):
            radius = window._draft_controls["radius_mm"]
            radius.setValue(radius.value() + 7)
            assert window._draft_dirty
        elif mutation == "incomplete_native":
            editor = window._draft_controls["radius_mm"].lineEdit()
            editor.selectAll()
            QTest.keyClick(editor, Qt.Key.Key_Backspace)
            assert window._draft_dirty
        elif mutation.startswith("expert"):
            editor.setText("5")
            assert window.expert_window.pending_changes
        elif mutation.startswith(("hypothesis", "rationale")):
            editor.setPlainText("New unapplied note")
        elif mutation == "channel":
            window.channel_selector.setCurrentIndex(1)
            assert window.project.id != project.id
        elif mutation == "alias":
            window._rename_history(entry.id, "Renamed while opening")
            assert next(row for row in window._history_entries if row.id == entry.id).alias == "Renamed while opening"
        elif mutation == "new":
            assert window.new_project(name="Replacement blank") is not None
        elif mutation == "template":
            window.load_template("cylinder")
            assert window.project.id != project.id
        elif mutation == "close_project":
            assert window.close_project()
        else:
            assert window.close() and window._closing
        current, current_assembly = window.project, window.assembly
        path, count = window.project_path, window.undo_stack.count()
        text = editor.text() if hasattr(editor, "text") else editor.toPlainText() if editor else None
        assert window._project_open_task is None and task not in window._foreground_work
        before_message = window.statusBar().currentMessage()
        window._project_open_progress(task.token, "Obsolete worker progress")
        window._project_open_ready(task.token, None, "Obsolete worker error")
        assert window.statusBar().currentMessage() == before_message and errors == []
    assert window.project is current and window.assembly is current_assembly
    assert window.project_path == path and window.undo_stack.count() == count
    if mutation not in {"new", "template", "close_project"}:
        assert window.project_path == previous_path
    if editor is not None:
        assert (editor.text() if hasattr(editor, "text") else editor.toPlainText()) == text
    assert errors == [] and window._project_open_task is None


@pytest.mark.parametrize("replacement", ["new_async_open", "synchronous_open", "cancel_button"])
def test_new_open_or_cancel_supersedes_an_older_worker(windows, incoming, tmp_path, monkeypatch, replacement):
    window = windows()
    previous = window.project
    previous_path = retain_path(window, tmp_path)
    latest = previous.duplicate("Newest requested project")
    latest_path = latest.save(tmp_path / "latest.coilgen.json")
    errors = []
    monkeypatch.setattr(window, "_error", errors.append)
    with held_document(monkeypatch, incoming.path) as held:
        old = window._start_project_open(incoming.path)
        wait_until(held.entered.is_set)
        if replacement == "new_async_open":
            new = window._start_project_open(latest_path)
            assert new.token != old.token
            window._project_open_progress(old.token, "Stale progress")
            assert window._project_open_task is new and "Stale progress" not in window.statusBar().currentMessage()
            wait_until(lambda: window._project_open_task is None)
        elif replacement == "synchronous_open":
            loaded = window.open_project(latest_path)
            assert loaded.id == latest.id  # Public Python API still returns its document synchronously.
        else:
            window._project_open_cancel.click()
            assert window.project is previous and window.project_path == previous_path
        current, path = window.project, window.project_path
        assert window._project_open_task is None
    assert window.project is current and window.project_path == path and errors == []
    assert window.project.id == (previous.id if replacement == "cancel_button" else latest.id)
    assert window.computation_indicator.isHidden() and window._project_open_cancel.isHidden()


@pytest.mark.parametrize("failure", ["malformed_json", "tampered_archive"])
def test_failed_async_open_preserves_current_draft_path_and_undo(windows, incoming, tmp_path, monkeypatch, failure):
    window = windows()
    previous_path = retain_path(window, tmp_path)
    assert window.apply_config({"target_gradient_strength": 0.031})
    radius = window._draft_controls["radius_mm"]
    radius.setValue(radius.value() + 9)
    previous, assembly, count = window.project, window.assembly, window.undo_stack.count()
    text = radius.lineEdit().text()
    path = tmp_path / "rejected.coilgen.json"
    if failure == "malformed_json":
        path.write_text("Invalid project JSON", encoding="utf-8")
    else:
        payload = json.loads(incoming.path.read_text(encoding="utf-8"))
        artifact = payload["results"][0]["files"]["result.json"]
        artifact["data"] = "e30="  # Valid base64 for {}, retaining the original declared digest.
        path.write_text(json.dumps(payload), encoding="utf-8")
    errors = []
    monkeypatch.setattr(window, "_error", errors.append)
    window._start_project_open(path)
    wait_until(lambda: window._project_open_task is None)
    assert window.project is previous and window.assembly is assembly
    assert window.project_path == previous_path and window.undo_stack.count() == count
    assert window._draft_dirty and radius.lineEdit().text() == text
    assert len(errors) == 1 and errors[0].startswith("Cannot open project:")
    assert window.computation_indicator.isHidden()


@pytest.mark.parametrize("failure", ["materialize", "cache_link", "cache_restore"])
def test_async_open_retains_diagnostics_and_independently_verified_embedded_fallback(
    windows, incoming, monkeypatch, failure
):
    window = windows()

    def unavailable(*_args, **_kwargs):
        raise OSError("Fixture evidence/cache inaccessible")

    if failure == "materialize":
        monkeypatch.setattr(RunArchive, "materialize", unavailable)
    elif failure == "cache_link":
        monkeypatch.setattr(window.history, "add_run", unavailable)
    else:
        monkeypatch.setattr(window.history, "restore", unavailable)
    window._start_project_open(incoming.path)
    wait_until(lambda: window._project_open_task is None)
    assert window.project.id == incoming.project.id and window.project_path == incoming.path
    assert window.messages.toPlainText().count("Fixture evidence/cache inaccessible") == 1
    if failure == "materialize":
        assert window._selected_run is None and not window._history_entries[0].verified
        assert "recorded run could not be restored" in window.messages.toPlainText()
        assert window.viewport.view_state().get("source") != "run"
    else:
        assert window._selected_run in window.runs and window._history_entries[0].verified
        assert window.viewport.view_state() == incoming.view
        selected = window.runs[window._selected_run]
        for name, values in incoming.record.arrays.items():
            np.testing.assert_array_equal(selected.arrays[name], values)
