"""Current global work remains visible across channels, windows and file loads."""

from threading import Event
import time

import pytest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QLineEdit, QVBoxLayout

from coilgen.application.stage_preview import MANIFEST_NAME
from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.gui.design_progress import ComputationIndicator

from test_workbench_usability import (
    application as application,
    prepared_run as prepared_run,
    windows as windows,
)


def _publish(window, *states):
    window.jobs.states = list(states)
    window.poll_jobs()


def _state(window, **changes):
    return {
        "id": "gx",
        "name": "Gx",
        "project_id": window.project.id,
        "project_fingerprint": window.project.fingerprint,
        "project_revision": window.project.revision,
        "pipeline_stage": "evaluated",
        "status": "running",
        "progress": 0.42,
        "stage": "Integrating wire fields",
        **changes,
    }


def _rows(indicator):
    return "\n".join(action.text() for action in indicator.tasks_menu.actions())


def _footer(dialog):
    return dialog.findChild(ComputationIndicator, "window_computation_indicator")


def _wait(application, predicate, *, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    pytest.fail("Background preview did not finish")


def test_closed_plan_and_historical_selection_keep_old_and_foreign_job_progress_visible(windows, monkeypatch):
    window = windows()
    old = _state(window, project_revision=window.project.revision - 1, name="Earlier revision", progress=0.37)
    foreign = _state(window, id="gy", name="Other project", project_id="foreign", progress=0.81, stage="Routing")
    window.design_plan_dock.hide()
    window.states["history"] = {"id": "history", "status": "completed", "progress": 1}
    monkeypatch.setattr(window, "_selected_job_id", lambda: "history")
    _publish(window, old, foreign)
    indicator = window.computation_indicator
    assert not indicator.isHidden() and window.design_plan_dock.isHidden()
    assert indicator.label.text() == "Computing (2 jobs)"
    assert "Earlier revision 37%" in indicator.details_label.text()
    assert "Other project 81%" in indicator.details_label.text()
    rows = _rows(indicator)
    assert "Integrating wire fields" in rows and "Routing" in rows
    assert "37%" in rows and "81%" in rows
    assert indicator.progress.maximum() == 0


def test_same_channel_name_jobs_retain_revision_and_identity_in_live_task_list(windows, application, monkeypatch):
    from coilgen.application import reuse

    entered, release = Event(), Event()
    original = reuse.plan_stopped_reuse

    def verify_stopped(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(reuse, "plan_stopped_reuse", verify_stopped)
    window = windows()
    _publish(
        window,
        _state(window, id="earlier-job", project_revision=1, progress=0.37),
        _state(window, id="current-job", project_revision=2, progress=0.81),
    )
    actions = window.computation_indicator.tasks_menu.actions()
    assert "revision 1 · job earlier-job" in actions[0].text()
    assert "revision 2 · job current-job" in actions[1].text()
    _publish(window, _state(window, id="current-job", project_revision=2, progress=0.85))
    assert len(window.computation_indicator.tasks_menu.actions()) == 1
    assert "85%" in window.computation_indicator.tasks_menu.actions()[0].text()
    try:
        _publish(window, _state(window, status="cancelled", progress=0.85))
        assert entered.wait(3)
        assert "Verifying stopped checkpoints" in _rows(window.computation_indicator)
        assert not window.computation_indicator.isHidden()
    finally:
        release.set()
        _wait(application, lambda: not window._stopped_reuse_tasks, timeout=15)
    assert not window.computation_indicator.tasks_menu.actions()
    assert window.computation_indicator.isHidden()


def test_evidence_and_nested_modal_windows_mirror_runtime_when_plan_closed(windows, application):
    window = windows()
    window.show()
    _publish(window, _state(window))
    window.design_plan_dock.hide()
    window._open_evidence(window.messages)
    evidence = window.evidence.window_for(window.messages)
    nested = QDialog(evidence)
    nested.setModal(True)
    nested.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    layout = QVBoxLayout(nested)
    layout.addWidget(QLineEdit("retained editor draft", nested))
    nested.show()
    application.processEvents()
    for dialog in (evidence, nested):
        footer = _footer(dialog)
        assert footer is not None and footer.isVisible()
        assert "Gx 42%" in footer.details_label.text()
        assert "Integrating wire fields" in _rows(footer)
    _publish(window, _state(window, progress=0.64, stage="Checking support intersections"))
    assert "Gx 64%" in _footer(nested).details_label.text()
    assert "Checking support intersections" in _rows(_footer(evidence))
    assert window.design_plan_dock.isHidden()
    nested.close()
    window.evidence.close_all()


def test_concurrent_field_analysis_has_visible_individual_progress_and_operations(windows):
    window = windows()
    _publish(window, _state(window, progress=0.53))
    window.design_plan_dock.hide()
    window._research_busy("Field analysis", True)
    try:
        window._research_progress("Field analysis", {"percent": 19, "message": "Slice integration"})
        indicator = window.computation_indicator
        assert "Gx 53%" in indicator.details_label.text()
        assert "Field analysis 19%" in indicator.details_label.text()
        assert "Slice integration" in _rows(indicator)
        assert "Integrating wire fields" in _rows(indicator)
        assert indicator.progress.maximum() == 0
    finally:
        window._research_busy("Field analysis", False)
    assert window.computation_indicator.label.text() == "Computing (53%)"


def test_inactive_system_support_loading_is_reported_independently(windows):
    window = windows()
    window.viewport._system_worker = object()
    window._refresh_computation_indicator()
    indicator = window.computation_indicator
    assert not indicator.isHidden()
    assert "Preparing system support meshes" in indicator.details_label.text()
    assert "Preparing system support meshes" in _rows(indicator)
    assert indicator.progress.maximum() == 0
    window.viewport._system_worker = None
    window._refresh_computation_indicator()
    assert indicator.isHidden()


def test_invalidated_live_preview_keeps_physical_archive_work_visible(windows, application, monkeypatch):
    window = windows()
    controller = window.live_preview
    entered, release = Event(), Event()

    def loader(_path, **_identity):
        entered.set()
        assert release.wait(3)
        return None

    monkeypatch.setattr(controller, "_loader", loader)
    state = _state(window, id="archive-preview", request_sha256="b" * 64)
    directory = window.jobs.directory / state["id"]
    directory.mkdir(parents=True)
    (directory / MANIFEST_NAME).write_text("captured stage", encoding="utf-8")
    try:
        controller.start(state)
        assert entered.wait(3)
        assert controller.loading and controller.working
        assert "Loading recorded stage preview" in window.computation_indicator.details_label.text()
        controller.pause()
        assert not controller.loading and controller.working
        assert not window.computation_indicator.isHidden()
        assert "Loading recorded stage preview" in _rows(window.computation_indicator)
    finally:
        release.set()
        _wait(application, lambda: not controller.working)
    assert window.computation_indicator.isHidden()


def test_synchronous_nested_work_is_visible_before_execution_and_recovers_after_error(windows):
    window = windows()
    indicator = window.computation_indicator
    with window._visible_work("Verify saved results"):
        assert not indicator.isHidden()
        assert "Verify saved results" in indicator.details_label.text()
        assert len(window._foreground_work) == 1
        with pytest.raises(ValueError, match="test failure"):
            with window._visible_work("Load geometry"):
                assert len(window._foreground_work) == 1
                assert "Load geometry" in indicator.details_label.text()
                assert "Verify saved results" not in indicator.details_label.text()
                assert indicator.label.text() == "Working…"
                raise ValueError("test failure")
        assert "Verify saved results" in indicator.details_label.text()
    assert not window._foreground_work and indicator.isHidden()


def test_load_run_paints_verification_activity_before_reading_saved_arrays(windows, prepared_run, monkeypatch):
    project, record = prepared_run
    window = windows(project)
    original = window.service.load_run
    inspected = []

    def load(path):
        indicator = window.computation_indicator
        assert not indicator.isHidden()
        assert "Verifying saved result" in indicator.details_label.text()
        assert record.directory.name in indicator.details_label.text()
        assert indicator.progress.maximum() == 0
        inspected.append(path)
        return original(path)

    monkeypatch.setattr(window.service, "load_run", load)
    loaded = window.load_run(record.directory, select=False)
    assert loaded.directory == record.directory and inspected == [record.directory]
    assert not window._foreground_work and window.computation_indicator.isHidden()


def test_open_project_paints_activity_before_deserialization_and_clears_on_invalid_input(windows, monkeypatch):
    window = windows()
    project = window.project

    def load(path):
        indicator = window.computation_indicator
        assert not indicator.isHidden()
        assert "Opening project and verifying embedded results" in indicator.details_label.text()
        assert path == "broken-project.json"
        raise ValueError("invalid project")

    monkeypatch.setattr(window, "_open_project", load)
    with pytest.raises(ValueError, match="invalid project"):
        window.open_project("broken-project.json")
    assert window.project is project
    assert not window._foreground_work and window.computation_indicator.isHidden()


@pytest.mark.parametrize("status, expected", (("running", "Computing 42%"), ("queued", "Queued")))
def test_disabled_channel_retains_already_submitted_job_status(windows, status, expected):
    window = windows()
    state = _state(window, status=status)
    _publish(window, state)
    window._channel_enabled_changed(False)
    window.poll_jobs()
    channel = window.assembly.channel(window._active_channel_id)
    assert not channel.enabled
    item = window.design_progress.channel_items[channel.id]
    assert item.text(1) == expected
    assert "next system compute" in item.toolTip(1)
    assert not window.computation_indicator.isHidden()


@pytest.mark.parametrize("failure", (False, True))
def test_multichannel_submission_has_visible_feedback_and_clears_guard_on_failure(windows, monkeypatch, failure):
    window = windows()
    window.load_template("cylindrical_xyz_gradient")
    seen, errors = [], []

    def submit(assembly, jobs):
        indicator = window.computation_indicator
        assert window._submitting and not indicator.isHidden()
        assert "Submitting study" in indicator.details_label.text()
        # Reentrant submission must not create a duplicate system run.
        assert window.compute_channels() is None
        seen.append(assembly.id)
        if failure:
            raise OSError("submission unavailable")
        states = {}
        for channel in assembly.channels:
            state = _state(
                window,
                id=channel.id,
                name=channel.name,
                project_id=channel.project.id,
                project_fingerprint=channel.project.fingerprint,
                project_revision=channel.project.revision,
                status="queued",
                progress=0,
            )
            states[channel.id] = state
            jobs.states.append(state)
        return states

    monkeypatch.setattr(window.service, "submit_assembly", submit, raising=False)
    monkeypatch.setattr(window, "_error", errors.append)
    result = window.compute_channels()
    assert len(seen) == 1 and not window._submitting
    if failure:
        assert result is None and errors == ["Cannot submit channels: submission unavailable"]
        assert window.computation_indicator.isHidden()
    else:
        assert len(result) == 3 and not errors
        assert window.computation_indicator.label.text() == "Queued (3 jobs)"
        assert "Submitting study" not in window.computation_indicator.details_label.text()


_DOCUMENT_WRITES = (
    ("save_project", "Saving project and results", ProjectDocument, "save", "project.json"),
    ("export_python", "Exporting project and Python inputs", ProjectDocument, "export_python", "design.py"),
    (
        "export_assembly_python",
        "Exporting channel system and Python inputs",
        ProjectAssembly,
        "save",
        "channels.py",
    ),
)


@pytest.mark.parametrize("method, label, writer_class, writer_method, filename", _DOCUMENT_WRITES)
def test_document_saves_and_exports_show_activity_before_packaging_and_disk_write(
    windows, monkeypatch, tmp_path, method, label, writer_class, writer_method, filename
):
    window = windows()
    original_pack = window._portable_assembly
    original_write = getattr(writer_class, writer_method)
    inspected = []

    def visible(stage):
        indicator = window.computation_indicator
        assert not indicator.isHidden()
        assert label in indicator.details_label.text()
        assert indicator.progress.maximum() == 0
        inspected.append(stage)

    def package():
        visible("package")
        return original_pack()

    def write(document, path, **options):
        visible("write")
        return original_write(document, path, **options)

    monkeypatch.setattr(window, "_portable_assembly", package)
    monkeypatch.setattr(writer_class, writer_method, write)
    path = tmp_path / filename
    options = {} if method == "save_project" else {"include_results": True}
    result = getattr(window, method)(path, **options)
    assert result == path and path.is_file()
    assert inspected == ["package", "write"]
    assert not window._foreground_work and window.computation_indicator.isHidden()


@pytest.mark.parametrize("method, label, writer_class, writer_method, filename", _DOCUMENT_WRITES)
@pytest.mark.parametrize("failure_stage", ("package", "write"))
def test_document_save_export_failure_clears_activity_and_preserves_original_error(
    windows, monkeypatch, tmp_path, method, label, writer_class, writer_method, filename, failure_stage
):
    window = windows()
    project = window.project
    original_pack = window._portable_assembly

    def fail(*_args, **_options):
        indicator = window.computation_indicator
        assert not indicator.isHidden() and label in indicator.details_label.text()
        raise OSError("test storage failure")

    if failure_stage == "package":
        monkeypatch.setattr(window, "_portable_assembly", fail)
    else:
        monkeypatch.setattr(window, "_portable_assembly", original_pack)
        monkeypatch.setattr(writer_class, writer_method, fail)
    with pytest.raises(OSError, match="^test storage failure$"):
        options = {} if method == "save_project" else {"include_results": True}
        getattr(window, method)(tmp_path / filename, **options)
    assert window.project is project
    assert not window._foreground_work and window.computation_indicator.isHidden()
