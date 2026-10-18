"""Finished evidence loading stays responsive and respects current view ownership."""

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox

import test_workbench_usability as usability
from coilgen.application import ProjectAssembly, ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.application.service import RunRecord
from coilgen.application.history import HistoryEntry
from coilgen.application.reuse import ReusePlan
from coilgen.gui.workbench import CoilGenWorkbench
from coilgen.gui.viewport import ScientificViewport
from coilgen.gui.run_loading import CompletedRunLoadTask, StoppedReuseTask


application = usability.application


def wait(predicate):
    deadline = time.monotonic() + 10
    while not predicate() and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)  # Yield the GIL to the Python archive/hash worker.
    assert predicate()


@pytest.fixture
def window(application, tmp_path):
    project = ProjectDocument.from_config(preset_config())
    instance = CoilGenWorkbench(
        project,
        jobs=usability.Queue(tmp_path / "jobs"),
        service=usability.Service(),
        viewport_factory=usability.Preview,
    )
    instance.timer.stop()
    instance.history = None
    yield instance
    QThreadPool.globalInstance().waitForDone(3000)
    application.processEvents()
    instance.live_preview.close()
    instance._draft_dirty = False
    instance.undo_stack.setClean()
    instance.close()
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def completed(window):
    state = window.compute()
    window.jobs.states[0].update(status="completed", stage="Complete", progress=1)
    return state


def record(window, state, **changes):
    return RunRecord(
        window.jobs_directory / state["id"],
        {**state, "pipeline_stage": "evaluated", "summary": {"accepted": False}, "artifacts": [], **changes},
        {},
    )


def blocker(window, captured):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def load(_directory):
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(10), "Test did not release archive loading."
        return captured

    window.service.load_run = load
    return entered, release, calls


def test_completion_verification_does_not_block_qt_or_publish_premature_outcome(window):
    state = completed(window)
    captured = record(window, state)
    entered, release, calls = blocker(window, captured)
    ticks = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    try:
        window.poll_jobs()
        wait(entered.is_set)
        for _ in range(3):
            window.poll_jobs()
            QTest.qWait(25)
        assert len(ticks) >= 3
        assert len(calls) == 1 and calls[0] != threading.get_ident()
        assert window._loading_job_id == state["id"]
        assert not window.runs
        assert not window.job_outcomes.outcomes
        assert "verifying saved evidence" in window.execution_label.text().lower()
    finally:
        release.set()
        timer.stop()
    wait(lambda: state["id"] in window.runs or window._run_verification_error(state["id"]))
    assert not window._run_verification_error(state["id"])
    assert window._loading_job_id is None
    assert window._selected_run == state["id"]
    assert window.runs[state["id"]].metadata["summary"]["accepted"] is False


@pytest.mark.parametrize("replacement", ("revision", "project", "close"))
def test_delayed_result_does_not_select_replaced_or_edited_model(window, replacement):
    state = completed(window)
    entered, release, _calls = blocker(window, record(window, state))
    window.poll_jobs()
    wait(entered.is_set)
    if replacement == "revision":
        window._set_project(window.project.update_config({"target_gradient_strength": 0.031}))
    elif replacement == "project":
        window._set_assembly(ProjectAssembly.new("Replacement", (ProjectDocument.from_config(preset_config()),)))
    else:
        window._clear_project()
    release.set()
    wait(lambda: not window._run_load_tasks)
    assert window._selected_run is None
    assert window.viewport.displayed is None
    assert (state["id"] in window.runs) == (replacement == "revision")


def test_delayed_result_respects_live_follow_pause(window):
    state = completed(window)
    entered, release, _calls = blocker(window, record(window, state))
    window.poll_jobs()
    wait(entered.is_set)
    window.live_preview.pause()
    release.set()
    wait(lambda: state["id"] in window.runs)
    assert window._selected_run is None
    assert window.viewport.displayed is None


@pytest.mark.parametrize(
    "key,value",
    (
        ("request_sha256", "changed"),
        ("project_id", "foreign"),
        ("project_revision", 29),
        ("project_fingerprint", "stale"),
    ),
)
def test_completed_archive_cannot_claim_another_submitted_job(window, key, value):
    state = completed(window)
    window.service.load_run = lambda _directory: record(window, state, **{key: value})
    if key == "request_sha256":
        window.jobs.states[0][key] = "submitted"
    window.poll_jobs()
    wait(lambda: not window._run_load_tasks)
    assert not window.runs
    assert key in window._run_verification_error(state["id"])
    assert "verification failed" in window.job_outcomes.selected.title.lower()


def test_loader_failure_is_terminal_and_retains_unverified_evidence_status(window):
    state = completed(window)

    def load(_directory):
        raise ValueError("Saved artifact checksum changed")

    window.service.load_run = load
    window.poll_jobs()
    wait(lambda: not window._run_load_tasks)
    assert not window.runs
    assert state["id"] in window._loaded_ids
    assert "checksum changed" in window._run_verification_error(state["id"])
    assert window._loading_job_id is None


def test_newer_explicit_integrity_failure_cannot_be_cleared_by_older_positive_load(window):
    state = completed(window)
    captured = record(window, state)
    entered, release, _calls = blocker(window, captured)
    window.poll_jobs()
    wait(entered.is_set)
    window._mark_run_verification_error(state["id"], "Newer explicit checksum failure", directory=captured.directory)
    release.set()
    wait(lambda: not window._run_load_tasks)
    assert not window.runs
    assert window._selected_run is None
    assert "Newer explicit checksum failure" in window._run_verification_error(state["id"])


def test_newer_reuse_refresh_is_not_overwritten_by_captured_completion_state(window, monkeypatch):
    state = completed(window)
    entered, release, _calls = blocker(window, record(window, state))
    window.poll_jobs()
    wait(entered.is_set)
    newer = ReusePlan((), (), ("Newly stopped second job evidence",))
    monkeypatch.setattr("coilgen.gui.workbench.execution_reuse", lambda *args, **kwargs: newer)
    window._refresh_stage_status()
    window.live_preview.pause()
    release.set()
    wait(lambda: not window._run_load_tasks)
    assert state["id"] in window.runs
    assert window._reuse_plan is newer


def test_alias_renamed_after_worker_snapshot_keeps_latest_display(window):
    state = completed(window)
    captured = record(window, state)
    restored, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()

    class History:
        entry = HistoryEntry(
            "existing-link", window.project.id, None, "Original alias", captured.directory, "", captured
        )

        def add_run(self, *_args, **_kwargs):
            return self.entry

        def restore(self, *_args, **_kwargs):
            snapshot = (self.entry,)
            if threading.get_ident() != main_thread:
                restored.set()
                assert release.wait(10)
            return snapshot

        def rename_run(self, _id, alias):
            self.entry = replace(self.entry, alias=alias)

        def save_session(self, *_args, **_kwargs):
            pass

    window.history = History()
    window.service.load_run = lambda _directory: captured
    window.poll_jobs()
    wait(restored.is_set)
    window._rename_history("existing-link", "Edited while loading")
    assert window._history_entries[0].alias == "Edited while loading"
    release.set()
    wait(lambda: state["id"] in window.runs)
    assert window._history_entries[0].alias == "Edited while loading"


def test_completed_record_preserves_production_viewport_detached_metadata_contract(window):
    state = completed(window)
    captured = record(window, state, artifacts=[{"name": "field.csv", "units": ["m", "T"]}])
    dataset = QComboBox()
    dataset.addItem("Saved target", "target")
    adapter = SimpleNamespace(
        dataset=dataset,
        _clear_inspection_task=lambda: None,
        _rebuild_capabilities=lambda: None,
        _preferred_run_dataset=lambda: "target",
        _render=lambda: None,
    )
    window.viewport.display_run = lambda arrays, metadata=None: ScientificViewport.display_run(
        adapter, arrays, metadata
    )
    window.service.load_run = lambda _directory: captured
    window.poll_jobs()
    wait(lambda: not window._run_load_tasks)
    assert not window._run_verification_error(state["id"])
    assert adapter._metadata["artifacts"][0]["units"] == ["m", "T"]
    adapter._metadata["artifacts"][0]["units"].append("display-only")
    assert captured.metadata["artifacts"][0]["units"] == ["m", "T"]


def test_stopped_checkpoint_verification_keeps_qt_responsive_and_resume_disabled(window, monkeypatch):
    window.compute()
    window.jobs.states[0].update(status="cancelled", stage="Stopped", completed_stages=["prepared"])
    entered, release = threading.Event(), threading.Event()
    calls, ticks = [], []
    verified = ReusePlan((), (), ("Captured checkpoint verified",))

    def verify(*_args, **_kwargs):
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(10)
        return verified

    monkeypatch.setattr("coilgen.application.execution.execution_reuse", verify)
    monkeypatch.setattr("coilgen.application.reuse.plan_stopped_reuse", lambda *_args, **_kwargs: verified)
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    try:
        window.poll_jobs()
        wait(entered.is_set)
        for _ in range(3):
            window.poll_jobs()
            QTest.qWait(25)
        assert len(ticks) >= 3
        assert len(calls) == 1 and calls[0] != threading.get_ident()
        assert not window.resume_action.isEnabled()
        assert "Verifying stopped checkpoint" in window.resume_action.toolTip()
        assert "Verifying stopped checkpoints" in window.computation_indicator.details_label.text()
    finally:
        timer.stop()
        release.set()
    wait(lambda: not window._stopped_reuse_tasks)
    assert window._reuse_plan is verified
    assert window._selected_resume_plan() is verified


@pytest.mark.parametrize("change", ("revision", "integrity"))
def test_stopped_checkpoint_snapshot_cannot_publish_after_newer_project_or_integrity_change(
    window, monkeypatch, change
):
    state = window.compute()
    window.jobs.states[0].update(status="cancelled", stage="Stopped", completed_stages=["prepared"])
    entered, release = threading.Event(), threading.Event()
    old = ReusePlan((), (), ("Old positive checkpoint snapshot",))

    def verify(*_args, **_kwargs):
        entered.set()
        assert release.wait(10)
        return old

    monkeypatch.setattr("coilgen.application.execution.execution_reuse", verify)
    monkeypatch.setattr("coilgen.application.reuse.plan_stopped_reuse", lambda *_args, **_kwargs: old)
    window.poll_jobs()
    wait(entered.is_set)
    if change == "revision":
        window._set_project(window.project.update_config({"target_gradient_strength": 0.031}))
    else:
        window._mark_run_verification_error(state["id"], "Newer checkpoint checksum failure")
    # Isolate publication of this capture from a subsequent current-state request.
    monkeypatch.setattr(window, "_start_stopped_reuse_load", lambda *_args, **_kwargs: None)
    release.set()
    wait(lambda: not window._stopped_reuse_tasks)
    assert window._reuse_plan is not old


@pytest.mark.parametrize("change", ("revision", "selected_job"))
def test_pending_stopped_capability_retries_current_source_after_older_task_finishes(window, monkeypatch, change):
    state = window.compute()
    window.jobs.states[0].update(status="cancelled", stage="Stopped", completed_stages=["prepared"])
    entered, release = threading.Event(), threading.Event()
    captures = []

    def verify(project, *_args, **_kwargs):
        captures.append(project.fingerprint)
        entered.set()
        assert release.wait(10)
        return ReusePlan((), (), (project.fingerprint,))

    monkeypatch.setattr("coilgen.application.execution.execution_reuse", verify)
    monkeypatch.setattr(
        "coilgen.application.reuse.plan_stopped_reuse",
        lambda _project, states, *_args, **_kwargs: ReusePlan((), (), (states[0]["id"],)),
    )
    window.poll_jobs()
    wait(entered.is_set)
    if change == "revision":
        window._set_project(window.project.update_config({"target_gradient_strength": 0.031}))
        expected_source = state["id"]
    else:
        second = {**window.jobs.states[0], "id": "b" * 32}
        window.jobs.states.append(second)
        window.poll_jobs()
        window._focus_job(second["id"], open_window=False)
        expected_source = second["id"]
    window._selected_resume_plan()
    assert "Verifying stopped checkpoint" in window._selected_resume_plan().diagnostics[0]
    release.set()
    wait(lambda: not window._stopped_reuse_tasks)
    assert len(captures) == 2
    assert window._selected_resume_plan().diagnostics == (expected_source,)


@pytest.mark.parametrize("kind", ("completed", "stopped"))
def test_background_evidence_task_finishing_after_qt_teardown_is_safe(window, monkeypatch, kind):
    import shiboken6

    state = completed(window)
    entered, release = threading.Event(), threading.Event()
    errors = []

    def blocked(*_args, **_kwargs):
        entered.set()
        assert release.wait(10)
        return record(window, state) if kind == "completed" else ReusePlan((), (), ())

    if kind == "completed":
        task = CompletedRunLoadTask(
            state, window.jobs_directory / state["id"], blocked, assembly_id=window.assembly.id, scopes={}
        )
    else:
        monkeypatch.setattr("coilgen.application.execution.execution_reuse", blocked)
        monkeypatch.setattr(
            "coilgen.application.reuse.plan_stopped_reuse", lambda *_args, **_kwargs: ReusePlan((), (), ())
        )
        task = StoppedReuseTask(window.project, (), (state,), window.jobs_directory, 1, state, ())

    def run():
        try:
            task.run()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    wait(entered.is_set)
    window.undo_stack.setClean()
    window.close()
    shiboken6.delete(task.signals)
    release.set()
    worker.join(10)
    assert not worker.is_alive()
    assert not errors
    assert not window.runs


def test_passive_source_health_borrows_captured_canonical_paths_without_filesystem_rechecks(window, monkeypatch):
    from pathlib import Path

    state = window.compute()
    captured = record(window, state)
    # Capture the runtime identity once; immutable records already captured theirs.
    assert window._run_verification_directory(state["id"]) == captured.directory
    window.runs["alias"] = captured

    def unexpected_resolve(*_args, **_kwargs):
        raise AssertionError("Passive source health resolved the same directory again")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", unexpected_resolve)
        for _ in range(3):
            assert window._run_verification_error(state["id"]) == ""
            assert window._run_verification_error("alias") == ""
            assert window._run_verification_error(captured) == ""
    window._mark_run_verification_error(state["id"], "Actual explicit checksum failure")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", unexpected_resolve)
        for _ in range(3):
            assert window._run_verification_error(state["id"]) == "Actual explicit checksum failure"
            assert window._run_verification_error("alias") == "Actual explicit checksum failure"
            assert window._run_verification_error(captured) == "Actual explicit checksum failure"
