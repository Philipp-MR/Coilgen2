"""Slow evidence verification cannot freeze job queries or cancellation."""

import queue
import threading
import time

import pytest

from coilgen.progress import ProgressReporter, progress_context, progress_scope
from coilgen.research.jobs import JobManager, _record_progress


class _Events(queue.Queue):
    def close(self):
        pass


class _Process:
    alive = True
    exitcode = 0

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    kill = terminate

    def join(self, timeout=None):
        pass


def _active_job(jobs, *, completed=False):
    process, events, cancel = _Process(), _Events(), threading.Event()
    id_ = "verification-test"
    (jobs.directory / id_).mkdir()
    with jobs._lock:
        jobs._states[id_] = {
            "id": id_,
            "name": "Shim",
            "status": "running",
            "stage": "Field evaluation",
            "progress": 0.99,
            "log": [],
            "pipeline_stage": "evaluated",
            "error": None,
        }
        jobs._active[id_] = (process, events, cancel, time.monotonic())
    if completed:
        events.put({"kind": "completed"})
    return id_, process


@pytest.mark.parametrize("cancel_during", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_verification_keeps_queries_and_cancellation_responsive(tmp_path, monkeypatch, cancel_during, valid):
    entered, release, queried = threading.Event(), threading.Event(), threading.Event()

    def verify(*_args):
        entered.set()
        assert release.wait(5), "Test did not release verification"
        return {} if valid else None

    with JobManager(tmp_path) as jobs:
        monkeypatch.setattr(jobs, "_verified_completion", verify)
        id_, process = _active_job(jobs, completed=True)
        assert entered.wait(2)
        snapshots = []

        def query():
            snapshots.append(jobs.get(id_))
            snapshots.append(jobs.list()[0])
            if cancel_during:
                snapshots.append(jobs.cancel(id_))
            queried.set()

        thread = threading.Thread(target=query)
        thread.start()
        try:
            assert queried.wait(0.5), "GUI query/cancel blocked behind evidence hashing"
            assert snapshots[0]["stage"] == "Verifying saved evidence"
            assert snapshots[0]["progress_indeterminate"]
        finally:
            release.set()
            thread.join(2)
        deadline = time.monotonic() + 2
        while jobs.get(id_)["status"] in {"running", "cancelling"} and time.monotonic() < deadline:
            time.sleep(0.01)
        state = jobs.get(id_)
        expected = "cancelled" if cancel_during else "completed" if valid else "failed"
        assert state["status"] == expected
        if not valid and not cancel_during:
            assert "integrity" in state["error"]
        process.alive = False


def test_process_observation_never_fabricates_numerical_progress(tmp_path):
    with JobManager(tmp_path) as jobs:
        id_, process = _active_job(jobs)
        deadline = time.monotonic() + 2
        while not jobs.get(id_).get("worker_observed_at") and time.monotonic() < deadline:
            time.sleep(0.01)
        first = jobs.get(id_)
        assert first["worker_alive"] is True
        assert "last_progress_at" not in first
        assert first["progress"] == 0.99
        observed = first["worker_observed_at"]
        while jobs.get(id_)["worker_observed_at"] == observed and time.monotonic() < deadline:
            time.sleep(0.01)
        second = jobs.get(id_)
        assert second["worker_observed_at"] != observed
        assert "last_progress_at" not in second
        assert second["progress"] == first["progress"]
        process.alive = False


@pytest.mark.parametrize("status", ("completed", "failed", "cancelled"))
def test_shutdown_preserves_terminal_outcome_with_worker_still_alive(tmp_path, status):
    jobs = JobManager(tmp_path)
    id_, _process = _active_job(jobs)
    with jobs._lock:
        jobs._states[id_].update(status=status, stage="Terminal evidence", error="retained diagnostic")
    jobs.close()
    state = jobs.get(id_)
    assert state["status"] == status
    assert state["stage"] == "Terminal evidence"
    assert state["error"] == "retained diagnostic"
    assert state["worker_alive"] is False


def test_shutdown_during_verification_cannot_publish_late_completion(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def verify(*_args):
        entered.set()
        assert release.wait(8)
        return {}

    jobs = JobManager(tmp_path)
    monkeypatch.setattr(jobs, "_verified_completion", verify)
    id_, _process = _active_job(jobs, completed=True)
    assert entered.wait(2)
    try:
        jobs.close()
        assert jobs.get(id_)["status"] == "interrupted"
    finally:
        release.set()
        jobs._thread.join(2)
    assert not jobs._thread.is_alive()
    assert jobs.get(id_)["status"] == "interrupted"


def test_named_operation_survives_inner_substeps_and_times_separately(monkeypatch):
    events = []
    with progress_context(events.append), progress_scope(0, 100, "Pipeline", overall=True):
        with progress_scope(0, 80, "Realize conductors", operation="Realize conductors"):
            with progress_scope(0, 50, "Prepare mounted centerline"):
                ProgressReporter(label="Sweep").report(None, "Check curved intervals")
    inner = next(event for event in events if event["message"] == "Check curved intervals")
    assert inner["operation"] == "Realize conductors"
    state = {"progress": 0}
    ticks = iter(["start", "later", "new"])
    monkeypatch.setattr("coilgen.research.jobs._now", lambda: next(ticks))
    _record_progress(state, inner)
    assert state["operation_started_at"] == "start"
    _record_progress(state, {**inner, "message": "Check next interval"})
    assert state["last_progress_at"] == "later"
    assert state["operation_started_at"] == "start"
    _record_progress(state, {**inner, "operation": "Build swept bodies"})
    assert state["operation_started_at"] == "new"
