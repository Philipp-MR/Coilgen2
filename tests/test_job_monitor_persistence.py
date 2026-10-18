"""A blocked state file cannot kill queue draining or another numerical job."""

import json
import threading
import time

import pytest

from coilgen.research import jobs as runtime
from coilgen.research.jobs import JobManager, RuntimePolicy
from test_runtime_responsiveness import _Events, _Process


def _wait(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("Job monitor did not finish the expected operation")


def _active(jobs, id_):
    process, events, cancel = _Process(), _Events(maxsize=256), threading.Event()
    directory = jobs.directory / id_
    directory.mkdir()
    state = {
        "id": id_,
        "name": id_,
        "status": "running",
        "stage": "Initial operation",
        "progress": 0,
        "log": [],
        "pipeline_stage": "evaluated",
        "error": None,
    }
    runtime.atomic_json(directory / "job.json", state)
    jobs._states[id_] = state
    jobs._active[id_] = (process, events, cancel, time.monotonic())
    return process, events


def _progress(index):
    return {
        "kind": "progress",
        "label": "CoilGen",
        "percent": 100 * index / 320,
        "message": f"Numerical update {index}",
    }


def test_blocked_state_write_drains_full_queue_and_retries_latest_reaped_terminal_state(tmp_path, monkeypatch):
    released, producer_done, write_failed = threading.Event(), threading.Event(), threading.Event()
    original, producer_errors = runtime.atomic_json, []
    with JobManager(tmp_path, RuntimePolicy(max_workers=2)) as jobs:
        with jobs._lock:
            blocked, events = _active(jobs, "blocked")
            other, other_events = _active(jobs, "other")
            blocked_path = tmp_path / "blocked" / "job.json"
            previous = blocked_path.read_bytes()

            def write(path, state):
                if path == blocked_path and not released.is_set():
                    write_failed.set()
                    raise OSError("Fixture state file inaccessible")
                return original(path, state)

            monkeypatch.setattr(runtime, "atomic_json", write)
            monkeypatch.setattr(jobs, "_verified_completion", lambda _id: {})
            # Fill the actual bounded monitor queue before releasing its lock.
            for index in range(256):
                events.put_nowait(_progress(index))

        def producer():
            try:
                for index in range(256, 320):
                    events.put(_progress(index), timeout=2)
                events.put({"kind": "completed"}, timeout=2)
                blocked.alive = False
            except Exception as exc:
                producer_errors.append(exc)
            finally:
                producer_done.set()

        thread = threading.Thread(target=producer)
        thread.start()
        try:
            other_events.put(_progress(319), timeout=2)
            other_events.put({"kind": "completed"}, timeout=2)
            other.alive = False
            assert producer_done.wait(3) and write_failed.wait(3)
            assert not producer_errors
            _wait(lambda: jobs.get("blocked")["status"] == jobs.get("other")["status"] == "completed")
            _wait(lambda: "blocked" not in jobs._active)
            state = jobs.get("blocked")
            assert jobs._thread.is_alive() and events.empty()
            assert "state file inaccessible" in state["persistence_error"]
            assert any(row.get("persistence_error") for row in jobs.list() if row["id"] == "blocked")
            assert state["log"][-1] == "CoilGen: Numerical update 319"
            assert blocked_path.read_bytes() == previous
            durable_other = json.loads((tmp_path / "other" / "job.json").read_text())
            assert durable_other["status"] == "completed" and "persistence_error" not in durable_other
            # Both processes have been reaped; retries must not depend on an active worker.
            released.set()
            _wait(lambda: "persistence_error" not in jobs.get("blocked"))
            durable = json.loads(blocked_path.read_text())
            assert durable["status"] == "completed" and durable["progress"] == 1
            assert durable["log"][-1] == "CoilGen: Numerical update 319"
            assert durable["worker_alive"] is False and "persistence_error" not in durable
        finally:
            released.set()
            blocked.alive = other.alive = False
            thread.join(3)


def test_permanent_state_denial_remains_visible_after_close_and_releases_runtime_owner(tmp_path, monkeypatch):
    jobs = JobManager(tmp_path)
    original, denied = runtime.atomic_json, threading.Event()
    try:
        with jobs._lock:
            process, events = _active(jobs, "blocked")
            path = tmp_path / "blocked" / "job.json"
            previous = path.read_bytes()

            def write(destination, state):
                if destination == path:
                    denied.set()
                    raise OSError("Fixture permanent state denial")
                return original(destination, state)

            monkeypatch.setattr(runtime, "atomic_json", write)
        events.put(_progress(12))
        assert denied.wait(3)
        _wait(lambda: "persistence_error" in jobs.get("blocked"))
        assert jobs._thread.is_alive() and jobs.get("blocked")["status"] == "running"
        jobs.close()
        state = jobs.get("blocked")
        assert state["status"] == "interrupted" and state["worker_alive"] is False
        assert "permanent state denial" in state["persistence_error"]
        assert not jobs._thread.is_alive() and not process.is_alive()
        assert path.read_bytes() == previous
        monkeypatch.setattr(runtime, "atomic_json", original)
        with JobManager(tmp_path) as reopened:
            assert reopened.get("blocked")["status"] == "interrupted"
    finally:
        monkeypatch.setattr(runtime, "atomic_json", original)
        jobs.close()


def test_explicit_cancel_still_raises_on_undurable_state_write(tmp_path, monkeypatch):
    original = runtime.atomic_json
    with JobManager(tmp_path) as jobs:
        with jobs._lock:
            process, _events = _active(jobs, "blocked")
            path = tmp_path / "blocked" / "job.json"

            def write(destination, state):
                if destination == path:
                    raise OSError("Fixture cancel write denied")
                return original(destination, state)

            monkeypatch.setattr(runtime, "atomic_json", write)
            with pytest.raises(OSError, match="cancel write denied"):
                jobs.cancel("blocked")
            assert jobs.get("blocked")["status"] == "cancelling"
            monkeypatch.setattr(runtime, "atomic_json", original)
            process.alive = False
