"""Background live display loading is coalesced and cannot revive stale jobs."""

from copy import deepcopy
from threading import Event, Lock, get_ident
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtTest import QTest

import test_native_workbench as native

from coilgen.application.stage_preview import MANIFEST_NAME, StagePreview
from coilgen.gui.live_preview import LivePreviewController

application = native.application


def _state(job="a" * 32, **changes):
    return {
        "id": job, "status": "running", "stage": "Preparing model", "progress": 0.2,
        "request_sha256": "b" * 64, "project_id": "c" * 32,
        "project_fingerprint": "d" * 64, "project_revision": 3, **changes,
    }


def _write_manifest(jobs, state, content="prepared"):
    directory = jobs / state["id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_NAME).write_text(content, encoding="utf-8")
    return directory


def _preview(directory, stage="prepared"):
    return StagePreview(directory, {"pipeline_stage": stage}, {"vertices_m": np.array([[0., 0., 0.]])})


def _wait(application, predicate, message="Background preview did not finish", timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    pytest.fail(message)


@pytest.fixture
def controllers(application, tmp_path):
    created, releases = [], []
    jobs = tmp_path / "jobs"

    def create(loader):
        controller = LivePreviewController(jobs, loader=loader)
        created.append(controller)
        return controller

    yield jobs, create, releases
    for release in releases:
        release.set()
    for controller in created:
        controller.close()
    _wait(application, lambda: all(controller._worker is None for controller in created))
    for controller in created:
        controller.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_loader_runs_off_ui_and_ready_returns_on_ui_with_detached_identity(controllers, application):
    jobs, create, releases = controllers
    state = _state()
    directory = _write_manifest(jobs, state)
    entered, release = Event(), Event()
    releases.append(release)
    calls, ready, changed = [], [], []
    main_thread = get_ident()

    def loader(path, **identity):
        calls.append((get_ident(), path, deepcopy(identity)))
        entered.set()
        assert release.wait(3)
        return _preview(path)

    controller = create(loader)
    controller.preview_ready.connect(lambda job, preview: ready.append((get_ident(), job, preview)))
    controller.changed.connect(lambda: changed.append(get_ident()))
    controller.start(state)
    assert entered.wait(3)
    assert controller.loading and controller.following
    assert controller.preview is None and controller.job_id == state["id"]
    state["project_revision"] = 99
    detached = controller.state
    detached["project_revision"] = 88
    assert controller.state["project_revision"] == 3
    release.set()
    _wait(application, lambda: not controller.loading and bool(ready))
    assert calls[0][0] != main_thread and ready[0][0] == main_thread
    assert calls[0][1] == directory.resolve()
    assert calls[0][2] == {
        "expected_request_sha256": "b" * 64, "expected_project_id": "c" * 32,
        "expected_project_fingerprint": "d" * 64, "expected_project_revision": 3,
    }
    assert controller.preview is ready[0][2] and controller.error is None
    assert changed and all(thread == main_thread for thread in changed)


def test_unchanged_manifest_is_loaded_once_and_missing_manifest_is_harmless(controllers, application):
    jobs, create, _releases = controllers
    state, calls = _state(), []
    controller = create(lambda path, **_identity: calls.append(path) or _preview(path))
    controller.start(state)
    assert not controller.loading and controller.preview is None and controller.error is None
    _write_manifest(jobs, state)
    controller.refresh(state)
    _wait(application, lambda: controller.preview is not None and not controller.loading)
    for progress in (0.3, 0.5, 0.9):
        controller.refresh({**state, "progress": progress})
    QTest.qWait(30)
    assert len(calls) == 1 and controller.state["progress"] == 0.9


def test_rapid_stage_announcements_load_only_latest_pending_manifest(controllers, application):
    jobs, create, releases = controllers
    state = _state()
    _write_manifest(jobs, state, "prepared")
    entered, release = Event(), Event()
    releases.append(release)
    calls, lock, active, maximum = [], Lock(), 0, 0

    def loader(path, **_identity):
        nonlocal active, maximum
        content = (path / MANIFEST_NAME).read_text(encoding="utf-8")
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append(content)
        if content == "prepared":
            entered.set()
            assert release.wait(3)
        with lock:
            active -= 1
        return _preview(path, content)

    controller = create(loader)
    controller.start(state)
    assert entered.wait(3)
    _write_manifest(jobs, state, "sheet boundary")
    controller.refresh(state)
    _write_manifest(jobs, state, "contours boundary is latest")
    controller.refresh(state)
    assert calls == ["prepared"] and controller.loading
    release.set()
    _wait(application, lambda: not controller.loading and len(calls) == 2)
    assert calls == ["prepared", "contours boundary is latest"] and maximum == 1
    assert controller.preview.metadata["pipeline_stage"] == "contours boundary is latest"


@pytest.mark.parametrize("action", ["pause", "stop", "close", "identity_changed"])
def test_invalidated_callbacks_do_not_revive_old_preview(controllers, application, action):
    jobs, create, releases = controllers
    state = _state()
    _write_manifest(jobs, state)
    entered, release = Event(), Event()
    releases.append(release)

    def loader(path, **_identity):
        entered.set()
        assert release.wait(3)
        return _preview(path)

    controller = create(loader)
    ready = []
    controller.preview_ready.connect(lambda *args: ready.append(args))
    controller.start(state)
    assert entered.wait(3)
    if action == "identity_changed":
        controller.refresh({**state, "project_revision": 4})
    else:
        getattr(controller, action)()
    assert not controller.following and not controller.loading
    release.set()
    _wait(application, lambda: controller._worker is None)
    assert ready == [] and controller.preview is None


@pytest.mark.parametrize("error", [MemoryError(), AttributeError("Unexpected archive data")])
def test_optional_loader_failure_always_finishes_and_can_load_a_new_commit(controllers, application, error):
    jobs, create, _releases = controllers
    state = _state()
    _write_manifest(jobs, state)
    calls = []

    def loader(path, **_identity):
        calls.append(path)
        if len(calls) == 1:
            raise error
        return _preview(path)

    controller = create(loader)
    controller.start(state)
    _wait(application, lambda: not controller.loading and controller.error is not None)
    assert controller.preview is None
    _write_manifest(jobs, state, "new completed boundary")
    controller.refresh(state)
    _wait(application, lambda: not controller.loading and controller.preview is not None)
    assert controller.error is None and len(calls) == 2


def test_pause_keeps_last_snapshot_and_resume_reads_latest_commit(controllers, application):
    jobs, create, _releases = controllers
    state, calls = _state(), []
    _write_manifest(jobs, state)

    def loader(path, **_identity):
        content = (path / MANIFEST_NAME).read_text(encoding="utf-8")
        calls.append(content)
        return _preview(path, content)

    controller = create(loader)
    controller.start(state)
    _wait(application, lambda: controller.preview is not None and not controller.loading)
    previous = controller.preview
    controller.pause()
    _write_manifest(jobs, state, "routed boundary")
    controller.refresh({**state, "progress": 0.8})
    QTest.qWait(30)
    assert controller.preview is previous and calls == ["prepared"] and not controller.following
    controller.resume()
    _wait(application, lambda: not controller.loading and len(calls) == 2)
    assert controller.following and controller.preview.metadata["pipeline_stage"] == "routed boundary"


def test_new_job_waits_for_old_worker_and_old_results_are_discarded(controllers, application):
    jobs, create, releases = controllers
    first, second = _state(), _state("e" * 32, project_revision=4, request_sha256="f" * 64)
    _write_manifest(jobs, first)
    _write_manifest(jobs, second)
    entered, release = Event(), Event()
    releases.append(release)
    calls, ready = [], []

    def loader(path, **identity):
        calls.append((path.name, identity["expected_project_revision"]))
        if path.name == first["id"]:
            entered.set()
            assert release.wait(3)
        return _preview(path)

    controller = create(loader)
    controller.preview_ready.connect(lambda job, preview: ready.append((job, preview)))
    controller.start(first)
    assert entered.wait(3)
    controller.start(second)
    assert calls == [(first["id"], 3)] and controller.job_id == second["id"]
    release.set()
    _wait(application, lambda: controller._worker is None and bool(ready))
    assert calls == [(first["id"], 3), (second["id"], 4)]
    assert [job for job, _preview_ in ready] == [second["id"]]


def test_corrupt_announcement_keeps_last_good_preview_and_recovers_on_new_commit(controllers, application):
    jobs, create, _releases = controllers
    state, calls = _state(), []
    _write_manifest(jobs, state)

    def loader(path, **_identity):
        content = (path / MANIFEST_NAME).read_text(encoding="utf-8")
        calls.append(content)
        if content == "corrupt archive":
            raise ValueError("Stage preview array integrity check failed")
        return _preview(path, content)

    controller = create(loader)
    controller.start(state)
    _wait(application, lambda: controller.preview is not None and not controller.loading)
    good = controller.preview
    _write_manifest(jobs, state, "corrupt archive")
    controller.refresh(state)
    _wait(application, lambda: not controller.loading and controller.error is not None)
    assert controller.preview is good and "integrity" in controller.error
    controller.refresh(state)
    QTest.qWait(30)
    assert calls == ["prepared", "corrupt archive"]
    _write_manifest(jobs, state, "new verified routed stage")
    controller.refresh(state)
    _wait(application, lambda: not controller.loading and len(calls) == 3)
    assert controller.preview is not good and controller.error is None


@pytest.mark.parametrize("changes", [
    {"id": "../outside"}, {"id": "C:/outside"}, {"request_sha256": "invalid"},
    {"project_id": None}, {"project_fingerprint": ""},
])
def test_unverified_job_identity_never_reads_files(controllers, changes):
    _jobs, create, _releases = controllers
    calls = []
    controller = create(lambda *_args, **_kwargs: calls.append(True))
    controller.start(_state(**changes))
    assert calls == [] and not controller.loading and controller.preview is None


def test_other_job_refresh_does_not_switch_active_followed_identity(controllers, application):
    jobs, create, _releases = controllers
    state = _state()
    _write_manifest(jobs, state)
    controller = create(lambda path, **_identity: _preview(path))
    controller.start(state)
    _wait(application, lambda: not controller.loading and controller.preview is not None)
    previous = controller.preview
    controller.refresh(_state("e" * 32, project_revision=4))
    assert controller.job_id == state["id"] and controller.preview is previous


def test_reentrant_ready_refresh_cannot_launch_two_loaders(controllers, application):
    jobs, create, releases = controllers
    state = _state()
    _write_manifest(jobs, state, "prepared")
    entered, release_first, release_later = Event(), Event(), Event()
    releases.extend((release_first, release_later))
    lock, calls, active, maximum = Lock(), [], 0, 0

    def loader(path, **_identity):
        nonlocal active, maximum
        content = (path / MANIFEST_NAME).read_text(encoding="utf-8")
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append(content)
        if content == "prepared":
            entered.set()
            assert release_first.wait(3)
        else:
            assert release_later.wait(3)
        with lock:
            active -= 1
        return _preview(path, content)

    controller = create(loader)

    def request_newer(_job, preview):
        if preview.metadata["pipeline_stage"] == "prepared":
            _write_manifest(jobs, state, "newest routed boundary")
            controller.refresh(state)

    controller.preview_ready.connect(request_newer)
    controller.start(state)
    assert entered.wait(3)
    _write_manifest(jobs, state, "sheet boundary")
    controller.refresh(state)
    release_first.set()
    _wait(application, lambda: len(calls) >= 2)
    QTest.qWait(40)
    assert maximum == 1, "A ready handler launched a worker while an old pending request launched another"
    release_later.set()
    _wait(application, lambda: controller._worker is None)
