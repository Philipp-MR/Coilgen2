"""Background project preparation retains application integrity and cache semantics."""

import json
from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QObject, QThread, QThreadPool, QTimer, Slot
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from shiboken6 import delete

from coilgen.application import ProjectAssembly
from coilgen.application.bundles import RunArchive
from coilgen.application.history import RunHistoryStore
from coilgen.gui import project_loading
from coilgen.gui.project_loading import ProjectOpenTask, load_document, prepare_history

from test_application_project_bundle import _legacy_project, _prepared_document


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    project, record, archive = _prepared_document(tmp_path_factory.mktemp("project-open-source"))
    return project.with_results((archive,)), record, archive


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["project-loading-test"])


@pytest.mark.parametrize("collection", (False, True))
@pytest.mark.parametrize("compact", (False, True))
def test_loader_retains_document_identity_and_arrays(prepared, tmp_path, collection, compact):
    project, record, _archive = prepared
    document = ProjectAssembly.new("Independent channels", (project,)) if collection else project
    path = document.save(tmp_path / ("owned.coilgen" if compact else "owned.json"))
    restored = load_document(path)
    assert type(restored) is type(document)
    assert restored.fingerprint == document.fingerprint
    owner = restored.channels[0].project if collection else restored
    evidence = owner.saved_runs[0].materialize(tmp_path / "cache")
    assert evidence.metadata == record.metadata
    for name, array in record.arrays.items():
        np.testing.assert_array_equal(evidence.arrays[name], array)
        assert evidence.arrays[name].dtype == array.dtype


@pytest.mark.parametrize("duplicate", (False, True))
def test_loader_reads_json_once_and_rejects_duplicate_keys(prepared, tmp_path, monkeypatch, duplicate):
    project, _record, _archive = prepared
    path = project.save(tmp_path / "single-read.json")
    if duplicate:
        path.write_text(path.read_text().replace('"name":', '"name": "Duplicate", "name":', 1))
    reads, original = [], Path.read_text

    def read(source, *args, **kwargs):
        if source.resolve() == path:
            reads.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    if duplicate:
        with pytest.raises(ValueError, match="Duplicate project JSON property: name"):
            load_document(path)
    else:
        assert load_document(path).fingerprint == project.fingerprint
    assert reads == [path]


def test_loader_preserves_relative_legacy_asset_resolution_and_rejects_escape(prepared, tmp_path, monkeypatch):
    project, _record, _archive = prepared
    source = tmp_path / "document"
    source.mkdir()
    path = source / "legacy.json"
    value = _legacy_project(project, path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    restored = load_document(Path("..") / "document" / path.name)
    assert restored.values == project.values
    assert restored.model_fingerprint == project.model_fingerprint
    descriptor = next(iter(value["assets"].values()))
    descriptor["path"] = "../outside.npz"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the project directory"):
        load_document(path)


def test_loader_rejects_corrupt_embedded_artifact_and_empty_collection(prepared, tmp_path):
    project, _record, _archive = prepared
    path = project.save(tmp_path / "corrupt.json")
    value = json.loads(path.read_text())
    files = value["results"][0]["files"]
    artifact = files["scientific_result.npz"]
    artifact["data"] = ("A" if artifact["data"][0] != "A" else "B") + artifact["data"][1:]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_document(path)
    empty = ProjectAssembly.new("Empty", ()).save(tmp_path / "empty.json")
    with pytest.raises(ValueError, match="no channels"):
        load_document(empty)


def test_prepared_history_is_detached_and_recursively_read_only(prepared, tmp_path):
    project, _record, archive = prepared
    history = RunHistoryStore(tmp_path / "history.json")
    first = prepare_history(project, (project.id, None), history, tmp_path / "cache")
    selected = first.embedded_ids[archive.id]
    history.save_session(project.id, selected_id=selected, view_state={"camera": [1, 2, 3]})
    result = prepare_history(project, (project.id, None), history, tmp_path / "cache")
    assert result.entries[0].verified and result.records[selected].directory == tmp_path / "cache" / archive.id
    assert result.ids[selected] == selected and result.session["selected_id"] == selected
    assert result.session["view_state"]["camera"] == (1, 2, 3)
    for mapping in (result.records, result.ids, result.embedded_ids, result.session, result.session["view_state"]):
        with pytest.raises(TypeError):
            mapping["changed"] = True
    history.save_session(project.id, view_state={"camera": [9]})
    assert result.session["view_state"]["camera"] == (1, 2, 3)
    with pytest.raises(ValueError, match="scope"):
        prepare_history(project, ("another-project", None), history, tmp_path / "cache")


@pytest.mark.parametrize("failure_stage", ("archive", "link", "restore", "session"))
def test_optional_cache_failures_keep_only_independently_verified_evidence(
    prepared, tmp_path, monkeypatch, failure_stage
):
    project, record, archive = prepared
    history = RunHistoryStore(tmp_path / "history.json")

    def fail(*_args, **_kwargs):
        raise OSError("Fixture cache inaccessible")

    if failure_stage == "archive":
        monkeypatch.setattr(RunArchive, "materialize", fail)
    else:
        monkeypatch.setattr(
            history, {"link": "add_run", "restore": "restore", "session": "session"}[failure_stage], fail
        )
    result = prepare_history(project, (project.id, None), history, tmp_path / "cache")
    assert any("cache inaccessible" in message for message in result.diagnostics)
    if failure_stage == "archive":
        assert not result.records and archive.id not in result.embedded_ids
        assert len(result.entries) == 1 and not result.entries[0].verified
    else:
        selected = result.embedded_ids[archive.id]
        assert selected in result.records and result.entries[0].verified
        assert result.records[selected].metadata == record.metadata
        if failure_stage != "link":
            assert selected != archive.id


@pytest.mark.parametrize("corrupt_files", (False, True))
def test_unavailable_history_evicts_all_aliases_without_bypassing_verification(prepared, tmp_path, corrupt_files):
    project, _record, archive = prepared
    history = RunHistoryStore(tmp_path / "history.json")
    first = prepare_history(project, (project.id, None), history, tmp_path / "cache")
    directory = first.records[archive.id].directory
    if corrupt_files:
        (directory / "scientific_result.npz").write_bytes(b"corrupt artifact")
    else:
        value = json.loads(history.path.read_text())
        value["records"][0]["request_sha256"] = "f" * 64
        history.path.write_text(json.dumps(value), encoding="utf-8")
    result = prepare_history(project, (project.id, None), history, tmp_path / "cache")
    assert not result.records
    assert result.unavailable_directories == (directory,)
    assert any(not entry.verified for entry in result.entries)
    assert result.diagnostics


def _wait(application, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    pytest.fail("Project preparation did not finish")


def test_worker_keeps_qt_heartbeat_running_and_delivers_on_gui_thread(application, prepared, tmp_path, monkeypatch):
    project, _record, _archive = prepared
    second = project.duplicate("Selected second")
    assembly = ProjectAssembly.new("Pair", (project, second))
    selected = assembly.channels[1]
    assembly = assembly.update_channel(
        selected.id,
        project=selected.project.with_results(selected.project.saved_runs, session={"active_channel_id": selected.id}),
    )
    path = assembly.save(tmp_path / "pair.json")
    entered, release = Event(), Event()
    original = project_loading.load_document
    worker_threads, heartbeats = [], []

    def blocked(*args, **kwargs):
        worker_threads.append(QThread.currentThread())
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(project_loading, "load_document", blocked)

    class Receiver(QObject):
        def __init__(self):
            super().__init__()
            self.messages, self.results, self.threads = [], [], []

        @Slot(object, str)
        def progress(self, token, message):
            self.messages.append((token, message))
            self.threads.append(QThread.currentThread())

        @Slot(object, object, str)
        def finished(self, token, result, error):
            self.results.append((token, result, error))
            self.threads.append(QThread.currentThread())

    token, receiver = object(), Receiver()
    task = ProjectOpenTask(path, tmp_path / "cache", token=token)
    task.signals.progress.connect(receiver.progress)
    task.signals.finished.connect(receiver.finished)
    timer = QTimer()
    timer.timeout.connect(lambda: heartbeats.append(time.monotonic()))
    timer.start(5)
    QThreadPool.globalInstance().start(task)
    try:
        assert entered.wait(5)
        _wait(application, lambda: len(heartbeats) >= 3)
        assert not receiver.results
    finally:
        release.set()
        _wait(application, lambda: bool(receiver.results))
        timer.stop()
    delivered_token, result, error = receiver.results[0]
    assert delivered_token is token and not error
    assert result.document.fingerprint == assembly.fingerprint
    assert result.active_channel_id == selected.id
    assert result.history.key == (selected.project.id, selected.id)
    assert all(thread == application.thread() for thread in receiver.threads)
    assert worker_threads[0] != application.thread()
    assert receiver.messages and all(id_ is token for id_, _message in receiver.messages)


@pytest.mark.parametrize("valid", (False, True))
def test_worker_finishes_safely_after_qt_destroys_signal_source(prepared, tmp_path, valid):
    project, _record, _archive = prepared
    path = project.save(tmp_path / "valid.json") if valid else tmp_path / "missing.json"
    task = ProjectOpenTask(path, tmp_path / "cache", token=object())
    delete(task.signals)
    task.run()
    assert (tmp_path / "cache").exists() is valid


@pytest.mark.parametrize("stage", ("before", "read", "history", "progress"))
def test_cancellation_cannot_publish_a_partially_prepared_document(prepared, tmp_path, monkeypatch, stage):
    project, _record, _archive = prepared
    path = project.save(tmp_path / "owned.json")
    token, delivered = object(), []
    task = ProjectOpenTask(path, tmp_path / "cache", token=token)
    task.signals.finished.connect(lambda id_, result, error: delivered.append((id_, result, error)))
    if stage == "before":
        task.cancel()
    elif stage == "progress":
        task.signals.progress.connect(lambda *_args: task.cancel())
    else:
        name = "load_document" if stage == "read" else "prepare_history"
        original = getattr(project_loading, name)

        def cancelled_after(*args, **kwargs):
            result = original(*args, **kwargs)
            task.cancel()
            return result

        monkeypatch.setattr(project_loading, name, cancelled_after)
    task.run()
    assert delivered == [(token, None, "Cancelled")]
    assert (tmp_path / "cache").exists() is (stage == "history")


def test_worker_captures_states_and_deduplicates_verified_reuse_sources(prepared, tmp_path, monkeypatch):
    from coilgen.application.reuse import ReusePlan

    project, original, archive = prepared
    cached = archive.materialize(tmp_path / "cache")
    state = {"id": "stopped", "status": "cancelled", "completed_stages": ["prepared"]}
    task = ProjectOpenTask(
        project.save(tmp_path / "owned.json"),
        tmp_path / "cache",
        records=(cached, cached, original),
        states=(state,),
        jobs_directory=tmp_path / "jobs",
        threads=2,
    )
    state["status"] = "running"
    state["completed_stages"].append("sheet")
    calls, delivered = [], []
    expected = ReusePlan((), (), ("fixture checkpoint verification",))

    def reuse(owner, records, states, **options):
        calls.append((owner, records, states, options))
        return expected

    monkeypatch.setattr(project_loading, "execution_reuse", reuse)
    task.signals.finished.connect(lambda _token, result, error: delivered.append((result, error)))
    task.run()
    result, error = delivered[0]
    assert not error and result.readiness is not None and result.reuse is expected
    owner, records, states, options = calls[0]
    assert owner.fingerprint == project.fingerprint
    assert len(records) == 2 and len({record.directory for record in records}) == 2
    assert states == ({"id": "stopped", "status": "cancelled", "completed_stages": ["prepared"]},)
    assert options == {"jobs_directory": tmp_path / "jobs", "threads": 2}


def test_worker_reuse_failure_is_diagnostic_without_losing_verified_results(prepared, tmp_path, monkeypatch):
    project, _record, archive = prepared
    task = ProjectOpenTask(project.save(tmp_path / "owned.json"), tmp_path / "cache")
    delivered = []

    def fail(*_args, **_kwargs):
        raise ValueError("Checkpoint integrity failed")

    monkeypatch.setattr(project_loading, "execution_reuse", fail)
    task.signals.finished.connect(lambda _token, result, error: delivered.append((result, error)))
    task.run()
    result, error = delivered[0]
    assert not error and archive.id in result.history.records
    assert not result.reuse.available and result.reuse.diagnostics == ("Checkpoint integrity failed",)


@pytest.mark.parametrize("operation", ("check_readiness", "execution_reuse"))
def test_cancellation_after_readiness_or_reuse_prevents_publication(prepared, tmp_path, monkeypatch, operation):
    project, _record, _archive = prepared
    task = ProjectOpenTask(project.save(tmp_path / "owned.json"), tmp_path / "cache")
    original, delivered = getattr(project_loading, operation), []

    def cancelled_after(*args, **kwargs):
        result = original(*args, **kwargs)
        task.cancel()
        return result

    monkeypatch.setattr(project_loading, operation, cancelled_after)
    task.signals.finished.connect(lambda _token, result, error: delivered.append((result, error)))
    task.run()
    assert delivered == [(None, "Cancelled")]


def test_cancellation_after_archive_verification_prevents_history_alias_writes(prepared, tmp_path, monkeypatch):
    project, _record, _archive = prepared
    history = RunHistoryStore(tmp_path / "history.json")
    task = ProjectOpenTask(project.save(tmp_path / "owned.json"), tmp_path / "cache", history=history)
    original, delivered = RunArchive.materialize, []

    def cancelled_after(*args, **kwargs):
        result = original(*args, **kwargs)
        task.cancel()
        return result

    monkeypatch.setattr(RunArchive, "materialize", cancelled_after)
    task.signals.finished.connect(lambda _token, result, error: delivered.append((result, error)))
    task.run()
    assert delivered == [(None, "Cancelled")]
    assert not history.path.exists()


def test_cancellation_after_cache_link_preserves_a_newer_display_alias(prepared, tmp_path, monkeypatch):
    project, _record, archive = prepared
    history = RunHistoryStore(tmp_path / "history.json")
    first = prepare_history(project, (project.id, None), history, tmp_path / "cache")
    history.rename_run(first.embedded_ids[archive.id], "User's newer alias")
    task = ProjectOpenTask(project.save(tmp_path / "owned.json"), tmp_path / "cache", history=history)
    original, delivered = history.add_run, []

    def cancelled_after(*args, **kwargs):
        result = original(*args, **kwargs)
        task.cancel()
        return result

    monkeypatch.setattr(history, "add_run", cancelled_after)
    task.signals.finished.connect(lambda _token, result, error: delivered.append((result, error)))
    task.run()
    assert delivered == [(None, "Cancelled")]
    assert history.restore(project.id)[0].alias == "User's newer alias"
