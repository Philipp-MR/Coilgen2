"""Project discard and atomic note saving retain the intended scientific owner."""

from hashlib import sha256
import json

import pytest
import test_workbench_usability as usability

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from coilgen.application import DesignService, ProjectAssembly, ProjectDocument
from coilgen.application.design_studies import design_study_state
from coilgen.application.presets import preset_config
from coilgen.persistence import atomic_json, file_digest
from coilgen.research.storage import result_metadata_digest
from test_run_history import write_run


application = usability.application
windows = usability.windows


def _owned_run(project, directory):
    """Construct static evidence with real integrity checks, without a solver."""
    record = write_run(directory, revision=project.revision, config=project.values)
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    request.update(project_id=project.id, name=project.name, **project.study)
    digest = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    atomic_json(directory / "request.json", request)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(request=request, request_sha256=digest)
    atomic_json(directory / "manifest.json", manifest)
    metadata = record.metadata
    metadata.update(
        name=project.name,
        project_id=project.id,
        project_fingerprint=project.fingerprint,
        project_model_fingerprint=project.model_fingerprint,
        request_sha256=digest,
    )
    for artifact in metadata["artifacts"]:
        artifact["sha256"] = file_digest(directory / artifact["name"])
    metadata["metadata_sha256"] = result_metadata_digest(metadata)
    atomic_json(directory / "result.json", metadata)
    return DesignService.load_run(directory)


def _notes(window, channel_index, record, hypothesis, rationale):
    window.channel_selector.setCurrentIndex(channel_index)
    window.load_run(record.directory)
    entry = next(entry for entry in window._history_entries if entry.directory == record.directory)
    assert window.history_panel.select_record(entry.id)
    assert window.history_panel.rationale.isEnabled()
    window.history_panel.hypothesis.setPlainText(hypothesis)
    window.history_panel.rationale.setPlainText(rationale)


@pytest.fixture
def owners(windows, tmp_path):
    first = ProjectDocument.from_config(preset_config(), name="First Gy")
    second = ProjectDocument.from_config(preset_config("biplanar"), name="Second Gy")
    assembly = ProjectAssembly.new("Independent comparison", (first, second))
    window = windows(assembly)
    records = tuple(
        _owned_run(project, tmp_path / f"static-run-{index}") for index, project in enumerate((first, second))
    )
    yield window, (first, second), records
    window._can_discard = lambda: True  # Teardown must not prompt for intentionally retained drafts.


def test_close_discard_then_reopen_does_not_revive_any_owner_note_drafts(owners, tmp_path, monkeypatch):
    window, projects, records = owners
    original = window.assembly
    path = original.save(tmp_path / "original.json")
    for index, record in enumerate(records):
        _notes(window, index, record, f"Discard hypothesis {index}", f"Discard rationale {index}")
    # Keep the other channel active so discard must clear both owner caches.
    assert window.project.id == projects[1].id
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Discard)
    assert window.close_project()
    assert window.project is None
    window.open_project(path)
    for index, project in enumerate(projects):
        window.channel_selector.setCurrentIndex(index)
        state = design_study_state(window.project)
        assert window.project.fingerprint == project.fingerprint
        assert state.hypothesis == "" and state.notes == ()
        assert not window.history_panel.has_note_drafts(project.id, state)
        assert window.history_panel.pending_notes is None
        assert window.history_panel.pending_note_updates == ()
        assert window.history_panel.hypothesis.toPlainText() == ""
    assert window.service.submissions == []


@pytest.mark.parametrize("operation", ["close", "new", "open", "template"])
def test_cancel_keeps_inactive_owner_drafts_and_original_document(owners, tmp_path, monkeypatch, operation):
    window, projects, records = owners
    _notes(window, 0, records[0], "Retain the earlier intent", "Retain its result decision")
    window.channel_selector.setCurrentIndex(1)
    original = window.assembly
    selected = window.project
    assert window.history_panel.pending_notes is None  # The only draft is in the inactive owner.
    prompts = []

    def cancel(*args):
        prompts.append(args[1])
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", cancel)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: pytest.fail("Cancel must stop before Open"))
    if operation == "close":
        assert not window.close_project()
    elif operation == "new":
        assert window.new_project(name="Unused") is None
    elif operation == "open":
        window._open_dialog()
    else:
        assert window.load_template("cylinder") is None
    assert prompts == ["Unsaved project changes"]
    assert window.assembly is original and window.project is selected
    window.channel_selector.setCurrentIndex(0)
    hypothesis, decisions = window.history_panel.note_drafts(projects[0].id, design_study_state(window.project))
    assert hypothesis == "Retain the earlier intent"
    assert [rationale for _entry, rationale in decisions] == ["Retain its result decision"]
    assert window.service.submissions == []


def test_failed_open_after_discard_choice_retains_notes_and_owner(owners, tmp_path, monkeypatch):
    window, projects, records = owners
    _notes(window, 0, records[0], "Keep this intent", "Keep this decision")
    original = window.assembly
    draft = window.history_panel.note_drafts(projects[0].id, design_study_state(window.project))
    path = tmp_path / "malformed.json"
    path.write_text("Invalid JSON", encoding="utf-8")
    errors = []
    monkeypatch.setattr(window, "_error", errors.append)
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Discard)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(path), ""))
    window._open_dialog()
    assert window.assembly is original and window.project.id == projects[0].id
    assert window.history_panel.note_drafts(projects[0].id, design_study_state(window.project)) == draft
    # File Open verifies off-thread; the failed parse must preserve drafts both
    # while loading and after the queued failure reaches the current editor.
    if not errors:
        loop = QEventLoop()
        poll, deadline = QTimer(), QTimer()
        poll.timeout.connect(lambda: loop.quit() if errors else None)
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        poll.start(5)
        deadline.start(3000)
        try:
            loop.exec()
        finally:
            poll.stop()
            deadline.stop()
    assert window._project_open_task is None
    assert window.assembly is original and window.project.id == projects[0].id
    assert window.history_panel.note_drafts(projects[0].id, design_study_state(window.project)) == draft
    assert len(errors) == 1 and errors[0].startswith("Cannot open project:")
    assert window.service.submissions == []


def test_save_from_one_channel_preserves_every_owners_pending_notes_in_one_json(owners, tmp_path):
    window, projects, records = owners
    for index, record in enumerate(records):
        _notes(window, index, record, f"Hypothesis {index}", f"Rationale {index}")
    before = window.assembly
    undo_count = window.undo_stack.count()
    path = window.save_project(tmp_path / "both-notes.json")
    loaded = ProjectAssembly.load(path)
    assert window.undo_stack.count() == undo_count + 1
    assert window.undo_stack.isClean()
    for index, (channel, original, record) in enumerate(zip(loaded.channels, projects, records)):
        state = design_study_state(channel.project)
        assert state.hypothesis == f"Hypothesis {index}"
        assert state.note_for(record).rationale == f"Rationale {index}"
        assert channel.project.revision == original.revision
        assert channel.project.values == original.values
        assert channel.project.fingerprint == original.fingerprint
        assert channel.project.model_fingerprint == original.model_fingerprint
        assert len(channel.project.saved_runs) == 1
        assert not window.history_panel.has_note_drafts(channel.project.id, state)
    # One atomic undo restores both note states, with their scientific revisions unchanged.
    window.undo_stack.undo()
    for channel, original in zip(window.assembly.channels, before.channels):
        assert channel.project.session == original.project.session
        assert channel.project.revision == original.project.revision
        assert channel.project.fingerprint == original.project.fingerprint
    assert window.service.submissions == []


@pytest.mark.parametrize("failure", ["hypothesis", "rationale", "artifact"])
def test_invalid_inactive_notes_fail_atomically_before_publishing_either_owner(owners, tmp_path, failure):
    window, projects, records = owners
    _notes(window, 0, records[0], "Valid first intent", "Valid first decision")
    _notes(
        window,
        1,
        records[1],
        "x" * 4001 if failure == "hypothesis" else "Second intent",
        "x" * 4001 if failure == "rationale" else "Second decision",
    )
    window.channel_selector.setCurrentIndex(0)
    before = window.assembly
    selected = window.project
    undo_count = window.undo_stack.count()
    if failure == "artifact":
        with (records[1].directory / "scientific_result.npz").open("ab") as stream:
            stream.write(b"Altered after note selection")
    path = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="4000|integrity"):
        window.save_project(path)
    assert not path.exists() and window.project_path is None
    assert window.assembly is before and window.project is selected
    assert window.undo_stack.count() == undo_count
    for channel, original in zip(window.assembly.channels, projects):
        assert channel.project.fingerprint == original.fingerprint
        assert channel.project.session == original.session
        assert channel.project.saved_runs == ()
        assert window.history_panel.has_note_drafts(channel.project.id, design_study_state(channel.project))
    assert window.service.submissions == []
