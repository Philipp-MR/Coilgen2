"""Modeless expert edits use one retained draft and atomic model comparison."""

from copy import deepcopy
import time

import pytest
import test_workbench_usability as usability
from test_native_workbench import FakeJobs

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog, QMessageBox, QPushButton
from shiboken6 import isValid

from coilgen.application import ProjectDocument
from coilgen.application.presets import EXAMPLE_TITLES, preset_config, preset_study
from coilgen.gui.workbench import CoilGenWorkbench


application = usability.application


@pytest.fixture
def expert_window(application, tmp_path):
    project = ProjectDocument.from_config(preset_config(), name=EXAMPLE_TITLES["cylinder"]).update_study(preset_study())
    window = CoilGenWorkbench(
        project,
        jobs=FakeJobs(tmp_path / "isolated-expert-jobs"),
        service=usability.Service(),
        viewport_factory=usability.Preview,
    )
    window.timer.stop()
    window._advanced_dialog()
    yield window
    window._can_discard = lambda: True
    window.close()
    window.undo_stack.clear()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _edit(expert, key="min_turn_distance", text="4"):
    expert.fields[key].control.setText(text)


def _wait_for_project_open(application, window):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        application.processEvents()
        if window._project_open_task is None:
            return
        QTest.qWait(5)
    pytest.fail("Background project opening did not finish")


def test_expert_window_is_modeless_and_reuses_uncommitted_draft_after_close(expert_window, application):
    window = expert_window
    original, expert = window.project, window.expert_window
    assert not expert.isModal() and expert.windowModality() == Qt.WindowModality.NonModal
    _edit(expert, text="4.12345678912345")
    draft = expert.draft
    assert window.project is original and window.undo_stack.count() == 0
    expert.close()
    assert expert.isHidden()
    window._advanced_dialog()
    application.processEvents()
    assert window.expert_window is expert and expert.isVisible()
    assert expert.draft == draft
    assert expert.fields["min_turn_distance"].control.text() == "4.12345678912345"
    assert window.project is original and window.undo_stack.count() == 0


def test_valid_expert_apply_is_one_atomic_undoable_edit_and_updates_comparison_baseline(expert_window):
    window = expert_window
    original, expert = window.project, window.expert_window
    original_strength = original.config.target_gradient_strength
    _edit(expert, text="4.12345678912345")
    assert expert.apply_changes()
    applied = window.project
    assert applied is not original and applied.revision == original.revision + 1
    assert applied.config.min_turn_distance == float("4.12345678912345") / 1000
    assert applied.config.target_gradient_strength == original_strength
    assert window.undo_stack.count() == 1
    assert expert.apply_changes()
    assert window.project is applied and window.undo_stack.count() == 1
    # A second Apply from the same window compares against its last successful
    # revision, rather than being rejected against the original opening state.
    _edit(expert, text="5")
    assert expert.apply_changes()
    assert window.project.config.min_turn_distance == pytest.approx(0.005)
    assert window.undo_stack.count() == 2
    window.undo_stack.undo()
    assert window.project is applied
    window.undo_stack.undo()
    assert window.project is original


@pytest.mark.parametrize("change", ("committed-model", "pending-main-draft"))
def test_concurrent_model_edit_rejects_expert_apply_and_retains_both_drafts(expert_window, change):
    window = expert_window
    expert = window.expert_window
    _edit(expert)
    draft = deepcopy(expert.draft)
    if change == "committed-model":
        assert window.apply_config({"levels": window.project.config.levels + 1})
    else:
        window._draft_controls["radius_mm"].setValue(91)
        assert window._draft_dirty
    baseline = window.project
    undo_count = window.undo_stack.count()
    assert not expert.apply_changes()
    assert window.project is baseline and window.undo_stack.count() == undo_count
    assert expert.draft == draft
    assert expert.fields["min_turn_distance"].control.text() == "4"
    assert "retained" in expert.error_label.text().casefold()
    assert "model changed" in window.messages.toPlainText().casefold()
    if change == "pending-main-draft":
        assert window._draft_controls["radius_mm"].value() == 91
        assert window._draft_dirty


def test_explicit_discard_and_reload_replaces_expert_baseline_with_current_model(expert_window, application):
    window = expert_window
    previous = window.expert_window
    _edit(previous, text="7")
    assert window.apply_config({"levels": 19, "min_turn_distance": 0.006})
    current = window.project
    undo_count = window.undo_stack.count()
    discard = next(
        button
        for button in previous.findChildren(QPushButton)
        if button.text() == "Discard expert draft & reload current model"
    )
    QTest.mouseClick(discard, Qt.MouseButton.LeftButton)
    application.processEvents()
    replacement = window.expert_window
    assert replacement is not previous
    assert replacement.draft["min_turn_distance"] == current.config.min_turn_distance
    assert replacement.draft["levels"] == 19
    assert float(replacement.fields["min_turn_distance"].control.text()) / 1000 == pytest.approx(
        current.config.min_turn_distance
    )
    assert not replacement.pending_changes
    assert window.project is current and window.undo_stack.count() == undo_count
    assert replacement.isVisible() and not replacement.isModal()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(previous)
    _edit(replacement, text="8")
    assert replacement.apply_changes()
    assert window.project.config.min_turn_distance == pytest.approx(0.008)
    assert window.project.config.levels == 19


def test_incomplete_expert_numeric_text_is_retained_on_close_and_reopen(expert_window):
    window = expert_window
    original, expert = window.project, window.expert_window
    _edit(expert, text="1e")
    assert not expert.apply_changes()
    expert.close()
    window._advanced_dialog()
    assert window.expert_window is expert
    assert expert.fields["min_turn_distance"].control.text() == "1e"
    assert window.project is original and window.undo_stack.count() == 0


@pytest.mark.parametrize("draft_kind", ("typed-valid", "typed-incomplete", "json-incomplete"))
@pytest.mark.parametrize("operation", ("single", "channels"))
def test_compute_reopens_unapplied_expert_draft_and_does_not_submit_old_values(expert_window, draft_kind, operation):
    window = expert_window
    original, expert = window.project, window.expert_window
    if draft_kind == "json-incomplete":
        expert.tabs.setCurrentIndex(1)
        expert.json_editor.setPlainText('{"min_turn_distance":')
    else:
        _edit(expert, text="4" if draft_kind == "typed-valid" else "1e")
    expert.close()
    if operation == "channels":

        def submit_assembly(*args):
            window.service.submissions.append(args)
            return {}

        window.service.submit_assembly = submit_assembly
        result = window.compute_channels()
    else:
        result = window.compute()
    assert result is None
    assert not window.service.submissions
    assert window.project is original and window.undo_stack.count() == 0
    assert window.expert_window is expert and expert.isVisible()
    if draft_kind == "json-incomplete":
        assert expert.json_editor.toPlainText() == '{"min_turn_distance":'
    else:
        assert expert.fields["min_turn_distance"].control.text() == ("4" if draft_kind == "typed-valid" else "1e")


def test_applied_expert_draft_allows_compute_with_new_values(expert_window):
    window = expert_window
    _edit(window.expert_window, text="4")
    assert window.expert_window.apply_changes()
    assert window.compute() is not None
    assert window.service.submissions[-1][0].config.min_turn_distance == pytest.approx(0.004)


def test_study_only_change_does_not_reject_expert_apply_or_overwrite_new_study(expert_window):
    window = expert_window
    original, expert = window.project, window.expert_window
    _edit(expert)
    assert window.apply_study({"pipeline_stage": "prepared", "fabricate": False, "validation_resolution": 19})
    changed = window.project
    study = deepcopy(changed.study)
    assert changed.model_fingerprint == original.model_fingerprint
    assert changed.fingerprint != original.fingerprint
    assert expert.apply_changes()
    assert window.project.config.min_turn_distance == pytest.approx(0.004)
    assert window.project.study == study
    assert window.project.study["pipeline_stage"] == "prepared"
    assert window.undo_stack.count() == 2


def test_single_json_save_applies_valid_expert_draft_to_owned_document(expert_window, tmp_path):
    window = expert_window
    expert = window.expert_window
    _edit(expert, text="4.12345678912345")
    path = tmp_path / "expert-model.coilgen.json"
    saved = window.save_project(path)
    assert saved == path and saved.is_file()
    restored = ProjectDocument.load(saved)
    assert restored.config.min_turn_distance == float("4.12345678912345") / 1000
    assert window.project.config.min_turn_distance == restored.config.min_turn_distance
    assert not expert.pending_changes
    assert window.undo_stack.count() == 1
    # Local GUI history is independent runtime state. The project itself must
    # load after copying just its single JSON to a fresh location.
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    copy = relocated / path.name
    copy.write_bytes(path.read_bytes())
    assert ProjectDocument.load(copy).config.min_turn_distance == restored.config.min_turn_distance


def test_python_export_companion_contains_valid_expert_draft(expert_window, tmp_path):
    window = expert_window
    _edit(window.expert_window, text="4.12345678912345")
    path = tmp_path / "expert-model.py"
    exported = window.export_python(path)
    assert exported == path and path.is_file()
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    restored = ProjectDocument.load(path.with_suffix(".coilgen.json"))
    assert restored.config.min_turn_distance == float("4.12345678912345") / 1000
    assert not window.expert_window.pending_changes
    assert window.undo_stack.count() == 1


@pytest.mark.parametrize("operation", ("save", "export"))
@pytest.mark.parametrize("failure", ("incomplete", "stale"))
def test_invalid_or_stale_expert_draft_prevents_document_output_and_is_retained(
    expert_window, tmp_path, operation, failure
):
    window = expert_window
    expert = window.expert_window
    if failure == "incomplete":
        _edit(expert, text="1e")
    else:
        _edit(expert, text="4")
        assert window.apply_config({"levels": window.project.config.levels + 1})
    baseline, undo_count = window.project, window.undo_stack.count()
    draft = expert.draft
    path = tmp_path / ("rejected.coilgen.json" if operation == "save" else "rejected.py")
    with pytest.raises(ValueError):
        (window.save_project if operation == "save" else window.export_python)(path)
    assert not path.exists() and not path.with_suffix(".coilgen.json").exists()
    assert window.project is baseline and window.undo_stack.count() == undo_count
    assert window.expert_window is expert and expert.draft == draft
    assert expert.pending_changes
    assert expert.fields["min_turn_distance"].control.text() == ("1e" if failure == "incomplete" else "4")


def test_canceling_main_window_close_keeps_owned_popups_and_drafts_open(expert_window, application):
    window = expert_window
    window.show()
    window._show_dock("settings_dock")
    window._open_evidence(window.history_panel)
    expert = window.expert_window
    _edit(expert, text="1e")
    focused = window._focused_sections[0]
    focused.open()
    history = window.evidence.window_for(window.history_panel)
    window._can_discard = lambda: False  # Equivalent to choosing Cancel in the close prompt.
    assert not window.close()
    application.processEvents()
    assert window.isVisible() and expert.isVisible()
    assert focused.dialog.isVisible() and history.isVisible()
    assert expert.fields["min_turn_distance"].control.text() == "1e"


def test_file_open_discard_removes_previous_owner_expert_and_retains_exact_new_inputs(
    expert_window, application, tmp_path, monkeypatch
):
    window = expert_window
    old_owner, old_expert = window.project, window.expert_window
    _edit(old_expert, text="1e")
    assert old_expert.pending_changes
    incoming = ProjectDocument.from_config(preset_config("biplanar"), name="Incoming Gy").update_study(
        preset_study("biplanar")
    )
    path = incoming.save(tmp_path / "incoming.json")
    prompts = []

    def discard(*args):
        prompts.append(args[1])
        return QMessageBox.StandardButton.Discard

    monkeypatch.setattr(QMessageBox, "question", discard)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(path), ""))
    window._open_dialog()
    _wait_for_project_open(application, window)
    assert prompts == ["Unsaved project changes"]
    assert window.project.id == incoming.id and window.project.id != old_owner.id
    assert window.project.fingerprint == incoming.fingerprint
    assert window.project.values == incoming.values
    assert window.expert_window is None
    assert not isValid(old_expert) or not old_expert.isVisible()
    assert window.undo_stack.isClean() and window.undo_stack.count() == 0
    assert window._check_expert_draft()
    saved = window.save_project(tmp_path / "new-owner-save.json")
    assert ProjectDocument.load(saved).fingerprint == incoming.fingerprint
    window._advanced_dialog()
    assert window.expert_window is not old_expert
    assert "Incoming Gy" in window.expert_window.windowTitle()
    assert window.expert_window.draft == incoming.values
    assert not window.expert_window.pending_changes
    assert window.service.submissions == []


@pytest.mark.parametrize("failure", ["missing", "malformed"])
def test_failed_file_open_preserves_previous_owner_expert_and_uncommitted_draft(
    expert_window, application, tmp_path, monkeypatch, failure
):
    window = expert_window
    owner, expert = window.project, window.expert_window
    _edit(expert, text="4.12345678912345")
    draft = deepcopy(expert.draft)
    title = expert.windowTitle()
    path = tmp_path / f"{failure}.json"
    if failure == "malformed":
        path.write_text('{"schema": invalid}', encoding="utf-8")
    errors = []
    monkeypatch.setattr(window, "_error", errors.append)
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Discard)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: (str(path), ""))
    window._open_dialog()
    _wait_for_project_open(application, window)
    assert window.project is owner and window.project_path is None
    assert window.expert_window is expert and isValid(expert) and expert.isVisible()
    assert expert.windowTitle() == title and expert.draft == draft and expert.pending_changes
    assert expert.fields["min_turn_distance"].control.text() == "4.12345678912345"
    assert window.undo_stack.count() == 0
    assert len(errors) == 1 and errors[0].startswith("Cannot open project:")
    assert window.service.submissions == []
