"""Source-qualified decisions and complete portable draft snapshots."""

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import test_workbench_usability as usability

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPlainTextEdit, QTableWidgetItem

from coilgen.application import ProjectDocument
from coilgen.application.design_studies import design_study_state
from coilgen.application.presets import preset_config
from coilgen.application.service import RunRecord
from coilgen.gui.saved_settings import SavedEvidenceError, SavedSettingsDialog, load_saved_request


application = usability.application
windows = usability.windows
prepared_run = usability.prepared_run


def test_saved_check_and_problem_actions_inspect_selected_record_without_editing_current(
    windows, tmp_path, monkeypatch
):
    window = windows()
    applied = window.project
    record = SimpleNamespace(directory=tmp_path / "other-channel", metadata={"project_id": "other-channel"}, arrays={})
    window.runs["saved"] = record
    window._selected_run = "saved"
    inspect = Mock()
    monkeypatch.setattr(window, "_inspect_saved_settings", inspect)
    window.checks_table.setRowCount(1)
    item = QTableWidgetItem("Field RMS error")
    item.setData(Qt.ItemDataRole.UserRole, "target")
    window.checks_table.setItem(0, 0, item)
    window._check_selected(0)
    inspect.assert_called_once_with(record, "target")
    window.problems_panel.navigate_requested.emit("winding")
    assert inspect.call_args.args == (record, "winding")
    assert window.project is applied and not window.service.submissions


@pytest.mark.parametrize(
    "stage,next_stage", [("prepared", "sheet"), ("sheet", "contours"), ("contours", "routed"), ("routed", "evaluated")]
)
def test_continue_advances_exactly_one_requested_boundary(windows, tmp_path, stage, next_stage):
    project = ProjectDocument.from_config(preset_config()).update_study({"pipeline_stage": stage, "fabricate": False})
    window = windows(project)
    record = RunRecord(
        tmp_path / "source",
        {
            "project_id": project.id,
            "project_revision": project.revision,
            "project_fingerprint": project.fingerprint,
            "pipeline_stage": stage,
            "summary": {"sheet_accepted": True},
        },
        {},
    )
    window.runs["source"] = record
    window.show_run("source")
    assert window.design_overview._state.action_key == "continue_full"
    assert "study complete" in window.design_status_label.text()
    assert window.project is project
    window.design_overview.action_button.click()
    submitted, controls = window.service.submissions[-1]
    assert submitted.study["pipeline_stage"] == next_stage
    assert controls["pipeline_stage"] == next_stage
    assert submitted.config.as_dict() == project.config.as_dict()


def test_hidden_advanced_draft_is_visible_and_resolves_before_compute(windows):
    window = windows()
    original = window.project
    window._advanced_dialog()
    expert = window.expert_window
    expert.fields["min_turn_distance"].control.setText("4")
    expert.close()
    assert window._has_input_drafts()
    assert "Advanced edits pending" in window.view_context.context.editing_text
    assert window.compute_button.text() == "Resolve Advanced draft"
    assert window.design_overview._state.action_key == "resolve_expert"
    assert window.isWindowModified()
    window.design_overview.action_button.click()
    assert not expert.isHidden()
    assert window.project is original and not window.service.submissions
    window._reload_expert()
    assert not window._has_input_drafts()
    assert "Advanced edits pending" not in window.view_context.context.editing_text
    assert window.compute_button.text() != "Resolve Advanced draft"


def test_resume_resolves_advanced_draft_before_applying_other_edits(windows):
    window = windows()
    original = window.project
    window._advanced_dialog()
    expert = window.expert_window
    expert.fields["min_turn_distance"].control.setText("4")
    expert.close()
    window._draft_controls["radius_mm"].setValue(174)
    assert window.resume_selected_run() is None
    assert window.project is original and window._draft_dirty
    assert not window.service.submissions and not expert.isHidden()
    window._reload_expert()


def test_unavailable_connected_wire_row_does_not_display_calculated_metrics(windows, tmp_path):
    window = windows()
    record = RunRecord(
        tmp_path / "run",
        {
            "wire_fields": {
                "routed": {
                    "available": False,
                    "reason": "No connected winding",
                    "current_A": 7.5,
                    "current_refitted": True,
                    "sample_count": 257,
                }
            },
        },
        {},
    )
    window.runs["source"] = record
    window.show_run("source")
    assert "unavailable" in window.wire_fields_table.item(0, 0).text()
    assert all(window.wire_fields_table.item(0, column).text() == "—" for column in range(1, 7))
    assert "No connected winding" in window.wire_fields_table.item(0, 7).text()


def test_python_export_includes_pending_history_notes_without_changing_physics(windows, tmp_path):
    window = windows()
    original = window.project
    window.history_panel.hypothesis.setPlainText("Refine the contour spacing after sheet validation")
    path = window.export_python(tmp_path / "replay.py")
    restored = ProjectDocument.load(path.with_suffix(".coilgen.json"))
    assert design_study_state(restored).hypothesis == "Refine the contour spacing after sheet validation"
    assert restored.fingerprint == original.fingerprint
    assert restored.revision == original.revision


def test_geometry_companion_includes_pending_history_notes(windows, tmp_path, monkeypatch):
    from coilgen.application import cad

    window = windows()
    record = SimpleNamespace(directory=tmp_path / "run", metadata={"project_id": window.project.id}, arrays={})
    window.runs["source"] = record
    window._selected_run = "source"
    window.history_panel.hypothesis.setPlainText("Preserve the fabrication decision with the geometry")
    monkeypatch.setattr(window, "_portable_assembly", lambda: window.assembly)
    export = Mock(return_value="package")
    monkeypatch.setattr(cad, "export_geometry", export)
    window.export_selected_geometry(tmp_path / "cad")
    companion = export.call_args.kwargs["project"]
    assert (
        design_study_state(companion.channels[0].project).hypothesis
        == "Preserve the fabrication decision with the geometry"
    )


def test_known_integrity_failure_blocks_geometry_export(windows, tmp_path, monkeypatch):
    from coilgen.application import cad

    window = windows()
    window.runs["source"] = SimpleNamespace(directory=tmp_path / "run", metadata={}, arrays={})
    window._selected_run = "source"
    window._job_verification_errors["source"] = "Artifact checksum changed"
    export = Mock()
    monkeypatch.setattr(cad, "export_geometry", export)
    with pytest.raises(ValueError, match="failed verification"):
        window.export_selected_geometry(tmp_path / "cad", include_project=False)
    export.assert_not_called()


def test_saved_settings_inspection_verifies_original_request_and_keeps_model_unchanged(
    windows, prepared_run, application
):
    project, record = prepared_run
    window = windows(project)
    original, undo_count = window.project, window.undo_stack.count()
    request = load_saved_request(record)
    assert request["project_id"] == project.id
    assert request["project_revision"] == project.revision
    window._inspect_saved_settings(record, "geometry")
    deadline = time.monotonic() + 10
    while window._saved_settings_task is not None and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert window._saved_settings_task is None
    dialogs = window.findChildren(SavedSettingsDialog)
    assert dialogs and all(editor.isReadOnly() for editor in dialogs[-1].findChildren(QPlainTextEdit))
    assert window.project is original and window.undo_stack.count() == undo_count
    dialogs[-1].close()


def test_saved_settings_refuses_request_changed_after_selection(prepared_run):
    _project, record = prepared_run
    path = record.directory / "request.json"
    payload = json.loads(path.read_bytes())
    payload["config"]["levels"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SavedEvidenceError, match="(integrity|checksum|changed|artifact|Artifact)"):
        load_saved_request(record)


def test_saved_settings_discovered_corruption_marks_all_aliases(windows, prepared_run, application):
    project, record = prepared_run
    window = windows(project)
    window.runs.update({"job": record, "history": record})
    window.show_run("history")
    path = record.directory / "request.json"
    path.write_bytes(path.read_bytes() + b" ")
    window._inspect_saved_settings(record, "target")
    deadline = time.monotonic() + 10
    while window._saved_settings_task is not None and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert window._saved_settings_task is None
    assert window._run_verification_error("job")
    assert window._run_verification_error("history")
    assert not window.acceptance_panel.settings.isEnabled()
    assert not window.findChildren(SavedSettingsDialog)
