"""Native history panel signals, visible broken links and scope-aware tables."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication

from coilgen.application.history import RunHistoryStore
from coilgen.application.design_studies import CandidateNote, DesignStudyState, RunReference
from coilgen.application.presets import preset_config
from coilgen.application.project import ProjectDocument
from coilgen.application.service import DesignService
from coilgen.gui.history import HistoryPanel
from coilgen.gui.history import _parameter
from test_run_history import write_run


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication(["history-test"])


def _select_two(panel):
    selection = panel.runs_table.selectionModel()
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    selection.select(panel.runs_table.model().index(0, 0), flags)
    selection.select(panel.runs_table.model().index(1, 0), flags)


def test_named_history_selection_alias_signal_and_verified_comparison(application, tmp_path):
    first, second = write_run(tmp_path / "a"), write_run(tmp_path / "b", config={"levels": 12}, refitted=True)
    store = RunHistoryStore(tmp_path / "session.json")
    linked = store.add_run("p", first, alias="Baseline")
    store.add_run("p", second, alias="More turns")
    panel = HistoryPanel()
    selected, renamed, compared = [], [], []
    panel.selected.connect(selected.append)
    panel.alias_changed.connect(lambda id_, alias: renamed.append((id_, alias)))
    panel.compare_requested.connect(lambda left, right: compared.append((left, right)))
    panel.set_records(store.restore("p"))
    assert panel.select_record(linked.id)
    assert selected[-1].id == linked.id
    panel.alias.setText("Original fit")
    panel.rename_button.click()
    assert renamed == [(linked.id, "Original fit")]
    assert first.metadata["name"] == "Original scientific name"
    panel.runs_table.clearSelection()
    _select_two(panel)
    assert panel.compare_button.isEnabled()
    comparison = panel.compare_selected()
    assert comparison is not None and len(compared) == 1
    assert compared[0][0].directory == first.directory
    assert compared[0][1].directory == second.directory
    assert panel.parameters_table.item(0, 0).text() == "Contour level count"
    assert "config.levels" in panel.parameters_table.item(0, 0).toolTip()
    row = next(index for index, metric in enumerate(comparison.metrics) if metric.name == "Contour RMS error")
    assert panel.metrics_table.item(row, 4).text() == "Separate scopes"
    assert "refitted=True" in panel.metrics_table.item(row, 4).toolTip()
    assert "do not claim independent acceptance" in panel.comparison_notes.text()
    panel.set_records(store.restore("p"))
    assert panel.comparison is not None and panel.metrics_table.rowCount() > 0
    assert len(compared) == 1  # Polling preserves evidence without repeating user actions.
    panel.close()


def test_pinned_baseline_actions_notes_and_comparison_are_owned_and_persistent(application, tmp_path):
    first = write_run(tmp_path / "a")
    second = write_run(tmp_path / "b", config={"conductor_cross_section_width": 0.003})
    store = RunHistoryStore(tmp_path / "session.json")
    baseline = store.add_run("source-project", first, alias="Reference")
    candidate = store.add_run("source-project", second, alias="Wider wire")
    panel = HistoryPanel()
    panel.set_records(store.restore("source-project"))
    state = DesignStudyState(
        baseline=RunReference.from_metadata(first.metadata),
        hypothesis="Test wider wire clearance",
        notes=(CandidateNote(RunReference.from_metadata(second.metadata), "Wider conductor", "Needs bend review"),),
    )
    panel.set_study_state(state, project_id="source-project")
    pinned, unpinned, variants, notes = [], [], [], []
    panel.baseline_requested.connect(pinned.append)
    panel.clear_baseline_requested.connect(lambda: unpinned.append(True))
    panel.variant_requested.connect(lambda entry, hypothesis: variants.append((entry.id, hypothesis)))
    panel.notes_changed.connect(lambda hypothesis, rationale: notes.append((hypothesis, rationale)))
    assert panel.select_record(candidate.id)
    assert panel.rationale.toPlainText() == "Needs bend review"
    assert panel.pin_baseline_button.isEnabled()
    assert panel.compare_baseline_button.isEnabled()
    assert panel.create_variant_button.isEnabled()
    assert "Reference" in panel.baseline_label.text()
    assert panel.runs_table.item(0, 0).text() == "Baseline · Reference"
    panel.pin_baseline_button.click()
    panel.create_variant_button.click()
    panel.save_notes_button.click()
    assert pinned[-1].id == candidate.id
    assert variants == [(candidate.id, "Test wider wire clearance")]
    assert notes == [("Test wider wire clearance", "Needs bend review")]
    comparison = panel.compare_to_baseline()
    assert comparison.left.directory == first.directory
    assert comparison.right.directory == second.directory
    assert "B − A" in panel.comparison_notes.text()
    row = next(
        index for index, metric in enumerate(comparison.metrics) if metric.name == "Sheet independent relative error"
    )
    assert panel.metrics_table.item(row, 5).text() == "0"
    delta_row = next(
        index
        for index, delta in enumerate(comparison.parameter_deltas)
        if delta.field == "config.conductor_cross_section_width"
    )
    assert panel.parameters_table.item(delta_row, 0).text() == "Conductor width"
    assert panel.parameters_table.item(delta_row, 2).text() == "3 mm"
    panel.set_records(store.restore("source-project"))
    assert panel.selected_entries[0].id == candidate.id
    assert panel.comparison is not None
    assert "Baseline ·" in panel.runs_table.item(0, 0).text()
    panel.clear_baseline_button.click()
    assert unpinned == [True]
    assert panel.select_record(baseline.id)
    assert not panel.compare_baseline_button.isEnabled()
    panel.close()


def test_incomparable_metric_has_no_delta_and_missing_comparison_is_cleared(application, tmp_path):
    first, second = write_run(tmp_path / "a"), write_run(tmp_path / "b", refitted=True)
    store = RunHistoryStore(tmp_path / "session.json")
    store.add_run("p", first)
    store.add_run("p", second)
    panel = HistoryPanel()
    panel.set_records(store.restore("p"))
    _select_two(panel)
    comparison = panel.compare_selected()
    row = next(index for index, metric in enumerate(comparison.metrics) if metric.name == "Contour RMS error")
    assert panel.metrics_table.item(row, 5).text() == "Unavailable"
    (second.directory / "result.json").unlink()
    panel.set_records(store.restore("p"))
    assert panel.comparison is None
    assert panel.metrics_table.rowCount() == 0
    assert "Previous comparison is unavailable" in panel.comparison_notes.text()
    panel.close()


def test_foreign_run_can_be_inspected_but_cannot_be_restored_or_selected_as_baseline(application, tmp_path):
    record = write_run(tmp_path / "run")
    store = RunHistoryStore(tmp_path / "session.json")
    linked = store.add_run("another-project", record)
    panel = HistoryPanel()
    panel.set_study_state(DesignStudyState(), project_id="another-project")
    panel.set_records(store.restore("another-project"))
    assert panel.select_record(linked.id)
    assert not panel.pin_baseline_button.isEnabled()
    assert not panel.create_variant_button.isEnabled()
    assert not panel.rationale.isEnabled()
    panel.close()


def test_missing_baseline_is_explicit_and_notes_length_is_bounded(application, tmp_path):
    record = write_run(tmp_path / "run")
    panel = HistoryPanel()
    panel.set_study_state(
        DesignStudyState(baseline=RunReference.from_metadata(record.metadata)), project_id="source-project"
    )
    assert "Pinned baseline is unavailable" in panel.baseline_label.text()
    assert panel.clear_baseline_button.isEnabled()
    assert not panel.compare_baseline_button.isEnabled()
    panel.hypothesis.setPlainText("x" * 4001)
    assert not panel.save_notes_button.isEnabled()
    panel.close()


def test_unsaved_design_notes_survive_polling_selection_changes_and_baseline_updates(application, tmp_path):
    first, second = write_run(tmp_path / "a"), write_run(tmp_path / "b", revision=2)
    store = RunHistoryStore(tmp_path / "session.json")
    a = store.add_run("source-project", first)
    b = store.add_run("source-project", second)
    panel = HistoryPanel()
    panel.set_study_state(DesignStudyState(hypothesis="Initial hypothesis"), project_id="source-project")
    panel.set_records(store.restore("source-project"))
    panel.select_record(a.id)
    panel.hypothesis.setPlainText("Try different cut placement")
    panel.rationale.setPlainText("Keep this run until bend clearance is understood")
    panel.comparison_tabs.setCurrentIndex(2)
    panel.set_records(store.restore("source-project"))
    panel.set_study_state(
        DesignStudyState(baseline=RunReference.from_metadata(first.metadata), hypothesis="Initial hypothesis"),
        project_id="source-project",
    )
    assert panel.hypothesis.toPlainText() == "Try different cut placement"
    assert panel.rationale.toPlainText() == "Keep this run until bend clearance is understood"
    assert panel.notes_status.text() == "Unsaved design notes"
    hypothesis, rationale, selected = panel.pending_notes
    assert hypothesis == "Try different cut placement"
    assert rationale == "Keep this run until bend clearance is understood"
    assert selected.id == a.id
    assert panel.owned_selection.id == a.id
    assert panel.comparison_tabs.currentIndex() == 2
    panel.select_record(b.id)
    assert panel.rationale.toPlainText() == ""
    panel.rationale.setPlainText("Inspect this candidate separately")
    panel.select_record(a.id)
    assert panel.rationale.toPlainText() == "Keep this run until bend clearance is understood"
    assert {entry.id: rationale for entry, rationale in panel.pending_note_updates} == {
        a.id: "Keep this run until bend clearance is understood",
        b.id: "Inspect this candidate separately",
    }
    saved = DesignStudyState(
        baseline=RunReference.from_metadata(first.metadata),
        hypothesis="Try different cut placement",
        notes=(
            CandidateNote(
                RunReference.from_metadata(first.metadata),
                "Try different cut placement",
                "Keep this run until bend clearance is understood",
            ),
        ),
    )
    panel.set_study_state(saved, project_id="source-project")
    assert panel.notes_status.text() == "Unsaved decisions for 1 other run(s)"
    assert panel.pending_notes is None
    assert {entry.id: rationale for entry, rationale in panel.pending_note_updates} == {
        b.id: "Inspect this candidate separately",
    }
    panel.select_record(b.id)
    assert panel.rationale.toPlainText() == "Inspect this candidate separately"
    panel.close()


def test_refresh_preserves_comparison_tab_and_notes_do_not_leak_between_projects(application, tmp_path):
    first, second = write_run(tmp_path / "a"), write_run(tmp_path / "b")
    store = RunHistoryStore(tmp_path / "session.json")
    store.add_run("source-project", first)
    store.add_run("source-project", second)
    panel = HistoryPanel()
    panel.set_study_state(DesignStudyState(), project_id="source-project")
    panel.set_records(store.restore("source-project"))
    _select_two(panel)
    panel.compare_selected()
    panel.comparison_tabs.setCurrentWidget(panel.parameters_table)
    panel.set_records(store.restore("source-project"))
    assert panel.comparison_tabs.currentWidget() is panel.parameters_table
    panel.hypothesis.setPlainText("Unsaved source project experiment")
    panel.set_study_state(DesignStudyState(hypothesis="Independent project intent"), project_id="other-project")
    assert panel.hypothesis.toPlainText() == "Independent project intent"
    assert panel.pending_note_updates == ()
    panel.set_study_state(DesignStudyState(), project_id="source-project")
    assert panel.hypothesis.toPlainText() == "Unsaved source project experiment"
    panel.close()


def test_physical_geometry_difference_uses_actual_cylinder_length():
    label, value = _parameter("config.cylinder_mesh_parameter_list", [0.21, 0.08, 24, 20, 1, 0, 0, 0])
    assert label == "Cylinder support"
    assert "Radius: 80 mm" in value
    assert "Length: 200 mm" in value
    assert "Angular divisions: 24" in value
    assert "Rotation angle: 0 °" in value


def test_pending_decision_is_preserved_when_its_disk_evidence_becomes_unavailable(application, tmp_path):
    record = write_run(tmp_path / "run")
    store = RunHistoryStore(tmp_path / "session.json")
    linked = store.add_run("source-project", record)
    panel = HistoryPanel()
    panel.set_study_state(DesignStudyState(), project_id="source-project")
    panel.set_records(store.restore("source-project"))
    panel.select_record(linked.id)
    panel.rationale.setPlainText("This candidate requires a clearance check")
    (record.directory / "scientific_result.npz").write_bytes(b"changed")
    panel.set_records(store.restore("source-project"))
    updates = panel.pending_note_updates
    assert len(updates) == 1
    assert updates[0][0].id == linked.id
    assert updates[0][1] == "This candidate requires a clearance check"
    panel.close()


def test_missing_result_link_is_visible_and_disabled_for_comparison(application, tmp_path):
    first, second = write_run(tmp_path / "a"), write_run(tmp_path / "b")
    store = RunHistoryStore(tmp_path / "session.json")
    store.add_run("p", first)
    store.add_run("p", second)
    (second.directory / "result.json").unlink()
    entries = store.restore("p")
    panel = HistoryPanel()
    panel.set_records(entries)
    assert panel.runs_table.rowCount() == 2
    assert not panel.diagnostics.isHidden()
    assert "result.json" in panel.diagnostics.text()
    assert panel.runs_table.item(1, 6).text() == "Unavailable"
    _select_two(panel)
    assert not panel.compare_button.isEnabled()
    assert panel.compare_selected() is None
    panel.close()


def test_comparison_reports_later_artifact_corruption_and_clears_old_values(application, tmp_path):
    first, second = write_run(tmp_path / "a"), write_run(tmp_path / "b")
    store = RunHistoryStore(tmp_path / "session.json")
    store.add_run("p", first)
    store.add_run("p", second)
    panel = HistoryPanel()
    panel.set_records(store.restore("p"))
    _select_two(panel)
    assert panel.compare_selected() is not None
    (second.directory / "scientific_result.npz").write_bytes(b"corrupt")
    assert panel.compare_selected() is None
    assert panel.metrics_table.rowCount() == 0
    assert "integrity" in panel.comparison_notes.text()
    panel.close()


def test_real_prepared_checkpoint_is_completion_without_acceptance_failure(application, tmp_path):
    project = ProjectDocument.new("Prepared support", preset_config()).update_study({"pipeline_stage": "prepared"})
    DesignService.run(project, tmp_path / "run")
    store = RunHistoryStore(tmp_path / "session.json")
    entry = store.add_run(project.id, tmp_path / "run")
    assert entry.run.metadata["summary"]["accepted"] is False
    panel = HistoryPanel()
    panel.set_records(store.restore(project.id))
    assert panel.runs_table.item(0, 1).text() == "prepared"
    assert panel.runs_table.item(0, 3).text() == "Unverified"
    panel.close()
