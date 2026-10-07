"""A portable engineering decision loop over the real shared application API."""

from dataclasses import replace
import shutil
import time

import numpy as np
import pytest
from PySide6.QtCore import QThreadPool

import test_workbench_project_bundle as bundle_tests
from coilgen.application import DesignService, ProjectDocument
from coilgen.application.design_studies import design_study_state
from coilgen.application.presets import preset_config

application = bundle_tests.application


@pytest.fixture
def prepared(application, tmp_path):
    config = replace(preset_config(), progress_flag=False)
    project = ProjectDocument.from_config(config).update_study({"pipeline_stage": "prepared"})
    directory = tmp_path / "source" / "jobs" / "baseline"
    DesignService.run(project, directory, threads=1)
    window = bundle_tests._window(project, directory.parent)
    record = window.load_run(directory)
    window.history_panel.select_record(window._history_entries[0].id)
    yield window, project, record
    QThreadPool.globalInstance().waitForDone(5000)
    application.processEvents()
    window._draft_dirty = False
    bundle_tests._close(window)


def _wait_task(window, application):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        application.processEvents()
        if window._design_study_task is None:
            return
        # QtTest.qWait can retain the GIL and starve Python QRunnable workers.
        time.sleep(0.01)
    pytest.fail("Design action did not finish")


def test_pinned_baseline_notes_and_evidence_survive_single_json_relocation(prepared, application, tmp_path):
    window, original, record = prepared
    baseline_arrays = {key: values.copy() for key, values in record.arrays.items()}
    window.pin_design_baseline()
    window._save_design_notes("Increase contour levels; check realization error", "Retain as the prepared reference")
    _wait_task(window, application)
    state = design_study_state(window.project)
    assert state.baseline.matches(record)
    assert state.note_for(record).rationale == "Retain as the prepared reference"
    assert window.project.fingerprint == original.fingerprint
    assert window.project.revision == original.revision
    saved = window.save_project(tmp_path / "source" / "study.json")
    moved = tmp_path / "moved" / "study.json"
    moved.parent.mkdir()
    shutil.copyfile(saved, moved)
    # Reopening must succeed without the original inputs, outputs or local history.
    shutil.rmtree(tmp_path / "source")
    loaded = ProjectDocument.load(moved)
    assert loaded.session["design_study"] == window.project.session["design_study"]
    reopened = bundle_tests._window(loaded, tmp_path / "fresh" / "jobs")
    try:
        restored = reopened.runs[reopened._selected_run]
        assert design_study_state(reopened.project).baseline.matches(restored)
        assert "Baseline" in reopened.history_panel.baseline_label.text()
        for key, values in baseline_arrays.items():
            np.testing.assert_array_equal(restored.arrays[key], values)
            assert restored.arrays[key].dtype == values.dtype
        assert list(moved.parent.iterdir()) == [moved]
    finally:
        bundle_tests._close(reopened)


def test_variant_restores_recorded_model_as_undoable_same_owner_revision(prepared):
    window, original, record = prepared
    window.pin_design_baseline()
    assert window.apply_config({"levels": original.config.levels + 8})
    before = window.project
    variant = window.create_design_variant(hypothesis="Test support length before changing contour density")
    assert variant.id == original.id
    assert variant.revision == before.revision + 1
    assert variant.config.levels == original.config.levels
    assert design_study_state(variant).source.matches(record)
    assert len(window.assembly.channels) == 1
    window.undo_stack.undo()
    assert window.project is before
    window.undo_stack.redo()
    assert window.project is variant


def test_baseline_button_runs_verification_in_background(prepared, application):
    window, original, record = prepared
    entry = window._history_entries[0]
    window.history_panel.select_record(entry.id)
    assert window.history_panel.pin_baseline_button.isEnabled()
    window.history_panel.pin_baseline_button.click()
    assert window._design_study_task is not None
    assert not window.history_panel.isEnabled()
    _wait_task(window, application)
    assert window.history_panel.isEnabled()
    assert design_study_state(window.project).baseline.matches(record)
    assert window.project.fingerprint == original.fingerprint


def test_background_variant_does_not_overwrite_edits_made_during_verification(prepared, application):
    window, original, _record = prepared
    entry = window._history_entries[0]
    window._create_history_variant(entry, "Explore support length")
    assert window.apply_config({"levels": original.config.levels + 4})
    changed = window.project
    _wait_task(window, application)
    assert window.project is changed
    assert "was not applied" in window.messages.toPlainText()


def test_acceptance_view_opens_without_computing_and_pending_inputs_stay_distinct(prepared):
    window, _original, record = prepared
    window.navigate_workflow("review")
    assert window.evidence.currentWidget() is window.acceptance_panel
    assert window.acceptance_panel.report.checks
    assert (
        "Prepared" in window.acceptance_panel.identity.text()
        or record.metadata["name"] in window.acceptance_panel.identity.text()
    )
    window.select_node("winding")
    window._draft_controls["levels"].setValue(window.project.config.levels + 2)
    assert "pending" in window.design_brief.revision_label.text()
    assert (
        "pending" in window.acceptance_panel.report.message.lower()
        or "pending" in window.acceptance_panel.report.identity.lower()
    )


def test_save_project_commits_pending_design_notes_for_the_history_selection(prepared, tmp_path):
    window, _original, record = prepared
    window.history_panel.hypothesis.setPlainText("Investigate a longer support at unchanged field budget")
    window.history_panel.rationale.setPlainText("Reference for comparing support dimensions")
    assert window.history_panel.pending_notes is not None
    path = window.save_project(tmp_path / "with-notes.json")
    loaded = ProjectDocument.load(path)
    state = design_study_state(loaded)
    assert state.hypothesis == "Investigate a longer support at unchanged field budget"
    assert state.note_for(record).rationale == "Reference for comparing support dimensions"
    assert window.history_panel.pending_notes is None


def test_save_project_retains_dirty_decisions_for_unselected_candidates(prepared, tmp_path):
    window, original, first = prepared
    window.history_panel.rationale.setPlainText("Keep the original support as reference")
    candidate_project = original.update_config({"levels": original.config.levels + 2})
    directory = tmp_path / "source" / "jobs" / "candidate"
    DesignService.run(candidate_project, directory, threads=1)
    second = window.load_run(directory)
    second_entry = next(entry for entry in window._history_entries if entry.run.directory == second.directory)
    window.history_panel.select_record(second_entry.id)
    window.history_panel.rationale.setPlainText("Retain this alternative for the contour-density comparison")
    assert len(window.history_panel.pending_note_updates) == 2
    path = window.save_project(tmp_path / "all-decisions.json")
    loaded = ProjectDocument.load(path)
    state = design_study_state(loaded)
    assert state.note_for(first).rationale == "Keep the original support as reference"
    assert state.note_for(second).rationale == "Retain this alternative for the contour-density comparison"
    assert not window.history_panel.pending_note_updates
