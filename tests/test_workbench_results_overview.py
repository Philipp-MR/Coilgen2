"""Prominent result actions retain the saved physical scope and model drafts."""

import time
from threading import Event, get_ident

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, Signal, Qt
from PySide6.QtWidgets import QDialog

from coilgen.application import DesignService, RunRecord
from test_workbench_usability import application as application, windows as windows


def computed_record(window, directory, *, component="Bz"):
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])
    target = np.zeros((2, 3))
    target[:, ("Bx", "By", "Bz").index(component)] = [1e-6, 2e-6]
    error = np.zeros((2, 3))
    error[:, ("Bx", "By", "Bz").index(component)] = 0.1e-6
    return RunRecord(
        directory,
        {
            "name": "Recorded shim",
            "project_id": window.project.id,
            "project_fingerprint": window.project.fingerprint,
            "project_model_fingerprint": window.project.model_fingerprint,
            "project_revision": window.project.revision,
            "config": {"target_field_component": component},
            "summary": {"accepted": True, "construction_released": False},
            "wire_fields": {
                "routed": {
                    "available": True,
                    "current_A": 8.0,
                    "current_refitted": True,
                    "component": component,
                    "rms_error_T": 1e-8,
                    "at_stream_current": {
                        "available": True,
                        "current_A": 3.0,
                        "current_refitted": False,
                        "component": component,
                        "rms_error_T": 1e-7,
                        "peak_error_T": 1e-7,
                        "relative_rms_error": 0.0632455532,
                        "sample_count": 2,
                    },
                }
            },
        },
        {
            "target_points_m": points,
            "target_field_T": target,
            "routed_field_T": target + error / 10,
            "routed_field_at_stream_current_T": target + error,
        },
    )


def select_record(window, record):
    window.runs["result"] = record
    window.show_run("result")


def test_results_appear_after_selection_with_direct_toolbar_and_tree_access(windows, tmp_path):
    window = windows()
    assert window.results_overview.isHidden()
    assert not window.field_results_button.isEnabled()
    record = computed_record(window, tmp_path / "saved")
    select_record(window, record)
    assert not window.results_overview.isHidden()
    assert window.workspace.isAncestorOf(window.results_overview)
    assert window.field_results_button.isEnabled()
    assert window.results_overview.source_key == "routed_at_stream_current"
    assert window.result_tree_items["metrics"].parent() is window.tree_items["results"]
    assert all(not item.isDisabled() for item in window.result_tree_items.values())


def test_field_map_uses_original_current_and_recorded_component_without_changing_evidence(windows, tmp_path):
    window = windows()
    record = computed_record(window, tmp_path / "saved", component="Bx")
    select_record(window, record)
    before = record.arrays
    project = window.project
    window._results_action("field")
    assert window.viewport.state == {
        "dataset": "routed_residual",
        "component": "Bx",
        "units": "µT",
        "wire_current": "Original contour current",
    }
    assert window.project is project
    assert window.undo_stack.count() == 0
    for key, value in before.items():
        np.testing.assert_array_equal(record.array(key), value)


def test_reloaded_snapshot_refreshes_without_comparing_scientific_arrays(windows, tmp_path):
    window = windows()
    original = computed_record(window, tmp_path / "saved")
    select_record(window, original)
    replacement = RunRecord(original.directory, original.metadata, original.arrays)
    select_record(window, replacement)
    assert window.runs["result"] is replacement
    assert window.results_overview.source_key == "routed_at_stream_current"
    assert window._results_overview_key[0] == id(replacement)


def test_results_actions_keep_pending_drafts_and_open_selected_page(windows, tmp_path, monkeypatch):
    window = windows()
    record = computed_record(window, tmp_path / "saved")
    select_record(window, record)
    window._draft_controls["radius_mm"].setValue(174)
    original = window.project
    requested = []
    monkeypatch.setattr(window, "_field_analysis_dialog", lambda **options: requested.append(options))
    window.field_results_button.click()
    item = window.result_tree_items["slices"]
    window.tree.setCurrentItem(item)
    window.tree.itemClicked.emit(item, 0)
    assert requested == [
        {"view": "metrics", "source": "routed_at_stream_current"},
        {"view": "slices", "source": "routed_at_stream_current"},
    ]
    assert window.project is original and window._draft_dirty
    assert window._draft_controls["radius_mm"].value() == 174
    assert window.undo_stack.count() == 0
    assert item.data(0, Qt.ItemDataRole.UserRole) == "result:slices"


def test_modified_evidence_disables_prominent_field_actions(windows, tmp_path):
    window = windows()
    record = computed_record(window, tmp_path / "saved")
    select_record(window, record)
    window._source_health.mark_failed(record.directory, "Scientific array checksum changed")
    window._update_execution_feedback()
    assert not window.field_results_button.isEnabled()
    assert not window.field_analysis_action.isEnabled()
    assert all(item.isDisabled() for item in window.result_tree_items.values())
    assert not window.results_overview.isHidden()


def test_target_only_result_does_not_enable_field_analysis(windows, tmp_path):
    window = windows()
    record = RunRecord(
        tmp_path / "prepared",
        {},
        {
            "target_points_m": np.zeros((2, 3)),
            "target_field_T": np.zeros((2, 3)),
        },
    )
    select_record(window, record)
    assert not window.field_results_button.isEnabled()
    assert window.result_tree_items["metrics"].isDisabled()


def test_coil_properties_remain_accessible_without_field_samples(windows, tmp_path, monkeypatch):
    window = windows()
    record = RunRecord(
        tmp_path / "geometry",
        {
            "name": "Saved geometry",
            "summary": {"resistance_ohm": 0.25, "wire_length_m": 2.0},
        },
        {},
    )
    select_record(window, record)
    requested = []
    monkeypatch.setattr(window, "_field_analysis_dialog", lambda **options: requested.append(options))
    assert not window.field_analysis_action.isEnabled()
    assert window.field_results_button.isEnabled()
    assert not window.results_overview.isHidden()
    window.field_results_button.click()
    assert requested == [{"view": "coil"}]


def test_projection_field_uses_analysis_without_invalid_viewport_selection(windows, tmp_path, monkeypatch):
    window = windows()
    original = computed_record(window, tmp_path / "saved")
    metadata = original.metadata
    metadata["config"]["target_field_component"] = "BdotB0"
    metadata["wire_fields"]["routed"]["component"] = "BdotB0"
    metadata["wire_fields"]["routed"]["at_stream_current"]["component"] = "BdotB0"
    select_record(window, RunRecord(original.directory, metadata, original.arrays))
    before = dict(window.viewport.state)
    requested = []
    monkeypatch.setattr(window, "_field_analysis_dialog", lambda **options: requested.append(options))
    window._results_action("field")
    assert window.viewport.state == before
    assert requested == [{"view": "field", "source": "routed_at_stream_current"}]


class Inspector(QDialog):
    busyChanged = Signal(bool)
    progressChanged = Signal(str)
    progressDetails = Signal(object)
    closed = Signal()

    def __init__(self, record, parent=None, **options):
        super().__init__(parent)
        self.record, self.options = record, options
        self.focused = []

    def focus_view(self, view, **options):
        self.focused.append((view, options))

    def request_close(self):
        self.closed.emit()
        self.close()


def test_rapid_page_choices_share_verification_and_last_page_wins(windows, tmp_path, monkeypatch):
    import coilgen.gui.field_analysis as field_gui

    window = windows()
    record = computed_record(window, tmp_path / "saved")
    select_record(window, record)
    pending = []
    monkeypatch.setattr(field_gui, "FieldAnalysisDialog", Inspector)
    monkeypatch.setattr(
        window, "_gradient_task", lambda operation, callback, *_args, **_options: pending.append(callback)
    )
    window._results_action("metrics")
    window._results_action("slices")
    assert len(pending) == 1
    pending.pop()(record)
    dialog = next(iter(window._field_analysis_windows.values()))
    assert dialog.options == {"initial_source": "routed_at_stream_current", "initial_view": "slices"}
    window._results_action("errors")
    pending.pop()(record)
    assert next(iter(window._field_analysis_windows.values())) is dialog
    assert dialog.focused == [("errors", {"source": "routed_at_stream_current"})]
    window._field_analysis_dialog(view="metrics")
    pending.pop()(record)
    assert dialog.focused[-1] == ("metrics", {})


@pytest.mark.parametrize("old_failure", [True, False])
def test_old_analysis_verification_cannot_override_newer_source_health(windows, tmp_path, monkeypatch, old_failure):
    import coilgen.gui.field_analysis as field_gui

    window = windows()
    record = computed_record(window, tmp_path / "saved")
    select_record(window, record)
    release = Event()

    def load(_directory):
        assert release.wait(5)
        if old_failure:
            raise ValueError("Obsolete checksum failure")
        return record

    monkeypatch.setattr(DesignService, "load_run", load)
    monkeypatch.setattr(field_gui, "FieldAnalysisDialog", Inspector)
    window._results_action("slices")
    if old_failure:
        window._clear_run_verification_error(record)
    else:
        window._mark_run_verification_error("result", "Newer checksum failure", directory=record.directory)
    generation = window._source_verification_generations[record.directory.resolve()]
    release.set()
    deadline = time.monotonic() + 5
    while window._field_analysis_verifications and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)
    assert not window._field_analysis_verifications
    assert not window._field_analysis_requests
    assert not getattr(window, "_field_analysis_windows", {})
    assert window._source_verification_generations[record.directory.resolve()] == generation
    assert window._run_verification_error(record) == ("" if old_failure else "Newer checksum failure")


def test_failed_analysis_verification_releases_request_and_blocks_aliases(windows, tmp_path, monkeypatch):
    window = windows()
    record = computed_record(window, tmp_path / "saved")
    select_record(window, record)
    window.runs["alias"] = record

    def reject(_directory):
        raise ValueError("Scientific array checksum changed")

    monkeypatch.setattr(DesignService, "load_run", reject)
    window._results_action("slices")
    deadline = time.monotonic() + 5
    while window._field_analysis_requests and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)
    assert not window._field_analysis_requests
    assert "checksum changed" in window._run_verification_error("alias")
    assert not window.field_results_button.isEnabled()
    assert not getattr(window, "_field_analysis_windows", {})


def test_analysis_snapshot_hashing_runs_outside_gui_thread(windows, tmp_path, monkeypatch):
    import coilgen.gui.field_analysis as field_gui
    import coilgen.gui.workbench as workbench_gui

    window = windows()
    record = computed_record(window, tmp_path / "saved")
    select_record(window, record)
    main_thread = get_ident()
    digest = workbench_gui.result_metadata_digest
    observed = []

    def check_thread(metadata):
        observed.append(get_ident())
        assert get_ident() != main_thread
        return digest(metadata)

    monkeypatch.setattr(workbench_gui, "result_metadata_digest", check_thread)
    monkeypatch.setattr(DesignService, "load_run", lambda _directory: record)
    monkeypatch.setattr(field_gui, "FieldAnalysisDialog", Inspector)
    window._results_action("coil")
    deadline = time.monotonic() + 5
    while not getattr(window, "_field_analysis_windows", {}) and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)
    assert len(observed) == 2
    assert next(iter(window._field_analysis_windows.values())).options["initial_view"] == "coil"
