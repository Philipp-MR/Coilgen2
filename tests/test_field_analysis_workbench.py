"""Field analysis binds saved evidence and keeps result analysis out of model edits."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen.application import DesignService, ProjectDocument, RunRecord
from coilgen.config import CoilGenConfig
from coilgen.target import evaluate_field_expression
from coilgen.field_units import field_values_to_tesla
from test_workbench_usability import application as application, windows as windows


def native_config():
    return CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 8, 4, 1, 0, 0, 0),
        field_shape_function="2*x",
        target_gradient_strength=0.03,
        target_region_radius=0.02,
        target_region_resolution=3,
        symmetry_reduction="off",
        gauss_order=3,
        progress_flag=False,
        levels=3,
        min_point_loop_number=3,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
    )


@pytest.mark.parametrize("stage", ["prepared", "evaluated"])
def test_executed_target_recipe_replays_original_samples_without_renormalizing(tmp_path, stage):
    project = ProjectDocument.from_config(native_config()).update_study({"pipeline_stage": stage})
    DesignService.run(project, tmp_path / stage)
    saved = DesignService.load_run(tmp_path / stage)
    recipe = saved.metadata["target_recipe"]
    points = saved.array("target_points_m")
    expected = field_values_to_tesla(
        evaluate_field_expression(recipe["field_shape_function"], points) * recipe["normalization_scale"],
        recipe["target_field_units"],
    )
    np.testing.assert_array_equal(expected, saved.array("target_field_T")[:, 2])
    assert recipe["field_component"] == "Bz" and recipe["coordinate_frame"] == "world"


def test_imported_target_never_gets_inert_native_recipe():
    from types import SimpleNamespace
    from coilgen.research.application import _target_analysis_recipe
    from coilgen.research.requests import ResearchRequest

    config = replace(native_config(), target_field_definition_file="unrelated.csv")
    request = ResearchRequest.from_config(config)
    target = SimpleNamespace(target_diagnostics={"target_gradient_normalization_scale": 7.0})
    assert not _target_analysis_recipe(request, target)


def test_analysis_action_requires_computed_healthy_evidence_including_real_zero_fields(windows, tmp_path):
    window = windows()
    arrays = {"target_points_m": np.zeros((2, 3)), "target_field_T": np.zeros((2, 3))}
    record = RunRecord(tmp_path / "target-only", {}, arrays)
    window.runs["source"] = record
    window._selected_run = "source"
    window._update_execution_feedback()
    assert not window.field_analysis_action.isEnabled()
    record = RunRecord(tmp_path / "computed-zero", {}, {**arrays, "sheet_field_T": np.zeros((2, 3))})
    window.runs["source"] = record
    window._update_execution_feedback()
    assert window.field_analysis_action.isEnabled()
    window._source_health.mark_failed(record.directory, "Modified scientific result")
    window._update_execution_feedback()
    assert not window.field_analysis_action.isEnabled()


@pytest.mark.parametrize(
    "current_policy, expected_source",
    [("Stored evaluation", "routed_field"), ("Original contour current", "routed_at_stream_current")],
)
def test_analysis_verifies_pinned_snapshot_without_applying_model_draft(
    windows, tmp_path, monkeypatch, current_policy, expected_source
):
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QDialog
    import coilgen.gui.field_analysis as field_gui

    class Inspector(QDialog):
        busyChanged = Signal(bool)
        progressChanged = Signal(str)
        closed = Signal()

        def __init__(self, record, parent=None, **options):
            super().__init__(parent)
            self.record, self.options = record, options

        def request_close(self):
            self.closed.emit()
            self.close()

    window = windows()
    original = window.project
    record = RunRecord(tmp_path / "saved", {"name": "Pinned", "result": "original"}, {})
    window.runs["source"] = record
    window._selected_run = "source"
    window.viewport.state["dataset"] = "routed_field"
    window.viewport.state["wire_current"] = current_policy
    window._draft_controls["radius_mm"].setValue(174)
    captured = []
    monkeypatch.setattr(field_gui, "FieldAnalysisDialog", Inspector)
    monkeypatch.setattr(
        window,
        "_gradient_task",
        lambda operation, callback, label, **opts: captured.append((operation, callback, opts)),
    )
    # Volatile job progress is intentionally outside scientific snapshot identity.
    refreshed = RunRecord(record.directory, {**record.metadata, "job": {"progress": 1.0}}, {})
    monkeypatch.setattr(DesignService, "load_run", lambda _path: refreshed)
    window._field_analysis_dialog()
    operation, callback, options = captured.pop()
    assert options["retain_receipt"]
    window._selected_run = None
    callback(operation(original))
    dialog = next(iter(window._field_analysis_windows.values()))
    assert dialog.record is refreshed
    assert dialog.options["initial_source"] == expected_source
    assert window.project is original and window._draft_dirty
    assert window.undo_stack.count() == 0
    callback(refreshed)
    assert len(window._field_analysis_windows) == 1  # An unchanged snapshot reuses its inspector.
    replacement = RunRecord(record.directory, {"name": "Pinned", "result": "replacement"}, {})
    window.runs["source"] = replacement
    window._selected_run = "source"
    monkeypatch.setattr(DesignService, "load_run", lambda _path: replacement)
    window._field_analysis_dialog()
    new_operation, new_callback, _ = captured.pop()
    new_callback(new_operation(original))
    assert len(window._field_analysis_windows) == 2
    assert {item.record.metadata["result"] for item in window._field_analysis_windows.values()} == {
        "original",
        "replacement",
    }
    dialog.busyChanged.emit(True)
    assert not window.computation_indicator.isHidden()
    dialog.busyChanged.emit(False)
    assert window.computation_indicator.isHidden()
    for inspector in tuple(window._field_analysis_windows.values()):
        inspector.request_close()
    assert not window._field_analysis_windows


def test_analysis_rejects_selected_snapshot_replaced_on_disk(windows, tmp_path, monkeypatch):
    window = windows()
    record = RunRecord(tmp_path / "replaced", {"name": "Before", "config": {"levels": 10}}, {})
    window.runs["source"] = record
    window._selected_run = "source"
    captured = []
    monkeypatch.setattr(window, "_gradient_task", lambda operation, *_args, **_kwargs: captured.append(operation))
    replacement = RunRecord(record.directory, {"name": "After", "config": {"levels": 11}}, {})
    monkeypatch.setattr(DesignService, "load_run", lambda _path: replacement)
    window._field_analysis_dialog()
    with pytest.raises(ValueError, match="changed on disk"):
        captured[0](window.project)
    assert not getattr(window, "_field_analysis_windows", {})
