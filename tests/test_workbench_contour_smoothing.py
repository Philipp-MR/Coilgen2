"""Pre-interconnection smoothing uses the shared config, SI values and draft history."""

from dataclasses import replace

import pytest
import test_workbench_usability as usability

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.gui.expert_settings import expert_fields


application = usability.application
windows = usability.windows


def test_smoothing_is_opt_in_and_opening_winding_does_not_modify_inputs(windows):
    window = windows()
    original = window.project
    window.select_node("winding")
    controls = window._draft_controls
    section = window._settings_decisions["Contour smoothing"]
    assert controls["contour_path_regularization_mode"].currentText() == "off"
    assert section.summary_label.text() == "Keep original contours"
    for name in ("contour_smoothing_mesh_gain", "contour_smoothing_factor", "contour_segment_mm"):
        assert not section.form.isRowVisible(controls[name].parentWidget())
    assert window.apply_settings()
    assert window.project is original and window.undo_stack.count() == 0


def test_enabling_surface_smoothing_commits_only_requested_fields_and_can_undo(windows):
    window = windows()
    window.select_node("winding")
    original = window.project
    controls = window._draft_controls
    section = window._settings_decisions["Contour smoothing"]
    controls["contour_path_regularization_mode"].setCurrentText("parameterized_bspline")
    assert controls["contour_smoothing_scale_mode"].currentText() == "mesh_coupled"
    assert section.form.isRowVisible(controls["contour_smoothing_mesh_gain"].parentWidget())
    assert not section.form.isRowVisible(controls["contour_smoothing_factor"].parentWidget())
    controls["contour_smoothing_mesh_gain"].setValue(0.2)
    controls["contour_displacement_mm"].setValue(0.35)
    controls["contour_segment_mm"].setValue(0.2)
    assert window.project is original and not window.service.submissions
    assert window.apply_settings()
    assert window.project.values == dict(
        original.values,
        contour_path_regularization_mode="parameterized_bspline",
        contour_smoothing_scale_mode="mesh_coupled",
        contour_smoothing_mesh_gain=0.2,
        contour_smoothing_max_displacement=0.35e-3,
        contour_resample_target_segment_length=0.2e-3,
    )
    assert window.undo_stack.count() == 1
    window.compute()
    submitted = window.service.submissions[-1][0].config
    assert submitted.contour_smoothing_max_displacement == pytest.approx(0.35e-3)
    assert submitted.contour_smoothing_mesh_gain == 0.2
    window.undo_stack.undo()
    assert window.project.values == original.values


def test_imported_legacy_strength_and_precision_are_preserved_by_unrelated_edit(windows):
    config = replace(
        preset_config(),
        contour_path_regularization_mode="smooth",
        contour_smoothing_scale_mode="legacy",
        contour_smoothing_factor=0.1234567891234567,
        contour_resample_target_segment_length=0.0001234567891234567,
        contour_smoothing_max_displacement=0.0002345678912345678,
    )
    project = ProjectDocument.from_config(config)
    window = windows(project)
    window.select_node("winding")
    section = window._settings_decisions["Contour smoothing"]
    controls = window._draft_controls
    assert section.form.isRowVisible(controls["contour_smoothing_factor"].parentWidget())
    assert not section.form.isRowVisible(controls["contour_smoothing_mesh_gain"].parentWidget())
    assert window.apply_settings() and window.project is project
    controls["levels"].setValue(config.levels + 1)
    assert window.apply_settings()
    assert window.project.values == dict(project.values, levels=config.levels + 1)


def test_resampling_does_not_enable_smoothing_or_discard_saved_strength(windows):
    project = ProjectDocument.from_config(replace(preset_config(), contour_smoothing_factor=0.4))
    window = windows(project)
    window.select_node("winding")
    controls = window._draft_controls
    section = window._settings_decisions["Contour smoothing"]
    controls["contour_path_regularization_mode"].setCurrentText("resample")
    controls["contour_segment_mm"].setValue(0.5)
    assert section.form.isRowVisible(controls["contour_segment_mm"].parentWidget())
    assert not section.form.isRowVisible(controls["contour_displacement_mm"].parentWidget())
    assert window.apply_settings()
    assert window.project.values == dict(
        project.values,
        contour_path_regularization_mode="resample",
        contour_resample_target_segment_length=0.5e-3,
    )


def test_expert_surface_smoothing_controls_use_explicit_units_and_types():
    fields = {field.key: field for field in expert_fields()}
    cap = fields["contour_smoothing_max_displacement"]
    assert cap.unit == "mm" and cap.display_scale == 1e3
    assert fields["contour_smoothing_scale_mode"].choices == ("mesh_coupled", "legacy")
    assert fields["contour_smoothing_work_samples_per_mesh_length"].kind == "int"
    assert fields["contour_smoothing_output_samples_per_mesh_length"].kind == "int"
    assert fields["contour_smoothing_preserve_min_turn_distance"].kind == "bool"
