"""Focused controls retain the headless configuration's SI engineering contract."""

from dataclasses import replace

import pytest
import test_workbench_usability as usability

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.engineering_limits import max_wire_current_A, min_wire_distance_m


application = usability.application
windows = usability.windows


def test_physical_placement_is_explicit_si_edit_with_exact_profile_and_undo(windows):
    points = ((0.0021, 0.0007), (0.0024, -0.0007), (0.0004, -0.0007), (0.0001, 0.0007))
    project = ProjectDocument.from_config(replace(preset_config(), cross_sectional_points=points))
    window = windows(project)
    window.select_node("material")
    controls = window._draft_controls
    section = window._settings_decisions["Physical placement"]
    controls["conductor_mounting_mode"].setCurrentText("surface_inner")
    controls["normal_shift_mm"].setValue(5.7)
    controls["return_path_lift"].setChecked(False)
    assert "Inside support" in section.summary_label.text()
    assert "return-path lift off" in section.summary_label.text()
    assert not section.form.isRowVisible(controls["normal_shift_mm"].parentWidget())
    assert window.project is project
    assert window.apply_settings()
    assert window.project.values == dict(
        project.values,
        conductor_mounting_mode="surface_inner",
        skip_normal_shift=True,
        normal_shift_length=5.7e-3,
    )
    assert tuple(map(tuple, window.project.config.cross_sectional_points)) == points
    assert window.service.submissions == []
    window.undo_stack.undo()
    assert window.project.values == project.values


def test_contour_policy_changes_only_requested_fields_with_si_spacing(windows):
    window = windows()
    window.select_node("winding")
    original = window.project
    controls = window._draft_controls
    section = window._settings_decisions["Contour policy"]
    controls["contour_discretization_method"].setCurrentText("engineering")
    controls["max_wire_current_A"].setValue(8.1)
    controls["min_wire_distance_mm"].setValue(0.8)
    assert "≤8.1 A" in section.summary_label.text()
    assert "≥0.8 mm centerline" in section.summary_label.text()
    assert not section.form.isRowVisible(controls["levels"].parentWidget())
    assert window.project is original
    assert window.apply_settings()
    assert window.project.values == dict(
        original.values,
        contour_discretization_method="engineering",
        max_wire_current_A=8.1,
        min_wire_distance_m=0.8e-3,
    )
    assert max_wire_current_A(window.project.config) == 8.1
    assert min_wire_distance_m(window.project.config) == pytest.approx(0.8e-3)
    assert not window.service.submissions
    window.undo_stack.undo()
    assert window.project.values == original.values


def test_effective_contour_summary_tracks_general_spacing_fallback(windows):
    window = windows()
    window.select_node("winding")
    controls = window._draft_controls
    controls["contour_discretization_method"].setCurrentText("engineering")
    controls["min_wire_distance_mm"].setValue(0)
    controls["min_turn_distance"].setValue(6.5)
    assert "≥6.5 mm centerline" in window._settings_decisions["Contour policy"].summary_label.text()
    assert window._draft_dirty and not window.service.submissions
