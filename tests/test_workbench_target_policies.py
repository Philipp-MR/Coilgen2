"""GUI policies preserve native target values and expose actual backend units."""

from dataclasses import replace
import re

import numpy as np
import pytest

import test_workbench_usability as usability_tests
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel
from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.field_units import PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
from coilgen.geometry import load_mesh_from_config
from coilgen.gui.conductor_preview import ConductorPreview
from coilgen.target import define_target_field

application = usability_tests.application
windows = usability_tests.windows


def target_window(windows, **overrides):
    config = replace(preset_config(), **overrides)
    window = windows(ProjectDocument.from_config(config))
    window.select_node("target")
    return window


@pytest.mark.parametrize("policy", ("off", "none"))
def test_absolute_field_disables_unused_gradient_controls_without_mutating_values(windows, policy):
    strength = 0.023456789123456789
    window = target_window(
        windows,
        field_shape_function="1e-4",
        target_gradient_normalization=policy,
        target_gradient_strength=strength,
    )
    assert not window._draft_controls["target_gradient_strength"].isEnabled()
    assert not window._draft_controls["target_gradient_axis"].isEnabled()
    window._draft_controls["target_region_resolution"].setValue(19)
    assert window.apply_settings()
    assert window.project.config.target_gradient_strength == strength
    field = define_target_field([], None, None, window.project.config)
    np.testing.assert_array_equal(field.b[:, 2], np.full(len(field.coords), 1e-4))


@pytest.mark.parametrize("units,suffix,display_scale", [("T", " mT/m", 1000), ("Hz", " Hz/m", 1)])
def test_gradient_display_matches_native_normalization_and_tesla_result(windows, units, suffix, display_scale):
    window = target_window(
        windows,
        field_shape_function="-y",
        target_gradient_normalization="linear_fit",
        target_gradient_axis="y",
        target_field_units=units,
        target_gradient_strength=0.02 if units == "T" else 1000,
    )
    widget = window._draft_controls["target_gradient_strength"]
    assert widget.suffix() == suffix
    assert widget.value() == pytest.approx(window.project.config.target_gradient_strength * display_scale)
    widget.setValue(25 if units == "T" else 1250)
    assert window.apply_settings()
    field = define_target_field([], None, None, window.project.config)
    slope_T_per_m = 0.025 if units == "T" else 1250 / PROTON_GYROMAGNETIC_RATIO_HZ_PER_T
    np.testing.assert_allclose(field.b[:, 2], -slope_T_per_m * field.coords[:, 1], rtol=1e-13, atol=1e-19)


def test_context_and_units_roundtrip_leave_high_precision_native_strength_unchanged(windows):
    strength = 0.023456789123456789
    window = target_window(windows, target_gradient_normalization="auto", target_gradient_strength=strength)
    original = window.project
    controls = window._draft_controls
    controls["design_context"].setCurrentText("mr_shim")
    assert controls["target_gradient_strength"].suffix() == ""
    assert controls["target_gradient_strength"].value() == pytest.approx(strength)
    assert not controls["target_gradient_axis"].isEnabled()
    controls["target_field_units"].setCurrentText("Hz")
    controls["design_context"].setCurrentText("mr_gradient")
    assert controls["target_gradient_strength"].suffix() == " Hz/m"
    controls["target_field_units"].setCurrentText("T")
    assert controls["target_gradient_strength"].suffix() == " mT/m"
    assert window.apply_settings()
    assert window.project is original
    assert window.project.config.target_gradient_strength == strength


def test_edited_amplitude_survives_policy_display_conversion_and_compute_submission(windows):
    window = target_window(windows, target_gradient_normalization="auto", target_gradient_strength=0.02)
    controls = window._draft_controls
    controls["target_gradient_strength"].setValue(30)
    controls["design_context"].setCurrentText("mr_shim")
    assert controls["target_gradient_strength"].value() == pytest.approx(0.03)
    controls["design_context"].setCurrentText("mr_gradient")
    assert controls["target_gradient_strength"].value() == pytest.approx(30)
    assert window.compute() is not None
    submitted, _study = window.service.submissions[-1]
    assert submitted.config.target_gradient_strength == pytest.approx(0.03)
    field = define_target_field([], None, None, submitted.config)
    np.testing.assert_allclose(field.b[:, 2], 0.03 * field.coords[:, 1], rtol=1e-13, atol=1e-19)


def test_compute_commits_numeric_text_without_requiring_focus_to_leave_the_field(windows):
    window = target_window(windows, target_gradient_normalization="linear_fit", target_gradient_strength=0.02)
    widget = window._draft_controls["target_gradient_strength"]
    widget.lineEdit().selectAll()
    QTest.keyClicks(widget.lineEdit(), "25")
    # Ctrl+Return can invoke Compute while the spin box retains keyboard focus.
    # The text must be interpreted before freezing the submitted model.
    assert window.compute() is not None
    submitted, _study = window.service.submissions[-1]
    assert submitted.config.target_gradient_strength == pytest.approx(0.025)


def test_compute_keeps_incomplete_numeric_text_and_does_not_submit_the_old_value(windows):
    window = target_window(windows, target_gradient_normalization="linear_fit", target_gradient_strength=0.02)
    widget = window._draft_controls["target_gradient_strength"]
    widget.lineEdit().selectAll()
    QTest.keyClicks(widget.lineEdit(), "1e")
    assert not widget.hasAcceptableInput()
    assert window.compute() is None
    assert not window.service.submissions
    assert widget.lineEdit().text().startswith("1e")
    assert window.project.config.target_gradient_strength == 0.02


def test_mouse_compute_keeps_incomplete_numeric_text_after_focus_leaves_field(windows, application):
    window = target_window(windows, target_gradient_normalization="linear_fit", target_gradient_strength=0.02)
    window.show()
    application.processEvents()
    widget = window._draft_controls["target_gradient_strength"]
    widget.setFocus()
    application.processEvents()
    widget.lineEdit().selectAll()
    QTest.keyClicks(widget.lineEdit(), "1e")
    QTest.mouseClick(window.compute_button, Qt.MouseButton.LeftButton)
    assert not window.service.submissions
    assert widget.lineEdit().text().startswith("1e")
    assert window.project.config.target_gradient_strength == 0.02


@pytest.mark.parametrize("correction", ["typing", "step", "set_value"])
def test_incomplete_scientific_draft_can_be_corrected_and_computed(windows, correction):
    window = target_window(windows, target_gradient_normalization="linear_fit", target_gradient_strength=0.02)
    widget = window._draft_controls["target_gradient_strength"]
    widget.lineEdit().selectAll()
    QTest.keyClicks(widget.lineEdit(), "1e")
    assert window.compute() is None
    if correction == "typing":
        widget.lineEdit().selectAll()
        QTest.keyClicks(widget.lineEdit(), "25")
        expected = 0.025
    elif correction == "step":
        widget.stepUp()
        expected = 0.021
    else:
        widget.setValue(25)
        expected = 0.025
    assert window.compute() is not None
    submitted, _study = window.service.submissions[-1]
    assert submitted.config.target_gradient_strength == pytest.approx(expected)


def test_display_policy_conversion_does_not_replace_incomplete_amplitude_with_old_value(windows):
    window = target_window(windows, target_gradient_normalization="auto", target_gradient_strength=0.02)
    widget = window._draft_controls["target_gradient_strength"]
    widget.lineEdit().selectAll()
    QTest.keyClicks(widget.lineEdit(), "1e")
    window._draft_controls["design_context"].setCurrentText("mr_shim")
    assert window.compute() is None
    assert not window.service.submissions
    assert widget.lineEdit().text().startswith("1e")
    widget.lineEdit().selectAll()
    QTest.keyClicks(widget.lineEdit(), "25")
    assert window.compute() is not None
    submitted, _study = window.service.submissions[-1]
    assert submitted.config.target_gradient_strength == pytest.approx(0.025)
    assert submitted.config.design_context == "mr_shim"


def test_winding_intention_commits_native_mode_and_preserves_ignored_route_preference(windows):
    window = windows(ProjectDocument.from_config(preset_config()))
    window.select_node("winding")
    controls = window._draft_controls
    assert controls["winding_intent"].currentText() == "Connected circuit"
    controls["winding_intent"].setCurrentText("Independent closed turns")
    assert window.apply_settings()
    assert window.project.config.contour_only_sweep_mode is True
    assert window.project.config.skip_postprocessing is False
    assert window._draft_controls["winding_intent"].currentText() == "Independent closed turns"
    window._draft_controls["winding_intent"].setCurrentText("Connected circuit")
    assert window.apply_settings()
    assert window.project.config.contour_only_sweep_mode is False
    assert window.project.config.skip_postprocessing is False


def test_biplanar_center_controls_move_actual_mesh_in_correct_world_axes(windows):
    config = preset_config("biplanar")
    window = windows(ProjectDocument.from_config(config))
    before = load_mesh_from_config(config, window.project.project_root)
    controls = window._draft_controls
    assert "rotation_angle_deg" not in controls
    displacement_m = np.array([0.013, -0.007, 0.011])
    for axis, value in zip("xyz", displacement_m * 1000):
        controls[f"center_{axis}_mm"].setValue(value)
    assert window.apply_settings()
    updated = window.project.config
    np.testing.assert_array_equal(updated.biplanar_mesh_parameter_list[4:7], config.biplanar_mesh_parameter_list[4:7])
    np.testing.assert_allclose(updated.biplanar_mesh_parameter_list[7:10], displacement_m, atol=0, rtol=1e-15)
    after = load_mesh_from_config(updated, window.project.project_root)
    np.testing.assert_allclose(after.vertices, before.vertices + displacement_m, rtol=1e-14, atol=1e-17)
    np.testing.assert_array_equal(after.faces, before.faces)


@pytest.mark.parametrize("normal", [(1, 2, 3), (0.6, 0, -0.8), (-1, 2, -3), (0, 0, -1)])
def test_biplanar_plane_normal_controls_orient_actual_backend_planes(windows, normal):
    window = windows(ProjectDocument.from_config(preset_config("biplanar")))
    requested_normal = np.array(normal, dtype=float)
    for axis, value in zip("xyz", requested_normal):
        window._draft_controls[f"plane_normal_{axis}"].setValue(value)
    assert window.apply_settings()
    mesh = load_mesh_from_config(window.project.config, window.project.project_root)
    unit_normal = requested_normal / np.linalg.norm(requested_normal)
    separation = window.project.config.biplanar_mesh_parameter_list[10]
    np.testing.assert_allclose(np.abs(mesh.vertices @ unit_normal), separation / 2, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(mesh.face_normals, np.tile(unit_normal, (len(mesh.faces), 1)), rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(window.project.config.biplanar_mesh_parameter_list[7:10], [0, 0, 0])


def test_cylinder_editor_length_radius_and_degree_rotation_match_backend_geometry(windows):
    window = windows(ProjectDocument.from_config(preset_config()))
    controls = window._draft_controls
    controls["length_mm"].setValue(250)
    controls["radius_mm"].setValue(90)
    controls["rotation_angle_deg"].setValue(90)
    assert window.apply_settings()
    values = window.project.config.cylinder_mesh_parameter_list
    # Native scripts retain their legacy generator-height parameter. The GUI
    # exposes its realized axial span and translates only intentional edits.
    assert values[0] == pytest.approx(0.25 * (values[3] + 1) / values[3])
    assert values[1] == 0.09
    assert values[7] == pytest.approx(np.pi / 2)
    mesh = load_mesh_from_config(window.project.config, window.project.project_root)
    np.testing.assert_allclose(np.min(mesh.vertices, axis=0), [-0.09, -0.125, -0.09], atol=1e-16)
    np.testing.assert_allclose(np.max(mesh.vertices, axis=0), [0.09, 0.125, 0.09], atol=1e-16)


def test_cylinder_axial_refinement_preserves_realized_length_and_untouched_height_bits(windows):
    config = preset_config()
    height = 0.23456789123456789
    values = list(config.cylinder_mesh_parameter_list)
    values[0] = height
    config.cylinder_mesh_parameter_list = tuple(values)
    window = windows(ProjectDocument.from_config(config))
    span = height * values[3] / (values[3] + 1)
    assert window._draft_controls["length_mm"].value() == pytest.approx(span * 1000)
    window._draft_controls["radius_mm"].setValue(90)
    assert window.apply_settings()
    assert window.project.config.cylinder_mesh_parameter_list[0] == height
    window._draft_controls["length_divisions"].setValue(24)
    assert window.apply_settings()
    mesh = load_mesh_from_config(window.project.config, window.project.project_root)
    assert np.ptp(mesh.vertices[:, 2]) == pytest.approx(span, rel=1e-14)
    updated = window.project.config.cylinder_mesh_parameter_list
    assert updated[0] == pytest.approx(span * 25 / 24, rel=1e-14)


def test_cone_editor_radii_rotation_and_world_center_follow_generator_order(windows):
    config = replace(
        preset_config(),
        coil_mesh_file="create conical frustum mesh",
        surface_is_cylinder_flag=False,
        conical_frustum_mesh_parameter_list=(0.24, 0.1, 0.06, 16, 8, 1, 0, 0, 0, 0, 0, 0),
    )
    window = windows(ProjectDocument.from_config(config))
    controls = window._draft_controls
    controls["bottom_radius_mm"].setValue(110)
    controls["top_radius_mm"].setValue(70)
    controls["rotation_angle_deg"].setValue(90)
    center = np.array([0.013, -0.007, 0.011])
    for axis, value in zip("xyz", center * 1000):
        controls[f"center_{axis}_mm"].setValue(value)
    assert window.apply_settings()
    values = window.project.config.conical_frustum_mesh_parameter_list
    assert values[1] == 0.11 and values[2] == 0.07
    assert values[8] == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(values[9:12], center, atol=0, rtol=1e-15)
    mesh = load_mesh_from_config(window.project.config, window.project.project_root)
    local = mesh.vertices - center
    np.testing.assert_allclose(np.min(local, axis=0), [-0.11, -0.12, -0.11], atol=1e-16)
    np.testing.assert_allclose(np.max(local, axis=0), [0.11, 0.12, 0.11], atol=1e-16)


def test_pending_validation_edits_survive_stage_change_and_submit_the_new_stage(windows):
    window = windows(ProjectDocument.from_config(preset_config()).update_study({"pipeline_stage": "evaluated"}))
    window.select_node("validation")
    window._draft_controls["validation_resolution"].setValue(23)
    window._draft_controls["design_margin"].setValue(0.75)
    window.stage_selector.setCurrentIndex(window.stage_selector.findData("sheet"))
    assert window.project.study["pipeline_stage"] == "sheet"
    assert window.project.study["validation_resolution"] == 23
    assert window.project.study["design_margin"] == 0.75
    assert window.fabricate is False
    assert window.compute() is not None
    _project, controls = window.service.submissions[-1]
    assert controls["pipeline_stage"] == "sheet"
    assert controls["validation_resolution"] == 23
    assert controls["design_margin"] == 0.75
    assert controls["fabricate"] is False


def _display_numbers(text):
    return [
        float(value) for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text.replace("−", "-"))
    ]


def _gy_values(label):
    lines = label.text().splitlines()
    slope = next(line for line in lines if "mT/m" in line)
    endpoints = next(line for line in lines if "mm" in line and "mT" in line)
    values = _display_numbers(endpoints)
    return _display_numbers(slope)[0], np.array(values[:3]) * 1e-3, np.array(values[3:]) * 1e-3


def test_gy_summary_tracks_uncommitted_inputs_and_matches_actual_target_poles_after_apply(windows):
    window = target_window(windows)
    label = window.findChild(QLabel, "linear_gradient_summary")
    assert label is not None and not label.isHidden()
    original = window.project
    amplitude = window._draft_controls["target_gradient_strength"]
    radius = window._draft_controls["target_region_radius"]
    for control, text in ((amplitude, "25"), (radius, "30")):
        control.lineEdit().selectAll()
        QTest.keyClicks(control.lineEdit(), text)
    assert window.project is original  # The explanation describes the pending inputs.
    slope_mT_per_m, positions_m, fields_T = _gy_values(label)
    assert slope_mT_per_m == 25
    np.testing.assert_allclose(positions_m, [-0.03, 0, 0.03], atol=0, rtol=1e-15)
    np.testing.assert_allclose(fields_T, [-0.00075, 0, 0.00075], atol=0, rtol=1e-15)
    assert window.apply_settings()
    target = define_target_field([], None, None, window.project.config)
    for y_m, displayed_T in zip(positions_m, fields_T):
        index = np.argmin(np.linalg.norm(target.coords - [0, y_m, 0], axis=1))
        np.testing.assert_allclose(target.coords[index], [0, y_m, 0], atol=1e-15)
        assert target.b[index, 2] == pytest.approx(displayed_T, rel=1e-13, abs=1e-19)
    np.testing.assert_allclose(target.b[:, 2], slope_mT_per_m * 1e-3 * target.coords[:, 1], rtol=1e-13, atol=1e-19)


@pytest.mark.parametrize(
    "key,value",
    [
        ("target_gradient_normalization", "off"),
        ("target_field_units", "Hz"),
        ("field_shape_function", "y*y"),
        ("target_gradient_axis", "x"),
        ("design_context", "mr_shim"),
    ],
)
def test_gy_summary_is_hidden_when_draft_no_longer_matches_the_linear_tesla_contract(windows, key, value):
    window = target_window(windows, target_gradient_normalization="auto")
    label = window.findChild(QLabel, "linear_gradient_summary")
    assert label is not None and not label.isHidden()
    control = window._draft_controls[key]
    if key == "field_shape_function":
        control.selectAll()
        QTest.keyClicks(control, value)
    else:
        control.setCurrentText(value)
    assert label.isHidden()


@pytest.mark.parametrize("key", ("target_gradient_strength", "target_region_radius"))
def test_gy_summary_does_not_report_old_values_for_incomplete_numeric_drafts(windows, key):
    window = target_window(windows)
    label = window.findChild(QLabel, "linear_gradient_summary")
    assert not label.isHidden()
    control = window._draft_controls[key]
    control.lineEdit().selectAll()
    QTest.keyClicks(control.lineEdit(), "1e")
    assert label.isHidden()
    assert window.compute() is None
    assert not window.service.submissions


def test_origin_specific_gy_summary_is_hidden_for_a_translated_mesh_centered_roi(windows):
    config = preset_config("biplanar")
    parameters = list(config.biplanar_mesh_parameter_list)
    parameters[8] = 0.04
    config.biplanar_mesh_parameter_list = tuple(parameters)
    config.set_roi_into_mesh_center = True
    window = windows(ProjectDocument.from_config(config))
    window.select_node("target")
    label = window.findChild(QLabel, "linear_gradient_summary")
    assert label is None or label.isHidden()
    support = load_mesh_from_config(config, window.project.project_root)
    target = define_target_field([support], None, None, config)
    assert np.min(target.coords[:, 1]) == pytest.approx(0.02)
    assert np.max(target.coords[:, 1]) == pytest.approx(0.06)
    assert np.min(target.b[:, 2]) == pytest.approx(0.0004)
    assert np.max(target.b[:, 2]) == pytest.approx(0.0012)


def test_spherical_gy_summary_is_hidden_for_an_explicit_target_mesh(windows, tmp_path):
    region = tmp_path / "target_region.npz"
    np.savez(
        region,
        schema="coilgen-mesh-v1",
        vertices_m=np.array([[0, -0.01, 0], [0, 0.02, 0], [0, 0, 0.005]]),
        faces=np.empty((0, 3), dtype=int),
    )
    window = target_window(windows, target_mesh_file=str(region))
    label = window.findChild(QLabel, "linear_gradient_summary")
    assert label is None or label.isHidden()


def test_scalar_gy_summary_is_hidden_for_a_vector_fitting_contract(windows):
    window = target_window(windows, target_field_component="vector")
    label = window.findChild(QLabel, "linear_gradient_summary")
    assert label is None or label.isHidden()


def test_gy_material_summary_shows_copper_area_and_nominal_insulated_rectangle(windows):
    window = windows(ProjectDocument.from_config(preset_config()))
    window.select_node("material")
    preview = window.findChild(ConductorPreview, "conductor_cross_section_preview")
    assert preview is not None
    label = preview.summary_label
    assert preview.dimensions == (2, 2, 0.1)
    values = _display_numbers(label.text())
    np.testing.assert_allclose(values[:5], [2, 2, 4, 2.2, 2.2], atol=0, rtol=1e-15)
    config = window.project.config
    assert config.conductor_cross_section_width * config.conductor_cross_section_height == pytest.approx(4e-6)
    assert config.conductor_cross_section_width + 2 * config.sweep_insulation_thickness == pytest.approx(0.0022)
    original = window.project
    for key, text in (("wire_width", "3"), ("wire_height", "1.5"), ("insulation", "0.2")):
        control = window._draft_controls[key]
        control.lineEdit().selectAll()
        QTest.keyClicks(control.lineEdit(), text)
    assert window.project is original
    np.testing.assert_allclose(_display_numbers(label.text())[:5], [3, 1.5, 4.5, 3.4, 1.9], atol=0, rtol=1e-15)
    assert preview.dimensions == (3, 1.5, 0.2)
    assert "wire_width" in window._changed_controls
    assert window.apply_settings()
    config = window.project.config
    assert config.conductor_cross_section_width * config.conductor_cross_section_height == pytest.approx(4.5e-6)
    assert config.sweep_insulation_thickness == pytest.approx(0.0002)
    assert config.conductor_thickness == original.config.conductor_thickness


def test_custom_conductor_profile_is_identified_without_claiming_fallback_rectangle_area(windows):
    config = replace(
        preset_config(),
        cross_sectional_points=((-0.0005, -0.0005), (0.0005, -0.0005), (0.0005, 0.0005), (-0.0005, 0.0005)),
    )
    window = windows(ProjectDocument.from_config(config))
    window.select_node("material")
    preview = window.findChild(ConductorPreview, "conductor_cross_section_preview")
    label = preview.summary_label
    assert "Polygon" in label.text()
    assert preview.custom and preview.dimensions is None
    assert preview.drawing._dimensions is None
    assert "Cu area 1 mm²" in label.text()
    assert "area 4 mm²" not in label.text()
    np.testing.assert_array_equal(preview.drawing._profile, np.asarray(config.cross_sectional_points) * 1000)
    assert "Outside" not in label.text()
    assert window._draft_controls["wire_width"].value() == 2
    window._draft_controls["wire_width"].setValue(3)
    assert window.apply_settings()
    np.testing.assert_array_equal(window.project.config.cross_sectional_points, config.cross_sectional_points)


def test_conductor_summary_requires_complete_dimensions_before_reporting_area(windows):
    window = windows(ProjectDocument.from_config(preset_config()))
    window.select_node("material")
    preview = window.findChild(ConductorPreview, "conductor_cross_section_preview")
    label = preview.summary_label
    width = window._draft_controls["wire_width"]
    width.lineEdit().selectAll()
    QTest.keyClicks(width.lineEdit(), "1e")
    assert "mm²" not in label.text()
    assert preview.dimensions is None and preview.drawing._dimensions is None
    assert window.compute() is None
    assert not window.service.submissions


def test_single_character_contour_count_edit_is_applied_to_the_submitted_model(windows):
    window = windows(ProjectDocument.from_config(preset_config()))
    window.select_node("winding")
    levels = window._draft_controls["levels"]
    levels.lineEdit().selectAll()
    QTest.keyClicks(levels.lineEdit(), "4")
    assert window.compute() is not None
    submitted, _study = window.service.submissions[-1]
    assert submitted.config.levels == 4
