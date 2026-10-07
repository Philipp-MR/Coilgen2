"""Progressive decisions preserve exact inputs and reject hidden invalid drafts."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import test_workbench_usability as usability

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox
from shiboken6 import isValid

from coilgen.application import ProjectDocument
from coilgen.application.presets import preset_config
from coilgen.application.targets import recognize_solid_harmonic_target, solid_harmonic_target
from coilgen.gui.focused_editors import control_form


application = usability.application
windows = usability.windows

PAGES = ("geometry", "target", "material", "winding", "optimization", "validation")


def _sections(window):
    return list(window._settings_decisions.values())


def _owned_section(window, key):
    control = window._draft_controls[key]
    return next(section for section in _sections(window) if control in section.controls)


def _edit_numeric(control, text, application):
    control.lineEdit().setFocus()
    control.lineEdit().selectAll()
    QTest.keyClicks(control.lineEdit(), text)
    application.processEvents()


@pytest.mark.parametrize("page", PAGES)
def test_every_primary_registered_setting_is_in_one_modeless_decision(windows, application, page):
    window = windows()
    window.select_node(page)
    original = window.project
    sections = _sections(window)
    main_form = window.settings.widget().form
    assert sections
    for key, control in window._draft_controls.items():
        owners = [section for section in sections if control in section.controls]
        assert len(owners) == 1, key
        assert control_form(window, control, main_form) is owners[0].form
    for index, section in enumerate(sections, 1):
        assert section.step_label.text() == str(index)
        assert not section.dialog.isModal()
        section.open()
        application.processEvents()
        assert sum(other.dialog.isVisible() for other in sections) == 1
        assert section.dialog.isVisible()
    sections[-1].dialog.close()
    assert window.project is original
    assert not window._draft_dirty and window.undo_stack.count() == 0


def test_switching_decision_windows_retains_incomplete_draft_and_widget_identity(windows, application):
    window = windows()
    original = window.project
    radius = window._draft_controls["radius_mm"]
    dimensions = _owned_section(window, "radius_mm")
    mesh = _owned_section(window, "angular_divisions")
    dimensions.open()
    application.processEvents()
    _edit_numeric(radius, "1e", application)
    assert window._draft_dirty and not radius.hasAcceptableInput()
    mesh.open()
    application.processEvents()
    assert not dimensions.dialog.isVisible() and mesh.dialog.isVisible()
    dimensions.open()
    application.processEvents()
    assert window._draft_controls["radius_mm"] is radius
    assert radius.lineEdit().text().startswith("1e")
    assert not radius.hasAcceptableInput()
    assert window.project is original and window.undo_stack.count() == 0


def test_apply_and_next_resolves_fresh_sections_after_one_atomic_revision(windows, application):
    window = windows()
    original = window.project
    old = _owned_section(window, "radius_mm")
    old_dialog = old.dialog
    old.open()
    application.processEvents()
    window._draft_controls["radius_mm"].setValue(172)
    QTest.mouseClick(old.next_button, Qt.MouseButton.LeftButton)
    application.processEvents()
    assert window.project.revision == original.revision + 1
    assert window.project.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.172)
    assert window.undo_stack.count() == 1 and not window._draft_dirty
    fresh = window._settings_decisions["Placement"]
    assert fresh.dialog.isVisible()
    assert sum(section.dialog.isVisible() for section in _sections(window)) == 1
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(old_dialog)


def test_compute_and_save_reopen_invalid_hidden_decision_without_submitting(windows, application, tmp_path):
    window = windows()
    original = window.project
    control = window._draft_controls["rotation_angle_deg"]
    placement = _owned_section(window, "rotation_angle_deg")
    dimensions = _owned_section(window, "radius_mm")
    placement.open()
    application.processEvents()
    _edit_numeric(control, "1e", application)
    dimensions.open()
    assert not placement.dialog.isVisible()
    assert window.compute() is None
    application.processEvents()
    assert window.service.submissions == []
    assert window.project is original and window.undo_stack.count() == 0
    assert placement.dialog.isVisible() and not dimensions.dialog.isVisible()
    assert not window._field_errors["rotation_angle_deg"].isHidden()
    destination = tmp_path / "invalid.coilgen.json"
    with pytest.raises(ValueError, match="pending settings"):
        window.save_project(destination)
    assert not destination.exists()
    assert control.lineEdit().text().startswith("1e")


def _precise_config(kind):
    config = replace(
        preset_config(),
        cylinder_mesh_parameter_list=(0.31234567891234567, 0.08123456789123457, 48, 18, 1, 0, 0, 0),
        target_gradient_strength=0.023456789123456789,
        conductor_cross_section_width=0.002345678912345678,
        conductor_cross_section_height=0.001234567891234567,
        specific_conductivity_conductor=1.7234567891234567e-8,
        conductor_thickness=0.001345678912345678,
        normal_shift_length=0.0001234567891234567,
        tikonov_reg_factor=6.234567891234567,
        max_current_A=7.123456789123456,
        max_wire_current_A=0.0,
        min_wire_distance_m=0.0,
        min_turn_distance=0.004234567891234567,
    )
    if kind == "harmonic":
        config = replace(
            config,
            **solid_harmonic_target(
                2,
                1,
                0.0001234567891234567,
                0.043234567891234565,
                (0.001234567891234567, -0.002345678912345678, 0.003456789123456789),
            ).to_config_overrides(),
        )
    elif kind == "absolute":
        config = replace(config, field_shape_function="1e-4 + 0.001*x*y", target_gradient_normalization="off")
    elif kind == "engineering_alias":
        config = replace(config, contour_discretization_method="wire_current_spacing")
    elif kind == "custom_profile":
        config = replace(
            config,
            cross_sectional_points=(
                (0.0021234567891234567, 0.001234567891234567),
                (0.0024234567891234567, -0.000765432108765433),
                (0.0004234567891234567, -0.000765432108765433),
                (0.0007234567891234567, 0.001234567891234567),
            ),
        )
    return config


@pytest.mark.parametrize("kind", ("linear", "harmonic", "absolute", "engineering_alias", "custom_profile"))
def test_viewing_all_decisions_and_saving_preserves_exact_inputs(windows, application, tmp_path, kind):
    project = ProjectDocument.from_config(_precise_config(kind))
    window = windows(project)
    expected = deepcopy(project.values)
    fingerprint = project.model_fingerprint
    for page in PAGES:
        window.select_node(page)
        for section in _sections(window):
            section.open()
            application.processEvents()
            section.dialog.close()
        assert window.apply_settings()
        assert window.project is project
        assert window.project.values == expected
        assert window.project.model_fingerprint == fingerprint
    assert window.undo_stack.count() == 0
    saved = window.save_project(tmp_path / (kind + ".coilgen.json"))
    restored = ProjectDocument.load(saved)
    assert restored.values == expected
    assert restored.model_fingerprint == fingerprint


def test_engineering_fallback_bounds_remain_raw_and_are_described_as_effective(windows):
    window = windows(ProjectDocument.from_config(_precise_config("engineering_alias")))
    window.select_node("winding")
    original = window.project
    section = window._settings_decisions["Contour policy"]
    controls = window._draft_controls
    assert controls["contour_discretization_method"].currentText() == "wire_current_spacing"
    assert controls["max_wire_current_A"].value() == 0.0
    assert controls["min_wire_distance_mm"].value() == 0.0
    assert "≤7.12346 A" in section.summary_label.text()
    assert "4.23457 mm centerline" in section.summary_label.text()
    assert not section.form.isRowVisible(controls["levels"].parentWidget())
    assert window.apply_settings() and window.project is original


def test_incomplete_engineering_bound_is_marked_in_the_decision_summary(windows, application):
    window = windows(ProjectDocument.from_config(_precise_config("engineering_alias")))
    window.select_node("winding")
    original = window.project
    section = window._settings_decisions["Contour policy"]
    section.open()
    application.processEvents()
    current = window._draft_controls["max_wire_current_A"]
    _edit_numeric(current, "1e", application)
    assert not current.hasAcceptableInput()
    assert "incomplete" in section.summary_label.text().lower()
    assert window.project is original


def test_imported_map_exposes_authoritative_component_and_preserves_ignored_preferences(windows, tmp_path):
    path = tmp_path / "target.csv"
    np.savetxt(
        path,
        np.asarray([[0, 0, 0, 0, 0, 1e-5], [0.01, 0, 0, 0, 0, 2e-5]]),
        delimiter=",",
        header="x,y,z,Bx,By,Bz",
        comments="",
    )
    config = replace(_precise_config("linear"), target_field_definition_file=str(path))
    project = ProjectDocument.from_config(config)
    window = windows(project)
    window.select_node("target")
    expected = deepcopy(project.values)
    assert set(window._settings_decisions) == {"Choose field", "Map component"}
    assert window._target_region_editor.gallery.isHidden()
    assert window._target_region_editor.sampling.isHidden()
    main = window.settings.widget().form
    for key in (
        "field_shape_function",
        "target_gradient_strength",
        "target_gradient_axis",
        "target_region_radius",
        "target_region_resolution",
    ):
        assert not main.isRowVisible(window._draft_controls[key].parentWidget())
    for section in _sections(window):
        section.open()
        section.dialog.close()
    assert window.apply_settings() and window.project is project
    assert window.project.values == expected and window.undo_stack.count() == 0


def test_target_scaling_opening_reapplies_policy_visibility_after_legacy_reveal(windows, application):
    window = windows()
    window.select_node("target")
    original = window.project
    scaling = window._settings_decisions["Target scaling"]
    policy = window._draft_controls["target_gradient_normalization"]
    axis = window._draft_controls["target_gradient_axis"]
    legacy = window.settings.widget().findChild(QCheckBox, "advanced_target_controls")
    # This compatibility control is hidden in the card UI, but still governs
    # field rows. Opening details must respect the more specific active policy.
    policy.setCurrentText("off")
    assert not scaling.form.isRowVisible(axis.parentWidget())
    scaling.open()
    application.processEvents()
    assert legacy.isChecked()
    assert not scaling.form.isRowVisible(axis.parentWidget())
    assert not axis.isEnabled()
    assert window.project is original


def test_choose_field_opens_original_tiles_directly_and_family_selection_rebuilds_safely(windows, application):
    window = windows()
    window.select_node("target")
    original = window.project
    section = window._settings_decisions["Choose field"]
    combo = section.controls[0]
    model, buttons = combo.model(), tuple(combo.buttons)
    section.open()
    application.processEvents()
    assert section.dialog.isVisible() and not combo.dialog.isVisible()
    assert combo.scroll.isVisible() and combo.isHidden()
    assert combo.model() is model and tuple(combo.buttons) == buttons
    assert window.project is original and not window._draft_dirty
    target = combo.findText("Spherical harmonic")
    QTest.keyClick(buttons[target], Qt.Key.Key_Return)
    application.processEvents()
    assert window.project.revision == original.revision + 1
    assert window.undo_stack.count() == 1
    assert recognize_solid_harmonic_target(window.project.config.field_shape_function) is not None
    rebuilt = window._settings_decisions["Choose field"]
    assert rebuilt is not section and rebuilt.dialog.isVisible()
    assert rebuilt.controls[0].currentText() == "Spherical harmonic"
    assert not rebuilt.controls[0].dialog.isVisible()
    assert sum(owner.dialog.isVisible() for owner in _sections(window)) == 1
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(combo)


def test_edited_invalid_regularization_stays_blocking_after_it_becomes_inactive(windows, application, tmp_path):
    config = replace(preset_config(), sf_opt_method="tikhonov", research_physics=None)
    window = windows(ProjectDocument.from_config(config))
    window.select_node("optimization")
    original = window.project
    numerical = window._settings_decisions["Numerical settings"]
    strategy = window._settings_decisions["Solve strategy"]
    regularization = window._draft_controls["tikonov_reg_factor"]
    numerical.open()
    application.processEvents()
    _edit_numeric(regularization, "1e", application)
    assert not regularization.hasAcceptableInput()
    strategy.open()
    window._draft_controls["sf_opt_method"].setCurrentText("constrained_quadratic")
    assert not regularization.isEnabled()
    assert not numerical.form.isRowVisible(regularization.parentWidget())
    assert not window.apply_settings()
    application.processEvents()
    assert numerical.dialog.isVisible() and not strategy.dialog.isVisible()
    assert not window._field_errors["tikonov_reg_factor"].isHidden()
    assert regularization.isEnabled() and regularization.property("repairInactiveDraft")
    assert window.project is original and window.undo_stack.count() == 0
    assert window.compute() is None and window.service.submissions == []
    destination = tmp_path / "inactive-invalid.coilgen.json"
    with pytest.raises(ValueError, match="pending settings"):
        window.save_project(destination)
    assert not destination.exists()
    assert regularization.lineEdit().text().startswith("1e")
    _edit_numeric(regularization, "6.5", application)
    assert window.apply_settings()
    assert window.project.config.sf_opt_method == "constrained_quadratic"
    assert window.project.config.tikonov_reg_factor == 6.5
    assert window.project.revision == original.revision + 1 and window.undo_stack.count() == 1
    restored = window._draft_controls["tikonov_reg_factor"]
    assert not restored.isEnabled()
    assert not _owned_section(window, "tikonov_reg_factor").form.isRowVisible(restored.parentWidget())


def test_graphical_study_output_changes_only_owned_study_and_is_undoable(windows, application):
    project = ProjectDocument.from_config(preset_config()).update_study({"pipeline_stage": "sheet", "fabricate": False})
    window = windows(project)
    window.select_node("stage:sheet")
    original = window.project
    inputs, model_fingerprint, study = deepcopy(original.values), original.model_fingerprint, original.study
    section = window._settings_decisions["Choose output"]
    combo = section.controls[0]
    section.open()
    application.processEvents()
    assert combo.isHidden() and combo.scroll.isVisible()
    assert not combo.dialog.isVisible()
    assert window.project is original and window.service.submissions == []
    selected = combo.findData("contours")
    QTest.keyClick(combo.buttons[selected], Qt.Key.Key_Return)
    application.processEvents()
    assert window.project.study == {**study, "pipeline_stage": "contours"}
    assert window.project.values == inputs and window.project.model_fingerprint == model_fingerprint
    assert window.project.revision == original.revision + 1
    assert window.stage_selector.currentData() == "contours"
    assert window.undo_stack.count() == 1 and window.service.submissions == []
    window.undo_stack.undo()
    application.processEvents()
    assert window.project.study == study
    assert window.project.values == inputs and window.project.model_fingerprint == model_fingerprint
    assert window.stage_selector.currentData() == "sheet"
    assert window.service.submissions == []


def test_field_region_choice_reuses_original_tiles_and_model_without_a_second_picker(windows, application):
    window = windows()
    window.select_node("target")
    original = window.project
    section = window._settings_decisions["Choose field region"]
    gallery = window._target_region_editor.gallery
    model, buttons = gallery.model(), tuple(gallery.buttons)
    QTest.mouseClick(section.title_label, Qt.MouseButton.LeftButton)
    application.processEvents()
    assert section.dialog.isVisible() and not gallery.dialog.isVisible()
    assert gallery.isHidden() and gallery.scroll.isVisible()
    assert gallery.model() is model and tuple(gallery.buttons) == buttons
    assert window.project is original and not window._draft_dirty
    section.dialog.close()
    gallery.showPopup()
    application.processEvents()
    assert section.dialog.isVisible() and not gallery.dialog.isVisible()
    assert window.project is original and window.service.submissions == []
    selected = gallery.findData("box")
    QTest.keyClick(buttons[selected], Qt.Key.Key_Space)
    application.processEvents()
    assert window.project.revision == original.revision + 1
    assert window.undo_stack.count() == 1 and window.service.submissions == []
    assert window.project.config.target_region_geometry["shape"] == "box"
    assert window.project.config.field_shape_function == original.config.field_shape_function
    rebuilt = window._settings_decisions["Choose field region"]
    assert rebuilt.dialog.isVisible()
    assert not window._target_region_editor.gallery.dialog.isVisible()
    assert sum(owner.dialog.isVisible() for owner in _sections(window)) == 1


def test_full_decision_walk_reaches_study_without_revisions_or_jobs(windows, application):
    window = windows()
    original = window.project
    values, study, fingerprint = deepcopy(original.values), original.study, original.model_fingerprint
    expected_pages = ("geometry", "target", "material", "winding", "optimization", "validation")
    visited = []
    current = _sections(window)[0]
    current.open()
    for _step in range(64):
        key = window._current_node.data(0, Qt.ItemDataRole.UserRole)
        if not visited or visited[-1] != key:
            visited.append(key)
        visible = [section for section in _sections(window) if section.dialog.isVisible()]
        assert len(visible) == 1
        current = visible[0]
        assert window.project is original and window.undo_stack.count() == 0
        assert window.project.values == values and window.project.study == study
        assert window.project.model_fingerprint == fingerprint and window.service.submissions == []
        if key.startswith("stage:"):
            assert current.dialog.windowTitle() == "Choose output"
            assert current.next_button.isHidden()
            break
        assert current.next_button.isVisible()
        QTest.mouseClick(current.next_button, Qt.MouseButton.LeftButton)
        application.processEvents()
    else:
        pytest.fail("The graphical decisions did not reach the requested study boundary.")
    assert visited == [*expected_pages, "stage:" + study["pipeline_stage"]]


def test_last_decision_applies_one_exact_model_edit_before_opening_next_page(windows, application):
    window = windows()
    original = window.project
    expected = deepcopy(original.values)
    parameters = list(expected["cylinder_mesh_parameter_list"])
    parameters[2] += 4
    expected["cylinder_mesh_parameter_list"] = type(expected["cylinder_mesh_parameter_list"])(parameters)
    mesh = _sections(window)[-1]
    assert mesh.dialog.windowTitle() == "Mesh"
    mesh.open()
    application.processEvents()
    window._draft_controls["angular_divisions"].setValue(parameters[2])
    QTest.mouseClick(mesh.next_button, Qt.MouseButton.LeftButton)
    application.processEvents()
    assert window._current_node.data(0, Qt.ItemDataRole.UserRole) == "target"
    assert window.project.revision == original.revision + 1
    assert window.project.values == expected and window.project.study == original.study
    assert window.undo_stack.count() == 1 and not window._draft_dirty
    fresh = _sections(window)[0]
    assert fresh.dialog.windowTitle() == "Choose field" and fresh.dialog.isVisible()
    assert sum(section.dialog.isVisible() for section in _sections(window)) == 1
    assert window.service.submissions == []


def test_invalid_last_decision_keeps_page_and_reveals_its_error(windows, application):
    window = windows()
    window.select_node("material")
    original = window.project
    last = _sections(window)[-1]
    assert last.dialog.windowTitle() == "Continuous-sheet material"
    last.open()
    application.processEvents()
    resistivity = window._draft_controls["resistivity"]
    _edit_numeric(resistivity, "1e", application)
    assert not resistivity.hasAcceptableInput()
    QTest.mouseClick(last.next_button, Qt.MouseButton.LeftButton)
    application.processEvents()
    assert window._current_node.data(0, Qt.ItemDataRole.UserRole) == "material"
    assert window.project is original and window.undo_stack.count() == 0
    assert last.dialog.isVisible() and window._draft_dirty
    assert not window._field_errors["resistivity"].isHidden()
    assert resistivity.lineEdit().text().startswith("1e")
    assert window.service.submissions == []
