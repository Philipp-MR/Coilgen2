"""Setup edits preserve native inputs and commit all sections as one transaction."""

from dataclasses import replace
import numpy as np

import pytest

from coilgen.application.design_setup import DesignSetupDraft, SetupInputError
from coilgen.application.presets import preset_config
from coilgen.application.project import ProjectDocument
from coilgen.application.conductor_profiles import round_wire_profile


@pytest.fixture
def project():
    return ProjectDocument.from_config(preset_config())


def test_opening_every_section_preserves_project_and_native_precision(project):
    project = project.update_config(
        {"tikonov_reg_factor": 1234.567891234567, "target_gradient_strength": 0.012345678912345}
    )
    draft = DesignSetupDraft(project)
    for section in draft.sections():
        assert section["fields"]
        assert len({field["key"] for field in section["fields"]}) == len(section["fields"])
    assert draft.build() is project
    assert not draft.dirty


def test_geometry_target_and_study_commit_together(project):
    draft = DesignSetupDraft(project)
    draft.update({"cylinder_radius_mm": "130", "target_amplitude_native": "35", "validation_resolution": "9"})
    after = draft.build()
    assert after.revision == project.revision + 1
    assert after.config.cylinder_mesh_parameter_list[1] == pytest.approx(0.13)
    assert after.config.target_gradient_strength == pytest.approx(0.035)
    assert after.study["validation_resolution"] == 9
    assert project.config.target_gradient_strength != after.config.target_gradient_strength
    assert after.config.tikonov_reg_factor == project.config.tikonov_reg_factor


@pytest.mark.parametrize(
    "key,value", [("levels", "3.5"), ("target_amplitude_native", "1e"), ("cylinder_radius_mm", "nan")]
)
def test_invalid_entry_preserves_draft_and_source(project, key, value):
    draft = DesignSetupDraft(project)
    draft.update({key: value})
    with pytest.raises(SetupInputError) as error:
        draft.build()
    assert error.value.key == key
    assert draft.values[key] == value
    assert draft.project is project


def test_round_wire_edit_updates_physical_profile_and_both_dimensions(project):
    profile = round_wire_profile(0.001, 24, equal_area=True)
    project = project.update_config(
        {
            "cross_sectional_points": profile,
            "conductor_cross_section_width": 0.001,
            "conductor_cross_section_height": 0.001,
        }
    )
    draft = DesignSetupDraft(project)
    assert draft.values["setup_profile"] == "round"
    draft.update({"conductor_width_mm": "1.4", "levels": "17"})
    after = draft.build()
    assert after.config.conductor_cross_section_width == after.config.conductor_cross_section_height == 0.0014
    np.testing.assert_array_equal(after.config.cross_sectional_points, round_wire_profile(0.0014, 24, equal_area=True))


def test_rectangle_uses_native_sentinel_and_never_a_metre_sized_polygon(project):
    project = project.update_config(
        {
            "cross_sectional_points": round_wire_profile(0.001),
            "conductor_cross_section_width": 0.001,
            "conductor_cross_section_height": 0.001,
        }
    )
    draft = DesignSetupDraft(project)
    draft.update({"setup_profile": "rectangle", "conductor_width_mm": "2", "conductor_height_mm": "1"})
    np.testing.assert_array_equal(draft.build().config.cross_sectional_points, ((0.0, 0.0),))


def test_support_change_retains_unrelated_draft_and_only_uses_selected_dimensions(project):
    draft = DesignSetupDraft(project)
    draft.update({"cylinder_radius_mm": "147", "levels": "16"})
    draft.update({"coil_mesh_file": "create bi-planar mesh"})
    assert draft.values["levels"] == "16"
    assert "cylinder_radius_mm" not in draft.changes()
    draft.update({"coil_mesh_file": project.config.coil_mesh_file})
    assert draft.values["cylinder_radius_mm"] == "147"
    assert draft.build().config.levels == 16


def test_hidden_smoothing_edits_do_not_change_an_inactive_feature(project):
    draft = DesignSetupDraft(project)
    draft.update({"contour_treatment": "off", "contour_smoothing_factor": "0.61"})
    assert "contour_smoothing_factor" not in draft.changes()
    assert draft.build().config.contour_smoothing_factor == project.config.contour_smoothing_factor


def test_method_change_never_silently_removes_physical_constraints(project):
    config = replace(project.config, research_physics={"objective": "power", "max_current_density_A_per_m": 400})
    project = ProjectDocument.from_config(config)
    draft = DesignSetupDraft(project)
    draft.update({"sf_opt_method": "target_error_constrained_efficiency"})
    with pytest.raises(SetupInputError, match="physical objective"):
        draft.build()
    assert project.config.research_physics["max_current_density_A_per_m"] == 400


def test_integer_text_and_float_text_equal_to_saved_values_are_not_changes(project):
    draft = DesignSetupDraft(project)
    draft.update({"levels": str(draft.values["levels"]) + ".0"})
    assert not draft.dirty
    assert draft.build() is project


def test_changed_field_does_not_reapply_previous_harmonic_origin(project):
    project = project.update_config({"field_shape_function": "x*z", "target_gradient_normalization": "none"})
    draft = DesignSetupDraft(project)
    draft.update({"field_origin_x_mm": "12"})
    draft.update({"setup_field": "y"})
    assert draft.build().config.field_shape_function == "y"


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_gradient_preset_updates_the_normalization_axis(project, axis):
    draft = DesignSetupDraft(project)
    draft.update({"setup_field": axis})
    after = draft.build()
    assert after.config.field_shape_function == axis
    assert after.config.target_gradient_axis == axis
    assert after.config.target_gradient_strength == project.config.target_gradient_strength


def test_explicit_error_budget_is_the_common_control(project):
    project = project.update_config({"target_error_constraint_metric": "rms", "target_error_rms_limit_T": 1e-6})
    draft = DesignSetupDraft(project)
    common = {field["key"] for field in draft.entries("optimization") if field["common"]}
    assert "target_error_rms_limit_T" in common
    assert "target_error_tolerance_percent" not in common
    assert draft.build() is project


def test_custom_field_constraints_do_not_offer_inactive_default_budgets(project):
    config = replace(
        project.config, research_physics={"objective": "power", "include_default_field_constraints": False}
    )
    draft = DesignSetupDraft(ProjectDocument.from_config(config))
    assert not any(item.key.startswith("target_error_") for item in draft.fields("optimization"))
    assert "Custom field constraints" in draft.summary("optimization")
