"""Template inputs retain SI targets, portable geometry and physical shielding."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from coilgen.application.assembly import ProjectAssembly
from coilgen.application.presets import EXAMPLE_TITLES, preset_config, preset_study
from coilgen.application.service import DesignService
from coilgen.application.templates import TEMPLATE_SPECS, template_assembly, template_spec
from coilgen.geometry import load_mesh_from_config, load_secondary_target_mesh, split_disconnected_mesh
from coilgen.pipeline_stages import PipelineCheckpoint
from coilgen.research.storage import json_value
from coilgen.target import define_target_field, evaluate_field_expression


@pytest.mark.parametrize("spec", TEMPLATE_SPECS, ids=lambda spec: spec.id)
def test_templates_validate_and_request_complete_studies(spec, tmp_path):
    assembly = template_assembly(spec.id, project_root=tmp_path)

    assert assembly.name == spec.title
    for channel in assembly.channels:
        project = channel.project
        DesignService.validate(project).raise_for_errors()
        assert project.study["pipeline_stage"] == "evaluated"
        assert project.study["fabricate"] is True
        assert project.config.constrained_quadratic_fallback_policy == "strict"
        assert project.config.field_evaluation_current_source == "stream_function"
        assert project.config.target_error_tolerance_percent / project.study["design_margin"] == pytest.approx(5.0)


@pytest.mark.parametrize("spec", TEMPLATE_SPECS, ids=lambda spec: spec.id)
def test_every_template_round_trips_as_one_json(spec, tmp_path):
    assembly = template_assembly(spec.id, project_root=tmp_path)
    path = assembly.save(tmp_path / f"{spec.id}.json")
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    restored = ProjectAssembly.from_dict(json.loads(path.read_text(encoding="utf-8")), project_root=relocated)

    assert restored.fingerprint == assembly.fingerprint
    assert [channel.project.model_fingerprint for channel in restored.channels] == [
        channel.project.model_fingerprint for channel in assembly.channels
    ]
    for channel in restored.channels:
        DesignService.validate(channel.project).raise_for_errors()
    if spec.id == "shielded_gy":
        project = restored.channels[0].project
        assert project.values["coil_mesh_file"].startswith("asset:")
        assert project.values["secondary_target_mesh_file"].startswith("asset:")
        assert len(project.to_dict()["assets"]) == 2
        assert len(split_disconnected_mesh(load_mesh_from_config(project.config, relocated))) == 2


@pytest.mark.parametrize("id_", EXAMPLE_TITLES)
def test_original_examples_keep_their_resolved_numerical_inputs(id_):
    project = template_assembly(id_).channels[0].project

    assert json_value(asdict(project.config)) == json_value(asdict(preset_config(id_)))
    assert {key: project.study[key] for key in preset_study(id_)} == preset_study(id_)


@pytest.mark.parametrize("support", ("cylinder", "biplanar"))
@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_gradient_templates_use_the_selected_world_axis(support, axis):
    id_ = support if axis == "y" else f"{support}_g{axis}"
    project = template_assembly(id_).channels[0].project
    config = project.config
    mesh = load_mesh_from_config(config, project.project_root)
    target = define_target_field(split_disconnected_mesh(mesh), None, None, config)

    np.testing.assert_allclose(target.b[:, 2], 0.02 * target.coords[:, "xyz".index(axis)], atol=1e-15)
    assert config.target_gradient_axis == axis
    assert config.target_gradient_strength == 0.02


@pytest.mark.parametrize("id_", ("shim_z0", "shim_z2", "shim_xz", "shim_yz", "shim_x2_y2", "shim_xy"))
def test_shim_targets_match_named_cartesian_modes_and_physical_peak(id_):
    radius = 0.02
    # Include extrema of the cross modes on the reference sphere, not just its poles.
    points = radius * np.array(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), (2**-0.5, 0, 2**-0.5), (0, 2**-0.5, 2**-0.5), (2**-0.5, 2**-0.5, 0))
    )
    config = template_assembly(id_).channels[0].project.config
    x, y, z = points.T
    expected = {
        "shim_z0": np.full(len(points), 100e-6),
        "shim_z2": 10e-6 * (2 * z**2 - x**2 - y**2) / (2 * radius**2),
        "shim_xz": 10e-6 * 2 * x * z / radius**2,
        "shim_yz": 10e-6 * 2 * y * z / radius**2,
        "shim_x2_y2": 10e-6 * (x**2 - y**2) / radius**2,
        "shim_xy": 10e-6 * 2 * x * y / radius**2,
    }[id_]

    actual = evaluate_field_expression(config.field_shape_function, points)

    np.testing.assert_allclose(actual, expected, atol=1e-19)
    assert np.max(np.abs(actual)) == pytest.approx(100e-6 if id_ == "shim_z0" else 10e-6)
    assert config.design_context == "mr_shim"
    assert config.target_region_radius == radius
    assert config.conductor_cross_section_width == config.conductor_cross_section_height == 0.001


def test_shielded_template_prepares_joint_supports_and_closed_external_zero_vector_target(tmp_path):
    project = template_assembly("shielded_gy", project_root=tmp_path).channels[0].project
    config = project.config
    mesh = load_mesh_from_config(config, tmp_path)
    parts = split_disconnected_mesh(mesh)
    suppression = load_secondary_target_mesh(config, tmp_path, coil_meshes=parts)
    assert len(parts) == 2
    for part, radius in zip(parts, (0.08, 0.11)):
        np.testing.assert_allclose(np.linalg.norm(part.vertices[:, :2], axis=1), radius, atol=1e-14)
        np.testing.assert_allclose(np.ptp(part.vertices[:, 2]), 0.3, atol=1e-14)
    assert suppression.boundaries == []
    # Check the actual finite cylinder, including its polygonal side facets.
    assert np.max(np.linalg.norm(mesh.vertices[:, :2], axis=1)) < 0.15 * np.cos(np.pi / 24)
    assert np.max(np.abs(mesh.vertices[:, 2])) < 0.2

    result = DesignService.run(project, tmp_path / "prepared", threads=1, pipeline_stage="prepared", fabricate=False)
    record = DesignService.load_run(tmp_path / "prepared")
    mask = record.arrays["target_suppressed"]
    assert result["pipeline_stage"] == "prepared"
    assert np.count_nonzero(mask) == suppression.n_vertices
    np.testing.assert_array_equal(record.arrays["target_field_T"][mask], 0)
    np.testing.assert_allclose(
        record.arrays["target_field_T"][~mask, 2], 0.02 * record.arrays["target_points_m"][~mask, 1], atol=1e-15
    )
    assert config.suppression_field_component == "vector"
    assert config.suppression_field_rms_limit_T == 20e-6
    assert config.routing_circuit_scope == "per_part"


def test_active_shield_sheet_reduces_external_vector_field_at_the_joint_solution_current(tmp_path):
    project = template_assembly("shielded_gy", project_root=tmp_path).channels[0].project
    result = DesignService.run(
        project, tmp_path / "sheet", threads=1, pipeline_stage="sheet", fabricate=False, validation_resolution=12
    )
    checkpoint = PipelineCheckpoint.load(tmp_path / "sheet" / "pipeline_checkpoint.cgstage")
    state = checkpoint.state
    mask = state.target_field.suppressed
    weights = state.target_field.suppression_area_weights[mask]
    fields = [np.einsum("cpi,i->pc", part.sensitivity_matrix, part.stream_function) for part in state.parts]

    def external_rms(field):
        return np.sqrt(weights @ np.sum(field[mask] ** 2, axis=1) / np.sum(weights))

    assert result["summary"]["sheet_accepted"] is True
    assert result["validation"]["passed"] is True
    assert result["validation"]["relative_field_error"] < 0.05
    assert all(np.ptp(part.stream_function) > 1 for part in state.parts)
    combined = external_rms(fields[0] + fields[1])
    assert combined <= project.config.suppression_field_rms_limit_T + 1e-8
    assert combined < external_rms(fields[0]) / 5
    np.testing.assert_allclose(combined, checkpoint.diagnostics["suppression_field_rms_T"], rtol=1e-11)
    np.testing.assert_allclose(fields[0] + fields[1], checkpoint.sf_b_field, atol=1e-15)


def test_unknown_template_is_rejected_without_side_effects():
    for factory in (template_spec, template_assembly):
        with pytest.raises(ValueError, match="Unknown design template"):
            factory("unknown")
