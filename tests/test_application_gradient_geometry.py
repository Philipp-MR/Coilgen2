"""Parameterized cylinder inputs preserve physical span, pose and owned assets."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen.application.gradient_geometry import CylinderSurface, CylindricalGeometry
from coilgen.application.presets import preset_config
from coilgen.application.project import ProjectDocument
from coilgen.application.service import DesignService
from coilgen.application.templates import template_assembly
from coilgen.geometry import load_mesh_from_config, load_secondary_target_mesh, split_disconnected_mesh
from coilgen.models import Mesh


def _project():
    return ProjectDocument.from_config(preset_config("cylinder"))


@pytest.mark.parametrize("axis", ((0, 0, 1), (0, 0, -1), (1, 2, 3)))
@pytest.mark.parametrize("axial", (1, 7, 16))
def test_requested_dimensions_pose_and_closed_boundary_are_exact(axis, axial):
    surface = CylinderSurface(0.08, 0.3, 12, axial, (0.1, -0.2, 0.3), axis)
    mesh = surface.mesh(closed=True)
    relative = mesh.vertices - surface.center_m
    axial_position = relative @ surface.axis
    radius = np.linalg.norm(relative - axial_position[:, None] * surface.axis, axis=1)
    np.testing.assert_allclose(np.ptp(axial_position), 0.3, atol=1e-14)
    np.testing.assert_allclose(radius[:-2], 0.08, atol=1e-14)
    assert mesh.boundaries == []
    assert np.all(
        np.linalg.norm(
            np.cross(*(mesh.vertices[mesh.faces][:, i] - mesh.vertices[mesh.faces][:, 0] for i in (1, 2))), axis=1
        )
        > 0
    )
    assert CylinderSurface.from_dict(surface.to_dict()) == surface


def test_geometry_applies_one_atomic_revision_and_round_trips_portably(tmp_path):
    project = _project()
    original = project.to_dict()
    spec = CylindricalGeometry(
        CylinderSurface(0.08, 0.3, 12, 6), CylinderSurface(0.11, 0.3, 12, 6), CylinderSurface(0.15, 0.4, 12, 6)
    )
    updated = spec.apply(project, expected_fingerprint=project.fingerprint)
    assert project.to_dict() == original
    assert updated.id == project.id and updated.revision == project.revision + 1
    assert updated.study == project.study
    for key in (
        "field_shape_function",
        "target_gradient_strength",
        "target_field_units",
        "research_physics",
        "target_error_tolerance_percent",
        "suppression_field_rms_limit_T",
    ):
        assert updated.values[key] == project.values[key]
    assert updated.values["suppression_field_component"] == "vector"
    assert updated.values["routing_circuit_scope"] == "per_part"
    assert len(updated.to_dict()["assets"]) == 2
    reopened = ProjectDocument.load(updated.save(tmp_path / "portable.json"))
    assert CylindricalGeometry.from_project(reopened) == spec
    assert spec.apply(reopened) is reopened
    parts = split_disconnected_mesh(load_mesh_from_config(reopened.config, tmp_path))
    assert len(parts) == 2
    assert load_secondary_target_mesh(reopened.config, tmp_path, coil_meshes=parts).boundaries == []
    with pytest.raises(ValueError, match="changed"):
        spec.apply(updated, expected_fingerprint=project.fingerprint)


@pytest.mark.parametrize("rotation", ((0, 0, 1, 0.7), (1, 2, 3, 1.3), (1, 0, 0, np.pi)))
def test_recognizing_native_cylinder_preserves_azimuth_as_well_as_axis(rotation):
    config = replace(preset_config("cylinder"), cylinder_mesh_parameter_list=(0.3, 0.08, 12, 7, *rotation))
    project = ProjectDocument.from_config(config)
    before = load_mesh_from_config(project.config, project.project_root)
    spec = CylindricalGeometry.from_project(project)
    updated = spec.apply(project)
    after = load_mesh_from_config(updated.config, project.project_root)
    np.testing.assert_allclose(after.vertices, before.vertices, atol=1e-14)
    np.testing.assert_array_equal(after.faces, before.faces)


def test_existing_shield_template_becomes_editable_without_geometry_change():
    project = template_assembly("shielded_gy").channels[0].project
    spec = CylindricalGeometry.from_project(project)
    assert spec is not None and spec.shield.radius_m == pytest.approx(0.11)
    before = load_mesh_from_config(project.config, project.project_root)
    updated = spec.apply(project)
    np.testing.assert_allclose(
        load_mesh_from_config(updated.config, project.project_root).vertices, before.vertices, atol=1e-14
    )
    assert updated.config.suppression_field_rms_limit_T == 20e-6


def test_arbitrary_imports_are_not_inferred_or_replaced():
    mesh = Mesh(np.asarray(((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0))), np.asarray(((0, 1, 2),)))
    project = _project().with_mesh(mesh, field="coil_mesh_file")
    before = project.to_dict()
    assert CylindricalGeometry.from_project(project) is None
    assert project.to_dict() == before


def test_suppression_polyhedron_must_enclose_supports_and_optional_removal_is_explicit():
    primary = CylinderSurface(0.1, 0.3)
    with pytest.raises(ValueError, match="enclose"):
        CylindricalGeometry(primary, suppression=CylinderSurface(0.11, 0.4, 4))
    with pytest.raises(ValueError, match="enclose"):
        CylindricalGeometry(primary, suppression=CylinderSurface(0.2, 0.3))
    with pytest.raises(ValueError, match="Shield radius"):
        CylindricalGeometry(primary, CylinderSurface(0.08, 0.3))
    with pytest.raises(ValueError, match="world center"):
        CylindricalGeometry(primary, CylinderSurface(0.2, 0.3, center_m=(0, 0, 0.01)))
    shielded = CylindricalGeometry(primary, CylinderSurface(0.15, 0.3), CylinderSurface(0.2, 0.4)).apply(_project())
    simple = CylindricalGeometry(primary).apply(shielded)
    assert simple.values["secondary_target_mesh_file"] == "none"
    assert len(simple.to_dict()["assets"]) == 1


@pytest.mark.parametrize(
    "changes",
    ({"radius_m": 0}, {"length_m": np.inf}, {"axis": (0, 0, 0)}, {"angular_divisions": True}, {"axial_divisions": 257}),
)
def test_invalid_geometry_is_rejected_before_asset_creation(changes):
    with pytest.raises(ValueError):
        CylinderSurface(**({"radius_m": 0.1, "length_m": 0.3} | changes))


def test_generated_geometry_prepares_same_physical_target_and_external_zero_samples(tmp_path):
    spec = CylindricalGeometry(
        CylinderSurface(0.08, 0.3, 8, 3), CylinderSurface(0.11, 0.3, 8, 3), CylinderSurface(0.15, 0.4, 8, 3)
    )
    project = spec.apply(_project()).update_config({"target_region_resolution": 3, "gauss_order": 1})
    DesignService.run(project, tmp_path / "prepared", threads=1, pipeline_stage="prepared", fabricate=False)
    arrays = DesignService.load_run(tmp_path / "prepared").arrays
    suppressed = arrays["target_suppressed"]
    assert np.count_nonzero(suppressed) == spec.suppression.mesh(closed=True).n_vertices
    np.testing.assert_array_equal(arrays["target_field_T"][suppressed], 0)
    np.testing.assert_allclose(
        arrays["target_field_T"][~suppressed, 2], 0.02 * arrays["target_points_m"][~suppressed, 1], atol=1e-15
    )
