"""Public cylinder dimensions stay physical; raw legacy config stays reproducible."""

from pathlib import Path

import numpy as np
import pytest

from coilgen import CoilGenConfig, CylinderCoilDesign, PipelineGeometrySpec, validate_config
from coilgen.geometry import load_mesh_from_config, rotation_matrix_by_vector


@pytest.mark.parametrize("axial", [1, 4, 8, 40, np.int64(12)])
@pytest.mark.parametrize("angle", [0.0, 0.73])
def test_public_cylinder_realizes_length_radius_and_center_under_refinement_and_rotation(axial, angle):
    axis = np.array([1.0, 2.0, -3.0])
    spec = PipelineGeometrySpec.cylinder(
        length_m=0.3,
        radius_m=0.2,
        angular_divisions=16,
        length_divisions=axial,
        rotation_axis=axis,
        rotation_angle_rad=angle,
    )
    config = CoilGenConfig(**spec.to_config_overrides())
    mesh = load_mesh_from_config(config, project_root=Path.cwd())
    direction = rotation_matrix_by_vector(axis, angle) @ np.array([0.0, 0.0, 1.0])
    positions = mesh.vertices @ direction
    radial = mesh.vertices - positions[:, None] * direction
    np.testing.assert_allclose([positions.min(), positions.max()], [-0.15, 0.15], atol=2e-15, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(radial, axis=1), 0.2, atol=2e-15, rtol=0)
    np.testing.assert_allclose(mesh.vertices.mean(axis=0), 0.0, atol=2e-15, rtol=0)
    assert mesh.n_vertices == 16 * (axial + 1)
    assert mesh.n_faces == 32 * axial


def test_design_defaults_and_design_to_spec_conversion_preserve_physical_length_without_double_compensation():
    design = CylinderCoilDesign()
    direct = design.geometry_config()
    assert PipelineGeometrySpec.from_coil_design(design).to_config_overrides() == direct
    mesh = load_mesh_from_config(design.to_config(), project_root=Path.cwd())
    assert np.ptp(mesh.vertices[:, 2]) == pytest.approx(design.length_m, rel=1e-14)


@pytest.mark.parametrize("axial", [0, -1, True, 1.5, "4", np.nan])
def test_public_cylinder_rejects_invalid_axial_counts_before_height_conversion(axial):
    with pytest.raises(ValueError, match="length_divisions must be a positive integer"):
        PipelineGeometrySpec.cylinder(length_m=0.3, radius_m=0.2, angular_divisions=8, length_divisions=axial)


def test_raw_config_and_from_config_values_retain_legacy_height_and_mesh():
    parameters = (0.3, 0.2, 8, 4, 1, 0, 0, 0)
    spec = PipelineGeometrySpec.from_config_values(
        "create cylinder mesh",
        surface_is_cylinder=True,
        parameters={"cylinder_mesh_parameter_list": parameters},
    )
    config = CoilGenConfig(**spec.to_config_overrides())
    assert config.cylinder_mesh_parameter_list == parameters
    mesh = load_mesh_from_config(config, project_root=Path.cwd())
    assert np.ptp(mesh.vertices[:, 2]) == pytest.approx(0.3 * 4 / 5, rel=1e-14)


def test_raw_noninteger_count_still_reaches_existing_config_validation_unchanged():
    parameters = (0.3, 0.2, 8, 1.5, 1, 0, 0, 0)
    spec = PipelineGeometrySpec.from_config_values(
        "create cylinder mesh",
        surface_is_cylinder=True,
        parameters={"cylinder_mesh_parameter_list": parameters},
    )
    config = CoilGenConfig(**spec.to_config_overrides())
    assert config.cylinder_mesh_parameter_list == parameters
    raw = CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        surface_is_cylinder_flag=True,
        cylinder_mesh_parameter_list=parameters,
    )
    assert validate_config(config) == validate_config(raw)
