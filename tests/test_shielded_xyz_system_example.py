"""Each foil channel owns a primary/shield pair and samples the same enclosure."""

import numpy as np

from coilgen.application import ProjectAssembly
from coilgen.application.gradient_geometry import CylindricalGeometry
from coilgen.application.templates import template_assembly
from coilgen.geometry import load_mesh_from_config, load_secondary_target_mesh, split_disconnected_mesh
from coilgen.target import define_target_field


def test_shielded_xyz_system_round_trip_keeps_six_layers_and_three_independent_targets(tmp_path):
    assembly = template_assembly("shielded_xyz_gradient", project_root=tmp_path)
    restored = ProjectAssembly.load(assembly.save(tmp_path / "xyz.coilgen"))
    assert [channel.id for channel in restored.channels] == ["gx", "gy", "gz"]
    assert len({channel.project.id for channel in restored.channels}) == 3
    reference = None
    for i, channel in enumerate(restored.channels):
        project, axis = channel.project, "xyz"[i]
        geometry = CylindricalGeometry.from_project(project)
        assert geometry.primary.radius_m == (0.080, 0.086, 0.092)[i]
        assert geometry.shield.radius_m == (0.110, 0.116, 0.122)[i]
        assert geometry.primary.length_m == geometry.shield.length_m == 0.300
        assert geometry.suppression.radius_m == 0.150 and geometry.suppression.length_m == 0.400
        config = project.config
        assert config.conductor_cross_section_width == 0.002
        assert config.conductor_cross_section_height == config.conductor_thickness == 0.0002
        assert config.routing_circuit_scope == "per_part" and config.level_set_method == "primary"
        assert config.suppression_field_component == "vector" and config.suppression_surface_weighting == "area"
        assert config.suppression_field_rms_limit_T == 20e-6
        assert config.target_gradient_axis == axis and config.target_field_component == "Bz"
        mesh = load_mesh_from_config(config, tmp_path)
        assert mesh.metadata["part_roles"] == ["primary", "active_shield"]
        parts = split_disconnected_mesh(mesh)
        suppression = load_secondary_target_mesh(config, tmp_path, coil_meshes=parts)
        assert suppression.boundaries == []
        target = define_target_field(parts, None, suppression, config)
        roi = ~target.suppressed
        np.testing.assert_allclose(target.b[roi, 2], 0.020 * target.coords[roi, i], atol=1e-15)
        np.testing.assert_array_equal(target.b[target.suppressed], 0)
        assert target.suppressed.sum() == suppression.n_vertices
        if reference is None:
            reference = target.coords
        np.testing.assert_array_equal(target.coords, reference)
