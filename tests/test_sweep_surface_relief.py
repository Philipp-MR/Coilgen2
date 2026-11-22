"""Physical relief must preserve direction, source corridor and hard limits."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import rotation_matrix_by_vector
from coilgen.manufacturing import create_swept_coil_bodies, conductor_cross_section_points, realize_coil_paths
from coilgen.models import CoilPart, WirePath, WirePathGeometry
from coilgen.sweep_surface_relief import _Envelope, _source_corridor, repair_surface_sweep
from _coilgen_test_support.group_route_support_cap import cap
from _coilgen_test_support.group_route_support_curved import _frustum


@pytest.mark.parametrize("mesh", [_frustum(), _frustum(0.018, 0.022), cap()])
def test_relief_envelope_uses_actual_geometry_and_is_rigid_transform_equivariant(mesh):
    envelope = _Envelope.from_mesh(mesh, 1e-8)
    assert envelope is not None
    points = mesh.vertices + 0.0007 * envelope.normal(mesh.vertices)
    before_height, before_normal = envelope.height(points), envelope.normal(points)
    rotation = rotation_matrix_by_vector(np.array([1.0, -2, 3]), 0.71)
    shift = np.array([1.2, -0.3, 0.8])
    mesh = deepcopy(mesh)
    mesh.vertices = mesh.vertices @ rotation.T + shift
    mesh.metadata = {"geometry_type": "incorrect"}
    other = _Envelope.from_mesh(mesh, 1e-8)
    assert other is not None
    np.testing.assert_allclose(other.height(points @ rotation.T + shift), before_height, atol=1e-12)
    np.testing.assert_allclose(other.normal(points @ rotation.T + shift), before_normal @ rotation.T, atol=1e-12)


def test_corridor_checks_skipped_source_corners_between_dense_candidate_vertices():
    source = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]) * 0.01
    # Replacing each corner with a diagonal can make every candidate vertex
    # coincide with the source, while source corners lie outside the corridor.
    candidate = (source[:-1] + source[1:]) / 2
    result = _source_corridor(source, candidate)
    assert result is not None
    assert result[0].max() >= 0.0035
    assert _source_corridor(source[:-1], candidate) is None


def test_radial_transition_preserves_mounting_height_without_changing_axial_boundary():
    envelope = _Envelope.from_mesh(_frustum(count=48), 1e-8)
    config = CoilGenConfig(sweep_min_bend_radius=0.003, normal_shift_length=0.004)
    section = np.array([[-0.001, -0.001], [0.001, -0.001], [0.001, 0.001], [-0.001, 0.001]])
    points = envelope.global_points(np.array([[envelope.radius + 0.001, 0, 0.005]]))
    broad = envelope.constrain(points, 0.001, section, config)
    narrow = envelope.constrain(points, 0.001, section, config, radial_width=0.003 / 16)
    np.testing.assert_allclose(envelope.local(narrow)[:, 2], envelope.local(broad)[:, 2], atol=1e-12)
    assert 0.001 <= envelope.height(narrow)[0] < 0.0012
    assert envelope.height(narrow)[0] < envelope.height(broad)[0]


def test_corridor_includes_complete_translated_closed_curve():
    theta = np.linspace(0, 2 * np.pi, 101)
    source = np.column_stack((0.02 * np.cos(theta), 0.02 * np.sin(theta), np.zeros(len(theta))))
    candidate = source[:-1] + [0, 0, 0.0004]
    result = _source_corridor(source, candidate)
    assert result is not None
    assert result[0].max() == pytest.approx(0.0004, abs=1e-12)


def test_closed_convex_relief_is_detached_and_cannot_relax_corridor_or_lift():
    theta = np.linspace(0, 2 * np.pi, 121)
    source = np.column_stack((0.0205 * np.cos(theta), 0.0205 * np.sin(theta), np.zeros(len(theta))))
    part = CoilPart(
        mesh=_frustum(count=48),
        wire_path=WirePath(source, segment_geometry=[WirePathGeometry.OFFSET_SURFACE] * (len(source) - 1)),
    )
    config = CoilGenConfig(
        conductor_mounting_mode="surface_outer",
        normal_shift_length=0.004,
        conductor_cross_section_width=0.0002,
        conductor_cross_section_height=0.0002,
        sweep_min_bend_radius=0.001,
        sweep_bend_mode="fillet",
        sweep_corner_cluster_max_source_deviation_m=0.002,
        sweep_max_chord_error=0.00005,
        sweep_max_frame_rotation_deg=10,
        sweep_max_segment_length=0.002,
        sweep_surface_realization="mesh",
    )
    create_swept_coil_bodies([part], config)
    original_path = part.fabrication_path
    original_config = config.as_dict()
    trial = repair_surface_sweep(part, conductor_cross_section_points(config), config, part.manufacturing_diagnostics)
    assert trial is not None
    assert trial.manufacturing_diagnostics["manufacturing_ok"]
    assert trial.manufacturing_diagnostics["exact_collision_completed"]
    assert trial.manufacturing_diagnostics["sweep_relief_source_deviation_bound_m"] < 0.002
    assert part.fabrication_path is original_path
    np.testing.assert_array_equal(part.wire_path.v, source)
    assert config.as_dict() == original_config
    path_only = deepcopy(part)
    realize_coil_paths([path_only], config)
    np.testing.assert_array_equal(path_only.fabrication_path.points_m, part.fabrication_path.points_m)
    np.testing.assert_array_equal(path_only.fabrication_path.normals, part.fabrication_path.normals)
    assert path_only.swept_body is None and path_only.swept_bodies == []
    assert "manufacturing_ok" not in path_only.manufacturing_diagnostics
    assert (
        repair_surface_sweep(
            part,
            conductor_cross_section_points(config),
            replace(config, sweep_corner_cluster_max_source_deviation_m=1e-8),
            {},
        )
        is None
    )
    assert (
        repair_surface_sweep(
            part, conductor_cross_section_points(config), replace(config, normal_shift_length=0.0001), {}
        )
        is None
    )


def test_modified_support_cannot_receive_an_analytic_relief_certificate():
    mesh = _frustum()
    mesh.vertices[0, 0] += 0.0001
    assert _Envelope.from_mesh(mesh, 1e-8) is None
