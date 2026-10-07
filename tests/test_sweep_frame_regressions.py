from __future__ import annotations

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import build_cylinder_mesh, normalize_mesh
from coilgen.manufacturing import (
    _enforce_periodic_path_frames,
    _frame_rotation_angles_deg,
    _mounting_centerline_offset,
    _parallel_transport_path_frames,
    _path_frames,
    _path_tangents,
    _sweep_ring_vertices,
)
from coilgen.models import CoilPart, Mesh, WirePathGeometry
from coilgen.sweep_frames import _connection_solve, constrained_frames, free_transport_frames, transport_normal


@pytest.mark.parametrize("strength", [1.0, 0.35])
def test_antipodal_attachments_distribute_twist_without_a_normalization_flip(strength):
    points = np.column_stack((np.arange(5) * 0.01, np.zeros(5), np.zeros(5)))
    tangent = np.tile([1.0, 0.0, 0.0], (5, 1))
    guides = np.tile([0.0, -1.0, 0.0], (5, 1))
    guides[:2] = [0.0, 1.0, 0.0]
    normals, transverse = constrained_frames(
        points,
        tangent,
        guides,
        np.array([True, False, False, True]),
        strength,
        0.002,
        closed=False,
    )
    np.testing.assert_allclose(normals[[0, 1, 3, 4]], guides[[0, 1, 3, 4]], atol=1e-12)
    np.testing.assert_allclose(np.abs(normals[2]), [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        _frame_rotation_angles_deg(tangent, normals, transverse, closed=False), [0.0, 90.0, 90.0, 0.0]
    )


def test_free_span_twist_uses_physical_arclength_and_preserves_single_ended_tails():
    positions = np.array([0.0, 0.01, 0.012, 0.03, 0.04, 0.05])
    points = np.column_stack((positions, np.zeros(6), np.zeros(6)))
    tangent = np.tile([1.0, 0.0, 0.0], (6, 1))
    guides = np.tile([0.0, -1.0, 0.0], (6, 1))
    guides[:2] = [0.0, 1.0, 0.0]
    normals, _transverse = constrained_frames(
        points,
        tangent,
        guides,
        np.array([True, False, False, True, False]),
        1.0,
        0.002,
        closed=False,
    )
    angle = np.pi * (0.012 - 0.01) / (0.03 - 0.01)
    np.testing.assert_allclose(normals[2], [0.0, np.cos(angle), np.sin(angle)], atol=1e-12)
    np.testing.assert_allclose(normals[-1], guides[-1], atol=1e-12)


def test_subprecision_surface_strength_has_a_finite_rmf_limit():
    points = np.column_stack((np.arange(4) / 64.0, np.zeros(4), np.zeros(4)))
    tangent = np.tile([1.0, 0.0, 0.0], (4, 1))
    guides = np.tile([0.0, 1.0, 0.0], (4, 1))
    with np.errstate(all="raise"):
        normals, transverse = constrained_frames(
            points,
            tangent,
            guides,
            np.ones(3, dtype=bool),
            1e-30,
            0.002,
            closed=False,
        )
    np.testing.assert_array_equal(normals, guides)
    np.testing.assert_allclose(np.linalg.norm(transverse, axis=1), 1.0)


def _plane() -> CoilPart:
    return CoilPart(
        mesh=normalize_mesh(
            Mesh(
                np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]),
                np.array([[0, 1, 2], [0, 2, 3]]),
            )
        )
    )


def _cylinder(count=32):
    theta = (np.arange(count) + 0.2) * 2 * np.pi / count
    points = np.column_stack((0.1 * np.cos(theta), 0.1 * np.sin(theta), 0.03 * np.sin(2 * theta)))
    parameters = (0.2, 0.1, 32, 8, 1.0, 0.0, 0.0, 0.0)
    part = CoilPart(mesh=normalize_mesh(build_cylinder_mesh(*parameters)))
    config = CoilGenConfig(
        coil_mesh_file="create cylinder mesh", cylinder_mesh_parameter_list=parameters, surface_is_cylinder_flag=True
    )
    guides = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(count)))
    return points, part, config, guides


def _assert_frames(tangents, normals, transverse):
    assert np.all(np.isfinite(normals))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=2e-12)
    np.testing.assert_allclose(np.sum(tangents * normals, axis=1), 0.0, atol=2e-12)
    np.testing.assert_allclose(np.cross(tangents, normals), transverse, atol=2e-12)


@pytest.mark.parametrize(("previous", "following"), [(0.01, 0.4), (0.4, 0.01), (1e-5, 0.1)])
def test_nonuniform_circle_tangent_is_not_biased_by_neighbor_lengths(previous, following):
    theta = np.array([-previous, 0.0, following])
    points = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(3)))
    tangent = _path_tangents(points, closed=False)[1]
    np.testing.assert_allclose(tangent, [0.0, 1.0, 0.0], atol=1e-10)


def test_full_frame_audit_includes_planar_tangent_rotation():
    theta = np.arange(8) * np.pi / 4
    points = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(8)))
    tangent = _path_tangents(points, closed=True)
    normals = np.tile([0.0, 0.0, 1.0], (8, 1))
    transverse = np.cross(tangent, normals)
    np.testing.assert_allclose(_frame_rotation_angles_deg(tangent, normals, transverse, closed=True), 45.0)
    assert len(_frame_rotation_angles_deg(tangent, normals, transverse, closed=False)) == 7


@pytest.mark.parametrize("strength", [0.35, 1.0])
def test_closed_cylinder_frames_are_cyclic_index_consistent_and_outward(strength):
    points, part, config, guides = _cylinder()
    config.sweep_surface_normal_constraint_strength = strength
    tangent = _path_tangents(points, closed=True)
    geometry = [WirePathGeometry.SURFACE] * len(points)
    normals, transverse = _path_frames(part, points, tangent, geometry, config, closed=True)
    _assert_frames(tangent, normals, transverse)
    assert np.min(np.sum(normals * guides, axis=1)) > 0.99
    for shift in (1, 7, 16):
        shifted, _ = _path_frames(
            part, np.roll(points, shift, axis=0), np.roll(tangent, shift, axis=0), geometry, config, closed=True
        )
        np.testing.assert_allclose(shifted, np.roll(normals, shift, axis=0), atol=1e-10)
    if strength == 1.0:
        projected = guides - tangent * np.sum(tangent * guides, axis=1)[:, None]
        projected /= np.linalg.norm(projected, axis=1)[:, None]
        np.testing.assert_allclose(normals, projected, atol=1e-12)


def test_supplied_closed_attached_frames_are_not_globally_twisted():
    points, _, _, guides = _cylinder(16)
    tangent = _path_tangents(points, closed=True)
    projected = guides - tangent * np.sum(tangent * guides, axis=1)[:, None]
    projected /= np.linalg.norm(projected, axis=1)[:, None]
    normals, _ = _enforce_periodic_path_frames(points, tangent, projected)
    np.testing.assert_allclose(normals, projected, atol=1e-12)


def test_surface_attachment_is_independent_of_collinear_tail_sample_count_and_keeps_mounting_side():
    prefix = np.array([[0.0, -0.04, 0.05], [-0.04, -0.04, 0.05], [-0.04, 0.0, 0.0], [0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])
    part = _plane()
    section = np.array([[-0.00098, -0.0009], [0.00098, -0.0009], [0.0008, 0.0009], [-0.0008, 0.0009]])
    config = CoilGenConfig(cross_sectional_points=section, conductor_mounting_mode="surface_outer")
    baseline = None
    for count in (2, 4, 8, 16):
        tail = np.column_stack((np.linspace(0.005, 0.1, count + 1)[1:], np.zeros(count), np.zeros(count)))
        points = np.vstack((prefix, tail))
        geometry = [WirePathGeometry.FREE_3D] * 3 + [WirePathGeometry.SURFACE] * (len(points) - 4)
        tangent = _path_tangents(points, closed=False)
        normals, transverse = _parallel_transport_path_frames(part, points, tangent, geometry, config, closed=False)
        _assert_frames(tangent, normals, transverse)
        if baseline is None:
            baseline = normals[: len(prefix)].copy()
        np.testing.assert_allclose(normals[: len(prefix)], baseline, atol=1e-11)
        np.testing.assert_allclose(normals[-1], [0.0, 0.0, 1.0], atol=1e-12)
        vertices = _sweep_ring_vertices(
            points, normals, transverse, section, _mounting_centerline_offset(config, section)
        )
        assert vertices[-len(section) :, 2].min() >= -1e-12


def test_double_reflection_preserves_planar_rmf_and_handles_exact_reversal():
    normal = np.array([0.0, 0.0, 1.0])
    for angle in (0.0, 0.3, np.pi, -np.pi / 2):
        tangent = np.array([np.cos(angle), np.sin(angle), 0.0])
        transported = transport_normal(normal, np.array([1.0, 0.0, 0.0]), tangent, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(transported, normal, atol=1e-12)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    tangent = _path_tangents(points, closed=False)
    normals, transverse = constrained_frames(
        points, tangent, np.tile(normal, (4, 1)), np.zeros(3), 0.0, 0.001, closed=False
    )
    _assert_frames(tangent, normals, transverse)


def test_unconstrained_closed_rmf_uses_geometric_not_index_dependent_anchor():
    points, _, _, guides = _cylinder(32)
    tangent = _path_tangents(points, closed=True)
    normals, transverse = constrained_frames(points, tangent, guides, np.zeros(32), 0.0, 0.001, closed=True)
    _assert_frames(tangent, normals, transverse)
    rolled, _ = constrained_frames(
        np.roll(points, 11, axis=0),
        np.roll(tangent, 11, axis=0),
        np.roll(guides, 11, axis=0),
        np.zeros(32),
        0.0,
        0.001,
        closed=True,
    )
    np.testing.assert_allclose(rolled, np.roll(normals, 11, axis=0), atol=1e-12)


def test_cyclic_solver_never_builds_dense_path_sized_system(monkeypatch):
    original = np.linalg.solve

    def small_solve(matrix, right):
        assert matrix.shape == (2, 2)
        return original(matrix, right)

    monkeypatch.setattr(np.linalg, "solve", small_solve)
    count = 1000
    theta = np.arange(count) * 2 * np.pi / count
    points = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(count)))
    tangent = _path_tangents(points, closed=True)
    guide = np.tile([0.0, 0.0, 1.0], (count, 1))
    normals, transverse = constrained_frames(points, tangent, guide, np.ones(count), 0.35, 0.002, closed=True)
    _assert_frames(tangent, normals, transverse)
    np.testing.assert_allclose(normals, guide, atol=1e-12)


def test_cyclic_connection_solve_matches_independent_small_dense_reference():
    rng = np.random.default_rng(943)
    count = 8
    lower = -0.3 * np.exp(1j * rng.uniform(-1.0, 1.0, count - 1))
    closing = -0.2 * np.exp(0.7j)
    diagonal = np.full(count, 2.0)
    right = rng.normal(size=count) + 1j * rng.normal(size=count)
    matrix = np.diag(diagonal.astype(complex)) + np.diag(lower, -1) + np.diag(np.conjugate(lower), 1)
    matrix[0, -1] = closing
    matrix[-1, 0] = np.conjugate(closing)
    np.testing.assert_allclose(
        _connection_solve(diagonal, lower, right, closing), np.linalg.solve(matrix, right), atol=1e-14
    )


def test_double_reflection_converges_at_fourth_order_to_analytic_helix_rmf():
    errors = []
    for count in (40, 80):
        parameter = np.linspace(0.0, 2 * np.pi, count + 1)
        pitch = 0.3
        speed = np.sqrt(1 + pitch**2)
        points = np.column_stack((np.cos(parameter), np.sin(parameter), pitch * parameter))
        tangent = np.column_stack((-np.sin(parameter), np.cos(parameter), np.full(count + 1, pitch))) / speed
        normal = np.column_stack((-np.cos(parameter), -np.sin(parameter), np.zeros(count + 1)))
        transverse = np.cross(tangent, normal)
        roll = -pitch / speed * parameter
        exact = np.cos(roll)[:, None] * normal + np.sin(roll)[:, None] * transverse
        actual, _ = free_transport_frames(points, tangent, normal, closed=False)
        errors.append(float(np.linalg.norm(actual - exact, axis=1).max()))
    assert errors[0] < 2e-7
    assert errors[1] < errors[0] / 12


def test_arclength_guide_energy_converges_under_sample_refinement():
    solutions = []
    for count in (17, 33, 65):
        x = np.linspace(0.0, 1.0, count)
        points = np.column_stack((x, np.zeros(count), np.zeros(count)))
        tangent = np.tile([1.0, 0.0, 0.0], (count, 1))
        angle = 0.3 * np.sin(2 * np.pi * x)
        guide = np.column_stack((np.zeros(count), np.sin(angle), np.cos(angle)))
        normals, _ = constrained_frames(points, tangent, guide, np.ones(count - 1), 0.35, 0.2, closed=False)
        solutions.append(normals)
    coarse_difference = np.linalg.norm(solutions[0] - solutions[1][::2], axis=1).max()
    fine_difference = np.linalg.norm(solutions[1] - solutions[2][::2], axis=1).max()
    assert coarse_difference < 0.003
    assert fine_difference < coarse_difference / 3
