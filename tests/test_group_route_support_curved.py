"""Verified curved supports never certify an exterior body by vertices alone."""

from copy import deepcopy

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import build_conical_frustum_mesh, normalize_mesh, refine_mesh, rotation_matrix_by_vector
from coilgen.group_route_convex_shell import certify_shell_body, convex_circular_shell
from coilgen.group_route_support import certify_group_route_support, check_group_route_support_domain
from coilgen.manufacturing import create_swept_coil_bodies
from coilgen.models import CoilPart, WirePath, WirePathGeometry


def _frustum(bottom=0.02, top=0.02, count=16):
    return normalize_mesh(build_conical_frustum_mesh(0.06, bottom, top, count, 2, 1, 0, 0, 0, 0, 0, 0))


@pytest.mark.parametrize("bottom,top", [(0.02, 0.02), (0.018, 0.022), (0.022, 0.018)])
@pytest.mark.parametrize("mode", ["surface_outer", "surface_inner"])
def test_complete_cylinder_and_frustum_shells_are_eligible_from_actual_geometry(bottom, top, mode):
    mesh = _frustum(bottom, top)
    mesh.metadata = {"geometry_type": "untrusted"}
    result = check_group_route_support_domain(mesh, CoilGenConfig(conductor_mounting_mode=mode))
    assert result.accepted, result
    assert result.diagnostics["support_realization"] == "actual_triangle_mesh"
    assert result.diagnostics["support_side_plane_count"] == 16


def test_preflight_is_invariant_to_rigid_transform_permutation_and_collinear_subdivision():
    mesh = refine_mesh(_frustum(0.018, 0.022), 1)
    rotation = rotation_matrix_by_vector(np.array([1, -3, 2.0]), 0.48)
    mesh.vertices = mesh.vertices @ rotation.T + [0.21, -0.06, 0.05]
    generator = np.random.default_rng(32)
    permutation = generator.permutation(len(mesh.vertices))
    inverse = np.argsort(permutation)
    mesh.vertices = mesh.vertices[permutation]
    mesh.faces = inverse[mesh.faces[generator.permutation(len(mesh.faces))]]
    result = check_group_route_support_domain(mesh, CoilGenConfig(conductor_mounting_mode="surface_outer"))
    assert result.accepted, result


@pytest.mark.parametrize("defect", ["hole", "inward", "noncircular", "bowed", "twisted", "duplicate", "stale_metadata"])
def test_unknown_modified_or_ambiguous_curved_supports_fail_closed(defect):
    mesh = _frustum()
    if defect == "hole":
        mesh.faces = mesh.faces[2:]
    elif defect == "inward":
        mesh.faces = mesh.faces[:, ::-1]
    elif defect == "noncircular":
        mesh.vertices[:, 0] *= 1.2
    elif defect == "bowed":
        mesh.vertices[16:32, :2] *= 1.01
    elif defect == "twisted":
        mesh.vertices[32:] = mesh.vertices[32:] @ rotation_matrix_by_vector(np.array([0, 0, 1.0]), 0.05).T
    elif defect == "duplicate":
        mesh.faces = np.vstack((mesh.faces, mesh.faces[0]))
    else:
        mesh.vertices[0, 0] += 1e-5
        mesh.metadata = {"geometry_type": "cylinder", "parameters": {"radius_m": 0.02}}
    result = check_group_route_support_domain(mesh, CoilGenConfig(conductor_mounting_mode="surface_outer"))
    assert not result.accepted, result


def test_exterior_body_triangle_crossing_solid_is_rejected_even_with_all_vertices_outside():
    mesh = _frustum()
    shell, reason = convex_circular_shell(mesh, 1e-8)
    assert not reason
    points = np.array([[0.025, 0, 0], [-0.025, 0.002, 0], [-0.025, -0.002, 0]])
    assert np.all(np.linalg.norm(points[:, :2], axis=1) > 0.02)
    diagnostics = {}
    assert certify_shell_body(shell, points, np.array([[0, 1, 2]]), "surface_outer", 1e-8, 10000, diagnostics) == (
        "body_support_penetration"
    )
    assert "body_support_penetration_witness_m" in diagnostics


def test_exterior_triangle_around_convex_corner_is_clipped_instead_of_rejected_by_conservative_plane_test():
    mesh = _frustum(count=4)
    shell, reason = convex_circular_shell(mesh, 1e-8)
    assert not reason
    # All points are outside and the whole triangle is above the diamond's top
    # corner, but neither adjacent side plane separates its complete extent.
    points = np.array([[-0.002, 0.0205, 0], [0.002, 0.0205, 0], [0, 0.022, 0]])
    diagnostics = {}
    assert certify_shell_body(shell, points, np.array([[0, 1, 2]]), "surface_outer", 1e-8, 10000, diagnostics) == ""
    assert diagnostics["body_triangles_requiring_solid_clipping"] == 1
    assert diagnostics["whole_body_exterior_to_support"]


def test_exterior_and_footprint_budget_exhaustion_are_explicit_rejections():
    shell, _ = convex_circular_shell(_frustum(), 1e-8)
    points = np.array([[0.025, 0, 0], [0.025, 0.001, 0], [0.025, 0, 0.001]])
    assert certify_shell_body(shell, points, np.array([[0, 1, 2]]), "surface_outer", 1e-8, 1, {}) == (
        "body_support_exterior_budget_exceeded"
    )
    points[2, 2] = 0.031
    assert certify_shell_body(shell, points, np.array([[0, 1, 2]]), "surface_outer", 1e-8, 10000, {}) == (
        "body_support_domain_violation"
    )


@pytest.mark.parametrize("mode,lift,radius", [("surface_outer", 0.002, 0.021), ("surface_inner", -0.002, 0.019)])
@pytest.mark.parametrize("bottom,top", [(0.02, 0.02), (0.018, 0.022)])
def test_actual_swept_body_retains_declared_mounting_and_signed_lift(mode, lift, radius, bottom, top):
    theta = np.linspace(0, 2 * np.pi, 73)
    points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros(len(theta))))
    config = CoilGenConfig(
        conductor_cross_section_width=0.0004,
        conductor_cross_section_height=0.0003,
        conductor_mounting_mode=mode,
        normal_shift_length=lift,
        sweep_min_bend_radius=0.001,
        sweep_bend_mode="fillet",
        sweep_max_chord_error=1e-5,
        sweep_max_frame_rotation_deg=10,
    )
    part = CoilPart(
        mesh=_frustum(bottom, top, 48),
        wire_path=WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * (len(points) - 1)),
    )
    create_swept_coil_bodies([part], config)
    before = deepcopy(part)
    result = certify_group_route_support(part, config)
    assert result.accepted, (result.reasons, result.diagnostics)
    assert result.diagnostics["certificate_scope"] == "curved_mesh_support_mounting_and_whole_body_footprint"
    assert result.diagnostics["whole_body_exterior_to_support"] == (mode == "surface_outer")
    assert not result.diagnostics["centered_profile_embedding_intended"]
    np.testing.assert_array_equal(part.swept_body.vertices, before.swept_body.vertices)
    # Changing only the lift side must not be hidden by profile mounting.
    config.normal_shift_length *= -1
    assert not certify_group_route_support(part, config).accepted


@pytest.mark.parametrize("defect", ["body_displacement", "invalid_tangent"])
def test_curved_certificate_retains_exact_body_frame_correspondence(defect):
    mesh = _frustum()
    theta = np.linspace(0, 2 * np.pi, 73)
    points = np.column_stack((0.021 * np.cos(theta), 0.021 * np.sin(theta), np.zeros(len(theta))))
    config = CoilGenConfig(conductor_mounting_mode="surface_outer", normal_shift_length=0.002)
    part = CoilPart(mesh=mesh, wire_path=WirePath(points))
    create_swept_coil_bodies([part], config)
    if defect == "body_displacement":
        part.swept_body.vertices[0, 0] += 1e-5
        reason = "body_section_frame_mismatch"
    else:
        part.fabrication_path.tangents = 0.5 * part.fabrication_path.tangents
        reason = "invalid_section_frames"
    assert certify_group_route_support(part, config).reasons == (reason,)


def test_preflight_work_budget_does_not_produce_an_incomplete_certificate():
    shell, reason = convex_circular_shell(_frustum(), 1e-8, maximum_work=1)
    assert shell is None
    assert reason == "support_face_edge_work_budget_exceeded"
