"""Actual convex cap geometry, complete triangles and unchanged lift limits."""

from copy import deepcopy

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import build_spherical_cap_mesh, normalize_mesh, rotation_matrix_by_vector
from coilgen.group_route_convex_cap import convex_mesh_cap, certify_cap_body
from coilgen.group_route_support import check_group_route_support_domain, certify_group_route_support
from coilgen.manufacturing import create_swept_coil_bodies
from coilgen.models import CoilPart, WirePath, WirePathGeometry


def cap():
    return normalize_mesh(build_spherical_cap_mesh(0.08, np.pi / 2, 24, 8, 0, 0, 1, 0, 0, 0, 0))


def test_convex_hemisphere_is_verified_from_actual_mesh_and_rigid_transform():
    mesh = cap()
    rotation = rotation_matrix_by_vector(np.array([1.0, 2, -1]), 0.73)
    mesh.vertices = mesh.vertices @ rotation.T + [0.2, -0.1, 0.03]
    mesh.metadata = {"geometry_type": "untrusted"}
    result = check_group_route_support_domain(mesh, CoilGenConfig(conductor_mounting_mode="surface_outer"))
    assert result.accepted, result
    assert result.diagnostics["support_geometry"] == "verified_convex_mesh_cap"


@pytest.mark.parametrize("defect", ["hole", "duplicate", "inward", "dented", "rim"])
def test_cap_preflight_rejects_invalid_or_nonconvex_geometry(defect):
    mesh = cap()
    if defect == "hole":
        mesh.faces = mesh.faces[1:]
    elif defect == "duplicate":
        mesh.faces = np.vstack((mesh.faces, mesh.faces[0]))
    elif defect == "inward":
        mesh.faces = mesh.faces[:, ::-1]
    elif defect == "dented":
        mesh.vertices[np.argmax(mesh.vertices[:, 2]), 2] -= 0.01
    else:
        mesh.vertices[np.argmin(mesh.vertices[:, 2]), 2] -= 0.002
    assert not check_group_route_support_domain(mesh, CoilGenConfig(conductor_mounting_mode="surface_outer")).accepted


def test_whole_triangle_crossing_cap_is_rejected_even_when_its_vertices_are_exterior():
    shell, reason = convex_mesh_cap(cap(), 1e-8)
    assert not reason
    points = np.array([[0.09, 0, 0.02], [-0.09, -0.001, 0.02], [-0.09, 0.001, 0.02]])
    diagnostics = {}
    assert (
        certify_cap_body(shell, points, np.array([[0, 1, 2]]), "surface_outer", 1e-8, 100000, diagnostics)
        == "body_support_penetration"
    )
    assert "body_support_penetration_witness_m" in diagnostics
    points[:, 2] = -0.001
    assert (
        certify_cap_body(shell, points, np.array([[0, 1, 2]]), "surface_outer", 1e-8, 100000, {})
        == "body_support_domain_violation"
    )


def test_actual_hemisphere_body_and_wrong_lift_side_are_independently_checked():
    mesh = cap()
    theta = np.linspace(0, 2 * np.pi, 181)
    z = 0.04
    r = np.sqrt(0.081**2 - z**2)
    points = np.column_stack((r * np.cos(theta), r * np.sin(theta), np.full(len(theta), z)))
    config = CoilGenConfig(
        conductor_mounting_mode="surface_outer",
        normal_shift_length=0.004,
        conductor_cross_section_width=0.0004,
        conductor_cross_section_height=0.0003,
        sweep_min_bend_radius=0.001,
        sweep_bend_mode="fillet",
        sweep_max_chord_error=1e-5,
        sweep_max_frame_rotation_deg=10,
    )
    part = CoilPart(
        mesh=mesh, wire_path=WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * (len(points) - 1))
    )
    create_swept_coil_bodies([part], config)
    before = deepcopy(part.swept_body)
    result = certify_group_route_support(part, config)
    assert result.accepted, (result.reasons, result.diagnostics)
    np.testing.assert_array_equal(part.swept_body.vertices, before.vertices)
    config.normal_shift_length *= -1
    assert not certify_group_route_support(part, config).accepted
