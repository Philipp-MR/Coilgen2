"""Whole-triangle support scope and centered-mounting regression checks."""

import numpy as np
import pytest

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh, rotation_matrix_by_vector
from coilgen.group_route_support import certify_group_route_support, check_group_route_support_domain
from coilgen.manufacturing import create_swept_coil_bodies
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry


def _plane():
    return normalize_mesh(
        Mesh(
            np.array([[-0.02, -0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0], [-0.02, 0.02, 0]]),
            np.array([[0, 1, 2], [0, 2, 3]]),
        )
    )


def _swept_circle(*, mode="centered", height=0.0, lift=0.002, radius=0.006, x_offset=0.0):
    theta = np.linspace(0, 2 * np.pi, 181)
    points = np.column_stack((x_offset + radius * np.cos(theta), radius * np.sin(theta), np.full(len(theta), height)))
    config = CoilGenConfig(
        conductor_cross_section_width=0.0006,
        conductor_cross_section_height=0.0004,
        conductor_mounting_mode=mode,
        normal_shift_length=lift,
        sweep_min_bend_radius=0.0012,
        sweep_bend_mode="fillet",
        sweep_max_frame_rotation_deg=5.0,
        sweep_max_chord_error=5e-6,
    )
    part = CoilPart(
        mesh=_plane(),
        wire_path=WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * (len(points) - 1)),
    )
    create_swept_coil_bodies([part], config)
    return part, config


@pytest.mark.parametrize("mode", ["centered", "surface_outer", "surface_inner"])
def test_declared_mounting_modes_certify_complete_planar_body(mode):
    part, config = _swept_circle(mode=mode)
    before = part.swept_body.vertices.copy()
    result = certify_group_route_support(part, config)
    assert result.accepted, result.reasons
    assert result.diagnostics["centered_profile_embedding_intended"] == (mode == "centered")
    assert result.diagnostics["whole_body_exterior_to_support_plane"] == (mode != "centered")
    np.testing.assert_array_equal(part.swept_body.vertices, before)
    if mode == "centered":
        assert np.min(part.swept_body.vertices[:, 2]) < 0


@pytest.mark.parametrize("height,lift", [(-0.001, 0.002), (0.003, 0.002), (0.001, -0.002)])
def test_reference_side_and_lift_are_independent_of_manufacturing_pass(height, lift):
    part, config = _swept_circle(height=height, lift=lift)
    # The earlier manufacturing diagnostic intentionally ignores penetration
    # in centered mode; the new reference-origin contract must still hold.
    assert part.manufacturing_diagnostics["manufacturing_ok"]
    result = certify_group_route_support(part, config)
    assert result.reasons == ("reference_support_side_or_lift_violation",)


def test_negative_lift_uses_declared_inward_reference_side():
    part, config = _swept_circle(height=-0.001, lift=-0.002)
    assert certify_group_route_support(part, config).accepted


def test_whole_body_footprint_rejects_overhang_even_when_centerline_is_inside():
    part, config = _swept_circle(radius=0.006, x_offset=0.0139)
    result = certify_group_route_support(part, config)
    assert result.reasons == ("body_support_domain_violation",)
    assert result.diagnostics["reference_domain_margin_m"] > 0
    assert result.diagnostics["body_domain_margin_m"] < 0


def test_centered_body_cannot_use_stale_section_frames_to_hide_a_displacement():
    part, config = _swept_circle()
    part.swept_body.vertices[0, 2] -= 0.001
    assert certify_group_route_support(part, config).reasons == ("body_section_frame_mismatch",)


def test_missing_final_frames_fail_closed():
    part, config = _swept_circle()
    part.fabrication_path = None
    assert certify_group_route_support(part, config).reasons == ("missing_final_section_frames",)


def test_convex_certificate_is_rigid_transform_invariant():
    part, config = _swept_circle()
    rotation = rotation_matrix_by_vector(np.array([1.0, 2.0, -1.0]), 0.63)
    offset = np.array([0.12, -0.4, 0.25])
    part.mesh = normalize_mesh(Mesh(part.mesh.vertices @ rotation.T + offset, part.mesh.faces.copy()))
    part.wire_path.v = part.wire_path.v @ rotation.T + offset
    # Regenerate the physical frames and mesh in the transformed coordinates.
    create_swept_coil_bodies([part], config)
    result = certify_group_route_support(part, config)
    assert result.accepted, result.reasons


def test_fast_preflight_rejects_nonplanar_support_before_body_exists():
    mesh = _plane()
    mesh.vertices[2, 2] = 0.001
    result = check_group_route_support_domain(mesh, CoilGenConfig(conductor_mounting_mode="surface_outer"))
    assert result.reasons == ("unsupported_nonplanar_support",)


def test_fast_preflight_rejects_a_hole_and_disconnected_domain():
    vertices = np.array([[x, y, 0] for y in [-0.02, -0.002, 0.002, 0.02] for x in [-0.02, -0.002, 0.002, 0.02]])
    faces = []
    for y in range(3):
        for x in range(3):
            if x == y == 1:
                continue
            a = y * 4 + x
            faces.extend([[a, a + 1, a + 5], [a, a + 5, a + 4]])
    mesh = normalize_mesh(Mesh(vertices, np.asarray(faces)))
    assert check_group_route_support_domain(mesh, CoilGenConfig(conductor_mounting_mode="surface_outer")).reasons == (
        "unsupported_holed_or_disconnected_support",
    )
    disconnected = _plane()
    disconnected.vertices = np.vstack((disconnected.vertices, disconnected.vertices + [0.05, 0, 0]))
    disconnected.faces = np.vstack((disconnected.faces, disconnected.faces + 4))
    assert check_group_route_support_domain(
        disconnected, CoilGenConfig(conductor_mounting_mode="surface_outer")
    ).reasons == ("unsupported_holed_or_disconnected_support",)


def test_fast_preflight_rejects_concavity_and_inconsistent_orientation():
    mesh = normalize_mesh(
        Mesh(np.array([[0, 0, 0], [0.02, 0, 0], [0.01, 0.005, 0], [0, 0.02, 0]]), np.array([[0, 1, 2], [0, 2, 3]]))
    )
    assert check_group_route_support_domain(mesh, CoilGenConfig(conductor_mounting_mode="surface_outer")).reasons == (
        "unsupported_nonconvex_support",
    )
    mesh = _plane()
    mesh.faces[1] = mesh.faces[1, ::-1]
    assert check_group_route_support_domain(mesh, CoilGenConfig()).reasons == ("inconsistent_support_orientation",)


def test_duplicate_faces_do_not_create_a_false_planar_coverage_certificate():
    mesh = _plane()
    mesh.faces = np.vstack((mesh.faces, mesh.faces[:1]))
    assert check_group_route_support_domain(mesh, CoilGenConfig()).reasons == ("nonmanifold_or_overlapping_support",)


def test_support_and_body_budgets_fail_explicitly():
    part, config = _swept_circle()
    assert not check_group_route_support_domain(part.mesh, config, maximum_support_faces=1).accepted
    assert certify_group_route_support(part, config, maximum_body_faces=1).reasons == ("missing_swept_body_or_budget",)


def _cylinder(capped=True, *, count=48):
    theta = np.arange(count) * 2 * np.pi / count
    ring = 0.02 * np.column_stack((np.cos(theta), np.sin(theta), np.zeros(count)))
    vertices = np.vstack((ring + [0, 0, -0.03], ring + [0, 0, 0.03]))
    faces = []
    for index in range(count):
        following = (index + 1) % count
        faces.extend([[index, following, count + following], [index, count + following, count + index]])
    if capped:
        vertices = np.vstack((vertices, [0, 0, -0.03], [0, 0, 0.03]))
        for index in range(count):
            following = (index + 1) % count
            faces.extend([[2 * count, following, index], [2 * count + 1, count + index, count + following]])
    return normalize_mesh(Mesh(vertices, np.asarray(faces)))


def _cylinder_coil(*, capped=True, z=0.0, radius=0.021):
    part, config = _swept_circle()
    part.mesh = _cylinder(capped)
    theta = np.linspace(0, 2 * np.pi, 181)
    points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.full(len(theta), z)))
    part.wire_path = WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * (len(points) - 1))
    create_swept_coil_bodies([part], config)
    return part, config


@pytest.mark.parametrize("capped", [True, False])
def test_centered_lifted_cylinder_has_a_whole_reference_and_body_footprint_certificate(capped):
    part, config = _cylinder_coil(capped=capped)
    result = certify_group_route_support(part, config)
    assert result.accepted, (result.reasons, result.diagnostics)
    assert result.diagnostics["certificate_scope"] == "centered_reference_support_and_whole_body_boundary_tube"
    assert result.diagnostics["whole_body_exterior_to_support"] is False
    assert result.diagnostics["support_boundary_edge_count"] == (0 if capped else 96)


def test_lifted_cylinder_near_cut_edge_does_not_confuse_height_with_footprint_clearance():
    part, config = _cylinder_coil(capped=False, z=0.0299)
    result = certify_group_route_support(part, config)
    assert result.reasons == ("body_support_footprint_uncertified",)
    assert not result.diagnostics["reference_support_reasons"]
    # Euclidean distance to the edge exceeds the profile radius, but much of
    # that distance is lift. The projected body can still overhang the cut.
    assert np.max(part.swept_body.vertices[:, 2]) > 0.03


def test_centered_general_mesh_requires_adjacent_ring_triangle_connectivity():
    part, config = _cylinder_coil()
    part.swept_body.faces[0] = [0, 4 * 20, 4 * 60]
    assert certify_group_route_support(part, config).reasons == ("body_section_loft_topology_mismatch",)


def test_asymmetric_profile_tube_includes_its_offset_from_the_reference_origin():
    part, config = _cylinder_coil(capped=False)
    section = np.array([[-0.0001, 0], [0.0005, 0], [0.0005, 0.0004], [-0.0001, 0.0004]])
    config.cross_sectional_points = section
    create_swept_coil_bodies([part], config)
    result = certify_group_route_support(part, config)
    assert result.accepted, (result.reasons, result.diagnostics)
    np.testing.assert_allclose(
        result.diagnostics["body_reference_tube_radius_m"], np.linalg.norm(section, axis=1).max(), atol=1e-12
    )


def test_curved_exterior_mounting_is_still_explicitly_unsupported():
    part, config = _cylinder_coil()
    config.conductor_mounting_mode = "surface_outer"
    assert check_group_route_support_domain(part.mesh, config).reasons == ("unsupported_nonplanar_support",)


def test_general_support_budget_never_turns_into_acceptance():
    part, config = _cylinder_coil()
    result = certify_group_route_support(part, config, maximum_support_evaluations=1)
    assert result.reasons == ("support_certificate_budget_exceeded",)


@pytest.mark.parametrize("phase", ["construction", "traversal"])
def test_bounded_support_scan_exhaustion_rejects_before_running_any_spatial_queries(monkeypatch, phase):
    import coilgen.group_route_support as support
    from coilgen.connector_bridge_validation import _SpatialSupport

    part, config = _cylinder_coil()
    original_checker = support._CenteredSpatialSupport

    class LimitedSupport(original_checker):
        def __init__(self, *args, **kwargs):
            kwargs["maximum_box_work"] = 1 if phase == "construction" else 50_000_000
            super().__init__(*args, **kwargs)
            if phase == "traversal":
                self.maximum_box_work = self.box_work + 1

    def unexpected(*args, **kwargs):
        raise AssertionError("Incomplete broad-phase support scan must reject before spatial queries")

    monkeypatch.setattr(support, "_CenteredSpatialSupport", LimitedSupport)
    monkeypatch.setattr(_SpatialSupport, "validate", unexpected)
    monkeypatch.setattr(_SpatialSupport, "sample", unexpected)
    result = certify_group_route_support(part, config)
    assert result.reasons == ("support_face_edge_work_budget_exceeded",)
    if phase == "traversal":
        assert (
            result.diagnostics["support_attempted_initial_chord_face_work"]
            > result.diagnostics["support_initial_chord_face_work"]
        )


def test_raw_surface_vertices_cannot_hide_an_off_support_input_behind_projected_frames():
    part, config = _cylinder_coil()
    source = part.wire_path.v[:3].copy()
    source[:, :2] *= 1.2
    part.wire_path = WirePath(source)
    result = certify_group_route_support(part, config)
    assert result.reasons == ("lift_limit_violation",)
    assert result.diagnostics["raw_surface_reference_vertex_count"] == 3


def test_inward_cylinder_uses_signed_lift_and_actual_faces_despite_stale_normal_cache():
    part, config = _cylinder_coil(radius=0.019)
    config.normal_shift_length = -0.002
    part.mesh.face_normals *= -1
    cached = part.mesh.face_normals.copy()
    result = certify_group_route_support(part, config)
    assert result.accepted, (result.reasons, result.diagnostics)
    assert result.diagnostics["minimum_sampled_reference_signed_height_m"] > 0
    np.testing.assert_array_equal(part.mesh.face_normals, cached)
    config.normal_shift_length = 0.002
    assert certify_group_route_support(part, config).reasons == ("lift_limit_violation",)


def test_general_open_loft_caps_are_certified_only_at_their_declared_end_rings():
    part, config = _cylinder_coil()
    points = np.column_stack((np.full(11, 0.021), np.zeros(11), np.linspace(-0.01, 0.01, 11)))
    part.wire_path = WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * 10)
    create_swept_coil_bodies([part], config)
    result = certify_group_route_support(part, config)
    assert result.accepted, (result.reasons, result.diagnostics)
    assert not part.fabrication_path.closed
    cap_vertex = len(part.swept_body.vertices) - 2
    cap_face = np.flatnonzero(np.any(part.swept_body.faces == cap_vertex, axis=1))[0]
    part.swept_body.faces[cap_face] = [cap_vertex, 4 * 5, 4 * 5 + 1]
    assert certify_group_route_support(part, config).reasons == ("body_section_loft_topology_mismatch",)


def test_crease_centered_reference_can_pass_without_claiming_substrate_exterior():
    part, config = _swept_circle(height=0.001)
    vertices = np.array(
        [[-0.02, -0.02, 0], [0, -0.02, 0], [0.02, -0.02, -0.003], [-0.02, 0.02, 0], [0, 0.02, 0], [0.02, 0.02, -0.003]]
    )
    part.mesh = normalize_mesh(Mesh(vertices, np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]])))
    config.normal_shift_length = 0.004
    create_swept_coil_bodies([part], config)
    result = certify_group_route_support(part, config)
    assert result.accepted, (result.reasons, result.diagnostics)
    assert result.diagnostics["centered_profile_embedding_intended"]


def test_centered_support_hole_is_allowed_only_when_full_body_tube_stays_clear():
    part, config = _swept_circle(radius=0.012, height=0.0005)
    vertices = np.array([[x, y, 0] for y in [-0.02, -0.002, 0.002, 0.02] for x in [-0.02, -0.002, 0.002, 0.02]])
    faces = []
    for y in range(3):
        for x in range(3):
            if x == y == 1:
                continue
            a = 4 * y + x
            faces.extend([[a, a + 1, a + 5], [a, a + 5, a + 4]])
    part.mesh = normalize_mesh(Mesh(vertices, np.asarray(faces)))
    create_swept_coil_bodies([part], config)
    result = certify_group_route_support(part, config)
    assert result.accepted, (result.reasons, result.diagnostics)
    # The same support cannot host a circle wholly over the central hole.
    theta = np.linspace(0, 2 * np.pi, 73)
    points = np.column_stack((0.001 * np.cos(theta), 0.001 * np.sin(theta), np.full(len(theta), 0.0005)))
    part.wire_path = WirePath(points, segment_geometry=[WirePathGeometry.FREE_3D] * (len(points) - 1))
    create_swept_coil_bodies([part], config)
    rejected = certify_group_route_support(part, config)
    assert not rejected.accepted
    assert rejected.reasons[0] in {"support_domain_violation", "support_certificate_budget_exceeded"}
