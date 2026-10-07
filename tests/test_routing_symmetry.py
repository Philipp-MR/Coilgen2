from copy import deepcopy

import numpy as np
import pytest

from coilgen.models import CoilPart, ContourLine, LoopGroup, Mesh
from coilgen.routing_symmetry import audit_routing_symmetry


def symmetric_part(*, center=(0.0, 0.0, 0.0), gauge=0.0, provenance=True):
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    faces = []
    for x in (0, 1):
        for y in (2, 3):
            for z in (4, 5):
                face = [x, y, z]
                triangle = vertices[face]
                if np.dot(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]), triangle.mean(axis=0)) < 0:
                    face[1], face[2] = face[2], face[1]
                faces.append(face)
    center = np.asarray(center, dtype=float)
    mesh = Mesh(vertices + center, np.asarray(faces))
    loops = []
    for x, index in ((0.5, 0), (-0.5, 1)):
        points = np.array([[x, 0.5, 0], [x, 0, 0.5], [x, -0.5, 0], [x, 0, -0.5], [x, 0.5, 0]]) + center
        edges = np.array([[index, y] for y in (2, 4, 3, 5, 2)]) if provenance else None
        loop = ContourLine(points, potential=x + gauge, surface_edge_vertices=edges)
        loops.append(loop)
    part = CoilPart(
        mesh=mesh,
        stream_function=vertices[:, 0] + gauge,
        contour_lines=loops,
        groups=[LoopGroup([loop], loop_indices=[i]) for i, loop in enumerate(loops)],
    )
    if provenance:
        attach_face_provenance(part)
    return part


def named(report, name):
    return next(check for check in report["transforms"] if check["name"] == name)


def attach_face_provenance(part):
    lookup = {tuple(sorted(face)): i for i, face in enumerate(part.mesh.faces)}
    for loop in part.contour_lines:
        face_indices, barycentric = [], []
        for i in range(len(loop.v) - 1):
            vertices = set(loop.surface_edge_vertices[i]) | set(loop.surface_edge_vertices[i + 1])
            face_index = lookup[tuple(sorted(vertices))]
            triangle = part.mesh.vertices[part.mesh.faces[face_index]]
            coefficients = np.vstack((triangle.T, np.ones(3)))
            face_indices.append(face_index)
            barycentric.append(
                [np.linalg.lstsq(coefficients, np.r_[point, 1.0], rcond=None)[0] for point in loop.v[i : i + 2]]
            )
        loop.segment_faces = np.asarray(face_indices)
        loop.segment_barycentric = np.asarray(barycentric)
        loop.surface_faces = np.r_[loop.segment_faces, loop.segment_faces[0]]
        loop.surface_barycentric = np.vstack((loop.segment_barycentric[:, 0], loop.segment_barycentric[0, 0]))
        loop.surface_edge_t = np.full(len(loop.v), 0.5)


def test_support_sf_and_directed_contours_must_all_match():
    part = symmetric_part(provenance=True)
    attach_face_provenance(part)
    report = audit_routing_symmetry(part)
    assert report["verified_transform_count"] == 6
    assert report["routing_reuse_allowed"] is False
    mirror = named(report, "mirror_x")
    assert mirror["stream_function"]["parity"] == -1
    assert mirror["contours"]["permutation"] == [1, 0]
    assert mirror["contours"]["current_direction_sign"] == [1, 1]
    assert mirror["contours"]["provenance_status"] == "verified"
    assert mirror["contours"]["sf_current_direction"]["status"] == "verified"
    assert mirror["group_permutation"] == [1, 0]
    assert named(report, "mirror_y")["contours"]["current_direction_sign"] == [-1, -1]


def test_center_and_sf_gauge_do_not_change_the_checks():
    report = audit_routing_symmetry(symmetric_part(center=(3.2, -5.1, 7.4), gauge=1e6))
    assert report["verified_transform_count"] == 6
    mirror = named(report, "mirror_x")
    np.testing.assert_allclose(mirror["center_m"], [3.2, -5.1, 7.4])
    assert mirror["stream_function"]["offset_A"] == pytest.approx(2e6)
    assert report["stream_function_tolerance_A"] == pytest.approx(2e-6)


def test_asymmetric_mesh_is_not_inferred_from_an_ideal_support_shape():
    part = symmetric_part()
    part.mesh.vertices[0, 0] += 0.02
    check = named(audit_routing_symmetry(part), "mirror_x")
    assert check["mesh"]["status"] == "failed"
    assert not check["verified"]
    assert check["group_permutation"] is None


def test_mesh_vertex_symmetry_does_not_certify_triangle_connectivity():
    part = symmetric_part()
    part.mesh.faces = part.mesh.faces[:-1]
    check = named(audit_routing_symmetry(part), "mirror_x")
    assert check["mesh"]["vertex_bijection"]
    assert not check["mesh"]["faces_ok"]
    assert not check["verified"]


def test_mesh_winding_and_stored_normals_are_checked():
    part = symmetric_part()
    part.mesh.faces[0] = part.mesh.faces[0, ::-1]
    check = named(audit_routing_symmetry(part), "mirror_x")
    assert check["mesh"]["faces_ok"]
    assert not check["mesh"]["oriented_normals_ok"]
    part = symmetric_part()
    part.mesh.vertex_normals = part.mesh.vertices.copy()
    part.mesh.vertex_normals[0] *= -1
    assert not named(audit_routing_symmetry(part), "mirror_x")["mesh"]["oriented_normals_ok"]


def test_sf_max_residual_cannot_be_hidden_by_gauge_or_rms():
    part = symmetric_part(gauge=1e6)
    part.stream_function[0] += 0.002
    check = named(audit_routing_symmetry(part), "mirror_x")
    assert check["mesh"]["status"] == "verified"
    assert check["stream_function"]["status"] == "failed"
    assert not check["verified"]


def test_approximate_sf_symmetry_is_reported_without_reuse_permission():
    part = symmetric_part()
    part.stream_function[0] += 1e-5
    check = named(audit_routing_symmetry(part), "mirror_x")
    assert check["stream_function"]["status"] == "approximate"
    assert not check["verified"]
    assert check["group_permutation"] is None


@pytest.mark.parametrize("failure", ["geometry", "level", "current", "turns", "provenance"])
def test_source_contour_constraints_are_checked_separately(failure):
    part = symmetric_part(provenance=True)
    loop = part.contour_lines[1]
    if failure == "geometry":
        loop.v[1, 2] -= 0.002
    elif failure == "level":
        loop.potential += 0.001
    elif failure == "current":
        loop.v = loop.v[::-1].copy()
        loop.surface_edge_vertices = loop.surface_edge_vertices[::-1].copy()
    elif failure == "turns":
        loop.turn_multiplicity = 2
    else:
        loop.surface_edge_vertices[1] = [1, 2]
    check = named(audit_routing_symmetry(part), "mirror_x")
    assert check["mesh"]["status"] == "verified"
    assert check["stream_function"]["status"] == "verified"
    assert check["contours"]["status"] == "failed"
    assert not check["verified"]


def test_cyclic_start_changes_are_supported_but_current_reversal_is_explicit():
    part = symmetric_part()
    loop = part.contour_lines[1]
    for name in ("v", "surface_faces", "surface_barycentric", "surface_edge_vertices", "surface_edge_t"):
        values = np.roll(getattr(loop, name)[:-1], 2, axis=0)
        setattr(loop, name, np.concatenate((values, values[:1]), axis=0))
    for name in ("segment_faces", "segment_barycentric"):
        setattr(loop, name, np.roll(getattr(loop, name), 2, axis=0))
    assert named(audit_routing_symmetry(part), "mirror_x")["verified"]
    loop.v = loop.v[::-1].copy()
    loop.current_orientation = -1.0
    check = named(audit_routing_symmetry(part), "mirror_x")
    assert not check["verified"]
    assert check["contours"]["status"] == "failed"


def test_missing_contour_provenance_limits_verification_to_geometry():
    check = named(audit_routing_symmetry(symmetric_part(provenance=False)), "mirror_x")
    assert check["mesh"]["status"] == "verified"
    assert check["stream_function"]["status"] == "verified"
    assert check["contours"]["status"] == "verified"
    assert check["contours"]["sf_current_direction"]["status"] == "not_checked"
    assert not check["verified"]
    assert check["group_permutation"] is None


def test_orthogonal_or_zero_local_sf_current_does_not_certify_direction():
    mesh = Mesh(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
    loop = ContourLine(
        np.array([[0.1, 0.1, 0], [0.2, 0.1, 0], [0.3, 0.1, 0], [0.1, 0.1, 0]]),
        potential=0.2,
        segment_faces=np.zeros(3, dtype=int),
    )
    part = CoilPart(mesh, stream_function=np.array([0.0, 1.0, 0.0]), contour_lines=[loop])
    current = named(audit_routing_symmetry(part), "mirror_x")["contours"]["sf_current_direction"]
    assert current["status"] == "not_checked"
    assert current["min_cosine_alignment"] == 0.0
    part.stream_function[:] = 0
    current = named(audit_routing_symmetry(part), "mirror_x")["contours"]["sf_current_direction"]
    assert current["status"] == "not_checked"
    assert current["unresolved_segment_count"] == 3


@pytest.mark.parametrize(
    "name",
    [
        "surface_faces",
        "surface_barycentric",
        "surface_edge_vertices",
        "surface_edge_t",
        "segment_faces",
        "segment_barycentric",
    ],
)
def test_malformed_provenance_is_a_diagnostic_failure_not_a_pipeline_exception(name):
    part = symmetric_part()
    loop = part.contour_lines[1]
    setattr(loop, name, getattr(loop, name)[:-1])
    check = named(audit_routing_symmetry(part), "mirror_x")
    assert not check["verified"]
    assert check["contours"]["status"] == "failed"


def test_global_contour_reversal_cannot_hide_from_sf_current_direction_check():
    part = symmetric_part(provenance=True)
    attach_face_provenance(part)
    for loop in part.contour_lines:
        loop.v = loop.v[::-1].copy()
        loop.surface_faces = loop.surface_faces[::-1].copy()
        loop.surface_barycentric = loop.surface_barycentric[::-1].copy()
        loop.surface_edge_vertices = loop.surface_edge_vertices[::-1].copy()
        loop.surface_edge_t = loop.surface_edge_t[::-1].copy()
        loop.segment_faces = loop.segment_faces[::-1].copy()
        loop.segment_barycentric = loop.segment_barycentric[::-1, ::-1].copy()
    check = named(audit_routing_symmetry(part), "mirror_x")
    assert check["mesh"]["status"] == "verified"
    assert check["stream_function"]["status"] == "verified"
    assert check["contours"]["sf_current_direction"]["status"] == "failed"
    assert not check["verified"]


@pytest.mark.parametrize("failure", ["edge_index", "face_index", "barycentric", "edge_fraction", "open_loop"])
def test_invalid_attachment_and_open_loops_are_not_certified(failure):
    part = symmetric_part(provenance=True)
    attach_face_provenance(part)
    loop = part.contour_lines[1]
    if failure == "edge_index":
        loop.surface_edge_vertices[0, 0] = -1
    elif failure == "face_index":
        loop.surface_faces[0] = -1
    elif failure == "barycentric":
        loop.surface_barycentric[0] += [0.1, -0.1, 0]
    elif failure == "edge_fraction":
        loop.surface_edge_t[0] = 0.75
    else:
        loop.v[-1] += [0, 0, 0.02]
    check = named(audit_routing_symmetry(part), "mirror_x")
    assert check["contours"]["status"] == "failed"
    assert not check["verified"]


def test_explicit_boundary_cycles_are_preserved():
    part = symmetric_part()
    part.mesh.boundaries = [np.array([0, 2, 4, 0]), np.array([1, 2, 4, 1])]
    assert named(audit_routing_symmetry(part), "mirror_x")["mesh"]["boundaries_ok"]
    part.mesh.boundaries.pop()
    assert not named(audit_routing_symmetry(part), "mirror_x")["mesh"]["boundaries_ok"]


def test_missing_or_constant_sf_is_not_a_current_symmetry_certificate():
    part = symmetric_part()
    part.stream_function = None
    assert audit_routing_symmetry(part)["verified_transform_count"] == 0
    part.stream_function = np.ones(len(part.mesh.vertices))
    assert named(audit_routing_symmetry(part), "mirror_x")["stream_function"]["status"] == "not_checked"


def test_audit_does_not_mutate_mesh_sf_contours_or_groups():
    part = symmetric_part(provenance=True)
    original = deepcopy(part)
    audit_routing_symmetry(part)
    np.testing.assert_array_equal(part.mesh.vertices, original.mesh.vertices)
    np.testing.assert_array_equal(part.mesh.faces, original.mesh.faces)
    np.testing.assert_array_equal(part.stream_function, original.stream_function)
    for loop, reference in zip(part.contour_lines, original.contour_lines):
        np.testing.assert_array_equal(loop.v, reference.v)
        np.testing.assert_array_equal(loop.surface_edge_vertices, reference.surface_edge_vertices)
    assert [group.loop_indices for group in part.groups] == [group.loop_indices for group in original.groups]


def test_tolerances_are_validated():
    with pytest.raises(ValueError, match="finite and non-negative"):
        audit_routing_symmetry(symmetric_part(), sf_rtol=-1)


def test_complex_sf_is_not_certified_by_discarding_its_imaginary_component():
    part = symmetric_part()
    part.stream_function = part.stream_function.astype(complex) + 1j * np.arange(len(part.mesh.vertices))
    report = audit_routing_symmetry(part)
    assert report["verified_transform_count"] == 0
    assert all(check["stream_function"]["status"] == "not_checked" for check in report["transforms"])
