"""Distinguish a disk side from a side containing handles."""

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.contours import extract_contours
from coilgen.geometry import normalize_mesh
from coilgen.models import CoilPart, Mesh
from coilgen.surface_topology import classify_loop_topology_signatures, validate_intrinsic_topology_inputs


def _extract(mesh, stream, level):
    part = CoilPart(
        mesh=normalize_mesh(mesh),
        stream_function=np.asarray(stream),
        potential_level_list=np.array([level]),
        contour_step=1.0,
    )
    extract_contours([part], CoilGenConfig(min_point_loop_number=4, progress_flag=False))
    assert part.contour_lines
    assert validate_intrinsic_topology_inputs(part).ok
    return part


def _octahedron():
    vertices = np.array([[1.0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    faces = np.array([[4, 0, 1], [4, 1, 2], [4, 2, 3], [4, 3, 0], [5, 1, 0], [5, 2, 1], [5, 3, 2], [5, 0, 3]])
    return Mesh(vertices, faces)


def _torus():
    nu, nv = 40, 24
    u, v = np.meshgrid(np.arange(nu) * 2 * np.pi / nu, np.arange(nv) * 2 * np.pi / nv, indexing="ij")
    vertices = np.column_stack(
        (
            ((2 + 0.6 * np.cos(v)) * np.cos(u)).ravel(),
            ((2 + 0.6 * np.cos(v)) * np.sin(u)).ravel(),
            (0.6 * np.sin(v)).ravel(),
        )
    )
    faces = []
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = i * nv + j, ((i + 1) % nu) * nv + j, ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
            faces.extend(((a, b, c), (a, c, d)))
    return Mesh(vertices, np.asarray(faces))


def test_sphere_regular_circle_has_two_disk_sides():
    mesh = _octahedron()
    part = _extract(mesh, mesh.vertices[:, 2], 0.31)
    (signature,) = classify_loop_topology_signatures(part)
    assert signature.kind == "contractible"
    assert signature.component_euler_characteristics == (1, 1)
    assert signature.has_boundary_free_side
    assert signature.component_boundary_sets == ((), ())


def test_contractible_torus_circle_distinguishes_disk_from_punctured_handle():
    mesh = _torus()
    part = _extract(mesh, mesh.vertices[:, 0], 2.31)
    (signature,) = classify_loop_topology_signatures(part)
    assert signature.kind == "contractible"
    assert sorted(signature.component_euler_characteristics) == [-1, 1]


def test_boundary_free_torus_side_does_not_imply_a_disk():
    mesh = _torus()
    centers = mesh.vertices[mesh.faces].mean(axis=1)
    mesh.faces = mesh.faces[centers[:, 0] <= 2.43]
    part = _extract(mesh, mesh.vertices[:, 0], 2.31)
    (signature,) = classify_loop_topology_signatures(part)
    assert signature.kind == "essential_separating"
    assert signature.has_boundary_free_side
    assert not signature.contractible_candidate
    assert sorted(signature.component_euler_characteristics) == [-1, 0]


def test_paired_essential_torus_level_components_remain_nonseparating():
    mesh = _torus()
    part = _extract(mesh, mesh.vertices[:, 2], 0.11)
    signatures = classify_loop_topology_signatures(part)
    assert len(signatures) == 2
    assert all(signature.kind == "nonseparating" for signature in signatures)
    assert all(not signature.contractible_candidate for signature in signatures)
    # No homology basis, class equality, or paired-current claim is inferred.
    assert all(signature.component_euler_characteristics == () for signature in signatures)


def test_disconnected_uncut_support_is_not_an_extra_contour_side():
    first, second = _octahedron(), _octahedron()
    mesh = Mesh(
        np.vstack((first.vertices, second.vertices + [4, 0, 0])),
        np.vstack((first.faces, second.faces + len(first.vertices))),
    )
    part = _extract(mesh, np.r_[first.vertices[:, 2], np.full(len(second.vertices), -2.0)], 0.31)
    (signature,) = classify_loop_topology_signatures(part)
    assert signature.kind == "contractible"
    assert signature.component_euler_characteristics == (1, 1)
    assert not signature.ambiguous
