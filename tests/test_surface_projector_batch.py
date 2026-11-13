"""Batch projection preserves face selection and agrees with scalar geometry to roundoff."""

import numpy as np
import pytest

from coilgen.geometry import build_cylinder_mesh, build_planar_mesh
from coilgen.manufacturing import _SurfaceProjector


def _scalar_reference(projector: _SurfaceProjector, points: np.ndarray):
    """The original per-point loop of ``project_with_details`` (k-d tree candidates)."""

    vertex_count = min(8, len(projector.vertices))
    face_count = min(8, len(projector.faces))
    _distances, vertex_candidates = projector.vertex_tree.query(points, k=vertex_count)
    _distances, centroid_candidates = projector.centroid_tree.query(points, k=face_count)
    outputs = [np.zeros_like(points) for _ in range(3)]
    face_indices = np.full(len(points), -1, dtype=np.int64)
    barycentrics = np.zeros((len(points), 3))
    for index, point in enumerate(points):
        candidate_faces = {int(face) for face in centroid_candidates[index]}
        for vertex_index in vertex_candidates[index]:
            candidate_faces.update(projector.vertex_faces[int(vertex_index)])
        face, closest, barycentric = projector._closest_face_from_candidates(point, candidate_faces)
        normal = barycentric @ projector.vertex_normals[projector.faces[face]]
        length = float(np.linalg.norm(normal))
        if length <= 1e-15:
            normal = projector.face_normals[face]
            length = float(np.linalg.norm(normal))
        outputs[0][index] = closest
        outputs[1][index] = normal / length if length > 1e-15 else np.array([0.0, 0.0, 1.0])
        outputs[2][index] = projector.face_normals[face]
        face_indices[index] = face
        barycentrics[index] = barycentric
    return (*outputs, face_indices, barycentrics)


def _sample_points(mesh, rng: np.random.Generator) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    triangles = vertices[faces]
    count = 1500
    face_index = rng.integers(0, len(faces), count)
    weights = rng.random((count, 2))
    flip = weights.sum(axis=1) > 1.0
    weights[flip] = 1.0 - weights[flip]
    on_surface = (
        triangles[face_index, 0]
        + weights[:, :1] * (triangles[face_index, 1] - triangles[face_index, 0])
        + weights[:, 1:] * (triangles[face_index, 2] - triangles[face_index, 0])
    )
    span = float(np.ptp(vertices, axis=0).max())
    near = on_surface + rng.normal(scale=0.02 * span, size=on_surface.shape)
    far = on_surface[:300] + rng.normal(scale=0.5 * span, size=(300, 3))
    # Exact vertices and edge midpoints make distance ties between neighbouring faces.
    exact_vertices = vertices[rng.integers(0, len(vertices), 300)]
    midpoints = (0.5 * (vertices[faces[:, 0]] + vertices[faces[:, 1]]))[rng.integers(0, len(faces), 300)]
    return np.concatenate((on_surface, near, far, exact_vertices, midpoints))


@pytest.mark.parametrize(
    "mesh",
    [
        build_cylinder_mesh(0.25, 0.066, 8, 3, 1, 0, 0, 0),
        build_cylinder_mesh(0.3, 0.12, 48, 24, 1, 0, 0, 0),
        build_planar_mesh(0.2, 0.3, 16, 20, 1, 0, 0, 0, 0, 0, 0.05),
    ],
    ids=["coarse_cylinder", "fine_cylinder", "plane"],
)
def test_batch_projection_matches_per_point_projection(mesh) -> None:
    projector = _SurfaceProjector(mesh)
    points = _sample_points(mesh, np.random.default_rng(7))

    batched = projector.project_with_details(points)
    reference = _scalar_reference(projector, points)

    for name, actual, expected in zip(
        ("projected", "normals", "face_normals", "face_indices", "barycentric"), batched, reference
    ):
        assert actual.dtype == expected.dtype, name
        if name in {"face_indices", "face_normals"}:
            np.testing.assert_array_equal(actual, expected, err_msg=name)
        else:
            # Scalar BLAS dot products and batched sums need not round identically.
            scale = np.max(np.abs(mesh.vertices)) if name == "projected" else 1.0
            epsilon = np.finfo(float).eps
            np.testing.assert_allclose(actual, expected, rtol=32 * epsilon, atol=32 * epsilon * scale, err_msg=name)


def test_batch_projection_handles_chunk_boundaries(monkeypatch) -> None:
    mesh = build_cylinder_mesh(0.3, 0.12, 24, 12, 1, 0, 0, 0)
    projector = _SurfaceProjector(mesh)
    points = _sample_points(mesh, np.random.default_rng(11))
    whole = projector.project_with_details(points)

    monkeypatch.setattr(_SurfaceProjector, "_BATCH_CHUNK", 7)
    chunked = projector.project_with_details(points)

    for actual, expected in zip(chunked, whole):
        np.testing.assert_array_equal(actual, expected)
