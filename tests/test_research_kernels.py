from __future__ import annotations

import numpy as np
import pytest

from coilgen import geometry_kernels
from coilgen.manufacturing import _swept_mesh_self_intersection_diagnostics
from coilgen.models import Mesh
from coilgen.research import kernels
from coilgen.research.fabrication import accelerated_mesh_collision_diagnostics


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e3])
def test_vectorized_segment_distances_match_scalar_reference(scale):
    rng = np.random.default_rng(141)
    first, second = rng.normal(size=(2, 100, 2, 3)) * scale
    first[0, 1] = first[0, 0]
    second[1] = first[1] + np.array([0, 0, scale])
    measured = geometry_kernels.segment_distances(
        first[:, 0], first[:, 1] - first[:, 0], second[:, 0], second[:, 1] - second[:, 0]
    )
    expected = kernels.segment_pair_distances(first, second, backend="reference").values_m
    np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-14 * scale)


@pytest.mark.skipif(not kernels.NUMBA_AVAILABLE, reason="Optional Numba unavailable")
@pytest.mark.parametrize("vertices,kernel", [(2, kernels.segment_pair_distances), (3, kernels.triangle_pair_distances)])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e3])
def test_compiled_reference_parity(vertices, kernel, scale):
    rng = np.random.default_rng(75)
    first, second = rng.normal(size=(2, 60, vertices, 3)) * scale
    first[0] = first[0, 0]  # Degenerate point primitive.
    second[1] = first[1]  # Coincident primitive.
    measured = kernel(first, second, backend="numba")
    expected = kernel(first, second, backend="reference")
    assert measured.backend == "numba_float64"
    np.testing.assert_allclose(measured.values_m, expected.values_m, rtol=2e-12, atol=1e-14 * scale)


@pytest.mark.parametrize(
    "backend",
    [
        "reference",
        pytest.param(
            "numba", marks=pytest.mark.skipif(not kernels.NUMBA_AVAILABLE, reason="Optional Numba unavailable")
        ),
    ],
)
def test_triangle_crossing_coplanar_and_face_distance(backend):
    triangle = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    first = np.repeat(triangle[None], 3, axis=0)
    second = np.array(
        [
            [[0.2, 0.2, -1], [0.2, 0.2, 1], [0.4, 0.2, 0]],
            [[0.1, 0.1, 0], [0.2, 0.1, 0], [0.1, 0.2, 0]],
            triangle + [0, 0, 0.25],
        ]
    )
    np.testing.assert_allclose(
        kernels.triangle_pair_distances(first, second, backend=backend).values_m, [0, 0, 0.25], atol=1e-15
    )


def test_safe_auto_fallback_and_context_reset(monkeypatch):
    def unavailable(*args):
        raise RuntimeError("test compiler unavailable")

    monkeypatch.setattr(geometry_kernels, "_compiled_kernel", lambda vertices: unavailable)
    points = np.zeros((1, 2, 3))
    with pytest.warns(RuntimeWarning, match="reference"):
        result = kernels.segment_pair_distances(points, points, backend="auto")
    assert result.backend == "reference_float64"
    with pytest.raises(RuntimeError, match="compiler unavailable"):
        kernels.segment_pair_distances(points, points, backend="numba")
    assert kernels.active_geometry_backend() is None
    with pytest.raises(ValueError), kernels.geometry_acceleration("reference"):
        assert kernels.active_geometry_backend() == "reference"
        raise ValueError()
    assert kernels.active_geometry_backend() is None


def test_invalid_distance_input_rejected():
    with pytest.raises(ValueError, match="matching"):
        kernels.segment_pair_distances(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        kernels.triangle_pair_distances(np.full((1, 3, 3), np.nan), np.zeros((1, 3, 3)))


def test_accelerated_self_collision_matches_original_crossing():
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, -1], [0.2, 0.2, 1], [0.4, 0.2, 0]]),
        faces=np.array([[0, 1, 2], [3, 4, 5]]),
    )
    original = _swept_mesh_self_intersection_diagnostics(mesh)
    accelerated = accelerated_mesh_collision_diagnostics(mesh, backend="reference")
    for key in original:
        assert accelerated[key] == original[key]
