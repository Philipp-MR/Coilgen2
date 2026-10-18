"""Connector refinement preserves the actual surface and distance API."""

import builtins
import sys

import numpy as np
import pytest

from coilgen.models import Mesh
from coilgen.surface_distance import SurfaceGeodesicSolver, _point_triangle_distance_sq
from test_geodesic_cache import _grid_mesh


def _length(points):
    return np.linalg.norm(np.diff(points, axis=0), axis=1).sum()


def test_missing_optional_scipy_has_actionable_refinement_error_and_preserves_numpy_paths(monkeypatch):
    original_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("Fixture without optional SciPy", name="scipy")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "coilgen.surface_corridor", raising=False)
    monkeypatch.setattr(builtins, "__import__", without_scipy)
    mesh = _grid_mesh()
    start, end = np.array([0.2, 0.3, 0.0]), np.array([4.6, 3.7, 0.0])
    ordinary = SurfaceGeodesicSolver(mesh)
    reference = ordinary.distance_between_points(start, end)
    np.testing.assert_array_equal(ordinary.route_between_points(start, end).path_v, reference.path_v)
    refined = SurfaceGeodesicSolver(mesh, refine_routes=True)
    with pytest.raises(ImportError, match=r'requires SciPy.*pip install "\.\[spatial\]"'):
        refined.route_between_points(start, end)
    np.testing.assert_array_equal(refined.distance_between_points(start, end).path_v, reference.path_v)


def test_planar_connector_removes_vertex_zigzags_without_changing_distance_queries():
    mesh = _grid_mesh()
    a, b = np.array([0.2, 0.3, 0.0]), np.array([4.6, 3.7, 0.0])
    solver = SurfaceGeodesicSolver(mesh, refine_routes=True)
    seed = solver.distance_between_points(a, b)
    route = solver.route_between_points(a, b)
    assert _length(route.path_v) < 0.99 * _length(seed.path_v)
    assert _length(route.path_v) == pytest.approx(np.linalg.norm(b - a), rel=1e-8)
    np.testing.assert_array_equal(route.path_v[[0, -1]], [a, b])
    np.testing.assert_array_equal(solver.distance_between_points(a, b).path_v, seed.path_v)
    assert solver.last_route_diagnostics["status"] == "triangle_corridor_refined"


def test_folded_surface_intervals_stay_on_triangles_and_never_shortcut_through_support():
    mesh = _grid_mesh()
    mesh.vertices[:, 2] = np.abs(mesh.vertices[:, 0] - 2.5)
    a = np.array([0.2, 0.3, 2.3])
    b = np.array([4.6, 3.7, 2.1])
    solver = SurfaceGeodesicSolver(mesh, refine_routes=True)
    seed = solver.distance_between_points(a, b)
    route = solver.route_between_points(a, b)
    assert _length(route.path_v) <= _length(seed.path_v) + 1e-12
    assert _length(route.path_v) > np.linalg.norm(b - a) + 0.1
    for left, right in zip(route.path_v[:-1], route.path_v[1:]):
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            point = left * (1 - fraction) + right * fraction
            assert min(_point_triangle_distance_sq(point, triangle) for triangle in mesh.vertices[mesh.faces]) < 1e-20


def test_regular_refinement_has_rigid_transform_equivalent_geometry():
    mesh = _grid_mesh()
    a, b = np.array([0.2, 0.3, 0.0]), np.array([4.6, 3.7, 0.0])
    reference = SurfaceGeodesicSolver(mesh, refine_routes=True).route_between_points(a, b)
    q, _ = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 1.0], [3.0, 1.0, 4.0]]))
    origin = np.array([0.3, -0.2, 4.0])
    transformed = Mesh(mesh.vertices @ q.T + origin, mesh.faces)
    actual = SurfaceGeodesicSolver(transformed, refine_routes=True).route_between_points(
        a @ q.T + origin, b @ q.T + origin
    )
    assert _length(actual.path_v) == pytest.approx(_length(reference.path_v), rel=1e-8)
    for point in (actual.path_v - origin) @ q:
        delta = b - a
        assert np.linalg.norm(np.cross(point - a, delta)) / np.linalg.norm(delta) < 1e-6


def test_disconnected_supports_and_off_surface_endpoints_do_not_gain_a_certificate():
    mesh = Mesh(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
        ),
        np.array([[0, 1, 2], [3, 4, 5]]),
    )
    solver = SurfaceGeodesicSolver(mesh, refine_routes=True)
    assert np.isinf(solver.route_between_points(np.array([0.2, 0.2, 0.0]), np.array([0.2, 0.2, 1.0])).distance)
    solver.route_between_points(np.array([0.2, 0.2, 0.05]), np.array([0.3, 0.3, 0.0]))
    assert solver.last_route_diagnostics["status"] == "endpoint_not_on_surface"


def test_refined_route_cache_is_owned_and_separate_from_distance_cache(monkeypatch):
    solver = SurfaceGeodesicSolver(_grid_mesh(), refine_routes=True)
    a, b = np.array([0.2, 0.3, 0.0]), np.array([4.6, 3.7, 0.0])
    first = solver.route_between_points(a, b)
    saved = first.path_v.copy()
    first.path_v[:] = 99
    monkeypatch.setattr(solver._route_corridor, "route", lambda *args: pytest.fail("unexpected repeated optimization"))
    np.testing.assert_array_equal(solver.route_between_points(a, b).path_v, saved)
    assert _length(solver.distance_between_points(a, b).path_v) > _length(saved)
    solver.clear_path_cache()
    assert not solver._route_cache and solver._route_cache_bytes == 0
