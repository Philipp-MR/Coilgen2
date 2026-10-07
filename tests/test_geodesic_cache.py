from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import coilgen.surface_distance as surface_distance
from coilgen.models import ContourLine, Mesh
from coilgen.surface_distance import GeodesicPathResult, SurfaceGeodesicSolver


def _grid_mesh(size: int = 6) -> Mesh:
    vertices = np.asarray([[x, y, 0.0] for y in range(size) for x in range(size)])
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            a = y * size + x
            faces.extend(((a, a + 1, a + size + 1), (a, a + size + 1, a + size)))
    return Mesh(vertices=vertices, faces=np.asarray(faces))


def _assert_identical(actual: GeodesicPathResult, expected: GeodesicPathResult) -> None:
    assert actual.distance == expected.distance
    assert (actual.index_a, actual.index_b) == (expected.index_a, expected.index_b)
    for name in ("point_a", "point_b", "path_v"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("kind", ["points", "sets", "contours", "point_to_contour"])
def test_repeated_queries_skip_dijkstra_and_match_uncached_result(kind: str) -> None:
    mesh = _grid_mesh()
    a = np.asarray([[0.2, 0.2, 0.0], [0.2, 1.2, 0.0]])
    b = np.asarray([[4.7, 4.7, 0.0], [4.7, 3.7, 0.0]])

    def query(solver: SurfaceGeodesicSolver) -> GeodesicPathResult:
        if kind == "points":
            return solver.distance_between_points(a[0], b[0])
        if kind == "sets":
            return solver.distance_between_point_sets(a, b)
        if kind == "contours":
            return solver.distance_between_contours(ContourLine(a, 0.0), ContourLine(b, 1.0))
        return solver.distance_point_to_contour(a[0], ContourLine(b, 1.0))

    with patch.object(surface_distance, "_PATH_CACHE_MAX_ENTRIES", 0):
        expected = query(SurfaceGeodesicSolver(mesh))
    solver = SurfaceGeodesicSolver(mesh)
    with patch.object(solver, "_multi_source_dijkstra", wraps=solver._multi_source_dijkstra) as dijkstra:
        first = query(solver)
        assert solver.last_dijkstra_settled_count > 0
        second = query(solver)
    assert dijkstra.call_count == 1
    assert solver.last_dijkstra_settled_count == 0
    assert (solver.path_cache_hits, solver.path_cache_misses) == (1, 1)
    _assert_identical(first, expected)
    _assert_identical(second, expected)


def test_direct_and_unreachable_results_are_reused_without_array_aliases() -> None:
    mesh = Mesh(
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]]),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]]),
    )
    solver = SurfaceGeodesicSolver(mesh)
    start = np.asarray([0.2, 0.2, 0.0])
    for end in (np.asarray([0.3, 0.3, 0.0]), start + [0.0, 0.0, 1.0]):
        first = solver.distance_between_points(start, end)
        with patch.object(solver, "_uncached_distance_between_samples", side_effect=AssertionError("cache miss")):
            second = solver.distance_between_points(start, end)
        _assert_identical(first, second)
        assert not np.shares_memory(first.point_a, second.point_a)
        assert not np.shares_memory(first.point_b, second.point_b)
        assert not np.shares_memory(first.path_v, second.path_v)
    assert np.isinf(second.distance)


def test_result_mutation_does_not_change_cached_or_later_results() -> None:
    solver = SurfaceGeodesicSolver(_grid_mesh())
    start, end = np.asarray([0.2, 0.2, 0.0]), np.asarray([4.7, 4.7, 0.0])
    first = solver.distance_between_points(start, end)
    expected = solver.distance_between_points(start, end)
    first.distance = -10.0
    first.index_a = -10
    for name in ("point_a", "point_b", "path_v"):
        getattr(first, name)[:] = -10.0
    second = solver.distance_between_points(start, end)
    _assert_identical(second, expected)
    second.path_v[:] = 100.0
    _assert_identical(solver.distance_between_points(start, end), expected)


def test_ordered_endpoints_and_point_set_ties_are_not_reverse_cached() -> None:
    mesh = _grid_mesh()
    solver = SurfaceGeodesicSolver(mesh)
    a = np.asarray([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    b = np.asarray([[5.0, 0.0, 0.0], [5.0, 5.0, 0.0]])
    with patch.object(solver, "_multi_source_dijkstra", wraps=solver._multi_source_dijkstra) as dijkstra:
        for source, target in ((a, b), (b, a), (a[::-1], b), (a, b[::-1])):
            actual = solver.distance_between_point_sets(source, target)
            fresh = SurfaceGeodesicSolver(mesh).distance_between_point_sets(source, target)
            _assert_identical(actual, fresh)
            _assert_identical(solver.distance_between_point_sets(source, target), fresh)
    assert dijkstra.call_count == 4
    assert (solver.path_cache_hits, solver.path_cache_misses) == (4, 4)


@pytest.mark.parametrize("provenance", ["surface_faces", "surface_edge_vertices"])
def test_changed_contour_geometry_and_support_provenance_use_new_results(provenance: str) -> None:
    mesh = _grid_mesh()
    solver = SurfaceGeodesicSolver(mesh)
    contour = ContourLine(v=np.asarray([[0.2, 0.2, 0.0]]), potential=0.0)
    if provenance == "surface_faces":
        contour.surface_faces = np.asarray([0])
        changed_support = np.asarray([15])
    else:
        contour.surface_edge_vertices = np.asarray([[0, 1]])
        changed_support = np.asarray([[12, 13]])
    target = np.asarray([[4.7, 4.7, 0.0]])
    solver.distance_between_contours(contour, target)
    getattr(contour, provenance)[:] = changed_support
    _assert_identical(
        solver.distance_between_contours(contour, target),
        SurfaceGeodesicSolver(mesh).distance_between_contours(contour, target),
    )
    contour.v += np.asarray([0.01, 0.0, 0.0])
    _assert_identical(
        solver.distance_between_contours(contour, target),
        SurfaceGeodesicSolver(mesh).distance_between_contours(contour, target),
    )
    assert (solver.path_cache_hits, solver.path_cache_misses) == (0, 3)


def test_input_point_mutation_leaves_previous_query_reusable() -> None:
    mesh = _grid_mesh()
    solver = SurfaceGeodesicSolver(mesh)
    start, end = np.asarray([0.2, 0.2, 0.0]), np.asarray([4.7, 4.7, 0.0])
    saved_start = start.copy()
    saved_result = solver.distance_between_points(start, end)
    start[0] += 1.0
    _assert_identical(
        solver.distance_between_points(start, end), SurfaceGeodesicSolver(mesh).distance_between_points(start, end)
    )
    _assert_identical(solver.distance_between_points(saved_start, end), saved_result)
    assert (solver.path_cache_hits, solver.path_cache_misses) == (1, 2)


def test_mesh_is_a_construction_snapshot_for_cached_and_uncached_queries() -> None:
    mesh = _grid_mesh()
    original = mesh.copy()
    solver = SurfaceGeodesicSolver(mesh)
    start, end = np.asarray([0.2, 0.2, 0.0]), np.asarray([4.7, 4.7, 0.0])
    expected = solver.distance_between_points(start, end)
    mesh.vertices[:, 2] += 3.0
    mesh.faces[:] = mesh.faces[:, ::-1]
    _assert_identical(solver.distance_between_points(start, end), expected)
    solver.clear_sample_cache()
    _assert_identical(solver.distance_between_points(start, end), expected)
    moved_start = start + [1.0, 0.0, 0.0]
    _assert_identical(
        solver.distance_between_points(moved_start, end),
        SurfaceGeodesicSolver(original).distance_between_points(moved_start, end),
    )
    assert SurfaceGeodesicSolver(mesh).distance_between_points(start, end).distance != expected.distance
    assert not np.shares_memory(solver.mesh.vertices, mesh.vertices)
    assert not np.shares_memory(solver.mesh.faces, mesh.faces)
    assert not solver.mesh.vertices.flags.writeable
    assert not solver.mesh.faces.flags.writeable


def test_path_cache_evicts_least_recently_used_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(surface_distance, "_PATH_CACHE_MAX_ENTRIES", 2)
    solver = SurfaceGeodesicSolver(_grid_mesh())
    end = np.asarray([4.7, 4.7, 0.0])
    starts = [np.asarray([x, 0.2, 0.0]) for x in (0.2, 1.2, 2.2)]
    for index in (0, 1, 0, 2, 0, 1):
        solver.distance_between_points(starts[index], end)
    assert (solver.path_cache_hits, solver.path_cache_misses) == (2, 4)
    assert len(solver._path_cache) == 2
    assert solver._path_cache_bytes == sum(size for _, size in solver._path_cache.values())


def test_path_cache_payload_bytes_are_bounded_and_oversize_results_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    solver = SurfaceGeodesicSolver(_grid_mesh())
    end = np.asarray([4.7, 4.7, 0.0])
    solver.distance_between_points(np.asarray([0.2, 0.2, 0.0]), end)
    one_entry_bytes = solver._path_cache_bytes
    solver.clear_path_cache()
    monkeypatch.setattr(surface_distance, "_PATH_CACHE_MAX_BYTES", one_entry_bytes)
    for x in (0.2, 1.2, 2.2):
        solver.distance_between_points(np.asarray([x, 0.2, 0.0]), end)
        assert solver._path_cache_bytes <= one_entry_bytes
    assert len(solver._path_cache) == 1
    solver.clear_path_cache()
    monkeypatch.setattr(surface_distance, "_PATH_CACHE_MAX_BYTES", 1)
    for _ in range(2):
        solver.distance_between_points(np.asarray([0.2, 0.2, 0.0]), end)
    assert not solver._path_cache
    assert solver._path_cache_bytes == 0
    assert (solver.path_cache_hits, solver.path_cache_misses) == (0, 2)


def test_clearing_sample_cache_also_releases_derived_paths_and_counters() -> None:
    solver = SurfaceGeodesicSolver(_grid_mesh())
    start, end = np.asarray([0.2, 0.2, 0.0]), np.asarray([4.7, 4.7, 0.0])
    expected = solver.distance_between_points(start, end)
    solver.distance_between_points(start, end)
    solver.clear_sample_cache()
    assert not solver._point_sample_cache
    assert not solver._contour_sample_cache
    assert not solver._path_cache
    assert solver._path_cache_bytes == 0
    assert (solver.path_cache_hits, solver.path_cache_misses, solver.sample_cache_hits, solver.sample_cache_misses) == (
        0,
    ) * 4
    _assert_identical(solver.distance_between_points(start, end), expected)


def test_empty_query_does_not_report_previous_dijkstra_work() -> None:
    solver = SurfaceGeodesicSolver(_grid_mesh())
    solver.distance_between_points(np.asarray([0.2, 0.2, 0.0]), np.asarray([4.7, 4.7, 0.0]))
    assert solver.last_dijkstra_settled_count > 0
    result = solver.distance_between_point_sets(np.empty((0, 3)), np.asarray([[4.7, 4.7, 0.0]]))
    assert np.isinf(result.distance)
    assert solver.last_dijkstra_settled_count == 0
