"""Vectorized and cached fast paths must reproduce the original per-item algorithms exactly."""

import heapq
import math

import numpy as np
import pytest

from coilgen.geometry import build_cylinder_mesh, build_planar_mesh
from coilgen.layout import _initial_wire_segment_distance_bound, _min_wire_segment_distance_pair_chunk, _wire_segment_pair_allowed, _wire_segment_length_prefix
from coilgen.manufacturing import _polyline_circumradii, _polyline_turn_angles_deg, _turn_angle_deg
from coilgen.surface_distance import SurfaceGeodesicSolver, _point_triangle_distance_sq
from coilgen.wire_provenance import nearest_source_segment_indices


def _wavy_loop(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, count)
    points = np.c_[np.cos(angles), np.sin(angles), 0.05 * np.sin(5.0 * angles)] * 0.1
    points[1:-1] += rng.normal(scale=2.0e-4, size=(count - 2, 3))
    points[count // 3] = points[count // 3 - 1]  # repeated point
    points[count // 2 : count // 2 + 5] = points[count // 2]  # short straight run
    return points


def _reference_circumradii(points, closed, minimum_turn_angle_deg):
    radii, kept = [], []
    indices = range(len(points)) if closed else range(1, len(points) - 1)
    for index in indices:
        previous, current, following = points[(index - 1) % len(points)], points[index], points[(index + 1) % len(points)]
        if _turn_angle_deg(previous, current, following) <= minimum_turn_angle_deg:
            continue
        first = float(np.linalg.norm(current - previous))
        second = float(np.linalg.norm(following - current))
        third = float(np.linalg.norm(following - previous))
        area_twice = float(np.linalg.norm(np.cross(current - previous, following - previous)))
        if area_twice <= 1e-30:
            continue
        radii.append(first * second * third / (2.0 * area_twice))
        kept.append(index)
    return np.asarray(radii, dtype=float), np.asarray(kept, dtype=np.int64)


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("minimum_turn_angle_deg", [1e-7, 2.0])
def test_vectorized_circumradii_and_turn_angles_match_per_vertex_loop(closed, minimum_turn_angle_deg) -> None:
    points = _wavy_loop(400, 1)
    radii, kept = _polyline_circumradii(points, closed=closed, minimum_turn_angle_deg=minimum_turn_angle_deg, return_indices=True)
    expected_radii, expected_kept = _reference_circumradii(points, closed, minimum_turn_angle_deg)
    np.testing.assert_array_equal(radii, expected_radii)
    np.testing.assert_array_equal(kept, expected_kept)

    indices = range(len(points)) if closed else range(1, len(points) - 1)
    expected_angles = [_turn_angle_deg(points[(i - 1) % len(points)], points[i], points[(i + 1) % len(points)]) for i in indices]
    np.testing.assert_array_equal(_polyline_turn_angles_deg(points, closed=closed), np.asarray(expected_angles))


def _reference_bound(starts, directions, valid, closed, local_exclusion):
    n_segments = len(starts)
    valid_indices = np.flatnonzero(valid)
    prefix = _wire_segment_length_prefix(directions)
    centers = starts + 0.5 * directions
    pairs = []
    for axis in range(3):
        order = valid_indices[np.argsort(centers[valid_indices, axis], kind="mergesort")]
        window = min(16, max(len(order) - 1, 1))
        for pos, seg_i in enumerate(order):
            for next_pos in range(pos + 1, min(pos + 1 + window, len(order))):
                seg_a, seg_b = int(min(seg_i, order[next_pos])), int(max(seg_i, order[next_pos]))
                if _wire_segment_pair_allowed(
                    seg_a, seg_b, n_segments, closed, starts=starts, directions=directions,
                    local_exclusion=local_exclusion, segment_length_prefix=prefix,
                ):
                    pairs.append((seg_a, seg_b))
    best = math.inf
    for begin in range(0, len(pairs), 65536):
        _a, _b, distance = _min_wire_segment_distance_pair_chunk(starts, directions, np.asarray(pairs[begin : begin + 65536]))
        best = min(best, distance)
    return best


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("local_exclusion", [0.0, 0.004])
def test_vectorized_wire_distance_bound_matches_pair_loop(closed, local_exclusion) -> None:
    points = _wavy_loop(600, 2)
    starts, directions = points[:-1], np.diff(points, axis=0)
    valid = np.linalg.norm(directions, axis=1) > 1e-12
    assert _initial_wire_segment_distance_bound(starts, directions, valid, closed, local_exclusion) == _reference_bound(
        starts, directions, valid, closed, local_exclusion
    )


def test_batched_segment_matching_matches_exhaustive_matching() -> None:
    source = _wavy_loop(900, 3)
    positions = np.linspace(0, len(source) - 1, 1700)
    target = np.c_[[np.interp(positions, np.arange(len(source)), source[:, axis]) for axis in range(3)]].T
    target += np.random.default_rng(4).normal(scale=1.0e-4, size=target.shape)
    indexed = nearest_source_segment_indices(source, target)  # large enough for the k-d tree path
    exhaustive = np.concatenate(
        [nearest_source_segment_indices(source, target[begin : begin + 401]) for begin in range(0, len(target) - 1, 400)]
    )
    np.testing.assert_array_equal(indexed, exhaustive)


def _reference_geodesic(solver: SurfaceGeodesicSolver, points_a, points_b):
    """The original per-point support selection and targeted list-free Dijkstra."""

    def support(points):
        nearest = solver._nearest_vertex_candidates(points, k=8)
        vertices, offsets, owners = [], [], []
        for index, point in enumerate(points):
            faces = set()
            for vertex in nearest[index]:
                faces.update(solver.vertex_faces[int(vertex)])
            best = min(faces, key=lambda f: _point_triangle_distance_sq(point, solver.mesh.vertices[solver.mesh.faces[f]]))
            for vertex in solver.mesh.faces[best]:
                vertices.append(int(vertex))
                offsets.append(float(np.linalg.norm(point - solver.mesh.vertices[int(vertex)])))
                owners.append(index)
        return np.asarray(vertices), np.asarray(offsets), np.asarray(owners)

    va, oa, ia = support(points_a)
    vb, ob, _ib = support(points_b)
    dist = np.full(solver.mesh.n_vertices, math.inf)
    heap = []
    for k, vertex in enumerate(va):
        if oa[k] < dist[vertex]:
            dist[vertex] = oa[k]
            heapq.heappush(heap, (float(oa[k]), int(vertex)))
    targets = {}
    for k, vertex in enumerate(vb):
        targets.setdefault(int(vertex), []).append(k)
    best, best_sample, settled = math.inf, -1, 0
    while heap:
        current, vertex = heapq.heappop(heap)
        if current > dist[vertex]:
            continue
        if current >= best:
            break
        settled += 1
        for k in targets.get(vertex, ()):
            if current + float(ob[k]) < best:
                best, best_sample = current + float(ob[k]), k
        if current >= best:
            break
        for neighbor, weight in solver.adjacency[vertex]:
            if current + weight < dist[neighbor]:
                dist[neighbor] = current + weight
                heapq.heappush(heap, (current + weight, neighbor))
    return best, best_sample, settled, (va, oa, ia)


@pytest.mark.parametrize(
    "mesh",
    [build_cylinder_mesh(0.3, 0.12, 32, 16, 1, 0, 0, 0), build_planar_mesh(0.2, 0.3, 14, 18, 1, 0, 0, 0, 0, 0, 0)],
    ids=["cylinder", "plane"],
)
def test_replayed_geodesic_queries_match_direct_search(mesh) -> None:
    rng = np.random.default_rng(5)
    vertices = np.asarray(mesh.vertices)
    solver = SurfaceGeodesicSolver(mesh)
    sources = [vertices[rng.integers(0, len(vertices), 1)] + rng.normal(scale=1e-3, size=(1, 3)) for _ in range(4)]
    sources.append(vertices[rng.integers(0, len(vertices), 6)])  # exact vertices: tied support faces
    for source in sources:
        for _ in range(6):  # repeated sources exercise the cached full search
            target = vertices[rng.integers(0, len(vertices), 3)] + rng.normal(scale=1e-3, size=(3, 3))
            result = solver.distance_between_point_sets(source, target)
            best, best_sample, settled, (va, oa, ia) = _reference_geodesic(solver, source, target)
            samples = solver.samples_from_points(source)
            np.testing.assert_array_equal(samples.vertices, va)
            np.testing.assert_array_equal(samples.offsets, oa)
            np.testing.assert_array_equal(samples.point_indices, ia)
            assert result.distance == best
            assert solver.last_dijkstra_settled_count == settled
            assert best_sample >= 0
