from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .models import ContourLine, Mesh


@dataclass
class GeodesicPathResult:
    distance: float
    point_a: np.ndarray
    index_a: int
    point_b: np.ndarray
    index_b: int
    path_v: np.ndarray


@dataclass
class _SurfaceSamples:
    points: np.ndarray
    vertices: np.ndarray
    offsets: np.ndarray
    point_indices: np.ndarray


@dataclass
class _DijkstraResult:
    dist: np.ndarray
    prev: np.ndarray
    source_index: np.ndarray
    source_vertex: np.ndarray
    best_target_sample: int
    settled_count: int
    # Recorded only for untargeted searches: vertices in the order they were
    # settled and their distances (nondecreasing), used to replay targeted queries.
    settled_vertices: np.ndarray | None = None
    settled_distances: np.ndarray | None = None


_SAMPLE_CACHE_MAX_ENTRIES = 256
_PATH_CACHE_MAX_ENTRIES = 512
_PATH_CACHE_MAX_BYTES = 16 * 1024 * 1024
_FULL_SEARCH_CACHE_MAX_BYTES = 64 * 1024 * 1024


class SurfaceGeodesicSolver:
    """Approximate geodesics over a construction-time mesh geometry snapshot.

    Construct a new solver after changing mesh vertices or faces. Keeping the
    geometry snapshot with its graph prevents cached paths, support samples and
    edge weights from referring to different mesh revisions. Ordered queries
    are reused exactly; reversing endpoints still performs its own search so
    that Dijkstra's directed tie choices are preserved.
    """

    def __init__(self, mesh: Mesh, *, refine_routes: bool = False):
        self.mesh = Mesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
        self.mesh.vertices.flags.writeable = False
        self.mesh.faces.flags.writeable = False
        self.adjacency = _build_vertex_adjacency(self.mesh)
        self.vertex_faces = _build_vertex_faces(self.mesh)
        self.last_dijkstra_settled_count = 0
        self.sample_cache_hits = 0
        self.sample_cache_misses = 0
        self._point_sample_cache: dict[tuple[object, ...], _SurfaceSamples] = {}
        self._contour_sample_cache: dict[tuple[object, ...], _SurfaceSamples] = {}
        self.path_cache_hits = 0
        self.path_cache_misses = 0
        self._path_cache: OrderedDict[tuple[object, ...], tuple[GeodesicPathResult, int]] = OrderedDict()
        self._path_cache_bytes = 0
        self.refine_routes = bool(refine_routes)
        self._route_corridor = None
        self._route_cache = OrderedDict()
        self._route_cache_bytes = 0
        self.last_route_diagnostics = {}
        self._full_search_cache: OrderedDict[tuple[object, ...], tuple[_DijkstraResult, np.ndarray, int]] = OrderedDict()
        self._full_search_cache_bytes = 0

    def route_between_points(self, point_a: np.ndarray, point_b: np.ndarray) -> GeodesicPathResult:
        """Refine only mutable connector proposals; distance queries are unchanged."""
        key = (np.asarray(point_a, dtype=float).tobytes(), np.asarray(point_b, dtype=float).tobytes())
        cached = self._route_cache.get(key) if self.refine_routes else None
        if cached is not None:
            self._route_cache.move_to_end(key)
            self.last_route_diagnostics = dict(cached[1])
            return _copy_result(cached[0])
        result = self.distance_between_points(point_a, point_b)
        if not self.refine_routes or not np.isfinite(result.distance) or len(result.path_v) < 2:
            self.last_route_diagnostics = {"status": "mesh_vertex_seed"}
            return result
        try:
            from .surface_corridor import SurfaceCorridor
        except ModuleNotFoundError as error:
            if error.name != "scipy" and not str(error.name).startswith("scipy."):
                raise
            raise ImportError(
                "Regular geodesic connector refinement requires SciPy. "
                'Install the spatial extra with: python -m pip install ".[spatial]"'
            ) from error

        if self._route_corridor is None:
            self._route_corridor = SurfaceCorridor(self.mesh, self.vertex_faces)
        source = self.samples_from_points(point_a).vertices
        target = self.samples_from_points(point_b).vertices
        route = self._route_corridor.route(result.point_a, result.point_b, source, target, result.path_v)
        self.last_route_diagnostics = {
            "status": route.status,
            "support_face_count": len(route.face_indices),
            "optimizer_converged": route.optimizer_converged,
            "scope": "connector_seed_inside_actual_triangles; not finite_body_or_field_certificate",
        }
        if route.face_indices:
            result.path_v = route.points.copy()
            result.distance = float(np.linalg.norm(np.diff(result.path_v, axis=0), axis=1).sum())
        size = result.path_v.nbytes + result.point_a.nbytes + result.point_b.nbytes + sum(map(len, key))
        if size <= _PATH_CACHE_MAX_BYTES and _PATH_CACHE_MAX_ENTRIES > 0:
            while self._route_cache and (
                len(self._route_cache) >= _PATH_CACHE_MAX_ENTRIES
                or self._route_cache_bytes + size > _PATH_CACHE_MAX_BYTES
            ):
                _, entry = self._route_cache.popitem(last=False)
                self._route_cache_bytes -= entry[2]
            self._route_cache[key] = (_copy_result(result), dict(self.last_route_diagnostics), size)
            self._route_cache_bytes += size
        return result

    def clear_sample_cache(self) -> None:
        """Release sampled endpoints and all paths derived from them."""
        self._point_sample_cache.clear()
        self._contour_sample_cache.clear()
        self.sample_cache_hits = 0
        self.sample_cache_misses = 0
        self.clear_path_cache()

    def clear_path_cache(self) -> None:
        """Release path results while retaining mesh and endpoint samples."""
        self._path_cache.clear()
        self._path_cache_bytes = 0
        self.path_cache_hits = 0
        self.path_cache_misses = 0
        self.last_dijkstra_settled_count = 0
        self._route_cache.clear()
        self._route_cache_bytes = 0
        self._full_search_cache.clear()
        self._full_search_cache_bytes = 0

    def distance_between_points(self, point_a: np.ndarray, point_b: np.ndarray) -> GeodesicPathResult:
        return self._distance_between_samples(self.samples_from_points(point_a), self.samples_from_points(point_b))

    def distance_point_to_contour(self, point: np.ndarray, contour: ContourLine | np.ndarray) -> GeodesicPathResult:
        return self._distance_between_samples(self.samples_from_points(point), self.samples_from_contour(contour))

    def distance_between_contours(
        self,
        contour_a: ContourLine | np.ndarray,
        contour_b: ContourLine | np.ndarray,
    ) -> GeodesicPathResult:
        return self._distance_between_samples(
            self.samples_from_contour(contour_a), self.samples_from_contour(contour_b)
        )

    def distance_between_point_sets(self, points_a: np.ndarray, points_b: np.ndarray) -> GeodesicPathResult:
        return self._distance_between_samples(self.samples_from_points(points_a), self.samples_from_points(points_b))

    def samples_from_points(self, points: np.ndarray) -> _SurfaceSamples:
        points_a = _as_point_array(points)
        if len(points_a) == 0:
            return _empty_samples()

        cache_key = _points_sample_cache_key(points_a)
        cached = self._point_sample_cache.get(cache_key)
        if cached is not None:
            self.sample_cache_hits += 1
            return cached

        self.sample_cache_misses += 1
        support_vertices, support_offsets, support_point_indices = self._support_vertices_from_points(points_a)
        samples = _SurfaceSamples(
            points=points_a,
            vertices=support_vertices,
            offsets=support_offsets,
            point_indices=support_point_indices,
        )
        _cache_store(self._point_sample_cache, cache_key, samples)
        return samples

    def samples_from_contour(self, contour: ContourLine | np.ndarray) -> _SurfaceSamples:
        if not isinstance(contour, ContourLine):
            return self.samples_from_points(contour)
        points = _as_point_array(contour.v)
        if len(points) == 0:
            return _empty_samples()

        cache_key = _contour_sample_cache_key(contour, points)
        cached = self._contour_sample_cache.get(cache_key)
        if cached is not None:
            self.sample_cache_hits += 1
            return cached

        self.sample_cache_misses += 1
        support_vertices: list[int] = []
        support_offsets: list[float] = []
        support_point_indices: list[int] = []
        edge_vertices = contour.surface_edge_vertices
        faces = contour.surface_faces
        for point_ind, point in enumerate(points):
            vertices: np.ndarray | None = None
            if edge_vertices is not None and point_ind < len(edge_vertices):
                vertices = np.asarray(edge_vertices[point_ind], dtype=np.int64)
                vertices = vertices[(0 <= vertices) & (vertices < self.mesh.n_vertices)]
            if (vertices is None or len(vertices) == 0) and faces is not None and point_ind < len(faces):
                face_ind = int(faces[point_ind])
                if 0 <= face_ind < self.mesh.n_faces:
                    vertices = self.mesh.faces[face_ind]
            if vertices is None or len(vertices) == 0:
                vertices = self.nearest_vertices(point)[0:1]

            for vertex in vertices:
                support_vertices.append(int(vertex))
                support_offsets.append(float(np.linalg.norm(point - self.mesh.vertices[int(vertex)])))
                support_point_indices.append(point_ind)

        samples = _SurfaceSamples(
            points=points,
            vertices=np.asarray(support_vertices, dtype=np.int64),
            offsets=np.asarray(support_offsets, dtype=float),
            point_indices=np.asarray(support_point_indices, dtype=np.int64),
        )
        _cache_store(self._contour_sample_cache, cache_key, samples)
        return samples

    def _distance_between_samples(self, samples_a: _SurfaceSamples, samples_b: _SurfaceSamples) -> GeodesicPathResult:
        self.last_dijkstra_settled_count = 0
        if (
            len(samples_a.points) == 0
            or len(samples_b.points) == 0
            or len(samples_a.vertices) == 0
            or len(samples_b.vertices) == 0
        ):
            return _empty_result()

        # Include support vertices, offsets and original point ordering, rather
        # than just geometric endpoints: contour provenance affects the graph
        # attachments and equal-distance candidates retain their original ties.
        arrays = (
            samples_a.points,
            samples_a.vertices,
            samples_a.offsets,
            samples_a.point_indices,
            samples_b.points,
            samples_b.vertices,
            samples_b.offsets,
            samples_b.point_indices,
        )
        cache_key = tuple(_array_cache_token(array) for array in arrays)
        cached = self._path_cache.get(cache_key)
        if cached is not None:
            self.path_cache_hits += 1
            self._path_cache.move_to_end(cache_key)
            return _copy_result(cached[0])

        self.path_cache_misses += 1
        result = self._uncached_distance_between_samples(samples_a, samples_b)
        cache_bytes = (
            sum(array.nbytes for array in arrays) + result.point_a.nbytes + result.point_b.nbytes + result.path_v.nbytes
        )
        if _PATH_CACHE_MAX_ENTRIES > 0 and cache_bytes <= _PATH_CACHE_MAX_BYTES:
            while self._path_cache and (
                len(self._path_cache) >= _PATH_CACHE_MAX_ENTRIES
                or self._path_cache_bytes + cache_bytes > _PATH_CACHE_MAX_BYTES
            ):
                _, (_, evicted_bytes) = self._path_cache.popitem(last=False)
                self._path_cache_bytes -= evicted_bytes
            self._path_cache[cache_key] = (_copy_result(result), cache_bytes)
            self._path_cache_bytes += cache_bytes
        return result

    def _uncached_distance_between_samples(
        self, samples_a: _SurfaceSamples, samples_b: _SurfaceSamples
    ) -> GeodesicPathResult:
        direct = self._same_face_point_path(samples_a, samples_b)
        if direct is not None:
            self.last_dijkstra_settled_count = 0
            return direct

        dijkstra = self._replayed_targeted_search(samples_a, samples_b)
        if dijkstra is None:
            dijkstra = self._multi_source_dijkstra(
                samples_a.vertices,
                samples_a.offsets,
                samples_a.point_indices,
                target_vertices=samples_b.vertices,
                target_offsets=samples_b.offsets,
            )
        self.last_dijkstra_settled_count = dijkstra.settled_count
        if dijkstra.best_target_sample >= 0:
            best_target_ind = dijkstra.best_target_sample
            target_vertex = int(samples_b.vertices[best_target_ind])
            best_distance = float(dijkstra.dist[target_vertex] + samples_b.offsets[best_target_ind])
        else:
            target_scores = dijkstra.dist[samples_b.vertices] + samples_b.offsets
            best_target_ind = int(np.argmin(target_scores))
            best_distance = float(target_scores[best_target_ind])
        if not math.isfinite(best_distance):
            return _empty_result()

        target_vertex = int(samples_b.vertices[best_target_ind])
        best_source_ind = int(dijkstra.source_index[target_vertex])
        best_target_point_ind = int(samples_b.point_indices[best_target_ind])
        if best_source_ind < 0:
            return _empty_result()

        start_vertex = int(dijkstra.source_vertex[target_vertex])
        path_v = self._path_vertices(start_vertex, target_vertex, dijkstra.prev)
        path_v = _with_endpoint(samples_a.points[best_source_ind], path_v, prepend=True)
        path_v = _with_endpoint(samples_b.points[best_target_point_ind], path_v, prepend=False)
        return GeodesicPathResult(
            distance=best_distance,
            point_a=samples_a.points[best_source_ind].copy(),
            index_a=best_source_ind,
            point_b=samples_b.points[best_target_point_ind].copy(),
            index_b=best_target_point_ind,
            path_v=path_v,
        )

    def _same_face_point_path(
        self, samples_a: _SurfaceSamples, samples_b: _SurfaceSamples
    ) -> GeodesicPathResult | None:
        """Use a straight interval only when one triangle contains both endpoints."""

        if len(samples_a.points) != 1 or len(samples_b.points) != 1:
            return None
        faces_a = {face for vertex in samples_a.vertices for face in self.vertex_faces[int(vertex)]}
        faces_b = {face for vertex in samples_b.vertices for face in self.vertex_faces[int(vertex)]}
        point_a = samples_a.points[0]
        point_b = samples_b.points[0]
        for face_index in sorted(faces_a & faces_b):
            triangle = self.mesh.vertices[self.mesh.faces[face_index]]
            edge_a, edge_b = triangle[1:] - triangle[0]
            edge_scale = float(np.linalg.norm(edge_a) * np.linalg.norm(edge_b))
            if np.linalg.norm(np.cross(edge_a, edge_b)) <= 64.0 * np.finfo(float).eps * edge_scale:
                continue
            scale = max(
                float(np.max(np.abs(triangle))),
                float(np.linalg.norm(edge_a)),
                float(np.linalg.norm(edge_b)),
                np.finfo(float).tiny,
            )
            tolerance_sq = (64.0 * np.finfo(float).eps * scale) ** 2
            if (
                _point_triangle_distance_sq(point_a, triangle) <= tolerance_sq
                and _point_triangle_distance_sq(point_b, triangle) <= tolerance_sq
            ):
                return GeodesicPathResult(
                    distance=float(np.linalg.norm(point_b - point_a)),
                    point_a=point_a.copy(),
                    index_a=0,
                    point_b=point_b.copy(),
                    index_b=0,
                    path_v=np.vstack((point_a, point_b)),
                )
        return None

    def nearest_vertices(self, points: np.ndarray, *, chunk_size: int = 4096) -> np.ndarray:
        return self._nearest_vertex_candidates(points, k=1, chunk_size=chunk_size)[:, 0]

    def _nearest_vertex_candidates(self, points: np.ndarray, *, k: int, chunk_size: int = 4096) -> np.ndarray:
        points = _as_point_array(points)
        if len(points) == 0:
            return np.zeros((0, 1), dtype=np.int64)
        k = max(1, min(int(k), self.mesh.n_vertices))
        nearest = np.zeros((len(points), k), dtype=np.int64)
        vertices = self.mesh.vertices
        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            distances = np.linalg.norm(points[start:stop, None, :] - vertices[None, :, :], axis=2)
            if k == 1:
                nearest[start:stop, 0] = np.argmin(distances, axis=1)
            else:
                candidate_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
                candidate_distances = np.take_along_axis(distances, candidate_indices, axis=1)
                order = np.argsort(candidate_distances, axis=1)
                nearest[start:stop] = np.take_along_axis(candidate_indices, order, axis=1)
        return nearest

    def _support_vertices_from_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mesh.n_faces == 0:
            nearest = self.nearest_vertices(points)
            return (
                nearest.astype(np.int64),
                np.linalg.norm(points - self.mesh.vertices[nearest], axis=1).astype(float),
                np.arange(len(points), dtype=np.int64),
            )

        nearest_candidates = self._nearest_vertex_candidates(points, k=8)
        batched = self._batched_support_faces(points, nearest_candidates)
        if batched is not None:
            return batched
        return self._support_vertices_point_by_point(points, nearest_candidates)

    def _vertex_face_table(self) -> np.ndarray:
        table = getattr(self, "_vertex_face_table_cache", None)
        if table is None:
            width = max((len(faces) for faces in self.vertex_faces), default=0)
            table = np.full((len(self.vertex_faces), max(width, 1)), -1, dtype=np.int64)
            for vertex, faces in enumerate(self.vertex_faces):
                table[vertex, : len(faces)] = faces
            self._vertex_face_table_cache = table
        return table

    def _batched_support_faces(
        self, points: np.ndarray, nearest_candidates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Vectorized ``_support_vertices_point_by_point`` with the same results.

        Every point-triangle distance repeats the scalar arithmetic. ``min`` over the
        candidate set keeps the first face in set order among equal distances, so rows
        with a tied or non-finite best distance are resolved by the scalar code.
        """

        table = self._vertex_face_table()
        candidates = table[nearest_candidates].reshape(len(points), -1)
        candidates.sort(axis=1)
        valid = candidates >= 0
        valid[:, 1:] &= candidates[:, 1:] != candidates[:, :-1]
        if not np.all(np.any(valid, axis=1)):
            return None
        vertices = self.mesh.vertices
        triangles = vertices[self.mesh.faces[np.where(valid, candidates, 0)]]
        distance_sq = np.where(valid, _point_triangle_distance_sq_batch(points, triangles), np.inf)
        best = np.argmin(distance_sq, axis=1)
        rows = np.arange(len(points))
        best_distance = distance_sq[rows, best]
        best_faces = candidates[rows, best]
        scalar_rows = ~np.isfinite(best_distance) | (np.count_nonzero(distance_sq == best_distance[:, None], axis=1) > 1)
        for row in np.flatnonzero(scalar_rows):
            point = points[row]
            candidate_faces: set[int] = set()
            for vertex in nearest_candidates[row]:
                candidate_faces.update(self.vertex_faces[int(vertex)])
            best_faces[row] = min(
                candidate_faces,
                key=lambda face_ind: _point_triangle_distance_sq(point, vertices[self.mesh.faces[int(face_ind)]]),
            )
        support_vertices = self.mesh.faces[best_faces].astype(np.int64)
        delta = points[:, None, :] - vertices[support_vertices]
        # sqrt of the sequential sum equals np.linalg.norm on each 3-vector.
        offsets = np.sqrt(delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1] + delta[..., 2] * delta[..., 2])
        return (
            support_vertices.reshape(-1),
            offsets.reshape(-1).astype(float),
            np.repeat(rows, support_vertices.shape[1]).astype(np.int64),
        )

    def _support_vertices_point_by_point(
        self, points: np.ndarray, nearest_candidates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        support_vertices: list[int] = []
        support_offsets: list[float] = []
        support_point_indices: list[int] = []
        for point_ind, point in enumerate(points):
            candidate_faces: set[int] = set()
            for vertex in nearest_candidates[point_ind]:
                candidate_faces.update(self.vertex_faces[int(vertex)])

            if not candidate_faces:
                vertices = nearest_candidates[point_ind, :1]
            else:
                best_face = min(
                    candidate_faces,
                    key=lambda face_ind: _point_triangle_distance_sq(
                        point, self.mesh.vertices[self.mesh.faces[int(face_ind)]]
                    ),
                )
                vertices = self.mesh.faces[int(best_face)]

            for vertex in np.asarray(vertices, dtype=np.int64):
                support_vertices.append(int(vertex))
                support_offsets.append(float(np.linalg.norm(point - self.mesh.vertices[int(vertex)])))
                support_point_indices.append(point_ind)

        return (
            np.asarray(support_vertices, dtype=np.int64),
            np.asarray(support_offsets, dtype=float),
            np.asarray(support_point_indices, dtype=np.int64),
        )

    def _replayed_targeted_search(
        self, samples_a: _SurfaceSamples, samples_b: _SurfaceSamples
    ) -> _DijkstraResult | None:
        """Answer a targeted search from a cached untargeted search from the same sources.

        A targeted search pops exactly the same entries as the untargeted one until it
        stops, so its settled vertices, their distances, predecessors and sources are
        the untargeted values. Replaying the settle order with the targeted stopping
        rule gives the same best target sample and settled count. The caller reads
        only settled vertices (the chosen target and its predecessor chain), or, when
        no target was reached, distances of a search that ran to completion.
        """

        key = tuple(
            _array_cache_token(array) for array in (samples_a.vertices, samples_a.offsets, samples_a.point_indices)
        )
        entry = self._full_search_cache.get(key)
        if entry is None:
            full = self._multi_source_dijkstra(
                samples_a.vertices, samples_a.offsets, samples_a.point_indices, record_settled_order=True
            )
            distances = full.settled_distances
            if len(distances) > 1 and not np.all(distances[1:] >= distances[:-1]):
                return None
            position = np.full(self.mesh.n_vertices, -1, dtype=np.int64)
            position[full.settled_vertices] = np.arange(len(full.settled_vertices))
            for array in (full.dist, full.prev, full.source_index, full.source_vertex, distances, position):
                array.flags.writeable = False
            size = sum(array.nbytes for array in (full.dist, full.prev, full.source_index, full.source_vertex))
            size += distances.nbytes + full.settled_vertices.nbytes + position.nbytes
            while self._full_search_cache and self._full_search_cache_bytes + size > _FULL_SEARCH_CACHE_MAX_BYTES:
                _, (_, _, evicted) = self._full_search_cache.popitem(last=False)
                self._full_search_cache_bytes -= evicted
            if size <= _FULL_SEARCH_CACHE_MAX_BYTES:
                self._full_search_cache[key] = (full, position, size)
                self._full_search_cache_bytes += size
            entry = (full, position, size)
        else:
            self._full_search_cache.move_to_end(key)
        full, position, _size = entry

        target_vertices = np.asarray(samples_b.vertices, dtype=np.int64)
        target_offsets = np.asarray(samples_b.offsets, dtype=float).tolist()
        n_vertices = self.mesh.n_vertices
        in_range = (target_vertices >= 0) & (target_vertices < n_vertices)
        target_position = np.full(len(target_vertices), -1, dtype=np.int64)
        target_position[in_range] = position[target_vertices[in_range]]
        reached = np.flatnonzero(target_position >= 0)
        # Targets in settle order; several samples on one vertex in sample order.
        reached = reached[np.lexsort((reached, target_position[reached]))].tolist()
        positions = target_position.tolist()
        distances = full.settled_distances
        best_distance = math.inf
        best_sample = -1
        settled_count = None
        index = 0
        while index < len(reached):
            settled = positions[reached[index]]
            current = float(distances[settled])
            if current >= best_distance:
                break
            while index < len(reached) and positions[reached[index]] == settled:
                sample = reached[index]
                score = current + target_offsets[sample]
                if score < best_distance:
                    best_distance = score
                    best_sample = sample
                index += 1
            if current >= best_distance:
                settled_count = settled + 1
                break
        if settled_count is None:
            # Stops at the first settled vertex at or beyond the best target (or runs out).
            settled_count = int(np.searchsorted(distances, best_distance, side="left"))
        return _DijkstraResult(
            dist=full.dist,
            prev=full.prev,
            source_index=full.source_index,
            source_vertex=full.source_vertex,
            best_target_sample=best_sample,
            settled_count=settled_count,
        )

    def _multi_source_dijkstra(
        self,
        source_vertices: np.ndarray,
        source_offsets: np.ndarray,
        source_point_indices: np.ndarray,
        *,
        target_vertices: np.ndarray | None = None,
        target_offsets: np.ndarray | None = None,
        record_settled_order: bool = False,
    ) -> _DijkstraResult:
        # Plain Python lists inside the loop: numpy element access is far slower,
        # and float64 values, comparisons and heap order are unchanged.
        n_vertices = self.mesh.n_vertices
        dist = [math.inf] * n_vertices
        prev = [-1] * n_vertices
        source_index = [-1] * n_vertices
        source_vertex = [-1] * n_vertices
        heap: list[tuple[float, int]] = []
        point_indices = np.asarray(source_point_indices).tolist()
        offsets = np.asarray(source_offsets, dtype=float).tolist()
        for point_ind, vertex in enumerate(np.asarray(source_vertices, dtype=np.int64).tolist()):
            offset = offsets[point_ind]
            if offset < dist[vertex]:
                dist[vertex] = offset
                source_index[vertex] = int(point_indices[point_ind])
                source_vertex[vertex] = vertex
                prev[vertex] = -1
                heapq.heappush(heap, (offset, vertex))

        target_samples_by_vertex: list[list[int]] | None = None
        target_offset_values: list[float] = []
        if target_vertices is not None and target_offsets is not None and len(target_vertices):
            target_samples_by_vertex = [[] for _ in range(n_vertices)]
            for target_sample_ind, vertex in enumerate(np.asarray(target_vertices, dtype=np.int64).tolist()):
                if 0 <= vertex < n_vertices:
                    target_samples_by_vertex[vertex].append(target_sample_ind)
            target_offset_values = np.asarray(target_offsets, dtype=float).tolist()

        best_target_sample = -1
        best_target_distance = math.inf
        settled_count = 0
        settled_vertices: list[int] = []
        settled_distances: list[float] = []
        record = record_settled_order and target_samples_by_vertex is None
        adjacency = self.adjacency
        heappop = heapq.heappop
        heappush = heapq.heappush
        while heap:
            current_dist, vertex = heappop(heap)
            if current_dist > dist[vertex]:
                continue
            if current_dist >= best_target_distance:
                break
            settled_count += 1
            if record:
                settled_vertices.append(vertex)
                settled_distances.append(current_dist)
            if target_samples_by_vertex is not None:
                for target_sample_ind in target_samples_by_vertex[vertex]:
                    target_score = current_dist + target_offset_values[target_sample_ind]
                    if target_score < best_target_distance:
                        best_target_distance = target_score
                        best_target_sample = target_sample_ind
                if current_dist >= best_target_distance:
                    break
            for neighbor, weight in adjacency[vertex]:
                candidate = current_dist + weight
                if candidate < dist[neighbor]:
                    dist[neighbor] = candidate
                    prev[neighbor] = vertex
                    source_index[neighbor] = source_index[vertex]
                    source_vertex[neighbor] = source_vertex[vertex]
                    heappush(heap, (candidate, neighbor))
        return _DijkstraResult(
            dist=np.asarray(dist, dtype=float),
            prev=np.asarray(prev, dtype=np.int64),
            source_index=np.asarray(source_index, dtype=np.int64),
            source_vertex=np.asarray(source_vertex, dtype=np.int64),
            best_target_sample=best_target_sample,
            settled_count=settled_count,
            settled_vertices=np.asarray(settled_vertices, dtype=np.int64) if record else None,
            settled_distances=np.asarray(settled_distances, dtype=float) if record else None,
        )

    def _path_vertices(self, source_vertex: int, target_vertex: int, prev: np.ndarray) -> np.ndarray:
        if source_vertex == target_vertex:
            return self.mesh.vertices[[source_vertex]].copy()
        path = []
        current = int(target_vertex)
        while current >= 0:
            path.append(current)
            if current == source_vertex:
                break
            current = int(prev[current])
        if not path or path[-1] != source_vertex:
            return np.zeros((0, 3), dtype=float)
        path.reverse()
        return self.mesh.vertices[np.asarray(path, dtype=np.int64)].copy()


def _build_vertex_adjacency(mesh: Mesh) -> list[list[tuple[int, float]]]:
    adjacency: list[dict[int, float]] = [dict() for _ in range(mesh.n_vertices)]
    for face in mesh.faces:
        for a, b in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            weight = float(np.linalg.norm(mesh.vertices[a] - mesh.vertices[b]))
            if b not in adjacency[a] or weight < adjacency[a][b]:
                adjacency[a][b] = weight
                adjacency[b][a] = weight
    return [[(neighbor, weight) for neighbor, weight in neighbors.items()] for neighbors in adjacency]


def _points_sample_cache_key(points: np.ndarray) -> tuple[object, ...]:
    return ("points", *_array_cache_token(np.asarray(points, dtype=float)))


def _contour_sample_cache_key(contour: ContourLine, points: np.ndarray) -> tuple[object, ...]:
    edge_vertices = contour.surface_edge_vertices
    faces = contour.surface_faces
    return (
        "contour",
        *_array_cache_token(np.asarray(points, dtype=float)),
        None if edge_vertices is None else _array_cache_token(np.asarray(edge_vertices, dtype=np.int64)),
        None if faces is None else _array_cache_token(np.asarray(faces, dtype=np.int64)),
    )


def _array_cache_token(array: np.ndarray) -> tuple[tuple[int, ...], str, bytes]:
    contiguous = np.ascontiguousarray(array)
    return contiguous.shape, contiguous.dtype.str, contiguous.tobytes()


def _cache_store(
    cache: dict[tuple[object, ...], _SurfaceSamples], key: tuple[object, ...], samples: _SurfaceSamples
) -> None:
    if len(cache) >= _SAMPLE_CACHE_MAX_ENTRIES:
        cache.pop(next(iter(cache)))
    cache[key] = samples


def _build_vertex_faces(mesh: Mesh) -> list[list[int]]:
    vertex_faces: list[list[int]] = [[] for _ in range(mesh.n_vertices)]
    for face_ind, face in enumerate(mesh.faces):
        for vertex in face:
            vertex_faces[int(vertex)].append(face_ind)
    return vertex_faces


def _dot3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Same summation order as np.dot on 3-vectors, so values match bit for bit.
    return left[..., 0] * right[..., 0] + left[..., 1] * right[..., 1] + left[..., 2] * right[..., 2]


def _point_triangle_distance_sq_batch(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """``_point_triangle_distance_sq`` for points ``(n, 3)`` against triangles ``(n, k, 3, 3)``.

    Uses the same operations and region order as ``_closest_point_on_triangle``.
    """

    point = points[:, None, :]
    a = triangles[:, :, 0]
    b = triangles[:, :, 1]
    c = triangles[:, :, 2]
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = _dot3(ab, ap)
    d2 = _dot3(ac, ap)
    bp = point - b
    d3 = _dot3(ab, bp)
    d4 = _dot3(ac, bp)
    vc = d1 * d4 - d3 * d2
    cp = point - c
    d5 = _dot3(ab, cp)
    d6 = _dot3(ac, cp)
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        closest = a + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None]
        # Lowest priority first; each later region overrides, so the first match in
        # the scalar order wins.
        regions = (
            (np.abs(denom) <= 1e-30, a),
            (
                (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0),
                b + ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None] * (c - b),
            ),
            ((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0), a + (d2 / (d2 - d6))[..., None] * ac),
            ((d6 >= 0.0) & (d5 <= d6), c),
            ((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0), a + (d1 / (d1 - d3))[..., None] * ab),
            ((d3 >= 0.0) & (d4 <= d3), b),
            ((d1 <= 0.0) & (d2 <= 0.0), a),
        )
        for mask, region_point in regions:
            if np.any(mask):
                closest = np.where(mask[..., None], region_point, closest)
    difference = point - closest
    return _dot3(difference, difference)


def _point_triangle_distance_sq(point: np.ndarray, triangle: np.ndarray) -> float:
    closest = _closest_point_on_triangle(np.asarray(point, dtype=float), np.asarray(triangle, dtype=float))
    return float(np.dot(point - closest, point - closest))


def _closest_point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a = triangle[0]
    b = triangle[1]
    c = triangle[2]
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    denom = va + vb + vc
    if abs(denom) <= 1e-30:
        return a
    v = vb / denom
    w = vc / denom
    return a + ab * v + ac * w


def _as_point_array(value: np.ndarray) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected a point array with shape (n, 3), got {points.shape}.")
    finite = np.all(np.isfinite(points), axis=1)
    return points[finite]


def _empty_result() -> GeodesicPathResult:
    return GeodesicPathResult(
        distance=math.inf,
        point_a=np.zeros(3, dtype=float),
        index_a=-1,
        point_b=np.zeros(3, dtype=float),
        index_b=-1,
        path_v=np.zeros((0, 3), dtype=float),
    )


def _copy_result(result: GeodesicPathResult) -> GeodesicPathResult:
    return GeodesicPathResult(
        distance=result.distance,
        point_a=result.point_a.copy(),
        index_a=result.index_a,
        point_b=result.point_b.copy(),
        index_b=result.index_b,
        path_v=result.path_v.copy(),
    )


def _empty_samples() -> _SurfaceSamples:
    return _SurfaceSamples(
        points=np.zeros((0, 3), dtype=float),
        vertices=np.zeros(0, dtype=np.int64),
        offsets=np.zeros(0, dtype=float),
        point_indices=np.zeros(0, dtype=np.int64),
    )


def _with_endpoint(endpoint: np.ndarray, path_v: np.ndarray, *, prepend: bool) -> np.ndarray:
    endpoint = np.asarray(endpoint, dtype=float).reshape(1, 3)
    if len(path_v) == 0:
        return endpoint.copy()
    compare = path_v[0] if prepend else path_v[-1]
    if np.linalg.norm(compare - endpoint[0]) <= 1e-12:
        return path_v
    return np.vstack((endpoint, path_v)) if prepend else np.vstack((path_v, endpoint))
