"""Refine connector seeds inside a bounded strip of actual mesh triangles.

Crossing points are affine parameters on shared edges. Each interval therefore
stays inside one convex triangle, including its interior. Length minimization
removes mesh-vertex zigzags without projecting or moving the winding contours.
This is a corridor-local proposal, not an obstacle, field or body certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class CorridorRoute:
    points: np.ndarray
    face_indices: tuple[int, ...]
    status: str
    optimizer_converged: bool = False


class SurfaceCorridor:
    def __init__(self, mesh, vertex_faces, *, maximum_faces=512, maximum_search_faces=16384):
        self.vertices, self.faces = mesh.vertices, mesh.faces
        self.vertex_faces = vertex_faces
        self.maximum_faces = maximum_faces
        self.maximum_search_faces = maximum_search_faces
        self.scale = float(np.linalg.norm(np.ptp(self.vertices, axis=0)))
        self.tolerance = max(self.scale * 128 * np.finfo(float).eps, 1e-13)
        self.centers = self.vertices[self.faces].mean(axis=1)
        self.neighbors = [[] for _ in self.faces]
        edges = {}
        for face_index, face in enumerate(self.faces):
            for a, b in zip(face, np.roll(face, -1)):
                edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(face_index)
        triangles = self.vertices[self.faces]
        self.valid = bool(
            self.scale > 0
            and len(self.faces)
            and np.isfinite(triangles).all()
            and np.all(
                np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
                > 0
            )
        )
        for edge, incident in edges.items():
            if len(incident) > 2:
                self.valid = False
            elif len(incident) == 2:
                a, b = incident
                midpoint = self.vertices[list(edge)].mean(axis=0)
                weight = float(np.linalg.norm(self.centers[a] - midpoint) + np.linalg.norm(self.centers[b] - midpoint))
                self.neighbors[a].append((b, edge, weight))
                self.neighbors[b].append((a, edge, weight))
        for values in self.neighbors:
            values.sort(key=lambda value: value[0])

    def _endpoint_faces(self, point, support_vertices):
        from .surface_distance import _point_triangle_distance_sq

        candidates = {face for vertex in support_vertices for face in self.vertex_faces[int(vertex)]}
        return sorted(
            face
            for face in candidates
            if _point_triangle_distance_sq(point, self.vertices[self.faces[face]]) <= self.tolerance**2
        )

    def _strip(self, source_faces, target_faces):
        targets = set(target_faces)
        distance, previous, heap = {}, {}, []
        for face in source_faces:
            distance[face] = 0.0
            heapq.heappush(heap, (0.0, face))
        evaluated = 0
        end = None
        while heap and evaluated < self.maximum_search_faces:
            cost, face = heapq.heappop(heap)
            if cost != distance.get(face):
                continue
            evaluated += 1
            if face in targets:
                end = face
                break
            for other, edge, weight in self.neighbors[face]:
                score = cost + weight
                if score < distance.get(other, np.inf):
                    distance[other] = score
                    previous[other] = (face, edge)
                    heapq.heappush(heap, (score, other))
        if end is None:
            return (), ()
        faces, portals = [end], []
        while faces[-1] in previous:
            face, edge = previous[faces[-1]]
            faces.append(face)
            portals.append(edge)
            if len(faces) > self.maximum_faces:
                return (), ()
        return tuple(reversed(faces)), tuple(reversed(portals))

    def route(self, start, end, source_vertices, target_vertices, seed):
        if not self.valid:
            return CorridorRoute(seed.copy(), (), "unsupported_mesh")
        source_faces = self._endpoint_faces(start, source_vertices)
        target_faces = self._endpoint_faces(end, target_vertices)
        if not source_faces or not target_faces:
            return CorridorRoute(seed.copy(), (), "endpoint_not_on_surface")
        faces, portals = self._strip(source_faces, target_faces)
        if not faces:
            return CorridorRoute(seed.copy(), (), "corridor_unavailable_or_budget")
        if not portals:
            return CorridorRoute(np.vstack((start, end)), faces, "same_face", True)
        origins = self.vertices[np.asarray(portals)[:, 0]]
        directions = self.vertices[np.asarray(portals)[:, 1]] - origins
        # Center and normalize to keep optimization tolerances independent of
        # translation and metre/millimetre-sized supports. Endpoint arrays are
        # restored exactly; no closest-surface projection is performed.
        a, b = np.zeros(3), (end - start) / self.scale
        origins_n, directions_n = (origins - start) / self.scale, directions / self.scale

        def objective(parameters):
            points = np.vstack((a, origins_n + parameters[:, None] * directions_n, b))
            delta = np.diff(points, axis=0)
            length = np.sqrt(np.einsum("ij,ij->i", delta, delta) + 1e-24)
            unit = delta / length[:, None]
            gradient = np.einsum("ij,ij->i", unit[:-1] - unit[1:], directions_n)
            return float(length.sum()), gradient

        solution = minimize(
            objective,
            np.full(len(portals), 0.5),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * len(portals),
            options={"maxiter": 200, "ftol": 1e-14, "gtol": 1e-9, "maxls": 40},
        )
        parameters = np.clip(solution.x, 0.0, 1.0)
        points = np.vstack((start, origins + parameters[:, None] * directions, end))
        new_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        old_length = float(np.linalg.norm(np.diff(seed, axis=0), axis=1).sum())
        if not np.isfinite(points).all() or new_length > old_length + self.tolerance:
            return CorridorRoute(seed.copy(), (), "seed_retained")
        # Keep the support proof explicit: points[i:i+2] lie in faces[i].
        # Coincident portals are retained here; ordinary path assembly removes
        # exact duplicate vertices without changing any physical interval.
        return CorridorRoute(points, faces, "triangle_corridor_refined", bool(solution.success))
