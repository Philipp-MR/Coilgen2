from __future__ import annotations

import numpy as np

from .models import BasisElement, CoilPart


MAX_CURRENT_DENSITY_MATRIX_BYTES = 512 * 1024**2


def calculate_one_ring_by_mesh(parts: list[CoilPart]) -> list[CoilPart]:
    for part in parts:
        mesh = part.mesh
        if mesh.vertex_normals is None:
            raise ValueError("Mesh vertex normals must be computed before one-ring construction.")
        node_triangles: list[list[int]] = [[] for _ in range(mesh.n_vertices)]
        for face_ind, face in enumerate(mesh.faces):
            for node in face:
                node_triangles[int(node)].append(face_ind)
        one_ring_list: list[np.ndarray] = []
        for node_ind, tri_inds in enumerate(node_triangles):
            pairs: list[list[int]] = []
            for face_ind in tri_inds:
                face = mesh.faces[face_ind]
                other = [int(node) for node in face if int(node) != node_ind]
                if len(other) != 2:
                    continue
                point_aa = mesh.vertices[other[0]]
                point_bb = mesh.vertices[other[1]]
                point_cc = mesh.vertices[node_ind]
                cross_vec = np.cross(point_bb - point_aa, point_aa - point_cc)
                if np.sign(np.dot(mesh.vertex_normals[node_ind], cross_vec)) > 0:
                    other.reverse()
                pairs.append(other)
            one_ring_list.append(np.asarray(pairs, dtype=np.int64))
        part.node_triangles = [np.asarray(tris, dtype=np.int64) for tris in node_triangles]
        part.one_ring_list = one_ring_list
    return parts


def calculate_basis_functions(parts: list[CoilPart]) -> list[CoilPart]:
    for part in parts:
        mesh = part.mesh
        dense_bytes = mesh.n_vertices * mesh.n_faces * 3 * np.dtype(float).itemsize
        current_density_mat = (
            np.zeros((mesh.n_vertices, mesh.n_faces, 3), dtype=float)
            if dense_bytes <= MAX_CURRENT_DENSITY_MATRIX_BYTES
            else None
        )
        basis_elements: list[BasisElement] = []
        for node_ind in range(mesh.n_vertices):
            node_point = mesh.vertices[node_ind]
            tri_inds = part.node_triangles[node_ind]
            one_ring = part.one_ring_list[node_ind]
            areas: list[float] = []
            face_normals: list[np.ndarray] = []
            triangle_points: list[np.ndarray] = []
            currents: list[np.ndarray] = []
            for local_tri_ind, face_ind in enumerate(tri_inds):
                point_b = mesh.vertices[one_ring[local_tri_ind, 0]]
                point_c = mesh.vertices[one_ring[local_tri_ind, 1]]
                area = np.linalg.norm(np.cross(point_c - node_point, point_b - node_point)) / 2.0
                if area <= 0:
                    current = np.zeros(3, dtype=float)
                    face_normal = np.zeros(3, dtype=float)
                else:
                    normal_vec = np.cross(point_c - node_point, point_c - point_b)
                    face_normal = normal_vec / max(np.linalg.norm(normal_vec), 1e-15)
                    current = (point_c - point_b) / (2.0 * area)
                areas.append(area)
                face_normals.append(face_normal)
                triangle_points.append(mesh.vertices[mesh.faces[face_ind]])
                currents.append(current)
                if current_density_mat is not None:
                    current_density_mat[node_ind, face_ind] = current
            basis_elements.append(
                BasisElement(
                    triangles=tri_inds,
                    one_ring=one_ring,
                    area=np.asarray(areas, dtype=float),
                    face_normal=np.asarray(face_normals, dtype=float),
                    triangle_points_abc=np.asarray(triangle_points, dtype=float),
                    current=np.asarray(currents, dtype=float),
                )
            )
        part.basis_elements = basis_elements
        part.current_density_mat = current_density_mat
    return parts


def current_density_from_basis(part: CoilPart) -> np.ndarray | None:
    if part.current_density is not None:
        return np.asarray(part.current_density, dtype=float)
    if part.stream_function is None or not part.basis_elements:
        return None
    stream_function = np.asarray(part.stream_function, dtype=float)
    if len(stream_function) != len(part.basis_elements):
        return None
    current_density = np.zeros((part.mesh.n_faces, 3), dtype=float)
    for node_ind, basis in enumerate(part.basis_elements):
        if len(basis.triangles) == 0:
            continue
        current_density[np.asarray(basis.triangles, dtype=np.int64)] += stream_function[node_ind] * np.asarray(
            basis.current, dtype=float
        )
    return current_density
