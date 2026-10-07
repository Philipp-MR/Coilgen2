from __future__ import annotations

import math
import json
import struct
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import CoilGenConfig
from .models import Mesh
from .object_registry import get_in_memory_mesh, is_in_memory_mesh_key
from .suppression import apply_suppression_clearance


GeometryBuilder = Callable[[CoilGenConfig], Mesh]


@dataclass(frozen=True, slots=True)
class GeometryFactory:
    """Registered generated-mesh factory for user-facing geometry APIs."""

    name: str
    description: str
    parameter_attribute: str
    builder: GeometryBuilder

    def build(self, config: CoilGenConfig) -> Mesh:
        return self.builder(config)


_GEOMETRY_FACTORIES: dict[str, GeometryFactory] = {}


def register_geometry_factory(factory: GeometryFactory, *, replace: bool = False) -> GeometryFactory:
    """Register a named generated geometry factory.

    This is the extension point for new user-facing geometry APIs. A new
    generator should expose one stable ``coil_mesh_file`` name, one parameter
    attribute on ``CoilGenConfig``, and one builder function returning a
    ``Mesh`` in SI units.
    """

    key = _geometry_factory_key(factory.name)
    if key in _GEOMETRY_FACTORIES and not replace:
        raise ValueError(f"Geometry factory {factory.name!r} is already registered.")
    _GEOMETRY_FACTORIES[key] = factory
    return factory


def geometry_factory(name: str) -> GeometryFactory | None:
    return _GEOMETRY_FACTORIES.get(_geometry_factory_key(name))


def available_geometry_factories() -> tuple[GeometryFactory, ...]:
    return tuple(sorted(_GEOMETRY_FACTORIES.values(), key=lambda factory: factory.name))


def _geometry_factory_key(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def rotation_matrix_by_vector(rot_vec: np.ndarray, rot_angle: float) -> np.ndarray:
    rot_vec = np.asarray(rot_vec, dtype=float)
    norm = np.linalg.norm(rot_vec)
    if norm < 1e-15 or abs(rot_angle) < 1e-15:
        return np.eye(3)
    ux, uy, uz = rot_vec / norm
    s = math.sin(rot_angle)
    c = math.cos(rot_angle)
    k = 1.0 - c
    return np.array(
        [
            [c + ux * ux * k, ux * uy * k - uz * s, ux * uz * k + uy * s],
            [uy * ux * k + uz * s, c + uy * uy * k, uy * uz * k - ux * s],
            [uz * ux * k - uy * s, uz * uy * k + ux * s, c + uz * uz * k],
        ],
        dtype=float,
    )


def normalize_mesh(mesh: Mesh) -> Mesh:
    mesh = make_unique_noded_mesh(mesh)
    mesh.face_normals = compute_face_normals(mesh)
    mesh.vertex_normals = compute_vertex_normals(mesh)
    mesh.boundaries = boundary_loops(mesh)
    return mesh


def make_unique_noded_mesh(mesh: Mesh) -> Mesh:
    key_to_index: dict[tuple[float, float, float], int] = {}
    vertices: list[np.ndarray] = []
    index_map: list[int] = []
    for vertex in mesh.vertices:
        key = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
        if key not in key_to_index:
            key_to_index[key] = len(vertices)
            vertices.append(vertex)
        index_map.append(key_to_index[key])
    index_map_array = np.asarray(index_map, dtype=np.int64)
    faces = index_map_array[mesh.faces]
    return Mesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=faces,
        unique_vert_inds=np.arange(len(vertices), dtype=np.int64),
        metadata=dict(mesh.metadata),
    )


def compute_face_normals(mesh: Mesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths > 0] /= lengths[lengths > 0, None]
    return normals


def compute_vertex_normals(mesh: Mesh) -> np.ndarray:
    face_normals = mesh.face_normals if mesh.face_normals is not None else compute_face_normals(mesh)
    normals = np.zeros_like(mesh.vertices)
    for face, normal in zip(mesh.faces, face_normals):
        normals[face] += normal
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths > 0] /= lengths[lengths > 0, None]
    return normals


def _metadata(
    geometry_type: str, parameters: dict[str, object], *, symmetry: tuple[str, ...] = ()
) -> dict[str, object]:
    return {
        "geometry_type": geometry_type,
        "parameters": dict(parameters),
        "units": "SI",
        "symmetry": tuple(symmetry),
        "topology": "orientable triangular surface",
    }


def _with_metadata(
    mesh: Mesh, geometry_type: str, parameters: dict[str, object], *, symmetry: tuple[str, ...] = ()
) -> Mesh:
    mesh.metadata.update(_metadata(geometry_type, parameters, symmetry=symmetry))
    return mesh


def _apply_rotation_center(
    vertices: np.ndarray,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> np.ndarray:
    rot = rotation_matrix_by_vector(
        np.asarray([rotation_vector_x, rotation_vector_y, rotation_vector_z], dtype=float),
        float(rotation_angle),
    )
    center = np.asarray([center_position_x, center_position_y, center_position_z], dtype=float)
    return np.asarray(vertices, dtype=float) @ rot.T + center


def _grid_faces(row_count: int, col_count: int, *, wrap_cols: bool = False, outward: bool = True) -> np.ndarray:
    faces = []
    max_col = col_count if wrap_cols else col_count - 1
    for row in range(row_count - 1):
        base = row * col_count
        next_base = (row + 1) * col_count
        for col in range(max_col):
            col2 = (col + 1) % col_count
            a = base + col
            b = base + col2
            c = next_base + col
            d = next_base + col2
            if outward:
                faces.append([a, b, d])
                faces.append([a, d, c])
            else:
                faces.append([a, d, b])
                faces.append([a, c, d])
    return np.asarray(faces, dtype=np.int64)


def build_parametric_surface_mesh(
    surface: Callable[[float, float], tuple[float, float, float] | np.ndarray],
    u_values: np.ndarray,
    v_values: np.ndarray,
    *,
    wrap_u: bool = False,
    wrap_v: bool = False,
    geometry_type: str = "parametric_surface",
    parameters: dict[str, object] | None = None,
) -> Mesh:
    """Sample an orientable parametric surface on a rectangular parameter grid.

    ``surface(u, v)`` must return one 3D point in metres. Faces use the
    parameter-grid orientation. If normals point inward for a particular
    parameterization, reverse the parameter order in the caller.
    """

    u = np.asarray(u_values, dtype=float).reshape(-1)
    v = np.asarray(v_values, dtype=float).reshape(-1)
    if len(u) < 2 or len(v) < 2:
        raise ValueError("Parametric surface requires at least two u and two v samples.")
    vertices = []
    for u_value in u:
        for v_value in v:
            point = np.asarray(surface(float(u_value), float(v_value)), dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError("Parametric surface callable must return finite 3D points.")
            vertices.append(point)
    if wrap_u:
        raise ValueError("wrap_u is not supported by this row-major sampler; put the periodic parameter on v.")
    faces = _grid_faces(len(u), len(v), wrap_cols=wrap_v, outward=True)
    mesh = Mesh(vertices=np.asarray(vertices, dtype=float), faces=faces)
    return _with_metadata(mesh, geometry_type, parameters or {})


def offset_mesh_along_vertex_normals(mesh: Mesh, separation: float, *, role: str = "shield") -> Mesh:
    """Return an offset copy with identical face topology.

    This is the generic paired-shield helper for primitives where the shield is
    defined by fixed outward separation from the primary support. The caller
    should pass a consistently oriented primary mesh.
    """

    if float(separation) <= 0.0:
        raise ValueError("Offset separation must be positive.")
    normalized = normalize_mesh(mesh.copy())
    normals = normalized.vertex_normals
    if normals is None or len(normals) != normalized.n_vertices:
        normals = compute_vertex_normals(normalized)
    offset = Mesh(
        vertices=normalized.vertices + float(separation) * normals,
        faces=normalized.faces.copy(),
        metadata={
            **dict(normalized.metadata),
            "role": role,
            "offset_from": dict(normalized.metadata),
            "offset_separation_m": float(separation),
            "same_topology_as_primary": True,
        },
    )
    return offset


def boundary_loops(mesh: Mesh) -> list[np.ndarray]:
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in mesh.faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (int(min(u, v)), int(max(u, v)))
            edge_count[edge] += 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    if not boundary_edges:
        return []
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    visited_edges: set[tuple[int, int]] = set()
    loops: list[np.ndarray] = []
    for start_a, start_b in boundary_edges:
        edge = (min(start_a, start_b), max(start_a, start_b))
        if edge in visited_edges:
            continue
        loop = [start_a]
        prev = -1
        current = start_a
        next_node = start_b
        while True:
            visited_edges.add((min(current, next_node), max(current, next_node)))
            loop.append(next_node)
            prev, current = current, next_node
            if current == start_a:
                break
            candidates = [node for node in adjacency[current] if node != prev]
            if not candidates:
                break
            unused = [node for node in candidates if (min(current, node), max(current, node)) not in visited_edges]
            next_node = unused[0] if unused else candidates[0]
            if len(loop) > mesh.n_vertices + 5:
                break
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def load_mesh_from_config(config: CoilGenConfig, project_root: Path) -> Mesh:
    mesh_name = config.coil_mesh_file
    if mesh_name == "none":
        raise ValueError("coil_mesh_file must be specified.")
    if is_in_memory_mesh_key(mesh_name):
        return normalize_mesh(get_in_memory_mesh(mesh_name))
    if mesh_name.lower().endswith(".npz"):
        return normalize_mesh(read_mesh_npz(config.resolved_geometry_source_path(project_root) / mesh_name))
    if mesh_name.lower().endswith(".stl"):
        mesh_path = config.resolved_geometry_source_path(project_root) / mesh_name
        mesh = read_stl(mesh_path)
        length_scale_to_m = float(config.imported_mesh_length_scale_to_m)
        if length_scale_to_m != 1.0:
            mesh.vertices = np.asarray(mesh.vertices, dtype=float) * length_scale_to_m
            mesh.metadata.update(
                {
                    "source_length_scale_to_m": length_scale_to_m,
                    "units": "SI",
                }
            )
        return normalize_mesh(mesh)
    registered = geometry_factory(mesh_name)
    if registered is not None:
        return normalize_mesh(registered.build(config))
    if mesh_name == "create cylinder mesh":
        return normalize_mesh(_rotate_generated_mesh(build_cylinder_mesh(*config.cylinder_mesh_parameter_list), config))
    if mesh_name == "create planary mesh":
        return normalize_mesh(_rotate_generated_mesh(build_planar_mesh(*config.planar_mesh_parameter_list), config))
    if mesh_name == "create circular mesh":
        return normalize_mesh(_rotate_generated_mesh(build_circular_mesh(*config.circular_mesh_parameter_list), config))
    if mesh_name == "create annular disc mesh":
        return normalize_mesh(build_annular_disc_mesh(*config.annular_disc_mesh_parameter_list))
    if mesh_name == "create partial cylinder mesh":
        return normalize_mesh(build_partial_cylinder_mesh(*config.partial_cylinder_mesh_parameter_list))
    if mesh_name == "create elliptical cylinder mesh":
        return normalize_mesh(build_elliptical_cylinder_mesh(*config.elliptical_cylinder_mesh_parameter_list))
    if mesh_name == "create surface of revolution mesh":
        return normalize_mesh(build_surface_of_revolution_mesh(*config.surface_of_revolution_mesh_parameter_list))
    if mesh_name == "create circular bi-planary mesh":
        return normalize_mesh(build_circular_biplanar_mesh(*config.circular_biplanar_mesh_parameter_list))
    if mesh_name == "create bi-planary mesh":
        return normalize_mesh(_rotate_generated_mesh(build_biplanar_mesh(*config.biplanar_mesh_parameter_list), config))
    if mesh_name == "create shielded bi-planary mesh":
        return normalize_mesh(
            _rotate_generated_mesh(
                build_shielded_biplanar_mesh(*config.shielded_circular_biplanar_mesh_parameter_list),
                config,
            )
        )
    if mesh_name == "create flanged split bore mesh":
        return normalize_mesh(
            _rotate_generated_mesh(build_split_bore_flange_mesh(*config.flanged_split_bore_mesh_parameter_list), config)
        )
    if mesh_name == "create shielded flanged split bore mesh":
        values = list(config.shielded_flanged_split_bore_mesh_parameter_list)
        if len(values) not in (6, 10):
            raise ValueError(
                "shielded_flanged_split_bore_mesh_parameter_list must contain "
                "either [edge_len r_inner r_outer gap half_length shield_separation] "
                "or the legacy [edge_len r_inner r_outer gap half_length shield_edge_len "
                "shield_r_inner shield_r_outer shield_gap shield_half_length]."
            )
        if len(values) == 6:
            edge_len, r_inner, r_outer, gap, half_length, shield_separation = values
            edge_len2 = edge_len
            r_inner2 = float(r_outer) + float(shield_separation)
            r_outer2 = np.nan
            gap2 = gap
            half_length2 = half_length
        else:
            edge_len, r_inner, r_outer, gap, half_length, edge_len2, r_inner2, r_outer2, gap2, half_length2 = values
            edge_len2 = edge_len if np.isnan(edge_len2) else edge_len2
            gap2 = gap if np.isnan(gap2) else gap2
            half_length2 = half_length if np.isnan(half_length2) else half_length2
        return normalize_mesh(
            _rotate_generated_mesh(
                build_split_bore_flange_mesh(
                    edge_len,
                    r_inner,
                    r_outer,
                    gap,
                    half_length,
                    shield=True,
                    shield_edge_len=edge_len2,
                    shield_r_inner=r_inner2,
                    shield_r_outer=r_outer2,
                    shield_gap=gap2,
                    shield_half_length=half_length2,
                ),
                config,
            )
        )
    raise ValueError(f"Unknown coil mesh source: {mesh_name!r}.")


def _rotate_generated_mesh(mesh: Mesh, config: CoilGenConfig) -> Mesh:
    angle = float(config.generated_mesh_rotation_angle)
    if abs(angle) < 1e-15:
        return mesh
    axis = np.asarray(config.generated_mesh_rotation_axis, dtype=float)
    if axis.shape != (3,) or np.linalg.norm(axis) < 1e-15:
        raise ValueError("generated_mesh_rotation_axis must contain a nonzero 3-vector.")
    rot = rotation_matrix_by_vector(axis, angle)
    mesh.vertices = mesh.vertices @ rot.T
    return mesh


def combine_meshes(meshes: list[Mesh]) -> Mesh:
    """Combine mesh parts while preserving each part's local vertex order."""

    vertices = []
    faces = []
    vertex_normals = []
    boundaries: list[np.ndarray] = []
    metadata_parts = []
    offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        if mesh.vertex_normals is not None:
            vertex_normals.append(mesh.vertex_normals)
        for boundary in mesh.boundaries:
            boundaries.append(boundary + offset)
        metadata_parts.append(dict(mesh.metadata))
        offset += mesh.n_vertices
    return Mesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        vertex_normals=np.vstack(vertex_normals) if vertex_normals else None,
        boundaries=boundaries,
        metadata={"combined_from": metadata_parts},
    )


def load_optional_target_mesh(file_name: str, config: CoilGenConfig, project_root: Path) -> Mesh | None:
    if file_name == "none":
        return None
    mesh_path = config.resolved_geometry_source_path(project_root) / file_name
    if is_in_memory_mesh_key(file_name):
        return normalize_mesh(get_in_memory_mesh(file_name))
    return normalize_mesh(read_mesh_npz(mesh_path) if mesh_path.suffix.lower() == ".npz" else read_stl(mesh_path))


def read_mesh_npz(path: str | Path) -> Mesh:
    """Read an owned triangular SI mesh without pickle or STL quantization.

    NPZ meshes are already in metres. The STL import scale is deliberately not
    applied to these portable numerical assets.
    """
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"].item()) != "coilgen-mesh-v1":
            raise ValueError("Unsupported CoilGen numerical mesh schema.")
        vertices = np.array(data["vertices_m"], dtype=float, copy=True)
        raw_faces = np.asarray(data["faces"])
        if raw_faces.dtype.kind not in "iu":
            raise ValueError("Mesh faces must contain integer indices.")
        faces = np.array(raw_faces, dtype=np.int64, copy=True)
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        normals = {
            name: np.array(data[name], dtype=float, copy=True)
            for name in ("vertex_normals", "face_normals")
            if name in data
        }
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh vertices must be finite (n, 3) coordinates in metres.")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Mesh faces must be valid zero-based triangular indices.")
    if not isinstance(metadata, dict):
        raise ValueError("Mesh metadata must be a JSON object.")
    for name, values in normals.items():
        count = len(vertices) if name == "vertex_normals" else len(faces)
        if values.shape != (count, 3) or not np.all(np.isfinite(values)):
            raise ValueError(f"Mesh {name} must contain finite ({count}, 3) directions.")
    return Mesh(vertices=vertices, faces=faces, metadata=metadata, **normals)


def load_secondary_target_mesh(
    config: CoilGenConfig,
    project_root: Path,
    *,
    coil_mesh: Mesh | None = None,
    coil_meshes: list[Mesh] | None = None,
) -> Mesh | None:
    if config.secondary_target_mesh_file != "none":
        return load_optional_target_mesh(config.secondary_target_mesh_file, config, project_root)
    if config.create_mesh_topology_field_suppression_surface == "true":
        return normalize_mesh(
            build_mesh_topology_suppression_surface(
                coil_mesh=coil_mesh,
                coil_meshes=coil_meshes,
                separation=float(config.mesh_topology_suppression_separation),
                source=str(config.mesh_topology_suppression_source),
                config=config,
                project_root=project_root,
            )
        )
    if config.create_split_bore_field_suppression_surface == "true":
        return normalize_mesh(
            _rotate_generated_mesh(create_split_cylinder_mesh(*config.split_bore_area_suppression_parameters), config)
        )
    if config.create_flanged_split_bore_field_suppression_surface == "true":
        return normalize_mesh(
            _rotate_generated_mesh(
                build_split_bore_flange_mesh(*config.flanged_split_bore_area_suppression_parameters),
                config,
            )
        )
    if config.create_circular_biplanar_field_suppression_surface == "true":
        values = tuple(config.circular_biplanar_area_suppression_parameters)
        if len(values) == 3:
            mesh = build_two_coaxial_discs_mesh(*values)
        elif len(values) == 9:
            mesh = build_circular_biplanar_mesh(*values)
        else:
            raise ValueError(
                "circular_biplanar_area_suppression_parameters must contain either "
                "[edge_len radius spacing] or [edge_len radius normal_xyz center_xyz spacing]."
            )
        return normalize_mesh(
            _rotate_generated_mesh(
                mesh,
                config,
            )
        )
    return None


def build_mesh_topology_suppression_surface(
    *,
    separation: float,
    source: str = "shield",
    coil_mesh: Mesh | None = None,
    coil_meshes: list[Mesh] | None = None,
    config: CoilGenConfig | None = None,
    project_root: Path | None = None,
) -> Mesh:
    """Build a zero-field target surface from coil/shield mesh topology.

    ``source="shield"`` is the active-shield mode. The input parts are assumed
    to be ordered as primary parts followed by matching shield parts. The
    suppression vertices are created from the shield vertices and offset by the
    primary-to-shield direction, so the surface stays a fixed distance outside
    both primary and shield with identical topology.

    ``source="all"`` offsets all available coil-surface parts along outward
    vertex normals. This is useful for unshielded reference calculations that
    need zero-weight fringe-field sample points at a fixed distance.
    """

    separation = float(separation)
    if not np.isfinite(separation) or separation <= 0.0:
        raise ValueError("mesh_topology_suppression_separation must be positive.")
    parts = _suppression_source_parts(
        coil_mesh=coil_mesh, coil_meshes=coil_meshes, config=config, project_root=project_root
    )
    mode = source.strip().lower()
    min_clearance = float(getattr(config, "mesh_topology_suppression_min_clearance", 0.0))
    clearance_action = str(getattr(config, "mesh_topology_suppression_clearance_action", "none"))
    if mode == "shield":
        return _suppression_surface_from_shield_parts(
            parts,
            separation,
            min_clearance=min_clearance,
            clearance_action=clearance_action,
        )
    if mode in {"shield_radial", "shield_radial_interface", "split_bore_radial_interface"}:
        return _suppression_surface_from_shield_parts(
            parts,
            separation,
            face_mode="radial",
            split_axis=_rotated_split_axis(config),
            min_clearance=min_clearance,
            clearance_action=clearance_action,
        )
    if mode in {"shield_split", "shield_split_interface", "split_bore_split_interface"}:
        return _suppression_surface_from_shield_parts(
            parts,
            separation,
            face_mode="split",
            split_axis=_rotated_split_axis(config),
            min_clearance=min_clearance,
            clearance_action=clearance_action,
        )
    if mode in {"shield_radial_split", "shield_radial_and_split", "split_bore_magnet_interface"}:
        return _suppression_surface_from_shield_parts(
            parts,
            separation,
            face_mode="radial_split",
            split_axis=_rotated_split_axis(config),
            min_clearance=min_clearance,
            clearance_action=clearance_action,
        )
    if mode == "all":
        return _suppression_surface_from_all_parts(
            parts,
            separation,
            min_clearance=min_clearance,
            clearance_action=clearance_action,
        )
    if mode in {"all_radial", "all_radial_interface"}:
        return _suppression_surface_from_all_parts(
            parts,
            separation,
            face_mode="radial",
            split_axis=_rotated_split_axis(config),
            min_clearance=min_clearance,
            clearance_action=clearance_action,
        )
    if mode in {"all_split", "all_split_interface"}:
        return _suppression_surface_from_all_parts(
            parts,
            separation,
            face_mode="split",
            split_axis=_rotated_split_axis(config),
            min_clearance=min_clearance,
            clearance_action=clearance_action,
        )
    if mode in {"all_radial_split", "all_radial_and_split"}:
        return _suppression_surface_from_all_parts(
            parts,
            separation,
            face_mode="radial_split",
            split_axis=_rotated_split_axis(config),
            min_clearance=min_clearance,
            clearance_action=clearance_action,
        )
    raise ValueError(
        "mesh_topology_suppression_source must be 'shield', 'all', "
        "'shield_radial', 'shield_split', 'shield_radial_split', "
        "'all_radial', 'all_split', or 'all_radial_split'."
    )


def _suppression_source_parts(
    *,
    coil_mesh: Mesh | None,
    coil_meshes: list[Mesh] | None,
    config: CoilGenConfig | None,
    project_root: Path | None,
) -> list[Mesh]:
    if coil_meshes is not None:
        parts = list(coil_meshes)
    else:
        if coil_mesh is None:
            if config is None:
                raise ValueError("A coil mesh, coil mesh parts, or config is required for topology suppression.")
            coil_mesh = load_mesh_from_config(config, Path.cwd() if project_root is None else project_root)
        parts = split_disconnected_mesh(coil_mesh)
    if not parts:
        raise ValueError("Cannot derive a suppression surface from an empty coil mesh.")
    return [normalize_mesh(part) for part in parts]


def _suppression_surface_from_shield_parts(
    parts: list[Mesh],
    separation: float,
    *,
    face_mode: str = "all",
    split_axis: np.ndarray | None = None,
    min_clearance: float = 0.0,
    clearance_action: str = "none",
) -> Mesh:
    if len(parts) < 2 or len(parts) % 2 != 0:
        raise ValueError(
            "Topology-derived shield suppression requires primary parts followed by matching shield parts."
        )
    half = len(parts) // 2
    suppression_parts: list[Mesh] = []
    for index, (primary, shield) in enumerate(zip(parts[:half], parts[half:])):
        if primary.n_vertices != shield.n_vertices or primary.n_faces != shield.n_faces:
            raise ValueError(
                "Primary and shield parts must have matching topology for topology-derived suppression "
                f"(pair {index + 1}: {primary.n_vertices}/{primary.n_faces} vs "
                f"{shield.n_vertices}/{shield.n_faces})."
            )
        directions = shield.vertices - primary.vertices
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths <= 1.0e-14):
            raise ValueError("Primary-to-shield separation contains coincident vertices.")
        directions = directions / lengths[:, None]
        faces = _split_bore_interface_faces(primary, shield.faces, face_mode=face_mode, split_axis=split_axis)
        suppression_parts.append(_mesh_from_selected_faces(shield.vertices + separation * directions, faces))
    candidate = merge_raw_meshes(suppression_parts)
    return apply_suppression_clearance(
        candidate,
        parts[half:],
        requested_clearance_m=separation,
        min_clearance_m=min_clearance,
        action=clearance_action,
        generator=f"paired_shield:{face_mode}",
    )


def _suppression_surface_from_all_parts(
    parts: list[Mesh],
    separation: float,
    *,
    face_mode: str = "all",
    split_axis: np.ndarray | None = None,
    min_clearance: float = 0.0,
    clearance_action: str = "none",
) -> Mesh:
    center = np.vstack([part.vertices for part in parts]).mean(axis=0)
    suppression_parts = []
    for part in parts:
        normalized = normalize_mesh(part)
        if face_mode == "all":
            normals = normalized.vertex_normals
            if normals is None:
                normals = compute_vertex_normals(normalized)
            directions = np.asarray(normals, dtype=float).copy()
            radial = normalized.vertices - center[None, :]
            flip = np.einsum("ij,ij->i", directions, radial) < 0.0
            directions[flip] *= -1.0
        else:
            directions = _split_bore_interface_vertex_directions(
                normalized.vertices, split_axis=split_axis, center=center
            )
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths <= 1.0e-14):
            raise ValueError("Cannot derive outward directions for topology suppression from mesh normals.")
        directions = directions / lengths[:, None]
        faces = _split_bore_interface_faces(normalized, normalized.faces, face_mode=face_mode, split_axis=split_axis)
        suppression_parts.append(_mesh_from_selected_faces(normalized.vertices + separation * directions, faces))
    candidate = merge_raw_meshes(suppression_parts)
    return apply_suppression_clearance(
        candidate,
        parts,
        requested_clearance_m=separation,
        min_clearance_m=min_clearance,
        action=clearance_action,
        generator=f"all_parts:{face_mode}",
    )


def _rotated_split_axis(config: CoilGenConfig | None) -> np.ndarray:
    axis = np.array([0.0, 0.0, 1.0], dtype=float)
    if config is None:
        return axis
    angle = float(config.generated_mesh_rotation_angle)
    if abs(angle) <= 1.0e-15:
        return axis
    rotation_axis = np.asarray(config.generated_mesh_rotation_axis, dtype=float)
    if np.linalg.norm(rotation_axis) <= 1.0e-15:
        return axis
    return rotation_matrix_by_vector(rotation_axis, angle) @ axis


def _split_bore_interface_faces(
    mesh: Mesh,
    faces: np.ndarray,
    *,
    face_mode: str,
    split_axis: np.ndarray | None,
) -> np.ndarray:
    mode = str(face_mode).strip().lower()
    faces = np.asarray(faces, dtype=np.int64)
    if mode == "all":
        return faces.copy()
    if len(faces) == 0:
        return faces.copy()
    axis = _normalized_axis(split_axis)
    face_mesh = Mesh(vertices=mesh.vertices, faces=faces)
    normals = compute_face_normals(face_mesh)
    alignment = np.abs(normals @ axis)
    radial_faces = alignment <= 0.45
    split_faces = alignment >= 0.55
    if mode == "radial":
        selected = faces[radial_faces]
    elif mode == "split":
        selected = faces[split_faces]
    elif mode == "radial_split":
        selected = faces[radial_faces | split_faces]
    else:
        raise ValueError(f"Unsupported split-bore suppression face mode: {face_mode!r}")
    if len(selected) == 0:
        raise ValueError(f"Split-bore suppression face mode {face_mode!r} selected no faces.")
    return selected


def _mesh_from_selected_faces(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("Cannot create a suppression surface from zero selected faces.")
    used_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return Mesh(vertices=np.asarray(vertices, dtype=float)[used_vertices], faces=inverse.reshape(faces.shape))


def _split_bore_interface_vertex_directions(
    vertices: np.ndarray,
    *,
    split_axis: np.ndarray | None,
    center: np.ndarray,
) -> np.ndarray:
    axis = _normalized_axis(split_axis)
    relative = np.asarray(vertices, dtype=float) - np.asarray(center, dtype=float)[None, :]
    axial = relative @ axis
    radial = relative - axial[:, None] * axis[None, :]
    radial_lengths = np.linalg.norm(radial, axis=1)
    directions = np.zeros_like(relative)
    radial_mask = radial_lengths > 1.0e-14
    radial_unit = np.zeros_like(relative)
    radial_unit[radial_mask] = radial[radial_mask] / radial_lengths[radial_mask, None]
    if np.any(radial_mask):
        inner_radius = float(np.min(radial_lengths[radial_mask]))
        radial_scale = max(float(np.max(radial_lengths[radial_mask])), inner_radius, 1.0)
        cylinder_mask = np.abs(radial_lengths - inner_radius) <= 1.0e-8 * radial_scale
        directions[cylinder_mask] += radial_unit[cylinder_mask]
    if np.any(np.abs(axial) > 1.0e-14):
        split_abs = float(np.min(np.abs(axial[np.abs(axial) > 1.0e-14])))
        scale = max(float(np.max(np.abs(axial))), 1.0)
        split_mask = np.abs(np.abs(axial) - split_abs) <= 1.0e-8 * scale
        signs = np.sign(axial[split_mask])
        signs[signs == 0.0] = 1.0
        directions[split_mask] += signs[:, None] * axis[None, :]
    missing = np.linalg.norm(directions, axis=1) <= 1.0e-14
    directions[missing] = radial_unit[missing]
    still_missing = np.linalg.norm(directions, axis=1) <= 1.0e-14
    directions[still_missing] = axis
    return directions


def _normalized_axis(axis: np.ndarray | None) -> np.ndarray:
    if axis is None:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.asarray(axis, dtype=float)
    length = np.linalg.norm(axis)
    if length <= 1.0e-15:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return axis / length


def read_stl(path: Path) -> Mesh:
    data = Path(path).read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL file is too short: {path}")
    face_count = struct.unpack("<I", data[80:84])[0]
    expected_binary_size = 84 + face_count * 50
    if expected_binary_size == len(data):
        return _read_binary_stl(data, face_count)
    return _read_ascii_stl(data)


def _read_binary_stl(data: bytes, face_count: int) -> Mesh:
    vertices = np.zeros((face_count * 3, 3), dtype=float)
    faces = np.zeros((face_count, 3), dtype=np.int64)
    offset = 84
    for face_ind in range(face_count):
        chunk = data[offset : offset + 50]
        values = struct.unpack("<12fH", chunk)
        vertices[3 * face_ind : 3 * face_ind + 3] = np.asarray(values[3:12], dtype=float).reshape(3, 3)
        faces[face_ind] = (3 * face_ind, 3 * face_ind + 1, 3 * face_ind + 2)
        offset += 50
    return Mesh(vertices=vertices, faces=faces)


def _read_ascii_stl(data: bytes) -> Mesh:
    text = data.decode("utf-8", errors="ignore")
    raw_vertices: list[list[float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            raw_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(raw_vertices) % 3 != 0:
        raise ValueError("ASCII STL contains an invalid number of vertex lines.")
    vertices = np.asarray(raw_vertices, dtype=float)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return Mesh(vertices=vertices, faces=faces)


def build_cylinder_mesh(
    cylinder_height: float,
    cylinder_radius: float,
    num_circular_divisions: int,
    num_longitudinal_divisions: int,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
) -> Mesh:
    nc = int(num_circular_divisions)
    nl = int(num_longitudinal_divisions)
    phi = np.linspace(0.0, 2.0 * np.pi, nc, endpoint=False)
    z_positions = np.linspace(-cylinder_height / 2.0, cylinder_height / 2.0, nl + 2)[:-1]
    vertices = []
    for z in z_positions:
        for angle in phi:
            vertices.append([math.sin(angle) * cylinder_radius, math.cos(angle) * cylinder_radius, z])
    vertices = np.asarray(vertices, dtype=float)
    vertices -= vertices.mean(axis=0)
    faces = []
    for ring in range(nl):
        base = ring * nc
        next_base = (ring + 1) * nc
        for col in range(nc):
            a = base + col
            b = base + ((col + 1) % nc)
            c = next_base + ((col + 1) % nc)
            d = next_base + col
            faces.append([b, a, c])
            faces.append([c, a, d])
    rot = rotation_matrix_by_vector(
        np.array([rotation_vector_x, rotation_vector_y, rotation_vector_z], dtype=float),
        float(rotation_angle),
    )
    vertices = vertices @ rot.T
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def build_planar_mesh(
    planar_height: float,
    planar_width: float,
    num_lateral_divisions: int,
    num_longitudinal_divisions: int,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> Mesh:
    nx = int(num_lateral_divisions)
    ny = int(num_longitudinal_divisions)
    xs = np.linspace(-planar_width / 2.0, planar_width / 2.0, nx + 1)
    ys = np.linspace(-planar_height / 2.0, planar_height / 2.0, ny + 1)
    vertices = []
    for y in ys:
        for x in xs:
            vertices.append([y, x, 0.0])
    vertices = np.asarray(vertices, dtype=float)
    rot = rotation_matrix_by_vector(
        np.array([rotation_vector_x, rotation_vector_y, rotation_vector_z], dtype=float),
        float(rotation_angle),
    )
    vertices = vertices @ rot.T + np.array([center_position_x, center_position_y, center_position_z])
    faces = []
    for row in range(ny):
        for col in range(nx):
            a = row * (nx + 1) + col
            b = a + 1
            c = (row + 1) * (nx + 1) + col
            d = c + 1
            faces.append([a, d, b])
            faces.append([a, c, d])
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def build_circular_mesh(
    radius: float,
    num_radial_divisions: int,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> Mesh:
    nr = int(num_radial_divisions)
    xs = np.linspace(-radius, radius, nr + 1)
    ys = np.linspace(-radius, radius, nr + 1)
    vertices = []
    old_to_new: dict[int, int] = {}
    old_ind = 0
    for y in ys:
        for x in xs:
            vertex = np.array([y, x, 0.0], dtype=float)
            if np.linalg.norm(vertex) <= radius * 0.99:
                old_to_new[old_ind] = len(vertices)
                vertices.append(vertex)
            old_ind += 1

    faces = []
    for row in range(nr):
        for col in range(nr):
            a = row * (nr + 1) + col
            b = a + 1
            c = (row + 1) * (nr + 1) + col
            d = c + 1
            for face in ([a, d, b], [a, c, d]):
                if all(node in old_to_new for node in face):
                    faces.append([old_to_new[node] for node in face])
    mesh = Mesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64))

    for loop in boundary_loops(mesh):
        for node in np.unique(loop):
            r = np.linalg.norm(mesh.vertices[node, :2])
            if r > 1e-15:
                mesh.vertices[node] *= radius / r

    rot = rotation_matrix_by_vector(
        np.array([rotation_vector_x, rotation_vector_y, rotation_vector_z], dtype=float),
        float(rotation_angle),
    )
    mesh.vertices = mesh.vertices @ rot.T + np.array([center_position_x, center_position_y, center_position_z])
    return mesh


def build_annular_disc_mesh(
    inner_radius: float,
    outer_radius: float,
    num_radial_divisions: int,
    num_circular_divisions: int,
    target_normal_x: float,
    target_normal_y: float,
    target_normal_z: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> Mesh:
    if inner_radius < 0.0 or outer_radius <= inner_radius:
        raise ValueError("Annular disc requires 0 <= inner_radius < outer_radius.")
    nr = max(1, int(num_radial_divisions))
    nt = max(3, int(num_circular_divisions))
    radii = np.linspace(float(inner_radius), float(outer_radius), nr + 1)
    theta = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    vertices = []
    for radius in radii:
        vertices.extend(np.column_stack((radius * cos_theta, radius * sin_theta, np.zeros(nt))).tolist())
    faces = []
    for radial in range(nr):
        base = radial * nt
        next_base = (radial + 1) * nt
        for col in range(nt):
            col2 = (col + 1) % nt
            v00 = base + col
            v10 = base + col2
            v01 = next_base + col
            v11 = next_base + col2
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    normal = np.asarray([target_normal_x, target_normal_y, target_normal_z], dtype=float)
    rot = rotation_from_z_to(normal)
    center = np.asarray([center_position_x, center_position_y, center_position_z], dtype=float)
    vertices_array = np.asarray(vertices, dtype=float) @ rot.T + center
    return _with_metadata(
        Mesh(vertices=vertices_array, faces=np.asarray(faces, dtype=np.int64)),
        "annular_disc",
        {
            "inner_radius_m": float(inner_radius),
            "outer_radius_m": float(outer_radius),
            "num_radial_divisions": int(num_radial_divisions),
            "num_circular_divisions": int(num_circular_divisions),
        },
        symmetry=("rotational",),
    )


def build_partial_cylinder_mesh(
    cylinder_height: float,
    cylinder_radius: float,
    start_angle: float,
    end_angle: float,
    num_circular_divisions: int,
    num_longitudinal_divisions: int,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> Mesh:
    if cylinder_height <= 0.0 or cylinder_radius <= 0.0:
        raise ValueError("Partial cylinder height and radius must be positive.")
    if float(end_angle) <= float(start_angle):
        raise ValueError("Partial cylinder end_angle must exceed start_angle.")
    nc = max(1, int(num_circular_divisions))
    nl = max(1, int(num_longitudinal_divisions))
    phi = np.linspace(float(start_angle), float(end_angle), nc + 1)
    z_positions = np.linspace(-cylinder_height / 2.0, cylinder_height / 2.0, nl + 1)
    vertices = []
    for z in z_positions:
        for angle in phi:
            vertices.append([math.cos(angle) * cylinder_radius, math.sin(angle) * cylinder_radius, z])
    faces = []
    cols = nc + 1
    for row in range(nl):
        base = row * cols
        next_base = (row + 1) * cols
        for col in range(nc):
            a = base + col
            b = base + col + 1
            c = next_base + col
            d = next_base + col + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    rot = rotation_matrix_by_vector(
        np.asarray([rotation_vector_x, rotation_vector_y, rotation_vector_z], dtype=float),
        float(rotation_angle),
    )
    center = np.asarray([center_position_x, center_position_y, center_position_z], dtype=float)
    vertices_array = np.asarray(vertices, dtype=float) @ rot.T + center
    return _with_metadata(
        Mesh(vertices=vertices_array, faces=np.asarray(faces, dtype=np.int64)),
        "partial_cylinder",
        {
            "cylinder_height_m": float(cylinder_height),
            "cylinder_radius_m": float(cylinder_radius),
            "start_angle_rad": float(start_angle),
            "end_angle_rad": float(end_angle),
            "num_circular_divisions": int(num_circular_divisions),
            "num_longitudinal_divisions": int(num_longitudinal_divisions),
        },
    )


def build_elliptical_cylinder_mesh(
    cylinder_height: float,
    radius_x: float,
    radius_y: float,
    num_circular_divisions: int,
    num_longitudinal_divisions: int,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> Mesh:
    if cylinder_height <= 0.0 or radius_x <= 0.0 or radius_y <= 0.0:
        raise ValueError("Elliptical cylinder height and radii must be positive.")
    nc = max(3, int(num_circular_divisions))
    nl = max(1, int(num_longitudinal_divisions))
    phi = np.linspace(0.0, 2.0 * np.pi, nc, endpoint=False)
    z_positions = np.linspace(-cylinder_height / 2.0, cylinder_height / 2.0, nl + 1)
    vertices = []
    for z in z_positions:
        for angle in phi:
            vertices.append([math.cos(angle) * radius_x, math.sin(angle) * radius_y, z])
    faces = []
    for row in range(nl):
        base = row * nc
        next_base = (row + 1) * nc
        for col in range(nc):
            a = base + col
            b = base + ((col + 1) % nc)
            c = next_base + col
            d = next_base + ((col + 1) % nc)
            faces.append([a, b, d])
            faces.append([a, d, c])
    rot = rotation_matrix_by_vector(
        np.asarray([rotation_vector_x, rotation_vector_y, rotation_vector_z], dtype=float),
        float(rotation_angle),
    )
    center = np.asarray([center_position_x, center_position_y, center_position_z], dtype=float)
    vertices_array = np.asarray(vertices, dtype=float) @ rot.T + center
    return _with_metadata(
        Mesh(vertices=vertices_array, faces=np.asarray(faces, dtype=np.int64)),
        "elliptical_cylinder",
        {
            "cylinder_height_m": float(cylinder_height),
            "radius_x_m": float(radius_x),
            "radius_y_m": float(radius_y),
            "num_circular_divisions": int(num_circular_divisions),
            "num_longitudinal_divisions": int(num_longitudinal_divisions),
        },
        symmetry=("mirror_xz", "mirror_yz"),
    )


def build_superellipse_cylinder_mesh(
    cylinder_height: float,
    radius_x: float,
    radius_y: float,
    exponent: float,
    num_circular_divisions: int,
    num_longitudinal_divisions: int,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> Mesh:
    if cylinder_height <= 0.0 or radius_x <= 0.0 or radius_y <= 0.0:
        raise ValueError("Superellipse cylinder height and radii must be positive.")
    if exponent <= 0.5:
        raise ValueError("Superellipse exponent must be > 0.5.")
    nc = max(8, int(num_circular_divisions))
    nl = max(1, int(num_longitudinal_divisions))
    phi = np.linspace(0.0, 2.0 * np.pi, nc, endpoint=False)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    power = 2.0 / float(exponent)
    x_profile = radius_x * np.sign(cos_phi) * np.abs(cos_phi) ** power
    y_profile = radius_y * np.sign(sin_phi) * np.abs(sin_phi) ** power
    z_positions = np.linspace(-cylinder_height / 2.0, cylinder_height / 2.0, nl + 1)
    vertices = []
    for z in z_positions:
        vertices.extend(np.column_stack((x_profile, y_profile, np.full(nc, z))).tolist())
    faces = _grid_faces(nl + 1, nc, wrap_cols=True, outward=True)
    vertices_array = _apply_rotation_center(
        np.asarray(vertices, dtype=float),
        rotation_vector_x,
        rotation_vector_y,
        rotation_vector_z,
        rotation_angle,
        center_position_x,
        center_position_y,
        center_position_z,
    )
    return _with_metadata(
        Mesh(vertices=vertices_array, faces=faces),
        "superellipse_cylinder",
        {
            "cylinder_height_m": float(cylinder_height),
            "radius_x_m": float(radius_x),
            "radius_y_m": float(radius_y),
            "exponent": float(exponent),
            "num_circular_divisions": int(num_circular_divisions),
            "num_longitudinal_divisions": int(num_longitudinal_divisions),
        },
        symmetry=("mirror_xz", "mirror_yz"),
    )


def build_conical_frustum_mesh(
    frustum_height: float,
    radius_bottom: float,
    radius_top: float,
    num_circular_divisions: int,
    num_longitudinal_divisions: int,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> Mesh:
    if frustum_height <= 0.0 or radius_bottom <= 0.0 or radius_top <= 0.0:
        raise ValueError("Conical frustum height and radii must be positive.")
    nc = max(3, int(num_circular_divisions))
    nl = max(1, int(num_longitudinal_divisions))
    phi = np.linspace(0.0, 2.0 * np.pi, nc, endpoint=False)
    z_positions = np.linspace(-frustum_height / 2.0, frustum_height / 2.0, nl + 1)
    radii = np.linspace(float(radius_bottom), float(radius_top), nl + 1)
    vertices = []
    for z, radius in zip(z_positions, radii):
        vertices.extend(np.column_stack((radius * np.cos(phi), radius * np.sin(phi), np.full(nc, z))).tolist())
    faces = _grid_faces(nl + 1, nc, wrap_cols=True, outward=True)
    vertices_array = _apply_rotation_center(
        np.asarray(vertices, dtype=float),
        rotation_vector_x,
        rotation_vector_y,
        rotation_vector_z,
        rotation_angle,
        center_position_x,
        center_position_y,
        center_position_z,
    )
    return _with_metadata(
        Mesh(vertices=vertices_array, faces=faces),
        "conical_frustum",
        {
            "frustum_height_m": float(frustum_height),
            "radius_bottom_m": float(radius_bottom),
            "radius_top_m": float(radius_top),
            "num_circular_divisions": int(num_circular_divisions),
            "num_longitudinal_divisions": int(num_longitudinal_divisions),
        },
        symmetry=("rotational",),
    )


def build_ellipsoidal_cap_mesh(
    radius_x: float,
    radius_y: float,
    radius_z: float,
    cap_polar_angle: float,
    num_azimuthal_divisions: int,
    num_polar_divisions: int,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> Mesh:
    if radius_x <= 0.0 or radius_y <= 0.0 or radius_z <= 0.0:
        raise ValueError("Ellipsoidal cap radii must be positive.")
    if cap_polar_angle <= 0.0 or cap_polar_angle > np.pi:
        raise ValueError("Ellipsoidal cap polar angle must be in (0, pi].")
    na = max(3, int(num_azimuthal_divisions))
    npolar = max(1, int(num_polar_divisions))
    phi = np.linspace(0.0, 2.0 * np.pi, na, endpoint=False)
    theta = np.linspace(float(cap_polar_angle) / npolar, float(cap_polar_angle), npolar)
    vertices = [[0.0, 0.0, radius_z]]
    for polar_angle in theta:
        sin_theta = math.sin(float(polar_angle))
        cos_theta = math.cos(float(polar_angle))
        vertices.extend(
            np.column_stack(
                (
                    radius_x * sin_theta * np.cos(phi),
                    radius_y * sin_theta * np.sin(phi),
                    np.full(na, radius_z * cos_theta),
                )
            ).tolist()
        )
    faces = []
    first_ring = 1
    for col in range(na):
        col2 = (col + 1) % na
        faces.append([0, first_ring + col, first_ring + col2])
    for ring in range(npolar - 1):
        base = 1 + ring * na
        next_base = base + na
        for col in range(na):
            col2 = (col + 1) % na
            v00 = base + col
            v10 = base + col2
            v01 = next_base + col
            v11 = next_base + col2
            faces.append([v00, v11, v10])
            faces.append([v00, v01, v11])
    vertices_array = _apply_rotation_center(
        np.asarray(vertices, dtype=float),
        rotation_vector_x,
        rotation_vector_y,
        rotation_vector_z,
        rotation_angle,
        center_position_x,
        center_position_y,
        center_position_z,
    )
    return _with_metadata(
        Mesh(vertices=vertices_array, faces=np.asarray(faces, dtype=np.int64)),
        "ellipsoidal_cap",
        {
            "radius_x_m": float(radius_x),
            "radius_y_m": float(radius_y),
            "radius_z_m": float(radius_z),
            "cap_polar_angle_rad": float(cap_polar_angle),
            "num_azimuthal_divisions": int(num_azimuthal_divisions),
            "num_polar_divisions": int(num_polar_divisions),
        },
        symmetry=("mirror_xz", "mirror_yz"),
    )


def build_spherical_cap_mesh(
    sphere_radius: float,
    cap_polar_angle: float,
    num_azimuthal_divisions: int,
    num_polar_divisions: int,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> Mesh:
    mesh = build_ellipsoidal_cap_mesh(
        sphere_radius,
        sphere_radius,
        sphere_radius,
        cap_polar_angle,
        num_azimuthal_divisions,
        num_polar_divisions,
        rotation_vector_x,
        rotation_vector_y,
        rotation_vector_z,
        rotation_angle,
        center_position_x,
        center_position_y,
        center_position_z,
    )
    mesh.metadata["geometry_type"] = "spherical_cap"
    mesh.metadata["parameters"] = {
        "sphere_radius_m": float(sphere_radius),
        "cap_polar_angle_rad": float(cap_polar_angle),
        "num_azimuthal_divisions": int(num_azimuthal_divisions),
        "num_polar_divisions": int(num_polar_divisions),
    }
    mesh.metadata["symmetry"] = ("rotational",)
    return mesh


def build_toroidal_segment_mesh(
    major_radius: float,
    minor_radius: float,
    start_angle: float,
    end_angle: float,
    tube_start_angle: float,
    tube_end_angle: float,
    num_major_divisions: int,
    num_minor_divisions: int,
    rotation_vector_x: float,
    rotation_vector_y: float,
    rotation_vector_z: float,
    rotation_angle: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
) -> Mesh:
    if major_radius <= 0.0 or minor_radius <= 0.0:
        raise ValueError("Toroidal segment radii must be positive.")
    if minor_radius >= major_radius:
        raise ValueError("Toroidal minor radius must be smaller than major radius to avoid self-intersection.")
    if end_angle <= start_angle or tube_end_angle <= tube_start_angle:
        raise ValueError("Toroidal segment end angles must exceed start angles.")
    if end_angle - start_angle > 2.0 * np.pi + 1e-12 or tube_end_angle - tube_start_angle > 2.0 * np.pi + 1e-12:
        raise ValueError("Toroidal angular spans cannot exceed 2*pi.")
    nu = max(1, int(num_major_divisions))
    nv = max(3, int(num_minor_divisions))
    wrap_v = abs((tube_end_angle - tube_start_angle) - 2.0 * np.pi) < 1e-12
    u_values = np.linspace(float(start_angle), float(end_angle), nu + 1)
    v_values = (
        np.linspace(float(tube_start_angle), float(tube_end_angle), nv, endpoint=False)
        if wrap_v
        else np.linspace(float(tube_start_angle), float(tube_end_angle), nv + 1)
    )
    vertices = []
    for u in u_values:
        cos_u = math.cos(float(u))
        sin_u = math.sin(float(u))
        for v in v_values:
            radial = major_radius + minor_radius * math.cos(float(v))
            vertices.append([radial * cos_u, radial * sin_u, minor_radius * math.sin(float(v))])
    faces = _grid_faces(len(u_values), len(v_values), wrap_cols=wrap_v, outward=False)
    vertices_array = _apply_rotation_center(
        np.asarray(vertices, dtype=float),
        rotation_vector_x,
        rotation_vector_y,
        rotation_vector_z,
        rotation_angle,
        center_position_x,
        center_position_y,
        center_position_z,
    )
    return _with_metadata(
        Mesh(vertices=vertices_array, faces=faces),
        "toroidal_segment",
        {
            "major_radius_m": float(major_radius),
            "minor_radius_m": float(minor_radius),
            "start_angle_rad": float(start_angle),
            "end_angle_rad": float(end_angle),
            "tube_start_angle_rad": float(tube_start_angle),
            "tube_end_angle_rad": float(tube_end_angle),
            "num_major_divisions": int(num_major_divisions),
            "num_minor_divisions": int(num_minor_divisions),
        },
    )


def build_ruled_surface_mesh(
    num_curve_samples: float,
    num_span_divisions: float,
    *curve_points: float,
) -> Mesh:
    """Build a ruled surface between two sampled 3D boundary curves.

    Parameter format is ``n_samples, n_span, curve_a_xyz..., curve_b_xyz...``.
    Both curves must contain ``n_samples`` points in metres.
    """

    n = int(num_curve_samples)
    span = int(num_span_divisions)
    if n < 2 or span < 1:
        raise ValueError("Ruled surface requires at least two curve samples and one span division.")
    expected = 6 * n
    if len(curve_points) != expected:
        raise ValueError(f"Ruled surface expected {expected} curve coordinate values, got {len(curve_points)}.")
    coords = np.asarray(curve_points, dtype=float)
    if not np.all(np.isfinite(coords)):
        raise ValueError("Ruled surface coordinates must be finite.")
    curve_a = coords[: 3 * n].reshape(n, 3)
    curve_b = coords[3 * n :].reshape(n, 3)
    vertices = []
    for i in range(n):
        for j in range(span + 1):
            t = j / span
            vertices.append((1.0 - t) * curve_a[i] + t * curve_b[i])
    faces = _grid_faces(n, span + 1, wrap_cols=False, outward=True)
    return _with_metadata(
        Mesh(vertices=np.asarray(vertices, dtype=float), faces=faces),
        "ruled_surface",
        {
            "num_curve_samples": n,
            "num_span_divisions": span,
            "curve_a_start_m": tuple(float(v) for v in curve_a[0]),
            "curve_a_end_m": tuple(float(v) for v in curve_a[-1]),
            "curve_b_start_m": tuple(float(v) for v in curve_b[0]),
            "curve_b_end_m": tuple(float(v) for v in curve_b[-1]),
        },
    )


def build_surface_of_revolution_mesh(
    num_circular_divisions: float,
    *z_radius_pairs: float,
) -> Mesh:
    """Build an open surface of revolution from flattened ``z, radius`` pairs."""

    if len(z_radius_pairs) < 4 or len(z_radius_pairs) % 2 != 0:
        raise ValueError("Surface of revolution requires at least two flattened z,radius pairs.")
    nt = max(3, int(num_circular_divisions))
    pairs = np.asarray(z_radius_pairs, dtype=float).reshape(-1, 2)
    z_values = pairs[:, 0]
    radii = pairs[:, 1]
    if np.any(~np.isfinite(pairs)):
        raise ValueError("Surface of revolution parameters must be finite.")
    if np.any(radii <= 0.0):
        raise ValueError("Surface of revolution radii must be positive.")
    if np.any(np.diff(z_values) <= 0.0):
        raise ValueError("Surface of revolution z values must be strictly increasing.")
    theta = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    vertices = []
    for z, radius in pairs:
        vertices.extend(np.column_stack((radius * cos_theta, radius * sin_theta, np.full(nt, z))).tolist())
    faces = []
    for row in range(len(pairs) - 1):
        base = row * nt
        next_base = (row + 1) * nt
        for col in range(nt):
            col2 = (col + 1) % nt
            v00 = base + col
            v10 = base + col2
            v01 = next_base + col
            v11 = next_base + col2
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return _with_metadata(
        Mesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64)),
        "surface_of_revolution",
        {
            "num_circular_divisions": int(num_circular_divisions),
            "z_radius_pairs_m": tuple((float(z), float(radius)) for z, radius in pairs),
        },
        symmetry=("rotational",),
    )


def build_biplanar_mesh(
    planar_height: float,
    planar_width: float,
    num_lateral_divisions: int,
    num_longitudinal_divisions: int,
    target_normal_x: float,
    target_normal_y: float,
    target_normal_z: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
    plane_distance: float,
) -> Mesh:
    base = build_planar_mesh(
        planar_height,
        planar_width,
        int(num_lateral_divisions),
        int(num_longitudinal_divisions),
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    normal = np.asarray([target_normal_x, target_normal_y, target_normal_z], dtype=float)
    normal_norm = np.linalg.norm(normal)
    normal = normal / normal_norm if normal_norm > 0 else np.array([0.0, 0.0, 1.0])
    # asin(|z × n|) loses the lower hemisphere and cannot represent -z.
    # Match the full-sphere normal convention of the circular biplanar support.
    rot = rotation_from_z_to(normal)
    center = np.array([center_position_x, center_position_y, center_position_z], dtype=float)
    vertices1 = base.vertices + np.array([0.0, 0.0, plane_distance / 2.0])
    vertices2 = base.vertices + np.array([0.0, 0.0, -plane_distance / 2.0])
    vertices = np.vstack((vertices1 @ rot.T + center, vertices2 @ rot.T + center))
    faces = np.vstack((base.faces, base.faces + len(base.vertices)))
    return Mesh(vertices=vertices, faces=faces)


def build_two_coaxial_discs_mesh(edge_len: float, outer_radius: float, spacing: float) -> Mesh:
    divisions = max(4, int(math.ceil((2.0 * outer_radius) / edge_len)))
    base = build_circular_mesh(outer_radius, divisions, 1, 0, 0, 0, 0, 0, 0)
    vertices1 = base.vertices + np.array([0.0, 0.0, -spacing / 2.0])
    vertices2 = base.vertices + np.array([0.0, 0.0, spacing / 2.0])
    faces1 = base.faces
    faces2 = base.faces[:, [0, 2, 1]] + len(vertices1)
    return Mesh(vertices=np.vstack((vertices1, vertices2)), faces=np.vstack((faces1, faces2)))


def build_circular_biplanar_mesh(
    edge_len: float,
    outer_radius: float,
    target_normal_x: float,
    target_normal_y: float,
    target_normal_z: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
    plane_distance: float,
) -> Mesh:
    mesh = build_two_coaxial_discs_mesh(edge_len, outer_radius, plane_distance)
    normal = np.asarray([target_normal_x, target_normal_y, target_normal_z], dtype=float)
    rot = rotation_from_z_to(normal)
    center = np.asarray([center_position_x, center_position_y, center_position_z], dtype=float)
    mesh.vertices = mesh.vertices @ rot.T + center
    return mesh


def build_shielded_biplanar_mesh(
    primary_outer_radius: float,
    primary_edge_len: float,
    shield_outer_radius: float,
    shield_edge_len: float,
    target_normal_x: float,
    target_normal_y: float,
    target_normal_z: float,
    center_position_x: float,
    center_position_y: float,
    center_position_z: float,
    plane_distance: float,
    shield_gap: float,
) -> Mesh:
    # Shield radius and edge length are accepted for compatibility with older
    # configuration files. The generated shield is an offset copy of the
    # primary topology; shield_gap is the physical separation distance.
    primary_divisions = max(4, int(math.ceil((2.0 * primary_outer_radius) / primary_edge_len)))
    primary = build_circular_mesh(primary_outer_radius, primary_divisions, 1, 0, 0, 0, 0, 0, 0)
    normal = np.array([target_normal_x, target_normal_y, target_normal_z], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0 or not np.isfinite(normal_norm):
        raise ValueError("target normal must be nonzero.")
    normal_unit = normal / normal_norm
    rot = rotation_from_z_to(normal)
    center = np.array([center_position_x, center_position_y, center_position_z], dtype=float)

    primary_plus = _place_disc(primary, plane_distance / 2.0, rot, center, normal_plus=False)
    primary_minus = _place_disc(primary, -plane_distance / 2.0, rot, center, normal_plus=True)
    separation = float(shield_gap)
    if not np.isfinite(separation) or separation <= 0.0:
        raise ValueError("shield_gap must be a positive shield separation.")
    meshes = [
        primary_plus,
        primary_minus,
        _offset_mesh(primary_plus, separation * normal_unit),
        _offset_mesh(primary_minus, -separation * normal_unit),
    ]
    return merge_raw_meshes(meshes)


def _place_disc(mesh: Mesh, z0: float, rot: np.ndarray, center: np.ndarray, normal_plus: bool) -> Mesh:
    vertices = mesh.vertices.copy()
    vertices[:, 2] = z0
    vertices = vertices @ rot.T + center
    faces = mesh.faces if normal_plus else mesh.faces[:, [0, 2, 1]]
    return Mesh(vertices=vertices, faces=faces.copy())


def _offset_mesh(mesh: Mesh, offset: np.ndarray) -> Mesh:
    offset = np.asarray(offset, dtype=float)
    if offset.shape == (3,):
        vertices = mesh.vertices + offset[None, :]
    else:
        vertices = mesh.vertices + offset
    return Mesh(vertices=vertices, faces=mesh.faces.copy())


def create_split_cylinder_mesh(edge_len: float, r_inner: float, gap: float, half_length: float) -> Mesh:
    if edge_len <= 0 or r_inner <= 0 or gap < 0 or half_length <= 0:
        raise ValueError("Invalid split-cylinder parameters.")
    n_theta = max(24, int(math.ceil((2.0 * math.pi * r_inner) / edge_len)))
    n_z = max(1, int(math.ceil(half_length / edge_len)))

    def build_half(z0: float, z1: float) -> Mesh:
        theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
        z = np.linspace(min(z0, z1), max(z0, z1), n_z + 1)
        vertices = []
        for zz in z:
            for th in theta:
                vertices.append([r_inner * math.cos(th), r_inner * math.sin(th), zz])
        faces = []
        for row in range(n_z):
            for col in range(n_theta):
                a = row * n_theta + col
                b = row * n_theta + ((col + 1) % n_theta)
                c = (row + 1) * n_theta + col
                d = (row + 1) * n_theta + ((col + 1) % n_theta)
                faces.append([a, d, b])
                faces.append([a, c, d])
        return Mesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64))

    negative = build_half(-gap / 2.0 - half_length, -gap / 2.0)
    positive = build_half(gap / 2.0, gap / 2.0 + half_length)
    return merge_raw_meshes([negative, positive])


def build_split_bore_flange_mesh(
    edge_len: float,
    r_inner: float,
    r_outer: float,
    gap: float,
    half_length: float,
    *,
    shield: bool = False,
    shield_edge_len: float | None = None,
    shield_r_inner: float | None = None,
    shield_r_outer: float | None = None,
    shield_gap: float | None = None,
    shield_half_length: float | None = None,
) -> Mesh:
    primary = _build_one_split_bore_flange(edge_len, r_inner, r_outer, gap, half_length)
    if not shield:
        return primary
    # Legacy shield shape parameters are accepted but do not create an
    # independently meshed shield. The shield is an outward offset copy of the
    # primary topology; shield_r_inner - r_outer is the separation distance.
    if shield_r_inner is None:
        raise ValueError("shield_r_inner is required when shield=True.")
    separation = float(shield_r_inner) - float(r_outer)
    if not np.isfinite(separation) or separation <= 0.0:
        raise ValueError("shield_r_inner must exceed r_outer; the difference is the shield separation.")
    shield_mesh = _offset_split_bore_flange_mesh(primary, separation, r_inner, gap)
    return merge_raw_meshes([primary, shield_mesh])


def _offset_split_bore_flange_mesh(primary: Mesh, separation: float, r_inner: float, gap: float) -> Mesh:
    vertices = np.asarray(primary.vertices, dtype=float)
    directions = np.zeros_like(vertices)
    radial = np.linalg.norm(vertices[:, :2], axis=1)
    radial_unit = np.zeros_like(vertices)
    radial_mask = radial > 1.0e-15
    radial_unit[radial_mask, 0] = vertices[radial_mask, 0] / radial[radial_mask]
    radial_unit[radial_mask, 1] = vertices[radial_mask, 1] / radial[radial_mask]

    scale = max(float(r_inner), float(np.max(radial)) if radial.size else 0.0, abs(float(gap)), 1.0)
    tolerance = 1.0e-8 * scale
    cylinder_mask = np.abs(radial - float(r_inner)) <= tolerance
    directions[cylinder_mask] += radial_unit[cylinder_mask]

    split_mask = np.abs(np.abs(vertices[:, 2]) - float(gap) / 2.0) <= tolerance
    directions[split_mask, 2] += np.sign(vertices[split_mask, 2])

    missing = np.linalg.norm(directions, axis=1) <= 1.0e-15
    directions[missing] = radial_unit[missing]
    lengths = np.linalg.norm(directions, axis=1)
    valid = lengths > 1.0e-15
    directions[valid] /= lengths[valid, None]
    return Mesh(vertices=vertices + float(separation) * directions, faces=primary.faces.copy())


def _build_one_split_bore_flange(
    edge_len: float, r_inner: float, r_outer: float, gap: float, half_length: float
) -> Mesh:
    if edge_len <= 0 or r_outer <= r_inner or gap < 0 or half_length <= 0:
        raise ValueError("Invalid split-bore flange parameters.")
    n_theta = max(24, int(math.ceil((2.0 * math.pi * r_outer) / edge_len)))
    n_z = max(1, int(math.ceil(half_length / edge_len)))
    n_r = max(1, int(math.ceil((r_outer - r_inner) / edge_len)))
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    def ring(radius: float, z: float) -> np.ndarray:
        return np.column_stack((radius * cos_theta, radius * sin_theta, np.full(n_theta, z)))

    def build_half(z_split: float, z_far: float) -> Mesh:
        z_levels = np.linspace(min(z_split, z_far), max(z_split, z_far), n_z + 1)
        split_row = n_z if z_far < z_split else 0
        vertices = [ring(r_inner, z) for z in z_levels]
        inner_idx = np.arange((n_z + 1) * n_theta, dtype=np.int64).reshape(n_z + 1, n_theta)
        faces: list[list[int]] = []
        faces.extend(_cylinder_faces(inner_idx, outward=False).tolist())

        rings = [inner_idx[split_row]]
        for rr in np.linspace(r_inner, r_outer, n_r + 1)[1:]:
            base = sum(len(block) for block in vertices)
            vertices.append(ring(rr, z_split))
            rings.append(np.arange(base, base + n_theta, dtype=np.int64))
        normal_plus_z = z_split < 0
        faces.extend(_annulus_faces(np.vstack(rings), normal_plus_z=normal_plus_z).tolist())
        return Mesh(vertices=np.vstack(vertices), faces=np.asarray(faces, dtype=np.int64))

    negative = build_half(-gap / 2.0, -gap / 2.0 - half_length)
    positive = build_half(gap / 2.0, gap / 2.0 + half_length)
    return merge_raw_meshes([negative, positive])


def _cylinder_faces(ring_idx: np.ndarray, outward: bool) -> np.ndarray:
    n_rows, n_theta = ring_idx.shape
    faces = []
    for row in range(n_rows - 1):
        for col in range(n_theta):
            col2 = (col + 1) % n_theta
            v00 = ring_idx[row, col]
            v10 = ring_idx[row, col2]
            v01 = ring_idx[row + 1, col]
            v11 = ring_idx[row + 1, col2]
            if (row + col) % 2 == 0:
                triangles = [[v00, v10, v11], [v00, v11, v01]]
            else:
                triangles = [[v00, v10, v01], [v10, v11, v01]]
            if not outward:
                triangles = [[a, c, b] for a, b, c in triangles]
            faces.extend(triangles)
    return np.asarray(faces, dtype=np.int64)


def _annulus_faces(rings: np.ndarray, normal_plus_z: bool) -> np.ndarray:
    n_rings, n_theta = rings.shape
    faces = []
    for radial in range(n_rings - 1):
        for col in range(n_theta):
            col2 = (col + 1) % n_theta
            v00 = rings[radial, col]
            v10 = rings[radial, col2]
            v01 = rings[radial + 1, col]
            v11 = rings[radial + 1, col2]
            if (radial + col) % 2 == 0:
                triangles = [[v00, v10, v11], [v00, v11, v01]]
            else:
                triangles = [[v00, v10, v01], [v10, v11, v01]]
            if not normal_plus_z:
                triangles = [[a, c, b] for a, b, c in triangles]
            faces.extend(triangles)
    return np.asarray(faces, dtype=np.int64)


def merge_raw_meshes(meshes: list[Mesh]) -> Mesh:
    vertices = []
    faces = []
    offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += mesh.n_vertices
    return Mesh(vertices=np.vstack(vertices), faces=np.vstack(faces))


def rotation_from_z_to(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm <= 0:
        raise ValueError("target normal must be nonzero.")
    normal = normal / norm
    ez = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(ez, normal), -1.0, 1.0))
    if abs(dot - 1.0) < 1e-14:
        return np.eye(3)
    if abs(dot + 1.0) < 1e-14:
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    axis = np.cross(ez, normal)
    angle = math.acos(dot)
    return rotation_matrix_by_vector(axis, angle)


def split_disconnected_mesh(mesh: Mesh) -> list[Mesh]:
    vertex_to_faces: dict[int, list[int]] = defaultdict(list)
    for face_ind, face in enumerate(mesh.faces):
        for vertex in face:
            vertex_to_faces[int(vertex)].append(face_ind)
    visited = np.zeros(mesh.n_faces, dtype=bool)
    parts: list[Mesh] = []
    for start_face in range(mesh.n_faces):
        if visited[start_face]:
            continue
        queue: deque[int] = deque([start_face])
        visited[start_face] = True
        component_faces: list[int] = []
        while queue:
            face_ind = queue.popleft()
            component_faces.append(face_ind)
            for vertex in mesh.faces[face_ind]:
                for neighbor_face in vertex_to_faces[int(vertex)]:
                    if not visited[neighbor_face]:
                        visited[neighbor_face] = True
                        queue.append(neighbor_face)
        old_faces = mesh.faces[component_faces]
        unique_vertices, inverse = np.unique(old_faces.reshape(-1), return_inverse=True)
        part_faces = inverse.reshape(old_faces.shape)
        part = Mesh(vertices=mesh.vertices[unique_vertices], faces=part_faces, unique_vert_inds=unique_vertices)
        parts.append(normalize_mesh(part))
    return parts


def refine_mesh(mesh: Mesh, iterations: int) -> Mesh:
    refined = mesh.copy()
    for _ in range(max(0, int(iterations))):
        refined = _subdivide_once(refined)
    return normalize_mesh(refined)


def _subdivide_once(mesh: Mesh) -> Mesh:
    vertices = [vertex.copy() for vertex in mesh.vertices]
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint_index(a: int, b: int) -> int:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if key not in midpoint_cache:
            midpoint_cache[key] = len(vertices)
            vertices.append((mesh.vertices[key[0]] + mesh.vertices[key[1]]) / 2.0)
        return midpoint_cache[key]

    faces = []
    for a, b, c in mesh.faces:
        ab = midpoint_index(a, b)
        bc = midpoint_index(b, c)
        ca = midpoint_index(c, a)
        faces.extend(
            [
                [a, ab, ca],
                [ab, b, bc],
                [ca, bc, c],
                [ab, bc, ca],
            ]
        )
    return Mesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64))


def _build_shielded_split_bore_from_config(config: CoilGenConfig) -> Mesh:
    values = list(config.shielded_flanged_split_bore_mesh_parameter_list)
    if len(values) not in (6, 10):
        raise ValueError(
            "shielded_flanged_split_bore_mesh_parameter_list must contain "
            "either [edge_len r_inner r_outer gap half_length shield_separation] "
            "or the legacy [edge_len r_inner r_outer gap half_length shield_edge_len "
            "shield_r_inner shield_r_outer shield_gap shield_half_length]."
        )
    if len(values) == 6:
        edge_len, r_inner, r_outer, gap, half_length, shield_separation = values
        edge_len2 = edge_len
        r_inner2 = float(r_outer) + float(shield_separation)
        r_outer2 = np.nan
        gap2 = gap
        half_length2 = half_length
    else:
        edge_len, r_inner, r_outer, gap, half_length, edge_len2, r_inner2, r_outer2, gap2, half_length2 = values
        edge_len2 = edge_len if np.isnan(edge_len2) else edge_len2
        gap2 = gap if np.isnan(gap2) else gap2
        half_length2 = half_length if np.isnan(half_length2) else half_length2
    return build_split_bore_flange_mesh(
        edge_len,
        r_inner,
        r_outer,
        gap,
        half_length,
        shield=True,
        shield_edge_len=edge_len2,
        shield_r_inner=r_inner2,
        shield_r_outer=r_outer2,
        shield_gap=gap2,
        shield_half_length=half_length2,
    )


def _register_builtin_geometry_factories() -> None:
    factories = (
        GeometryFactory(
            "create cylinder mesh",
            "Open cylindrical surface.",
            "cylinder_mesh_parameter_list",
            lambda config: _rotate_generated_mesh(build_cylinder_mesh(*config.cylinder_mesh_parameter_list), config),
        ),
        GeometryFactory(
            "create planary mesh",
            "Single rectangular planar surface.",
            "planar_mesh_parameter_list",
            lambda config: _rotate_generated_mesh(build_planar_mesh(*config.planar_mesh_parameter_list), config),
        ),
        GeometryFactory(
            "create circular mesh",
            "Single circular planar surface.",
            "circular_mesh_parameter_list",
            lambda config: _rotate_generated_mesh(build_circular_mesh(*config.circular_mesh_parameter_list), config),
        ),
        GeometryFactory(
            "create annular disc mesh",
            "Single annular planar surface.",
            "annular_disc_mesh_parameter_list",
            lambda config: build_annular_disc_mesh(*config.annular_disc_mesh_parameter_list),
        ),
        GeometryFactory(
            "create partial cylinder mesh",
            "Open cylindrical sector surface.",
            "partial_cylinder_mesh_parameter_list",
            lambda config: build_partial_cylinder_mesh(*config.partial_cylinder_mesh_parameter_list),
        ),
        GeometryFactory(
            "create elliptical cylinder mesh",
            "Open elliptical cylindrical surface.",
            "elliptical_cylinder_mesh_parameter_list",
            lambda config: build_elliptical_cylinder_mesh(*config.elliptical_cylinder_mesh_parameter_list),
        ),
        GeometryFactory(
            "create superellipse cylinder mesh",
            "Open rounded-rectangle/superellipse cylindrical surface.",
            "superellipse_cylinder_mesh_parameter_list",
            lambda config: build_superellipse_cylinder_mesh(*config.superellipse_cylinder_mesh_parameter_list),
        ),
        GeometryFactory(
            "create spherical cap mesh",
            "Open spherical cap surface.",
            "spherical_cap_mesh_parameter_list",
            lambda config: build_spherical_cap_mesh(*config.spherical_cap_mesh_parameter_list),
        ),
        GeometryFactory(
            "create ellipsoidal cap mesh",
            "Open ellipsoidal cap surface.",
            "ellipsoidal_cap_mesh_parameter_list",
            lambda config: build_ellipsoidal_cap_mesh(*config.ellipsoidal_cap_mesh_parameter_list),
        ),
        GeometryFactory(
            "create conical frustum mesh",
            "Open tapered cylindrical/conical-frustum surface.",
            "conical_frustum_mesh_parameter_list",
            lambda config: build_conical_frustum_mesh(*config.conical_frustum_mesh_parameter_list),
        ),
        GeometryFactory(
            "create toroidal segment mesh",
            "Open toroidal segment surface.",
            "toroidal_segment_mesh_parameter_list",
            lambda config: build_toroidal_segment_mesh(*config.toroidal_segment_mesh_parameter_list),
        ),
        GeometryFactory(
            "create ruled surface mesh",
            "Ruled surface between two sampled 3D boundary curves.",
            "ruled_surface_mesh_parameter_list",
            lambda config: build_ruled_surface_mesh(*config.ruled_surface_mesh_parameter_list),
        ),
        GeometryFactory(
            "create surface of revolution mesh",
            "Open surface of revolution from z/radius samples.",
            "surface_of_revolution_mesh_parameter_list",
            lambda config: build_surface_of_revolution_mesh(*config.surface_of_revolution_mesh_parameter_list),
        ),
        GeometryFactory(
            "create circular bi-planary mesh",
            "Two circular planar coil surfaces.",
            "circular_biplanar_mesh_parameter_list",
            lambda config: build_circular_biplanar_mesh(*config.circular_biplanar_mesh_parameter_list),
        ),
        GeometryFactory(
            "create bi-planary mesh",
            "Two rectangular planar coil surfaces.",
            "biplanar_mesh_parameter_list",
            lambda config: _rotate_generated_mesh(build_biplanar_mesh(*config.biplanar_mesh_parameter_list), config),
        ),
        GeometryFactory(
            "create shielded bi-planary mesh",
            "Circular biplanar primary surfaces plus offset-copy active shields.",
            "shielded_circular_biplanar_mesh_parameter_list",
            lambda config: _rotate_generated_mesh(
                build_shielded_biplanar_mesh(*config.shielded_circular_biplanar_mesh_parameter_list),
                config,
            ),
        ),
        GeometryFactory(
            "create flanged split bore mesh",
            "Split-bore flanged gradient surface.",
            "flanged_split_bore_mesh_parameter_list",
            lambda config: _rotate_generated_mesh(
                build_split_bore_flange_mesh(*config.flanged_split_bore_mesh_parameter_list),
                config,
            ),
        ),
        GeometryFactory(
            "create shielded flanged split bore mesh",
            "Split-bore flanged primary surfaces plus offset-copy active shields.",
            "shielded_flanged_split_bore_mesh_parameter_list",
            lambda config: _rotate_generated_mesh(_build_shielded_split_bore_from_config(config), config),
        ),
    )
    for factory in factories:
        register_geometry_factory(factory, replace=True)


_register_builtin_geometry_factories()
