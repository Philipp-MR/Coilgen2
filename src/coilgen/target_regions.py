"""Explicit SI sampling regions for analytic targets, separate from the field.

Region placement changes sample coordinates only. Expressions and harmonics
remain in world coordinates. All samples have equal objective weights; these
point clouds are not volume or surface integration quadratures.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .models import Mesh


_DIMENSIONS = {
    "sphere": {"radius_m"},
    "ellipsoid": {"semi_axes_m"},
    "box": {"size_m"},
    "cylinder": {"radius_m", "length_m"},
    "enclosing_mesh": set(),
}
_PLACEMENT = {"center_m", "rotation_axis", "rotation_angle_rad"}
_PHASE = (np.sqrt(5.0) - 1.0) / 2.0


def _triple(value, name: str, *, positive: bool = False) -> tuple[float, float, float]:
    original = np.asarray(value, dtype=object)
    if original.shape != (3,) or any(
        not isinstance(item, Real) or isinstance(item, (bool, np.bool_)) for item in original
    ):
        raise ValueError(f"{name} must contain three finite SI numbers.")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain three finite SI values.") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)) or (positive and np.any(array <= 0)):
        raise ValueError(f"{name} must contain three finite {'positive ' if positive else ''}SI values.")
    return tuple(float(item) for item in array)


def _scalar(value, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number.")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite SI number.") from exc
    if not np.isfinite(scalar) or (positive and scalar <= 0):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number.")
    return scalar


@dataclass(frozen=True, slots=True)
class TargetRegion:
    """Immutable target-region geometry; lengths are metres, angles radians.

    Cylinder axis is local z; box dimensions are full lengths. Imported
    enclosing meshes already contain SI world coordinates and cannot receive
    implicit centering or rotation through this specification.
    """

    shape: str
    sampling: str = "volume"
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    rotation_angle_rad: float = 0.0
    radius_m: float | None = None
    semi_axes_m: tuple[float, float, float] | None = None
    size_m: tuple[float, float, float] | None = None
    length_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, str) or self.shape not in _DIMENSIONS:
            raise ValueError(f"Unknown target-region shape {self.shape!r}.")
        if not isinstance(self.sampling, str) or self.sampling not in {"volume", "surface"}:
            raise ValueError("Target-region sampling must be 'volume' or 'surface'.")
        object.__setattr__(self, "center_m", _triple(self.center_m, "center_m"))
        object.__setattr__(self, "rotation_axis", _triple(self.rotation_axis, "rotation_axis"))
        object.__setattr__(self, "rotation_angle_rad", _scalar(self.rotation_angle_rad, "rotation_angle_rad"))
        if self.rotation_angle_rad and np.max(np.abs(self.rotation_axis)) == 0:
            raise ValueError("A nonzero region rotation requires a nonzero rotation_axis.")
        for name in {"radius_m", "semi_axes_m", "size_m", "length_m"}:
            value = getattr(self, name)
            if name not in _DIMENSIONS[self.shape]:
                if value is not None:
                    raise ValueError(f"{name} does not apply to target-region shape {self.shape!r}.")
                continue
            if value is None:
                raise ValueError(f"Target-region {self.shape} requires {name} in metres.")
            object.__setattr__(
                self,
                name,
                _triple(value, name, positive=True)
                if name.endswith("axes_m") or name == "size_m"
                else _scalar(value, name, positive=True),
            )
        if self.shape == "enclosing_mesh" and (any(self.center_m) or self.rotation_angle_rad != 0):
            raise ValueError(
                "Enclosing meshes use their saved SI world coordinates; set center_m and rotation to zero."
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | TargetRegion) -> TargetRegion:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("target_region_geometry must be a JSON object or None.")
        shape = value.get("shape")
        if not isinstance(shape, str) or shape not in _DIMENSIONS:
            raise ValueError(f"Unknown target-region shape {shape!r}.")
        allowed = {"shape", "sampling", *_PLACEMENT, *_DIMENSIONS[shape]}
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Target-region setting names must be strings.")
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown target-region settings: {', '.join(sorted(unknown))}.")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "shape": self.shape,
            "sampling": self.sampling,
            "center_m": list(self.center_m),
            "rotation_axis": list(self.rotation_axis),
            "rotation_angle_rad": self.rotation_angle_rad,
        }
        for name in sorted(_DIMENSIONS[self.shape]):
            value = getattr(self, name)
            result[name] = list(value) if isinstance(value, tuple) else value
        return result

    @property
    def rotation_matrix(self) -> np.ndarray:
        if self.rotation_angle_rad == 0:
            return np.eye(3)
        # Scale first so finite, very large axes cannot overflow their norm.
        axis = np.asarray(self.rotation_axis) / np.max(np.abs(self.rotation_axis))
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return (
            np.eye(3)
            + np.sin(self.rotation_angle_rad) * cross
            + (1 - np.cos(self.rotation_angle_rad)) * (cross @ cross)
        )

    @property
    def half_extents_m(self) -> np.ndarray:
        if self.shape == "sphere":
            return np.repeat(self.radius_m, 3)
        if self.shape == "ellipsoid":
            return np.asarray(self.semi_axes_m)
        if self.shape == "box":
            return np.asarray(self.size_m) / 2
        if self.shape == "cylinder":
            return np.array([self.radius_m, self.radius_m, self.length_m / 2])
        raise ValueError("Enclosing-mesh extents require the actual mesh.")


def _resolution(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
        raise ValueError("Explicit target-region resolution must be an integer >=2.")
    return int(value)


def _axis(n: int, independent: bool) -> np.ndarray:
    # Interior Cartesian samples avoid singular on-triangle winding tests.
    return 2 * (np.arange(n) + (_PHASE if independent else 0.5)) / n - 1


def _surface_sphere(n: int, independent: bool) -> np.ndarray:
    count = max(12, 6 * n * n)
    offset = _PHASE if independent else 0.5
    z = 1 - 2 * (np.arange(count) + offset) / count
    angle = (np.arange(count) + (_PHASE if independent else 0)) * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(np.maximum(0, 1 - z * z))
    return np.column_stack((r * np.cos(angle), r * np.sin(angle), z))


def _analytic_surface(region: TargetRegion, n: int, independent: bool) -> np.ndarray:
    if region.shape in {"sphere", "ellipsoid"}:
        return _surface_sphere(n, independent) * region.half_extents_m
    grid = _axis(n, independent)
    if region.shape == "box":
        uv = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
        faces = []
        for axis in range(3):
            for side in (-1, 1):
                face = np.empty((len(uv), 3))
                face[:, axis] = side
                face[:, [other for other in range(3) if other != axis]] = uv
                faces.append(face * region.half_extents_m)
        return np.vstack(faces)
    count = max(8, 4 * n)
    angle = (np.arange(count) + (_PHASE if independent else 0)) * 2 * np.pi / count
    a, z = np.meshgrid(angle, grid * region.length_m / 2, indexing="ij")
    side = np.column_stack((region.radius_m * np.cos(a.ravel()), region.radius_m * np.sin(a.ravel()), z.ravel()))
    xy = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    xy = xy[np.sum(xy * xy, axis=1) < 1] * region.radius_m
    caps = [np.column_stack((xy, np.repeat(sign * region.length_m / 2, len(xy)))) for sign in (-1, 1)]
    return np.vstack([side, *caps])


def _validated_surface(mesh: Mesh) -> Mesh:
    vertices = np.asarray(mesh.vertices, dtype=float)
    raw_faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)) or len(vertices) < 3:
        raise ValueError("Target mesh requires at least three finite SI world vertices.")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or len(raw_faces) < 1 or raw_faces.dtype.kind not in "iu":
        raise ValueError("Target mesh requires integer triangular faces.")
    if np.any(raw_faces < 0) or np.any(raw_faces >= len(vertices)):
        raise ValueError("Enclosing mesh face indices are outside its vertex array.")
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[raw_faces]
    extent = float(np.max(np.ptp(vertices, axis=0)))
    tri = vertices[faces]
    if extent <= 0 or np.any(
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        <= 64 * np.finfo(float).eps * extent**2
    ):
        raise ValueError("Enclosing mesh contains degenerate triangles; repair the shell before import.")
    return Mesh(vertices=vertices, faces=faces)


def validate_enclosing_mesh(mesh: Mesh) -> Mesh:
    """Return an exactly welded, closed, consistently oriented mesh copy.

    Open, nonmanifold, disconnected and zero-volume shells fail with an
    actionable message. Topological closure is not a self-intersection or
    fabrication certificate; ambiguous winding is rejected during sampling.
    Both globally inward and outward triangle orientation are accepted.
    """
    surface = _validated_surface(mesh)
    vertices, faces = surface.vertices, surface.faces
    extent = float(np.max(np.ptp(vertices, axis=0)))
    tri = vertices[faces]
    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, indices, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise ValueError(
            "Enclosing mesh must be watertight: every edge must belong to exactly two triangles. Close holes or choose an analytic region."
        )
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    if np.any(np.bincount(indices, weights=signs, minlength=len(edges)) != 0):
        raise ValueError(
            "Enclosing mesh has inconsistent face orientation; orient its triangles consistently before import."
        )
    # Edge closure alone misses two shells touching at a single vertex. Each
    # used vertex must have one connected circular link for a manifold shell.
    links = [{} for _ in vertices]
    for face in faces:
        for i in range(3):
            vertex, a, b = int(face[i]), int(face[(i + 1) % 3]), int(face[(i + 2) % 3])
            links[vertex].setdefault(a, set()).add(b)
            links[vertex].setdefault(b, set()).add(a)
    for link in links:
        if not link:
            continue
        reached, queue = set(), [next(iter(link))]
        while queue:
            node = queue.pop()
            if node not in reached:
                reached.add(node)
                queue.extend(link[node] - reached)
        if len(reached) != len(link) or any(len(adjacent) != 2 for adjacent in link.values()):
            raise ValueError(
                "Enclosing mesh has a nonmanifold vertex; separate touching shells or repair the boundary."
            )
    neighbours = [[] for _ in vertices]
    for a, b in edges:
        neighbours[a].append(int(b))
        neighbours[b].append(int(a))
    seen, pending = {int(faces[0, 0])}, [int(faces[0, 0])]
    while pending:
        for vertex in neighbours[pending.pop()]:
            if vertex not in seen:
                seen.add(vertex)
                pending.append(vertex)
    if len(seen) != len(np.unique(faces)):
        raise ValueError(
            "Enclosing mesh must be one connected closed shell; nested or disconnected shells are not supported."
        )
    shifted = tri - vertices.mean(axis=0)
    volume = np.sum(np.einsum("ij,ij->i", shifted[:, 0], np.cross(shifted[:, 1], shifted[:, 2]))) / 6
    if abs(volume) <= 64 * np.finfo(float).eps * extent**3:
        raise ValueError("Enclosing mesh has zero or numerically unresolved enclosed volume.")
    return Mesh(vertices=vertices, faces=faces)


def validate_target_region_mesh(mesh: Mesh, sampling: str = "volume") -> Mesh:
    """Linear topology/units preflight without constructing a dense point cloud.

    Volume requires a closed manifold; surface accepts open triangles. Actual
    volume containment and nonempty sampling are checked by the sampler.
    """
    if not isinstance(sampling, str) or sampling not in {"volume", "surface"}:
        raise ValueError("Target-region sampling must be 'volume' or 'surface'.")
    return validate_enclosing_mesh(mesh) if sampling == "volume" else _validated_surface(mesh)


def _inside_enclosure(points: np.ndarray, mesh: Mesh) -> np.ndarray:
    triangles = mesh.vertices[mesh.faces]
    result = np.zeros(len(points), dtype=bool)
    # Solid angle winding handles concave boundaries without optional rtree.
    for start in range(0, len(points), 64):
        query = points[start : start + 64]
        winding = np.zeros(len(query))
        on_boundary = np.zeros(len(query), bool)
        for face_start in range(0, len(triangles), 256):
            vectors = triangles[None, face_start : face_start + 256] - query[:, None, None, :]
            lengths = np.linalg.norm(vectors, axis=-1)
            a, b, c = vectors[:, :, 0], vectors[:, :, 1], vectors[:, :, 2]
            la, lb, lc = lengths[:, :, 0], lengths[:, :, 1], lengths[:, :, 2]
            numerator = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
            denominator = (
                la * lb * lc
                + np.einsum("pfi,pfi->pf", a, b) * lc
                + np.einsum("pfi,pfi->pf", b, c) * la
                + np.einsum("pfi,pfi->pf", c, a) * lb
            )
            winding += np.sum(2 * np.arctan2(numerator, denominator), axis=1) / (4 * np.pi)
            normal = np.cross(b - a, c - a)
            normal_length = np.linalg.norm(normal, axis=-1)
            tolerance = 64 * np.finfo(float).eps * np.max(np.ptp(mesh.vertices, axis=0))
            on_plane = np.abs(np.einsum("pfi,pfi->pf", a, normal)) <= tolerance * normal_length
            inside_triangle = on_plane
            for edge, start_vector in ((b - a, a), (c - b, b), (a - c, c)):
                side = np.einsum("pfi,pfi->pf", np.cross(edge, -start_vector), normal)
                inside_triangle &= side >= -tolerance * normal_length * np.linalg.norm(edge, axis=-1)
            on_boundary |= np.any(inside_triangle, axis=1)
        magnitude = np.abs(winding)
        ambiguous = (~on_boundary) & (np.minimum(magnitude, np.abs(magnitude - 1)) > 1e-6)
        if np.any(ambiguous) or np.any((magnitude > 1 + 1e-6) & ~on_boundary):
            raise ValueError(
                "Enclosing-mesh containment is ambiguous at sampled points; repair self-intersections or boundary degeneracies, or change resolution."
            )
        result[start : start + len(query)] = (magnitude > 0.5) & ~on_boundary
    return result


def sample_target_region(
    spec: Mapping[str, Any] | TargetRegion, resolution: int, *, independent: bool = False, mesh: Mesh | None = None
) -> np.ndarray:
    """Deterministic equal-weight world samples, with a distinct validation phase.

    Resolution is Cartesian bins per local bounding-box axis for volumes;
    analytic surfaces use proportional angular/face resolution. Imported
    surfaces have a global 6*resolution**2 area-distributed point budget.
    """
    region, n = TargetRegion.from_dict(spec), _resolution(resolution)
    if region.shape == "enclosing_mesh":
        if mesh is None:
            kind = "an SI triangular surface" if region.sampling == "surface" else "a closed SI mesh"
            raise ValueError(f"An imported target region requires target_mesh_file containing {kind}.")
        if region.sampling == "surface":
            surface = _validated_surface(mesh)
            triangles = surface.vertices[surface.faces]
            areas = np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
            )
            cumulative = np.cumsum(areas) / np.sum(areas)
            count = max(12, 6 * n * n)
            index = np.arange(count)
            phase = _PHASE if independent else 0.5
            selected = np.searchsorted(cumulative, (index + phase) / count)
            # Deterministic triangle-interior locations; area proportional
            # coverage avoids dependence on arbitrary local triangulation.
            u = (phase + index * _PHASE) % 1
            v = (phase + index * np.sqrt(2)) % 1
            root_u = np.sqrt(u)
            bary = np.column_stack((1 - root_u, root_u * (1 - v), root_u * v))
            points = np.einsum("nk,nkj->nj", bary, triangles[selected])
        else:
            enclosing = validate_enclosing_mesh(mesh)
            low, high = enclosing.vertices.min(axis=0), enclosing.vertices.max(axis=0)
            axes = [low[i] + (_axis(n, independent) + 1) * (high[i] - low[i]) / 2 for i in range(3)]
            candidates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            points = candidates[_inside_enclosure(candidates, enclosing)]
    elif region.sampling == "surface":
        points = _analytic_surface(region, n, independent)
        points = points @ region.rotation_matrix.T + region.center_m
    else:
        unit = np.stack(np.meshgrid(*[_axis(n, independent)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
        keep = np.ones(len(unit), bool)
        if region.shape in {"sphere", "ellipsoid"}:
            keep = np.sum(unit * unit, axis=1) < 1
        elif region.shape == "cylinder":
            keep = np.sum(unit[:, :2] ** 2, axis=1) < 1
        points = (unit[keep] * region.half_extents_m) @ region.rotation_matrix.T + region.center_m
    if len(points) == 0:
        raise ValueError(
            "Target region contains no sample points at this resolution; increase resolution or repair the enclosure."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("Target-region coordinates overflowed; use physically reasonable SI dimensions and placement.")
    return points


def sample_target_region_from_config(
    config,
    *,
    resolution: int | None = None,
    independent: bool = False,
    mesh: Mesh | None = None,
    project_root: str | Path | None = None,
) -> np.ndarray:
    """Load an owned enclosure if needed, then preflight or generate samples."""
    region = TargetRegion.from_dict(config.target_region_geometry)
    if region.shape == "enclosing_mesh" and mesh is None:
        from .geometry import load_optional_target_mesh

        mesh = load_optional_target_mesh(
            config.target_mesh_file, config, Path.cwd() if project_root is None else Path(project_root)
        )
    return sample_target_region(
        region,
        config.target_region_resolution if resolution is None else resolution,
        independent=independent,
        mesh=mesh,
    )


def target_region_summary(config) -> str:
    """Compact physical sampling description without filesystem access."""
    if str(config.target_field_definition_file).strip().lower() not in {"", "none"}:
        return "Imported field map · saved world coordinates and values"
    if config.target_region_geometry is None:
        if str(config.target_mesh_file).strip().lower() not in {"", "none"}:
            return (
                "Legacy target mesh · vertices"
                if config.use_only_target_mesh_verts
                else "Legacy target mesh · bounding-box grid + vertices"
            )
        placement = "support mean" if config.set_roi_into_mesh_center else "world origin"
        return f"Sphere · volume · radius {config.target_region_radius * 1000:g} mm · {placement} (legacy grid)"
    region = TargetRegion.from_dict(config.target_region_geometry)
    dimensions = ""
    if region.shape == "sphere":
        dimensions = f" · radius {region.radius_m * 1000:g} mm"
    elif region.shape == "ellipsoid":
        dimensions = " · semi-axes " + " × ".join(f"{x * 1000:g}" for x in region.semi_axes_m) + " mm"
    elif region.shape == "box":
        dimensions = " · size " + " × ".join(f"{x * 1000:g}" for x in region.size_m) + " mm"
    elif region.shape == "cylinder":
        dimensions = f" · radius {region.radius_m * 1000:g}, length {region.length_m * 1000:g} mm"
    if region.shape == "enclosing_mesh":
        title = (
            "Imported mesh boundary samples" if region.sampling == "surface" else "Closed imported enclosure · volume"
        )
        return f"{title} · saved SI world coordinates"
    center = ", ".join(f"{x * 1000:g}" for x in region.center_m)
    angle = np.degrees(region.rotation_angle_rad)
    axis = ", ".join(f"{x:g}" for x in region.rotation_axis)
    rotation = f" · {angle:g}° about [{axis}]" if angle else " · rotation 0°"
    return f"{region.shape.capitalize()} · {region.sampling}{dimensions} · center [{center}] mm{rotation}"
