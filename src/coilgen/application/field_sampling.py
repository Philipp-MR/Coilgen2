"""Bounded slices of an unchanged saved physical field and its owned desired target.

Direct evaluations use stored stream functions or exact electrical polylines
and signed currents. Display interpolation is separately labelled, confined to
the sampling hull and screened for large sampling gaps. Neither is a new design
or independent acceptance test. Singular current-sheet/wire samples stay masked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from numbers import Real
from typing import Mapping

import numpy as np

from ..config import CoilGenConfig
from ..models import CoilPart, Mesh, TargetField
from ..progress import DesignCancelled, progress_context
from ..target_regions import TargetRegion
from .gradient_studies import RigidTransform
from .project import _freeze

MAX_GRID_POINTS = 512 * 1024
MAX_OPERATOR_BYTES = 64 * 1024**2
MAX_SOURCE_BYTES = 256 * 1024**2
MAX_PHYSICAL_INTERACTIONS = 128 * 1024**2
_PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}


def _readonly(value, dtype=None):
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class FieldSlice:
    offset_m: float
    coordinates_m: np.ndarray
    field_T: np.ndarray
    target_T: np.ndarray | None
    available_mask: np.ndarray
    target_valid_mask: np.ndarray
    roi_mask: np.ndarray | None
    outlines_m: tuple = ()
    support_outlines_m: tuple = ()
    reasons: Mapping = field(default_factory=dict)

    def __post_init__(self):
        shape = np.shape(self.coordinates_m)
        if len(shape) != 3 or shape[2] != 3 or np.shape(self.field_T) != shape:
            raise ValueError("Slices require matching rectangular NxMx3 coordinates and fields.")
        for name in ("coordinates_m", "field_T", "target_T", "available_mask", "target_valid_mask", "roi_mask"):
            value = getattr(self, name)
            if value is None:
                continue
            expected = shape if name.endswith("_T") or name == "coordinates_m" else shape[:2]
            if np.shape(value) != expected:
                raise ValueError(f"Incompatible {name} grid shape.")
            object.__setattr__(self, name, _readonly(value, bool if name.endswith("mask") else float))
        for name in ("outlines_m", "support_outlines_m"):
            object.__setattr__(self, name, tuple(_readonly(line, float) for line in getattr(self, name)))
        object.__setattr__(self, "reasons", _freeze(dict(self.reasons)))


@dataclass(frozen=True, slots=True)
class SliceCollection:
    source_key: str
    label: str
    plane: str
    slices: tuple[FieldSlice, ...]
    representation: str
    target_representation: str
    roi_representation: str
    provenance: Mapping

    def __post_init__(self):
        object.__setattr__(self, "slices", tuple(self.slices))
        object.__setattr__(self, "provenance", _freeze(dict(self.provenance)))


def _cancel(check):
    if check is not None and check():
        raise DesignCancelled("Field sampling cancelled by the user.")


def _numbers(value, count, name, *, positive=False):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (count,) or any(isinstance(item, (bool, np.bool_)) or not isinstance(item, Real) for item in raw):
        raise ValueError(f"{name} requires {count} finite SI numbers.")
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)) or positive and np.any(result <= 0):
        raise ValueError(f"{name} requires finite {'positive ' if positive else ''}SI numbers.")
    return result


def _request(record, supplied):
    from .evidence import load_saved_request

    if any(item.get("name") == "request.json" for item in record.metadata.get("artifacts", ())):
        owned = load_saved_request(record)
        if supplied is not None and json.dumps(owned, sort_keys=True) != json.dumps(supplied, sort_keys=True):
            raise ValueError("Slice settings must come from the unchanged selected run's original request.")
        return owned
    # Legacy/synthetic snapshots can still display their saved fields and actual
    # geometry. A current model never supplies an inferred analytic target.
    return supplied


def _registered(points, transform, *, inverse=False):
    rotation, translation = np.asarray(transform.rotation), np.asarray(transform.translation_m)
    return (points - translation) @ rotation if inverse else points @ rotation.T + translation


def _scenario(dataset, request):
    row = dataset.provenance.get("requirements_row")
    if row is None:
        return None, None
    if request is None or request.get("requirements") is None:
        return row, None
    from .research_requirements import ResearchRequirements

    spec = ResearchRequirements.from_dict(request["requirements"])
    scenario = next((item for item in spec.scenarios if item.name == row["scenario"]), None)
    if scenario is None and row["scenario"] != "Nominal":
        raise ValueError("Selected regional field names an unknown original physical scenario.")
    return row, scenario


def _support_points(points, scenario):
    if scenario is None:
        return np.asarray(points)
    centre, scale = np.asarray(scenario.support_scale_center_m), np.asarray(scenario.support_scale_xyz)
    return _registered((points - centre) * scale + centre, scenario.support_transform)


def _sheet_source(record, config, scenario):
    from ..basis import calculate_basis_functions, calculate_one_ring_by_mesh
    from ..geometry import compute_face_normals, compute_vertex_normals

    names = {"vertices_m", "faces", "stream_function_A"}
    if not names <= set(record.array_names):
        raise ValueError(
            "Saved surface vertices, triangles and nodal stream function are required for direct sheet evaluation."
        )
    arrays = record.select_arrays(names, copy=False)
    if sum(value.nbytes for value in arrays.values()) > MAX_SOURCE_BYTES:
        raise ValueError("Saved sheet exceeds the explicit 256 MiB direct-source budget.")
    vertices = _support_points(arrays["vertices_m"], scenario)
    faces, sf = np.asarray(arrays["faces"]), np.asarray(arrays["stream_function_A"])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.all(np.isfinite(vertices)):
        raise ValueError("Saved sheet vertices are invalid.")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or faces.dtype.kind not in "iu"
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Saved sheet triangles are invalid.")
    if sf.shape != (len(vertices),) or sf.dtype.kind not in "fiu" or not np.all(np.isfinite(sf)):
        raise ValueError("Saved nodal stream function must match the exact saved vertex order.")
    mesh = Mesh(vertices=vertices, faces=faces)
    mesh.face_normals, mesh.vertex_normals = compute_face_normals(mesh), compute_vertex_normals(mesh)
    if np.any(
        np.linalg.norm(
            np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]),
            axis=1,
        )
        == 0
    ):
        raise ValueError("Saved sheet contains degenerate triangles.")
    part = CoilPart(mesh)
    calculate_one_ring_by_mesh([part])
    calculate_basis_functions([part])
    return part, sf


def _ordered_names(record, prefix):
    import re

    names = [name for name in record.array_names if name.startswith(prefix) and name.endswith("_points_m")]
    return sorted(names, key=lambda name: tuple(int(value) for value in re.findall(r"\d+", name)))


def _wire_source(record, dataset, row, scenario):
    """Select exactly the represented paths; never substitute fabricated for contour geometry."""
    metadata = record.metadata
    paths = []
    if row is not None:
        names = (
            _ordered_names(record, "review_electrical_")
            if row["scope"] in {"complete_electrical_paths", "partial_electrical_paths"}
            else _ordered_names(record, "evaluated_winding_")
        )
        currents = row.get("currents_A", ())
        expected = row.get("path_sha256", ())
        if not names or len(names) != len(currents) or len(names) != len(expected):
            raise ValueError("Regional winding/current/path identity is incomplete.")
        for name, current, digest in zip(names, currents, expected, strict=True):
            points = record.array(name, copy=False)
            if sha256(np.asarray(points, dtype="<f8").tobytes()).hexdigest() != digest:
                raise ValueError("Regional field references different saved winding geometry.")
            paths.append((points, current))
    elif dataset.key.startswith("contour"):
        # Multiplicities live in the typed checkpoint, not the display arrays.
        from .evidence import read_verified_artifact
        from ..pipeline_stages import PipelineCheckpoint

        checkpoint = PipelineCheckpoint.load(
            BytesIO(read_verified_artifact(record, "pipeline_checkpoint.cgstage")), max_bytes=MAX_SOURCE_BYTES
        )
        current = dataset.provenance.get("current_A")
        if current is None:
            evidence = metadata.get("wire_fields", {}).get("contour", {})
            current = (
                evidence.get("at_stream_current", {}) if dataset.key.endswith("at_stream_current") else evidence
            ).get("current_A")
        for part in checkpoint.state.parts:
            for contour in part.contour_lines:
                paths.append((contour.v, current * contour.turn_multiplicity if current is not None else None))
    elif dataset.key.startswith("routed"):
        evidence = metadata.get("wire_fields", {}).get("routed", {})
        current = (
            evidence.get("at_stream_current", {}) if dataset.key.endswith("at_stream_current") else evidence
        ).get("current_A")
        paths = [(record.array(name, copy=False), current) for name in _ordered_names(record, "evaluated_winding_")]
    else:
        from .gradient_studies import _saved_winding

        winding, currents, _source, _complete = _saved_winding(record, "original")
        paths = [(points, currents[name]) for name, points in winding.items()]
    if not paths:
        raise ValueError("No saved physical winding paths are available for this field source.")
    result = []
    for points, current in paths:
        points = _support_points(points, scenario)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 2
            or not np.all(np.isfinite(points))
            or not np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-15)
        ):
            raise ValueError("Saved winding requires finite nonzero-length physical segments.")
        if isinstance(current, (bool, np.bool_)) or not isinstance(current, Real) or not np.isfinite(current):
            raise ValueError("Every saved winding path requires its unchanged finite signed current.")
        result.append((points, float(current)))
    if sum(points.nbytes for points, _ in result) > MAX_SOURCE_BYTES:
        raise ValueError("Saved winding exceeds the explicit 256 MiB direct-source budget.")
    return result


def _sheet_mask(points, part, check=None):
    from scipy.spatial import cKDTree
    from ..geometry_kernels import _point_triangle_sq_numpy

    triangles = part.mesh.vertices[part.mesh.faces]
    centres = triangles.mean(axis=1)
    radius = np.max(np.linalg.norm(triangles - centres[:, None], axis=2))
    tolerance = 64 * np.finfo(float).eps * max(1, np.max(np.abs(points)), np.max(np.abs(part.mesh.vertices)))
    tree = cKDTree(centres)
    valid = np.ones(len(points), bool)
    for start in range(0, len(points), 128):
        _cancel(check)
        query = points[start : start + 128]
        candidates = tree.query_ball_point(query, radius + tolerance)
        counts = np.asarray([len(ids) for ids in candidates])
        if not counts.sum():
            continue
        if counts.sum() > 1000000:
            raise ValueError("Sheet proximity candidates exceed the bounded slice workspace.")
        ids = np.concatenate([np.asarray(ids, int) for ids in candidates])
        repeated = np.repeat(query, counts, axis=0)
        near = _point_triangle_sq_numpy(repeated, triangles[ids]) <= tolerance**2
        origin = np.repeat(np.arange(len(query)), counts)
        valid[start + np.unique(origin[near])] = False
    return valid


def _wire_mask(points, paths, check=None):
    from scipy.spatial import cKDTree

    valid = np.ones(len(points), bool)
    scale = max(1, np.max(np.abs(points)), *(np.max(np.abs(path)) for path, _ in paths))
    tolerance = 64 * np.finfo(float).eps * scale
    _cancel(check)
    starts = np.vstack([path[:-1] for path, _ in paths])
    vectors = np.vstack([np.diff(path, axis=0) for path, _ in paths])
    squared = np.einsum("ij,ij->i", vectors, vectors)
    nonzero = squared > 1e-30
    starts, vectors, squared = starts[nonzero], vectors[nonzero], squared[nonzero]
    tree = cKDTree(starts + vectors / 2)
    radius = np.sqrt(squared.max()) / 2 + tolerance
    for start in range(0, len(points), 128):
        _cancel(check)
        query = points[start : start + 128]
        candidates = tree.query_ball_point(query, radius)
        counts = np.asarray([len(ids) for ids in candidates])
        if not counts.sum():
            continue
        if counts.sum() > 1000000:
            raise ValueError("Wire proximity candidates exceed the bounded slice workspace.")
        ids = np.concatenate([np.asarray(ids, int) for ids in candidates])
        offset = np.repeat(query, counts, axis=0) - starts[ids]
        fraction = np.clip(np.einsum("ij,ij->i", offset, vectors[ids]) / squared[ids], 0, 1)
        near = np.linalg.norm(offset - fraction[:, None] * vectors[ids], axis=1) <= tolerance
        origin = np.repeat(np.arange(len(query)), counts)
        valid[start + np.unique(origin[near])] = False
    return valid


def _physical(points, kind, source, config, check, progress=None):
    from ..physics import biot_savart_wire_path, calculate_sensitivity_matrix

    values = np.full_like(points, np.nan)
    valid = _sheet_mask(points, source[0], check) if kind == "sheet" else _wire_mask(points, source, check)
    indices = np.flatnonzero(valid)
    count = len(source[1]) if kind == "sheet" else 1
    chunk_size = max(1, min(256, MAX_OPERATOR_BYTES // max(3 * count * 8, 1)))
    for start in range(0, len(indices), chunk_size):
        _cancel(check)
        selected = indices[start : start + chunk_size]
        samples = points[selected]
        with progress_context(lambda _event: None, cancel_check=check):
            if kind == "sheet":
                part, sf = source
                target = TargetField(
                    samples,
                    np.zeros_like(samples),
                    np.ones(len(samples)),
                    np.zeros(len(samples), bool),
                    np.zeros(len(samples), int),
                )
                calculate_sensitivity_matrix([part], target, config)
                values[selected] = np.einsum("ctv,v->tc", part.sensitivity_matrix, sf)
                part.sensitivity_matrix = None
            else:
                field_values = np.zeros_like(samples)
                for path, current in source:
                    _cancel(check)
                    for segment_start in range(0, len(path) - 1, 256):
                        _cancel(check)
                        segment_path = path[segment_start : min(len(path), segment_start + 257)]
                        field_values += current * biot_savart_wire_path(
                            segment_path, samples, worker_count=1, chunk_size=128
                        )
                values[selected] = field_values
        if progress is not None:
            progress(min(start + chunk_size, len(indices)) / max(len(indices), 1))
    valid &= np.all(np.isfinite(values), axis=1)
    values[~valid] = np.nan
    return values, valid


class _Interpolator:
    """Affine-rank aware linear interpolation without hull extrapolation or nearest fill."""

    def __init__(self, points, values, max_gap_m=None):
        from scipy.spatial import cKDTree, Delaunay

        points, values = np.asarray(points), np.asarray(values)
        if len(points) > 100000:
            raise ValueError("Display interpolation supports at most 100000 saved samples.")
        unique, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
        if len(unique) != len(points):
            reduced = values[first]
            if not np.allclose(values, reduced[inverse], rtol=1e-12, atol=1e-15):
                raise ValueError("Duplicate saved coordinates have incompatible field values.")
            points, values = unique, reduced
        self.origin, self.values, self.points = points.mean(axis=0), values, points
        self.tree = cKDTree(points)
        self.tolerance = 128 * np.finfo(float).eps * max(1, np.max(np.abs(points)))
        _, singular, vh = np.linalg.svd(points - self.origin, full_matrices=False)
        self.rank = int(np.count_nonzero(singular > self.tolerance * max(np.sqrt(len(points)), 1)))
        self.basis = vh[: self.rank].T
        self.local = (points - self.origin) @ self.basis
        if max_gap_m is not None:
            self.max_gap_m = float(_numbers([max_gap_m], 1, "Interpolation maximum gap", positive=True)[0])
        elif len(points) > 1:
            spacing = self.tree.query(points, k=2)[0][:, 1]
            self.max_gap_m = 3 * float(np.median(spacing[spacing > 0]))
        else:
            self.max_gap_m = 0.0
        self.triangulation = Delaunay(self.local) if self.rank >= 2 else None
        self.sorted = np.argsort(self.local[:, 0]) if self.rank == 1 else None

    def sample(self, points):
        points = np.asarray(points)
        local = (points - self.origin) @ self.basis
        projected = self.origin + local @ self.basis.T
        on_span = np.linalg.norm(points - projected, axis=1) <= self.tolerance
        valid = np.zeros(len(points), bool)
        values = np.full((len(points), 3), np.nan)
        if self.rank == 0:
            valid = np.linalg.norm(points - self.points[0], axis=1) <= self.tolerance
            values[valid] = self.values[0]
        elif self.rank == 1:
            axis = self.local[self.sorted, 0]
            interval = np.searchsorted(axis, local[:, 0], side="right") - 1
            interval = np.clip(interval, 0, len(axis) - 2)
            gap = axis[interval + 1] - axis[interval]
            valid = (
                on_span
                & (local[:, 0] >= axis[0] - self.tolerance)
                & (local[:, 0] <= axis[-1] + self.tolerance)
                & (gap <= self.max_gap_m + self.tolerance)
            )
            fraction = (local[:, 0] - axis[interval]) / gap
            values[valid] = (1 - fraction[valid, None]) * self.values[self.sorted[interval[valid]]] + fraction[
                valid, None
            ] * self.values[self.sorted[interval[valid] + 1]]
        else:
            simplex = self.triangulation.find_simplex(local, tol=self.tolerance)
            candidates = np.flatnonzero(on_span & (simplex >= 0))
            if len(candidates):
                simplices = simplex[candidates]
                vertices = self.triangulation.simplices[simplices]
                largest = np.zeros(len(vertices))
                for a in range(self.rank + 1):
                    for b in range(a + 1, self.rank + 1):
                        largest = np.maximum(
                            largest, np.linalg.norm(self.points[vertices[:, a]] - self.points[vertices[:, b]], axis=1)
                        )
                candidates, simplices, vertices = (
                    candidates[largest <= self.max_gap_m + self.tolerance],
                    simplices[largest <= self.max_gap_m + self.tolerance],
                    vertices[largest <= self.max_gap_m + self.tolerance],
                )
                transforms = self.triangulation.transform[simplices]
                barycentric = np.einsum(
                    "nij,nj->ni", transforms[:, : self.rank], local[candidates] - transforms[:, self.rank]
                )
                barycentric = np.column_stack((barycentric, 1 - barycentric.sum(axis=1)))
                values[candidates] = np.einsum("ni,nic->nc", barycentric, self.values[vertices])
                valid[candidates] = True
        # Exact saved nodes remain observations even where adjacent unsampled
        # simplices are masked. This is exact coincidence, never nearest fill.
        distance, index = self.tree.query(points, k=1)
        observed = distance <= self.tolerance
        values[observed], valid[observed] = self.values[index[observed]], True
        return values, valid


def _target_evaluator(record, dataset, request, row, scenario):
    from ..target import evaluate_field_expression, field_values_to_tesla

    if row is not None and request is not None and request.get("requirements") is not None:
        from .research_requirements import ResearchRequirements

        spec = ResearchRequirements.from_dict(request["requirements"])
        region = next((item for item in spec.regions if item.name == row["region"]), None)
        owned = (
            region.validation
            if region is not None and row["sampling"] == "independent_validation"
            else region.design
            if region
            else None
        )
        if owned is not None and owned.recipe is not None:
            if row.get("dataset_sha256") != owned.digest:
                raise ValueError("Selected regional target does not match its owned desired dataset.")
            sample = scenario.sample_transform if scenario is not None else RigidTransform()

            def evaluate(points):
                source = _registered(_registered(points, sample, inverse=True), owned.registration, inverse=True)
                values = np.column_stack(
                    [evaluate_field_expression(expression, source) for expression in owned.recipe["expressions"]]
                )
                return values @ np.asarray(owned.registration.rotation).T @ np.asarray(sample.rotation).T

            return evaluate
    recipe = record.metadata.get("target_recipe")
    if recipe is not None:
        if recipe.get("coordinate_frame") != "world" or recipe.get("field_component") != "Bz":
            raise ValueError("Unsupported saved analytic target coordinate/component contract.")
        scale = recipe.get("normalization_scale")
        if isinstance(scale, bool) or not isinstance(scale, Real) or not np.isfinite(scale):
            raise ValueError("Saved target normalization scale is unavailable.")

        def evaluate(points):
            values = np.zeros_like(points)
            values[:, 2] = evaluate_field_expression(recipe["field_shape_function"], points) * scale
            return field_values_to_tesla(values, recipe["target_field_units"])

        return evaluate
    # Explicit no-normalization is replayable without a saved scale. No old
    # extrema/linear-fit scale is estimated from the new display grid.
    config = (request or {}).get("config", {})
    if (
        config.get("target_gradient_normalization") == "none"
        and str(config.get("target_field_definition_file", "none")).lower() in {"", "none"}
        and (request or {}).get("requirements") is None
    ):

        def evaluate(points):
            values = np.zeros_like(points)
            values[:, 2] = evaluate_field_expression(config.get("field_shape_function", "x"), points)
            return field_values_to_tesla(values, config.get("target_field_units", "T"))

        return evaluate
    return None


def _region_level(points, geometry):
    region = TargetRegion.from_dict(geometry.get("region", geometry))
    source = np.asarray(points)
    for key in ("sample_transform", "registration"):
        source = _registered(source, RigidTransform(**dict(geometry.get(key, {}))), inverse=True)
    local = (source - region.center_m) @ region.rotation_matrix
    if region.shape in {"sphere", "ellipsoid"}:
        return np.linalg.norm(local / region.half_extents_m, axis=1) - 1
    if region.shape == "box":
        return np.max(np.abs(local) / region.half_extents_m, axis=1) - 1
    if region.shape == "cylinder":
        return (
            np.maximum(
                np.linalg.norm(local[:, :2], axis=1) / region.radius_m, np.abs(local[:, 2]) / (region.length_m / 2)
            )
            - 1
        )
    raise ValueError("An enclosing target volume needs its verified saved boundary mesh.")


def _check_target_recipe(record, dataset, function, row):
    if function is None:
        return
    points, desired = dataset.points_m, dataset.target_T
    active = np.ones(len(points), bool)
    primary = (
        row is None
        and {"target_points_m", "target_field_T"} <= set(record.array_names)
        and (
            dataset.provenance.get("points_key") == "target_points_m"
            or np.array_equal(points, record.array("target_points_m", copy=False))
        )
    )
    if primary:
        points, desired = record.array("target_points_m", copy=False), record.array("target_field_T", copy=False)
        active = np.ones(len(points), bool)
        if "target_suppressed" in record.array_names:
            active &= ~record.array("target_suppressed", copy=False).astype(bool)
    if desired is None or not np.any(active):
        return
    candidates = np.flatnonzero(active)
    selected = candidates[np.linspace(0, len(candidates) - 1, min(len(candidates), 24), dtype=int)]
    replayed = function(points[selected])
    scale = max(float(np.max(np.abs(desired[selected]))), 1e-15)
    if not np.all(np.isfinite(replayed)) or not np.allclose(
        replayed, desired[selected], rtol=1e-10, atol=scale * 1e-10 + 1e-15
    ):
        raise ValueError("The saved analytic target recipe does not reproduce its primary desired samples.")


def _marching_outline(grid, level, check=None):
    """Piecewise linear slice silhouette of an explicitly known implicit geometry."""
    segments = []
    corners = ((0, 0), (0, 1), (1, 1), (1, 0))
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    adjacent = np.stack((level[:-1, :-1], level[:-1, 1:], level[1:, 1:], level[1:, :-1]), axis=-1)
    active = np.argwhere(np.any(adjacent <= 0, axis=-1) & np.any(adjacent > 0, axis=-1))
    for index, (i, j) in enumerate(active):
        if index % 128 == 0:
            _cancel(check)
        points = np.asarray([grid[i + di, j + dj] for di, dj in corners])
        values = np.asarray([level[i + di, j + dj] for di, dj in corners])
        crossings = []
        for a, b in edges:
            if (values[a] <= 0) != (values[b] <= 0):
                fraction = values[a] / (values[a] - values[b])
                crossings.append(points[a] + fraction * (points[b] - points[a]))
        if len(crossings) == 2:
            segments.append(np.asarray(crossings))
        elif len(crossings) == 4:
            segments.extend((np.asarray(crossings[:2]), np.asarray(crossings[2:])))
    return tuple(segments)


def _mesh_section(vertices, faces, plane_axis, offset, check=None):
    segments = []
    tolerance = 64 * np.finfo(float).eps * max(1, np.max(np.abs(vertices)), abs(offset))
    triangles = np.asarray(vertices)[faces]
    coordinate = triangles[:, :, plane_axis]
    selected = triangles[
        (coordinate.min(axis=1) <= offset + tolerance) & (coordinate.max(axis=1) >= offset - tolerance)
    ]
    if len(selected) > 100000:
        raise ValueError("Slice intersections exceed the 100000 triangle display-context budget.")
    for index, triangle in enumerate(selected):
        if index % 128 == 0:
            _cancel(check)
        signed = triangle[:, plane_axis] - offset
        if np.all(signed > tolerance) or np.all(signed < -tolerance):
            continue
        crossings = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if abs(signed[a]) <= tolerance:
                crossings.append(triangle[a])
            if signed[a] * signed[b] < 0:
                crossings.append(triangle[a] + signed[a] / (signed[a] - signed[b]) * (triangle[b] - triangle[a]))
        unique = []
        for point in crossings:
            if not any(np.linalg.norm(point - other) <= tolerance for other in unique):
                unique.append(point)
        if len(unique) == 2:
            segments.append(np.asarray(unique))
        elif len(unique) == 3:
            segments.extend(np.asarray([unique[a], unique[b]]) for a, b in ((0, 1), (1, 2), (2, 0)))
    return tuple(segments)


def _owned_enclosure(record, geometry):
    """Load the exact historical boundary, including relocated hash-addressed inputs."""
    from pathlib import Path
    import struct
    from zipfile import ZipFile
    from .evidence import _checked_snapshot_metadata, EvidenceIntegrityError
    from ..geometry import read_mesh_npz, normalize_mesh, _read_ascii_stl, _read_binary_stl
    from ..target_regions import validate_target_region_mesh

    _metadata, manifest, _report = _checked_snapshot_metadata(record)
    item = manifest.get("input_files", {}).get("target_mesh_file")
    if item is None:
        raise ValueError("The original enclosing boundary is not recorded as an owned input.")
    path = record.directory / "portable_inputs" / item["sha256"]
    if not path.exists():
        path = Path(item["path"])
    with path.open("rb") as handle:
        data = handle.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError("Target boundary exceeds the 256 MiB source budget.")
    if sha256(data).hexdigest() != item["sha256"]:
        raise EvidenceIntegrityError("The original target boundary changed after its integrity check.")
    if Path(item["path"]).suffix.lower() == ".npz":
        with ZipFile(BytesIO(data)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > MAX_SOURCE_BYTES:
                raise ValueError("Expanded target boundary exceeds the 256 MiB source budget.")
        mesh = read_mesh_npz(BytesIO(data))
    else:
        count = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else 0
        mesh = _read_binary_stl(data, count) if 84 + 50 * count == len(data) else _read_ascii_stl(data)
        mesh = normalize_mesh(mesh)
    region = TargetRegion.from_dict(geometry.get("region", geometry))
    mesh = validate_target_region_mesh(mesh, region.sampling)
    for key in ("registration", "sample_transform"):
        mesh.vertices = _registered(mesh.vertices, RigidTransform(**dict(geometry.get(key, {}))))
    return mesh


def _enclosure_mask(points, mesh, check):
    from ..target_regions import _inside_enclosure

    result = np.zeros(len(points), bool)
    if len(points) * len(mesh.faces) > MAX_PHYSICAL_INTERACTIONS:
        raise ValueError("Enclosing-volume containment exceeds the bounded slice budget; reduce grid resolution.")
    for start in range(0, len(points), 64):
        _cancel(check)
        result[start : start + 64] = _inside_enclosure(points[start : start + 64], mesh)
    return result


def sample_slices(
    record,
    dataset,
    *,
    plane="xy",
    offsets_m=(0.0,),
    centre_m=(0.0, 0.0, 0.0),
    size_m=(0.1, 0.1),
    resolution=64,
    representation="auto",
    request=None,
    interpolation_max_gap_m=None,
    progress=None,
    cancel_check=None,
):
    """Return rectangular slices from the selected immutable scientific snapshot.

    Offsets are absolute world coordinates on the normal axis; centre_m locates
    the two in-plane axes. ``direct`` refuses unavailable physical source data;
    ``auto`` explicitly records a display interpolation fallback. Target masks
    are separate: absent imported target coverage never hides a valid field.
    """
    from .field_analysis import FieldDataset, source_catalog
    from .evidence import EvidenceIntegrityError

    if plane not in _PLANES or representation not in {"auto", "direct", "interpolation"}:
        raise ValueError("Select xy/xz/yz and auto/direct/interpolation representation.")
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)) or not 3 <= resolution <= 256:
        raise ValueError("Slice resolution must be an integer from 3 to 256.")
    centre, size = _numbers(centre_m, 3, "Slice centre"), _numbers(size_m, 2, "Slice size", positive=True)
    offsets = np.asarray(offsets_m, dtype=object)
    if offsets.ndim != 1 or not 1 <= len(offsets) <= 25:
        raise ValueError("Choose 1–25 finite absolute slice offsets.")
    offsets = _numbers(offsets_m, len(offsets), "Slice offsets")
    if len(set(offsets)) != len(offsets) or len(offsets) * resolution**2 > MAX_GRID_POINTS:
        raise ValueError("Slices must have distinct offsets and at most 524288 combined grid points.")
    _cancel(cancel_check)
    request = _request(record, request)
    if isinstance(dataset, str):
        dataset = next((item for item in source_catalog(record, request=request) if item.key == dataset), None)
    if not isinstance(dataset, FieldDataset):
        raise ValueError("Select an available saved FieldDataset.")
    row, scenario = _scenario(dataset, request)
    config_values = dict((request or {}).get("config") or record.metadata.get("config", {}))
    from dataclasses import fields, replace

    config = CoilGenConfig(
        **{key: value for key, value in config_values.items() if key in {item.name for item in fields(CoilGenConfig)}}
    )
    config = replace(config, parallel_worker_count=1, progress_flag=False)
    kind = dataset.provenance.get("physical_representation", "samples")
    source, fallback_reason, mask_source = None, "", None
    target_function = _target_evaluator(record, dataset, request, row, scenario)
    _check_target_recipe(record, dataset, target_function, row)
    if dataset.key == "target" and target_function is not None and representation != "interpolation":
        kind, source, chosen = "target", target_function, "owned_analytic_target"
    elif representation != "interpolation" and kind in {"sheet", "wire"}:
        try:
            _cancel(cancel_check)
            source = (
                _sheet_source(record, config, scenario)
                if kind == "sheet"
                else _wire_source(record, dataset, row, scenario)
            )
            mask_source = (kind, source)
            source_count = len(source[1]) if kind == "sheet" else sum(len(path) - 1 for path, _ in source)
            if source_count * len(offsets) * resolution**2 > MAX_PHYSICAL_INTERACTIONS:
                raise ValueError(
                    "Physical reevaluation exceeds the bounded interaction budget; reduce slice count/resolution."
                )
            # Reproduce a few original map values before associating new physical
            # evaluations with this particular saved representation/current.
            check_indices = np.unique(
                np.linspace(0, len(dataset.points_m) - 1, min(24, len(dataset.points_m)), dtype=int)
            )
            check_field, valid = _physical(dataset.points_m[check_indices], kind, source, config, cancel_check)
            scale = max(float(np.max(np.abs(dataset.field_T[check_indices]))), 1e-15)
            if not np.any(valid):
                raise ValueError(
                    "No nonsingular original samples are available to verify this physical representation."
                )
            if not np.allclose(
                check_field[valid], dataset.field_T[check_indices][valid], rtol=1e-8, atol=scale * 1e-8 + 1e-15
            ):
                raise ValueError("Saved geometry/current does not reproduce the selected map representation.")
            chosen = "direct_sheet" if kind == "sheet" else "direct_wire"
        except EvidenceIntegrityError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            if representation == "direct":
                raise ValueError(f"Direct saved-field sampling unavailable: {exc}") from exc
            fallback_reason, source = str(exc), None
    else:
        chosen = "display_interpolation"
    if representation == "interpolation" and kind in {"sheet", "wire"}:
        try:
            saved_source = (
                _sheet_source(record, config, scenario)
                if kind == "sheet"
                else _wire_source(record, dataset, row, scenario)
            )
            mask_source = (kind, saved_source)
        except EvidenceIntegrityError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            fallback_reason = "Physical singularity mask unavailable: " + str(exc)
    if source is None:
        if representation == "direct":
            raise ValueError("This source has no saved physical reevaluation contract.")
        chosen = "display_interpolation"
        interpolator = _Interpolator(dataset.points_m, dataset.field_T, interpolation_max_gap_m)
    else:
        interpolator = None
    _cancel(cancel_check)
    geometry = dict(dataset.target_geometry) if dataset.target_geometry is not None else None
    roi_representation = "owned_geometry" if geometry is not None else "sampling_hull"
    enclosure, roi_reason = None, ""
    region = TargetRegion.from_dict(geometry.get("region", geometry)) if geometry is not None else None
    if region is not None and region.shape == "enclosing_mesh":
        try:
            enclosure = _owned_enclosure(record, geometry)
        except EvidenceIntegrityError:
            raise
        except (ValueError, OSError) as exc:
            roi_reason, roi_representation = str(exc), "unavailable_owned_enclosure"
    # A convex sampling hull is a coverage overlay, never an authored target volume.
    hull = _Interpolator(dataset.points_m, dataset.field_T, interpolation_max_gap_m) if geometry is None else None
    target_interpolator = None
    if dataset.target_T is not None and target_function is None:
        target_interpolator = _Interpolator(dataset.points_m, dataset.target_T, interpolation_max_gap_m)
    target_representation = (
        "owned_analytic"
        if target_function is not None
        else "display_interpolation"
        if target_interpolator is not None
        else "unavailable"
    )
    horizontal, vertical, normal = _PLANES[plane]
    u = np.linspace(centre[horizontal] - size[0] / 2, centre[horizontal] + size[0] / 2, resolution)
    v = np.linspace(centre[vertical] - size[1] / 2, centre[vertical] + size[1] / 2, resolution)
    uv = np.meshgrid(u, v, indexing="xy")
    slices = []
    if progress is not None:
        progress(0, f"Sampling {dataset.label} · {chosen}")
    for index, offset in enumerate(offsets):
        _cancel(cancel_check)
        coordinates = np.empty((resolution, resolution, 3))
        coordinates[:, :, horizontal], coordinates[:, :, vertical], coordinates[:, :, normal] = uv[0], uv[1], offset
        points = coordinates.reshape(-1, 3)
        if not np.all(np.isfinite(points)):
            raise ValueError("Slice centre and size overflow finite world coordinates.")
        if kind == "target" and source is not None:
            actual = source(points)
            valid = np.all(np.isfinite(actual), axis=1)
        elif source is not None:

            def sampled(fraction):
                if progress is not None:
                    progress(100 * (index + fraction) / len(offsets), f"Sampling slice {index + 1}/{len(offsets)}")

            actual, valid = _physical(points, kind, source, config, cancel_check, sampled)
        else:
            actual, valid = interpolator.sample(points)
            if mask_source is not None:
                mask_kind, saved_source = mask_source
                valid &= (
                    _sheet_mask(points, saved_source[0], cancel_check)
                    if mask_kind == "sheet"
                    else _wire_mask(points, saved_source, cancel_check)
                )
                actual[~valid] = np.nan
        if geometry is not None:
            try:
                if region.shape == "enclosing_mesh":
                    if enclosure is None:
                        raise ValueError(roi_reason)
                    roi = (
                        _enclosure_mask(points, enclosure, cancel_check).reshape(resolution, resolution)
                        if region.sampling == "volume"
                        else ~_sheet_mask(points, CoilPart(enclosure), cancel_check).reshape(resolution, resolution)
                    )
                    outlines = _mesh_section(enclosure.vertices, enclosure.faces, normal, offset, cancel_check)
                else:
                    level = _region_level(points, geometry).reshape(resolution, resolution)
                    roi = level <= 64 * np.finfo(float).eps
                    outlines = _marching_outline(coordinates, level, cancel_check)
            except ValueError as exc:
                roi, outlines, roi_representation, roi_reason = None, (), "unavailable_owned_enclosure", str(exc)
        else:
            _, coverage = hull.sample(points)
            roi = coverage.reshape(resolution, resolution)
            outlines = _marching_outline(coordinates, np.where(roi, -1.0, 1.0), cancel_check)
        desired = None
        target_valid = np.zeros(len(points), bool)
        if target_function is not None:
            desired = target_function(points)
            target_valid = np.all(np.isfinite(desired), axis=1)
            if roi is not None:
                target_valid &= roi.ravel()
            elif geometry is not None:
                target_valid[:] = False
        elif target_interpolator is not None:
            desired, target_valid = target_interpolator.sample(points)
            if geometry is not None and roi is not None:
                target_valid &= roi.ravel()
            elif geometry is not None:
                target_valid[:] = False
        if desired is not None:
            desired[~target_valid] = np.nan
        context = ()
        if "vertices_m" in record.array_names and "faces" in record.array_names:
            context = _mesh_section(
                _support_points(record.array("vertices_m", copy=False), scenario),
                record.array("faces", copy=False),
                normal,
                offset,
                cancel_check,
            )
        slices.append(
            FieldSlice(
                float(offset),
                coordinates,
                actual.reshape(resolution, resolution, 3),
                desired.reshape(resolution, resolution, 3) if desired is not None else None,
                valid.reshape(resolution, resolution),
                target_valid.reshape(resolution, resolution),
                roi,
                outlines,
                context,
                {
                    "unavailable_field_samples": int(np.count_nonzero(~valid)),
                    "unavailable_target_samples": int(np.count_nonzero(~target_valid)),
                },
            )
        )
        if progress is not None:
            progress(100 * (index + 1) / len(offsets), f"Sampled slice {index + 1}/{len(offsets)}")
    provenance = {
        "request_sha256": record.metadata.get("request_sha256"),
        "source_scope": dataset.scope,
        "source_key": dataset.key,
        "analysis_current_refitted": False,
        "saved_current_refitted": dataset.provenance.get("current_refitted"),
        "current_policy": dataset.provenance.get("current_policy"),
        "current_source": dataset.provenance.get("current_source", dataset.provenance.get("current_A")),
        "source_provenance": dataset.provenance,
        "representation": chosen,
        "target_representation": target_representation,
        "roi_representation": roi_representation,
        "fallback_reason": fallback_reason,
        "sampling": "fresh_regular_display_grid",
        "independent_validation_claim": False,
        "interpolation_extrapolation": False,
        "interpolation_gap_limit_m": interpolator.max_gap_m if interpolator else None,
        "roi_outline": "piecewise_linear_slice_of_owned_geometry"
        if geometry
        else "sampling_coverage_hull_with_gap_mask",
        "roi_sampling": region.sampling if region is not None else "saved_point_cloud",
        "target_domain_scope": "authored expression inside target boundary; original constraints sampled on surface"
        if region is not None
        and region.sampling == "surface"
        and region.shape != "enclosing_mesh"
        and target_function is not None
        else "owned boundary samples only"
        if region is not None and region.sampling == "surface"
        else "owned target volume"
        if region is not None
        else "saved sampling hull",
        "roi_unavailable_reason": roi_reason,
        "component": dataset.provenance.get("component", "Bz"),
    }
    return SliceCollection(
        dataset.key, dataset.label, plane, tuple(slices), chosen, target_representation, roi_representation, provenance
    )
