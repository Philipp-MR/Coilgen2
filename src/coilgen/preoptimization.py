from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .basis import current_density_from_basis
from .config import CoilGenConfig
from .geometry import combine_meshes
from .models import CoilPart, Mesh, TargetField


def has_preoptimized_stream_function(config: CoilGenConfig) -> bool:
    return preoptimized_stream_function_source(config) != "none"


def preoptimized_stream_function_source(config: CoilGenConfig) -> str:
    source = str(getattr(config, "preoptimized_stream_function_file", "none")).strip()
    if source and source.lower() != "none":
        return source
    matlab_alias = str(getattr(config, "sf_source_file", "none")).strip()
    return matlab_alias if matlab_alias else "none"


def apply_preoptimized_stream_function(
    parts: list[CoilPart],
    target_field: TargetField,
    config: CoilGenConfig,
    project_root: Path,
) -> tuple[list[CoilPart], Mesh, np.ndarray, dict[str, Any]]:
    """Load stream-function values and continue with the Python layout pipeline.

    The imported stream function is assumed to be defined on the already loaded
    coil mesh vertices. CSV/TSV files may specify values by part-local
    ``vertex_index`` or by coordinates. JSON/NPZ files may provide a single
    global vector in combined-mesh order or one vector per mesh part.
    """

    source = _resolve_preoptimized_path(preoptimized_stream_function_source(config), project_root)
    stream_functions = load_preoptimized_stream_function_values(source, parts, config)
    for part, stream_function in zip(parts, stream_functions):
        part.stream_function = np.asarray(stream_function, dtype=float).reshape(-1)
        if part.current_density_mat is not None:
            part.current_density = np.tensordot(part.stream_function, part.current_density_mat, axes=(0, 0))
        else:
            part.current_density = current_density_from_basis(part)

    combined_mesh = combine_meshes([part.mesh for part in parts])
    combined_mesh.stream_function = np.concatenate([part.stream_function for part in parts])
    sf_b_field = _evaluate_stream_function_field(parts, target_field)
    diagnostics = {
        "selection_method": "preoptimized",
        "candidate_solver": "preoptimized",
        "fallback": False,
        "fallback_reason": "",
        "selected_index": 0,
        "selected_factor": float("nan"),
        "selected_lambda": float("nan"),
        "candidate_factors": [],
        "candidate_lambdas": [],
        "residual_norms": [],
        "regularization_norms": [],
        "curvature": [],
        "stream_function_scheme": "preoptimized",
        "stream_function_cost_function": "preoptimized_stream_function",
        "cost_function_formula": "loaded stream_function values; no inverse solve performed",
        "preoptimized_stream_function_file": str(source),
        "preoptimized_stream_function_field_name": str(config.preoptimized_stream_function_field_name),
        "unknown_count": int(sum(part.mesh.n_vertices for part in parts)),
        "target_sample_count": int(len(target_field.coords)),
    }
    return parts, combined_mesh, sf_b_field, diagnostics


def load_preoptimized_stream_function_values(
    path: str | Path,
    parts: list[CoilPart],
    config: CoilGenConfig | None = None,
) -> list[np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preoptimized stream-function file not found: {path}")
    field_name = "stream_function" if config is None else str(config.preoptimized_stream_function_field_name)
    tolerance = 1e-8 if config is None else float(config.preoptimized_stream_function_match_tolerance)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_preoptimized_json(path, parts, field_name)
    if suffix == ".npz":
        return _load_preoptimized_npz(path, parts, field_name, tolerance)
    if suffix in {".csv", ".tsv"}:
        return _load_preoptimized_table(path, parts, field_name, tolerance, "\t" if suffix == ".tsv" else ",")
    raise NotImplementedError("Preoptimized stream functions are supported as .csv, .tsv, .json, or .npz files.")


def _resolve_preoptimized_path(source: str, project_root: Path) -> Path:
    path = Path(source)
    if path.is_absolute() or path.exists():
        return path
    candidates = [
        project_root / path,
        project_root / "Pre_Optimized_Solutions" / path,
        project_root / "preoptimized_stream_functions" / path,
        project_root / "outputs" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_preoptimized_json(path: Path, parts: list[CoilPart], field_name: str) -> list[np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "parts" in raw:
        part_values: list[np.ndarray | None] = [None] * len(parts)
        for entry in raw["parts"]:
            part_index = _normalize_single_index(int(entry.get("part", entry.get("part_index", 1))), len(parts), "part")
            values = _entry_stream_function(entry, field_name)
            part_values[part_index] = np.asarray(values, dtype=float).reshape(-1)
        return _validate_part_stream_functions(part_values, parts)
    values = np.asarray(_entry_stream_function(raw, field_name), dtype=float).reshape(-1)
    return _split_global_stream_function(values, parts)


def _load_preoptimized_npz(
    path: Path,
    parts: list[CoilPart],
    field_name: str,
    tolerance: float,
) -> list[np.ndarray]:
    with np.load(path) as loaded:
        key = field_name if field_name in loaded else "stream_function"
        if key not in loaded:
            raise ValueError(f"NPZ preoptimized stream function is missing '{field_name}' or 'stream_function'.")
        values = np.asarray(loaded[key], dtype=float).reshape(-1)
        if "part" in loaded or "part_index" in loaded or "vertex_index" in loaded or "global_vertex_index" in loaded:
            table = {
                "stream_function": values,
                "part": np.asarray(
                    loaded["part"] if "part" in loaded else loaded["part_index"] if "part_index" in loaded else []
                ),
                "vertex_index": np.asarray(loaded["vertex_index"] if "vertex_index" in loaded else []),
                "global_vertex_index": np.asarray(
                    loaded["global_vertex_index"] if "global_vertex_index" in loaded else []
                ),
            }
            return _assign_table_arrays(table, parts, tolerance)
        return _split_global_stream_function(values, parts)


def _load_preoptimized_table(
    path: Path,
    parts: list[CoilPart],
    field_name: str,
    tolerance: float,
    delimiter: str,
) -> list[np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
    if not rows or reader.fieldnames is None:
        raise ValueError("Preoptimized stream-function CSV/TSV must contain a header and at least one row.")
    columns = _normalized_column_lookup(reader.fieldnames)
    sf_col = _stream_function_column(columns, field_name)
    part_col = _optional_column(columns, ("part", "part_index", "coil_part", "coil_part_index"))
    local_col = _optional_column(
        columns, ("vertex_index", "node_index", "local_vertex_index", "index", "vertex", "node")
    )
    global_col = _optional_column(columns, ("global_vertex_index", "global_node_index", "global_index"))
    coord_cols = _optional_coordinate_columns(columns)

    values = np.asarray([float(row[sf_col]) for row in rows], dtype=float)
    total_vertices = sum(part.mesh.n_vertices for part in parts)
    if part_col is None and local_col is None and global_col is None and coord_cols is None:
        if len(values) != total_vertices:
            raise ValueError(
                "Stream-function CSV without part/index/coordinate columns must contain exactly "
                f"{total_vertices} rows in combined vertex order."
            )
        return _split_global_stream_function(values, parts)

    part_values = [np.full(part.mesh.n_vertices, np.nan, dtype=float) for part in parts]
    assigned = [np.zeros(part.mesh.n_vertices, dtype=bool) for part in parts]
    global_indices = None
    if global_col is not None:
        global_indices = _normalize_indices(
            [int(row[global_col]) for row in rows], total_vertices, "global_vertex_index"
        )
    part_indices = None
    if part_col is not None:
        part_indices = _normalize_indices([int(row[part_col]) for row in rows], len(parts), "part")
    local_indices = None
    if local_col is not None and part_col is not None:
        raw_local = np.asarray([int(row[local_col]) for row in rows], dtype=np.int64)
        local_indices = np.zeros(len(rows), dtype=np.int64)
        for part_ind in np.unique(part_indices):
            mask = part_indices == part_ind
            local_indices[mask] = _normalize_indices(
                raw_local[mask],
                parts[int(part_ind)].mesh.n_vertices,
                "vertex_index",
            )
    elif local_col is not None and len(parts) == 1:
        local_indices = _normalize_indices(
            [int(row[local_col]) for row in rows], parts[0].mesh.n_vertices, "vertex_index"
        )

    offsets = np.cumsum([0] + [part.mesh.n_vertices for part in parts])
    for row_ind, row in enumerate(rows):
        if global_indices is not None:
            part_ind = int(np.searchsorted(offsets[1:], global_indices[row_ind], side="right"))
            local_ind = int(global_indices[row_ind] - offsets[part_ind])
        elif local_indices is not None:
            part_ind = int(part_indices[row_ind]) if part_indices is not None else 0
            local_ind = int(local_indices[row_ind])
        elif coord_cols is not None:
            xyz = np.array([float(row[col]) for col in coord_cols], dtype=float)
            part_ind, local_ind = _nearest_mesh_vertex(
                parts, xyz, tolerance, None if part_indices is None else int(part_indices[row_ind])
            )
        else:
            raise ValueError(
                "Stream-function CSV needs vertex_index, global_vertex_index, or x/y/z coordinate columns."
            )
        part_values[part_ind][local_ind] = values[row_ind]
        assigned[part_ind][local_ind] = True

    missing = [int(np.count_nonzero(~mask)) for mask in assigned]
    if any(count > 0 for count in missing):
        raise ValueError(
            f"Preoptimized stream-function file does not cover every mesh vertex; missing per part: {missing}."
        )
    return part_values


def _assign_table_arrays(table: dict[str, np.ndarray], parts: list[CoilPart], tolerance: float) -> list[np.ndarray]:
    values = np.asarray(table["stream_function"], dtype=float).reshape(-1)
    total_vertices = sum(part.mesh.n_vertices for part in parts)
    global_indices = np.asarray(table.get("global_vertex_index", []), dtype=np.int64).reshape(-1)
    if len(global_indices):
        global_indices = _normalize_indices(global_indices, total_vertices, "global_vertex_index")
        part_values = [np.full(part.mesh.n_vertices, np.nan, dtype=float) for part in parts]
        offsets = np.cumsum([0] + [part.mesh.n_vertices for part in parts])
        for global_ind, value in zip(global_indices, values):
            part_ind = int(np.searchsorted(offsets[1:], global_ind, side="right"))
            local_ind = int(global_ind - offsets[part_ind])
            part_values[part_ind][local_ind] = value
        return _validate_no_missing(part_values)

    part_indices = np.asarray(table.get("part", []), dtype=np.int64).reshape(-1)
    vertex_indices = np.asarray(table.get("vertex_index", []), dtype=np.int64).reshape(-1)
    if len(part_indices) and len(vertex_indices):
        part_indices = _normalize_indices(part_indices, len(parts), "part")
        normalized_vertex_indices = np.zeros(len(vertex_indices), dtype=np.int64)
        for part_ind in np.unique(part_indices):
            mask = part_indices == part_ind
            normalized_vertex_indices[mask] = _normalize_indices(
                vertex_indices[mask],
                parts[int(part_ind)].mesh.n_vertices,
                "vertex_index",
            )
        part_values = [np.full(part.mesh.n_vertices, np.nan, dtype=float) for part in parts]
        for part_ind, local_ind, value in zip(part_indices, normalized_vertex_indices, values):
            part_values[int(part_ind)][int(local_ind)] = value
        return _validate_no_missing(part_values)
    return _split_global_stream_function(values, parts)


def _entry_stream_function(entry: dict[str, Any], field_name: str) -> Any:
    if field_name in entry:
        return entry[field_name]
    for key in ("stream_function", "sf", "psi"):
        if key in entry:
            return entry[key]
    raise ValueError(f"Preoptimized stream-function JSON is missing field '{field_name}' or 'stream_function'.")


def _split_global_stream_function(values: np.ndarray, parts: list[CoilPart]) -> list[np.ndarray]:
    total_vertices = sum(part.mesh.n_vertices for part in parts)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(parts) == 1 and len(values) == parts[0].mesh.n_vertices:
        return [values.copy()]
    if len(values) != total_vertices:
        raise ValueError(f"Expected {total_vertices} stream-function values, got {len(values)}.")
    split = []
    offset = 0
    for part in parts:
        split.append(values[offset : offset + part.mesh.n_vertices].copy())
        offset += part.mesh.n_vertices
    return split


def _validate_part_stream_functions(values: list[np.ndarray | None], parts: list[CoilPart]) -> list[np.ndarray]:
    if any(value is None for value in values):
        missing = [ind + 1 for ind, value in enumerate(values) if value is None]
        raise ValueError(f"Preoptimized stream-function JSON is missing parts: {missing}.")
    out = [np.asarray(value, dtype=float).reshape(-1) for value in values if value is not None]
    for part_ind, (stream_function, part) in enumerate(zip(out, parts), start=1):
        if len(stream_function) != part.mesh.n_vertices:
            raise ValueError(
                f"Part {part_ind} stream-function length {len(stream_function)} does not match "
                f"mesh vertex count {part.mesh.n_vertices}."
            )
    return out


def _validate_no_missing(values: list[np.ndarray]) -> list[np.ndarray]:
    missing = [int(np.count_nonzero(~np.isfinite(part_values))) for part_values in values]
    if any(count > 0 for count in missing):
        raise ValueError(
            f"Preoptimized stream-function file does not cover every mesh vertex; missing per part: {missing}."
        )
    return values


def _evaluate_stream_function_field(parts: list[CoilPart], target_field: TargetField) -> np.ndarray:
    field = np.zeros((len(target_field.coords), 3), dtype=float)
    for part in parts:
        if part.sensitivity_matrix is None or part.stream_function is None:
            raise ValueError("Sensitivity matrix and stream_function are required to evaluate a preoptimized field.")
        field[:, 0] += part.sensitivity_matrix[0] @ part.stream_function
        field[:, 1] += part.sensitivity_matrix[1] @ part.stream_function
        field[:, 2] += part.sensitivity_matrix[2] @ part.stream_function
    return field


def _nearest_mesh_vertex(
    parts: list[CoilPart],
    point: np.ndarray,
    tolerance: float,
    forced_part: int | None,
) -> tuple[int, int]:
    candidates = [forced_part] if forced_part is not None else list(range(len(parts)))
    best_part = -1
    best_vertex = -1
    best_distance = np.inf
    for part_ind in candidates:
        distances = np.linalg.norm(parts[part_ind].mesh.vertices - point, axis=1)
        vertex_ind = int(np.argmin(distances))
        distance = float(distances[vertex_ind])
        if distance < best_distance:
            best_part = part_ind
            best_vertex = vertex_ind
            best_distance = distance
    if best_distance > tolerance:
        raise ValueError(
            f"Could not match stream-function coordinate {point.tolist()} to a mesh vertex "
            f"within tolerance {tolerance:g}; nearest distance is {best_distance:g}."
        )
    return best_part, best_vertex


def _normalize_single_index(value: int, size: int, label: str) -> int:
    return int(_normalize_indices([value], size, label)[0])


def _normalize_indices(values: Any, size: int, label: str) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return indices
    if np.any(indices < 0):
        raise ValueError(f"{label} values must be non-negative or one-based positive indices.")
    if np.any(indices == 0):
        normalized = indices
    else:
        normalized = indices - 1
    if np.any(normalized < 0) or np.any(normalized >= size):
        raise ValueError(f"{label} values are outside the valid range for size {size}.")
    return normalized


def _normalized_column_lookup(fieldnames: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in fieldnames:
        lookup.setdefault(_normalize_column_name(name), name)
    return lookup


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _optional_column(columns: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        column = columns.get(_normalize_column_name(alias))
        if column is not None:
            return column
    return None


def _stream_function_column(columns: dict[str, str], field_name: str) -> str:
    field = str(field_name).strip()
    aliases = ("stream_function", "streamfunction", "sf", "psi", "potential")
    if field and field.lower() != "none":
        column = _optional_column(columns, (field, f"{field}_stream_function", f"{field}_sf", f"{field}_psi") + aliases)
    else:
        column = _optional_column(columns, aliases)
    if column is None:
        raise ValueError("Preoptimized stream-function table is missing a stream_function/sf/psi column.")
    return column


def _optional_coordinate_columns(columns: dict[str, str]) -> tuple[str, str, str] | None:
    resolved = [
        _optional_column(columns, ("x", "x_m", "xm", "coord_x", "vertex_x")),
        _optional_column(columns, ("y", "y_m", "ym", "coord_y", "vertex_y")),
        _optional_column(columns, ("z", "z_m", "zm", "coord_z", "vertex_z")),
    ]
    if all(column is not None for column in resolved):
        return resolved[0], resolved[1], resolved[2]  # type: ignore[return-value]
    if any(column is not None for column in resolved):
        raise ValueError("Coordinate-based stream-function matching requires all three x/y/z columns.")
    return None
