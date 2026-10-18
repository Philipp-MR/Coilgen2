from __future__ import annotations

import ast
import csv
import json
import math
import re
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .config import CoilGenConfig
from .design_context import design_context_key
from .field_units import field_values_to_tesla
from .models import Mesh, TargetField
from .object_registry import get_in_memory_target_field, is_in_memory_target_key


def _solid_harmonic_parameters(degree, order, radius, center):
    """Validate the explicit convention used by analytic harmonic targets."""
    if isinstance(degree, bool) or not isinstance(degree, Integral) or not 0 <= degree <= 16:
        raise ValueError("Solid harmonic degree must be an integer from 0 to 16.")
    if isinstance(order, bool) or not isinstance(order, Integral) or not -degree <= order <= degree:
        raise ValueError("Solid harmonic order must be an integer between -degree and degree.")
    if isinstance(radius, bool) or not isinstance(radius, Real) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Solid harmonic reference radius must be finite and strictly positive [m].")
    try:
        center = tuple(center)
    except TypeError as exc:
        raise ValueError("Solid harmonic center must contain three finite coordinates [m].") from exc
    if len(center) != 3 or any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in center
    ):
        raise ValueError("Solid harmonic center must contain three finite coordinates [m].")
    return int(degree), int(order), float(radius), tuple(float(value) for value in center)


def solid_harmonic(degree, order, x, y, z, radius, center_x=0.0, center_y=0.0, center_z=0.0):
    """Evaluate one normalized real solid harmonic with explicit SI geometry.

    Uses the same real-Ylm normalization and Condon--Shortley phase as
    ``solid_spherical_harmonic_basis``: the dimensionless term is
    (r/radius)**degree * real_Ylm. A multiplier in tesla is an orthonormal
    harmonic coefficient, rather than the pole amplitude. Degrees 0--16
    are supported; degree, order, radius and center must be scalar constants.
    """
    from .spherical_harmonics import solid_spherical_harmonic_basis

    degree, order, radius, center = _solid_harmonic_parameters(degree, order, radius, (center_x, center_y, center_z))
    coordinates = np.broadcast_arrays(x, y, z)
    if any(np.iscomplexobj(value) for value in coordinates):
        raise ValueError("Solid harmonic coordinates must be finite real values [m].")
    points = np.column_stack([np.asarray(value, dtype=float).reshape(-1) for value in coordinates])
    if not np.all(np.isfinite(points)):
        raise ValueError("Solid harmonic coordinates must be finite real values [m].")
    if not len(points):
        return np.empty(coordinates[0].shape, dtype=float)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            basis, _, _, _ = solid_spherical_harmonic_basis(points, degree, center=center, radius=radius)
        except FloatingPointError as exc:
            raise ValueError("Solid harmonic geometry exceeds the finite numerical range.") from exc
    return basis[:, degree * degree + degree + order].reshape(coordinates[0].shape)


_ALLOWED_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "abs": np.abs,
    "solid_harmonic": solid_harmonic,
    "pi": math.pi,
}


def define_target_field(
    coil_meshes: list[Mesh],
    target_mesh: Mesh | None,
    secondary_target_mesh: Mesh | None,
    config: CoilGenConfig,
) -> TargetField:
    if config.target_field_definition_file != "none":
        return load_target_field_definition(config.target_field_definition_file, secondary_target_mesh, config)

    if config.target_region_geometry is not None:
        from .target_regions import sample_target_region

        target_points = sample_target_region(
            config.target_region_geometry, config.target_region_resolution, mesh=target_mesh
        )
    elif target_mesh is not None:
        target_points = _target_points_from_mesh(target_mesh, config)
    else:
        target_points = _spherical_target_points(config)
        if config.set_roi_into_mesh_center:
            all_vertices = np.vstack([mesh.vertices for mesh in coil_meshes])
            # Mesh and field-expression coordinates are in the world frame.
            # Move the origin-centred ROI to the mesh, preserving that frame.
            target_points = target_points + all_vertices.mean(axis=0)

    target_points = _unique_rows_stable(target_points)
    target_b = np.zeros((len(target_points), 3), dtype=float)
    target_b[:, 2] = evaluate_field_expression(config.field_shape_function, target_points)

    suppressed = np.zeros(len(target_points), dtype=bool)
    suppression_normals = None
    suppression_area_weights = None
    if secondary_target_mesh is not None:
        suppress_points = secondary_target_mesh.vertices
        secondary_normals, secondary_area_weights = _secondary_target_surface_metadata(secondary_target_mesh)
        suppression_normals = np.vstack((np.zeros((len(target_points), 3), dtype=float), secondary_normals))
        suppression_area_weights = np.concatenate((np.zeros(len(target_points), dtype=float), secondary_area_weights))
        target_points = np.vstack((target_points, suppress_points))
        target_b = np.vstack((target_b, np.zeros((len(suppress_points), 3), dtype=float)))
        suppressed = np.concatenate((suppressed, np.ones(len(suppress_points), dtype=bool)))

    gradient = symbolic_gradient_fallback(config.field_shape_function, target_points)
    target_b, gradient, target_diagnostics = _normalise_target_gradient_field(
        target_points,
        target_b,
        gradient,
        suppressed,
        config,
        source="expression",
    )
    target_b = _finite_target_array(field_values_to_tesla(target_b, config.target_field_units), "Target b field")
    gradient = field_values_to_tesla(gradient, config.target_field_units)
    weights = np.ones(len(target_points), dtype=float)
    weights[suppressed] = config.secondary_target_weight
    group_inds = np.ones(len(target_points), dtype=np.int64)
    group_inds[suppressed] = 2
    return TargetField(
        coords=target_points,
        b=target_b,
        weights=weights,
        suppressed=suppressed,
        target_field_group_inds=group_inds,
        target_gradient_dbdxyz=gradient,
        suppression_normals=suppression_normals,
        suppression_area_weights=suppression_area_weights,
        target_diagnostics=target_diagnostics,
    )


def load_target_field_definition(
    path: str | Path,
    secondary_target_mesh: Mesh | None,
    config: CoilGenConfig,
) -> TargetField:
    if is_in_memory_target_key(path):
        target = get_in_memory_target_field(str(path))
        return _append_secondary_target_mesh(target, secondary_target_mesh, config)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target field definition file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        # Owned assets retain the already resolved SI target exactly, including
        # optional gradients and suppression-surface data. Re-normalizing these
        # would change the problem when freezing a process-local input.
        with np.load(path, allow_pickle=False) as loaded:
            if "schema" in loaded:
                if str(loaded["schema"].item()) != "coilgen-target-v1":
                    raise ValueError("Unsupported CoilGen target asset schema.")
                values = {
                    key: np.array(loaded[key], copy=True)
                    for key in (
                        "coords",
                        "b",
                        "weights",
                        "suppressed",
                        "target_field_group_inds",
                        "target_gradient_dbdxyz",
                        "suppression_normals",
                        "suppression_area_weights",
                    )
                    if key in loaded
                }
                required = {"coords", "b", "weights", "suppressed", "target_field_group_inds"}
                if not required.issubset(values):
                    raise ValueError("Owned target asset lacks required numerical arrays.")
                values.update(_validate_target_data(values))
                values["target_diagnostics"] = json.loads(str(loaded["target_diagnostics_json"].item()))
                if not isinstance(values["target_diagnostics"], dict):
                    raise ValueError("Owned target diagnostics must be a JSON object.")
                return _append_secondary_target_mesh(TargetField(**values), secondary_target_mesh, config)
        data = _load_target_npz(path)
    elif suffix == ".json":
        data = _load_target_json(path)
    elif suffix in {".csv", ".tsv"}:
        data = _load_target_table(
            path,
            delimiter="\t" if suffix == ".tsv" else ",",
            field_name=config.target_field_definition_field_name,
        )
    else:
        raise NotImplementedError(
            "Target-field files are supported as .npz, .json, .csv, or .tsv. "
            "MAT-file loading requires scipy and is not enabled in this dependency-light port."
        )

    coords = data["coords"]
    target_b = data["b"]
    suppressed = data.get("suppressed", np.zeros(len(coords), dtype=bool))
    weights = data.get("weights", np.ones(len(coords), dtype=float))
    group_inds = data.get("target_field_group_inds", np.ones(len(coords), dtype=np.int64))
    gradient = data.get("target_gradient_dbdxyz")

    target_b, gradient, target_diagnostics = _normalise_target_gradient_field(
        coords,
        target_b,
        gradient,
        suppressed,
        config,
        source="file",
    )
    target_b = _finite_target_array(field_values_to_tesla(target_b, config.target_field_units), "Target b field")
    if gradient is None:
        gradient = np.zeros((len(coords), 3), dtype=float)
    else:
        gradient = _finite_target_array(field_values_to_tesla(gradient, config.target_field_units), "Target gradient")
    return _append_secondary_target_mesh(
        TargetField(
            coords=coords,
            b=target_b,
            weights=weights,
            suppressed=suppressed,
            target_field_group_inds=group_inds,
            target_gradient_dbdxyz=gradient,
            suppression_normals=data.get("suppression_normals"),
            suppression_area_weights=data.get("suppression_area_weights"),
            target_diagnostics=target_diagnostics,
        ),
        secondary_target_mesh,
        config,
    )


def _append_secondary_target_mesh(
    target_field: TargetField,
    secondary_target_mesh: Mesh | None,
    config: CoilGenConfig,
) -> TargetField:
    if secondary_target_mesh is None:
        return target_field
    suppress_points = secondary_target_mesh.vertices
    secondary_normals, secondary_area_weights = _secondary_target_surface_metadata(secondary_target_mesh)
    group_index = (
        int(np.max(target_field.target_field_group_inds)) + 1 if len(target_field.target_field_group_inds) else 2
    )
    gradient = target_field.target_gradient_dbdxyz
    if gradient is not None:
        gradient = np.vstack((gradient, np.zeros((len(suppress_points), 3), dtype=float)))
    existing_normals = target_field.suppression_normals
    if existing_normals is None:
        existing_normals = np.zeros((len(target_field.coords), 3), dtype=float)
    existing_area_weights = target_field.suppression_area_weights
    if existing_area_weights is None:
        existing_area_weights = np.zeros(len(target_field.coords), dtype=float)
    return TargetField(
        coords=np.vstack((target_field.coords, suppress_points)),
        b=np.vstack((target_field.b, np.zeros((len(suppress_points), 3), dtype=float))),
        weights=np.concatenate(
            (target_field.weights, np.full(len(suppress_points), float(config.secondary_target_weight)))
        ),
        suppressed=np.concatenate((target_field.suppressed, np.ones(len(suppress_points), dtype=bool))),
        target_field_group_inds=np.concatenate(
            (target_field.target_field_group_inds, np.full(len(suppress_points), group_index, dtype=np.int64))
        ),
        target_gradient_dbdxyz=gradient,
        suppression_normals=np.vstack((existing_normals, secondary_normals)),
        suppression_area_weights=np.concatenate((existing_area_weights, secondary_area_weights)),
        target_diagnostics=dict(getattr(target_field, "target_diagnostics", {})),
    )


def _secondary_target_surface_metadata(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    normals = _mesh_vertex_normals(mesh)
    area_weights = _mesh_vertex_area_weights(mesh)
    return normals, area_weights


def _mesh_vertex_normals(mesh: Mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if mesh.vertex_normals is not None and np.asarray(mesh.vertex_normals).shape == vertices.shape:
        normals = np.asarray(mesh.vertex_normals, dtype=float).copy()
    else:
        normals = np.zeros_like(vertices)
        face_normals = _mesh_face_normals(mesh)
        for face, normal in zip(np.asarray(mesh.faces, dtype=np.int64), face_normals):
            normals[face] += normal
    lengths = np.linalg.norm(normals, axis=1)
    safe = np.isfinite(lengths) & (lengths > 1.0e-15)
    out = np.zeros_like(vertices)
    out[safe] = normals[safe] / lengths[safe, None]
    return out


def _mesh_face_normals(mesh: Mesh) -> np.ndarray:
    if mesh.face_normals is not None and np.asarray(mesh.face_normals).shape == (mesh.n_faces, 3):
        normals = np.asarray(mesh.face_normals, dtype=float).copy()
    else:
        tri = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces, dtype=np.int64)]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    safe = np.isfinite(lengths) & (lengths > 1.0e-15)
    out = np.zeros_like(normals)
    out[safe] = normals[safe] / lengths[safe, None]
    return out


def _mesh_vertex_area_weights(mesh: Mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    weights = np.zeros(len(vertices), dtype=float)
    if len(faces) == 0:
        return np.ones(len(vertices), dtype=float)
    tri = vertices[faces]
    face_areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = np.isfinite(face_areas) & (face_areas > 0.0)
    for face, area in zip(faces[valid], face_areas[valid]):
        weights[face] += float(area) / 3.0
    positive = weights[np.isfinite(weights) & (weights > 0.0)]
    if positive.size == 0:
        return np.ones(len(vertices), dtype=float)
    fallback = float(np.median(positive))
    weights[~np.isfinite(weights) | (weights <= 0.0)] = fallback
    return weights


def _load_target_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        coords = _required_array(loaded, ("coords", "target_points", "target_coords"), 3)
        b_field = _required_array(loaded, ("b", "target_b", "target_field"), 3)
        data = {"coords": coords, "b": b_field}
        for key in (
            "weights",
            "suppressed",
            "target_field_group_inds",
            "target_gradient_dbdxyz",
            "suppression_normals",
            "suppression_area_weights",
        ):
            if key in loaded:
                data[key] = np.asarray(loaded[key])
        return _validate_target_data(data)


def _load_target_json(path: Path) -> dict[str, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    coords = np.asarray(raw.get("coords", raw.get("target_points", raw.get("target_coords"))), dtype=float)
    b_field = np.asarray(raw.get("b", raw.get("target_b", raw.get("target_field"))), dtype=float)
    data = {"coords": coords, "b": b_field}
    for key in (
        "weights",
        "suppressed",
        "target_field_group_inds",
        "target_gradient_dbdxyz",
        "suppression_normals",
        "suppression_area_weights",
    ):
        if key in raw:
            data[key] = np.asarray(raw[key])
    return _validate_target_data(data)


def _load_target_table(path: Path, delimiter: str, field_name: str = "none") -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
    if not rows or reader.fieldnames is None:
        raise ValueError("Target CSV/TSV must contain a header and at least one row.")

    columns = _normalized_column_lookup(reader.fieldnames)
    x_col = _required_column(columns, ("x", "coord_x", "target_x", "x_m", "xm"))
    y_col = _required_column(columns, ("y", "coord_y", "target_y", "y_m", "ym"))
    z_col = _required_column(columns, ("z", "coord_z", "target_z", "z_m", "zm"))
    vector_cols, scalar_bz_col = _target_field_columns(columns, field_name)

    coords = np.asarray([[float(row[x_col]), float(row[y_col]), float(row[z_col])] for row in rows])
    if vector_cols is not None:
        bx_col, by_col, bz_col = vector_cols
        b_field = np.asarray([[float(row[bx_col]), float(row[by_col]), float(row[bz_col])] for row in rows])
    else:
        b_field = np.zeros((len(rows), 3), dtype=float)
        b_field[:, 2] = np.asarray([float(row[scalar_bz_col]) for row in rows], dtype=float)
    data: dict[str, np.ndarray] = {"coords": coords, "b": b_field}
    weight_col = _optional_column(columns, ("weight", "weights", "target_weight"))
    if weight_col is not None:
        data["weights"] = np.asarray([float(row[weight_col]) for row in rows], dtype=float)
    suppressed_col = _optional_column(columns, ("suppressed", "suppress", "is_suppressed", "secondary"))
    if suppressed_col is not None:
        data["suppressed"] = np.asarray([_parse_bool(row[suppressed_col]) for row in rows], dtype=bool)
    group_col = _optional_column(columns, ("group", "group_ind", "group_index", "target_field_group_inds"))
    if group_col is not None:
        data["target_field_group_inds"] = np.asarray([int(row[group_col]) for row in rows], dtype=np.int64)
    gradient_cols = _optional_gradient_columns(columns)
    if gradient_cols is not None:
        gx_col, gy_col, gz_col = gradient_cols
        data["target_gradient_dbdxyz"] = np.asarray(
            [[float(row[gx_col]), float(row[gy_col]), float(row[gz_col])] for row in rows],
            dtype=float,
        )
    return _validate_target_data(data)


def _normalized_column_lookup(fieldnames: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in fieldnames:
        normalized = _normalize_column_name(name)
        lookup.setdefault(normalized, name)
    return lookup


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _required_column(columns: dict[str, str], aliases: tuple[str, ...]) -> str:
    column = _optional_column(columns, aliases)
    if column is None:
        raise ValueError(f"Target CSV/TSV is missing a required column matching one of: {', '.join(aliases)}.")
    return column


def _optional_column(columns: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        column = columns.get(_normalize_column_name(alias))
        if column is not None:
            return column
    return None


def _target_field_columns(columns: dict[str, str], field_name: str) -> tuple[tuple[str, str, str] | None, str]:
    vector = tuple(_optional_field_column(columns, field_name, axis) for axis in ("x", "y", "z"))
    if all(column is not None for column in vector):
        return (vector[0], vector[1], vector[2]), vector[2]  # type: ignore[return-value]
    if any(column is not None for column in vector):
        missing = [axis for axis, column in zip(("x", "y", "z"), vector) if column is None]
        if "z" not in missing:
            return None, vector[2]  # type: ignore[return-value]
        raise ValueError(
            "Target CSV/TSV vector field columns must provide bx, by, and bz. "
            "For scalar target maps, provide only a named Bz column such as lin_y or main_bz."
        )
    scalar = _optional_scalar_bz_column(columns, field_name)
    if scalar is not None:
        return None, scalar
    label = f"named field '{field_name}'" if str(field_name).strip().lower() not in {"", "none"} else "field"
    raise ValueError(
        f"Target CSV/TSV is missing {label} columns. Expected bx/by/bz vector columns or one scalar Bz map column."
    )


def _optional_field_column(columns: dict[str, str], field_name: str, axis: str) -> str | None:
    field_name = str(field_name).strip()
    base_aliases = (
        f"b{axis}",
        f"b_{axis}",
        f"target_b{axis}",
        f"target_b_{axis}",
        f"target_field_{axis}",
        f"field_{axis}",
        f"b{axis}_t",
    )
    if field_name and field_name.lower() != "none":
        prefixed_aliases = (
            f"{field_name}_b{axis}",
            f"{field_name}_b_{axis}",
            f"{field_name}.b{axis}",
            f"{field_name}:b{axis}",
            f"{field_name}_field_{axis}",
            f"{field_name}.{axis}",
            f"{field_name}:{axis}",
            f"b{axis}_{field_name}",
            f"b_{axis}_{field_name}",
            f"field_{field_name}_{axis}",
            f"{axis}_{field_name}",
        )
        column = _optional_column(columns, prefixed_aliases + base_aliases)
    else:
        column = _optional_column(columns, base_aliases)
    return column


def _optional_scalar_bz_column(columns: dict[str, str], field_name: str) -> str | None:
    field_name = str(field_name).strip()
    base_aliases = (
        "bz",
        "b_z",
        "target_bz",
        "target_b_z",
        "target_field",
        "target_field_z",
        "field",
        "field_z",
        "scalar_bz",
    )
    if field_name and field_name.lower() != "none":
        aliases = (
            field_name,
            f"{field_name}_bz",
            f"{field_name}_b_z",
            f"{field_name}.bz",
            f"{field_name}:bz",
            f"{field_name}_field",
            f"{field_name}_target_field",
            f"bz_{field_name}",
            f"b_z_{field_name}",
            f"field_{field_name}",
            f"target_field_{field_name}",
        ) + base_aliases
    else:
        aliases = base_aliases
    return _optional_column(columns, aliases)


def _optional_gradient_columns(columns: dict[str, str]) -> tuple[str, str, str] | None:
    aliases = (
        ("dbdx", "dbzdx", "d_bz_dx", "gradient_x", "target_gradient_x", "target_gradient_dbdx"),
        ("dbdy", "dbzdy", "d_bz_dy", "gradient_y", "target_gradient_y", "target_gradient_dbdy"),
        ("dbdz", "dbzdz", "d_bz_dz", "gradient_z", "target_gradient_z", "target_gradient_dbdz"),
    )
    resolved = [_optional_column(columns, axis_aliases) for axis_aliases in aliases]
    if all(column is not None for column in resolved):
        return resolved[0], resolved[1], resolved[2]  # type: ignore[return-value]
    if any(column is not None for column in resolved):
        raise ValueError("Target CSV/TSV gradient columns must provide all three dBz/dx,dBz/dy,dBz/dz components.")
    return None


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Could not parse boolean value {value!r}.")


def _required_array(loaded, names: tuple[str, ...], shape_last: int) -> np.ndarray:
    for name in names:
        if name in loaded:
            value = _finite_target_array(loaded[name], name)
            if value.ndim != 2 or value.shape[1] != shape_last:
                raise ValueError(f"{name} must have shape (n, {shape_last}).")
            return value
    raise ValueError(f"Target definition is missing one of: {', '.join(names)}.")


def _validate_target_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate real physical samples before normalization or mask coercion.

    Zero weights deliberately disable samples. Zero surface metadata is valid
    for ordinary fitting points; normals are not required to be unit vectors
    here. Group numbers are labels, so zero/negative integer labels are valid.
    """
    coords = _finite_target_array(data["coords"], "Target coords")
    b_field = _finite_target_array(data["b"], "Target b field")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Target coords must have shape (n, 3).")
    if b_field.ndim != 2 or b_field.shape[1] != 3 or len(b_field) != len(coords):
        raise ValueError("Target b field must have shape (n, 3) and match coords.")
    normalized: dict[str, np.ndarray] = {"coords": coords, "b": b_field}
    if "weights" in data:
        weights = _finite_target_array(data["weights"], "Target weights").reshape(-1)
        if len(weights) != len(coords):
            raise ValueError("Target weights must match coords length.")
        if np.any(weights < 0.0):
            raise ValueError("Target weights must be nonnegative; use zero to disable a sample.")
        normalized["weights"] = weights
    if "suppressed" in data:
        flags = _finite_target_array(data["suppressed"], "Target suppressed flags").reshape(-1)
        if np.any((flags != 0.0) & (flags != 1.0)):
            raise ValueError("Target suppressed flags must be booleans or numeric 0/1 values.")
        suppressed = flags.astype(bool)
        if len(suppressed) != len(coords):
            raise ValueError("Target suppressed flags must match coords length.")
        normalized["suppressed"] = suppressed
    if "target_field_group_inds" in data:
        raw_groups = np.asarray(data["target_field_group_inds"]).reshape(-1)
        if raw_groups.dtype.kind in "biu":
            if raw_groups.dtype.kind == "u" and np.any(raw_groups >= np.uint64(2**63)):
                raise ValueError("Target group indices must fit in signed 64-bit integers.")
            group_inds = raw_groups.astype(np.int64)
        else:
            groups = _finite_target_array(raw_groups, "Target group indices")
            if np.any(groups != np.floor(groups)):
                raise ValueError("Target group indices must be integer labels.")
            if np.any(groups >= float(2**63)) or np.any(groups < float(-(2**63))):
                raise ValueError("Target group indices must fit in signed 64-bit integers.")
            group_inds = groups.astype(np.int64)
        if len(group_inds) != len(coords):
            raise ValueError("Target group indices must match coords length.")
        normalized["target_field_group_inds"] = group_inds
    if "target_gradient_dbdxyz" in data:
        gradient = _finite_target_array(data["target_gradient_dbdxyz"], "Target gradient")
        if gradient.shape != coords.shape:
            raise ValueError("Target gradient must have shape (n, 3).")
        normalized["target_gradient_dbdxyz"] = gradient
    if "suppression_normals" in data:
        normals = _finite_target_array(data["suppression_normals"], "Suppression normals")
        if normals.shape != coords.shape:
            raise ValueError("Suppression normals must have shape (n, 3).")
        normalized["suppression_normals"] = normals
    if "suppression_area_weights" in data:
        areas = _finite_target_array(data["suppression_area_weights"], "Suppression area weights").reshape(-1)
        if len(areas) != len(coords):
            raise ValueError("Suppression area weights must match coords length.")
        if np.any(areas < 0.0):
            raise ValueError("Suppression area weights must be nonnegative.")
        normalized["suppression_area_weights"] = areas
    return normalized


def _finite_target_array(value, label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must contain real values.")
    try:
        array = np.asarray(value, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must contain real finite numerical values.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain finite values.")
    return array


def _target_points_from_mesh(mesh: Mesh, config: CoilGenConfig) -> np.ndarray:
    if config.use_only_target_mesh_verts:
        return mesh.vertices.copy()
    mins = mesh.vertices.min(axis=0)
    maxs = mesh.vertices.max(axis=0)
    resolution = int(config.target_region_resolution)
    axes = [np.linspace(mins[i], maxs[i], resolution) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="xy")
    grid_points = np.column_stack([axis.reshape(-1) for axis in grid])
    return np.vstack((mesh.vertices, grid_points))


def _spherical_target_points(config: CoilGenConfig) -> np.ndarray:
    resolution = int(config.target_region_resolution)
    radius = float(config.target_region_radius)
    coords = np.linspace(-2.0 * radius, 2.0 * radius, resolution)
    grid = np.meshgrid(coords, coords, coords, indexing="xy")
    points = np.column_stack([axis.reshape(-1) for axis in grid])
    tolerance = 32.0 * np.finfo(float).eps * max(abs(radius), 1.0)
    inside = np.linalg.norm(points, axis=1) <= radius + tolerance
    return points[inside]


def _unique_rows_stable(points: np.ndarray) -> np.ndarray:
    seen: set[tuple[float, float, float]] = set()
    rows: list[np.ndarray] = []
    for point in points:
        key = (float(point[0]), float(point[1]), float(point[2]))
        if key not in seen:
            seen.add(key)
            rows.append(point)
    return np.asarray(rows, dtype=float)


def evaluate_field_expression(expression: str, points: np.ndarray) -> np.ndarray:
    expression = _normalize_field_expression(expression)
    local_scope = dict(_ALLOWED_FUNCTIONS)
    local_scope.update({"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]})
    try:
        if "solid_harmonic" in expression:
            # Python callables expose module globals through attributes. Keep
            # this new expression facility numerical without changing the
            # evaluation semantics of existing expression targets.
            allowed_nodes = (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Name,
                ast.Load,
                ast.Constant,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.Mod,
                ast.FloorDiv,
                ast.UAdd,
                ast.USub,
                ast.Compare,
                ast.Eq,
                ast.NotEq,
                ast.Lt,
                ast.LtE,
                ast.Gt,
                ast.GtE,
                ast.BitAnd,
                ast.BitOr,
                ast.BitXor,
                ast.Invert,
            )
            for node in ast.walk(ast.parse(expression, mode="eval")):
                if (
                    not isinstance(node, allowed_nodes)
                    or isinstance(node, ast.Name)
                    and node.id not in local_scope
                    or isinstance(node, ast.Call)
                    and (not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS)
                    or isinstance(node, ast.Constant)
                    and (not isinstance(node.value, Real) or not math.isfinite(node.value))
                ):
                    raise ValueError(
                        "Solid harmonic expressions support finite numerical functions and arithmetic only."
                    )
        values = eval(expression, {"__builtins__": {}}, local_scope)
    except Exception as exc:
        raise ValueError(f"Could not evaluate field_shape_function {expression!r}.") from exc
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(len(points), float(values))
    return values


def _normalize_field_expression(expression: str) -> str:
    expression = str(expression).strip()
    expression = re.sub(r"\.\s*\^", "**", expression)
    expression = re.sub(r"\.\s*([*/])", r"\1", expression)
    expression = expression.replace("^", "**")
    expression = re.sub(r"(?<=\d)(?=[xyz])", "*", expression)
    expression = re.sub(r"(?<=[xyz])(?=[xyz])", "*", expression)
    expression = re.sub(r"(?<=[xyz])(?=\()", "*", expression)
    expression = re.sub(r"(?<=\))(?=[xyz])", "*", expression)
    return expression


def _normalise_target_gradient_field(
    points: np.ndarray,
    target_b: np.ndarray,
    gradient: np.ndarray | None,
    suppressed: np.ndarray,
    config: CoilGenConfig,
    *,
    source: str,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    points = np.asarray(points, dtype=float)
    target_b = np.asarray(target_b, dtype=float)
    suppressed = np.asarray(suppressed, dtype=bool).reshape(-1)
    gradient_values = None if gradient is None else np.asarray(gradient, dtype=float)
    method = _target_gradient_normalization_method(config, source=source)
    requested = _positive_float(getattr(config, "target_gradient_strength", 1.0), default=1.0)
    raw_fit = _fit_bz_linear_gradient(points, target_b, suppressed)
    axis = _target_gradient_axis(config, raw_fit, source=source)
    raw_strength = _gradient_strength_for_axis(raw_fit, axis)
    gradient_tolerance = _linear_fit_gradient_tolerance(points, target_b, suppressed)
    fallback = False

    if method == "none":
        scale = 1.0
    elif method == "linear_fit":
        if np.isfinite(raw_strength) and abs(raw_strength) > gradient_tolerance:
            scale = requested / abs(raw_strength)
        else:
            scale = requested
            fallback = True
        if gradient_values is None and np.all(np.isfinite(raw_fit)):
            gradient_values = np.tile(raw_fit, (len(points), 1))
    elif method == "legacy_extrema":
        scale, fallback = _legacy_target_gradient_scale(points, target_b, requested)
    else:
        scale = requested

    if not np.isfinite(scale):
        scale = 1.0
        fallback = True
    scaled_b = target_b * float(scale)
    scaled_gradient = None if gradient_values is None else gradient_values * float(scale)
    diagnostics = _target_gradient_normalization_diagnostics(
        config,
        source=source,
        method=method,
        axis=axis,
        requested=requested,
        scale=float(scale),
        raw_fit=raw_fit,
        raw_strength=raw_strength,
        fallback=fallback,
    )
    return scaled_b, scaled_gradient, diagnostics


def _linear_fit_gradient_tolerance(
    points: np.ndarray,
    target_b: np.ndarray,
    suppressed: np.ndarray,
) -> float:
    """Return a scale-aware floor below which a fitted gradient is numerical noise.

    A constant target can acquire a tiny nonzero slope from the affine least-
    squares fit. Dividing the requested gradient strength by that slope would
    amplify the constant field catastrophically. The floor is tied to field and
    coordinate scales, with a conservative allowance for the fit conditioning.
    """

    points = np.asarray(points, dtype=float)
    target_b = np.asarray(target_b, dtype=float)
    suppressed = np.asarray(suppressed, dtype=bool).reshape(-1)
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(target_b[:, 2])
    if len(suppressed) == len(points):
        finite &= ~suppressed
    if not np.any(finite):
        return 1.0e-30
    selected_points = points[finite]
    centered = selected_points - np.mean(selected_points, axis=0, keepdims=True)
    coordinate_scale = float(np.max(np.linalg.norm(centered, axis=1)))
    field_scale = float(np.max(np.abs(target_b[finite, 2])))
    if coordinate_scale <= 0.0 or not np.isfinite(coordinate_scale):
        return max(field_scale, 1.0) * np.finfo(float).eps * 100.0
    return max(field_scale, 1.0e-300) * np.finfo(float).eps * 100.0 / coordinate_scale


def _target_gradient_normalization_method(config: CoilGenConfig, *, source: str) -> str:
    raw = str(getattr(config, "target_gradient_normalization", "auto")).strip().lower().replace("-", "_")
    if raw in {"", "auto"}:
        if (
            design_context_key(getattr(config, "design_context", "mr_gradient")) == "mr_gradient"
            and source == "expression"
        ):
            return "linear_fit"
        return "scale"
    if raw in {"linear", "linear_fit", "fit", "gradient_fit"}:
        return "linear_fit"
    if raw in {"legacy", "legacy_extrema", "extrema", "max_min"}:
        return "legacy_extrema"
    if raw in {"none", "off", "disabled"}:
        return "none"
    return "scale"


def _target_gradient_axis(config: CoilGenConfig, raw_fit: np.ndarray, *, source: str) -> str:
    raw = str(getattr(config, "target_gradient_axis", "auto")).strip().lower().replace("-", "_")
    if raw.startswith("g") and len(raw) == 2:
        raw = raw[1]
    if raw in {"x", "y", "z", "magnitude", "norm"}:
        return "magnitude" if raw == "norm" else raw
    if source == "expression":
        expression_axis = _axis_from_expression(getattr(config, "field_shape_function", ""))
        if expression_axis is not None:
            return expression_axis
    if np.all(np.isfinite(raw_fit)) and np.linalg.norm(raw_fit) > 1.0e-30:
        return ("x", "y", "z")[int(np.argmax(np.abs(raw_fit)))]
    return "magnitude"


def _axis_from_expression(expression: object) -> str | None:
    text = _normalize_field_expression(str(expression)).replace(" ", "").lower()
    if text in {"x", "+x", "1*x", "+1*x"}:
        return "x"
    if text in {"y", "+y", "1*y", "+1*y"}:
        return "y"
    if text in {"z", "+z", "1*z", "+1*z"}:
        return "z"
    return None


def _gradient_strength_for_axis(gradient: np.ndarray, axis: str) -> float:
    gradient = np.asarray(gradient, dtype=float).reshape(3)
    if axis == "x":
        return float(gradient[0])
    if axis == "y":
        return float(gradient[1])
    if axis == "z":
        return float(gradient[2])
    return float(np.linalg.norm(gradient))


def _fit_bz_linear_gradient(points: np.ndarray, target_b: np.ndarray, suppressed: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    target_b = np.asarray(target_b, dtype=float)
    suppressed = np.asarray(suppressed, dtype=bool).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(target_b):
        return np.zeros(3, dtype=float)
    bz = target_b[:, 2]
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(bz)
    if len(suppressed) == len(points):
        finite &= ~suppressed
    if np.count_nonzero(finite) < 2:
        return np.zeros(3, dtype=float)
    fit_points = points[finite]
    fit_bz = bz[finite]
    # Separate the affine intercept before fitting gradients. Large world
    # offsets otherwise make [1,x,y,z] numerically rank deficient and can mix
    # the field offset into its derivatives. Lstsq retains the minimum-norm
    # gradient for spatially rank-deficient samples.
    fit_points = fit_points - np.mean(fit_points, axis=0, keepdims=True)
    fit_bz = fit_bz - np.mean(fit_bz)
    try:
        coefficients, *_ = np.linalg.lstsq(fit_points, fit_bz, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=float)
    gradient = np.asarray(coefficients, dtype=float)
    if gradient.shape != (3,) or not np.all(np.isfinite(gradient)):
        return np.zeros(3, dtype=float)
    return gradient


def _legacy_target_gradient_scale(
    points: np.ndarray,
    target_b: np.ndarray,
    requested: float,
) -> tuple[float, bool]:
    z_field = np.asarray(target_b, dtype=float)[:, 2]
    if len(z_field) == 0:
        return 1.0, True
    max_ind = int(np.argmax(z_field))
    min_ind = int(np.argmin(z_field))
    max_target_distance = float(np.linalg.norm(points[max_ind] - points[min_ind]))
    max_field_difference = float(z_field[max_ind] - z_field[min_ind])
    if abs(max_field_difference) > 1e-10 and max_target_distance > 0.0:
        return float(requested / (max_field_difference / max_target_distance)), False
    return float(requested), True


def _target_gradient_normalization_diagnostics(
    config: CoilGenConfig,
    *,
    source: str,
    method: str,
    axis: str,
    requested: float,
    scale: float,
    raw_fit: np.ndarray,
    raw_strength: float,
    fallback: bool,
) -> dict[str, object]:
    raw_fit = np.asarray(raw_fit, dtype=float).reshape(3)
    units = getattr(config, "target_field_units", "T")
    requested_T = _field_scalar_to_tesla(requested, units)
    raw_fit_T = field_values_to_tesla(raw_fit, units)
    raw_strength_T = _field_scalar_to_tesla(raw_strength, units)
    effective_T = _field_scalar_to_tesla(raw_strength * scale, units)
    return {
        "target_gradient_normalization_source": str(source),
        "target_gradient_normalization_method": str(method),
        "target_gradient_axis": str(axis),
        "target_gradient_requested_input": float(requested),
        "target_gradient_requested_T_per_m": float(abs(requested_T)),
        "target_gradient_raw_fit_x_T_per_m": float(raw_fit_T[0]),
        "target_gradient_raw_fit_y_T_per_m": float(raw_fit_T[1]),
        "target_gradient_raw_fit_z_T_per_m": float(raw_fit_T[2]),
        "target_gradient_raw_strength_T_per_m": float(abs(raw_strength_T)),
        "target_gradient_normalization_scale": float(scale),
        "target_gradient_effective_T_per_m": float(abs(effective_T)),
        "target_gradient_normalization_fallback": bool(fallback),
    }


def _field_scalar_to_tesla(value: float, units: object) -> float:
    return float(field_values_to_tesla(np.asarray([float(value)], dtype=float), units)[0])


def _positive_float(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number) or number <= 0.0:
        return float(default)
    return float(number)


def symbolic_gradient_fallback(expression: str, points: np.ndarray) -> np.ndarray:
    eps = 1e-6
    gradient = np.zeros((len(points), 3), dtype=float)
    for dim in range(3):
        plus = points.copy()
        minus = points.copy()
        plus[:, dim] += eps
        minus[:, dim] -= eps
        gradient[:, dim] = (
            evaluate_field_expression(expression, plus) - evaluate_field_expression(expression, minus)
        ) / (2 * eps)
    return gradient
