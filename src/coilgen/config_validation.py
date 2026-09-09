from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import CoilGenConfig
from .design_context import CoilApplication, normalize_design_context
from .engineering_limits import LEVEL_COUNT_METHODS, ENGINEERING_METHODS, contour_discretization_method
from .field_components import normalize_field_component
from .field_units import normalize_field_units
from .object_registry import is_in_memory_mesh_key, is_in_memory_target_key


@dataclass(frozen=True, slots=True)
class ConfigValidationMessage:
    level: str
    field: str
    message: str
    suggestion: str = ""

    def format(self) -> str:
        text = f"[{self.level.upper()}] {self.field}: {self.message}"
        return f"{text} Suggestion: {self.suggestion}" if self.suggestion else text


@dataclass(frozen=True, slots=True)
class ConfigValidationReport:
    messages: tuple[ConfigValidationMessage, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ConfigValidationMessage, ...]:
        return tuple(message for message in self.messages if message.level == "error")

    @property
    def warnings(self) -> tuple[ConfigValidationMessage, ...]:
        return tuple(message for message in self.messages if message.level == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Invalid CoilGen configuration:\n" + self.format(levels=("error",)))

    def format(self, *, levels: Iterable[str] = ("error", "warning")) -> str:
        allowed = set(levels)
        lines = [message.format() for message in self.messages if message.level in allowed]
        return "\n".join(lines)


def validate_config(config: CoilGenConfig, *, project_root: str | Path | None = None) -> ConfigValidationReport:
    """Validate a CoilGenConfig before numerical execution.

    The checks deliberately focus on parameters that are common sources of
    unclear failures for experimental users: physical units, mesh sizes,
    option names, file formats, and incompatible import modes.
    """

    messages: list[ConfigValidationMessage] = []

    def error(field: str, message: str, suggestion: str = "") -> None:
        messages.append(ConfigValidationMessage("error", field, message, suggestion))

    def warning(field: str, message: str, suggestion: str = "") -> None:
        messages.append(ConfigValidationMessage("warning", field, message, suggestion))

    _validate_mesh_settings(config, error, warning, project_root=project_root)
    _validate_target_settings(config, error, warning)
    _validate_design_context(config, error, warning)
    _validate_optimization_settings(config, error, warning)
    if config.research_physics is not None:
        from .research.convex import ResearchPhysicsSpec

        try:
            ResearchPhysicsSpec.from_dict(config.research_physics)
        except (TypeError, ValueError) as exc:
            error("research_physics", str(exc))
        if any(
            str(value).strip().lower() not in {"", "none"}
            for value in (config.preoptimized_stream_function_file, config.sf_source_file)
        ):
            error(
                "research_physics",
                "Physical optimization cannot be combined with a preoptimized stream-function source; "
                "that source bypasses optimization.",
            )
    _validate_routing_settings(config, error, warning)
    _validate_performance_settings(config, error, warning)
    _validate_manufacturing_settings(config, error, warning)
    return ConfigValidationReport(tuple(messages))


def _validate_design_context(config: CoilGenConfig, error, warning) -> None:
    try:
        context = normalize_design_context(getattr(config, "design_context", "mr_gradient"))
    except ValueError as exc:
        error("design_context", str(exc))
        return
    if context.application == CoilApplication.MR_RF:
        warning(
            "design_context",
            "MR RF context is recognized, but the current CoilGen pipeline evaluates magnetostatic fields only.",
            "Use this context to suppress gradient/shim diagnostics until RF-specific evaluation is implemented.",
        )


def _validate_mesh_settings(config: CoilGenConfig, error, warning, *, project_root: str | Path | None) -> None:
    mesh_name = str(config.coil_mesh_file).strip()
    generated_meshes = {
        "create cylinder mesh",
        "create planary mesh",
        "create circular mesh",
        "create annular disc mesh",
        "create partial cylinder mesh",
        "create elliptical cylinder mesh",
        "create superellipse cylinder mesh",
        "create spherical cap mesh",
        "create ellipsoidal cap mesh",
        "create conical frustum mesh",
        "create toroidal segment mesh",
        "create ruled surface mesh",
        "create surface of revolution mesh",
        "create circular bi-planary mesh",
        "create bi-planary mesh",
        "create shielded bi-planary mesh",
        "create flanged split bore mesh",
        "create shielded flanged split bore mesh",
    }
    if not mesh_name or mesh_name == "none":
        error("coil_mesh_file", "No coil surface is configured.", "Choose a generated geometry or an STL mesh.")
    elif is_in_memory_mesh_key(mesh_name):
        pass
    elif mesh_name in generated_meshes:
        _validate_generated_mesh_parameters(config, error)
    elif not mesh_name.lower().endswith(".stl"):
        error(
            "coil_mesh_file",
            f"Unknown mesh source {mesh_name!r}.",
            "Use a supported generated mesh name or an STL filename.",
        )
    elif project_root is not None:
        mesh_path = config.resolved_geometry_source_path(Path(project_root)) / mesh_name
        if not mesh_path.exists():
            warning(
                "coil_mesh_file", f"STL file was not found at {mesh_path}.", "Check geometry_source_path or filename."
            )

    if int(config.iteration_num_mesh_refinement) < 0:
        error("iteration_num_mesh_refinement", "Mesh refinement iterations must be non-negative.")
    try:
        imported_mesh_length_scale_to_m = float(config.imported_mesh_length_scale_to_m)
    except (TypeError, ValueError):
        error("imported_mesh_length_scale_to_m", "Imported-mesh length scale must be numeric.")
    else:
        if not np.isfinite(imported_mesh_length_scale_to_m) or imported_mesh_length_scale_to_m <= 0.0:
            error("imported_mesh_length_scale_to_m", "Imported-mesh length scale must be finite and positive.")
    try:
        rotation_angle = float(config.generated_mesh_rotation_angle)
    except (TypeError, ValueError):
        error("generated_mesh_rotation_angle", "Generated-mesh rotation angle must be numeric.")
        rotation_angle = 0.0
    try:
        rotation_axis = np.asarray(config.generated_mesh_rotation_axis, dtype=float)
    except (TypeError, ValueError):
        error("generated_mesh_rotation_axis", "Generated-mesh rotation axis must be numeric.")
    else:
        if rotation_axis.shape != (3,):
            error("generated_mesh_rotation_axis", "Generated-mesh rotation axis must contain exactly three values.")
        elif not np.all(np.isfinite(rotation_axis)):
            error("generated_mesh_rotation_axis", "Generated-mesh rotation axis contains non-finite values.")
        elif abs(rotation_angle) > 1e-15 and np.linalg.norm(rotation_axis) < 1e-15:
            error("generated_mesh_rotation_axis", "Generated-mesh rotation axis must be nonzero when rotation is used.")
    if not np.isfinite(rotation_angle):
        error("generated_mesh_rotation_angle", "Generated-mesh rotation angle must be finite.")
    try:
        b0_direction = np.asarray(config.b_0_direction, dtype=float)
    except (TypeError, ValueError):
        error("b_0_direction", "B0 direction must be numeric.")
    else:
        if b0_direction.shape != (3,):
            error("b_0_direction", "B0 direction must contain exactly three values.")
        elif not np.all(np.isfinite(b0_direction)):
            error("b_0_direction", "B0 direction contains non-finite values.")
        elif np.linalg.norm(b0_direction) < 1e-15:
            error("b_0_direction", "B0 direction must be nonzero.")
    _positive(error, "main B0 field strength", "b_0_strength_T", config.b_0_strength_T)


def _validate_generated_mesh_parameters(config: CoilGenConfig, error) -> None:
    mesh_name = str(config.coil_mesh_file).strip()
    expected_lengths = {
        "create cylinder mesh": ("cylinder_mesh_parameter_list", 8),
        "create planary mesh": ("planar_mesh_parameter_list", 11),
        "create circular mesh": ("circular_mesh_parameter_list", 9),
        "create annular disc mesh": ("annular_disc_mesh_parameter_list", 10),
        "create partial cylinder mesh": ("partial_cylinder_mesh_parameter_list", 13),
        "create elliptical cylinder mesh": ("elliptical_cylinder_mesh_parameter_list", 12),
        "create superellipse cylinder mesh": ("superellipse_cylinder_mesh_parameter_list", 13),
        "create spherical cap mesh": ("spherical_cap_mesh_parameter_list", 11),
        "create ellipsoidal cap mesh": ("ellipsoidal_cap_mesh_parameter_list", 13),
        "create conical frustum mesh": ("conical_frustum_mesh_parameter_list", 12),
        "create toroidal segment mesh": ("toroidal_segment_mesh_parameter_list", 15),
        "create ruled surface mesh": ("ruled_surface_mesh_parameter_list", None),
        "create surface of revolution mesh": ("surface_of_revolution_mesh_parameter_list", None),
        "create circular bi-planary mesh": ("circular_biplanar_mesh_parameter_list", 9),
        "create bi-planary mesh": ("biplanar_mesh_parameter_list", 11),
        "create shielded bi-planary mesh": ("shielded_circular_biplanar_mesh_parameter_list", 12),
        "create flanged split bore mesh": ("flanged_split_bore_mesh_parameter_list", 5),
        "create shielded flanged split bore mesh": ("shielded_flanged_split_bore_mesh_parameter_list", None),
    }
    field_name, expected_len = expected_lengths[mesh_name]
    values = tuple(getattr(config, field_name))
    if mesh_name == "create surface of revolution mesh":
        if len(values) < 5 or (len(values) - 1) % 2 != 0:
            error(
                field_name,
                "Expected [num_circular_divisions, z0, radius0, z1, radius1, ...] with at least two z/radius pairs.",
            )
            return
    elif mesh_name == "create ruled surface mesh":
        if len(values) < 14:
            error(field_name, "Expected [num_curve_samples, num_span_divisions, curve_a_xyz..., curve_b_xyz...].")
            return
        try:
            n_samples = int(values[0])
        except (TypeError, ValueError):
            error(field_name, "Ruled-surface sample count must be an integer.")
            return
        expected = 2 + 6 * n_samples
        if len(values) != expected:
            error(field_name, f"Expected {expected} values for {n_samples} ruled-surface samples, got {len(values)}.")
            return
    elif mesh_name == "create shielded flanged split bore mesh":
        if len(values) not in {6, 10}:
            error(field_name, f"Expected 6 or 10 values for {mesh_name}, got {len(values)}.")
            return
    elif expected_len is not None and len(values) != expected_len:
        error(field_name, f"Expected {expected_len} values for {mesh_name}, got {len(values)}.")
        return
    try:
        numeric_values = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        error(field_name, "Mesh parameter list must contain numeric values.")
        return
    if len(values) and not np.all(np.isfinite(numeric_values)):
        error(field_name, "Mesh parameter list contains non-finite values.")
        return
    if mesh_name == "create cylinder mesh" and len(values) == 8:
        height, radius, circular_divisions, height_divisions, *_ = values
        _positive(error, "cylinder height", "cylinder_mesh_parameter_list[0]", height)
        _positive(error, "cylinder radius", "cylinder_mesh_parameter_list[1]", radius)
        _at_least(error, "circular divisions", "cylinder_mesh_parameter_list[2]", circular_divisions, 3)
        _at_least(error, "longitudinal divisions", "cylinder_mesh_parameter_list[3]", height_divisions, 1)
    if mesh_name == "create planary mesh" and len(values) == 11:
        height, width, lateral_divisions, longitudinal_divisions, *_ = values
        _positive(error, "planar height", "planar_mesh_parameter_list[0]", height)
        _positive(error, "planar width", "planar_mesh_parameter_list[1]", width)
        _at_least(error, "lateral divisions", "planar_mesh_parameter_list[2]", lateral_divisions, 1)
        _at_least(error, "longitudinal divisions", "planar_mesh_parameter_list[3]", longitudinal_divisions, 1)
    if mesh_name == "create circular mesh" and len(values) == 9:
        radius, radial_divisions, *rotation_and_center = values
        _positive(error, "circular disc radius", "circular_mesh_parameter_list[0]", radius)
        _at_least(error, "radial divisions", "circular_mesh_parameter_list[1]", radial_divisions, 2)
        _validate_rotation_vector(
            error, "circular_mesh_parameter_list[2:5]", rotation_and_center[:3], rotation_and_center[3]
        )
    if mesh_name == "create annular disc mesh" and len(values) == 10:
        inner_radius, outer_radius, radial_divisions, circular_divisions, nx, ny, nz, *_center = values
        if float(inner_radius) < 0.0:
            error("annular_disc_mesh_parameter_list[0]", "Annular inner radius must be non-negative.")
        _positive(error, "annular outer radius", "annular_disc_mesh_parameter_list[1]", outer_radius)
        if float(outer_radius) <= float(inner_radius):
            error("annular_disc_mesh_parameter_list[1]", "Annular outer radius must exceed inner radius.")
        _at_least(error, "annular radial divisions", "annular_disc_mesh_parameter_list[2]", radial_divisions, 1)
        _at_least(error, "annular circular divisions", "annular_disc_mesh_parameter_list[3]", circular_divisions, 3)
        if np.linalg.norm(np.asarray([nx, ny, nz], dtype=float)) < 1e-15:
            error("annular_disc_mesh_parameter_list[4:7]", "Annular disc normal vector must be nonzero.")
    if mesh_name == "create partial cylinder mesh" and len(values) == 13:
        height, radius, start_angle, end_angle, circular_divisions, longitudinal_divisions, *rotation_and_center = (
            values
        )
        _positive(error, "partial-cylinder height", "partial_cylinder_mesh_parameter_list[0]", height)
        _positive(error, "partial-cylinder radius", "partial_cylinder_mesh_parameter_list[1]", radius)
        if float(end_angle) <= float(start_angle):
            error("partial_cylinder_mesh_parameter_list[3]", "Partial-cylinder end angle must exceed start angle.")
        if float(end_angle) - float(start_angle) > 2.0 * np.pi + 1e-12:
            error("partial_cylinder_mesh_parameter_list[2:4]", "Partial-cylinder angular span cannot exceed 2*pi.")
        _at_least(
            error,
            "partial-cylinder angular divisions",
            "partial_cylinder_mesh_parameter_list[4]",
            circular_divisions,
            1,
        )
        _at_least(
            error,
            "partial-cylinder longitudinal divisions",
            "partial_cylinder_mesh_parameter_list[5]",
            longitudinal_divisions,
            1,
        )
        _validate_rotation_vector(
            error, "partial_cylinder_mesh_parameter_list[6:9]", rotation_and_center[:3], rotation_and_center[3]
        )
    if mesh_name == "create elliptical cylinder mesh" and len(values) == 12:
        height, radius_x, radius_y, circular_divisions, longitudinal_divisions, *rotation_and_center = values
        _positive(error, "elliptical-cylinder height", "elliptical_cylinder_mesh_parameter_list[0]", height)
        _positive(error, "elliptical-cylinder x radius", "elliptical_cylinder_mesh_parameter_list[1]", radius_x)
        _positive(error, "elliptical-cylinder y radius", "elliptical_cylinder_mesh_parameter_list[2]", radius_y)
        _at_least(
            error,
            "elliptical-cylinder circular divisions",
            "elliptical_cylinder_mesh_parameter_list[3]",
            circular_divisions,
            3,
        )
        _at_least(
            error,
            "elliptical-cylinder longitudinal divisions",
            "elliptical_cylinder_mesh_parameter_list[4]",
            longitudinal_divisions,
            1,
        )
        _validate_rotation_vector(
            error, "elliptical_cylinder_mesh_parameter_list[5:8]", rotation_and_center[:3], rotation_and_center[3]
        )
    if mesh_name == "create superellipse cylinder mesh" and len(values) == 13:
        height, radius_x, radius_y, exponent, circular_divisions, longitudinal_divisions, *rotation_and_center = values
        _positive(error, "superellipse-cylinder height", "superellipse_cylinder_mesh_parameter_list[0]", height)
        _positive(error, "superellipse-cylinder x radius", "superellipse_cylinder_mesh_parameter_list[1]", radius_x)
        _positive(error, "superellipse-cylinder y radius", "superellipse_cylinder_mesh_parameter_list[2]", radius_y)
        if float(exponent) <= 0.5:
            error("superellipse_cylinder_mesh_parameter_list[3]", "Superellipse exponent must be > 0.5.")
        _at_least(
            error,
            "superellipse-cylinder circular divisions",
            "superellipse_cylinder_mesh_parameter_list[4]",
            circular_divisions,
            8,
        )
        _at_least(
            error,
            "superellipse-cylinder longitudinal divisions",
            "superellipse_cylinder_mesh_parameter_list[5]",
            longitudinal_divisions,
            1,
        )
        _validate_rotation_vector(
            error, "superellipse_cylinder_mesh_parameter_list[6:9]", rotation_and_center[:3], rotation_and_center[3]
        )
    if mesh_name == "create spherical cap mesh" and len(values) == 11:
        radius, polar_angle, azimuthal_divisions, polar_divisions, *rotation_and_center = values
        _positive(error, "spherical-cap radius", "spherical_cap_mesh_parameter_list[0]", radius)
        _validate_polar_angle(error, "spherical_cap_mesh_parameter_list[1]", polar_angle)
        _at_least(
            error, "spherical-cap azimuthal divisions", "spherical_cap_mesh_parameter_list[2]", azimuthal_divisions, 3
        )
        _at_least(error, "spherical-cap polar divisions", "spherical_cap_mesh_parameter_list[3]", polar_divisions, 1)
        _validate_rotation_vector(
            error, "spherical_cap_mesh_parameter_list[4:7]", rotation_and_center[:3], rotation_and_center[3]
        )
    if mesh_name == "create ellipsoidal cap mesh" and len(values) == 13:
        radius_x, radius_y, radius_z, polar_angle, azimuthal_divisions, polar_divisions, *rotation_and_center = values
        _positive(error, "ellipsoidal-cap x radius", "ellipsoidal_cap_mesh_parameter_list[0]", radius_x)
        _positive(error, "ellipsoidal-cap y radius", "ellipsoidal_cap_mesh_parameter_list[1]", radius_y)
        _positive(error, "ellipsoidal-cap z radius", "ellipsoidal_cap_mesh_parameter_list[2]", radius_z)
        _validate_polar_angle(error, "ellipsoidal_cap_mesh_parameter_list[3]", polar_angle)
        _at_least(
            error,
            "ellipsoidal-cap azimuthal divisions",
            "ellipsoidal_cap_mesh_parameter_list[4]",
            azimuthal_divisions,
            3,
        )
        _at_least(
            error, "ellipsoidal-cap polar divisions", "ellipsoidal_cap_mesh_parameter_list[5]", polar_divisions, 1
        )
        _validate_rotation_vector(
            error, "ellipsoidal_cap_mesh_parameter_list[6:9]", rotation_and_center[:3], rotation_and_center[3]
        )
    if mesh_name == "create conical frustum mesh" and len(values) == 12:
        height, radius_bottom, radius_top, circular_divisions, longitudinal_divisions, *rotation_and_center = values
        _positive(error, "conical-frustum height", "conical_frustum_mesh_parameter_list[0]", height)
        _positive(error, "conical-frustum bottom radius", "conical_frustum_mesh_parameter_list[1]", radius_bottom)
        _positive(error, "conical-frustum top radius", "conical_frustum_mesh_parameter_list[2]", radius_top)
        _at_least(
            error, "conical-frustum circular divisions", "conical_frustum_mesh_parameter_list[3]", circular_divisions, 3
        )
        _at_least(
            error,
            "conical-frustum longitudinal divisions",
            "conical_frustum_mesh_parameter_list[4]",
            longitudinal_divisions,
            1,
        )
        _validate_rotation_vector(
            error, "conical_frustum_mesh_parameter_list[5:8]", rotation_and_center[:3], rotation_and_center[3]
        )
    if mesh_name == "create toroidal segment mesh" and len(values) == 15:
        (
            major_radius,
            minor_radius,
            start_angle,
            end_angle,
            tube_start,
            tube_end,
            major_divisions,
            minor_divisions,
            *rotation_and_center,
        ) = values
        _positive(error, "toroidal major radius", "toroidal_segment_mesh_parameter_list[0]", major_radius)
        _positive(error, "toroidal minor radius", "toroidal_segment_mesh_parameter_list[1]", minor_radius)
        if float(minor_radius) >= float(major_radius):
            error("toroidal_segment_mesh_parameter_list[1]", "Toroidal minor radius must be smaller than major radius.")
        _validate_angle_span(error, "toroidal_segment_mesh_parameter_list[2:4]", start_angle, end_angle)
        _validate_angle_span(error, "toroidal_segment_mesh_parameter_list[4:6]", tube_start, tube_end)
        _at_least(error, "toroidal major divisions", "toroidal_segment_mesh_parameter_list[6]", major_divisions, 1)
        _at_least(error, "toroidal minor divisions", "toroidal_segment_mesh_parameter_list[7]", minor_divisions, 3)
        _validate_rotation_vector(
            error, "toroidal_segment_mesh_parameter_list[8:11]", rotation_and_center[:3], rotation_and_center[3]
        )
    if mesh_name == "create ruled surface mesh" and len(values) >= 14:
        n_samples, span_divisions, *coords = values
        _at_least(error, "ruled-surface curve samples", "ruled_surface_mesh_parameter_list[0]", n_samples, 2)
        _at_least(error, "ruled-surface span divisions", "ruled_surface_mesh_parameter_list[1]", span_divisions, 1)
        if len(coords) == 6 * int(n_samples):
            curve_a = np.asarray(coords[: 3 * int(n_samples)], dtype=float).reshape(int(n_samples), 3)
            curve_b = np.asarray(coords[3 * int(n_samples) :], dtype=float).reshape(int(n_samples), 3)
            if np.any(np.linalg.norm(curve_a - curve_b, axis=1) < 1e-12):
                error("ruled_surface_mesh_parameter_list[curves]", "Boundary curves must not touch at sampled points.")
    if mesh_name == "create surface of revolution mesh" and len(values) >= 5 and (len(values) - 1) % 2 == 0:
        circular_divisions = values[0]
        pairs = np.asarray(values[1:], dtype=float).reshape(-1, 2)
        _at_least(
            error,
            "surface-of-revolution circular divisions",
            "surface_of_revolution_mesh_parameter_list[0]",
            circular_divisions,
            3,
        )
        if np.any(pairs[:, 1] <= 0.0):
            error("surface_of_revolution_mesh_parameter_list[radii]", "Surface-of-revolution radii must be positive.")
        if np.any(np.diff(pairs[:, 0]) <= 0.0):
            error(
                "surface_of_revolution_mesh_parameter_list[z]",
                "Surface-of-revolution z samples must be strictly increasing.",
            )
    if mesh_name == "create bi-planary mesh" and len(values) == 11:
        height, width, lateral_divisions, longitudinal_divisions, *_rest, plane_distance = values
        _positive(error, "biplanar height", "biplanar_mesh_parameter_list[0]", height)
        _positive(error, "biplanar width", "biplanar_mesh_parameter_list[1]", width)
        _at_least(error, "lateral divisions", "biplanar_mesh_parameter_list[2]", lateral_divisions, 1)
        _at_least(error, "longitudinal divisions", "biplanar_mesh_parameter_list[3]", longitudinal_divisions, 1)
        _positive(error, "plane distance", "biplanar_mesh_parameter_list[10]", plane_distance)
    if mesh_name == "create circular bi-planary mesh" and len(values) == 9:
        edge_len, radius, nx, ny, nz, *_center, plane_distance = values
        _positive(error, "circular biplanar edge length", "circular_biplanar_mesh_parameter_list[0]", edge_len)
        _positive(error, "circular biplanar radius", "circular_biplanar_mesh_parameter_list[1]", radius)
        _positive(error, "circular biplanar plane distance", "circular_biplanar_mesh_parameter_list[8]", plane_distance)
        if np.linalg.norm(np.asarray([nx, ny, nz], dtype=float)) < 1e-15:
            error("circular_biplanar_mesh_parameter_list[2:5]", "Circular biplanar normal vector must be nonzero.")


def _validate_rotation_vector(error, field: str, axis_values: Any, angle_value: Any) -> None:
    try:
        axis = np.asarray(axis_values, dtype=float)
        angle = float(angle_value)
    except (TypeError, ValueError):
        error(field, "Rotation axis and angle must be numeric.")
        return
    if axis.shape != (3,):
        error(field, "Rotation axis must contain exactly three values.")
        return
    if not np.all(np.isfinite(axis)) or not np.isfinite(angle):
        error(field, "Rotation axis and angle must be finite.")
    elif abs(angle) > 1e-15 and np.linalg.norm(axis) < 1e-15:
        error(field, "Rotation axis must be nonzero when rotation angle is nonzero.")


def _validate_polar_angle(error, field: str, value: Any) -> None:
    try:
        angle = float(value)
    except (TypeError, ValueError):
        error(field, "Polar angle must be numeric.")
        return
    if not np.isfinite(angle) or angle <= 0.0 or angle > np.pi:
        error(field, "Polar angle must be finite and in (0, pi].")


def _validate_angle_span(error, field: str, start_value: Any, end_value: Any) -> None:
    try:
        start = float(start_value)
        end = float(end_value)
    except (TypeError, ValueError):
        error(field, "Angular span values must be numeric.")
        return
    if not np.isfinite(start) or not np.isfinite(end):
        error(field, "Angular span values must be finite.")
    elif end <= start:
        error(field, "Angular end must exceed angular start.")
    elif end - start > 2.0 * np.pi + 1e-12:
        error(field, "Angular span cannot exceed 2*pi.")


def _validate_target_settings(config: CoilGenConfig, error, warning) -> None:
    _positive(error, "target gradient strength", "target_gradient_strength", config.target_gradient_strength)
    _enum(
        error,
        "target_gradient_normalization",
        getattr(config, "target_gradient_normalization", "auto"),
        {"auto", "linear_fit", "fit", "gradient_fit", "legacy_extrema", "legacy", "scale", "multiplier", "none", "off"},
    )
    _enum(
        error,
        "target_gradient_axis",
        getattr(config, "target_gradient_axis", "auto"),
        {"auto", "x", "y", "z", "gx", "gy", "gz", "magnitude", "norm"},
    )
    _positive(error, "target region radius", "target_region_radius", config.target_region_radius)
    _at_least(error, "target region resolution", "target_region_resolution", config.target_region_resolution, 1)
    _validate_field_units(error, "target_field_units", config.target_field_units)
    _validate_field_units(error, "field_display_units", config.field_display_units)
    try:
        component = normalize_field_component(config.target_field_component)
    except ValueError as exc:
        error("target_field_component", str(exc))
        component = "Bz"
    if component in {"Bx", "By", "BdotB0", "vector"}:
        warning(
            "target_field_component",
            f"Target component {component} uses generalized field-component rows.",
            "Confirm reports and plots are interpreted with this useful-field convention.",
        )
    if str(config.field_shape_function).strip() == "" and str(config.target_field_definition_file).strip() == "none":
        error("field_shape_function", "Target field expression is empty.", "Use e.g. 'x', 'y', 'z', '1', or a CSV map.")
    target_file = str(config.target_field_definition_file).strip()
    if target_file and target_file.lower() != "none":
        if is_in_memory_target_key(target_file):
            pass
        elif Path(target_file).suffix.lower() not in {".csv", ".tsv", ".json", ".npz"}:
            error(
                "target_field_definition_file",
                "Unsupported target-field file extension.",
                "Use .csv, .tsv, .json, or .npz.",
            )
        if str(config.field_shape_function).strip().lower() != "none":
            warning(
                "field_shape_function",
                "field_shape_function is ignored when target_field_definition_file is set.",
                "Set field_shape_function='none' for clarity.",
            )
    _validate_secondary_suppression_surfaces(config, error)


def _validate_field_units(error, field: str, value) -> None:
    try:
        normalize_field_units(value)
    except ValueError as exc:
        error(field, str(exc), "Use 'T' for tesla or 'Hz' for proton-frequency field maps.")


def _validate_secondary_suppression_surfaces(config: CoilGenConfig, error) -> None:
    enabled = [
        str(config.create_split_bore_field_suppression_surface).strip().lower() == "true",
        str(config.create_flanged_split_bore_field_suppression_surface).strip().lower() == "true",
        str(config.create_circular_biplanar_field_suppression_surface).strip().lower() == "true",
        str(config.create_mesh_topology_field_suppression_surface).strip().lower() == "true",
    ]
    if sum(enabled) > 1:
        error(
            "secondary_target_mesh",
            "Only one generated field-suppression surface can be enabled at a time.",
            "Use either split-bore, flanged split-bore, circular biplanar, topology-derived, or a mesh file.",
        )
    if enabled[0]:
        values = tuple(config.split_bore_area_suppression_parameters)
        _validate_parameter_list(error, "split_bore_area_suppression_parameters", values, 4)
        if len(values) == 4:
            edge_len, radius, gap, half_length = values
            _positive(
                error, "split-bore suppression edge length", "split_bore_area_suppression_parameters[0]", edge_len
            )
            _positive(error, "split-bore suppression radius", "split_bore_area_suppression_parameters[1]", radius)
            if float(gap) < 0.0:
                error("split_bore_area_suppression_parameters[2]", "Split-bore suppression gap must be non-negative.")
            _positive(
                error, "split-bore suppression half length", "split_bore_area_suppression_parameters[3]", half_length
            )
    if enabled[1]:
        values = tuple(config.flanged_split_bore_area_suppression_parameters)
        _validate_parameter_list(error, "flanged_split_bore_area_suppression_parameters", values, 5)
        if len(values) == 5:
            edge_len, inner_radius, outer_radius, gap, half_length = values
            _positive(
                error, "flanged suppression edge length", "flanged_split_bore_area_suppression_parameters[0]", edge_len
            )
            _positive(
                error,
                "flanged suppression inner radius",
                "flanged_split_bore_area_suppression_parameters[1]",
                inner_radius,
            )
            _positive(
                error,
                "flanged suppression outer radius",
                "flanged_split_bore_area_suppression_parameters[2]",
                outer_radius,
            )
            if float(outer_radius) <= float(inner_radius):
                error(
                    "flanged_split_bore_area_suppression_parameters[2]",
                    "Flanged suppression outer radius must exceed inner radius.",
                )
            if float(gap) < 0.0:
                error(
                    "flanged_split_bore_area_suppression_parameters[3]", "Flanged suppression gap must be non-negative."
                )
            _positive(
                error,
                "flanged suppression half length",
                "flanged_split_bore_area_suppression_parameters[4]",
                half_length,
            )
    if enabled[2]:
        values = tuple(config.circular_biplanar_area_suppression_parameters)
        if len(values) not in {3, 9}:
            error(
                "circular_biplanar_area_suppression_parameters",
                "Expected 3 values [edge_len radius spacing] or 9 values [edge_len radius normal_xyz center_xyz spacing].",
            )
        elif len(values) == 3:
            edge_len, radius, spacing = values
            _positive(
                error,
                "circular biplanar suppression edge length",
                "circular_biplanar_area_suppression_parameters[0]",
                edge_len,
            )
            _positive(
                error,
                "circular biplanar suppression radius",
                "circular_biplanar_area_suppression_parameters[1]",
                radius,
            )
            _positive(
                error,
                "circular biplanar suppression spacing",
                "circular_biplanar_area_suppression_parameters[2]",
                spacing,
            )
        else:
            edge_len, radius, nx, ny, nz, *_center, spacing = values
            _positive(
                error,
                "circular biplanar suppression edge length",
                "circular_biplanar_area_suppression_parameters[0]",
                edge_len,
            )
            _positive(
                error,
                "circular biplanar suppression radius",
                "circular_biplanar_area_suppression_parameters[1]",
                radius,
            )
            _positive(
                error,
                "circular biplanar suppression spacing",
                "circular_biplanar_area_suppression_parameters[8]",
                spacing,
            )
            if np.linalg.norm(np.asarray([nx, ny, nz], dtype=float)) < 1e-15:
                error(
                    "circular_biplanar_area_suppression_parameters[2:5]",
                    "Circular biplanar suppression normal vector must be nonzero.",
                )
    if enabled[3]:
        _positive(
            error,
            "mesh-topology suppression separation",
            "mesh_topology_suppression_separation",
            config.mesh_topology_suppression_separation,
        )
        source = str(config.mesh_topology_suppression_source).strip().lower()
        if source not in {
            "shield",
            "all",
            "shield_radial",
            "shield_radial_interface",
            "split_bore_radial_interface",
            "shield_split",
            "shield_split_interface",
            "split_bore_split_interface",
            "shield_radial_split",
            "shield_radial_and_split",
            "split_bore_magnet_interface",
            "all_radial",
            "all_radial_interface",
            "all_split",
            "all_split_interface",
            "all_radial_split",
            "all_radial_and_split",
        }:
            error(
                "mesh_topology_suppression_source",
                "Mesh-topology suppression source must be 'shield', 'all', "
                "'shield_radial', 'shield_split', 'shield_radial_split', "
                "'all_radial', 'all_split', or 'all_radial_split'.",
            )
        clearance_action = str(config.mesh_topology_suppression_clearance_action).strip().lower()
        if clearance_action not in {"none", "report_only", "filter_faces"}:
            error(
                "mesh_topology_suppression_clearance_action",
                "Mesh-topology suppression clearance action must be 'none', 'report_only', or 'filter_faces'.",
            )
        if float(config.mesh_topology_suppression_min_clearance) < 0.0:
            error(
                "mesh_topology_suppression_min_clearance",
                "Mesh-topology suppression minimum clearance must be non-negative.",
            )


def _validate_optimization_settings(config: CoilGenConfig, error, warning) -> None:
    _enum(
        error,
        "regularization_selection_method",
        config.regularization_selection_method,
        {"auto", "l_curve", "fixed"},
    )
    _enum(
        error,
        "l_curve_scan_method",
        config.l_curve_scan_method,
        {"auto", "normal_equations", "normal", "svd", "transformed_svd"},
    )
    _enum(
        error,
        "sf_opt_method",
        config.sf_opt_method,
        {
            "tikhonov",
            "tikkonov",
            "tikhonov_regularized_least_squares",
            "target_error_constrained_efficiency",
            "target-error-constrained-efficiency",
            "efficiency_with_target_error_constraint",
            "fixed_error_efficiency",
            "minimum_resistance_subject_to_field_error",
            "constrained_quadratic",
            "cvxpy_constrained",
            "minimum_resistance_constrained",
            "minimum_resistance_subject_to_field_error_socp",
            "direct_fixed_error",
            "socp",
            "second_order_cone",
        },
    )
    _enum(
        error,
        "stream_function_cost_function",
        config.stream_function_cost_function,
        {"standard", "field_error_plus_resistance", "least_squares_plus_resistance"},
    )
    _enum(error, "optimization_solver", config.optimization_solver, {"auto", "solve", "pinv"})
    _enum(
        error,
        "constrained_quadratic_fallback_policy",
        config.constrained_quadratic_fallback_policy,
        {
            "target_error_scan",
            "target-error-scan",
            "tikhonov_scan",
            "tikhonov-scan",
            "diagnostic",
            "strict",
            "fail",
            "error",
        },
    )
    _enum(
        error,
        "constrained_quadratic_objective_mode",
        getattr(config, "constrained_quadratic_objective_mode", "balanced"),
        {
            "balanced",
            "resistance",
            "resistance_plus_penalties",
            "smoothness_priority",
            "smoothness-priority",
            "smoothest_feasible",
            "contourability_priority",
            "minimax_current_density",
            "minimax-current-density",
        },
    )
    _enum(
        error,
        "target_error_constraint_mode",
        config.target_error_constraint_mode,
        {
            "combined",
            "weighted",
            "weighted_combined",
            "legacy",
            "separate",
            "split",
            "dsv_and_suppression",
            "separate_dsv_suppression",
        },
    )
    _enum(
        error,
        "target_error_constraint_metric",
        config.target_error_constraint_metric,
        {
            "norm",
            "l2",
            "l2_norm",
            "residual_norm",
            "rms",
            "root_mean_square",
            "std",
            "standard_deviation",
            "standard-deviation",
            "homogeneity",
        },
    )
    _enum(
        error,
        "mri_shim_metric_mode",
        getattr(config, "mri_shim_metric_mode", "off"),
        {"off", "report", "optimize"},
    )
    mri_cvar_alpha = _finite_float(getattr(config, "mri_shim_cvar_alpha", 0.95))
    if not np.isfinite(mri_cvar_alpha) or not 0.5 <= mri_cvar_alpha < 1.0:
        error("mri_shim_cvar_alpha", "MRI shim CVaR alpha must be finite and in [0.5, 1).")
    mri_objective_weights = []
    for name in (
        "mri_shim_residual_rms_weight",
        "mri_shim_residual_cvar_weight",
        "mri_shim_spatial_gradient_weight",
    ):
        value = _finite_float(getattr(config, name, 0.0))
        mri_objective_weights.append(value)
        if not np.isfinite(value) or value < 0.0:
            error(name, f"{name} must be finite and non-negative.")
    mri_mode = str(getattr(config, "mri_shim_metric_mode", "off")).strip().lower().replace("-", "_")
    if mri_mode == "optimize" and not any(np.isfinite(value) and value > 0.0 for value in mri_objective_weights):
        error(
            "mri_shim_metric_mode",
            "MRI shim optimize mode requires at least one positive objective weight.",
        )
    mri_spatial_scale = _finite_float(getattr(config, "mri_shim_spatial_scale_m", 0.0))
    if not np.isfinite(mri_spatial_scale) or mri_spatial_scale < 0.0:
        error(
            "mri_shim_spatial_scale_m",
            "MRI shim spatial scale must be finite and non-negative; zero selects an ROI-derived scale.",
        )
    mri_reference_hz = _finite_float(getattr(config, "mri_shim_reference_frequency_hz", 50.0))
    if not np.isfinite(mri_reference_hz) or mri_reference_hz <= 0.0:
        error(
            "mri_shim_reference_frequency_hz",
            "MRI shim reference frequency must be finite and positive.",
        )
    _at_least(
        error,
        "MRI shim derivative neighbor count",
        "mri_shim_derivative_neighbor_count",
        getattr(config, "mri_shim_derivative_neighbor_count", 18),
        4,
    )
    _enum(
        error,
        "constrained_optimization_solver",
        config.constrained_optimization_solver,
        {"auto", "clarabel", "ecos", "scs"},
    )
    _positive(error, "Tikhonov regularization factor", "tikonov_reg_factor", config.tikonov_reg_factor)
    _positive(
        error,
        "constrained optimization tolerance",
        "constrained_optimization_tolerance",
        config.constrained_optimization_tolerance,
    )
    _positive(
        error,
        "constrained optimization maximum iterations",
        "constrained_optimization_max_iterations",
        config.constrained_optimization_max_iterations,
    )
    _at_least(
        error,
        "constrained optimization thread count",
        "constrained_optimization_threads",
        config.constrained_optimization_threads,
        0,
    )
    shield_fraction = _finite_float(getattr(config, "shield_gradient_max_fraction_of_target", 0.0))
    if not np.isfinite(shield_fraction):
        error(
            "shield_gradient_max_fraction_of_target",
            "Shield gradient opposition fraction must be finite.",
        )
    target_error_percent = _finite_float(config.target_error_tolerance_percent)
    target_error_absolute = _finite_float(config.target_error_tolerance_absolute_T)
    has_absolute_norm_limit = bool(np.isfinite(target_error_absolute) and target_error_absolute > 0.0)
    target_metric = str(config.target_error_constraint_metric).strip().lower().replace("-", "_")
    has_absolute_field_limit = (
        target_metric in {"std", "standard_deviation", "homogeneity"} and float(config.target_error_std_limit_T) > 0.0
    ) or (target_metric in {"rms", "root_mean_square"} and float(config.target_error_rms_limit_T) > 0.0)
    if not np.isfinite(target_error_percent) or target_error_percent < 0.0:
        error("target_error_tolerance_percent", "Target error tolerance percent must be a finite non-negative number.")
    elif target_error_percent == 0.0 and not has_absolute_norm_limit and not has_absolute_field_limit:
        error(
            "target_error_tolerance_percent",
            "Target error tolerance is zero and no positive absolute target-error limit is configured.",
            "Set target_error_tolerance_percent, target_error_tolerance_absolute_T, target_error_std_limit_T, or target_error_rms_limit_T.",
        )
    if target_error_percent > 0.0 and target_error_percent >= 100.0:
        error(
            "target_error_tolerance_percent",
            "Target error tolerance must be below 100% for a meaningful efficiency-constrained design.",
            "Use for example 2.0, 5.0, or 10.0.",
        )
    if float(config.target_error_tolerance_absolute_T) < 0.0:
        error("target_error_tolerance_absolute_T", "Absolute target error tolerance must be non-negative.")
    if float(config.target_error_std_limit_T) < 0.0:
        error("target_error_std_limit_T", "Absolute DSV target-error standard-deviation limit must be non-negative.")
    if float(config.target_error_rms_limit_T) < 0.0:
        error("target_error_rms_limit_T", "Absolute DSV target-error RMS limit must be non-negative.")
    for name in (
        "layout_target_error_std_limit_T",
        "layout_target_error_rms_limit_T",
        "layout_target_error_acceptance_absolute_T",
    ):
        value = _finite_float(getattr(config, name, 0.0))
        if not np.isfinite(value) or value < 0.0:
            error(name, f"{name} must be a finite non-negative number.")
    objective_mode = str(getattr(config, "constrained_quadratic_objective_mode", "balanced"))
    for name in ("target_error_constraint_acceptance_absolute_T", "target_error_constraint_acceptance_tolerance"):
        value = _finite_float(getattr(config, name, 0.0))
        if not np.isfinite(value) or value < 0.0:
            error(name, f"{name} must be finite and non-negative.")
    objective_mode = objective_mode.strip().lower().replace("-", "_")
    if objective_mode == "minimax_current_density":
        if str(config.sf_opt_method).lower().replace("-", "_") not in {
            "constrained_quadratic",
            "cvxpy_constrained",
            "minimum_resistance_constrained",
            "minimum_resistance_subject_to_field_error_socp",
            "direct_fixed_error",
            "socp",
            "second_order_cone",
        }:
            error("sf_opt_method", "Minimax current density requires the direct constrained_quadratic method.")
        for name in ("stream_function_smoothness_factor", "turn_spacing_regularization_factor"):
            if _finite_float(getattr(config, name, 0.0)) > 0.0:
                error(name, "Pure minimax current density requires zero quadratic penalty factors.")
    tiebreak = _finite_float(getattr(config, "constrained_quadratic_resistance_tiebreak_factor", 1.0e-3))
    if not np.isfinite(tiebreak) or tiebreak < 0.0:
        error(
            "constrained_quadratic_resistance_tiebreak_factor",
            "Constrained-quadratic resistance tie-break factor must be finite and non-negative.",
        )
    elif objective_mode in {"smoothness_priority", "smoothest_feasible", "contourability_priority"}:
        if tiebreak <= 0.0:
            error(
                "constrained_quadratic_resistance_tiebreak_factor",
                "Smoothness-priority mode requires a positive resistance tie-break factor.",
            )
        if float(getattr(config, "stream_function_smoothness_factor", 0.0)) <= 0.0:
            error(
                "stream_function_smoothness_factor",
                "Smoothness-priority mode requires a positive stream-function smoothness factor.",
            )
    layout_std_limit = float(getattr(config, "layout_target_error_std_limit_T", 0.0))
    if layout_std_limit > 0.0 and float(config.target_error_std_limit_T) >= layout_std_limit:
        warning(
            "layout_target_error_std_limit_T",
            "The continuous stream-function standard-deviation limit leaves no margin for contouring and routing.",
            "Set target_error_std_limit_T below layout_target_error_std_limit_T.",
        )
    layout_rms_limit = float(getattr(config, "layout_target_error_rms_limit_T", 0.0))
    if layout_rms_limit > 0.0 and float(config.target_error_rms_limit_T) >= layout_rms_limit:
        warning(
            "layout_target_error_rms_limit_T",
            "The continuous stream-function RMS limit leaves no margin for contouring and routing.",
            "Set target_error_rms_limit_T below layout_target_error_rms_limit_T.",
        )
    for field_name in (
        "gradient_strength_tolerance_percent",
        "linearity_reference_radius_m",
        "linearity_rms_tolerance_percent",
        "linearity_peak_tolerance_percent",
        "linearity_cross_gradient_tolerance_percent",
    ):
        if float(getattr(config, field_name, 0.0)) < 0.0:
            error(field_name, "Gradient linearity tolerances and reference radius must be non-negative.")
    _enum(
        error,
        "linearity_region",
        getattr(config, "linearity_region", "dsv"),
        {"dsv", "fov", "roi", "target", "primary", "internal", "all", "any", "suppression", "secondary", "shield"},
    )
    _enum(
        error,
        "linearity_normalization",
        getattr(config, "linearity_normalization", "requested_gradient_times_radius"),
        {
            "requested_gradient_times_radius",
            "requested_gradient_radius",
            "gradient_radius",
            "requested",
            "target_peak",
            "peak_target",
            "target",
        },
    )
    if bool(getattr(config, "gradient_slice_maps_enabled", False)):
        _enum(
            error,
            "gradient_slice_map_axis",
            getattr(config, "gradient_slice_map_axis", "z"),
            {"x", "y", "z", "0", "1", "2", "gx", "gy", "gz"},
        )
        _at_least(
            error,
            "gradient slice-map slice count",
            "gradient_slice_map_slice_count",
            getattr(config, "gradient_slice_map_slice_count", 9),
            1,
        )
        _validate_gradient_slice_grid_size(error, getattr(config, "gradient_slice_map_grid_size", 96))
        _enum(
            error,
            "gradient_slice_map_domain",
            getattr(config, "gradient_slice_map_domain", "dsv"),
            {"dsv", "fov", "roi", "target", "primary", "suppression", "secondary", "shield", "all", "any", "full"},
        )
        _enum(
            error,
            "gradient_slice_map_source",
            getattr(config, "gradient_slice_map_source", "layout"),
            {
                "layout",
                "wire",
                "wire_path",
                "connected",
                "connected_wire",
                "contours",
                "contour",
                "loops",
                "unconnected",
                "stream",
                "stream_function",
                "sf",
                "target",
                "ideal",
            },
        )
        _enum(
            error,
            "gradient_slice_map_current",
            getattr(config, "gradient_slice_map_current", "target"),
            {
                "target",
                "requested",
                "operating",
                "fitted",
                "per_amp",
                "per_a",
                "1a",
                "unit",
                "stream",
                "stream_function",
                "contour_step",
                "sf",
            },
        )
        _enum(
            error,
            "gradient_slice_map_gradient_method",
            getattr(config, "gradient_slice_map_gradient_method", "finite_difference"),
            {"finite_difference", "finite", "fd"},
        )
        gradient_step = _finite_float(getattr(config, "gradient_slice_map_gradient_step_m", 0.0) or 0.0)
        if not np.isfinite(gradient_step) or gradient_step < 0.0:
            error(
                "gradient_slice_map_gradient_step_m", "Gradient slice-map finite-difference step must be non-negative."
            )
        _enum(
            error,
            "gradient_slice_map_overlay_mode",
            getattr(config, "gradient_slice_map_overlay_mode", "slice_slab"),
            {"slice_slab", "slab", "slice", "project_all", "all", "none", "off"},
        )
        slab_thickness = _finite_float(getattr(config, "gradient_slice_map_overlay_slab_thickness_m", 0.0) or 0.0)
        if not np.isfinite(slab_thickness) or slab_thickness < 0.0:
            error(
                "gradient_slice_map_overlay_slab_thickness_m",
                "Gradient slice-map overlay slab thickness must be non-negative.",
            )
    if float(config.suppression_error_tolerance_percent) < 0.0:
        error("suppression_error_tolerance_percent", "Suppression error tolerance percent must be non-negative.")
    if float(config.suppression_error_tolerance_percent) >= 100.0:
        warning(
            "suppression_error_tolerance_percent",
            "Suppression error tolerance is >=100%; active shielding may become weak.",
            "For paper studies use a small explicit suppression budget, e.g. 2-10%.",
        )
    if float(config.suppression_error_tolerance_absolute_T) < 0.0:
        error("suppression_error_tolerance_absolute_T", "Absolute suppression error tolerance must be non-negative.")
    if float(config.suppression_field_rms_limit_T) < 0.0:
        error("suppression_field_rms_limit_T", "Suppression RMS field limit must be non-negative.")
    if float(getattr(config, "suppression_field_peak_limit_T", 0.0)) < 0.0:
        error("suppression_field_peak_limit_T", "Suppression peak field limit must be non-negative.")
    if float(getattr(config, "target_error_constraint_acceptance_absolute_T", 0.0)) < 0.0:
        error(
            "target_error_constraint_acceptance_absolute_T",
            "Constraint post-check absolute acceptance tolerance must be non-negative.",
        )
    _enum(
        error,
        "suppression_field_component",
        getattr(config, "suppression_field_component", "target"),
        {
            "target",
            "same",
            "useful",
            "bx",
            "by",
            "bz",
            "b0",
            "bdotb0",
            "b_dot_b0",
            "b_parallel",
            "bnormal",
            "b_normal",
            "normal",
            "vector",
            "components",
            "componentwise",
            "component_wise",
            "cartesian_components",
            "xyz",
            "bx_by_bz",
            "bmag",
            "magnitude",
        },
    )
    _enum(
        error,
        "suppression_surface_weighting",
        getattr(config, "suppression_surface_weighting", "uniform"),
        {"uniform", "none", "area", "mesh_area", "surface_area"},
    )
    _enum(
        error,
        "suppression_normal_source",
        getattr(config, "suppression_normal_source", "mesh"),
        {"mesh", "target", "metadata", "box", "radial", "cylindrical"},
    )
    if (
        float(config.suppression_field_rms_limit_T) == 0.0
        and float(getattr(config, "suppression_field_peak_limit_T", 0.0)) == 0.0
        and float(config.suppression_error_tolerance_percent) == 0.0
        and float(config.suppression_error_tolerance_absolute_T) == 0.0
    ):
        warning(
            "suppression_field_rms_limit_T",
            "No positive active-shield suppression budget is configured.",
            "Set suppression_field_rms_limit_T for a physical RMS fringe-field constraint.",
        )
    _at_least(
        error,
        "target error constraint candidate count",
        "target_error_constraint_factor_count",
        config.target_error_constraint_factor_count,
        3,
    )
    _positive(
        error,
        "target-error constraint factor minimum multiplier",
        "target_error_constraint_factor_min_multiplier",
        config.target_error_constraint_factor_min_multiplier,
    )
    _positive(
        error,
        "target-error constraint factor maximum multiplier",
        "target_error_constraint_factor_max_multiplier",
        config.target_error_constraint_factor_max_multiplier,
    )
    for field_name, label in (
        ("regularization_auto_range_min_singular_relative", "auto regularization minimum singular-value ratio"),
        ("regularization_auto_range_target_projection_relative", "auto regularization target-projection ratio"),
        ("regularization_auto_range_lower_padding", "auto regularization lower padding"),
        ("regularization_auto_range_upper_padding", "auto regularization upper padding"),
    ):
        _positive(error, label, field_name, getattr(config, field_name))
    for field_name in (
        "regularization_auto_range_min_factor",
        "regularization_auto_range_max_factor",
        "regularization_auto_range_max_contour_step_A",
    ):
        if float(getattr(config, field_name)) < 0.0:
            error(field_name, "Auto regularization guard values must be non-negative.")
    if (
        float(config.regularization_auto_range_max_factor) > 0.0
        and float(config.regularization_auto_range_min_factor) > 0.0
        and float(config.regularization_auto_range_max_factor) <= float(config.regularization_auto_range_min_factor)
    ):
        error(
            "regularization_auto_range_max_factor",
            "Auto regularization maximum factor must exceed the minimum factor when both are enabled.",
        )
    if float(config.target_error_constraint_factor_max_multiplier) <= float(
        config.target_error_constraint_factor_min_multiplier
    ):
        error(
            "target_error_constraint_factor_max_multiplier",
            "Target-error constraint factor maximum multiplier must exceed the minimum multiplier.",
            "Use a wide range such as 1e-10 to 1e10 when exploring residual-limited designs.",
        )
    _enum(
        error,
        "target_error_infeasible_fallback",
        config.target_error_infeasible_fallback,
        {"minimum_residual", "best_residual", "residual_floor", "near_minimum_residual", "l_curve"},
    )
    if float(config.target_error_infeasible_residual_margin_percent) < 0.0:
        error(
            "target_error_infeasible_residual_margin_percent",
            "Target-error infeasible residual margin must be non-negative.",
        )
    _at_least(
        error,
        "target-error automatic rerun count",
        "target_error_auto_retry_max_reruns",
        config.target_error_auto_retry_max_reruns,
        0,
    )
    _at_least(
        error,
        "target-error automatic rerun maximum candidate count",
        "target_error_auto_retry_max_candidate_count",
        config.target_error_auto_retry_max_candidate_count,
        3,
    )
    try:
        retry_shrink = float(config.target_error_auto_retry_factor_shrink)
    except (TypeError, ValueError):
        retry_shrink = float("nan")
    if not np.isfinite(retry_shrink) or retry_shrink <= 0.0 or retry_shrink >= 1.0:
        error(
            "target_error_auto_retry_factor_shrink",
            "Target-error automatic retry shrink factor must be finite and between 0 and 1.",
        )
    _at_least_float(
        error,
        "target-error automatic retry candidate growth",
        "target_error_auto_retry_candidate_growth",
        config.target_error_auto_retry_candidate_growth,
        1.0,
    )
    _at_least(error, "contour levels", "levels", config.levels, 1)
    raw_contour_method = (
        str(getattr(config, "contour_discretization_method", "level_count")).strip().lower().replace("-", "_")
    )
    _enum(
        error,
        "contour_discretization_method",
        raw_contour_method,
        LEVEL_COUNT_METHODS | ENGINEERING_METHODS | {"auto"},
    )
    if float(getattr(config, "max_current_A", 0.0)) < 0.0:
        error("max_current_A", "Maximum current must be non-negative.")
    if float(getattr(config, "max_wire_current_A", 0.0)) < 0.0:
        error("max_wire_current_A", "Maximum wire current must be non-negative.")
    if float(getattr(config, "min_wire_distance_m", 0.0)) < 0.0:
        error("min_wire_distance_m", "Minimum wire distance must be non-negative.")
    if float(getattr(config, "surface_current_density_limit_A_per_m", 0.0)) < 0.0:
        error("surface_current_density_limit_A_per_m", "Surface-current-density limit must be non-negative.")
    if float(getattr(config, "engineering_contour_step_safety_factor", 1.0)) <= 0.0:
        error("engineering_contour_step_safety_factor", "Engineering contour-step safety factor must be positive.")
    if float(getattr(config, "engineering_contour_step_safety_factor", 1.0)) > 1.0:
        error(
            "engineering_contour_step_safety_factor",
            "Engineering contour-step safety factor cannot exceed 1.",
            "Use 1.0 to run at the configured maximum current or a lower value for margin.",
        )
    _enum(
        error,
        "gradient_hardware_infeasible_policy",
        getattr(config, "gradient_hardware_infeasible_policy", "scale_to_limit"),
        {"scale_to_limit", "scale", "cap_target", "report_only", "report", "raise", "error"},
    )
    _at_least(
        error,
        "engineering contour maximum levels",
        "engineering_contour_max_levels",
        config.engineering_contour_max_levels,
        0,
    )
    if contour_discretization_method(config) == "engineering":
        if max(float(config.max_current_A), float(config.max_wire_current_A)) <= 0.0:
            error(
                "max_current_A",
                "Engineering contouring requires a positive maximum current.",
                "Set max_current_A or max_wire_current_A.",
            )
        if max(float(config.min_wire_distance_m), float(config.min_turn_distance)) <= 0.0:
            error(
                "min_wire_distance_m",
                "Engineering contouring requires a positive minimum wire distance.",
                "Set min_wire_distance_m or min_turn_distance.",
            )
    _enum(
        error,
        "zero_crossing_level_strategy",
        config.zero_crossing_level_strategy,
        {"paired", "symmetric", "full_range", "centered"},
    )
    if float(config.pot_offset_factor) < 0.0 or float(config.pot_offset_factor) >= 1.0:
        error("pot_offset_factor", "Contour offset factor must be in [0, 1).", "Use 0.25 or 0.5 for normal designs.")
    part_phase_deltas = tuple(float(value) for value in config.contour_part_phase_offset_deltas)
    if any((not np.isfinite(value)) or value < 0.0 or value >= 1.0 for value in part_phase_deltas):
        error(
            "contour_part_phase_offset_deltas",
            "Per-part contour phase deltas must be finite values in [0, 1).",
            "Use (0.0, 0.5) to interleave two current-surface layers.",
        )
    if float(config.min_turn_distance) < 0.0:
        error("min_turn_distance", "Minimum turn distance must be non-negative.")
    level_range_percentile = float(getattr(config, "contour_level_range_percentile", 100.0))
    if level_range_percentile <= 0.0 or level_range_percentile > 100.0:
        error(
            "contour_level_range_percentile",
            "Contour level range percentile must be in (0, 100].",
            "Use 100 for full stream-function range, or e.g. 95 to ignore isolated seam extrema.",
        )
    density_percentile = float(config.turn_spacing_density_percentile)
    if density_percentile <= 0.0 or density_percentile > 100.0:
        error(
            "turn_spacing_density_percentile",
            "Turn-spacing density percentile must be in (0, 100].",
            "Use 100 for strict max-density spacing, or e.g. 95-99 to ignore isolated mesh-current spikes.",
        )
    if float(config.stream_function_smoothness_factor) < 0.0:
        error("stream_function_smoothness_factor", "Stream-function smoothness factor must be non-negative.")
    _enum(
        error,
        "stream_function_basis",
        getattr(config, "stream_function_basis", "vertex"),
        {
            "vertex",
            "vertices",
            "full",
            "cotangent_spectral_lowpass",
            "spectral_lowpass",
            "laplace_beltrami_lowpass",
            "laplace-beltrami-lowpass",
        },
    )
    if float(getattr(config, "stream_function_spectral_cutoff_length_m", 0.0)) < 0.0:
        error(
            "stream_function_spectral_cutoff_length_m", "Spectral stream-function cutoff length must be non-negative."
        )
    _enum(
        error,
        "stream_function_spectral_cutoff_policy",
        getattr(config, "stream_function_spectral_cutoff_policy", "hard"),
        {"hard", "minimum_modes"},
    )
    if int(getattr(config, "stream_function_spectral_max_modes", 0)) < 0:
        error("stream_function_spectral_max_modes", "Spectral stream-function max modes must be non-negative.")
    if int(getattr(config, "stream_function_spectral_min_modes", 1)) < 1:
        error("stream_function_spectral_min_modes", "Spectral stream-function min modes must be at least one.")
    if int(getattr(config, "contourability_max_contour_count", 0)) < 0:
        error("contourability_max_contour_count", "Contourability maximum contour count must be non-negative.")
    _enum(
        error,
        "contour_level_grid_mode",
        getattr(config, "contour_level_grid_mode", "centered_margin"),
        {"centered_margin", "fixed_step_phase"},
    )
    _enum(error, "mesh_quality_policy", getattr(config, "mesh_quality_policy", "warn"), {"off", "warn", "strict"})
    _at_least_float(
        error,
        "symmetry plane snap tolerance",
        "symmetry_reduction_plane_snap_tolerance",
        getattr(config, "symmetry_reduction_plane_snap_tolerance", 0.0),
        0.0,
    )
    _positive(
        error,
        "mesh quality maximum triangle edge ratio",
        "mesh_quality_max_triangle_edge_ratio",
        getattr(config, "mesh_quality_max_triangle_edge_ratio", 25.0),
    )
    _positive(
        error,
        "mesh quality maximum altitude aspect ratio",
        "mesh_quality_max_altitude_aspect_ratio",
        getattr(config, "mesh_quality_max_altitude_aspect_ratio", 50.0),
    )
    _positive(
        error,
        "mesh quality maximum edge-length ratio",
        "mesh_quality_max_edge_length_ratio",
        getattr(config, "mesh_quality_max_edge_length_ratio", 100.0),
    )
    _enum(
        error,
        "contour_path_regularization_mode",
        config.contour_path_regularization_mode,
        {
            "off",
            "resample",
            "parameterized_bspline",
            "parameterised_bspline",
            "bspline",
            "smooth",
            "resample_smooth",
            "smooth_resample",
        },
    )
    if float(config.contour_resample_target_segment_length) < 0.0:
        error("contour_resample_target_segment_length", "Contour resampling segment length must be non-negative.")
    if int(config.contour_smoothing_iterations) < 0:
        error("contour_smoothing_iterations", "Contour smoothing iterations must be non-negative.")
    factor = float(config.contour_smoothing_factor)
    if factor < 0.0 or factor > 1.0:
        error("contour_smoothing_factor", "Contour smoothing factor must be in [0, 1].")
    _enum(
        error,
        "contour_smoothing_scale_mode",
        getattr(config, "contour_smoothing_scale_mode", "legacy"),
        {"legacy", "mesh_coupled"},
    )
    mesh_gain = float(getattr(config, "contour_smoothing_mesh_gain", 0.1))
    if mesh_gain <= 0.0 or mesh_gain > 1.0:
        error(
            "contour_smoothing_mesh_gain",
            "Contour mesh-scale gain must be in (0, 1].",
        )
    if int(getattr(config, "contour_smoothing_work_samples_per_mesh_length", 6)) < 4:
        error(
            "contour_smoothing_work_samples_per_mesh_length",
            "Contour smoothing requires at least four working samples per mesh length.",
        )
    if int(getattr(config, "contour_smoothing_output_samples_per_mesh_length", 8)) < 2:
        error(
            "contour_smoothing_output_samples_per_mesh_length",
            "Contour smoothing requires at least two output samples per mesh length.",
        )
    if int(getattr(config, "contour_smoothing_patch_rings", 4)) < 1:
        error("contour_smoothing_patch_rings", "Contour smoothing patch rings must be at least one.")
    if float(config.contour_smoothing_max_displacement) < 0.0:
        error("contour_smoothing_max_displacement", "Contour smoothing max displacement must be non-negative.")
    if float(config.turn_spacing_regularization_factor) < 0.0:
        error("turn_spacing_regularization_factor", "Turn-spacing regularization factor must be non-negative.")
    if int(config.turn_spacing_max_iterations) < 1:
        error("turn_spacing_max_iterations", "Turn-spacing optimization attempts must be at least one.")
    _enum(error, "contour_symmetry_validation", config.contour_symmetry_validation, {"auto", "strict", "off"})
    _positive(
        error,
        "contour symmetry geometry tolerance",
        "contour_symmetry_geometry_tolerance",
        config.contour_symmetry_geometry_tolerance,
    )
    _positive(
        error,
        "contour symmetry stream-function tolerance",
        "contour_symmetry_stream_function_tolerance",
        config.contour_symmetry_stream_function_tolerance,
    )
    _enum(error, "contourability_validation", config.contourability_validation, {"auto", "strict", "off"})
    _enum(
        error,
        "contour_loop_pruning_mode",
        getattr(config, "contour_loop_pruning_mode", "off"),
        {"off", "residual_budget"},
    )
    _at_least_float(
        error,
        "contour-loop pruning relative residual tolerance",
        "contour_loop_pruning_residual_relative_tolerance",
        getattr(config, "contour_loop_pruning_residual_relative_tolerance", 0.0),
        0.0,
    )
    _at_least_float(
        error,
        "contour-loop pruning absolute residual tolerance",
        "contour_loop_pruning_residual_absolute_tolerance_T",
        getattr(config, "contour_loop_pruning_residual_absolute_tolerance_T", 0.0),
        0.0,
    )
    _at_least(
        error,
        "contour-loop pruning maximum removed loops",
        "contour_loop_pruning_max_removed_loops",
        getattr(config, "contour_loop_pruning_max_removed_loops", 0),
        0,
    )
    sparsification_mode = str(getattr(config, "contour_sparsification_mode", "off")).strip().lower()
    _enum(
        error,
        "contour_sparsification_mode",
        sparsification_mode,
        {"off", "pairwise"},
    )
    _enum(
        error,
        "contour_sparsification_positioning",
        getattr(config, "contour_sparsification_positioning", "midpoint_level"),
        {"midpoint_level", "field_fit"},
    )
    _at_least_float(
        error,
        "contour-sparsification RMS field-error limit",
        "contour_sparsification_max_rms_field_error_percent",
        getattr(config, "contour_sparsification_max_rms_field_error_percent", 1.0),
        0.0,
    )
    _at_least_float(
        error,
        "contour-sparsification peak field-error limit",
        "contour_sparsification_max_peak_field_error_percent",
        getattr(config, "contour_sparsification_max_peak_field_error_percent", 5.0),
        0.0,
    )
    _at_least(
        error,
        "contour-sparsification maximum merges",
        "contour_sparsification_max_merges",
        getattr(config, "contour_sparsification_max_merges", 0),
        0,
    )
    _at_least(
        error,
        "contour-sparsification maximum passes",
        "contour_sparsification_max_passes",
        getattr(config, "contour_sparsification_max_passes", 1),
        1,
    )
    _at_least(
        error,
        "contour-sparsification maximum turn multiplicity",
        "contour_sparsification_max_turn_multiplicity",
        getattr(config, "contour_sparsification_max_turn_multiplicity", 2),
        2,
    )
    _at_least(
        error,
        "contour-sparsification field-fit candidate count",
        "contour_sparsification_field_fit_candidate_count",
        getattr(config, "contour_sparsification_field_fit_candidate_count", 5),
        1,
    )
    _enum(
        error,
        "contour_sparsification_realization",
        getattr(config, "contour_sparsification_realization", "ideal_weighted"),
        {"ideal_weighted"},
    )
    for field_name, label in (
        (
            "contour_sparsification_residual_relative_tolerance",
            "contour-sparsification cumulative relative residual tolerance",
        ),
        (
            "contour_sparsification_residual_absolute_tolerance_T",
            "contour-sparsification cumulative absolute residual tolerance",
        ),
    ):
        value = getattr(config, field_name, None)
        if value is not None:
            _at_least_float(error, label, field_name, value, 0.0)
    sparsification_pruning_mode = str(getattr(config, "contour_sparsification_pruning_mode", "off")).strip().lower()
    _enum(
        error,
        "contour_sparsification_pruning_mode",
        sparsification_pruning_mode,
        {"off", "residual_budget"},
    )
    _at_least_float(
        error,
        "post-sparsification pruning relative residual tolerance",
        "contour_sparsification_pruning_residual_relative_tolerance",
        getattr(config, "contour_sparsification_pruning_residual_relative_tolerance", 0.0),
        0.0,
    )
    _at_least_float(
        error,
        "post-sparsification pruning absolute residual tolerance",
        "contour_sparsification_pruning_residual_absolute_tolerance_T",
        getattr(config, "contour_sparsification_pruning_residual_absolute_tolerance_T", 0.0),
        0.0,
    )
    _at_least(
        error,
        "post-sparsification pruning maximum removed geometries",
        "contour_sparsification_pruning_max_removed_geometries",
        getattr(config, "contour_sparsification_pruning_max_removed_geometries", 0),
        0,
    )
    _at_least(
        error,
        "post-sparsification pruning target geometry count",
        "contour_sparsification_pruning_target_geometry_count",
        getattr(config, "contour_sparsification_pruning_target_geometry_count", 0),
        0,
    )
    _enum(
        error,
        "contour_sparsification_pruning_search_mode",
        getattr(config, "contour_sparsification_pruning_search_mode", "auto"),
        {"auto", "exact", "beam"},
    )
    _at_least(
        error,
        "post-sparsification exact-search maximum symmetry orbits",
        "contour_sparsification_pruning_exact_max_orbits",
        getattr(config, "contour_sparsification_pruning_exact_max_orbits", 16),
        0,
    )
    _at_least(
        error,
        "post-sparsification beam-search width",
        "contour_sparsification_pruning_beam_width",
        getattr(config, "contour_sparsification_pruning_beam_width", 128),
        1,
    )
    post_pruning_merge_mode = (
        str(getattr(config, "contour_sparsification_post_pruning_merge_mode", "off")).strip().lower()
    )
    _enum(
        error,
        "contour_sparsification_post_pruning_merge_mode",
        post_pruning_merge_mode,
        {"off", "retained_adjacent"},
    )
    _at_least(
        error,
        "post-pruning contour-merge maximum passes",
        "contour_sparsification_post_pruning_max_passes",
        getattr(config, "contour_sparsification_post_pruning_max_passes", 1),
        1,
    )
    if sparsification_pruning_mode != "off" and sparsification_mode == "off":
        error(
            "contour_sparsification_pruning_mode",
            "Post-sparsification pruning requires active contour sparsification.",
            "Set contour_sparsification_mode='pairwise' or disable post-sparsification pruning.",
        )
    if post_pruning_merge_mode != "off" and sparsification_pruning_mode == "off":
        error(
            "contour_sparsification_post_pruning_merge_mode",
            "Post-pruning contour consolidation requires active post-sparsification pruning.",
            "Set contour_sparsification_pruning_mode='residual_budget' or disable post-pruning merging.",
        )
    if sparsification_mode != "off" and not bool(getattr(config, "skip_postprocessing", False)):
        error(
            "skip_postprocessing",
            "Pairwise contour sparsification currently represents ideal weighted turns only.",
            "Set skip_postprocessing=True until separated physical turn realization and routing are selected.",
        )
    if sparsification_mode != "off" and bool(getattr(config, "contour_only_sweep_mode", False)):
        warning(
            "contour_only_sweep_mode",
            "Contour-only sweep will export one solid per ideal weighted representative; turn multiplicity is metadata and is not expanded into separated physical tracks.",
            "Use the exported multiplicity diagnostics when interpreting the bodies, or disable sparsification for one body per physical unit turn.",
        )
    for name in (
        "contourability_max_rms_error_percent",
        "contourability_max_mean_error_percent",
        "contourability_max_peak_error_percent",
        "contourability_max_tiny_loop_fraction",
        "contourability_min_loop_length_m",
        "contourability_max_closure_gap_m",
        "contourability_max_surface_error_m",
    ):
        if float(getattr(config, name)) < 0.0:
            error(name, f"{name} must be non-negative.")
    if float(config.contourability_max_tiny_loop_fraction) > 1.0:
        error("contourability_max_tiny_loop_fraction", "Tiny-loop fraction must be in [0, 1].")
    smoothing_candidates = tuple(getattr(config, "contour_field_smoothing_candidates", ()))
    for index, candidate in enumerate(smoothing_candidates):
        try:
            iterations, factor = candidate
            iterations_float = float(iterations)
            iterations_int = int(iterations)
            factor = float(factor)
        except (TypeError, ValueError):
            error(
                f"contour_field_smoothing_candidates[{index}]",
                "Each field-constrained smoothing candidate must be (positive integer iterations, factor in (0, 1]).",
            )
            continue
        if (
            iterations_int < 1
            or iterations_float != float(iterations_int)
            or not np.isfinite(factor)
            or factor <= 0.0
            or factor > 1.0
        ):
            error(
                f"contour_field_smoothing_candidates[{index}]",
                "Each field-constrained smoothing candidate must be (positive integer iterations, factor in (0, 1]).",
            )
    if bool(getattr(config, "contour_field_constrained_smoothing_enabled", False)):
        if not smoothing_candidates:
            error(
                "contour_field_smoothing_candidates",
                "Field-constrained contour smoothing requires at least one candidate.",
            )
        if str(config.contourability_validation).strip().lower() == "off":
            warning(
                "contour_field_constrained_smoothing_enabled",
                "Field-constrained smoothing is inactive while contourability validation is off.",
                "Use contourability_validation='auto' or 'strict'.",
            )
    min_smoothing_improvement = _finite_float(getattr(config, "contour_field_smoothing_min_improvement_percent", 0.0))
    if not np.isfinite(min_smoothing_improvement) or min_smoothing_improvement < 0.0:
        error(
            "contour_field_smoothing_min_improvement_percent",
            "Minimum contour-smoothing improvement must be finite and non-negative.",
        )
    repair_factors = tuple(float(value) for value in config.contourability_repair_level_factors)
    if not repair_factors:
        error("contourability_repair_level_factors", "At least one repair level factor is required.")
    elif any((not np.isfinite(value)) or value <= 0.0 for value in repair_factors):
        error("contourability_repair_level_factors", "Repair level factors must be finite positive values.")
    repair_phase_offsets = tuple(float(value) for value in config.contourability_repair_phase_offsets)
    if any((not np.isfinite(value)) or value < 0.0 or value >= 1.0 for value in repair_phase_offsets):
        error("contourability_repair_phase_offsets", "Repair phase offsets must be finite values in [0, 1).")
    repair_range_percentiles = tuple(float(value) for value in config.contourability_repair_range_percentiles)
    if any((not np.isfinite(value)) or value <= 0.0 or value > 100.0 for value in repair_range_percentiles):
        error(
            "contourability_repair_range_percentiles",
            "Repair range percentiles must be finite values in (0, 100].",
        )
    valid_zero_crossing = {"paired", "symmetric", "full_range", "centered"}
    for strategy in config.contourability_repair_zero_crossing_strategies:
        if str(strategy).strip().lower().replace("-", "_") not in valid_zero_crossing:
            error(
                "contourability_repair_zero_crossing_strategies",
                "Repair zero-crossing strategies must be paired/symmetric or full_range/centered.",
            )
            break
    _at_least(
        error,
        "contourability repair minimum levels",
        "contourability_repair_min_levels",
        config.contourability_repair_min_levels,
        1,
    )
    _at_least(
        error,
        "contourability repair maximum levels",
        "contourability_repair_max_levels",
        config.contourability_repair_max_levels,
        1,
    )
    if int(config.contourability_repair_max_levels) < int(config.contourability_repair_min_levels):
        error("contourability_repair_max_levels", "Repair maximum levels must be >= repair minimum levels.")
    _enum(error, "symmetry_reduction", config.symmetry_reduction, {"auto", "strict", "off"})
    _enum(
        error,
        "symmetry_reduction_source",
        config.symmetry_reduction_source,
        {"generated_parts", "single_mesh", "half_mesh", "auto"},
    )
    _enum(
        error, "symmetry_reduction_half_space", config.symmetry_reduction_half_space, {"positive", "negative", "auto"}
    )
    _positive(
        error,
        "symmetry reduction clip tolerance",
        "symmetry_reduction_clip_tolerance",
        config.symmetry_reduction_clip_tolerance,
    )
    _enum(error, "symmetry_reduction_target_parity", config.symmetry_reduction_target_parity, {"auto", "even", "odd"})
    _enum(
        error,
        "symmetry_reduction_target_parity_policy",
        config.symmetry_reduction_target_parity_policy,
        {"require", "diagnostic"},
    )
    _validate_vector3(error, "symmetry_reduction_mirror_plane_point", config.symmetry_reduction_mirror_plane_point)
    plane_normal = _validate_vector3(
        error, "symmetry_reduction_mirror_plane_normal", config.symmetry_reduction_mirror_plane_normal
    )
    if plane_normal is not None and float(np.linalg.norm(plane_normal)) <= 0.0:
        error("symmetry_reduction_mirror_plane_normal", "Mirror-plane normal must be non-zero.")
    _positive(
        error,
        "symmetry reduction target tolerance",
        "symmetry_reduction_target_tolerance",
        config.symmetry_reduction_target_tolerance,
    )
    for name in (
        "symmetry_reduction_target_geometry_tolerance_m",
        "symmetry_reduction_target_parity_tolerance",
        "symmetry_reduction_target_weight_tolerance",
    ):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value < 0.0:
            error(name, f"{name} must be finite and non-negative.")
    source = str(config.preoptimized_stream_function_file).strip()
    alias = str(config.sf_source_file).strip()
    if source and source.lower() != "none" and alias and alias.lower() != "none":
        warning(
            "preoptimized_stream_function_file",
            "Both preoptimized_stream_function_file and sf_source_file are set; the explicit Python name is used.",
        )
    active_source = source if source and source.lower() != "none" else alias
    if active_source and active_source.lower() != "none":
        if Path(active_source).suffix.lower() not in {".csv", ".tsv", ".json", ".npz"}:
            error(
                "preoptimized_stream_function_file",
                "Unsupported preoptimized stream-function file extension.",
                "Use .csv, .tsv, .json, or .npz.",
            )


def _validate_routing_settings(config: CoilGenConfig, error, warning) -> None:
    _enum(
        error,
        "topological_grouping_method",
        config.topological_grouping_method,
        {
            "auto",
            "containment",
            "legacy",
            "nested",
            "loop_in_loop",
            "parallel",
            "parallel_surface",
            "surface_parallel",
            "boundary_aware",
        },
    )
    _enum(
        error,
        "inter_group_connection_path_method",
        config.inter_group_connection_path_method,
        {
            "direct_3d",
            "direct",
            "straight",
            "geodesic",
            "surface_geodesic_or_direct",
            "mesh_geodesic_or_direct",
            "relaxed_geodesic",
            "surface",
            "surface_geodesic",
            "mesh_geodesic",
            "strict_geodesic",
        },
    )
    _at_least(
        error,
        "inter-group candidate count",
        "inter_group_connection_candidate_count",
        config.inter_group_connection_candidate_count,
        1,
    )
    _enum(
        error,
        "inter_group_cut_alignment",
        config.inter_group_cut_alignment,
        {
            "auto",
            "off",
            "none",
            "disabled",
            "false",
            "0",
            "on",
            "true",
            "1",
            "cylinder",
            "cylindrical",
            "cylinder_seam",
            "cylindrical_seam",
            "seam",
        },
    )
    if float(config.inter_group_cut_alignment_distance_rel_tol) < 0.0:
        error(
            "inter_group_cut_alignment_distance_rel_tol",
            "Inter-group cut alignment distance tolerance must be non-negative.",
        )
    _at_least(
        error,
        "inter-group cut alignment candidate factor",
        "inter_group_cut_alignment_candidate_factor",
        config.inter_group_cut_alignment_candidate_factor,
        1,
    )
    _at_least(
        error,
        "within-group cut candidate count",
        "within_group_cut_candidate_count",
        config.within_group_cut_candidate_count,
        1,
    )
    if float(config.interconnection_cut_width) <= 0.0:
        error("interconnection_cut_width", "Opening cut width must be positive.")
    if config.within_group_cut_width is not None and float(config.within_group_cut_width) <= 0.0:
        error("within_group_cut_width", "Within-group opening width must be positive when specified.")
    if config.inter_group_cut_width is not None and float(config.inter_group_cut_width) <= 0.0:
        error("inter_group_cut_width", "Inter-group opening width must be positive when specified.")
    if float(config.normal_shift_length) < 0.0:
        error("normal_shift_length", "Normal shift length must be non-negative.")
    if float(config.min_inter_group_connector_clearance) < 0.0:
        error("min_inter_group_connector_clearance", "Connector clearance threshold must be non-negative.")
    terminal_strategy = str(getattr(config, "layout_terminal_strategy", "undefined")).strip().lower()
    if terminal_strategy not in {"undefined", "waived", "pads", "wire_stubs"}:
        error("layout_terminal_strategy", "Terminal strategy must be undefined, waived, pads, or wire_stubs.")
    _at_least(error, "layout terminal pad count", "layout_terminal_pad_count", config.layout_terminal_pad_count, 0)
    _at_least(error, "layout terminal exit count", "layout_terminal_exit_count", config.layout_terminal_exit_count, 0)
    _at_least(
        error, "layout keepout region count", "layout_keepout_region_count", config.layout_keepout_region_count, 0
    )
    if not config.skip_postprocessing and terminal_strategy == "undefined":
        warning(
            "layout_terminal_strategy",
            "Connected-layout routing has no terminal handling strategy.",
            "Set pads, wire_stubs, or waived when the result is intended for manufacturing review.",
        )
    if config.skip_postprocessing:
        warning(
            "skip_postprocessing",
            "Topology/routing postprocessing is disabled.",
            "Fields and optional inductance are evaluated from unconnected contours.",
        )
        if bool(getattr(config, "save_wire_trace_csv", False)):
            warning(
                "save_wire_trace_csv",
                "Trace export is enabled while connected-layout routing is disabled.",
                "Unswept contour STEP wires will be saved; routed-path CSV and STEP files require skip_postprocessing=False.",
            )
    wire_trace_format = str(getattr(config, "wire_trace_csv_format", "detailed")).strip().lower()
    if wire_trace_format not in {"detailed", "xyz_no_header"}:
        error(
            "wire_trace_csv_format",
            "Wire-trace CSV format must be detailed or xyz_no_header.",
        )
    wire_trace_unit = str(getattr(config, "wire_trace_csv_length_unit", "m")).strip().lower()
    if wire_trace_unit not in {"m", "cm", "mm"}:
        error("wire_trace_csv_length_unit", "Wire-trace CSV length unit must be m, cm, or mm.")
    wire_trace_decimal_places = int(getattr(config, "wire_trace_csv_decimal_places", 9))
    _at_least(
        error,
        "wire-trace CSV decimal places",
        "wire_trace_csv_decimal_places",
        wire_trace_decimal_places,
        0,
    )
    if wire_trace_decimal_places > 15:
        error("wire_trace_csv_decimal_places", "Wire-trace CSV decimal places must not exceed 15.")


def _validate_performance_settings(config: CoilGenConfig, error, warning) -> None:
    _enum(
        error, "sensitivity_calculation_method", config.sensitivity_calculation_method, {"auto", "vectorized", "scalar"}
    )
    _enum(error, "wire_distance_method", config.wire_distance_method, {"auto", "spatial", "all_pairs"})
    _at_least(
        error,
        "point-cloud plot maximum point count",
        "plot_pointcloud_max_points",
        config.plot_pointcloud_max_points,
        0,
    )
    _at_least(
        error,
        "point-cloud plot random seed",
        "plot_pointcloud_random_seed",
        config.plot_pointcloud_random_seed,
        0,
    )
    _enum(
        error,
        "field_evaluation_current_source",
        config.field_evaluation_current_source,
        {
            "target_fit",
            "target_projection",
            "projection",
            "least_squares",
            "target_std_projection",
            "std_projection",
            "homogeneity_projection",
            "stream_function",
            "contour_step",
        },
    )
    _at_least(error, "Biot-Savart chunk size", "biot_savart_chunk_size", config.biot_savart_chunk_size, 1)
    _at_least(error, "inductance chunk size", "inductance_chunk_size", config.inductance_chunk_size, 1)
    _at_least(error, "parallel minimum work size", "parallel_min_work_size", config.parallel_min_work_size, 0)
    _at_least(
        error, "sensitivity chunks per worker", "sensitivity_chunks_per_worker", config.sensitivity_chunks_per_worker, 1
    )
    _positive(error, "parallel memory cap", "parallel_max_chunk_memory_mb", config.parallel_max_chunk_memory_mb)
    if int(config.parallel_worker_count) < -256:
        warning("parallel_worker_count", "Very negative worker count requested.", "Use -1 to keep one CPU core free.")


def _validate_manufacturing_settings(config: CoilGenConfig, error, warning) -> None:
    _enum(error, "inductance_source", config.inductance_source, {"auto", "wire_path", "unconnected_contours"})
    _enum(
        error, "conductor_mounting_mode", config.conductor_mounting_mode, {"centered", "surface_outer", "surface_inner"}
    )
    _enum(error, "sweep_path_simplification", config.sweep_path_simplification, {"off", "exact", "toleranced"})
    _enum(error, "sweep_surface_realization", config.sweep_surface_realization, {"analytic", "mesh"})
    _enum(error, "sweep_bend_mode", config.sweep_bend_mode, {"miter", "fillet"})
    _enum(
        error,
        "sweep_frame_method",
        config.sweep_frame_method,
        {"surface_normal", "surface_constrained_parallel_transport"},
    )
    _enum(error, "sweep_self_intersection_check", config.sweep_self_intersection_check, {"off", "capsule_fast"})
    _positive(error, "conductor width", "conductor_cross_section_width", config.conductor_cross_section_width)
    _positive(error, "conductor height", "conductor_cross_section_height", config.conductor_cross_section_height)
    _at_least_float(
        error, "sweep collinearity tolerance", "sweep_collinearity_tolerance", config.sweep_collinearity_tolerance, 0.0
    )
    _at_least_float(
        error,
        "sweep tangent angle tolerance",
        "sweep_tangent_angle_tolerance_deg",
        config.sweep_tangent_angle_tolerance_deg,
        0.0,
    )
    _at_least_float(
        error,
        "sweep frame angle tolerance",
        "sweep_frame_angle_tolerance_deg",
        config.sweep_frame_angle_tolerance_deg,
        0.0,
    )
    _at_least_float(error, "sweep max segment length", "sweep_max_segment_length", config.sweep_max_segment_length, 0.0)
    _at_least_float(error, "sweep max chord error", "sweep_max_chord_error", config.sweep_max_chord_error, 0.0)
    _positive(
        error,
        "sweep max frame rotation",
        "sweep_max_frame_rotation_deg",
        config.sweep_max_frame_rotation_deg,
    )
    if float(config.sweep_max_frame_rotation_deg) > 180.0:
        error("sweep_max_frame_rotation_deg", "sweep max frame rotation must be <= 180 degrees.")
    _at_least_float(error, "sweep minimum bend radius", "sweep_min_bend_radius", config.sweep_min_bend_radius, 0.0)
    _positive(error, "sweep bend arc angle", "sweep_bend_arc_angle_deg", config.sweep_bend_arc_angle_deg)
    _positive(
        error,
        "sweep corner detection angle",
        "sweep_corner_detection_angle_deg",
        config.sweep_corner_detection_angle_deg,
    )
    if float(config.sweep_corner_detection_angle_deg) >= 180.0:
        error("sweep_corner_detection_angle_deg", "sweep corner detection angle must be < 180 degrees.")
    _positive(
        error,
        "sweep maximum curvature-envelope ratio",
        "sweep_max_curvature_envelope_ratio",
        config.sweep_max_curvature_envelope_ratio,
    )
    if float(config.sweep_max_curvature_envelope_ratio) >= 1.0:
        error(
            "sweep_max_curvature_envelope_ratio",
            "sweep maximum curvature-envelope ratio must be < 1.",
        )
    _at_least(
        error,
        "sweep normal smoothing iterations",
        "sweep_surface_normal_smoothing_iterations",
        config.sweep_surface_normal_smoothing_iterations,
        0,
    )
    _at_least_float(
        error,
        "sweep normal constraint strength",
        "sweep_surface_normal_constraint_strength",
        config.sweep_surface_normal_constraint_strength,
        0.0,
    )
    if float(config.sweep_surface_normal_constraint_strength) > 1.0:
        error("sweep_surface_normal_constraint_strength", "sweep normal constraint strength must be <= 1.")
    _at_least_float(
        error, "sweep insulation thickness", "sweep_insulation_thickness", config.sweep_insulation_thickness, 0.0
    )
    _at_least_float(
        error,
        "sweep manufacturing tolerance",
        "sweep_manufacturing_tolerance",
        config.sweep_manufacturing_tolerance,
        0.0,
    )
    _at_least_float(
        error, "sweep bend clearance margin", "sweep_bend_clearance_margin", config.sweep_bend_clearance_margin, 0.0
    )
    if (
        not config.skip_sweep
        and not bool(getattr(config, "contour_only_sweep_mode", False))
        and str(getattr(config, "layout_terminal_strategy", "undefined")).strip().lower() == "undefined"
    ):
        warning(
            "layout_terminal_strategy",
            "Swept-body export is enabled without explicit terminal handling.",
            "Define terminal pads/wire stubs or mark terminal handling as waived.",
        )
    if bool(getattr(config, "contour_only_sweep_mode", False)) and bool(config.skip_sweep):
        warning(
            "contour_only_sweep_mode",
            "Contour-only sweep mode is enabled while sweep generation is disabled.",
            "Set skip_sweep=False to generate independent closed-contour bodies.",
        )
    if float(config.conductor_cross_section_width) > 0.05 or float(config.conductor_cross_section_height) > 0.05:
        warning("conductor dimensions", "Conductor dimensions exceed 5 cm.", "Check that SI meters are being used.")
    if float(config.min_turn_distance) > 0.0 and float(config.min_turn_distance) < max(
        float(config.conductor_cross_section_width), float(config.conductor_cross_section_height)
    ):
        warning(
            "min_turn_distance",
            "Requested turn spacing is smaller than the conductor cross-section dimension.",
            "Use a spacing at least as large as the physical conductor width/height.",
        )
    conductor_envelope, conductor_description = _conductor_envelope_diameter(config)
    required_centerline_clearance = (
        conductor_envelope
        + 2.0 * float(config.sweep_insulation_thickness)
        + float(config.sweep_manufacturing_tolerance)
        + float(config.sweep_bend_clearance_margin)
    )
    requested_centerline_clearance = max(
        float(getattr(config, "min_turn_distance", 0.0)),
        float(getattr(config, "min_wire_distance_m", 0.0)),
    )
    if requested_centerline_clearance > 0.0 and requested_centerline_clearance < required_centerline_clearance:
        warning(
            "min_turn_distance",
            "Requested centerline spacing is below the swept conductor/insulation/tolerance envelope.",
            f"Use at least {required_centerline_clearance:.6g} m for the {conductor_description}.",
        )
    if str(config.conductor_mounting_mode).strip().lower().replace("-", "_") != "centered":
        warning(
            "conductor_mounting_mode",
            "Manufacturing conductor centerline is offset from the electromagnetic wire path.",
            "Use centered mode for exact field-evaluation consistency, or re-evaluate fields with the shifted manufacturing path.",
        )


def _conductor_envelope_diameter(config: CoilGenConfig) -> tuple[float, str]:
    """Return the radial sweep envelope for either a custom polygon or rectangle."""

    points = np.asarray(config.cross_sectional_points, dtype=float)
    if points.ndim == 2:
        if points.shape[0] == 2 and points.shape[1] != 2:
            points = points.T
        if points.shape[1] == 2 and len(points) >= 3 and np.all(np.isfinite(points)):
            x = points[:, 0]
            y = points[:, 1]
            area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            if area > 0.0:
                return 2.0 * float(np.max(np.linalg.norm(points, axis=1))), "configured polygonal conductor"
    return (
        float(np.hypot(config.conductor_cross_section_width, config.conductor_cross_section_height)),
        "configured rectangular conductor",
    )


def _validate_parameter_list(error, field: str, values: tuple[Any, ...], expected_len: int) -> None:
    if len(values) != expected_len:
        error(field, f"Expected {expected_len} values, got {len(values)}.")
        return
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        error(field, "Parameter list must contain numeric values.")
        return
    if not np.all(np.isfinite(numeric)):
        error(field, "Parameter list contains non-finite values.")


def _enum(error, field: str, value: Any, allowed: set[str]) -> None:
    text = str(value).strip().lower().replace("-", "_")
    normalised_allowed = {item.strip().lower().replace("-", "_") for item in allowed}
    if text not in normalised_allowed:
        error(field, f"Unsupported value {value!r}.", f"Allowed values: {', '.join(sorted(allowed))}.")


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _positive(error, quantity: str, field: str, value: Any) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        error(field, f"{quantity} must be a finite positive number.")
        return
    if not np.isfinite(number) or number <= 0.0:
        error(field, f"{quantity} must be a finite positive number.")


def _at_least(error, quantity: str, field: str, value: Any, minimum: int) -> None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        error(field, f"{quantity} must be an integer >= {minimum}.")
        return
    if number < int(minimum):
        error(field, f"{quantity} must be an integer >= {minimum}.")


def _at_least_float(error, quantity: str, field: str, value: Any, minimum: float) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        error(field, f"{quantity} must be a finite number >= {minimum:g}.")
        return
    if not np.isfinite(number) or number < float(minimum):
        error(field, f"{quantity} must be a finite number >= {minimum:g}.")


def _validate_gradient_slice_grid_size(error, value: Any) -> None:
    try:
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                error("gradient_slice_map_grid_size", "Grid size must be an integer or a two-entry sequence.")
                return
            x_count = int(value[0])
            y_count = int(value[1])
        else:
            x_count = y_count = int(value)
    except (TypeError, ValueError):
        error("gradient_slice_map_grid_size", "Grid size must be an integer or a two-entry sequence.")
        return
    if x_count < 8 or y_count < 8:
        error("gradient_slice_map_grid_size", "Gradient slice-map grid dimensions must be >= 8 pixels.")


def _validate_vector3(error, field: str, value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        error(field, "Value must contain three finite numeric entries.")
        return None
    if vector.shape != (3,):
        error(field, "Value must contain exactly three entries.")
        return None
    if not np.all(np.isfinite(vector)):
        error(field, "Value must contain finite numeric entries.")
        return None
    return vector
