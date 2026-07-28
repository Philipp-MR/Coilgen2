from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .field_components import normalize_field_component, normalized_b0_direction
from .models import Mesh, TargetField


PROBLEM_SCHEMA_VERSION = "coilgen-problem-v1"

FieldComponent = Literal["Bx", "By", "Bz", "BdotB0", "vector", "Bmag"]
MeshRole = Literal["primary", "shield", "suppression", "auxiliary"]
RegionRole = Literal["useful_field", "suppression", "diagnostic_only", "engineering_limit"]
TerminalStrategy = Literal["undefined", "waived", "pads", "wire_stubs"]


def _as_points(value: Any, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must be an (n, 3) array in SI coordinates [m].")
    return points


def _as_field(value: Any, *, point_count: int, name: str) -> np.ndarray:
    field_values = np.asarray(value, dtype=float)
    if field_values.ndim == 1:
        if len(field_values) != point_count:
            raise ValueError(f"{name} vector length must match the number of points.")
        expanded = np.zeros((point_count, 3), dtype=float)
        expanded[:, 2] = field_values
        return expanded
    if field_values.shape != (point_count, 3):
        raise ValueError(f"{name} must be an (n, 3) field array [T] or an n-vector for Bz [T].")
    return field_values


def _as_weights(value: Any | None, *, point_count: int, name: str) -> np.ndarray:
    if value is None:
        return np.ones(point_count, dtype=float)
    weights = np.asarray(value, dtype=float).reshape(-1)
    if len(weights) != point_count:
        raise ValueError(f"{name} must have one weight per point.")
    if np.any(weights < 0):
        raise ValueError(f"{name} must be non-negative.")
    return weights


@dataclass(frozen=True, slots=True)
class PipelineGeometrySpec:
    """Current-pipeline geometry mapping for supported generated geometries.

    This keeps high-level problem objects explicit: if a support body was made
    by one of CoilGen's supported generators, conversion can preserve that
    generator and its parameters instead of falling back to an in-memory mesh.
    """

    coil_mesh_file: str
    surface_is_cylinder: bool
    parameters: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.coil_mesh_file).strip():
            raise ValueError("PipelineGeometrySpec.coil_mesh_file must be non-empty.")

    def to_config_overrides(self) -> dict[str, Any]:
        values = dict(self.parameters)
        values["coil_mesh_file"] = self.coil_mesh_file
        values["surface_is_cylinder_flag"] = bool(self.surface_is_cylinder)
        return values

    @classmethod
    def from_config_values(
        cls,
        coil_mesh_file: str,
        *,
        surface_is_cylinder: bool,
        parameters: Mapping[str, Any] | None = None,
        description: str = "",
    ) -> "PipelineGeometrySpec":
        return cls(
            coil_mesh_file=coil_mesh_file,
            surface_is_cylinder=surface_is_cylinder,
            parameters={} if parameters is None else dict(parameters),
            description=description,
        )

    @classmethod
    def from_coil_design(cls, design: Any) -> "PipelineGeometrySpec":
        """Create a geometry spec from an existing user-facing design class."""

        values = dict(design.geometry_config())
        mesh_file = str(values.pop("coil_mesh_file"))
        surface_is_cylinder = bool(values.pop("surface_is_cylinder_flag", False))
        return cls(
            coil_mesh_file=mesh_file,
            surface_is_cylinder=surface_is_cylinder,
            parameters=values,
            description=type(design).__name__,
        )

    @classmethod
    def cylinder(
        cls,
        *,
        length_m: float,
        radius_m: float,
        angular_divisions: int,
        length_divisions: int,
        rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        rotation_angle_rad: float = 0.0,
    ) -> "PipelineGeometrySpec":
        return cls(
            "create cylinder mesh",
            True,
            {
                "cylinder_mesh_parameter_list": (
                    length_m,
                    radius_m,
                    angular_divisions,
                    length_divisions,
                    *tuple(rotation_axis),
                    rotation_angle_rad,
                )
            },
            "cylindrical surface",
        )

    @classmethod
    def planar(
        cls,
        *,
        width_m: float,
        height_m: float,
        lateral_divisions: int,
        longitudinal_divisions: int,
        rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        rotation_angle_rad: float = 0.0,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create planary mesh",
            False,
            {
                "planar_mesh_parameter_list": (
                    height_m,
                    width_m,
                    lateral_divisions,
                    longitudinal_divisions,
                    *tuple(rotation_axis),
                    rotation_angle_rad,
                    *tuple(center_m),
                )
            },
            "rectangular planar surface",
        )

    @classmethod
    def circular(
        cls,
        *,
        radius_m: float,
        radial_divisions: int,
        rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        rotation_angle_rad: float = 0.0,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create circular mesh",
            False,
            {
                "circular_mesh_parameter_list": (
                    radius_m,
                    radial_divisions,
                    *tuple(rotation_axis),
                    rotation_angle_rad,
                    *tuple(center_m),
                )
            },
            "circular planar surface",
        )

    @classmethod
    def annular_disc(
        cls,
        *,
        inner_radius_m: float,
        outer_radius_m: float,
        radial_divisions: int,
        circular_divisions: int,
        normal: Sequence[float] = (0.0, 0.0, 1.0),
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create annular disc mesh",
            False,
            {
                "annular_disc_mesh_parameter_list": (
                    inner_radius_m,
                    outer_radius_m,
                    radial_divisions,
                    circular_divisions,
                    *tuple(normal),
                    *tuple(center_m),
                )
            },
            "annular planar surface",
        )

    @classmethod
    def partial_cylinder(
        cls,
        *,
        length_m: float,
        radius_m: float,
        start_angle_rad: float,
        end_angle_rad: float,
        angular_divisions: int,
        length_divisions: int,
        rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        rotation_angle_rad: float = 0.0,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create partial cylinder mesh",
            True,
            {
                "partial_cylinder_mesh_parameter_list": (
                    length_m,
                    radius_m,
                    start_angle_rad,
                    end_angle_rad,
                    angular_divisions,
                    length_divisions,
                    *tuple(rotation_axis),
                    rotation_angle_rad,
                    *tuple(center_m),
                )
            },
            "open partial cylindrical surface",
        )

    @classmethod
    def elliptical_cylinder(
        cls,
        *,
        length_m: float,
        radius_x_m: float,
        radius_y_m: float,
        angular_divisions: int,
        length_divisions: int,
        rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        rotation_angle_rad: float = 0.0,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create elliptical cylinder mesh",
            True,
            {
                "elliptical_cylinder_mesh_parameter_list": (
                    length_m,
                    radius_x_m,
                    radius_y_m,
                    angular_divisions,
                    length_divisions,
                    *tuple(rotation_axis),
                    rotation_angle_rad,
                    *tuple(center_m),
                )
            },
            "elliptical cylindrical surface",
        )

    @classmethod
    def superellipse_cylinder(
        cls,
        *,
        length_m: float,
        radius_x_m: float,
        radius_y_m: float,
        exponent: float,
        angular_divisions: int,
        length_divisions: int,
        rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        rotation_angle_rad: float = 0.0,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create superellipse cylinder mesh",
            True,
            {
                "superellipse_cylinder_mesh_parameter_list": (
                    length_m,
                    radius_x_m,
                    radius_y_m,
                    exponent,
                    angular_divisions,
                    length_divisions,
                    *tuple(rotation_axis),
                    rotation_angle_rad,
                    *tuple(center_m),
                )
            },
            "superellipse cylindrical surface",
        )

    @classmethod
    def spherical_cap(
        cls,
        *,
        radius_m: float,
        cap_polar_angle_rad: float,
        azimuthal_divisions: int,
        polar_divisions: int,
        rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        rotation_angle_rad: float = 0.0,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create spherical cap mesh",
            False,
            {
                "spherical_cap_mesh_parameter_list": (
                    radius_m,
                    cap_polar_angle_rad,
                    azimuthal_divisions,
                    polar_divisions,
                    *tuple(rotation_axis),
                    rotation_angle_rad,
                    *tuple(center_m),
                )
            },
            "spherical cap surface",
        )

    @classmethod
    def ellipsoidal_cap(
        cls,
        *,
        radius_x_m: float,
        radius_y_m: float,
        radius_z_m: float,
        cap_polar_angle_rad: float,
        azimuthal_divisions: int,
        polar_divisions: int,
        rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        rotation_angle_rad: float = 0.0,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create ellipsoidal cap mesh",
            False,
            {
                "ellipsoidal_cap_mesh_parameter_list": (
                    radius_x_m,
                    radius_y_m,
                    radius_z_m,
                    cap_polar_angle_rad,
                    azimuthal_divisions,
                    polar_divisions,
                    *tuple(rotation_axis),
                    rotation_angle_rad,
                    *tuple(center_m),
                )
            },
            "ellipsoidal cap surface",
        )

    @classmethod
    def conical_frustum(
        cls,
        *,
        length_m: float,
        radius_bottom_m: float,
        radius_top_m: float,
        angular_divisions: int,
        length_divisions: int,
        rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        rotation_angle_rad: float = 0.0,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create conical frustum mesh",
            True,
            {
                "conical_frustum_mesh_parameter_list": (
                    length_m,
                    radius_bottom_m,
                    radius_top_m,
                    angular_divisions,
                    length_divisions,
                    *tuple(rotation_axis),
                    rotation_angle_rad,
                    *tuple(center_m),
                )
            },
            "conical frustum surface",
        )

    @classmethod
    def toroidal_segment(
        cls,
        *,
        major_radius_m: float,
        minor_radius_m: float,
        start_angle_rad: float,
        end_angle_rad: float,
        tube_start_angle_rad: float,
        tube_end_angle_rad: float,
        major_divisions: int,
        minor_divisions: int,
        rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        rotation_angle_rad: float = 0.0,
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create toroidal segment mesh",
            False,
            {
                "toroidal_segment_mesh_parameter_list": (
                    major_radius_m,
                    minor_radius_m,
                    start_angle_rad,
                    end_angle_rad,
                    tube_start_angle_rad,
                    tube_end_angle_rad,
                    major_divisions,
                    minor_divisions,
                    *tuple(rotation_axis),
                    rotation_angle_rad,
                    *tuple(center_m),
                )
            },
            "toroidal segment surface",
        )

    @classmethod
    def ruled_surface(
        cls,
        *,
        curve_a_m: Sequence[Sequence[float]],
        curve_b_m: Sequence[Sequence[float]],
        span_divisions: int,
    ) -> "PipelineGeometrySpec":
        curve_a = _as_points(curve_a_m, name="curve_a_m")
        curve_b = _as_points(curve_b_m, name="curve_b_m")
        if len(curve_a) != len(curve_b):
            raise ValueError("ruled_surface curves must have the same number of samples.")
        flattened: list[float] = [float(len(curve_a)), float(span_divisions)]
        flattened.extend(float(value) for value in curve_a.reshape(-1))
        flattened.extend(float(value) for value in curve_b.reshape(-1))
        return cls(
            "create ruled surface mesh",
            False,
            {"ruled_surface_mesh_parameter_list": tuple(flattened)},
            "ruled surface between two boundary curves",
        )

    @classmethod
    def surface_of_revolution(
        cls,
        *,
        circular_divisions: int,
        z_radius_pairs_m: Sequence[Sequence[float]],
    ) -> "PipelineGeometrySpec":
        flattened: list[float] = [float(circular_divisions)]
        for z_value, radius in z_radius_pairs_m:
            flattened.extend((float(z_value), float(radius)))
        return cls(
            "create surface of revolution mesh",
            True,
            {"surface_of_revolution_mesh_parameter_list": tuple(flattened)},
            "surface of revolution",
        )

    @classmethod
    def biplanar(
        cls,
        *,
        width_m: float,
        height_m: float,
        lateral_divisions: int,
        longitudinal_divisions: int,
        plane_distance_m: float,
        normal: Sequence[float] = (0.0, 0.0, 1.0),
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create bi-planary mesh",
            False,
            {
                "biplanar_mesh_parameter_list": (
                    height_m,
                    width_m,
                    lateral_divisions,
                    longitudinal_divisions,
                    *tuple(normal),
                    *tuple(center_m),
                    plane_distance_m,
                )
            },
            "rectangular biplanar surface",
        )

    @classmethod
    def circular_biplanar(
        cls,
        *,
        outer_radius_m: float,
        edge_length_m: float,
        plane_distance_m: float,
        normal: Sequence[float] = (0.0, 0.0, 1.0),
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create circular bi-planary mesh",
            False,
            {
                "circular_biplanar_mesh_parameter_list": (
                    edge_length_m,
                    outer_radius_m,
                    *tuple(normal),
                    *tuple(center_m),
                    plane_distance_m,
                )
            },
            "circular biplanar surface",
        )

    @classmethod
    def shielded_circular_biplanar(
        cls,
        *,
        primary_outer_radius_m: float,
        primary_edge_length_m: float,
        shield_outer_radius_m: float,
        shield_edge_length_m: float,
        plane_distance_m: float,
        shield_gap_m: float,
        normal: Sequence[float] = (0.0, 0.0, 1.0),
        center_m: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "PipelineGeometrySpec":
        return cls(
            "create shielded bi-planary mesh",
            False,
            {
                "shielded_circular_biplanar_mesh_parameter_list": (
                    primary_outer_radius_m,
                    primary_edge_length_m,
                    shield_outer_radius_m,
                    shield_edge_length_m,
                    *tuple(normal),
                    *tuple(center_m),
                    plane_distance_m,
                    shield_gap_m,
                )
            },
            "shielded circular biplanar surface",
        )

    @classmethod
    def flanged_split_bore(
        cls,
        *,
        edge_length_m: float,
        inner_radius_m: float,
        outer_radius_m: float,
        gap_m: float,
        half_length_m: float,
        b0_direction: Sequence[float] = (0.0, 0.0, 1.0),
        mesh_rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        mesh_rotation_angle_rad: float = 0.0,
    ) -> "PipelineGeometrySpec":
        return cls(
            "create flanged split bore mesh",
            True,
            {
                "flanged_split_bore_mesh_parameter_list": (
                    edge_length_m,
                    inner_radius_m,
                    outer_radius_m,
                    gap_m,
                    half_length_m,
                ),
                "b_0_direction": tuple(b0_direction),
                "generated_mesh_rotation_axis": tuple(mesh_rotation_axis),
                "generated_mesh_rotation_angle": mesh_rotation_angle_rad,
            },
            "flanged split-bore surface",
        )

    @classmethod
    def shielded_flanged_split_bore(
        cls,
        *,
        edge_length_m: float,
        inner_radius_m: float,
        outer_radius_m: float,
        gap_m: float,
        half_length_m: float,
        shield_edge_length_m: float,
        shield_inner_radius_m: float,
        shield_outer_radius_m: float,
        shield_gap_m: float,
        shield_half_length_m: float,
        b0_direction: Sequence[float] = (0.0, 0.0, 1.0),
        mesh_rotation_axis: Sequence[float] = (1.0, 0.0, 0.0),
        mesh_rotation_angle_rad: float = 0.0,
    ) -> "PipelineGeometrySpec":
        return cls(
            "create shielded flanged split bore mesh",
            True,
            {
                "shielded_flanged_split_bore_mesh_parameter_list": (
                    edge_length_m,
                    inner_radius_m,
                    outer_radius_m,
                    gap_m,
                    half_length_m,
                    shield_edge_length_m,
                    shield_inner_radius_m,
                    shield_outer_radius_m,
                    shield_gap_m,
                    shield_half_length_m,
                ),
                "b_0_direction": tuple(b0_direction),
                "generated_mesh_rotation_axis": tuple(mesh_rotation_axis),
                "generated_mesh_rotation_angle": mesh_rotation_angle_rad,
            },
            "shielded flanged split-bore surface",
        )

    @classmethod
    def imported_mesh(
        cls,
        *,
        mesh_file: str,
        geometry_source_path: str | Path | None = None,
        surface_is_cylinder: bool = False,
    ) -> "PipelineGeometrySpec":
        parameters: dict[str, Any] = {}
        if geometry_source_path is not None:
            parameters["geometry_source_path"] = Path(geometry_source_path)
        return cls(mesh_file, surface_is_cylinder, parameters, "imported STL mesh")

    @staticmethod
    def supported_geometry_names() -> tuple[str, ...]:
        """Geometry names with explicit current-pipeline conversion support."""

        return (
            "cylinder",
            "planar",
            "circular",
            "annular_disc",
            "partial_cylinder",
            "elliptical_cylinder",
            "superellipse_cylinder",
            "spherical_cap",
            "ellipsoidal_cap",
            "conical_frustum",
            "toroidal_segment",
            "ruled_surface",
            "surface_of_revolution",
            "biplanar",
            "circular_biplanar",
            "shielded_circular_biplanar",
            "flanged_split_bore",
            "shielded_flanged_split_bore",
            "imported_mesh",
        )


@dataclass(frozen=True, slots=True)
class DesignMetadata:
    """Human-readable bookkeeping for a coil design or parameter-study point."""

    name: str = "coil_design"
    project: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return self.name if not self.project else f"{self.project}: {self.name}"


@dataclass(frozen=True, slots=True)
class FieldComponentSpec:
    """Field component used for target fitting and reporting.

    ``component`` accepts ``Bx``, ``By``, ``Bz``, ``BdotB0`` and ``vector`` for
    optimization targets. ``BdotB0`` uses ``b0_direction`` as the projection
    axis in scanner coordinates.
    """

    component: str = "Bz"
    b0_direction: Sequence[float] = (0.0, 0.0, 1.0)
    description: str = ""

    def __post_init__(self) -> None:
        resolved = normalize_field_component(self.component)
        direction = tuple(float(value) for value in normalized_b0_direction(self.b0_direction))
        object.__setattr__(self, "component", resolved)
        object.__setattr__(self, "b0_direction", direction)

    def to_config_value(self) -> str:
        return str(self.component)


def _component_spec(value: FieldComponentSpec | str | None) -> FieldComponentSpec:
    if isinstance(value, FieldComponentSpec):
        return value
    return FieldComponentSpec("Bz" if value is None else str(value))


@dataclass(frozen=True, slots=True)
class TargetRegion:
    """Named target-field region with explicit optimization semantics."""

    name: str
    role: RegionRole
    coords: np.ndarray
    target_b: np.ndarray | None = None
    component: FieldComponentSpec | str = field(default_factory=FieldComponentSpec)
    weight: float = 1.0
    hard_limit: float | None = None
    units: str = "T"
    metric_policy: str = "default"
    group_index: int = 1
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed_roles = {"useful_field", "suppression", "diagnostic_only", "engineering_limit"}
        if self.role not in allowed_roles:
            known = ", ".join(sorted(allowed_roles))
            raise ValueError(f"TargetRegion.role must be one of: {known}.")
        coords = _as_points(self.coords, name=f"{self.name}.coords")
        object.__setattr__(self, "coords", coords)
        target_b = (
            np.zeros((len(coords), 3), dtype=float)
            if self.target_b is None
            else _as_field(self.target_b, point_count=len(coords), name=f"{self.name}.target_b")
        )
        object.__setattr__(self, "target_b", target_b)
        object.__setattr__(self, "component", _component_spec(self.component))
        if float(self.weight) < 0:
            raise ValueError("target-region weight must be non-negative.")
        if self.hard_limit is not None and float(self.hard_limit) < 0:
            raise ValueError("target-region hard_limit must be non-negative when provided.")
        if int(self.group_index) < 1:
            raise ValueError("target-region group_index must be positive.")

    @property
    def point_count(self) -> int:
        return int(len(self.coords))

    @property
    def participates_in_optimization(self) -> bool:
        return self.role != "diagnostic_only"

    @property
    def suppressed_in_pipeline(self) -> bool:
        return self.role in {"suppression", "engineering_limit"}

    def weights(self) -> np.ndarray:
        return np.full(self.point_count, float(self.weight), dtype=float)

    def as_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "point_count": self.point_count,
            "component": self.component.component,
            "weight": float(self.weight),
            "hard_limit": None if self.hard_limit is None else float(self.hard_limit),
            "units": self.units,
            "group_index": int(self.group_index),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class SuppressionRegion:
    """Field-suppression samples, typically representing sensitive magnet surfaces.

    Coordinates are SI positions in metres. The target field defaults to zero
    field, which is the normal active-shielding use case.
    """

    name: str
    coords: np.ndarray
    target_b: np.ndarray | None = None
    weight: float = 1.0
    component: FieldComponent = "Bz"
    group_index: int = 2
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = _as_points(self.coords, name=f"{self.name}.coords")
        object.__setattr__(self, "coords", coords)
        if self.target_b is None:
            target_b = np.zeros((len(coords), 3), dtype=float)
        else:
            target_b = _as_field(self.target_b, point_count=len(coords), name=f"{self.name}.target_b")
        object.__setattr__(self, "target_b", target_b)
        object.__setattr__(self, "component", normalize_field_component(self.component))
        if float(self.weight) < 0:
            raise ValueError("suppression-region weight must be non-negative.")
        if int(self.group_index) < 1:
            raise ValueError("suppression-region group_index must be positive.")

    @property
    def point_count(self) -> int:
        return int(len(self.coords))

    def weights(self) -> np.ndarray:
        return np.full(self.point_count, float(self.weight), dtype=float)


@dataclass(frozen=True, slots=True)
class TargetFieldSpec:
    """Target-field definition with DSV samples and optional suppression regions.

    This class is the user-facing representation of target-field data. It
    can be converted to the internal ``TargetField`` model used by the solver.
    DSV and suppression points stay explicitly separated for evaluation.
    """

    name: str
    dsv_coords: np.ndarray
    dsv_target_b: np.ndarray
    dsv_weights: np.ndarray | None = None
    suppression_regions: tuple[SuppressionRegion, ...] = ()
    target_gradient_dbdxyz: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    component: FieldComponentSpec | str = field(default_factory=FieldComponentSpec)
    regions: tuple[TargetRegion, ...] = ()
    units: str = "T"

    def __post_init__(self) -> None:
        dsv_coords = _as_points(self.dsv_coords, name=f"{self.name}.dsv_coords")
        object.__setattr__(self, "dsv_coords", dsv_coords)
        dsv_target_b = _as_field(self.dsv_target_b, point_count=len(dsv_coords), name=f"{self.name}.dsv_target_b")
        object.__setattr__(self, "dsv_target_b", dsv_target_b)
        dsv_weights = _as_weights(self.dsv_weights, point_count=len(dsv_coords), name=f"{self.name}.dsv_weights")
        object.__setattr__(self, "dsv_weights", dsv_weights)
        object.__setattr__(self, "suppression_regions", tuple(self.suppression_regions))
        object.__setattr__(self, "component", _component_spec(self.component))
        object.__setattr__(self, "regions", tuple(self.regions))
        if self.target_gradient_dbdxyz is not None:
            gradient = np.asarray(self.target_gradient_dbdxyz, dtype=float)
            if gradient.shape != (len(dsv_coords), 3):
                raise ValueError("target_gradient_dbdxyz must be an (n_dsv, 3) array.")
            object.__setattr__(self, "target_gradient_dbdxyz", gradient)

    @classmethod
    def from_arrays(
        cls,
        coords: Any,
        target_b: Any,
        *,
        name: str = "target_field",
        weights: Any | None = None,
        suppression: Sequence[SuppressionRegion] = (),
        target_gradient_dbdxyz: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        component: FieldComponentSpec | str = "Bz",
        regions: Sequence[TargetRegion] = (),
        units: str = "T",
    ) -> "TargetFieldSpec":
        return cls(
            name=name,
            dsv_coords=np.asarray(coords, dtype=float),
            dsv_target_b=np.asarray(target_b, dtype=float),
            dsv_weights=None if weights is None else np.asarray(weights, dtype=float),
            suppression_regions=tuple(suppression),
            target_gradient_dbdxyz=None
            if target_gradient_dbdxyz is None
            else np.asarray(target_gradient_dbdxyz, dtype=float),
            metadata={} if metadata is None else dict(metadata),
            component=component,
            regions=tuple(regions),
            units=units,
        )

    @classmethod
    def from_regions(
        cls,
        regions: Sequence[TargetRegion],
        *,
        name: str = "target_field",
        component: FieldComponentSpec | str | None = None,
        target_gradient_dbdxyz: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
        units: str = "T",
    ) -> "TargetFieldSpec":
        semantic_regions = tuple(regions)
        component_spec = _component_spec(component or (semantic_regions[0].component if semantic_regions else "Bz"))
        useful_regions = tuple(region for region in semantic_regions if region.role == "useful_field")
        constrained_regions = tuple(
            region for region in semantic_regions if region.role in {"suppression", "engineering_limit"}
        )
        if useful_regions:
            dsv_coords = np.vstack([region.coords for region in useful_regions])
            dsv_target_b = np.vstack([region.target_b for region in useful_regions])
            dsv_weights = np.concatenate([region.weights() for region in useful_regions])
        else:
            dsv_coords = np.empty((0, 3), dtype=float)
            dsv_target_b = np.empty((0, 3), dtype=float)
            dsv_weights = np.empty((0,), dtype=float)

        suppression: list[SuppressionRegion] = []
        next_group = 2
        for region in constrained_regions:
            group_index = int(region.group_index)
            if group_index < 2:
                group_index = next_group
            next_group = max(next_group, group_index + 1)
            suppression.append(
                SuppressionRegion(
                    name=region.name,
                    coords=region.coords,
                    target_b=region.target_b,
                    weight=region.weight,
                    component=region.component.component,
                    group_index=group_index,
                    source=region.source,
                    metadata={
                        **dict(region.metadata),
                        "region_role": region.role,
                        "hard_limit": region.hard_limit,
                        "metric_policy": region.metric_policy,
                    },
                )
            )

        return cls(
            name=name,
            dsv_coords=dsv_coords,
            dsv_target_b=dsv_target_b,
            dsv_weights=dsv_weights,
            suppression_regions=tuple(suppression),
            target_gradient_dbdxyz=None
            if target_gradient_dbdxyz is None
            else np.asarray(target_gradient_dbdxyz, dtype=float),
            metadata={} if metadata is None else dict(metadata),
            component=component_spec,
            regions=semantic_regions,
            units=units,
        )

    @classmethod
    def from_target_field(cls, target_field: TargetField, *, name: str = "target_field") -> "TargetFieldSpec":
        suppressed = np.asarray(target_field.suppressed, dtype=bool)
        dsv_mask = ~suppressed
        regions: list[SuppressionRegion] = []
        for group_index in sorted(set(np.asarray(target_field.target_field_group_inds, dtype=np.int64)[suppressed])):
            group_mask = suppressed & (np.asarray(target_field.target_field_group_inds, dtype=np.int64) == group_index)
            regions.append(
                SuppressionRegion(
                    name=f"suppression_group_{int(group_index)}",
                    coords=target_field.coords[group_mask],
                    target_b=target_field.b[group_mask],
                    weight=float(np.mean(target_field.weights[group_mask])) if np.any(group_mask) else 1.0,
                    group_index=int(group_index),
                )
            )
        gradient = None
        if target_field.target_gradient_dbdxyz is not None:
            gradient = target_field.target_gradient_dbdxyz[dsv_mask]
        return cls(
            name=name,
            dsv_coords=target_field.coords[dsv_mask],
            dsv_target_b=target_field.b[dsv_mask],
            dsv_weights=target_field.weights[dsv_mask],
            suppression_regions=tuple(regions),
            target_gradient_dbdxyz=gradient,
        )

    @property
    def dsv_point_count(self) -> int:
        return int(len(self.dsv_coords))

    @property
    def suppression_point_count(self) -> int:
        return int(sum(region.point_count for region in self.suppression_regions))

    @property
    def diagnostic_point_count(self) -> int:
        return int(sum(region.point_count for region in self.regions if region.role == "diagnostic_only"))

    @property
    def point_count(self) -> int:
        return self.dsv_point_count + self.suppression_point_count

    def with_suppression_region(self, region: SuppressionRegion) -> "TargetFieldSpec":
        return replace(self, suppression_regions=self.suppression_regions + (region,))

    def regions_by_role(self, role: RegionRole) -> tuple[TargetRegion, ...]:
        return tuple(region for region in self.regions if region.role == role)

    def optimization_regions(self) -> tuple[TargetRegion, ...]:
        return tuple(region for region in self.regions if region.participates_in_optimization)

    def diagnostic_regions(self) -> tuple[TargetRegion, ...]:
        return self.regions_by_role("diagnostic_only")

    def region_summaries(self) -> tuple[dict[str, Any], ...]:
        if self.regions:
            return tuple(region.as_summary() for region in self.regions)
        summaries: list[dict[str, Any]] = [
            {
                "name": "dsv",
                "role": "useful_field",
                "point_count": self.dsv_point_count,
                "component": self.component.component,
                "weight": float(np.mean(self.dsv_weights)) if self.dsv_point_count else 0.0,
                "hard_limit": None,
                "units": self.units,
                "group_index": 1,
                "source": "",
            }
        ]
        summaries.extend(
            {
                "name": region.name,
                "role": str(region.metadata.get("region_role", "suppression")),
                "point_count": region.point_count,
                "component": str(region.component),
                "weight": float(region.weight),
                "hard_limit": region.metadata.get("hard_limit"),
                "units": self.units,
                "group_index": int(region.group_index),
                "source": region.source,
            }
            for region in self.suppression_regions
        )
        return tuple(summaries)

    def combined_coords(self) -> np.ndarray:
        if not self.suppression_regions:
            return self.dsv_coords.copy()
        return np.vstack([self.dsv_coords] + [region.coords for region in self.suppression_regions])

    def combined_target_b(self) -> np.ndarray:
        if not self.suppression_regions:
            return self.dsv_target_b.copy()
        return np.vstack([self.dsv_target_b] + [region.target_b for region in self.suppression_regions])

    def combined_weights(self) -> np.ndarray:
        if not self.suppression_regions:
            return self.dsv_weights.copy()
        return np.concatenate([self.dsv_weights] + [region.weights() for region in self.suppression_regions])

    def suppression_mask(self) -> np.ndarray:
        mask = np.zeros(self.point_count, dtype=bool)
        if self.suppression_point_count:
            mask[self.dsv_point_count :] = True
        return mask

    def target_group_indices(self) -> np.ndarray:
        groups = [np.ones(self.dsv_point_count, dtype=np.int64)]
        groups.extend(
            np.full(region.point_count, int(region.group_index), dtype=np.int64) for region in self.suppression_regions
        )
        return np.concatenate(groups)

    def to_target_field(self) -> TargetField:
        gradient = None
        if self.target_gradient_dbdxyz is not None:
            gradient = np.vstack(
                [self.target_gradient_dbdxyz]
                + [np.zeros((region.point_count, 3), dtype=float) for region in self.suppression_regions]
            )
        return TargetField(
            coords=self.combined_coords(),
            b=self.combined_target_b(),
            weights=self.combined_weights(),
            suppressed=self.suppression_mask(),
            target_field_group_inds=self.target_group_indices(),
            target_gradient_dbdxyz=gradient,
        )


@dataclass(frozen=True, slots=True)
class ConductorSpec:
    """Physical conductor parameters used for resistance and layout constraints."""

    width_m: float = 0.002
    height_m: float = 0.002
    resistivity_ohm_m: float = 1.8e-8
    insulation_thickness_m: float = 0.0
    min_turn_spacing_m: float = 0.0
    material: str = "copper"

    def __post_init__(self) -> None:
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("Conductor width and height must be positive SI lengths [m].")
        if self.resistivity_ohm_m <= 0:
            raise ValueError("Conductor resistivity must be positive [Ohm m].")
        if self.insulation_thickness_m < 0 or self.min_turn_spacing_m < 0:
            raise ValueError("Conductor insulation and minimum spacing cannot be negative.")

    @property
    def cross_section_area_m2(self) -> float:
        return float(self.width_m * self.height_m)

    @property
    def equivalent_round_radius_m(self) -> float:
        return float(np.sqrt(self.cross_section_area_m2 / np.pi))

    def resistance_ohm(self, length_m: float) -> float:
        return float(self.resistivity_ohm_m * float(length_m) / self.cross_section_area_m2)


@dataclass(frozen=True, slots=True)
class SupportMesh:
    """A mesh plus its physical role in the coil design problem."""

    mesh: Mesh
    role: MeshRole = "primary"
    name: str = ""
    part_index: int = 0
    paired_part_index: int | None = None
    separation_from_primary_m: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return self.mesh.n_vertices

    @property
    def face_count(self) -> int:
        return self.mesh.n_faces

    @property
    def bounds_m(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mesh.vertices.min(axis=0), self.mesh.vertices.max(axis=0)

    def has_same_topology(self, other: "SupportMesh") -> bool:
        return self.mesh.vertices.shape == other.mesh.vertices.shape and np.array_equal(
            self.mesh.faces, other.mesh.faces
        )

    def topology_report(self, other: "SupportMesh") -> dict[str, bool | int | str]:
        same_vertices = self.mesh.vertices.shape == other.mesh.vertices.shape
        same_faces = self.mesh.faces.shape == other.mesh.faces.shape and np.array_equal(
            self.mesh.faces, other.mesh.faces
        )
        return {
            "mesh": self.name or str(self.part_index),
            "other_mesh": other.name or str(other.part_index),
            "same_vertex_count": bool(same_vertices),
            "same_face_connectivity": bool(same_faces),
            "vertex_count": int(self.mesh.n_vertices),
            "other_vertex_count": int(other.mesh.n_vertices),
            "face_count": int(self.mesh.n_faces),
            "other_face_count": int(other.mesh.n_faces),
        }


@dataclass(frozen=True, slots=True)
class CoilSupportSpec:
    """Support-body meshes for primary, shield, and suppression surfaces."""

    name: str
    meshes: tuple[SupportMesh, ...] = ()
    pipeline_geometry: PipelineGeometrySpec | None = None
    coordinate_frame: str = "scanner"
    symmetry: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.meshes and self.pipeline_geometry is None:
            raise ValueError("CoilSupportSpec requires at least one mesh or a PipelineGeometrySpec.")
        object.__setattr__(self, "meshes", tuple(self.meshes))

    def by_role(self, role: MeshRole) -> tuple[SupportMesh, ...]:
        return tuple(mesh for mesh in self.meshes if mesh.role == role)

    @property
    def primary_meshes(self) -> tuple[SupportMesh, ...]:
        return self.by_role("primary")

    @property
    def shield_meshes(self) -> tuple[SupportMesh, ...]:
        return self.by_role("shield")

    @property
    def suppression_meshes(self) -> tuple[SupportMesh, ...]:
        return self.by_role("suppression")

    def active_shield_topology_reports(self) -> list[dict[str, bool | int | str]]:
        """Report whether each shield mesh shares topology with its primary mesh."""

        reports: list[dict[str, bool | int | str]] = []
        primaries = {mesh.part_index: mesh for mesh in self.primary_meshes}
        for shield in self.shield_meshes:
            if shield.paired_part_index is None:
                reports.append({"mesh": shield.name or str(shield.part_index), "paired_primary_missing": True})
                continue
            primary = primaries.get(shield.paired_part_index)
            if primary is None:
                reports.append({"mesh": shield.name or str(shield.part_index), "paired_primary_missing": True})
                continue
            reports.append(shield.topology_report(primary))
        return reports

    def validate_active_shield_topology(self) -> None:
        for report in self.active_shield_topology_reports():
            if report.get("paired_primary_missing"):
                raise ValueError(f"Shield mesh {report['mesh']} has no paired primary mesh.")
            if not report.get("same_face_connectivity", False):
                raise ValueError(
                    "Shield and primary meshes must share topology for topology-derived active shielding: "
                    f"{report['mesh']} vs {report['other_mesh']}."
                )


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    """Stream-function optimization choices."""

    method: str = "tikhonov"
    cost_function: str = "field_error_plus_resistance"
    regularization_selection: str = "l_curve"
    regularization_factor: float | None = None
    target_error_tolerance_percent: float | None = None
    min_turn_spacing_m: float = 0.0
    use_symmetry_reduction: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    """Contour extraction and routing choices."""

    contour_count: int = 10
    inter_group_connection_path: Literal[
        "direct_3d",
        "geodesic",
        "surface_geodesic_or_direct",
        "surface_geodesic",
    ] = "direct_3d"
    within_group_cut_candidate_count: int = 8
    postprocess_connections: bool = True
    normal_shift_returns: bool = True
    terminal_strategy: TerminalStrategy = "undefined"
    terminal_pad_count: int = 0
    terminal_exit_count: int = 0
    connector_clearance_m: float = 0.0
    min_bend_radius_m: float = 0.0
    keepout_region_count: int = 0
    manufacturing_intent: str = "contour_screening"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.contour_count) < 1:
            raise ValueError("contour_count must be positive.")
        allowed_terminal_strategies = {"undefined", "waived", "pads", "wire_stubs"}
        if self.terminal_strategy not in allowed_terminal_strategies:
            known = ", ".join(sorted(allowed_terminal_strategies))
            raise ValueError(f"terminal_strategy must be one of: {known}.")
        for label, value in (
            ("terminal_pad_count", self.terminal_pad_count),
            ("terminal_exit_count", self.terminal_exit_count),
            ("keepout_region_count", self.keepout_region_count),
        ):
            if int(value) < 0:
                raise ValueError(f"{label} must be non-negative.")
        for label, value in (
            ("connector_clearance_m", self.connector_clearance_m),
            ("min_bend_radius_m", self.min_bend_radius_m),
        ):
            if float(value) < 0:
                raise ValueError(f"{label} must be non-negative.")

    @property
    def terminal_handling_defined(self) -> bool:
        return self.terminal_strategy != "undefined"

    @property
    def layout_readiness_level(self) -> str:
        if not self.postprocess_connections:
            return "contour_screening"
        if not self.terminal_handling_defined:
            return "connected_without_terminals"
        if self.connector_clearance_m > 0.0 or self.min_bend_radius_m > 0.0 or self.keepout_region_count > 0:
            return "manufacturing_review"
        return "layout_ready"


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """Performance-evaluation choices and units."""

    evaluate_layout: bool = True
    evaluate_unconnected_contours: bool = True
    evaluate_inductance: bool = True
    evaluate_resistance: bool = True
    dsv_component: FieldComponent = "Bz"
    suppression_component: FieldComponent = "Bz"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CoilDesignProblem:
    """Complete high-level coil-design problem description.

    This is intentionally independent from the current procedural pipeline. It
    gives future solver, plotting, and study APIs a clear object to consume.
    """

    target_field: TargetFieldSpec
    support: CoilSupportSpec
    conductor: ConductorSpec = field(default_factory=ConductorSpec)
    optimization: OptimizationSpec = field(default_factory=OptimizationSpec)
    layout: LayoutSpec = field(default_factory=LayoutSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    metadata: DesignMetadata = field(default_factory=DesignMetadata)
    problem_schema_version: str = PROBLEM_SCHEMA_VERSION

    def validate(self) -> None:
        if self.problem_schema_version != PROBLEM_SCHEMA_VERSION:
            raise ValueError(f"Unsupported CoilDesignProblem schema version: {self.problem_schema_version!r}.")
        if self.target_field.dsv_point_count == 0:
            raise ValueError("A coil design problem requires at least one DSV target point.")
        normalize_field_component(self.target_field.component.component)
        if not self.support.primary_meshes and self.support.pipeline_geometry is None:
            raise ValueError("A coil design problem requires a primary support mesh or a supported pipeline geometry.")
        if self.support.shield_meshes:
            self.support.validate_active_shield_topology()

    def summary(self) -> dict[str, Any]:
        return {
            "problem_schema_version": self.problem_schema_version,
            "name": self.metadata.name,
            "project": self.metadata.project,
            "target_field_component": self.target_field.component.component,
            "target_region_count": len(self.target_field.regions),
            "dsv_point_count": self.target_field.dsv_point_count,
            "suppression_point_count": self.target_field.suppression_point_count,
            "diagnostic_point_count": self.target_field.diagnostic_point_count,
            "support_mesh_count": len(self.support.meshes),
            "primary_mesh_count": len(self.support.primary_meshes),
            "shield_mesh_count": len(self.support.shield_meshes),
            "pipeline_geometry": None
            if self.support.pipeline_geometry is None
            else self.support.pipeline_geometry.coil_mesh_file,
            "conductor_area_m2": self.conductor.cross_section_area_m2,
            "optimization_method": self.optimization.method,
            "layout_contour_count": self.layout.contour_count,
            "layout_readiness_level": self.layout.layout_readiness_level,
            "terminal_strategy": self.layout.terminal_strategy,
        }

    def to_config(self, **kwargs: Any):
        """Convert this problem to the current ``CoilGenConfig`` pipeline input."""

        from .conversion import coil_design_problem_to_config

        return coil_design_problem_to_config(self, **kwargs)

    def run(self, *, project_root: str | Path | None = None, **config_kwargs: Any):
        """Run this problem through the current CoilGen pipeline."""

        from .pipeline import run_coilgen

        config = self.to_config(**config_kwargs)
        return run_coilgen(config, project_root=Path(project_root) if project_root is not None else None)
