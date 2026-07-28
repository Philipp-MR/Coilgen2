from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class CoilParameterSection:
    """Base class for editable project parameter sections."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def set(self, **values: Any) -> "CoilParameterSection":
        for key, value in values.items():
            if not hasattr(self, key):
                raise ValueError(f"{type(self).__name__} has no parameter {key!r}.")
            setattr(self, key, value)
        return self


@dataclass(slots=True)
class MeshParameters(CoilParameterSection):
    """Shared support-mesh parameters for project channels."""

    mesh_size: int | None = None
    angular_divisions: int | None = None
    length_divisions: int | None = None
    radial_divisions: int | None = None
    circular_divisions: int | None = None
    lateral_divisions: int | None = None
    longitudinal_divisions: int | None = None
    width_divisions: int | None = None
    height_divisions: int | None = None
    radius_m: float | None = None
    radius_x_m: float | None = None
    radius_y_m: float | None = None
    inner_radius_m: float | None = None
    outer_radius_m: float | None = None
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    plane_distance_m: float | None = None
    center_m: tuple[float, float, float] | None = None
    rotation_axis: tuple[float, float, float] | None = None
    rotation_angle_rad: float | None = None

    def to_design_overrides(self, supported: set[str]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        mesh_size = _optional_int(self.mesh_size)
        if mesh_size is not None:
            for key in (
                "angular_divisions",
                "length_divisions",
                "radial_divisions",
                "circular_divisions",
                "lateral_divisions",
                "longitudinal_divisions",
                "width_divisions",
                "height_divisions",
            ):
                if key in supported:
                    values[key] = mesh_size
        _put_if_supported(values, supported, "angular_divisions", _optional_int(self.angular_divisions))
        _put_if_supported(values, supported, "length_divisions", _optional_int(self.length_divisions))
        _put_if_supported(values, supported, "radial_divisions", _optional_int(self.radial_divisions))
        _put_if_supported(values, supported, "circular_divisions", _optional_int(self.circular_divisions))
        _put_if_supported(values, supported, "lateral_divisions", _optional_int(self.lateral_divisions))
        _put_if_supported(values, supported, "longitudinal_divisions", _optional_int(self.longitudinal_divisions))
        _put_if_supported(values, supported, "width_divisions", _optional_int(self.width_divisions))
        _put_if_supported(values, supported, "height_divisions", _optional_int(self.height_divisions))
        for key in (
            "radius_m",
            "radius_x_m",
            "radius_y_m",
            "inner_radius_m",
            "outer_radius_m",
            "length_m",
            "width_m",
            "height_m",
            "plane_distance_m",
            "rotation_angle_rad",
        ):
            _put_if_supported(values, supported, key, _optional_float(getattr(self, key)))
        _put_if_supported(values, supported, "center_m", _optional_vector3(self.center_m))
        _put_if_supported(values, supported, "rotation_axis", _optional_vector3(self.rotation_axis))
        return values


@dataclass(slots=True)
class TargetParameters(CoilParameterSection):
    """Shared target-field parameters for project channels."""

    expression: str | None = None
    scale: float | None = None
    gradient_normalization: str | None = None
    gradient_axis: str | None = None
    region_radius_m: float | None = None
    region_resolution: int | None = None
    component: str | None = None
    field_file: str | None = None
    field_name: str | None = None
    field_units: str | None = None
    display_units: str | None = None

    def to_design_overrides(self, supported: set[str]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        _put_if_supported(values, supported, "target_bz_expression", self.expression)
        _put_if_supported(values, supported, "target_scale", _optional_float(self.scale))
        _put_if_supported(values, supported, "target_gradient_normalization", self.gradient_normalization)
        _put_if_supported(values, supported, "target_gradient_axis", self.gradient_axis)
        _put_if_supported(values, supported, "target_region_radius_m", _optional_float(self.region_radius_m))
        _put_if_supported(values, supported, "target_region_resolution", _optional_int(self.region_resolution))
        _put_if_supported(values, supported, "target_field_component", self.component)
        _put_if_supported(values, supported, "target_field_file", self.field_file)
        _put_if_supported(values, supported, "target_field_name", self.field_name)
        _put_if_supported(values, supported, "target_field_units", self.field_units)
        _put_if_supported(values, supported, "field_display_units", self.display_units)
        return values


@dataclass(slots=True)
class OptimizationParameters(CoilParameterSection):
    """Shared optimization and contouring parameters for project channels."""

    method: str | None = None
    regularization_method: str | None = None
    regularization_factor: float | None = None
    l_curve_candidate_count: int | None = None
    l_curve_smoothing_window: int | None = None
    contour_discretization_method: str | None = None
    contour_levels: int | None = None
    engineering_contour_step_safety_factor: float | None = None
    engineering_contour_max_levels: int | None = None
    min_turn_spacing_m: float | None = None
    potential_offset_factor: float | None = None
    min_contour_points: int | None = None
    contour_sparsification_mode: str | None = None
    contour_sparsification_positioning: str | None = None
    contour_sparsification_max_rms_field_error_percent: float | None = None
    contour_sparsification_max_peak_field_error_percent: float | None = None
    contour_sparsification_max_merges: int | None = None
    contour_sparsification_max_passes: int | None = None
    contour_sparsification_max_turn_multiplicity: int | None = None
    contour_sparsification_field_fit_candidate_count: int | None = None
    contour_sparsification_preserve_symmetry: bool | None = None
    skip_postprocessing: bool | None = None
    skip_sweep: bool | None = None
    skip_inductance: bool | None = None
    inductance_source: str | None = None
    target_error_tolerance_percent: float | None = None
    target_error_tolerance_absolute_T: float | None = None
    target_error_constraint_candidate_count: int | None = None
    gradient_strength_tolerance_percent: float | None = None
    linearity_region: str | None = None
    linearity_reference_radius_m: float | None = None
    linearity_rms_tolerance_percent: float | None = None
    linearity_peak_tolerance_percent: float | None = None
    linearity_cross_gradient_tolerance_percent: float | None = None
    linearity_normalization: str | None = None
    linearity_fit_offset: bool | None = None
    constrained_optimization_threads: int | None = None

    def to_design_overrides(self, supported: set[str]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        _put_if_supported(values, supported, "optimization_method", self.method)
        _put_if_supported(values, supported, "regularization_method", self.regularization_method)
        _put_if_supported(values, supported, "regularization_factor", _optional_float(self.regularization_factor))
        _put_if_supported(values, supported, "l_curve_candidate_count", _optional_int(self.l_curve_candidate_count))
        _put_if_supported(values, supported, "l_curve_smoothing_window", _optional_int(self.l_curve_smoothing_window))
        _put_if_supported(values, supported, "contour_discretization_method", self.contour_discretization_method)
        _put_if_supported(values, supported, "contour_levels", _optional_int(self.contour_levels))
        _put_if_supported(
            values,
            supported,
            "engineering_contour_step_safety_factor",
            _optional_float(self.engineering_contour_step_safety_factor),
        )
        _put_if_supported(
            values, supported, "engineering_contour_max_levels", _optional_int(self.engineering_contour_max_levels)
        )
        _put_if_supported(values, supported, "min_turn_spacing_m", _optional_float(self.min_turn_spacing_m))
        _put_if_supported(values, supported, "potential_offset_factor", _optional_float(self.potential_offset_factor))
        _put_if_supported(values, supported, "min_contour_points", _optional_int(self.min_contour_points))
        _put_if_supported(values, supported, "contour_sparsification_mode", self.contour_sparsification_mode)
        _put_if_supported(
            values, supported, "contour_sparsification_positioning", self.contour_sparsification_positioning
        )
        _put_if_supported(
            values,
            supported,
            "contour_sparsification_max_rms_field_error_percent",
            _optional_float(self.contour_sparsification_max_rms_field_error_percent),
        )
        _put_if_supported(
            values,
            supported,
            "contour_sparsification_max_peak_field_error_percent",
            _optional_float(self.contour_sparsification_max_peak_field_error_percent),
        )
        _put_if_supported(
            values,
            supported,
            "contour_sparsification_max_merges",
            _optional_int(self.contour_sparsification_max_merges),
        )
        _put_if_supported(
            values,
            supported,
            "contour_sparsification_max_passes",
            _optional_int(self.contour_sparsification_max_passes),
        )
        _put_if_supported(
            values,
            supported,
            "contour_sparsification_max_turn_multiplicity",
            _optional_int(self.contour_sparsification_max_turn_multiplicity),
        )
        _put_if_supported(
            values,
            supported,
            "contour_sparsification_field_fit_candidate_count",
            _optional_int(self.contour_sparsification_field_fit_candidate_count),
        )
        _put_if_supported(
            values,
            supported,
            "contour_sparsification_preserve_symmetry",
            self.contour_sparsification_preserve_symmetry,
        )
        _put_if_supported(values, supported, "skip_postprocessing", self.skip_postprocessing)
        _put_if_supported(values, supported, "skip_sweep", self.skip_sweep)
        _put_if_supported(values, supported, "skip_inductance", self.skip_inductance)
        _put_if_supported(values, supported, "inductance_source", self.inductance_source)
        _put_if_supported(
            values,
            supported,
            "target_error_tolerance_percent",
            _optional_float(self.target_error_tolerance_percent),
        )
        _put_if_supported(
            values,
            supported,
            "target_error_tolerance_absolute_T",
            _optional_float(self.target_error_tolerance_absolute_T),
        )
        _put_if_supported(
            values,
            supported,
            "target_error_constraint_candidate_count",
            _optional_int(self.target_error_constraint_candidate_count),
        )
        _put_if_supported(
            values,
            supported,
            "gradient_strength_tolerance_percent",
            _optional_float(self.gradient_strength_tolerance_percent),
        )
        _put_if_supported(values, supported, "linearity_region", self.linearity_region)
        _put_if_supported(
            values, supported, "linearity_reference_radius_m", _optional_float(self.linearity_reference_radius_m)
        )
        _put_if_supported(
            values,
            supported,
            "linearity_rms_tolerance_percent",
            _optional_float(self.linearity_rms_tolerance_percent),
        )
        _put_if_supported(
            values,
            supported,
            "linearity_peak_tolerance_percent",
            _optional_float(self.linearity_peak_tolerance_percent),
        )
        _put_if_supported(
            values,
            supported,
            "linearity_cross_gradient_tolerance_percent",
            _optional_float(self.linearity_cross_gradient_tolerance_percent),
        )
        _put_if_supported(values, supported, "linearity_normalization", self.linearity_normalization)
        _put_if_supported(values, supported, "linearity_fit_offset", self.linearity_fit_offset)
        _put_if_supported(
            values,
            supported,
            "constrained_optimization_threads",
            _optional_int(self.constrained_optimization_threads),
        )
        return values


@dataclass(slots=True)
class ConductorParameters(CoilParameterSection):
    """Shared conductor parameters for project channels."""

    width_m: float | None = None
    height_m: float | None = None
    resistivity_ohm_m: float | None = None
    max_current_A: float | None = None
    max_wire_current_A: float | None = None
    min_wire_distance_m: float | None = None
    surface_current_density_limit_A_per_m: float | None = None
    engineering_contour_step_safety_factor: float | None = None
    gradient_hardware_infeasible_policy: str | None = None

    def to_design_overrides(self, supported: set[str]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        _put_if_supported(values, supported, "conductor_width_m", _optional_float(self.width_m))
        _put_if_supported(values, supported, "conductor_height_m", _optional_float(self.height_m))
        _put_if_supported(values, supported, "conductor_resistivity_ohm_m", _optional_float(self.resistivity_ohm_m))
        _put_if_supported(values, supported, "max_current_A", _optional_float(self.max_current_A))
        _put_if_supported(values, supported, "max_wire_current_A", _optional_float(self.max_wire_current_A))
        _put_if_supported(values, supported, "min_wire_distance_m", _optional_float(self.min_wire_distance_m))
        _put_if_supported(
            values,
            supported,
            "surface_current_density_limit_A_per_m",
            _optional_float(self.surface_current_density_limit_A_per_m),
        )
        _put_if_supported(
            values,
            supported,
            "engineering_contour_step_safety_factor",
            _optional_float(self.engineering_contour_step_safety_factor),
        )
        _put_if_supported(
            values, supported, "gradient_hardware_infeasible_policy", self.gradient_hardware_infeasible_policy
        )
        return values


@dataclass(slots=True)
class SystemParameters(CoilParameterSection):
    """Shared scanner/system parameters for project channels."""

    main_field_strength_T: float | None = None
    b0_direction: tuple[float, float, float] | None = None

    def to_design_overrides(self, supported: set[str]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        _put_if_supported(values, supported, "main_field_strength_T", _optional_float(self.main_field_strength_T))
        _put_if_supported(values, supported, "b0_direction", _optional_vector3(self.b0_direction))
        return values


@dataclass(slots=True)
class NumericsParameters(CoilParameterSection):
    """Shared numerical execution parameters for project channels."""

    cpu_workers: int | None = None
    progress: bool | None = None

    def to_design_overrides(self, supported: set[str]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        _put_if_supported(values, supported, "cpu_workers", _optional_int(self.cpu_workers))
        _put_if_supported(values, supported, "progress", self.progress)
        return values


@dataclass(slots=True)
class CoilProjectParameters(CoilParameterSection):
    """Structured shared defaults for a CoilProject.

    Scripts can edit these attributes directly:

    ``project.parameters.mesh.mesh_size = 24``
    ``project.parameters.target.scale = 0.010``

    Values are applied only to design objects that support the mapped field.
    Channel-specific overrides are still applied afterward and therefore win.
    """

    mesh: MeshParameters = field(default_factory=MeshParameters)
    target: TargetParameters = field(default_factory=TargetParameters)
    optimization: OptimizationParameters = field(default_factory=OptimizationParameters)
    conductor: ConductorParameters = field(default_factory=ConductorParameters)
    system: SystemParameters = field(default_factory=SystemParameters)
    numerics: NumericsParameters = field(default_factory=NumericsParameters)

    def to_design_overrides(self, design: Any) -> dict[str, Any]:
        supported = _supported_fields(design)
        values: dict[str, Any] = {}
        for section in (self.mesh, self.target, self.optimization, self.conductor, self.system, self.numerics):
            values.update(section.to_design_overrides(supported))
        return values

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "CoilProjectParameters":
        parameters = cls()
        if not values:
            return parameters
        for section_name, section_values in values.items():
            if not hasattr(parameters, section_name):
                raise ValueError(f"CoilProjectParameters has no section {section_name!r}.")
            section = getattr(parameters, section_name)
            if isinstance(section_values, Mapping):
                section.set(**dict(section_values))
            else:
                raise TypeError(f"CoilProjectParameters.{section_name} must be a mapping.")
        return parameters


def _supported_fields(design: Any) -> set[str]:
    if is_dataclass(design) and not isinstance(design, type):
        return {item.name for item in fields(design)}
    return {name for name in dir(design) if not name.startswith("_")}


def _put_if_supported(values: dict[str, Any], supported: set[str], key: str, value: Any) -> None:
    if value is not None and key in supported:
        values[key] = value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_vector3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    items = tuple(float(item) for item in value)
    if len(items) != 3:
        raise ValueError("Expected a 3-component vector.")
    return items
