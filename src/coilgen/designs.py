from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import CoilGenConfig
from .config_validation import ConfigValidationReport, validate_config
from .design_context import design_context_key
from .gradient_linearity import GradientLinearitySpec, gradient_linearity_spec_from_value
from .gradient_slice_maps import GradientSliceMapSpec, gradient_slice_map_spec_from_value
from .geometry import (
    build_parametric_surface_mesh,
    combine_meshes,
    load_mesh_from_config,
    offset_mesh_along_vertex_normals,
)
from .models import CoilGenResult
from .object_registry import register_in_memory_mesh
from .pipeline import run_coilgen


@dataclass(slots=True)
class CoilDesign:
    """User-facing base class for a complete CoilGen design request.

    Subclasses only need to describe the coil surface geometry. All values use
    SI units. Extra expert parameters may be passed through
    ``advanced_parameters`` without changing this API.
    """

    name: str = "coil_design"
    target_bz_expression: str = "y"
    target_scale: float = 1.0
    target_slew_rate_T_per_m_per_s: float = 0.0
    target_gradient_normalization: str = "auto"
    target_gradient_axis: str = "auto"
    target_region_radius_m: float = 0.08
    target_region_resolution: int = 5
    contour_discretization_method: str = "level_count"
    contour_levels: int = 8
    min_turn_spacing_m: float = 0.0
    max_current_A: float = 0.0
    max_wire_current_A: float = 0.0
    min_wire_distance_m: float = 0.0
    surface_current_density_limit_A_per_m: float = 0.0
    engineering_contour_step_safety_factor: float = 1.0
    engineering_contour_max_levels: int = 0
    gradient_hardware_infeasible_policy: str = "scale_to_limit"
    potential_offset_factor: float = 0.25
    min_contour_points: int = 4
    conductor_width_m: float = 0.001
    conductor_height_m: float = 0.001
    conductor_resistivity_ohm_m: float = 0.018e-6
    main_field_strength_T: float = 1.0
    b0_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    optimization_method: str = "tikhonov"
    regularization_method: str = "l_curve"
    regularization_factor: float = 50.0
    l_curve_candidate_count: int = 21
    l_curve_smoothing_window: int = 5
    target_error_constraint_mode: str = "separate"
    target_error_constraint_metric: str = "norm"
    target_error_tolerance_percent: float = 5.0
    target_error_tolerance_absolute_T: float = 0.0
    target_error_std_limit_T: float = 0.0
    target_error_rms_limit_T: float = 0.0
    suppression_error_tolerance_percent: float = 5.0
    suppression_error_tolerance_absolute_T: float = 0.0
    suppression_field_rms_limit_T: float = 0.0
    suppression_field_component: str = "target"
    suppression_field_peak_limit_T: float = 0.0
    suppression_surface_weighting: str = "uniform"
    suppression_normal_source: str = "mesh"
    target_error_constraint_candidate_count: int = 61
    target_error_auto_retry_enabled: bool = True
    target_error_auto_retry_max_reruns: int = 10
    target_error_auto_retry_factor_shrink: float = 0.1
    target_error_auto_retry_candidate_growth: float = 1.2
    target_error_auto_retry_max_candidate_count: int = 201
    constrained_quadratic_fallback_policy: str = "target_error_scan"
    constrained_optimization_threads: int = 0
    gradient_strength_tolerance_percent: float = 0.0
    linearity_region: str = "dsv"
    linearity_reference_radius_m: float = 0.0
    linearity_rms_tolerance_percent: float = 0.0
    linearity_peak_tolerance_percent: float = 0.0
    linearity_cross_gradient_tolerance_percent: float = 0.0
    linearity_normalization: str = "requested_gradient_times_radius"
    linearity_fit_offset: bool = True
    slice_maps: GradientSliceMapSpec | dict[str, Any] | None = None
    target_field_file: str = "none"
    target_field_name: str = "none"
    target_field_units: str = "T"
    field_display_units: str = "T"
    field_evaluation_current_source: str = "target_fit"
    preoptimized_stream_function_file: str = "none"
    preoptimized_stream_function_field_name: str = "stream_function"
    cpu_workers: int = 0
    progress: bool = True
    skip_postprocessing: bool = False
    skip_sweep: bool = False
    skip_inductance: bool = False
    inductance_source: str = "auto"
    save_stl: bool = False
    save_wire_trace_csv: bool = False
    show_internal_plots: bool = False
    advanced_parameters: dict[str, Any] = field(default_factory=dict)
    design_context: str = "mr_gradient"
    linearity: GradientLinearitySpec | dict[str, Any] | None = None

    def geometry_config(self) -> dict[str, Any]:
        raise NotImplementedError("Subclasses must provide coil surface geometry parameters.")

    def to_config(self) -> CoilGenConfig:
        values: dict[str, Any] = {
            "design_context": design_context_key(self.design_context),
            "field_shape_function": self.target_bz_expression,
            "target_gradient_strength": self.target_scale,
            "target_gradient_normalization": self.target_gradient_normalization,
            "target_gradient_axis": self.target_gradient_axis,
            "target_region_radius": self.target_region_radius_m,
            "target_region_resolution": self.target_region_resolution,
            "contour_discretization_method": self.contour_discretization_method,
            "levels": self.contour_levels,
            "max_current_A": self.max_current_A,
            "max_wire_current_A": self.max_wire_current_A if self.max_wire_current_A > 0.0 else self.max_current_A,
            "min_wire_distance_m": self.min_wire_distance_m,
            "min_turn_distance": self.min_wire_distance_m
            if self.min_wire_distance_m > 0.0
            else self.min_turn_spacing_m,
            "surface_current_density_limit_A_per_m": self.surface_current_density_limit_A_per_m,
            "engineering_contour_step_safety_factor": self.engineering_contour_step_safety_factor,
            "engineering_contour_max_levels": self.engineering_contour_max_levels,
            "gradient_hardware_infeasible_policy": self.gradient_hardware_infeasible_policy,
            "pot_offset_factor": self.potential_offset_factor,
            "min_point_loop_number": self.min_contour_points,
            "conductor_cross_section_width": self.conductor_width_m,
            "conductor_cross_section_height": self.conductor_height_m,
            "specific_conductivity_conductor": self.conductor_resistivity_ohm_m,
            "b_0_strength_T": self.main_field_strength_T,
            "b_0_direction": self.b0_direction,
            "regularization_selection_method": self.regularization_method,
            "tikonov_reg_factor": self.regularization_factor,
            "l_curve_factor_count": self.l_curve_candidate_count,
            "l_curve_curvature_smoothing_window": self.l_curve_smoothing_window,
            "target_field_definition_file": self.target_field_file,
            "target_field_definition_field_name": self.target_field_name,
            "target_field_units": self.target_field_units,
            "field_display_units": self.field_display_units,
            "field_evaluation_current_source": self.field_evaluation_current_source,
            "preoptimized_stream_function_file": self.preoptimized_stream_function_file,
            "preoptimized_stream_function_field_name": self.preoptimized_stream_function_field_name,
            "parallel_worker_count": self.cpu_workers,
            "progress_flag": self.progress,
            "skip_postprocessing": self.skip_postprocessing,
            "skip_sweep": self.skip_sweep,
            "skip_inductance_calculation": self.skip_inductance,
            "inductance_source": self.inductance_source,
            "save_stl_flag": self.save_stl,
            "save_wire_trace_csv": self.save_wire_trace_csv,
            "plot_flag": self.show_internal_plots,
            "gauss_order": 1,
            "sf_opt_method": self.optimization_method,
            "stream_function_cost_function": "field_error_plus_resistance",
            "l_curve_scan_method": "normal_equations",
            "target_error_constraint_mode": self.target_error_constraint_mode,
            "target_error_constraint_metric": self.target_error_constraint_metric,
            "target_error_tolerance_percent": self.target_error_tolerance_percent,
            "target_error_tolerance_absolute_T": self.target_error_tolerance_absolute_T,
            "target_error_std_limit_T": self.target_error_std_limit_T,
            "target_error_rms_limit_T": self.target_error_rms_limit_T,
            "suppression_error_tolerance_percent": self.suppression_error_tolerance_percent,
            "suppression_error_tolerance_absolute_T": self.suppression_error_tolerance_absolute_T,
            "suppression_field_rms_limit_T": self.suppression_field_rms_limit_T,
            "suppression_field_component": self.suppression_field_component,
            "suppression_field_peak_limit_T": self.suppression_field_peak_limit_T,
            "suppression_surface_weighting": self.suppression_surface_weighting,
            "suppression_normal_source": self.suppression_normal_source,
            "target_error_constraint_factor_count": self.target_error_constraint_candidate_count,
            "target_error_auto_retry_enabled": self.target_error_auto_retry_enabled,
            "target_error_auto_retry_max_reruns": self.target_error_auto_retry_max_reruns,
            "target_error_auto_retry_factor_shrink": self.target_error_auto_retry_factor_shrink,
            "target_error_auto_retry_candidate_growth": self.target_error_auto_retry_candidate_growth,
            "target_error_auto_retry_max_candidate_count": self.target_error_auto_retry_max_candidate_count,
            "constrained_quadratic_fallback_policy": self.constrained_quadratic_fallback_policy,
            "constrained_optimization_threads": self.constrained_optimization_threads,
            "gradient_strength_tolerance_percent": self.gradient_strength_tolerance_percent,
            "linearity_region": self.linearity_region,
            "linearity_reference_radius_m": self.linearity_reference_radius_m,
            "linearity_rms_tolerance_percent": self.linearity_rms_tolerance_percent,
            "linearity_peak_tolerance_percent": self.linearity_peak_tolerance_percent,
            "linearity_cross_gradient_tolerance_percent": self.linearity_cross_gradient_tolerance_percent,
            "linearity_normalization": self.linearity_normalization,
            "linearity_fit_offset": self.linearity_fit_offset,
            "inter_group_connection_path_method": "direct_3d",
        }
        if self.linearity is not None:
            values.update(gradient_linearity_spec_from_value(self.linearity).to_config_overrides())
            if self.gradient_strength_tolerance_percent > 0.0:
                values["gradient_strength_tolerance_percent"] = self.gradient_strength_tolerance_percent
        if self.slice_maps is not None:
            values.update(gradient_slice_map_spec_from_value(self.slice_maps).to_config_overrides())
        values.update(self.geometry_config())
        if str(self.contour_discretization_method).strip().lower().replace("-", "_") != "level_count" and (
            self.min_wire_distance_m > 0.0 or self.min_turn_spacing_m > 0.0
        ):
            values.setdefault("turn_spacing_regularization_factor", 1.0)
            values.setdefault("skip_calculation_min_winding_distance", False)
        values.update(self.advanced_parameters)
        if self.target_field_file != "none":
            values.setdefault("field_shape_function", "none")
            values["field_shape_function"] = "none"
        return CoilGenConfig(**values)

    def validate(self, *, project_root: str | Path | None = None) -> ConfigValidationReport:
        return validate_config(self.to_config(), project_root=project_root)

    def run(
        self,
        *,
        project_root: str | Path | None = None,
        print_validation_warnings: bool = True,
    ) -> CoilGenResult:
        config = self.to_config()
        report = validate_config(config, project_root=project_root)
        report.raise_for_errors()
        if print_validation_warnings and report.warnings:
            print("CoilGen configuration warnings:")
            for warning in report.warnings:
                print(f"  {warning.format()}")
        return run_coilgen(config, project_root=Path(project_root) if project_root is not None else None)

    def as_user_parameters(self) -> dict[str, Any]:
        """Return the visible user-facing settings as a plain dictionary."""

        return asdict(self)


@dataclass(slots=True)
class CoilGenConfigDesign:
    """Project/channel adapter for an explicit ``CoilGenConfig``.

    This is useful for expert scripts that already build a full config but
    still want project-level outputs, channel summaries, and plot books.
    """

    name: str = "coilgen_config_design"
    config: CoilGenConfig = field(default_factory=CoilGenConfig)
    target_bz_expression: str | None = None
    target_scale: float | None = None
    design_context: str | None = None

    def to_config(self) -> CoilGenConfig:
        config = deepcopy(self.config)
        if self.design_context is not None:
            config.design_context = design_context_key(self.design_context)
        if self.target_bz_expression is not None:
            config.field_shape_function = str(self.target_bz_expression)
        if self.target_scale is not None:
            config.target_gradient_strength = float(self.target_scale)
        return config

    def validate(self, *, project_root: str | Path | None = None) -> ConfigValidationReport:
        return validate_config(self.to_config(), project_root=project_root)

    def run(
        self,
        *,
        project_root: str | Path | None = None,
        print_validation_warnings: bool = True,
    ) -> CoilGenResult:
        config = self.to_config()
        report = validate_config(config, project_root=project_root)
        report.raise_for_errors()
        if print_validation_warnings and report.warnings:
            print("CoilGen configuration warnings:")
            for warning in report.warnings:
                print(f"  {warning.format()}")
        return run_coilgen(config, project_root=Path(project_root) if project_root is not None else None)

    def as_user_parameters(self) -> dict[str, Any]:
        values = self.to_config().as_dict()
        values["name"] = self.name
        if self.design_context is not None:
            values["design_context"] = design_context_key(self.design_context)
        if self.target_bz_expression is not None:
            values["target_bz_expression"] = self.target_bz_expression
        if self.target_scale is not None:
            values["target_scale"] = self.target_scale
        return values


@dataclass(slots=True)
class CylinderCoilDesign(CoilDesign):
    radius_m: float = 0.25
    length_m: float = 0.5
    angular_divisions: int = 40
    length_divisions: int = 40
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_angle_rad: float = 0.0

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create cylinder mesh",
            "surface_is_cylinder_flag": True,
            "cylinder_mesh_parameter_list": (
                self.length_m,
                self.radius_m,
                self.angular_divisions,
                self.length_divisions,
                *self.rotation_axis,
                self.rotation_angle_rad,
            ),
        }


@dataclass(slots=True)
class PlanarCoilDesign(CoilDesign):
    width_m: float = 0.25
    height_m: float = 0.25
    lateral_divisions: int = 40
    longitudinal_divisions: int = 40
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_angle_rad: float = 0.0
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create planary mesh",
            "surface_is_cylinder_flag": False,
            "planar_mesh_parameter_list": (
                self.height_m,
                self.width_m,
                self.lateral_divisions,
                self.longitudinal_divisions,
                *self.rotation_axis,
                self.rotation_angle_rad,
                *self.center_m,
            ),
        }


@dataclass(slots=True)
class CircularDiscCoilDesign(CoilDesign):
    """Single circular planar support surface."""

    radius_m: float = 0.25
    radial_divisions: int = 40
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_angle_rad: float = 0.0
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create circular mesh",
            "surface_is_cylinder_flag": False,
            "circular_mesh_parameter_list": (
                self.radius_m,
                self.radial_divisions,
                *self.rotation_axis,
                self.rotation_angle_rad,
                *self.center_m,
            ),
        }


@dataclass(slots=True)
class AnnularDiscCoilDesign(CoilDesign):
    """Single annular planar support surface with an inner opening."""

    inner_radius_m: float = 0.08
    outer_radius_m: float = 0.25
    radial_divisions: int = 6
    circular_divisions: int = 48
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create annular disc mesh",
            "surface_is_cylinder_flag": False,
            "annular_disc_mesh_parameter_list": (
                self.inner_radius_m,
                self.outer_radius_m,
                self.radial_divisions,
                self.circular_divisions,
                *self.normal,
                *self.center_m,
            ),
        }


@dataclass(slots=True)
class PartialCylinderCoilDesign(CoilDesign):
    """Open cylindrical sector support surface."""

    radius_m: float = 0.25
    length_m: float = 0.5
    start_angle_rad: float = -1.5707963267948966
    end_angle_rad: float = 1.5707963267948966
    angular_divisions: int = 36
    length_divisions: int = 24
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_angle_rad: float = 0.0
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create partial cylinder mesh",
            "surface_is_cylinder_flag": True,
            "partial_cylinder_mesh_parameter_list": (
                self.length_m,
                self.radius_m,
                self.start_angle_rad,
                self.end_angle_rad,
                self.angular_divisions,
                self.length_divisions,
                *self.rotation_axis,
                self.rotation_angle_rad,
                *self.center_m,
            ),
        }


@dataclass(slots=True)
class EllipticalCylinderCoilDesign(CoilDesign):
    """Open elliptical cylindrical support surface."""

    radius_x_m: float = 0.28
    radius_y_m: float = 0.18
    length_m: float = 0.5
    angular_divisions: int = 48
    length_divisions: int = 24
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_angle_rad: float = 0.0
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create elliptical cylinder mesh",
            "surface_is_cylinder_flag": True,
            "elliptical_cylinder_mesh_parameter_list": (
                self.length_m,
                self.radius_x_m,
                self.radius_y_m,
                self.angular_divisions,
                self.length_divisions,
                *self.rotation_axis,
                self.rotation_angle_rad,
                *self.center_m,
            ),
        }


@dataclass(slots=True)
class SurfaceOfRevolutionCoilDesign(CoilDesign):
    """Open axisymmetric support surface from z/radius samples."""

    circular_divisions: int = 48
    z_radius_pairs_m: tuple[tuple[float, float], ...] = ((-0.25, 0.18), (0.0, 0.25), (0.25, 0.18))

    def geometry_config(self) -> dict[str, Any]:
        flattened: list[float] = [float(self.circular_divisions)]
        for z_value, radius in self.z_radius_pairs_m:
            flattened.extend((float(z_value), float(radius)))
        return {
            "coil_mesh_file": "create surface of revolution mesh",
            "surface_is_cylinder_flag": True,
            "surface_of_revolution_mesh_parameter_list": tuple(flattened),
        }


@dataclass(slots=True)
class SuperellipseCylinderCoilDesign(CoilDesign):
    """Rounded-rectangle/superellipse cylindrical support surface."""

    radius_x_m: float = 0.28
    radius_y_m: float = 0.18
    length_m: float = 0.5
    exponent: float = 4.0
    angular_divisions: int = 48
    length_divisions: int = 24
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_angle_rad: float = 0.0
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create superellipse cylinder mesh",
            "surface_is_cylinder_flag": True,
            "superellipse_cylinder_mesh_parameter_list": (
                self.length_m,
                self.radius_x_m,
                self.radius_y_m,
                self.exponent,
                self.angular_divisions,
                self.length_divisions,
                *self.rotation_axis,
                self.rotation_angle_rad,
                *self.center_m,
            ),
        }


@dataclass(slots=True)
class SphericalCapCoilDesign(CoilDesign):
    """Open spherical cap support surface."""

    radius_m: float = 0.25
    cap_polar_angle_rad: float = 1.5707963267948966
    azimuthal_divisions: int = 48
    polar_divisions: int = 16
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_angle_rad: float = 0.0
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create spherical cap mesh",
            "surface_is_cylinder_flag": False,
            "spherical_cap_mesh_parameter_list": (
                self.radius_m,
                self.cap_polar_angle_rad,
                self.azimuthal_divisions,
                self.polar_divisions,
                *self.rotation_axis,
                self.rotation_angle_rad,
                *self.center_m,
            ),
        }


@dataclass(slots=True)
class EllipsoidalCapCoilDesign(CoilDesign):
    """Open ellipsoidal cap support surface."""

    radius_x_m: float = 0.25
    radius_y_m: float = 0.20
    radius_z_m: float = 0.18
    cap_polar_angle_rad: float = 1.5707963267948966
    azimuthal_divisions: int = 48
    polar_divisions: int = 16
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_angle_rad: float = 0.0
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create ellipsoidal cap mesh",
            "surface_is_cylinder_flag": False,
            "ellipsoidal_cap_mesh_parameter_list": (
                self.radius_x_m,
                self.radius_y_m,
                self.radius_z_m,
                self.cap_polar_angle_rad,
                self.azimuthal_divisions,
                self.polar_divisions,
                *self.rotation_axis,
                self.rotation_angle_rad,
                *self.center_m,
            ),
        }


@dataclass(slots=True)
class ConicalFrustumCoilDesign(CoilDesign):
    """Open tapered cylindrical support surface."""

    radius_bottom_m: float = 0.22
    radius_top_m: float = 0.16
    length_m: float = 0.5
    angular_divisions: int = 48
    length_divisions: int = 24
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_angle_rad: float = 0.0
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create conical frustum mesh",
            "surface_is_cylinder_flag": True,
            "conical_frustum_mesh_parameter_list": (
                self.length_m,
                self.radius_bottom_m,
                self.radius_top_m,
                self.angular_divisions,
                self.length_divisions,
                *self.rotation_axis,
                self.rotation_angle_rad,
                *self.center_m,
            ),
        }


@dataclass(slots=True)
class ToroidalSegmentCoilDesign(CoilDesign):
    """Open toroidal segment support surface."""

    major_radius_m: float = 0.35
    minor_radius_m: float = 0.08
    start_angle_rad: float = -1.5707963267948966
    end_angle_rad: float = 1.5707963267948966
    tube_start_angle_rad: float = 0.0
    tube_end_angle_rad: float = 6.283185307179586
    major_divisions: int = 36
    minor_divisions: int = 16
    rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    rotation_angle_rad: float = 0.0
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create toroidal segment mesh",
            "surface_is_cylinder_flag": False,
            "toroidal_segment_mesh_parameter_list": (
                self.major_radius_m,
                self.minor_radius_m,
                self.start_angle_rad,
                self.end_angle_rad,
                self.tube_start_angle_rad,
                self.tube_end_angle_rad,
                self.major_divisions,
                self.minor_divisions,
                *self.rotation_axis,
                self.rotation_angle_rad,
                *self.center_m,
            ),
        }


@dataclass(slots=True)
class RuledSurfaceCoilDesign(CoilDesign):
    """Support surface linearly interpolated between two sampled 3D curves."""

    curve_a_m: tuple[tuple[float, float, float], ...] = ((-0.2, -0.1, 0.0), (0.2, -0.1, 0.0))
    curve_b_m: tuple[tuple[float, float, float], ...] = ((-0.2, 0.1, 0.0), (0.2, 0.1, 0.0))
    span_divisions: int = 8
    surface_is_cylinder: bool = False

    def geometry_config(self) -> dict[str, Any]:
        curve_a = np.asarray(self.curve_a_m, dtype=float)
        curve_b = np.asarray(self.curve_b_m, dtype=float)
        if curve_a.shape != curve_b.shape or curve_a.ndim != 2 or curve_a.shape[1] != 3:
            raise ValueError("RuledSurfaceCoilDesign curves must both be (n, 3) point arrays.")
        flattened: list[float] = [float(len(curve_a)), float(self.span_divisions)]
        flattened.extend(float(value) for value in curve_a.reshape(-1))
        flattened.extend(float(value) for value in curve_b.reshape(-1))
        return {
            "coil_mesh_file": "create ruled surface mesh",
            "surface_is_cylinder_flag": self.surface_is_cylinder,
            "ruled_surface_mesh_parameter_list": tuple(flattened),
        }


@dataclass(slots=True)
class ParametricSurfaceCoilDesign(CoilDesign):
    """Support surface sampled from a Python callable ``surface(u, v)``.

    The callable must return finite 3D SI coordinates. This route uses an
    in-memory mesh because arbitrary Python callables cannot be represented in a
    static `CoilGenConfig` parameter list.
    """

    surface: Callable[[float, float], tuple[float, float, float] | np.ndarray] | None = None
    u_range: tuple[float, float] = (0.0, 1.0)
    v_range: tuple[float, float] = (0.0, 1.0)
    u_divisions: int = 20
    v_divisions: int = 20
    wrap_v: bool = False
    surface_is_cylinder: bool = False

    def geometry_config(self) -> dict[str, Any]:
        if self.surface is None:
            raise ValueError("ParametricSurfaceCoilDesign.surface must be a callable surface(u, v).")
        if self.u_divisions < 1 or self.v_divisions < 1:
            raise ValueError("Parametric surface divisions must be positive.")
        u_values = np.linspace(float(self.u_range[0]), float(self.u_range[1]), int(self.u_divisions) + 1)
        v_count = int(self.v_divisions) if self.wrap_v else int(self.v_divisions) + 1
        v_values = np.linspace(float(self.v_range[0]), float(self.v_range[1]), v_count, endpoint=not self.wrap_v)
        mesh = build_parametric_surface_mesh(
            self.surface,
            u_values,
            v_values,
            wrap_v=self.wrap_v,
            geometry_type="parametric_surface",
            parameters={
                "u_range": self.u_range,
                "v_range": self.v_range,
                "u_divisions": self.u_divisions,
                "v_divisions": self.v_divisions,
                "wrap_v": self.wrap_v,
            },
        )
        key = register_in_memory_mesh(mesh, key=f"{self.name}:parametric_surface")
        return {
            "coil_mesh_file": key,
            "surface_is_cylinder_flag": self.surface_is_cylinder,
        }


@dataclass(slots=True)
class ShieldedCoilDesign(CoilDesign):
    """Generic active-shield wrapper for any generated/in-memory primary design.

    The shield is an outward normal offset of the primary support mesh and uses
    identical face topology. This keeps primary and shield stream-function
    channels structurally comparable and avoids independently meshed shield
    layers with mismatched contour counts.
    """

    primary_design: CoilDesign | None = None
    shield_separation_m: float = 0.05
    mesh_project_root: str | Path | None = None

    def geometry_config(self) -> dict[str, Any]:
        if self.primary_design is None:
            raise ValueError("ShieldedCoilDesign.primary_design must be set.")
        if float(self.shield_separation_m) <= 0.0:
            raise ValueError("ShieldedCoilDesign.shield_separation_m must be positive.")

        primary_config = self.primary_design.to_config()
        project_root = Path(self.mesh_project_root) if self.mesh_project_root is not None else Path(".")
        primary_mesh = load_mesh_from_config(primary_config, project_root=project_root)
        primary_mesh.metadata = {
            **dict(primary_mesh.metadata),
            "role": "primary",
            "shielded_wrapper_primary": True,
        }
        shield_mesh = offset_mesh_along_vertex_normals(primary_mesh, self.shield_separation_m, role="shield")
        combined = combine_meshes([primary_mesh, shield_mesh])
        combined.metadata.update(
            {
                "geometry_type": "generic_shielded_offset_pair",
                "units": "SI",
                "primary_geometry": dict(primary_mesh.metadata),
                "shield_separation_m": float(self.shield_separation_m),
                "same_topology_primary_shield": True,
                "source_design_class": type(self.primary_design).__name__,
            }
        )
        key = register_in_memory_mesh(combined, key=f"{self.name}:generic_shielded_offset_pair")
        return {
            "coil_mesh_file": key,
            "surface_is_cylinder_flag": bool(primary_config.surface_is_cylinder_flag),
        }


@dataclass(slots=True)
class BiplanarCoilDesign(CoilDesign):
    width_m: float = 0.25
    height_m: float = 0.25
    lateral_divisions: int = 40
    longitudinal_divisions: int = 40
    plane_distance_m: float = 0.2
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create bi-planary mesh",
            "surface_is_cylinder_flag": False,
            "biplanar_mesh_parameter_list": (
                self.height_m,
                self.width_m,
                self.lateral_divisions,
                self.longitudinal_divisions,
                *self.normal,
                *self.center_m,
                self.plane_distance_m,
            ),
        }


@dataclass(slots=True)
class CircularBiplanarCoilDesign(CoilDesign):
    outer_radius_m: float = 0.7
    edge_length_m: float = 0.08
    plane_distance_m: float = 1.0
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create circular bi-planary mesh",
            "surface_is_cylinder_flag": False,
            "circular_biplanar_mesh_parameter_list": (
                self.edge_length_m,
                self.outer_radius_m,
                *self.normal,
                *self.center_m,
                self.plane_distance_m,
            ),
        }


@dataclass(slots=True)
class ShieldedCircularBiplanarCoilDesign(CoilDesign):
    """Open planar gradient geometry with primary and shield disc pairs.

    The generated shield discs are offset copies of the primary discs with the
    same topology. ``shield_gap_m`` is the primary-to-shield normal separation
    distance. When a mesh-topology suppression surface is enabled through
    ``advanced_parameters``, its separation is measured outward from the shield
    surface, so primary-to-suppression distance is
    ``shield_gap_m + mesh_topology_suppression_separation``. The shield radius
    and edge length are retained for compatibility with older scripts but do
    not create an independently meshed shield layer.
    """

    primary_outer_radius_m: float = 0.7
    primary_edge_length_m: float = 0.08
    shield_outer_radius_m: float = 0.8
    shield_edge_length_m: float = 0.08
    plane_distance_m: float = 1.0
    shield_gap_m: float = 0.08
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create shielded bi-planary mesh",
            "surface_is_cylinder_flag": False,
            "shielded_circular_biplanar_mesh_parameter_list": (
                self.primary_outer_radius_m,
                self.primary_edge_length_m,
                self.shield_outer_radius_m,
                self.shield_edge_length_m,
                *self.normal,
                *self.center_m,
                self.plane_distance_m,
                self.shield_gap_m,
            ),
        }


@dataclass(slots=True)
class FlangedSplitBoreCoilDesign(CoilDesign):
    """Split-bore gradient geometry with cylindrical sections and flanges."""

    edge_length_m: float = 0.08
    inner_radius_m: float = 0.55
    outer_radius_m: float = 0.85
    gap_m: float = 0.5
    half_length_m: float = 0.6
    b0_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    mesh_rotation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    mesh_rotation_angle_rad: float = 0.0

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create flanged split bore mesh",
            "surface_is_cylinder_flag": True,
            "flanged_split_bore_mesh_parameter_list": (
                self.edge_length_m,
                self.inner_radius_m,
                self.outer_radius_m,
                self.gap_m,
                self.half_length_m,
            ),
            "b_0_direction": self.b0_direction,
            "generated_mesh_rotation_axis": self.mesh_rotation_axis,
            "generated_mesh_rotation_angle": self.mesh_rotation_angle_rad,
        }


@dataclass(slots=True)
class ShieldedFlangedSplitBoreCoilDesign(FlangedSplitBoreCoilDesign):
    """Split-bore geometry with an active shield offset from the primary layer.

    The generated shield is an outward offset copy of the primary split-bore
    topology. ``shield_inner_radius_m - outer_radius_m`` defines the separation
    distance. The shield radial thickness should normally match the primary
    thickness ``outer_radius_m - inner_radius_m``. If a mesh-topology
    suppression surface is enabled, its separation is measured outward from the
    shield outer surface, so primary outer radius to suppression inner radius is
    ``shield_separation + primary_thickness + suppression_separation``. Other
    shield shape parameters are retained for compatibility and
    suppression-surface bookkeeping, but they do not create a separate topology.
    For flanged or otherwise non-smooth supports, enable the topology
    suppression clearance filter when neighboring faces near corners must be
    excluded from the zero-field target surface.
    """

    shield_edge_length_m: float = 0.08
    shield_inner_radius_m: float = 0.9
    shield_outer_radius_m: float = 1.05
    shield_gap_m: float = 0.5
    shield_half_length_m: float = 0.6

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": "create shielded flanged split bore mesh",
            "surface_is_cylinder_flag": True,
            "shielded_flanged_split_bore_mesh_parameter_list": (
                self.edge_length_m,
                self.inner_radius_m,
                self.outer_radius_m,
                self.gap_m,
                self.half_length_m,
                self.shield_edge_length_m,
                self.shield_inner_radius_m,
                self.shield_outer_radius_m,
                self.shield_gap_m,
                self.shield_half_length_m,
            ),
            "b_0_direction": self.b0_direction,
            "generated_mesh_rotation_axis": self.mesh_rotation_axis,
            "generated_mesh_rotation_angle": self.mesh_rotation_angle_rad,
        }


@dataclass(slots=True)
class ImportedMeshCoilDesign(CoilDesign):
    mesh_file: str = "coil_surface.stl"
    geometry_source_path: str | Path | None = None
    surface_is_cylinder: bool = False

    def geometry_config(self) -> dict[str, Any]:
        return {
            "coil_mesh_file": self.mesh_file,
            "geometry_source_path": None if self.geometry_source_path is None else Path(self.geometry_source_path),
            "surface_is_cylinder_flag": self.surface_is_cylinder,
        }


def coil_geometry_api_template(geometry_name: str, *, class_name: str | None = None) -> str:
    """Return a Python template for adding a new user-facing geometry API."""

    clean_name = "".join(part.capitalize() for part in geometry_name.replace("-", " ").replace("_", " ").split())
    class_name = class_name or f"{clean_name}CoilDesign"
    mesh_key = geometry_name.strip().lower().replace("_", " ").replace("-", " ")
    if not mesh_key.startswith("create "):
        mesh_key = f"create {mesh_key} mesh"
    return f'''from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coilgen import CoilDesign


@dataclass(slots=True)
class {class_name}(CoilDesign):
    """User-facing settings for {geometry_name}.

    Replace the placeholder parameters with physically meaningful SI-unit
    fields for the new geometry. Keep expert-only numerical controls in
    advanced_parameters rather than adding them here.
    """

    geometry_length_m: float = 0.25
    geometry_width_m: float = 0.25
    mesh_divisions: int = 40

    # Stream-function optimization strategy.
    # Choose exactly one active line:
    #   "tikhonov" uses the regularized inverse solve.
    #   "constrained_quadratic" directly minimizes current/resistance with the
    #   DSV field error in the solver constraint. This is preferred for fair
    #   geometry-efficiency comparisons when CVXPY is installed.
    #   "target_error_constrained_efficiency" finds the lowest-current solution
    #   by scanning Tikhonov multipliers. This is the dependency-free fallback.
    optimization_method: str = "tikhonov"
    # optimization_method: str = "constrained_quadratic"
    # optimization_method: str = "target_error_constrained_efficiency"

    # Used by the fixed-error constrained optimization methods.
    target_error_constraint_metric: str = "norm"  # "norm", "rms", or "std"
    target_error_tolerance_percent: float = 5.0
    target_error_tolerance_absolute_T: float = 0.0
    target_error_std_limit_T: float = 0.0
    target_error_rms_limit_T: float = 0.0
    target_error_constraint_candidate_count: int = 61
    target_error_auto_retry_enabled: bool = True
    target_error_auto_retry_max_reruns: int = 10
    target_error_auto_retry_factor_shrink: float = 0.1
    target_error_auto_retry_candidate_growth: float = 1.2
    target_error_auto_retry_max_candidate_count: int = 201
    constrained_quadratic_fallback_policy: str = "target_error_scan"
    constrained_optimization_threads: int = 0

    def geometry_config(self) -> dict[str, Any]:
        return {{
            "coil_mesh_file": "{mesh_key}",
            # Replace this parameter list with the low-level CoilGenConfig
            # fields consumed by coilgen.geometry.load_mesh_from_config().
            "new_geometry_mesh_parameter_list": (
                self.geometry_length_m,
                self.geometry_width_m,
                self.mesh_divisions,
            ),
        }}
'''


def write_coil_geometry_api_template(
    geometry_name: str,
    output_path: str | Path,
    *,
    class_name: str | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(coil_geometry_api_template(geometry_name, class_name=class_name), encoding="utf-8")
    return path
