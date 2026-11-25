"""Configuration groups and stage dependencies, independent of clients and solvers.

CoilGenConfig declares values/defaults; this module owns their interpretation
for grouped views and invalidation. Unknown fields conservatively reprepare.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .config import CoilGenConfig


class PipelineStage(str, Enum):
    PREPARED = "prepared"
    SHEET = "sheet"
    CONTOURS = "contours"
    ROUTED = "routed"
    EVALUATED = "evaluated"

    @property
    def index(self) -> int:
        return tuple(type(self)).index(self)


# These are explicit exceptions to conservative preparation invalidation.
# In particular, wire area, sheet material, current and spacing constraints are
# deliberately not classified as routing-only settings.
_DISPLAY_FIELDS = frozenset(
    {
        "progress_flag",
        "progress_min_interval_seconds",
        "progress_percent_step",
        "plot_flag",
        "plot_pointcloud_max_points",
        "plot_pointcloud_random_seed",
        "field_display_units",
        "output_directory",
        "save_stl_flag",
        "save_wire_trace_csv",
        "wire_trace_csv_format",
        "wire_trace_csv_length_unit",
        "wire_trace_csv_decimal_places",
        "chunked_output_writing",
    }
)
_CONTOUR_FIELDS = frozenset(
    {
        "levels",
        "pot_offset_factor",
        "contour_part_phase_offset_deltas",
        "level_set_method",
        "zero_crossing_level_strategy",
        "min_point_loop_number",
        "min_loop_signifcance",
        "smooth_flag",
        "smooth_factor",
        "skip_calculation_min_winding_distance",
        "skip_unconnected_contour_wire_distance",
        "wire_distance_method",
        "wire_distance_chunk_size",
        "wire_distance_spatial_min_segments",
        "area_perimeter_deletion_ratio",
        "max_allowed_angle_within_coil_track",
        "min_allowed_angle_within_coil_track",
        "tiny_segment_length_percentage",
    }
)
_ROUTING_FIELDS = frozenset(
    {
        "interconnection_cut_width",
        "interconnection_axis",
        "within_group_cut_width",
        "inter_group_cut_width",
        "interconnection_method",
        "group_interconnection_method",
        "normal_shift_length",
        "normal_shift_smooth_factors",
        "normal_sweep_smooth_factor",
        "force_cut_selection",
        "within_group_cut_candidate_count",
        "min_opening_cut_span",
        "fail_on_route_quality_issues",
        "skip_postprocessing",
        "skip_normal_shift",
        "skip_sweep",
        "make_cylndrical_pcb",
        "contour_only_sweep_mode",
        "pcb_interconnection_method",
        "pcb_spiral_end_shift_factor",
        "track_width_factor",
        "min_inter_group_connector_clearance",
        "suspicious_inter_group_connector_ratio",
    }
)
_EVALUATION_FIELDS = frozenset(
    {
        "skip_inductance_calculation",
        "inductance_source",
        "low_memory_mode",
        "retain_heavy_fields",
    }
)
_SHEET_FIELDS = frozenset(
    {
        "sf_opt_method",
        "stream_function_cost_function",
        "tikonov_reg_factor",
        "regularization_selection_method",
        "conductor_thickness",
        "specific_conductivity_conductor",
        "conductor_cross_section_width",
        "conductor_cross_section_height",
        "cross_sectional_points",
        "max_current_A",
        "max_wire_current_A",
        "min_wire_distance_m",
        "min_turn_distance",
        "surface_current_density_limit_A_per_m",
        "engineering_contour_step_safety_factor",
        "engineering_contour_max_levels",
        "gradient_hardware_infeasible_policy",
    }
)


_LEVEL_FIELDS = frozenset({"levels", "pot_offset_factor", "contour_level_grid_mode", "contour_level_range_percentile"})


def config_field_stage(name: str, config: CoilGenConfig | None = None) -> PipelineStage | None:
    """Earliest affected stage; unknown/new inputs invalidate all operators."""
    if name in _DISPLAY_FIELDS:
        return None
    if name in _LEVEL_FIELDS:
        return PipelineStage.SHEET if config is None or _level_settings_affect_sheet(config) else PipelineStage.CONTOURS
    if name in {"skip_postprocessing", "contour_only_sweep_mode", "skip_sweep"}:
        # These modes decide whether contour resampling is deferred and whether
        # sparsification/pruning checks candidate solids before route creation.
        return PipelineStage.CONTOURS
    if name in _CONTOUR_FIELDS or name.startswith(("contour_path_", "contour_resample_", "contour_smoothing_")):
        return PipelineStage.CONTOURS
    if name in {"routing_group_plan", "routing_editor_plan"}:
        return PipelineStage.ROUTED
    if name in _ROUTING_FIELDS or name.startswith(("joint_group_routing", "routing_", "inter_group_", "sweep_")):
        if config is None or _contours_check_manufacturing(config):
            return PipelineStage.CONTOURS
        return PipelineStage.ROUTED
    if name in _EVALUATION_FIELDS or name.startswith("gradient_slice_map"):
        return PipelineStage.EVALUATED
    if name in _SHEET_FIELDS or name.startswith(
        (
            "constrained_",
            "regularization_",
            "l_curve_",
            "stream_function_",
            "optimization_solver",
            "turn_spacing_",
            "target_error_",
            "suppression_error_",
        )
    ):
        return PipelineStage.SHEET
    return PipelineStage.PREPARED


def _level_settings_affect_sheet(config: CoilGenConfig) -> bool:
    # Coupling includes hardware rescaling of the preoptimized stream function and automatic
    # factor selection. Reusing a stream function is permitted only when every linked
    # current/spacing/contour-step guard is disabled.
    controls = (
        "max_current_A",
        "max_wire_current_A",
        "min_wire_distance_m",
        "min_turn_distance",
        "surface_current_density_limit_A_per_m",
        "turn_spacing_regularization_factor",
        "regularization_auto_range_max_contour_step_A",
    )
    return any(float(getattr(config, name, 0.0)) > 0.0 for name in controls)


def _contours_check_manufacturing(config: CoilGenConfig) -> bool:
    return bool(
        config.contour_only_sweep_mode
        and not config.skip_sweep
        and any(
            str(getattr(config, name, "off")).strip().lower() != "off"
            for name in (
                "contour_sparsification_mode",
                "contour_sparsification_pruning_mode",
                "contour_sparsification_post_pruning_merge_mode",
            )
        )
    )


@dataclass(frozen=True, slots=True)
class ConfigField:
    name: str
    groups: tuple[str, ...]

    def affected_stage(self, config: CoilGenConfig | None = None) -> PipelineStage | None:
        return config_field_stage(self.name, config)


@lru_cache(maxsize=1)
def settings_groups() -> Mapping[str, type]:
    from .config import (
        MeshSettings,
        TargetSettings,
        OptimizationSettings,
        RoutingSettings,
        PerformanceSettings,
        ManufacturingSettings,
    )

    return MappingProxyType(
        {
            "mesh": MeshSettings,
            "target": TargetSettings,
            "optimization": OptimizationSettings,
            "routing": RoutingSettings,
            "performance": PerformanceSettings,
            "manufacturing": ManufacturingSettings,
        }
    )


@lru_cache(maxsize=1)
def config_fields() -> Mapping[str, ConfigField]:
    """Read-only catalog derived from the actual dataclasses, never a parallel field list."""
    from .config import CoilGenConfig

    names = {item.name for item in fields(CoilGenConfig)}
    memberships = {name: [] for name in names}
    for group, settings_type in settings_groups().items():
        for item in fields(settings_type):
            if item.name not in names:
                raise ValueError(f"Unknown config field {item.name!r} in {settings_type.__name__}.")
            memberships[item.name].append(group)
    return MappingProxyType(
        {item.name: ConfigField(item.name, tuple(memberships[item.name])) for item in fields(CoilGenConfig)}
    )


def settings_view(config: CoilGenConfig, settings_type: type):
    """Build a typed group from the same catalog used for dependency inspection."""
    group = next(name for name, value in settings_groups().items() if value is settings_type)
    return settings_type(
        **{name: getattr(config, name) for name, item in config_fields().items() if group in item.groups}
    )
