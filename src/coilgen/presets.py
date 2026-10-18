from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .config import CoilGenConfig


@dataclass(frozen=True, slots=True)
class RecommendedSettingsPreset:
    """Named CoilGen configuration overlay for common design workflows."""

    name: str
    description: str
    values: Mapping[str, Any]

    def apply_to_mapping(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(values or {})
        merged.update(dict(self.values))
        return merged

    def apply_to_config(self, config: CoilGenConfig | None = None) -> CoilGenConfig:
        base = {} if config is None else asdict(config)
        base.update(dict(self.values))
        return CoilGenConfig(**base)


RECOMMENDED_SETTINGS_PRESETS: dict[str, RecommendedSettingsPreset] = {
    "quick": RecommendedSettingsPreset(
        name="quick",
        description="Fast smoke-test mode for cross-project API, units, routing-contract, and report checks.",
        values={
            "target_region_resolution": 5,
            "levels": 4,
            "min_point_loop_number": 3,
            "gauss_order": 1,
            "regularization_selection_method": "fixed",
            "tikonov_reg_factor": 50.0,
            "skip_postprocessing": True,
            "skip_sweep": True,
            "skip_inductance_calculation": True,
            "plot_flag": False,
            "save_stl_flag": False,
            "low_memory_mode": True,
            "retain_heavy_fields": False,
            "contour_only_sweep_mode": True,
        },
    ),
    "quick_debug": RecommendedSettingsPreset(
        name="quick_debug",
        description="Small meshes, low target resolution, contour-only output for fast API/debug checks.",
        values={
            "target_region_resolution": 5,
            "levels": 4,
            "min_point_loop_number": 3,
            "gauss_order": 1,
            "regularization_selection_method": "fixed",
            "tikonov_reg_factor": 50.0,
            "skip_postprocessing": True,
            "skip_sweep": True,
            "skip_inductance_calculation": True,
            "plot_flag": False,
            "save_stl_flag": False,
            "low_memory_mode": True,
            "retain_heavy_fields": False,
            "contour_only_sweep_mode": True,
        },
    ),
    "validation": RecommendedSettingsPreset(
        name="validation",
        description="Moderate deterministic mode for cross-project regression tests and schema validation.",
        values={
            "target_region_resolution": 7,
            "levels": 8,
            "min_point_loop_number": 4,
            "gauss_order": 1,
            "regularization_selection_method": "l_curve",
            "l_curve_factor_count": 21,
            "target_error_constraint_factor_count": 41,
            "skip_postprocessing": False,
            "skip_sweep": False,
            "skip_inductance_calculation": False,
            "plot_flag": False,
            "save_stl_flag": False,
            "retain_heavy_fields": False,
        },
    ),
    "balanced": RecommendedSettingsPreset(
        name="balanced",
        description="Default interactive design settings with full routing and standard reports.",
        values={
            "target_region_resolution": 7,
            "levels": 8,
            "min_point_loop_number": 4,
            "gauss_order": 1,
            "regularization_selection_method": "l_curve",
            "l_curve_factor_count": 21,
            "skip_postprocessing": False,
            "skip_sweep": False,
            "skip_inductance_calculation": False,
            "plot_flag": False,
            "save_stl_flag": False,
        },
    ),
    "paper_quality": RecommendedSettingsPreset(
        name="paper_quality",
        description="Higher target resolution and regularization scans for publication figures.",
        values={
            "target_region_resolution": 9,
            "levels": 12,
            "min_point_loop_number": 5,
            "gauss_order": 2,
            "regularization_selection_method": "l_curve",
            "l_curve_factor_count": 41,
            "target_error_constraint_factor_count": 91,
            "skip_postprocessing": False,
            "skip_sweep": False,
            "skip_inductance_calculation": False,
            "plot_flag": False,
            "save_stl_flag": False,
            "retain_heavy_fields": True,
        },
    ),
    "manufacturing_review": RecommendedSettingsPreset(
        name="manufacturing_review",
        description="Full routing, connector-clearance, sweep, and export diagnostics for layout review.",
        values={
            "target_region_resolution": 7,
            "levels": 10,
            "min_point_loop_number": 5,
            "skip_postprocessing": False,
            "skip_sweep": False,
            "skip_inductance_calculation": False,
            "skip_calculation_min_winding_distance": False,
            "within_group_cut_candidate_count": 12,
            "inter_group_connection_candidate_count": 24,
            "min_inter_group_connector_clearance": 0.0,
            "sweep_self_intersection_check": "capsule_fast",
            "plot_flag": False,
            "save_stl_flag": True,
            "retain_heavy_fields": True,
        },
    ),
    "layout_review": RecommendedSettingsPreset(
        name="layout_review",
        description="Full routing/manufacturing diagnostics for checking final contour connectivity.",
        values={
            "target_region_resolution": 7,
            "levels": 10,
            "min_point_loop_number": 5,
            "skip_postprocessing": False,
            "skip_sweep": False,
            "skip_inductance_calculation": False,
            "skip_calculation_min_winding_distance": False,
            "within_group_cut_candidate_count": 12,
            "inter_group_connection_candidate_count": 24,
            "plot_flag": False,
            "save_stl_flag": True,
            "retain_heavy_fields": True,
        },
    ),
    "active_shielded_gradient": RecommendedSettingsPreset(
        name="active_shielded_gradient",
        description="Gradient-coil mode with separate DSV/suppression constraints and stream-current reporting.",
        values={
            "target_field_component": "Bz",
            "field_evaluation_current_source": "stream_function",
            "target_region_resolution": 7,
            "levels": 10,
            "min_point_loop_number": 4,
            "gauss_order": 1,
            "regularization_selection_method": "l_curve",
            "l_curve_factor_count": 31,
            "target_error_constraint_mode": "separate",
            "target_error_tolerance_percent": 5.0,
            "suppression_error_tolerance_percent": 5.0,
            "skip_postprocessing": False,
            "skip_sweep": False,
            "skip_inductance_calculation": False,
            "plot_flag": False,
            "save_stl_flag": False,
            "retain_heavy_fields": True,
        },
    ),
    "local_b0_shim": RecommendedSettingsPreset(
        name="local_b0_shim",
        description="Hz-based local B0 shim workflow with dense target residual reporting and plot-gallery diagnostics.",
        values={
            "target_field_units": "Hz",
            "field_display_units": "Hz",
            "target_gradient_strength": 1.0,
            "target_region_resolution": 7,
            "levels": 10,
            "min_point_loop_number": 4,
            "gauss_order": 1,
            "regularization_selection_method": "l_curve",
            "l_curve_factor_count": 31,
            "target_error_constraint_mode": "separate",
            "target_error_tolerance_percent": 5.0,
            "skip_postprocessing": True,
            "skip_sweep": True,
            "skip_inductance_calculation": True,
            "plot_flag": False,
            "save_stl_flag": False,
            "retain_heavy_fields": True,
        },
    ),
}


def recommended_settings(name: str) -> RecommendedSettingsPreset:
    """Return one of the named recommended CoilGen settings presets."""

    key = str(name).strip().lower()
    try:
        return RECOMMENDED_SETTINGS_PRESETS[key]
    except KeyError as exc:
        known = ", ".join(sorted(RECOMMENDED_SETTINGS_PRESETS))
        raise KeyError(f"Unknown CoilGen settings preset {name!r}. Known presets: {known}") from exc


def apply_recommended_settings(
    values: Mapping[str, Any] | CoilGenConfig | None = None,
    *,
    preset: str = "balanced",
) -> dict[str, Any] | CoilGenConfig:
    """Apply a recommended settings preset to a dict or ``CoilGenConfig``."""

    selected = recommended_settings(preset)
    if isinstance(values, CoilGenConfig):
        return selected.apply_to_config(values)
    return selected.apply_to_mapping(values)
