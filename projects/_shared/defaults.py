from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any


COMMON_COILGEN_DEFAULTS: dict[str, Any] = {
    "field_shape_function": "y",
    "target_gradient_strength": 1.0,
    "coil_mesh_file": "create cylinder mesh",
    "cylinder_mesh_parameter_list": (0.5, 0.25, 40, 40, 1, 0, 0, 0),
    "target_region_radius": 0.08,
    "target_region_resolution": 5,
    "levels": 8,
    "contour_discretization_method": "level_count",
    "max_current_A": 0.0,
    "max_wire_current_A": 0.0,
    "min_wire_distance_m": 0.0,
    "surface_current_density_limit_A_per_m": 0.0,
    "engineering_contour_step_safety_factor": 1.0,
    "engineering_contour_max_levels": 0,
    "pot_offset_factor": 0.25,
    "min_point_loop_number": 4,
    "gauss_order": 1,
    "sf_opt_method": "tikhonov",
    "stream_function_cost_function": "field_error_plus_resistance",
    "preoptimized_stream_function_file": "none",
    "preoptimized_stream_function_field_name": "stream_function",
    "preoptimized_stream_function_match_tolerance": 1e-8,
    "sf_source_file": "none",
    "tikonov_reg_factor": 50.0,
    "regularization_selection_method": "l_curve",
    "l_curve_scan_method": "normal_equations",
    "l_curve_curvature_smoothing_window": 5,
    "l_curve_factor_count": 21,
    "l_curve_factor_min_multiplier": 1e-4,
    "l_curve_factor_max_multiplier": 1e4,
    "l_curve_regularization_factors": (),
    "min_turn_distance": 0.0,
    "turn_spacing_regularization_factor": 0.0,
    "turn_spacing_max_iterations": 4,
    "turn_spacing_penalty_growth": 10.0,
    "parallel_worker_count": 0,
    "parallel_min_work_size": 50000,
    "sensitivity_chunks_per_worker": 4,
    "sensitivity_calculation_method": "vectorized",
    "biot_savart_chunk_size": 512,
    "inductance_chunk_size": 256,
    "wire_distance_method": "auto",
    "wire_distance_chunk_size": 512,
    "wire_distance_spatial_min_segments": 3000,
    "parallel_max_chunk_memory_mb": 64.0,
    "optimization_solver": "auto",
    "optimization_solver_residual_tolerance": 1e-8,
    # Alternative optimization mode:
    # sf_opt_method="target_error_constrained_efficiency" minimizes current
    # norm while keeping the weighted target-field residual below this budget.
    "target_error_tolerance_percent": 5.0,
    "target_error_tolerance_absolute_T": 0.0,
    "target_error_constraint_factor_count": 61,
    "target_error_auto_retry_enabled": True,
    "target_error_auto_retry_max_reruns": 10,
    "target_error_auto_retry_factor_shrink": 0.1,
    "target_error_auto_retry_candidate_growth": 1.2,
    "target_error_auto_retry_max_candidate_count": 201,
    "progress_flag": True,
    "progress_min_interval_seconds": 1.0,
    "progress_percent_step": 5.0,
    "inter_group_connection_path_method": "direct_3d",
    "skip_postprocessing": False,
    "skip_sweep": False,
    # Keep final thin-wire inductance/resistance evaluation active by default.
    # "auto" uses the connected routed wire path when postprocessing is enabled
    # and closed unconnected contours when contour-only studies skip routing.
    "skip_inductance_calculation": False,
    "inductance_source": "auto",
    "skip_calculation_min_winding_distance": True,
    # Rectangular copper conductor used for final resistance and equivalent
    # round-wire radius in the Neumann/Biot-Savart inductance estimate.
    "conductor_cross_section_width": 0.001,
    "conductor_cross_section_height": 0.001,
    "specific_conductivity_conductor": 0.018e-6,
    "plot_flag": False,
    "save_stl_flag": False,
}


COMMON_OUTPUT_DEFAULTS: dict[str, Any] = {
    "name": "coilgen_example",
    "save_svg_gallery": False,
}


COMMON_PLOT_DEFAULTS: dict[str, Any] = {
    "show_active": True,
    "close_previous": True,
    "mode": "book",
    "include": (
        "wire",
        "wire_sequence",
        "contour_current_orientation",
        "contour_groups",
        "field_pointclouds",
        "spherical_harmonics",
        "error_maps",
        "performance_table",
    ),
}


def common_coilgen_defaults(**overrides: Any) -> dict[str, Any]:
    return merge_dicts(COMMON_COILGEN_DEFAULTS, overrides)


def common_output_defaults(**overrides: Any) -> dict[str, Any]:
    return merge_dicts(COMMON_OUTPUT_DEFAULTS, overrides)


def common_plot_defaults(**overrides: Any) -> dict[str, Any]:
    return merge_dicts(COMMON_PLOT_DEFAULTS, overrides)


def merge_dicts(*parts: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = merge_dicts(merged[key], value)
            else:
                merged[key] = deepcopy(value)
    return merged
