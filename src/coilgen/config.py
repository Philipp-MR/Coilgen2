from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class MeshSettings:
    coil_mesh_file: str
    imported_mesh_length_scale_to_m: float
    surface_is_cylinder_flag: bool
    cylinder_mesh_parameter_list: Sequence[float]
    planar_mesh_parameter_list: Sequence[float]
    circular_mesh_parameter_list: Sequence[float]
    annular_disc_mesh_parameter_list: Sequence[float]
    partial_cylinder_mesh_parameter_list: Sequence[float]
    elliptical_cylinder_mesh_parameter_list: Sequence[float]
    superellipse_cylinder_mesh_parameter_list: Sequence[float]
    spherical_cap_mesh_parameter_list: Sequence[float]
    ellipsoidal_cap_mesh_parameter_list: Sequence[float]
    conical_frustum_mesh_parameter_list: Sequence[float]
    toroidal_segment_mesh_parameter_list: Sequence[float]
    ruled_surface_mesh_parameter_list: Sequence[float]
    surface_of_revolution_mesh_parameter_list: Sequence[float]
    circular_biplanar_mesh_parameter_list: Sequence[float]
    biplanar_mesh_parameter_list: Sequence[float]
    iteration_num_mesh_refinement: int
    generated_mesh_rotation_axis: Sequence[float]
    generated_mesh_rotation_angle: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TargetSettings:
    field_shape_function: str
    target_gradient_strength: float
    target_gradient_normalization: str
    target_gradient_axis: str
    target_region_radius: float
    target_region_resolution: int
    target_field_definition_file: str
    target_field_definition_field_name: str
    target_field_units: str
    field_display_units: str
    target_region_geometry: dict[str, Any] | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    sf_opt_method: str
    stream_function_cost_function: str
    research_physics: dict[str, Any] | None
    preoptimized_stream_function_file: str
    preoptimized_stream_function_field_name: str
    tikonov_reg_factor: float
    regularization_selection_method: str
    l_curve_scan_method: str
    regularization_auto_range: bool
    regularization_auto_range_min_singular_relative: float
    regularization_auto_range_target_projection_relative: float
    regularization_auto_range_lower_padding: float
    regularization_auto_range_upper_padding: float
    regularization_auto_range_min_factor: float
    regularization_auto_range_max_factor: float
    regularization_auto_range_max_contour_step_A: float
    target_error_constraint_mode: str
    target_error_constraint_metric: str
    mri_shim_metric_mode: str
    mri_shim_cvar_alpha: float
    mri_shim_residual_rms_weight: float
    mri_shim_residual_cvar_weight: float
    mri_shim_spatial_gradient_weight: float
    mri_shim_spatial_scale_m: float
    mri_shim_reference_frequency_hz: float
    mri_shim_derivative_neighbor_count: int
    target_error_tolerance_percent: float
    target_error_tolerance_absolute_T: float
    target_error_std_limit_T: float
    target_error_rms_limit_T: float
    layout_target_error_std_limit_T: float
    layout_target_error_rms_limit_T: float
    layout_target_error_acceptance_absolute_T: float
    suppression_error_tolerance_percent: float
    suppression_error_tolerance_absolute_T: float
    suppression_field_rms_limit_T: float
    suppression_field_component: str
    suppression_field_peak_limit_T: float
    suppression_surface_weighting: str
    suppression_normal_source: str
    target_error_constraint_acceptance_absolute_T: float
    target_error_constraint_acceptance_tolerance: float
    target_error_constraint_factor_count: int
    target_error_constraint_factor_min_multiplier: float
    target_error_constraint_factor_max_multiplier: float
    target_error_infeasible_fallback: str
    target_error_infeasible_residual_margin_percent: float
    target_error_auto_retry_enabled: bool
    target_error_auto_retry_max_reruns: int
    target_error_auto_retry_factor_shrink: float
    target_error_auto_retry_candidate_growth: float
    target_error_auto_retry_max_candidate_count: int
    gradient_strength_tolerance_percent: float
    linearity_region: str
    linearity_reference_radius_m: float
    linearity_rms_tolerance_percent: float
    linearity_peak_tolerance_percent: float
    linearity_cross_gradient_tolerance_percent: float
    linearity_normalization: str
    linearity_fit_offset: bool
    gradient_slice_maps_enabled: bool
    gradient_slice_map_axis: str
    gradient_slice_map_slice_count: int
    gradient_slice_map_grid_size: int | Sequence[int]
    gradient_slice_map_domain: str
    gradient_slice_map_source: str
    gradient_slice_map_current: str
    gradient_slice_map_field_components: Sequence[str]
    gradient_slice_map_gradient_components: Sequence[str]
    gradient_slice_map_concomitant_components: Sequence[str]
    gradient_slice_map_gradient_method: str
    gradient_slice_map_gradient_step_m: float
    gradient_slice_map_overlay: Sequence[str]
    gradient_slice_map_overlay_mode: str
    gradient_slice_map_overlay_slab_thickness_m: float
    optimization_solver: str
    constrained_optimization_solver: str
    constrained_optimization_tolerance: float
    constrained_optimization_max_iterations: int
    constrained_optimization_threads: int
    constrained_optimization_verbose: bool
    constrained_quadratic_fallback_policy: str
    constrained_quadratic_objective_mode: str
    constrained_quadratic_resistance_tiebreak_factor: float
    shield_gradient_opposition_enabled: bool
    shield_gradient_max_fraction_of_target: float
    stream_function_smoothness_factor: float
    stream_function_basis: str
    stream_function_spectral_cutoff_length_m: float
    stream_function_spectral_cutoff_policy: str
    stream_function_spectral_max_modes: int
    stream_function_spectral_min_modes: int
    contour_discretization_method: str
    max_current_A: float
    max_wire_current_A: float
    min_wire_distance_m: float
    surface_current_density_limit_A_per_m: float
    engineering_contour_step_safety_factor: float
    engineering_contour_max_levels: int
    gradient_hardware_infeasible_policy: str
    min_turn_distance: float
    turn_spacing_density_percentile: float
    turn_spacing_regularization_factor: float
    turn_spacing_max_iterations: int
    turn_spacing_penalty_growth: float
    contour_level_range_percentile: float
    contour_level_grid_mode: str
    zero_crossing_level_strategy: str
    contour_symmetry_validation: str
    contour_symmetry_geometry_tolerance: float
    contour_symmetry_stream_function_tolerance: float
    contourability_validation: str
    contourability_repair_enabled: bool
    contourability_max_rms_error_percent: float
    contourability_max_mean_error_percent: float
    contourability_max_peak_error_percent: float
    contourability_max_tiny_loop_fraction: float
    contourability_max_contour_count: int
    contourability_min_loop_length_m: float
    contourability_max_closure_gap_m: float
    contourability_max_surface_error_m: float
    contourability_repair_level_factors: Sequence[float]
    contourability_repair_phase_offsets: Sequence[float]
    contourability_repair_range_percentiles: Sequence[float]
    contourability_repair_zero_crossing_strategies: Sequence[str]
    contourability_repair_min_levels: int
    contourability_repair_max_levels: int
    contour_field_constrained_smoothing_enabled: bool
    contour_field_smoothing_candidates: Sequence[Sequence[float]]
    contour_field_smoothing_min_improvement_percent: float
    contour_loop_pruning_mode: str
    contour_loop_pruning_residual_relative_tolerance: float
    contour_loop_pruning_residual_absolute_tolerance_T: float
    contour_loop_pruning_max_removed_loops: int
    contour_loop_pruning_preserve_symmetry: bool
    contour_sparsification_mode: str
    contour_sparsification_positioning: str
    contour_sparsification_max_rms_field_error_percent: float
    contour_sparsification_max_peak_field_error_percent: float
    contour_sparsification_max_merges: int
    contour_sparsification_max_passes: int
    contour_sparsification_max_turn_multiplicity: int
    contour_sparsification_field_fit_candidate_count: int
    contour_sparsification_preserve_symmetry: bool
    contour_sparsification_realization: str
    contour_sparsification_residual_relative_tolerance: float | None
    contour_sparsification_residual_absolute_tolerance_T: float | None
    contour_sparsification_pruning_mode: str
    contour_sparsification_pruning_residual_relative_tolerance: float
    contour_sparsification_pruning_residual_absolute_tolerance_T: float
    contour_sparsification_pruning_max_removed_geometries: int
    contour_sparsification_pruning_target_geometry_count: int
    contour_sparsification_pruning_preserve_symmetry: bool
    contour_sparsification_pruning_search_mode: str
    contour_sparsification_pruning_exact_max_orbits: int
    contour_sparsification_pruning_beam_width: int
    contour_sparsification_post_pruning_merge_mode: str
    contour_sparsification_post_pruning_max_passes: int
    symmetry_reduction: str
    symmetry_reduction_source: str
    symmetry_reduction_mirror_plane_point: Sequence[float]
    symmetry_reduction_mirror_plane_normal: Sequence[float]
    symmetry_reduction_half_space: str
    symmetry_reduction_clip_tolerance: float
    symmetry_reduction_plane_snap_tolerance: float
    symmetry_reduction_target_parity: str
    symmetry_reduction_target_tolerance: float
    symmetry_reduction_target_parity_policy: str
    symmetry_reduction_target_geometry_tolerance_m: float
    symmetry_reduction_target_parity_tolerance: float
    symmetry_reduction_target_weight_tolerance: float
    mesh_quality_policy: str
    mesh_quality_max_triangle_edge_ratio: float
    mesh_quality_max_altitude_aspect_ratio: float
    mesh_quality_max_edge_length_ratio: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoutingSettings:
    routing_circuit_scope: str
    joint_group_routing: bool
    joint_group_routing_search_mode: str
    joint_group_routing_field_acceptance: str
    joint_group_routing_clearance_m: float | None
    joint_group_routing_max_field_change_rel: float
    joint_group_routing_max_longitudinal_field_change_rel: float | None
    joint_group_routing_max_candidates: int
    joint_group_routing_b0_weight: float
    joint_group_routing_search_cuts: bool
    routing_jump_height_tolerance_m: float
    routing_jump_sample_spacing_m: float
    topological_grouping_method: str
    inter_group_connection_path_method: str
    inter_group_connection_candidate_count: int
    inter_group_cut_alignment: str
    inter_group_cut_alignment_distance_rel_tol: float
    inter_group_cut_alignment_candidate_factor: int
    within_group_cut_candidate_count: int
    min_inter_group_connector_clearance: float
    skip_postprocessing: bool
    skip_normal_shift: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PerformanceSettings:
    parallel_worker_count: int
    parallel_min_work_size: int
    sensitivity_chunks_per_worker: int
    sensitivity_calculation_method: str
    biot_savart_chunk_size: int
    inductance_chunk_size: int
    parallel_max_chunk_memory_mb: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ManufacturingSettings:
    skip_sweep: bool
    contour_only_sweep_mode: bool
    skip_inductance_calculation: bool
    inductance_source: str
    conductor_cross_section_width: float
    conductor_cross_section_height: float
    cross_sectional_points: Sequence[Sequence[float]]
    conductor_mounting_mode: str
    sweep_surface_realization: str
    sweep_path_simplification: str
    sweep_collinearity_tolerance: float
    sweep_tangent_angle_tolerance_deg: float
    sweep_frame_angle_tolerance_deg: float
    sweep_max_segment_length: float
    sweep_max_chord_error: float
    sweep_max_frame_rotation_deg: float
    sweep_bend_mode: str
    sweep_min_bend_radius: float
    sweep_corner_cluster_max_source_deviation_m: float
    sweep_bend_arc_angle_deg: float
    sweep_corner_detection_angle_deg: float
    sweep_max_curvature_envelope_ratio: float
    sweep_frame_method: str
    sweep_surface_normal_smoothing_iterations: int
    sweep_surface_normal_constraint_strength: float
    sweep_self_intersection_check: str
    sweep_insulation_thickness: float
    sweep_manufacturing_tolerance: float
    sweep_bend_clearance_margin: float
    make_cylndrical_pcb: bool
    save_stl_flag: bool
    save_wire_trace_csv: bool
    wire_trace_csv_format: str
    wire_trace_csv_length_unit: str
    wire_trace_csv_decimal_places: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class CoilGenConfig:
    """Configuration matching the main MATLAB CoilGen name-value inputs."""

    coil_mesh_file: str = "none"
    field_shape_function: str = "x"
    target_gradient_strength: float = 1.0
    target_gradient_normalization: str = "auto"
    target_gradient_axis: str = "auto"
    pot_offset_factor: float = 0.5
    contour_part_phase_offset_deltas: Sequence[float] = field(default_factory=tuple)
    target_mesh_file: str = "none"
    secondary_target_mesh_file: str = "none"
    secondary_target_weight: float = 1.0
    use_only_target_mesh_verts: bool = False
    target_field_definition_file: str = "none"
    target_field_definition_field_name: str = "none"
    target_field_component: str = "Bz"
    problem_schema_version: str = ""
    design_context: str = "mr_gradient"
    target_field_units: str = "T"
    field_display_units: str = "T"
    sf_opt_method: str = "tikhonov"
    stream_function_cost_function: str = "field_error_plus_resistance"
    research_physics: dict[str, Any] | None = None
    preoptimized_stream_function_file: str = "none"
    preoptimized_stream_function_field_name: str = "stream_function"
    preoptimized_stream_function_match_tolerance: float = 1e-8
    sf_source_file: str = "none"
    tikonov_reg_factor: float = 1.0
    regularization_selection_method: str = "l_curve"
    l_curve_scan_method: str = "normal_equations"
    regularization_auto_range: bool = False
    regularization_auto_range_min_singular_relative: float = 1.0e-4
    regularization_auto_range_target_projection_relative: float = 1.0e-4
    regularization_auto_range_lower_padding: float = 5.0e-2
    regularization_auto_range_upper_padding: float = 1.0e2
    regularization_auto_range_min_factor: float = 0.0
    regularization_auto_range_max_factor: float = 0.0
    regularization_auto_range_max_contour_step_A: float = 0.0
    l_curve_curvature_smoothing_window: int = 5
    l_curve_factor_count: int = 21
    l_curve_factor_min_multiplier: float = 1e-4
    l_curve_factor_max_multiplier: float = 1e4
    l_curve_regularization_factors: Sequence[float] = field(default_factory=tuple)
    field_evaluation_current_source: str = "target_fit"
    target_error_constraint_mode: str = "separate"
    target_error_constraint_metric: str = "norm"
    mri_shim_metric_mode: str = "off"
    mri_shim_cvar_alpha: float = 0.95
    mri_shim_residual_rms_weight: float = 1.0
    mri_shim_residual_cvar_weight: float = 0.25
    mri_shim_spatial_gradient_weight: float = 0.10
    mri_shim_spatial_scale_m: float = 0.0
    mri_shim_reference_frequency_hz: float = 50.0
    mri_shim_derivative_neighbor_count: int = 18
    target_error_tolerance_percent: float = 5.0
    target_error_tolerance_absolute_T: float = 0.0
    target_error_std_limit_T: float = 0.0
    target_error_rms_limit_T: float = 0.0
    layout_target_error_std_limit_T: float = 0.0
    layout_target_error_rms_limit_T: float = 0.0
    layout_target_error_acceptance_absolute_T: float = 0.0
    suppression_error_tolerance_percent: float = 5.0
    suppression_error_tolerance_absolute_T: float = 0.0
    suppression_field_rms_limit_T: float = 0.0
    suppression_field_component: str = "target"
    suppression_field_peak_limit_T: float = 0.0
    suppression_surface_weighting: str = "uniform"
    suppression_normal_source: str = "mesh"
    target_error_constraint_acceptance_absolute_T: float = 1.0e-7
    target_error_constraint_acceptance_tolerance: float = 0.0
    target_error_constraint_factor_count: int = 61
    target_error_constraint_factor_min_multiplier: float = 1e-10
    target_error_constraint_factor_max_multiplier: float = 1e10
    target_error_infeasible_fallback: str = "residual_floor"
    target_error_infeasible_residual_margin_percent: float = 100.0
    target_error_auto_retry_enabled: bool = True
    target_error_auto_retry_max_reruns: int = 10
    target_error_auto_retry_factor_shrink: float = 0.1
    target_error_auto_retry_candidate_growth: float = 1.2
    target_error_auto_retry_max_candidate_count: int = 201
    gradient_strength_tolerance_percent: float = 0.0
    linearity_region: str = "dsv"
    linearity_reference_radius_m: float = 0.0
    linearity_rms_tolerance_percent: float = 0.0
    linearity_peak_tolerance_percent: float = 0.0
    linearity_cross_gradient_tolerance_percent: float = 0.0
    linearity_normalization: str = "requested_gradient_times_radius"
    linearity_fit_offset: bool = True
    gradient_slice_maps_enabled: bool = False
    gradient_slice_map_axis: str = "z"
    gradient_slice_map_slice_count: int = 9
    gradient_slice_map_grid_size: int | Sequence[int] = 96
    gradient_slice_map_domain: str = "dsv"
    gradient_slice_map_source: str = "layout"
    gradient_slice_map_current: str = "target"
    gradient_slice_map_field_components: Sequence[str] = ("Bx", "By", "Bz")
    gradient_slice_map_gradient_components: Sequence[str] = ("dBz_dx", "dBz_dy", "dBz_dz", "gradBz_mag")
    gradient_slice_map_concomitant_components: Sequence[str] = ("concomitant_Bz", "concomitant_grad_mag")
    gradient_slice_map_gradient_method: str = "finite_difference"
    gradient_slice_map_gradient_step_m: float = 0.0
    gradient_slice_map_overlay: Sequence[str] = ("wire_path", "contours")
    gradient_slice_map_overlay_mode: str = "slice_slab"
    gradient_slice_map_overlay_slab_thickness_m: float = 0.0
    constrained_optimization_solver: str = "auto"
    constrained_optimization_tolerance: float = 1e-7
    constrained_optimization_max_iterations: int = 20000
    constrained_optimization_threads: int = 0
    constrained_optimization_verbose: bool = False
    constrained_quadratic_fallback_policy: str = "target_error_scan"
    constrained_quadratic_objective_mode: str = "balanced"
    constrained_quadratic_resistance_tiebreak_factor: float = 1.0e-3
    shield_gradient_opposition_enabled: bool = False
    shield_gradient_max_fraction_of_target: float = 0.0
    stream_function_smoothness_factor: float = 0.0
    stream_function_basis: str = "vertex"
    stream_function_spectral_cutoff_length_m: float = 0.0
    stream_function_spectral_cutoff_policy: str = "hard"
    stream_function_spectral_max_modes: int = 0
    stream_function_spectral_min_modes: int = 1
    contour_discretization_method: str = "level_count"
    max_current_A: float = 0.0
    max_wire_current_A: float = 0.0
    min_wire_distance_m: float = 0.0
    surface_current_density_limit_A_per_m: float = 0.0
    engineering_contour_step_safety_factor: float = 1.0
    engineering_contour_max_levels: int = 0
    gradient_hardware_infeasible_policy: str = "scale_to_limit"
    min_turn_distance: float = 0.0
    turn_spacing_density_percentile: float = 100.0
    turn_spacing_regularization_factor: float = 0.0
    turn_spacing_max_iterations: int = 4
    turn_spacing_penalty_growth: float = 10.0
    contour_level_range_percentile: float = 100.0
    contour_level_grid_mode: str = "centered_margin"
    contour_symmetry_validation: str = "auto"
    contour_symmetry_geometry_tolerance: float = 1e-6
    contour_symmetry_stream_function_tolerance: float = 1e-5
    contourability_validation: str = "off"
    contourability_repair_enabled: bool = True
    contourability_max_rms_error_percent: float = 30.0
    contourability_max_mean_error_percent: float = 25.0
    contourability_max_peak_error_percent: float = 100.0
    contourability_max_tiny_loop_fraction: float = 0.5
    contourability_max_contour_count: int = 0
    contourability_min_loop_length_m: float = 0.0
    contourability_max_closure_gap_m: float = 0.0
    contourability_max_surface_error_m: float = 0.0
    contourability_repair_level_factors: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    contourability_repair_phase_offsets: Sequence[float] = field(default_factory=tuple)
    contourability_repair_range_percentiles: Sequence[float] = field(default_factory=tuple)
    contourability_repair_zero_crossing_strategies: Sequence[str] = field(default_factory=tuple)
    contourability_repair_min_levels: int = 4
    contourability_repair_max_levels: int = 160
    contour_field_constrained_smoothing_enabled: bool = False
    contour_field_smoothing_candidates: Sequence[Sequence[float]] = (
        (1, 0.10),
        (2, 0.15),
        (3, 0.20),
    )
    contour_field_smoothing_min_improvement_percent: float = 0.0
    contour_loop_pruning_mode: str = "off"
    contour_loop_pruning_residual_relative_tolerance: float = 0.0
    contour_loop_pruning_residual_absolute_tolerance_T: float = 0.0
    contour_loop_pruning_max_removed_loops: int = 0
    contour_loop_pruning_preserve_symmetry: bool = True
    contour_sparsification_mode: str = "off"
    contour_sparsification_positioning: str = "midpoint_level"
    contour_sparsification_max_rms_field_error_percent: float = 1.0
    contour_sparsification_max_peak_field_error_percent: float = 5.0
    contour_sparsification_max_merges: int = 0
    contour_sparsification_max_passes: int = 1
    contour_sparsification_max_turn_multiplicity: int = 2
    contour_sparsification_field_fit_candidate_count: int = 5
    contour_sparsification_preserve_symmetry: bool = True
    contour_sparsification_realization: str = "ideal_weighted"
    contour_sparsification_residual_relative_tolerance: float | None = None
    contour_sparsification_residual_absolute_tolerance_T: float | None = None
    contour_sparsification_pruning_mode: str = "off"
    contour_sparsification_pruning_residual_relative_tolerance: float = 0.0
    contour_sparsification_pruning_residual_absolute_tolerance_T: float = 0.0
    contour_sparsification_pruning_max_removed_geometries: int = 0
    contour_sparsification_pruning_target_geometry_count: int = 0
    contour_sparsification_pruning_preserve_symmetry: bool = True
    contour_sparsification_pruning_search_mode: str = "auto"
    contour_sparsification_pruning_exact_max_orbits: int = 16
    contour_sparsification_pruning_beam_width: int = 128
    contour_sparsification_post_pruning_merge_mode: str = "off"
    contour_sparsification_post_pruning_max_passes: int = 1
    topological_grouping_method: str = "auto"
    symmetry_reduction: str = "auto"
    symmetry_reduction_source: str = "generated_parts"
    symmetry_reduction_mirror_plane_point: Sequence[float] = (0.0, 0.0, 0.0)
    symmetry_reduction_mirror_plane_normal: Sequence[float] = (1.0, 0.0, 0.0)
    symmetry_reduction_half_space: str = "positive"
    symmetry_reduction_clip_tolerance: float = 1e-9
    symmetry_reduction_plane_snap_tolerance: float = 0.0
    symmetry_reduction_target_parity: str = "auto"
    symmetry_reduction_target_tolerance: float = 1e-6
    symmetry_reduction_target_parity_policy: str = "require"
    symmetry_reduction_target_geometry_tolerance_m: float = 0.0
    symmetry_reduction_target_parity_tolerance: float = 0.0
    symmetry_reduction_target_weight_tolerance: float = 1e-8
    mesh_quality_policy: str = "warn"
    mesh_quality_max_triangle_edge_ratio: float = 25.0
    mesh_quality_max_altitude_aspect_ratio: float = 50.0
    mesh_quality_max_edge_length_ratio: float = 100.0
    levels: int = 10
    level_set_method: str = "primary"
    zero_crossing_level_strategy: str = "paired"
    fieldtype_to_evaluate: str = "field"
    surface_is_cylinder_flag: bool = True
    interconnection_cut_width: float = 0.01
    within_group_cut_width: float | None = None
    inter_group_cut_width: float | None = None
    target_region_radius: float = 0.15
    target_region_resolution: int = 10
    # None preserves MATLAB-compatible legacy sphere/target-mesh sampling.
    target_region_geometry: dict[str, Any] | None = None
    normal_shift_length: float = 0.001
    min_point_loop_number: int = 20
    min_loop_signifcance: float = 0.0
    area_perimeter_deletion_ratio: float = 5.0
    max_allowed_angle_within_coil_track: float = 120.0
    min_allowed_angle_within_coil_track: float = 0.0001
    tiny_segment_length_percentage: float = 0.0
    iteration_num_mesh_refinement: int = 0
    b_0_direction: Sequence[float] = (0.0, 0.0, 1.0)
    b_0_strength_T: float = 1.0
    geometry_source_path: Path | None = None
    imported_mesh_length_scale_to_m: float = 1.0
    output_directory: Path | None = None
    progress_flag: bool = False
    progress_min_interval_seconds: float = 1.0
    progress_percent_step: float = 5.0
    low_memory_mode: bool = False
    retain_heavy_fields: bool = True
    contour_only_sweep_mode: bool = False
    chunked_output_writing: bool = False
    smooth_flag: bool = True
    smooth_factor: float = 1.0
    contour_path_regularization_mode: str = "off"
    contour_resample_target_segment_length: float = 0.0
    contour_smoothing_iterations: int = 0
    contour_smoothing_factor: float = 0.0
    contour_smoothing_scale_mode: str = "legacy"
    contour_smoothing_mesh_gain: float = 0.1
    contour_smoothing_work_samples_per_mesh_length: int = 6
    contour_smoothing_output_samples_per_mesh_length: int = 8
    contour_smoothing_patch_rings: int = 4
    contour_smoothing_max_displacement: float = 0.0
    contour_smoothing_preserve_min_turn_distance: bool = True
    parallel_worker_count: int = 0
    parallel_min_work_size: int = 50000
    sensitivity_chunks_per_worker: int = 4
    sensitivity_calculation_method: str = "vectorized"
    biot_savart_chunk_size: int = 512
    inductance_chunk_size: int = 256
    wire_distance_method: str = "auto"
    wire_distance_chunk_size: int = 512
    wire_distance_spatial_min_segments: int = 3000
    parallel_max_chunk_memory_mb: float = 64.0
    optimization_solver: str = "auto"
    optimization_solver_residual_tolerance: float = 1e-8
    save_stl_flag: bool = True
    save_wire_trace_csv: bool = False
    wire_trace_csv_format: str = "detailed"
    wire_trace_csv_length_unit: str = "m"
    wire_trace_csv_decimal_places: int = 9
    plot_flag: bool = False
    # Display-only budget; zero preserves every sample. Numerical arrays are unchanged.
    plot_pointcloud_max_points: int = 2000
    plot_pointcloud_random_seed: int = 1729
    interconnection_method: str = "regular"
    group_interconnection_method: str = "crossed"
    # A connected layout is one series circuit; independent circuits are explicit.
    routing_circuit_scope: str = "single"
    # Adopt a joint return route only after an isolated physical/field preview.
    joint_group_routing: bool = True
    # Cheap circuit ranking followed by staged curve generation. Thorough keeps
    # the wider eager transition search for comparison and difficult cases.
    joint_group_routing_search_mode: str = "progressive"
    # Absolute qualification by default. Explicit non_regression permits only
    # inherited failures that do not worsen; budget_only allows measured scalar
    # failures subject to the hard routing delta budgets. All failures are reported.
    joint_group_routing_field_acceptance: str = "absolute"
    # Route-only spacing replaces both routing spacing aliases; source contour
    # optimization keeps its original min_wire_distance_m/min_turn_distance.
    joint_group_routing_clearance_m: float | None = None
    # Maximum pointwise vector change / peak active DSV target, at fixed current.
    joint_group_routing_max_field_change_rel: float = 0.001
    # Optional pointwise B0-component change with the same target/current scaling.
    joint_group_routing_max_longitudinal_field_change_rel: float | None = None
    joint_group_routing_max_candidates: int = 6
    # Search guide only; final selection uses jumps, longitudinal field RMS, length.
    joint_group_routing_b0_weight: float = 1.0
    # Reopen the same directed contours on alternate, unforced cut rails.
    joint_group_routing_search_cuts: bool = True
    # Physical excursion counting, relative to the nominal attachment surface.
    routing_jump_height_tolerance_m: float = 1e-6
    routing_jump_sample_spacing_m: float = 0.001
    inter_group_connection_path_method: str = "direct_3d"
    inter_group_connection_candidate_count: int = 16
    inter_group_cut_alignment: str = "auto"
    inter_group_cut_alignment_distance_rel_tol: float = 5.0e-2
    inter_group_cut_alignment_candidate_factor: int = 4
    suspicious_inter_group_connector_ratio: float = 2.0
    min_inter_group_connector_clearance: float = 0.0
    within_group_cut_candidate_count: int = 8
    min_opening_cut_span: float = 1e-10
    fail_on_route_quality_issues: bool = False
    skip_calculation_min_winding_distance: bool = True
    skip_unconnected_contour_wire_distance: bool = False
    skip_postprocessing: bool = False
    skip_inductance_calculation: bool = False
    inductance_source: str = "auto"
    skip_normal_shift: bool = False
    normal_shift_smooth_factors: Sequence[float] = (2.0, 3.0, 2.0)
    normal_sweep_smooth_factor: float = 5.0
    skip_sweep: bool = False
    make_cylndrical_pcb: bool = False
    pcb_interconnection_method: str = "spiral_in_out"
    pcb_spiral_end_shift_factor: float = 10.0
    force_cut_selection: Sequence[str] = field(default_factory=tuple)
    gauss_order: int = 2
    set_roi_into_mesh_center: bool = False
    track_width_factor: float = 0.5
    conductor_cross_section_width: float = 0.002
    conductor_cross_section_height: float = 0.002
    conductor_mounting_mode: str = "centered"
    sweep_path_simplification: str = "off"
    sweep_surface_realization: str = "analytic"
    sweep_collinearity_tolerance: float = 1e-8
    sweep_tangent_angle_tolerance_deg: float = 0.5
    sweep_frame_angle_tolerance_deg: float = 0.5
    sweep_max_segment_length: float = 0.0
    sweep_max_chord_error: float = 5.0e-5
    sweep_max_frame_rotation_deg: float = 10.0
    sweep_bend_mode: str = "miter"
    sweep_min_bend_radius: float = 0.0
    # Explicit two-way source corridor for compound intrinsic corner fillets.
    # Zero retains the existing per-corner realization without this fallback.
    sweep_corner_cluster_max_source_deviation_m: float = 0.0
    sweep_bend_arc_angle_deg: float = 5.0
    sweep_corner_detection_angle_deg: float = 15.0
    sweep_max_curvature_envelope_ratio: float = 0.5
    sweep_frame_method: str = "surface_constrained_parallel_transport"
    sweep_surface_normal_smoothing_iterations: int = 0
    # Signed support-normal attachment: 0 uses free RMF transport, 1 fixes
    # attached frames, and intermediate strengths use arclength-weighted
    # guidance over the actual profile-envelope diameter (not per-sample decay).
    sweep_surface_normal_constraint_strength: float = 0.35
    sweep_self_intersection_check: str = "capsule_fast"
    sweep_insulation_thickness: float = 0.0
    sweep_manufacturing_tolerance: float = 0.0
    sweep_bend_clearance_margin: float = 0.0
    layout_terminal_strategy: str = "undefined"
    layout_terminal_pad_count: int = 0
    layout_terminal_exit_count: int = 0
    layout_keepout_region_count: int = 0
    layout_manufacturing_intent: str = ""
    specific_conductivity_conductor: float = 0.018e-6
    conductor_thickness: float = 0.005
    cross_sectional_points: Sequence[Sequence[float]] = ((0.0, 0.0),)
    cylinder_mesh_parameter_list: Sequence[float] = (0.8, 0.3, 20, 20, 1, 0, 0, 0)
    planar_mesh_parameter_list: Sequence[float] = (0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0)
    circular_mesh_parameter_list: Sequence[float] = (0.25, 20, 1, 0, 0, 0, 0, 0, 0)
    annular_disc_mesh_parameter_list: Sequence[float] = (0.05, 0.25, 4, 40, 0, 0, 1, 0, 0, 0)
    partial_cylinder_mesh_parameter_list: Sequence[float] = (
        0.5,
        0.25,
        -1.5707963267948966,
        1.5707963267948966,
        24,
        20,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    elliptical_cylinder_mesh_parameter_list: Sequence[float] = (0.5, 0.25, 0.18, 40, 20, 1, 0, 0, 0, 0, 0, 0)
    superellipse_cylinder_mesh_parameter_list: Sequence[float] = (0.5, 0.25, 0.18, 4.0, 40, 20, 1, 0, 0, 0, 0, 0, 0)
    spherical_cap_mesh_parameter_list: Sequence[float] = (0.25, 1.5707963267948966, 40, 12, 1, 0, 0, 0, 0, 0, 0)
    ellipsoidal_cap_mesh_parameter_list: Sequence[float] = (
        0.25,
        0.20,
        0.18,
        1.5707963267948966,
        40,
        12,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    conical_frustum_mesh_parameter_list: Sequence[float] = (0.5, 0.22, 0.16, 40, 20, 1, 0, 0, 0, 0, 0, 0)
    toroidal_segment_mesh_parameter_list: Sequence[float] = (
        0.35,
        0.08,
        -1.5707963267948966,
        1.5707963267948966,
        0,
        6.283185307179586,
        32,
        12,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    ruled_surface_mesh_parameter_list: Sequence[float] = (2, 4, -0.2, -0.1, 0, 0.2, -0.1, 0, -0.2, 0.1, 0, 0.2, 0.1, 0)
    surface_of_revolution_mesh_parameter_list: Sequence[float] = (40, -0.25, 0.18, 0.0, 0.25, 0.25, 0.18)
    circular_biplanar_mesh_parameter_list: Sequence[float] = (0.08, 0.25, 0, 0, 1, 0, 0, 0, 0.2)
    biplanar_mesh_parameter_list: Sequence[float] = (0.25, 0.25, 20, 20, 1, 0, 0, 0, 0, 0, 0.2)
    flanged_split_bore_mesh_parameter_list: Sequence[float] = field(default_factory=tuple)
    shielded_flanged_split_bore_mesh_parameter_list: Sequence[float] = field(default_factory=tuple)
    shielded_circular_biplanar_mesh_parameter_list: Sequence[float] = field(default_factory=tuple)
    generated_mesh_rotation_axis: Sequence[float] = (1.0, 0.0, 0.0)
    generated_mesh_rotation_angle: float = 0.0
    create_split_bore_field_suppression_surface: str = "none"
    split_bore_area_suppression_parameters: Sequence[float] = field(default_factory=tuple)
    create_flanged_split_bore_field_suppression_surface: str = "none"
    flanged_split_bore_area_suppression_parameters: Sequence[float] = field(default_factory=tuple)
    create_circular_biplanar_field_suppression_surface: str = "none"
    circular_biplanar_area_suppression_parameters: Sequence[float] = field(default_factory=tuple)
    create_mesh_topology_field_suppression_surface: str = "none"
    mesh_topology_suppression_separation: float = 0.0
    mesh_topology_suppression_source: str = "shield"
    mesh_topology_suppression_min_clearance: float = 0.0
    mesh_topology_suppression_clearance_action: str = "none"

    def resolved_geometry_source_path(self, project_root: Path) -> Path:
        if self.geometry_source_path is not None:
            return Path(self.geometry_source_path)
        roots = (project_root, project_root.parent)
        for root in roots:
            shared_geometry = root / "project_data" / "_shared" / "legacy_coilgen" / "Geometry_Data"
            if shared_geometry.exists():
                return shared_geometry
        # Retain read compatibility for checkouts that have not migrated their
        # local data yet. New input assets belong below project_data/.
        for root in roots:
            reference_geometry = root / "reference" / "CoilGen 3_6_2026" / "Geometry_Data"
            if reference_geometry.exists():
                return reference_geometry
        return project_root / "Geometry_Data"

    def resolved_output_directory(self, project_root: Path) -> Path:
        return Path(self.output_directory) if self.output_directory is not None else project_root

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def mesh_settings(self) -> MeshSettings:
        return MeshSettings(**{item.name: getattr(self, item.name) for item in fields(MeshSettings)})

    @property
    def target_settings(self) -> TargetSettings:
        return TargetSettings(**{item.name: getattr(self, item.name) for item in fields(TargetSettings)})

    @property
    def optimization_settings(self) -> OptimizationSettings:
        return OptimizationSettings(**{item.name: getattr(self, item.name) for item in fields(OptimizationSettings)})

    @property
    def routing_settings(self) -> RoutingSettings:
        return RoutingSettings(**{item.name: getattr(self, item.name) for item in fields(RoutingSettings)})

    @property
    def performance_settings(self) -> PerformanceSettings:
        return PerformanceSettings(**{item.name: getattr(self, item.name) for item in fields(PerformanceSettings)})

    @property
    def manufacturing_settings(self) -> ManufacturingSettings:
        return ManufacturingSettings(**{item.name: getattr(self, item.name) for item in fields(ManufacturingSettings)})
