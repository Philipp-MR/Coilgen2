from __future__ import annotations

from typing import Any, TypedDict


class RegularizationDiagnostics(TypedDict, total=False):
    selection_method: str
    candidate_solver: str
    fallback: bool
    fallback_reason: str
    factor_scale: float
    selected_index: int
    selected_factor: float
    selected_lambda: float
    candidate_factors: list[float]
    candidate_lambdas: list[float]
    residual_norms: list[float]
    regularization_norms: list[float]
    curvature: list[float]
    l_curve_knee_type: str
    l_curve_residual_log_span: float
    l_curve_regularization_log_span: float
    l_curve_selected_residual_position: float
    l_curve_selected_regularization_position: float
    stream_function_scheme: str
    stream_function_cost_function: str
    cost_function_formula: str
    unknown_count: int
    target_sample_count: int
    turn_spacing_penalty_enabled: bool
    min_turn_distance: float
    turn_spacing_reference_length: float
    turn_spacing_regularization_factor: float
    turn_spacing_attempts: int
    turn_spacing_threshold_met: bool
    estimated_min_turn_distance: float
    measured_min_contour_distance: float
    turn_spacing_measured_threshold_met: bool
    turn_spacing_penalty_weights: list[float]
    estimated_min_turn_distances: list[float]


class RouteDiagnostics(TypedDict, total=False):
    point_count: int
    segment_count: int
    length: float
    min_segment_length: float
    closed_v: bool
    self_contact_count: int
    shifted_point_count: int
    max_normal_shift_factor: float
    min_wire_wire_distance: float
    min_wire_wire_segment_a: int
    min_wire_wire_segment_b: int
    wire_clearance_failure_count: int
    wire_clearance_failure_threshold: float
    wire_clearance_warning_count: int
    wire_clearance_warning_threshold: float
    opening_cut_count: int
    degenerate_opening_cut_count: int
    inter_group_cut_count: int
    inter_group_geodesic_connector_count: int
    inter_group_direct_connector_count: int
    inter_group_direct_fallback_count: int
    inter_group_clearance_fallback_count: int
    route_quality_ok: bool
    route_quality_failure_count: int
    route_quality_failure_reasons: str
    route_quality_warning_count: int
    route_quality_status: str
    route_topology_ok: bool
    route_topology_failure_count: int


class ManufacturingDiagnostics(TypedDict, total=False):
    manufacturing_ok: bool
    manufacturing_failure_reasons: str
    route_design_quality_status: str
    route_design_target_failure: bool
    manufacturing_route_topology_failure: bool
    manufacturing_route_topology_failure_reasons: str
    mesh_solid_ok: bool
    sweep_geometry_backend: str
    sweep_geometry_role: str
    analytic_solid_created: bool
    open_edge_count: int
    nonmanifold_edge_count: int
    misoriented_edge_count: int
    short_edge_count: int
    sweep_degenerate_ring_points_removed: int
    sweep_unrepaired_short_lateral_edge_count: int
    sweep_min_lateral_edge_length_m: float
    sweep_ring_edge_tolerance_m: float
    signed_volume_m3: float
    positive_volume: bool
    self_intersection_count: int
    min_clearance: float
    conductor_width: float
    conductor_height: float


class TopologyDiagnostics(TypedDict, total=False):
    method: str
    warnings: list[str]
    errors: list[str]
    non_separating_loops: list[int]
    face_component_count: int
    boundary_loop_count: int


class DiagnosticCache(TypedDict, total=False):
    field_maps: Any
    gradient_maps: Any
    relative_error_maps: Any
