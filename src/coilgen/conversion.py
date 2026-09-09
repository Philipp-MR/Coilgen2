from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .config import CoilGenConfig
from .geometry import combine_meshes
from .object_registry import register_in_memory_mesh, register_in_memory_target_field

if TYPE_CHECKING:
    from .problem import CoilDesignProblem, TargetFieldSpec


def coil_design_problem_to_config(
    problem: CoilDesignProblem,
    *,
    base_config: CoilGenConfig | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    include_support_suppression_meshes: bool = True,
) -> CoilGenConfig:
    """Convert a high-level ``CoilDesignProblem`` to the current pipeline config.

    The current solver pipeline consumes ``CoilGenConfig``. This function is the
    explicit bridge from the object model to that config. Meshes and target
    fields are registered in memory and referenced by stable same-process keys;
    no temporary STL/CSV files are written.
    """

    problem.validate()
    target_spec = (
        _target_with_support_suppression(problem) if include_support_suppression_meshes else problem.target_field
    )
    target_key = register_in_memory_target_field(target_spec.to_target_field(), key=f"{problem.metadata.name}:target")

    config = base_config if base_config is not None else CoilGenConfig()
    values = asdict(config)
    values.update(
        {
            "field_shape_function": "none",
            "target_gradient_strength": 1.0,
            "target_region_radius": _target_radius(target_spec),
            "target_region_resolution": max(1, int(round(target_spec.dsv_point_count ** (1.0 / 3.0)))),
            "target_field_definition_file": target_key,
            "target_field_definition_field_name": "none",
            "target_field_component": target_spec.component.to_config_value(),
            "b_0_direction": tuple(target_spec.component.b0_direction),
            "problem_schema_version": problem.problem_schema_version,
            "secondary_target_mesh_file": "none",
            "secondary_target_weight": 1.0,
            "surface_is_cylinder_flag": bool(problem.support.metadata.get("surface_is_cylinder", False)),
            "levels": int(problem.layout.contour_count),
            "min_turn_distance": max(
                float(problem.conductor.min_turn_spacing_m),
                float(problem.optimization.min_turn_spacing_m),
            ),
            "conductor_cross_section_width": float(problem.conductor.width_m),
            "conductor_cross_section_height": float(problem.conductor.height_m),
            "sweep_insulation_thickness": float(problem.conductor.insulation_thickness_m),
            "specific_conductivity_conductor": float(problem.conductor.resistivity_ohm_m),
            "sf_opt_method": str(problem.optimization.method),
            "stream_function_cost_function": str(problem.optimization.cost_function),
            "regularization_selection_method": str(problem.optimization.regularization_selection),
            "target_error_tolerance_percent": _default_if_none(
                problem.optimization.target_error_tolerance_percent,
                config.target_error_tolerance_percent,
            ),
            "symmetry_reduction": "strict" if problem.optimization.use_symmetry_reduction else "off",
            "inter_group_connection_path_method": str(problem.layout.inter_group_connection_path),
            "within_group_cut_candidate_count": int(problem.layout.within_group_cut_candidate_count),
            "skip_postprocessing": not bool(problem.layout.postprocess_connections),
            "skip_normal_shift": not bool(problem.layout.normal_shift_returns),
            "min_inter_group_connector_clearance": float(problem.layout.connector_clearance_m),
            "sweep_min_bend_radius": float(problem.layout.min_bend_radius_m),
            "layout_terminal_strategy": str(problem.layout.terminal_strategy),
            "layout_terminal_pad_count": int(problem.layout.terminal_pad_count),
            "layout_terminal_exit_count": int(problem.layout.terminal_exit_count),
            "layout_keepout_region_count": int(problem.layout.keepout_region_count),
            "layout_manufacturing_intent": str(problem.layout.manufacturing_intent),
            "skip_inductance_calculation": not bool(problem.evaluation.evaluate_inductance),
            "skip_sweep": not bool(problem.evaluation.evaluate_layout),
        }
    )
    values.update(_support_geometry_config(problem))
    if problem.optimization.regularization_factor is not None:
        values["tikonov_reg_factor"] = float(problem.optimization.regularization_factor)
    if problem.conductor.effective_sheet_thickness_m is not None:
        values["conductor_thickness"] = float(problem.conductor.effective_sheet_thickness_m)
    optimization_fields = {
        "constrained_objective_mode": "constrained_quadratic_objective_mode",
        "conic_solver": "constrained_optimization_solver",
        "conic_tolerance": "constrained_optimization_tolerance",
        "conic_max_iterations": "constrained_optimization_max_iterations",
        "conic_fallback_policy": "constrained_quadratic_fallback_policy",
        "target_error_mode": "target_error_constraint_mode",
        "target_error_metric": "target_error_constraint_metric",
        "target_error_tolerance_absolute_T": "target_error_tolerance_absolute_T",
        "target_error_constraint_acceptance_absolute_T": "target_error_constraint_acceptance_absolute_T",
        "target_error_constraint_acceptance_tolerance": "target_error_constraint_acceptance_tolerance",
        "target_error_rms_limit_T": "target_error_rms_limit_T",
        "target_error_std_limit_T": "target_error_std_limit_T",
        "suppression_rms_limit_T": "suppression_field_rms_limit_T",
        "suppression_peak_limit_T": "suppression_field_peak_limit_T",
        "surface_current_density_limit_A_per_m": "surface_current_density_limit_A_per_m",
        "smoothness_factor": "stream_function_smoothness_factor",
        "basis": "stream_function_basis",
        "spectral_cutoff_length_m": "stream_function_spectral_cutoff_length_m",
        "spectral_cutoff_policy": "stream_function_spectral_cutoff_policy",
        "spectral_min_modes": "stream_function_spectral_min_modes",
        "spectral_max_modes": "stream_function_spectral_max_modes",
    }
    for spec_field, config_field in optimization_fields.items():
        value = getattr(problem.optimization, spec_field)
        if value is not None:
            values[config_field] = value
    values.update(_metadata_config_overrides(problem))
    if config_overrides:
        values.update(dict(config_overrides))
    return CoilGenConfig(**values)


def _default_if_none(value: Any, default: Any) -> Any:
    return default if value is None else value


def _metadata_config_overrides(problem: CoilDesignProblem) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for mapping in (
        problem.metadata.attributes.get("config_overrides", {}),
        problem.support.metadata.get("config_overrides", {}),
        problem.target_field.metadata.get("config_overrides", {}),
        problem.optimization.metadata.get("config_overrides", {}),
        problem.layout.metadata.get("config_overrides", {}),
        problem.evaluation.metadata.get("config_overrides", {}),
    ):
        if mapping:
            overrides.update(dict(mapping))
    return overrides


def _coil_support_meshes(problem: CoilDesignProblem):
    primaries = sorted(problem.support.primary_meshes, key=lambda mesh: mesh.part_index)
    shields = sorted(problem.support.shield_meshes, key=lambda mesh: mesh.part_index)
    auxiliaries = sorted(problem.support.by_role("auxiliary"), key=lambda mesh: mesh.part_index)
    return tuple(primaries + shields + auxiliaries)


def _support_geometry_config(problem: CoilDesignProblem) -> dict[str, Any]:
    if problem.support.pipeline_geometry is not None:
        return problem.support.pipeline_geometry.to_config_overrides()

    coil_meshes = [support_mesh.mesh for support_mesh in _coil_support_meshes(problem)]
    if not coil_meshes:
        raise ValueError("Cannot convert problem without primary or shield support meshes.")
    mesh_key = register_in_memory_mesh(combine_meshes(coil_meshes), key=f"{problem.metadata.name}:coil_mesh")
    return {
        "coil_mesh_file": mesh_key,
        "surface_is_cylinder_flag": bool(problem.support.metadata.get("surface_is_cylinder", False)),
    }


def _target_with_support_suppression(problem: CoilDesignProblem) -> TargetFieldSpec:
    from .problem import SuppressionRegion

    target = problem.target_field
    next_group = int(np.max(target.target_group_indices())) + 1 if target.point_count else 2
    for support_mesh in sorted(problem.support.suppression_meshes, key=lambda mesh: mesh.part_index):
        weight = float(support_mesh.metadata.get("weight", 1.0))
        target = target.with_suppression_region(
            SuppressionRegion(
                name=support_mesh.name or f"suppression_part_{support_mesh.part_index}",
                coords=support_mesh.mesh.vertices,
                target_b=np.zeros((support_mesh.mesh.n_vertices, 3), dtype=float),
                weight=weight,
                group_index=next_group,
                source=str(support_mesh.metadata.get("source", "support_mesh")),
                metadata=dict(support_mesh.metadata),
            )
        )
        next_group += 1
    return target


def _target_radius(target: TargetFieldSpec) -> float:
    if target.dsv_point_count == 0:
        return 1.0
    radius = float(np.max(np.linalg.norm(target.dsv_coords - np.mean(target.dsv_coords, axis=0), axis=1)))
    return max(radius, 1e-9)
