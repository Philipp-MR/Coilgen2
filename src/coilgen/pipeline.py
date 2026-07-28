from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .basis import calculate_basis_functions, calculate_one_ring_by_mesh
from .config import CoilGenConfig
from .config_validation import validate_config
from .contourability import validate_and_repair_contourability
from .contour_angles import analyze_contour_angle_stability
from .contour_sparsification import sparsify_contours
from .contours import (
    calc_potential_levels,
    extract_contours,
    filter_loop_significance,
    regularize_contour_paths,
)
from .design_context import EvaluationFeature, normalize_design_context
from .evaluation import calculate_layout_gradient, evaluate_field_errors, layout_target_residual_diagnostics
from .geometry import (
    load_mesh_from_config,
    load_optional_target_mesh,
    load_secondary_target_mesh,
    normalize_mesh,
    refine_mesh,
    split_disconnected_mesh,
)
from .gradient_slice_maps import evaluate_gradient_slice_maps, gradient_slice_maps_enabled
from .models import CoilGenResult, CoilPart, Mesh
from .routing import (
    calculate_unconnected_contour_wire_distances,
    calculate_group_centers,
    cleanup_wire_paths,
    find_minimal_contour_distance,
    interconnect_among_groups,
    interconnect_within_groups,
    shift_return_paths,
    topological_loop_grouping,
)
from .inductance import calculate_inductance_by_coil_layout
from .manufacturing import create_swept_coil_meshes
from .pcb import generate_cylindrical_pcb_print
from .optimization import (
    enforce_stream_function_current_density_limit,
    stream_function_current_density_diagnostics,
    stream_function_optimization,
)
from .physics import calculate_resistance_matrix, calculate_sensitivity_matrix
from .preoptimization import apply_preoptimized_stream_function, has_preoptimized_stream_function
from .progress import ProgressReporter
from .symmetry import mirror_symmetric_mesh_from_half
from .surface_topology import mesh_quality_report
from .target import define_target_field


class CoilGenPipeline:
    """Orchestrates the Python CoilGen workflow."""

    def __init__(self, config: CoilGenConfig, project_root: Path | None = None):
        self.config = config
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()

    def run(self) -> CoilGenResult:
        validate_config(self.config, project_root=self.project_root).raise_for_errors()
        progress = ProgressReporter.from_config(self.config)
        stage_index = 0
        stage_count = 22
        start_time = time.perf_counter()
        last_stage_time = start_time
        timing_summary: list[dict[str, float | str]] = [
            {"stage": "start", "stage_seconds": 0.0, "elapsed_seconds": 0.0}
        ]

        def record_timing(message: str) -> None:
            nonlocal last_stage_time
            now = time.perf_counter()
            timing_summary.append(
                {
                    "stage": message,
                    "stage_seconds": float(now - last_stage_time),
                    "elapsed_seconds": float(now - start_time),
                }
            )
            last_stage_time = now

        def stage(message: str) -> None:
            nonlocal stage_index
            stage_index += 1
            record_timing(message)
            progress.report(100.0 * stage_index / stage_count, message, force=True)

        progress.report(0.0, "start", force=True)
        coil_mesh = load_mesh_from_config(self.config, self.project_root)
        coil_mesh, symmetry_mesh_diagnostics = _prepare_half_mesh_symmetry_if_requested(coil_mesh, self.config)
        target_mesh = load_optional_target_mesh(self.config.target_mesh_file, self.config, self.project_root)
        stage("loaded meshes")

        prepared_meshes = [
            refine_mesh(mesh, self.config.iteration_num_mesh_refinement) for mesh in split_disconnected_mesh(coil_mesh)
        ]
        prepared_meshes, refinement_symmetry_diagnostics = _restore_refined_single_mesh_symmetry(
            prepared_meshes,
            self.config,
        )
        symmetry_mesh_diagnostics.update(refinement_symmetry_diagnostics)
        mesh_quality_diagnostics = _validate_prepared_mesh_quality(prepared_meshes, self.config)
        symmetry_mesh_diagnostics.update(mesh_quality_diagnostics)
        parts = [CoilPart(mesh=mesh) for mesh in prepared_meshes]
        secondary_target_mesh = load_secondary_target_mesh(
            self.config,
            self.project_root,
            coil_mesh=coil_mesh,
            coil_meshes=prepared_meshes,
        )
        suppression_surface_diagnostics = _suppression_surface_diagnostics(secondary_target_mesh)
        stage("prepared coil meshes")
        target_field = define_target_field(
            [part.mesh for part in parts], target_mesh, secondary_target_mesh, self.config
        )
        stage("defined target field")

        parts = calculate_one_ring_by_mesh(parts)
        stage("built one-ring mesh neighborhoods")
        parts = calculate_basis_functions(parts)
        stage("built stream-function basis")
        parts = calculate_sensitivity_matrix(parts, target_field, self.config)
        sensitivity_diagnostics = _sensitivity_matrix_diagnostics(parts, target_field)
        stage("calculated sensitivity matrix")
        if has_preoptimized_stream_function(self.config):
            parts, combined_mesh, sf_b_field, regularization_diagnostics = apply_preoptimized_stream_function(
                parts,
                target_field,
                self.config,
                self.project_root,
            )
            parts, combined_mesh, sf_b_field, density_limit_diagnostics = enforce_stream_function_current_density_limit(
                parts,
                combined_mesh,
                sf_b_field,
                self.config,
            )
            regularization_diagnostics.update(density_limit_diagnostics)
            stage("loaded preoptimized stream function")
        else:
            parts = calculate_resistance_matrix(parts, self.config)
            stage("calculated resistance matrix")
            parts, combined_mesh, sf_b_field, regularization_diagnostics = stream_function_optimization(
                parts, target_field, self.config
            )
            stage("optimized stream function")
        regularization_diagnostics.update(stream_function_current_density_diagnostics(parts, self.config))
        regularization_diagnostics.update(dict(getattr(target_field, "target_diagnostics", {})))
        _update_gradient_hardware_target_diagnostics(regularization_diagnostics)
        regularization_diagnostics.update(sensitivity_diagnostics)
        regularization_diagnostics.update(suppression_surface_diagnostics)
        regularization_diagnostics.update(symmetry_mesh_diagnostics)
        parts, primary_surface_ind = calc_potential_levels(parts, combined_mesh, self.config)
        if 0 <= primary_surface_ind < len(parts):
            regularization_diagnostics.update(dict(parts[primary_surface_ind].contour_level_diagnostics))
        stage("calculated potential levels")
        parts = extract_contours(parts, self.config)
        if 0 <= primary_surface_ind < len(parts):
            regularization_diagnostics.update(dict(parts[primary_surface_ind].contour_level_diagnostics))
        stage("extracted contours")
        parts = filter_loop_significance(parts, target_field, self.config)
        stage("filtered loop significance")
        parts = regularize_contour_paths(parts, self.config)
        if 0 <= primary_surface_ind < len(parts):
            regularization_diagnostics.update(dict(parts[primary_surface_ind].contour_level_diagnostics))
        stage("regularized contour paths")
        parts, contourability_diagnostics = validate_and_repair_contourability(
            parts,
            combined_mesh,
            target_field,
            sf_b_field,
            self.config,
        )
        if 0 <= primary_surface_ind < len(parts):
            regularization_diagnostics.update(dict(parts[primary_surface_ind].contour_level_diagnostics))
        regularization_diagnostics.update(contourability_diagnostics)
        stage("validated contourability")
        parts = sparsify_contours(
            parts,
            target_field,
            self.config,
            progress=ProgressReporter.from_config(self.config, label="Contour sparsification"),
        )
        parts = analyze_contour_angle_stability(parts)
        if 0 <= primary_surface_ind < len(parts):
            regularization_diagnostics.update(dict(parts[primary_surface_ind].contour_level_diagnostics))
        stage("processed contour sparsification")
        if self.config.skip_calculation_min_winding_distance:
            regularization_diagnostics["contour_distance_diagnostics_skipped"] = True
        else:
            parts = find_minimal_contour_distance(
                parts,
                self.config,
                progress=ProgressReporter.from_config(self.config, label="Contour distance"),
            )
            parts = calculate_unconnected_contour_wire_distances(
                parts,
                self.config,
                progress=ProgressReporter.from_config(self.config, label="Unconnected wire distance"),
            )
            min_turn_distance = float(getattr(self.config, "min_turn_distance", 0.0))
            if min_turn_distance > 0.0:
                measured_distances = [
                    float(part.minimal_contour_distance)
                    for part in parts
                    if len(part.contour_lines) > 1 and np.isfinite(part.minimal_contour_distance)
                ]
                measured_min_distance = min(measured_distances) if measured_distances else 0.0
                regularization_diagnostics["measured_min_contour_distance"] = measured_min_distance
                regularization_diagnostics["turn_spacing_measured_threshold_met"] = (
                    measured_min_distance >= min_turn_distance
                )
        stage("calculated contour distances")
        skip_postprocessing = self.config.skip_postprocessing or self.config.contour_only_sweep_mode
        if skip_postprocessing:
            for part in parts:
                part.wire_path = None
                part.connected_groups = []
                part.opening_cuts_among_groups = []
            stage("skipped topology/routing postprocessing")
        else:
            parts = topological_loop_grouping(
                parts,
                config=self.config,
                progress=ProgressReporter.from_config(self.config, label="Topology"),
            )
            stage("grouped contour topology")
            parts = calculate_group_centers(parts)
            parts = interconnect_within_groups(
                parts,
                self.config,
                progress=ProgressReporter.from_config(self.config, label="Within-group routing"),
            )
            stage("interconnected within loop groups")
            parts = interconnect_among_groups(
                parts,
                self.config,
                progress=ProgressReporter.from_config(self.config, label="Inter-group routing"),
            )
            stage("interconnected among loop groups")
            parts = shift_return_paths(parts, self.config)
            stage("shifted return paths")
        parts = cleanup_wire_paths(
            parts,
            config=self.config,
            progress=ProgressReporter.from_config(self.config, label="Route cleanup"),
        )
        stage("cleaned wire paths")
        parts = generate_cylindrical_pcb_print(parts, self.config)
        parts = create_swept_coil_meshes(parts, self.config)
        stage("generated swept visualization meshes")
        parts, inductance_matrix, resistance, coil_lengths, coil_cross_sections = calculate_inductance_by_coil_layout(
            parts, self.config
        )
        stage("calculated inductance and resistance")

        (
            field_by_layout,
            field_by_unconnected_loops,
            field_layout_per_amp,
            field_loops_per_amp,
            error_vals,
            opt_current_layout,
            sf_b_field_1a,
            target_field_1a,
        ) = evaluate_field_errors(parts, self.config, target_field, sf_b_field)
        regularization_diagnostics.update(
            layout_target_residual_diagnostics(field_by_layout, target_field, self.config)
        )
        design_context = normalize_design_context(getattr(self.config, "design_context", "mr_gradient"))
        layout_gradient = (
            calculate_layout_gradient(field_layout_per_amp, target_field)
            if design_context.enabled(EvaluationFeature.GRADIENT_FIT)
            else None
        )
        _apply_result_memory_policy(parts, self.config)
        record_timing("evaluated field errors")
        progress.report(100.0, "evaluated field errors", force=True)
        progress.finish()

        potential_step = parts[primary_surface_ind].contour_step or np.nan
        result = CoilGenResult(
            coil_parts=parts,
            num_levels=int(len(parts[primary_surface_ind].potential_level_list)),
            potential_step=float(potential_step),
            primary_surface=primary_surface_ind,
            target_field=target_field,
            field_by_layout=field_by_layout,
            field_by_unconnected_loops=field_by_unconnected_loops,
            field_layout_per1Amp=field_layout_per_amp,
            field_loops_per1Amp=field_loops_per_amp,
            needed_current_layout=opt_current_layout,
            b_field_opt_sf=sf_b_field,
            b_field_opt_sf_1A=sf_b_field_1a,
            regularization_factor=float(regularization_diagnostics.get("selected_factor", np.nan)),
            regularization_lambda=float(regularization_diagnostics.get("selected_lambda", np.nan)),
            regularization_diagnostics=regularization_diagnostics,
            target_field_1A=target_field_1a,
            error_vals=error_vals,
            input_data=self.config,
            layout_gradient=layout_gradient,
            combined_mesh=combined_mesh,
            inductance_matrix=inductance_matrix,
            resistance=resistance,
            coil_lengths=coil_lengths,
            coil_cross_sections=coil_cross_sections,
            timing_summary=timing_summary,
        )
        if gradient_slice_maps_enabled(self.config):
            slice_start = time.perf_counter()
            slice_maps = evaluate_gradient_slice_maps(result)
            result.diagnostic_cache["gradient_slice_maps"] = slice_maps
            result.regularization_diagnostics.update(dict(slice_maps.summary))
            result.timing_summary.append(
                {
                    "stage": "evaluated gradient slice maps",
                    "stage_seconds": float(time.perf_counter() - slice_start),
                    "elapsed_seconds": float(time.perf_counter() - start_time),
                }
            )
        return result


def _restore_refined_single_mesh_symmetry(
    meshes: list[Mesh],
    config: CoilGenConfig,
) -> tuple[list[Mesh], dict[str, object]]:
    """Rebuild a subdivided single mesh from one half to preserve FEM parity.

    Independent midpoint insertion keeps mirrored coordinates exact but can
    change the mirrored face pairing used by the nodal current basis. Rebuilding
    the refined support from one half restores the required axial-field parity.
    """

    iterations = max(int(getattr(config, "iteration_num_mesh_refinement", 0)), 0)
    mode = str(getattr(config, "symmetry_reduction", "off")).strip().lower()
    source = str(getattr(config, "symmetry_reduction_source", "generated_parts")).strip().lower()
    if iterations == 0 or mode != "strict" or source != "single_mesh" or len(meshes) != 1:
        return meshes, {"refinement_symmetry_reconstruction_enabled": False}

    clip_tolerance = max(float(getattr(config, "symmetry_reduction_clip_tolerance", 1.0e-9)), 1.0e-12)
    snap_tolerance = float(getattr(config, "symmetry_reduction_plane_snap_tolerance", 0.0))
    if not np.isfinite(snap_tolerance) or snap_tolerance <= 0.0:
        snap_tolerance = clip_tolerance
    merge_tolerance = max(min(0.01 * clip_tolerance, 0.01 * snap_tolerance), 1.0e-12)
    rebuilt, summary = mirror_symmetric_mesh_from_half(
        meshes[0],
        plane_point=np.asarray(config.symmetry_reduction_mirror_plane_point, dtype=float),
        plane_normal=np.asarray(config.symmetry_reduction_mirror_plane_normal, dtype=float),
        half_space=str(getattr(config, "symmetry_reduction_half_space", "positive")),
        clip_tolerance=clip_tolerance,
        merge_tolerance=merge_tolerance,
        plane_snap_tolerance=max(snap_tolerance, clip_tolerance),
    )
    rebuilt = normalize_mesh(rebuilt)
    diagnostics: dict[str, object] = {
        "refinement_symmetry_reconstruction_enabled": True,
        "refinement_symmetry_reconstruction_iterations": iterations,
        "refinement_symmetry_reconstruction_vertex_count": int(rebuilt.n_vertices),
        "refinement_symmetry_reconstruction_face_count": int(rebuilt.n_faces),
    }
    diagnostics.update({f"refinement_{key}": value for key, value in summary.items()})
    return [rebuilt], diagnostics


def run_coilgen(config: CoilGenConfig, project_root: Path | None = None) -> CoilGenResult:
    return CoilGenPipeline(config, project_root=project_root).run()


def _sensitivity_matrix_diagnostics(parts: list[CoilPart], target_field) -> dict[str, float | int | str]:
    matrices = [part.sensitivity_matrix for part in parts if part.sensitivity_matrix is not None]
    vertex_count = int(sum(part.mesh.n_vertices for part in parts))
    target_count = int(len(target_field.coords))
    byte_count = int(sum(matrix.nbytes for matrix in matrices))
    unknown_count = int(sum(matrix.shape[2] for matrix in matrices if matrix.ndim == 3))
    shapes = "; ".join("x".join(str(dim) for dim in matrix.shape) for matrix in matrices)
    return {
        "sensitivity_target_point_count": target_count,
        "sensitivity_mesh_vertex_count": vertex_count,
        "sensitivity_unknown_count": unknown_count,
        "sensitivity_matrix_part_count": len(matrices),
        "sensitivity_matrix_bytes": byte_count,
        "sensitivity_matrix_MiB": float(byte_count / (1024.0 * 1024.0)),
        "sensitivity_matrix_shapes": shapes,
    }


def _update_gradient_hardware_target_diagnostics(diagnostics: dict[str, object]) -> None:
    try:
        target_gradient = float(
            diagnostics.get(
                "target_gradient_effective_T_per_m",
                diagnostics.get("target_gradient_requested_T_per_m", float("nan")),
            )
        )
        scale = float(
            diagnostics.get(
                "gradient_hardware_scale_factor",
                diagnostics.get("surface_current_density_scaling_factor", 1.0),
            )
        )
    except (TypeError, ValueError):
        return
    if not np.isfinite(target_gradient) or not np.isfinite(scale):
        return
    diagnostics["target_gradient_requested_after_normalization_T_per_m"] = target_gradient
    diagnostics["target_gradient_hardware_effective_T_per_m"] = target_gradient * scale


def _suppression_surface_diagnostics(mesh) -> dict[str, float | int | str | bool]:
    if mesh is None:
        return {
            "suppression_clearance_status": "not_applicable",
            "suppression_clearance_requested_m": float("nan"),
            "suppression_clearance_min_allowed_m": float("nan"),
            "suppression_clearance_min_m": float("nan"),
            "suppression_clearance_p05_m": float("nan"),
            "suppression_clearance_median_m": float("nan"),
            "suppression_clearance_violation_count": 0,
            "suppression_clearance_dropped_face_count": 0,
        }
    report = dict(getattr(mesh, "metadata", {}).get("suppression_clearance_report", {}) or {})
    if not report:
        return {
            "suppression_clearance_status": "not_checked",
            "suppression_clearance_requested_m": float("nan"),
            "suppression_clearance_min_allowed_m": float("nan"),
            "suppression_clearance_min_m": float("nan"),
            "suppression_clearance_p05_m": float("nan"),
            "suppression_clearance_median_m": float("nan"),
            "suppression_clearance_violation_count": 0,
            "suppression_clearance_dropped_face_count": 0,
        }
    if "suppression_clearance_violating_output_vertex_count" in report:
        report["suppression_clearance_violation_count"] = report["suppression_clearance_violating_output_vertex_count"]
    return report


def _prepare_half_mesh_symmetry_if_requested(coil_mesh, config: CoilGenConfig):
    source = str(getattr(config, "symmetry_reduction_source", "generated_parts")).strip().lower().replace("-", "_")
    mode = str(getattr(config, "symmetry_reduction", "auto")).strip().lower()
    if source not in {"half_mesh", "half"} or mode == "off":
        return coil_mesh, {}
    prepared, summary = mirror_symmetric_mesh_from_half(
        coil_mesh,
        getattr(config, "symmetry_reduction_mirror_plane_point", (0.0, 0.0, 0.0)),
        getattr(config, "symmetry_reduction_mirror_plane_normal", (1.0, 0.0, 0.0)),
        half_space=getattr(config, "symmetry_reduction_half_space", "positive"),
        clip_tolerance=float(getattr(config, "symmetry_reduction_clip_tolerance", 1e-9)),
        plane_snap_tolerance=float(getattr(config, "symmetry_reduction_plane_snap_tolerance", 0.0)),
    )
    diagnostics = {f"symmetry_half_mesh_{key}": value for key, value in summary.items()}
    diagnostics["symmetry_half_mesh_active"] = True
    return prepared, diagnostics


def _validate_prepared_mesh_quality(meshes, config: CoilGenConfig) -> dict[str, object]:
    policy = str(getattr(config, "mesh_quality_policy", "warn")).strip().lower().replace("-", "_")
    if policy == "off":
        return {"mesh_quality_policy": "off", "mesh_quality_checked": False}
    diagnostics: dict[str, object] = {
        "mesh_quality_policy": policy,
        "mesh_quality_checked": True,
        "mesh_quality_part_count": int(len(meshes)),
        "mesh_quality_warning_count": 0,
        "mesh_quality_error_count": 0,
    }
    messages: list[str] = []
    for part_index, mesh in enumerate(meshes, start=1):
        report = mesh_quality_report(
            mesh,
            max_triangle_edge_ratio=float(getattr(config, "mesh_quality_max_triangle_edge_ratio", 25.0)),
            max_altitude_aspect_ratio=float(getattr(config, "mesh_quality_max_altitude_aspect_ratio", 50.0)),
            max_edge_length_ratio=float(getattr(config, "mesh_quality_max_edge_length_ratio", 100.0)),
        )
        diagnostics[f"mesh_quality_part{part_index}"] = report.as_dict()
        diagnostics["mesh_quality_warning_count"] = int(diagnostics["mesh_quality_warning_count"]) + len(
            report.warnings
        )
        diagnostics["mesh_quality_error_count"] = int(diagnostics["mesh_quality_error_count"]) + len(report.errors)
        messages.extend(f"part {part_index}: {message}" for message in (*report.errors, *report.warnings))
    diagnostics["mesh_quality_messages"] = messages
    if messages and policy == "strict":
        raise ValueError("Prepared current-surface mesh failed strict quality checks: " + "; ".join(messages))
    if messages and bool(getattr(config, "progress_flag", False)):
        for message in messages:
            print(f"Mesh quality warning: {message}")
    return diagnostics


def _apply_result_memory_policy(parts: list[CoilPart], config: CoilGenConfig) -> None:
    if bool(config.retain_heavy_fields) and not bool(config.low_memory_mode):
        return
    for part in parts:
        part.sensitivity_matrix = None
        part.resistance_matrix = None
        part.basis_elements = []
        part.current_density_mat = None
        part.contour_distance_matrix = None
        part.loop_in_loop_matrix = None
