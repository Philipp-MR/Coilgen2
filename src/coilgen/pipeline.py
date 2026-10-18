from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from .basis import calculate_basis_functions, calculate_one_ring_by_mesh
from .config import CoilGenConfig
from .config_validation import validate_config
from .contourability import validate_and_repair_contourability
from .contour_angles import analyze_contour_angle_stability
from .contour_sparsification import (
    consolidate_pruned_contours,
    prune_sparsified_contours,
    sparsify_contours,
)
from .contours import (
    calc_potential_levels,
    extract_contours,
    filter_loop_significance,
    regularize_contour_paths,
)
from .design_context import EvaluationFeature, normalize_design_context
from .evaluation import (
    calculate_layout_gradient,
    evaluate_field_errors,
    final_field_diagnostics,
    layout_target_residual_diagnostics,
)
from .geometry import (
    load_mesh_from_config,
    load_optional_target_mesh,
    load_secondary_target_mesh,
    normalize_mesh,
    refine_mesh,
    split_disconnected_mesh,
)
from .gradient_slice_maps import gradient_slice_map_result, gradient_slice_maps_enabled
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
from .manufacturing import create_swept_coil_meshes, realize_coil_paths
from .joint_group_adoption import improve_group_routes, refresh_retained_route_diagnostics
from .routing_assembly import assemble_routing_parts
from .routing_jumps import fabricated_jump_metrics
from .pcb import generate_cylindrical_pcb_print
from .optimization import (
    enforce_stream_function_current_density_limit,
    stream_function_current_density_diagnostics,
    stream_function_optimization,
)
from .physics import calculate_resistance_matrix, calculate_sensitivity_matrix
from .preoptimization import apply_preoptimized_stream_function, has_preoptimized_stream_function
from .progress import ProgressReporter, progress_scope
from .pipeline_stages import PipelineCheckpoint, PipelineStage, PipelineState, stage_fingerprints
from .pipeline_failure import retain_pipeline_failure
from .symmetry import mirror_symmetric_mesh_from_half
from .surface_topology import mesh_quality_report
from .target import define_target_field


class _ExecutionContext:
    """Per-execution reporting; checkpoint times never masquerade as rerun work."""

    def __init__(self, config: CoilGenConfig):
        self.progress = ProgressReporter.from_config(config)
        self.realize_for_routing = (
            config.joint_group_routing and not config.skip_postprocessing and not config.make_cylndrical_pcb
        )
        self.start_time = time.perf_counter()
        self.last_stage_time = self.start_time
        self.failed_operation = None
        self.timing_summary: list[dict[str, float | str]] = [
            {"stage": "start", "stage_seconds": 0.0, "elapsed_seconds": 0.0}
        ]

    def record_timing(self, message: str) -> None:
        now = time.perf_counter()
        self.timing_summary.append(
            {
                "stage": message,
                "stage_seconds": float(now - self.last_stage_time),
                "elapsed_seconds": float(now - self.start_time),
            }
        )
        self.last_stage_time = now

    def stage(self, message: str) -> None:
        self.record_timing(message)
        # The stages have very different and data-dependent costs. Completion
        # counts are not an elapsed-work percentage, especially after resume.
        self.progress.report(None, message, force=True)

    def operations(self, weights):
        """Allocate a stage's work; completed inner units supply its updates."""
        total = sum(value for _, value in weights)
        position = 0.0
        self._operation_ranges = {}
        for name, value in weights:
            end = position + 100 * value / total
            self._operation_ranges[name] = (position, end)
            position = end

    def compute(self, name, operation, *args, **kwargs):
        start, end = self._operation_ranges[name]
        with progress_scope(start, end, name, label="CoilGen", operation=name):
            try:
                return operation(*args, **kwargs)
            except Exception:
                self.failed_operation = name
                raise


class CoilGenPipeline:
    """One numerical workflow for legacy scripts and restartable applications.

    Configuration is owned by this instance; changes to a caller's config never
    change an in-progress run. Stage execution copies its checkpoint before any
    mutable numerical operations, so one solved sheet can feed multiple studies.
    """

    def __init__(self, config: CoilGenConfig, project_root: Path | None = None, *, requirements=None):
        self.config = deepcopy(config)
        self.project_root = (Path(project_root) if project_root is not None else Path.cwd()).resolve()
        if requirements is not None:
            from .application.research_requirements import ResearchRequirements

            requirements = ResearchRequirements.from_dict(requirements)
        self.requirements = requirements

    def stage_fingerprints(self):
        if self.requirements is None:
            return stage_fingerprints(self.config, self.project_root)
        from .application.research_requirements import requirements_stage_fingerprints

        return requirements_stage_fingerprints(self.config, self.project_root, self.requirements)

    def run(self) -> CoilGenResult:
        state, _ = self._execute(PipelineStage.EVALUATED)
        assert state.result is not None
        return state.result

    def run_stage(
        self, stage: PipelineStage | str, checkpoint: PipelineCheckpoint | None = None, *, checkpoint_callback=None
    ) -> PipelineCheckpoint:
        """Execute through a selected boundary, optionally reusing completed work.

        A checkpoint cannot be rewound or reused after upstream settings, asset
        contents or numerical source change. Recompute those stages explicitly.
        """
        selected = PipelineStage(stage)
        fingerprints = self.stage_fingerprints()

        def completed(boundary, state):
            if checkpoint_callback is not None:
                checkpoint_callback(
                    PipelineCheckpoint(boundary, state, self.config, self.project_root, fingerprints=fingerprints)
                )

        state, _ = self._execute(selected, checkpoint, completed if checkpoint_callback is not None else None)
        return PipelineCheckpoint(selected, state, self.config, self.project_root, fingerprints=fingerprints)

    def _execute(self, selected: PipelineStage, checkpoint: PipelineCheckpoint | None = None, completed=None):
        validation_config = self.config
        if self.requirements is not None:
            from .application.research_requirements import requirements_validation_config

            validation_config = requirements_validation_config(self.config)
        validate_config(validation_config, project_root=self.project_root).raise_for_errors()
        context = _ExecutionContext(self.config)
        state = PipelineState()
        first_stage = 0
        if checkpoint is not None:
            if not isinstance(checkpoint, PipelineCheckpoint):
                raise TypeError("checkpoint must be a PipelineCheckpoint.")
            if selected.index < checkpoint.stage.index:
                raise ValueError("A checkpoint cannot be rewound; provide an earlier stage artifact.")
            checkpoint.assert_compatible(self.config, self.project_root, fingerprints=self.stage_fingerprints())
            state = deepcopy(checkpoint._state)
            first_stage = checkpoint.stage.index + 1
            context.progress.report(0.0, f"reusing {checkpoint.stage.value} checkpoint", force=True)
        else:
            context.progress.report(0.0, "start", force=True)
        methods = (self._prepare, self._solve_sheet, self._extract_contours, self._route_and_realize, self._evaluate)
        # These are work allocations, not a time prediction. Reused and
        # unrequested stages are excluded; local kernels report actual units.
        routing = not (self.config.skip_postprocessing or self.config.contour_only_sweep_mode)
        weights = (30, 20, 15, 25 if routing else 8 if not self.config.skip_sweep else 2, 10)
        remaining_weight = sum(weights[first_stage : selected.index + 1])
        position = 0.0
        for index in range(first_stage, selected.index + 1):
            canonical_config = self.config
            if self.requirements is not None and index > 0:
                from .application.research_requirements import requirements_evaluation_config

                self.config = requirements_evaluation_config(canonical_config)
            try:
                end = position + 100 * weights[index] / remaining_weight
                with progress_scope(
                    position,
                    end,
                    f"Computing {tuple(PipelineStage)[index].value}",
                    label="CoilGen",
                    stage=tuple(PipelineStage)[index].value,
                    overall=True,
                ):
                    if completed is None:
                        methods[index](state, context)
                    else:
                        with progress_scope(0, 95, "Computing numerical stage", label="CoilGen"):
                            methods[index](state, context)
                        with progress_scope(95, 100, "Retaining checkpoint and stage preview", label="CoilGen"):
                            completed(tuple(PipelineStage)[index], state)
                position = end
            except Exception as error:
                retain_pipeline_failure(
                    error,
                    state=state,
                    config=self.config,
                    stage=tuple(PipelineStage)[index],
                    source_config=canonical_config,
                    requirements=self.requirements,
                    operation=context.failed_operation,
                    timings=context.timing_summary,
                )
                raise
            finally:
                self.config = canonical_config
        if selected is not PipelineStage.EVALUATED:
            context.progress.finish(f"{selected.value} stage completed")
        return state, context

    def _prepare(self, state: PipelineState, context: _ExecutionContext) -> None:
        context.operations(
            (
                ("Load meshes", 5),
                ("Prepare meshes", 10),
                ("Sample desired field", 5),
                ("Build mesh neighborhoods", 10),
                ("Build stream-function basis", 10),
                ("Assemble field sensitivity", 60),
            )
        )
        stage = context.stage
        coil_mesh = context.compute("Load meshes", load_mesh_from_config, self.config, self.project_root)
        coil_mesh, symmetry_mesh_diagnostics = _prepare_half_mesh_symmetry_if_requested(coil_mesh, self.config)
        target_mesh = (
            load_optional_target_mesh(self.config.target_mesh_file, self.config, self.project_root)
            if self.requirements is None
            else None
        )
        stage("loaded meshes")

        prepared_meshes = context.compute(
            "Prepare meshes",
            lambda: [
                refine_mesh(mesh, self.config.iteration_num_mesh_refinement)
                for mesh in split_disconnected_mesh(coil_mesh)
            ],
        )
        prepared_meshes, refinement_symmetry_diagnostics = _restore_refined_single_mesh_symmetry(
            prepared_meshes,
            self.config,
        )
        symmetry_mesh_diagnostics.update(refinement_symmetry_diagnostics)
        if self.config.research_physics is not None:
            from .research.convex import preflight_research_physics

            preflight_research_physics(self.config.research_physics, sum(mesh.n_faces for mesh in prepared_meshes))
        state.parts = [CoilPart(mesh=mesh) for mesh in prepared_meshes]
        mesh_quality_diagnostics = _validate_prepared_mesh_quality(prepared_meshes, self.config)
        symmetry_mesh_diagnostics.update(mesh_quality_diagnostics)
        parts = state.parts
        secondary_target_mesh = (
            load_secondary_target_mesh(
                self.config,
                self.project_root,
                coil_mesh=coil_mesh,
                coil_meshes=prepared_meshes,
            )
            if self.requirements is None
            else None
        )
        suppression_surface_diagnostics = _suppression_surface_diagnostics(secondary_target_mesh)
        stage("prepared coil meshes")
        target_field = context.compute(
            "Sample desired field",
            lambda: (
                define_target_field([part.mesh for part in parts], target_mesh, secondary_target_mesh, self.config)
                if self.requirements is None
                else self._requirements_target()
            ),
        )
        state.target_field = target_field
        stage("defined target field")

        parts = context.compute("Build mesh neighborhoods", calculate_one_ring_by_mesh, parts)
        stage("built one-ring mesh neighborhoods")
        parts = context.compute("Build stream-function basis", calculate_basis_functions, parts)
        stage("built stream-function basis")
        parts = context.compute(
            "Assemble field sensitivity", calculate_sensitivity_matrix, parts, target_field, self.config
        )
        sensitivity_diagnostics = _sensitivity_matrix_diagnostics(parts, target_field)
        stage("calculated sensitivity matrix")
        state.parts = parts
        state.target_field = target_field
        state.prepared_diagnostics = {
            **sensitivity_diagnostics,
            **suppression_surface_diagnostics,
            **symmetry_mesh_diagnostics,
        }

    def _requirements_target(self):
        from .application.research_requirements import requirements_target

        return requirements_target(self.requirements)

    def _solve_sheet(self, state: PipelineState, context: _ExecutionContext) -> None:
        preoptimized = has_preoptimized_stream_function(self.config)
        named_weight = 10 if self.requirements is not None else 0
        context.operations(
            (("Prepare named operators", named_weight), ("Load stream function", 90))
            if preoptimized
            else (
                ("Prepare named operators", named_weight),
                ("Assemble resistance", 25),
                ("Optimize stream function", 65),
            )
        )
        if self.requirements is not None:
            from .application.research_requirements import prepare_requirement_constraints

            self.config = context.compute(
                "Prepare named operators", prepare_requirement_constraints, state.parts, self.config, self.requirements
            )
        parts = state.parts
        target_field = state.target_field
        combined_mesh = state.combined_mesh
        sf_b_field = state.sf_b_field
        regularization_diagnostics = state.regularization_diagnostics
        primary_surface_ind = state.primary_surface_index
        stage = context.stage
        if has_preoptimized_stream_function(self.config):
            parts, combined_mesh, sf_b_field, regularization_diagnostics = context.compute(
                "Load stream function",
                apply_preoptimized_stream_function,
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
            parts = context.compute("Assemble resistance", calculate_resistance_matrix, parts, self.config)
            stage("calculated resistance matrix")
            parts, combined_mesh, sf_b_field, regularization_diagnostics = context.compute(
                "Optimize stream function", stream_function_optimization, parts, target_field, self.config
            )
            stage("optimized stream function")
        regularization_diagnostics.update(stream_function_current_density_diagnostics(parts, self.config))
        regularization_diagnostics.update(dict(getattr(target_field, "target_diagnostics", {})))
        _update_gradient_hardware_target_diagnostics(regularization_diagnostics)
        regularization_diagnostics.update(state.prepared_diagnostics)
        state.parts = parts
        state.target_field = target_field
        state.combined_mesh = combined_mesh
        state.sf_b_field = sf_b_field
        state.regularization_diagnostics = regularization_diagnostics
        state.primary_surface_index = primary_surface_ind

    def _extract_contours(self, state: PipelineState, context: _ExecutionContext) -> None:
        distance_weight = 0 if self.config.skip_calculation_min_winding_distance else 10
        context.operations(
            (
                ("Select potential levels", 5),
                ("Extract contours", 20),
                ("Check loop significance", 15),
                ("Smooth contours", 10),
                ("Check contourability", 10),
                ("Sparsify contours", 10 if self.config.contour_sparsification_mode != "off" else 0),
                ("Prune contours", 5 if self.config.contour_sparsification_pruning_mode != "off" else 0),
                (
                    "Merge retained contours",
                    5 if self.config.contour_sparsification_post_pruning_merge_mode != "off" else 0,
                ),
                ("Measure contour spacing", distance_weight),
                ("Measure wire spacing", distance_weight),
            )
        )
        parts = state.parts
        target_field = state.target_field
        combined_mesh = state.combined_mesh
        sf_b_field = state.sf_b_field
        regularization_diagnostics = state.regularization_diagnostics
        primary_surface_ind = state.primary_surface_index
        stage = context.stage
        parts, primary_surface_ind = context.compute(
            "Select potential levels", calc_potential_levels, parts, combined_mesh, self.config
        )
        if 0 <= primary_surface_ind < len(parts):
            regularization_diagnostics.update(dict(parts[primary_surface_ind].contour_level_diagnostics))
        stage("calculated potential levels")
        parts = context.compute("Extract contours", extract_contours, parts, self.config)
        if 0 <= primary_surface_ind < len(parts):
            regularization_diagnostics.update(dict(parts[primary_surface_ind].contour_level_diagnostics))
        stage("extracted contours")
        parts = context.compute("Check loop significance", filter_loop_significance, parts, target_field, self.config)
        stage("filtered loop significance")
        parts = context.compute("Smooth contours", regularize_contour_paths, parts, self.config)
        if 0 <= primary_surface_ind < len(parts):
            regularization_diagnostics.update(dict(parts[primary_surface_ind].contour_level_diagnostics))
        stage("regularized contour paths")
        parts, contourability_diagnostics = context.compute(
            "Check contourability",
            validate_and_repair_contourability,
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
        parts = context.compute(
            "Sparsify contours",
            sparsify_contours,
            parts,
            target_field,
            self.config,
            progress=ProgressReporter.from_config(self.config, label="Contour sparsification"),
        )
        parts = context.compute(
            "Prune contours",
            prune_sparsified_contours,
            parts,
            target_field,
            self.config,
            progress=ProgressReporter.from_config(self.config, label="Sparse contour pruning"),
        )
        parts = context.compute(
            "Merge retained contours",
            consolidate_pruned_contours,
            parts,
            target_field,
            self.config,
            progress=ProgressReporter.from_config(self.config, label="Post-pruning contour merge"),
        )
        parts = analyze_contour_angle_stability(parts)
        if 0 <= primary_surface_ind < len(parts):
            regularization_diagnostics.update(dict(parts[primary_surface_ind].contour_level_diagnostics))
        stage("processed contour sparsification, pruning, and retained-contour consolidation")
        if self.config.skip_calculation_min_winding_distance:
            regularization_diagnostics["contour_distance_diagnostics_skipped"] = True
        else:
            parts = context.compute(
                "Measure contour spacing",
                find_minimal_contour_distance,
                parts,
                self.config,
                progress=ProgressReporter.from_config(self.config, label="Contour distance"),
            )
            parts = context.compute(
                "Measure wire spacing",
                calculate_unconnected_contour_wire_distances,
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
        state.parts = parts
        state.target_field = target_field
        state.combined_mesh = combined_mesh
        state.sf_b_field = sf_b_field
        state.regularization_diagnostics = regularization_diagnostics
        state.primary_surface_index = primary_surface_ind

    def _route_and_realize(self, state: PipelineState, context: _ExecutionContext) -> None:
        routing = not (self.config.skip_postprocessing or self.config.contour_only_sweep_mode)
        context.operations(
            (
                ("Group contour topology", 10 if routing else 0),
                ("Connect loop groups", 15 if routing else 0),
                ("Connect circuits", 15 if routing else 0),
                ("Shift return paths", 5 if routing else 0),
                ("Clean wire paths", 5),
                ("Realize conductors", 5 if context.realize_for_routing else 0),
                ("Select joint routes", 25 if routing else 0),
                ("Sweep conductors", 15 if not self.config.skip_sweep else 0),
                ("Validate circuit assembly", 4 if routing else 0),
                ("Generate PCB", 1 if self.config.make_cylndrical_pcb else 0),
            )
        )
        parts = state.parts
        target_field = state.target_field
        combined_mesh = state.combined_mesh
        sf_b_field = state.sf_b_field
        regularization_diagnostics = state.regularization_diagnostics
        primary_surface_ind = state.primary_surface_index
        stage = context.stage
        realize_for_routing = context.realize_for_routing
        skip_postprocessing = self.config.skip_postprocessing or self.config.contour_only_sweep_mode
        if skip_postprocessing:
            for part in parts:
                part.wire_path = None
                part.connected_groups = []
                part.opening_cuts_among_groups = []
            stage("skipped topology/routing postprocessing")
        else:
            parts = context.compute(
                "Group contour topology",
                topological_loop_grouping,
                parts,
                config=self.config,
                progress=ProgressReporter.from_config(self.config, label="Topology"),
            )
            stage("grouped contour topology")
            parts = calculate_group_centers(parts)
            regularization_diagnostics["routing_circuit_scope"] = self.config.routing_circuit_scope
            regularization_diagnostics["routing_source_part_count"] = len(parts)
            if self.config.routing_circuit_scope == "single" and len(parts) > 1:
                parts = assemble_routing_parts(parts, primary_surface_index=primary_surface_ind)
                state.parts = parts
                primary_surface_ind = 0
                regularization_diagnostics["routing_parts_assembled"] = True
            parts = context.compute(
                "Connect loop groups",
                interconnect_within_groups,
                parts,
                self.config,
                progress=ProgressReporter.from_config(self.config, label="Within-group routing"),
            )
            stage("interconnected within loop groups")
            parts = context.compute(
                "Connect circuits",
                interconnect_among_groups,
                parts,
                self.config,
                progress=ProgressReporter.from_config(self.config, label="Inter-group routing"),
            )
            stage("interconnected among loop groups")
            parts = context.compute("Shift return paths", shift_return_paths, parts, self.config)
            stage("shifted return paths")
        parts = context.compute(
            "Clean wire paths",
            cleanup_wire_paths,
            parts,
            config=self.config,
            progress=ProgressReporter.from_config(self.config, label="Route cleanup"),
        )
        stage("cleaned wire paths")
        if realize_for_routing:
            parts = context.compute("Realize conductors", realize_coil_paths, parts, self.config)
            stage("realized fabrication centerlines")
        parts = context.compute("Select joint routes", improve_group_routes, parts, self.config, target_field)
        stage("selected joint group routes")
        # Parts arrive without a mesh. A body returned by route
        # selection belongs to an independently validated candidate; retain its
        # exact geometry, including any routing-specific clearance contract.
        unswept_parts = [part for part in parts if part.swept_body is None]
        if unswept_parts:
            context.compute("Sweep conductors", create_swept_coil_meshes, unswept_parts, self.config)
        refresh_retained_route_diagnostics(parts)
        stage("finalized swept conductor meshes")
        # Retain the actual final candidate before a strict assembly gate can
        # reject it. This is failure evidence, never a completed checkpoint.
        state.parts = parts
        if not skip_postprocessing:
            from .routing_assembly import validate_routing_assembly

            context.compute("Validate circuit assembly", validate_routing_assembly, parts, self.config)
        parts = context.compute("Generate PCB", generate_cylindrical_pcb_print, parts, self.config)
        state.parts = parts
        state.target_field = target_field
        state.combined_mesh = combined_mesh
        state.sf_b_field = sf_b_field
        state.regularization_diagnostics = regularization_diagnostics
        state.primary_surface_index = primary_surface_ind

    def _evaluate(self, state: PipelineState, context: _ExecutionContext) -> None:
        context.operations(
            (
                ("Evaluate inductance and resistance", 30),
                ("Evaluate winding fields", 50),
                ("Check final field diagnostics", 15),
                ("Fit gradient", 5),
                ("Evaluate gradient slices", 20 if gradient_slice_maps_enabled(self.config) else 0),
            )
        )
        parts = state.parts
        target_field = state.target_field
        combined_mesh = state.combined_mesh
        sf_b_field = state.sf_b_field
        regularization_diagnostics = state.regularization_diagnostics
        primary_surface_ind = state.primary_surface_index
        stage = context.stage
        skip_postprocessing = self.config.skip_postprocessing or self.config.contour_only_sweep_mode
        parts, inductance_matrix, resistance, coil_lengths, coil_cross_sections = context.compute(
            "Evaluate inductance and resistance", calculate_inductance_by_coil_layout, parts, self.config
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
        ) = context.compute(
            "Evaluate winding fields", evaluate_field_errors, parts, self.config, target_field, sf_b_field
        )
        if any("routing_assembly" in part.topology_diagnostics for part in parts):
            from .group_route_fields import assess_group_route_field

            assembly_field = assess_group_route_field(
                field_layout_per_amp,
                field_layout_per_amp,
                target_field,
                self.config,
                parts[0].contour_step,
                0.0,
            )
            regularization_diagnostics["routing_assembly_fixed_current_field"] = assembly_field.metrics
            if not assembly_field.accepted:
                raise ValueError(
                    "Single-circuit assembly violates fixed-current field constraints: "
                    + "; ".join(assembly_field.reasons)
                )
            for part in parts:
                certificate = part.route_diagnostics["routing_assembly_validation"]
                certificate["field_acceptance_checked"] = True
                certificate["final_fixed_current_field"] = assembly_field.metrics
        regularization_diagnostics.update(
            layout_target_residual_diagnostics(field_by_layout, target_field, self.config)
        )
        regularization_diagnostics.update(
            context.compute(
                "Check final field diagnostics",
                final_field_diagnostics,
                parts,
                self.config,
                target_field,
                field_layout_per_amp,
                field_loops_per_amp,
                opt_current_layout,
            )
        )
        if not skip_postprocessing and all(part.fabrication_path is not None for part in parts):
            try:
                jump_metrics = [fabricated_jump_metrics(part, self.config) for part in parts]
                regularization_diagnostics.update(
                    routing_jump_count=sum(values["jump_count"] for values in jump_metrics),
                    routing_jump_metrics_by_circuit=jump_metrics,
                    routing_jump_count_checked=True,
                    routing_closed_circuit_count=len(parts),
                )
            except ValueError as error:
                regularization_diagnostics.update(
                    routing_jump_count_checked=False, routing_jump_count_reason=str(error)
                )
        design_context = normalize_design_context(getattr(self.config, "design_context", "mr_gradient"))
        layout_gradient = (
            context.compute("Fit gradient", calculate_layout_gradient, field_layout_per_amp, target_field)
            if design_context.enabled(EvaluationFeature.GRADIENT_FIT)
            else None
        )
        _apply_result_memory_policy(parts, self.config)
        context.record_timing("evaluated field errors")

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
            input_data=deepcopy(self.config),
            layout_gradient=layout_gradient,
            combined_mesh=combined_mesh,
            inductance_matrix=inductance_matrix,
            resistance=resistance,
            coil_lengths=coil_lengths,
            coil_cross_sections=coil_cross_sections,
            timing_summary=context.timing_summary,
        )
        if gradient_slice_maps_enabled(self.config):
            slice_start = time.perf_counter()
            slice_maps = context.compute("Evaluate gradient slices", gradient_slice_map_result, result)
            result.regularization_diagnostics.update(dict(slice_maps.summary))
            result.timing_summary.append(
                {
                    "stage": "evaluated gradient slice maps",
                    "stage_seconds": float(time.perf_counter() - slice_start),
                    "elapsed_seconds": float(time.perf_counter() - context.start_time),
                }
            )
        state.result = result
        context.progress.finish()


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
