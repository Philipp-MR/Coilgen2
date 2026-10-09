from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from ..field_components import field_component_values
from ..pipeline import CoilGenPipeline
from ..pipeline_stages import PipelineCheckpoint, PipelineStage
from ..progress import DesignCancelled, ProgressReporter, progress_scope
from ..reporting import write_design_report
from ..scientific_arrays import scientific_arrays as _scientific_arrays
from ..application.stage_preview import MANIFEST_NAME as STAGE_PREVIEW_MANIFEST, publish_stage_preview
from .requests import ResearchRequest
from .storage import (
    DirectoryOwner,
    atomic_json,
    file_digest,
    json_value,
    request_identity,
    result_metadata_digest,
    verify_input_files,
)
from .validation import (
    validate_on_independent_grid,
    validate_fabricated_constraints,
    validate_winding_on_independent_grid,
)


def _pipeline_sweep_evidence(parts, config) -> dict:
    """Retain recorded native sweep checks, independently of fabrication review.

    The mesh enumeration matches ``_scientific_arrays``. Missing legacy checks
    remain unknown; neither a skipped sweep nor a mesh alone proves readiness.
    No validation, geometry conditioning or current fit is performed here.
    """

    def recorded_bool(value):
        return bool(value) if isinstance(value, (bool, np.bool_)) else None

    bodies = []
    for part_index, part in enumerate(parts):
        native = [
            (body.mesh, body.contour_index, body.turn_multiplicity, body.diagnostics) for body in part.swept_bodies
        ]
        if not native and part.swept_body is not None:
            native = [(part.swept_body, None, 1, part.manufacturing_diagnostics)]
        for body_index, (_, contour_index, multiplicity, diagnostics) in enumerate(native):
            bodies.append(
                {
                    "body_id": f"pipeline_part{part_index + 1:03d}_body{body_index + 1:04d}",
                    "part_index": part_index + 1,
                    "body_index": body_index + 1,
                    "contour_index": contour_index,
                    "source_kind": "contour" if contour_index is not None else "wire_path",
                    "turn_multiplicity": multiplicity,
                    "vertices_key": f"swept_{part_index}_{body_index}_vertices_m",
                    "faces_key": f"swept_{part_index}_{body_index}_faces",
                    "manufacturing_ok": recorded_bool(diagnostics.get("manufacturing_ok")),
                    "mesh_solid_ok": recorded_bool(diagnostics.get("mesh_solid_ok")),
                    "diagnostics": json_value(diagnostics),
                    "route_diagnostics": json_value(part.route_diagnostics),
                }
            )
    statuses = [body["manufacturing_ok"] for body in bodies]
    return {
        "scope": "pipeline_sweep_geometry",
        "surface_realization": config.sweep_surface_realization,
        "body_count": len(bodies),
        "checked_body_count": sum(status is not None for status in statuses),
        "checks_passed": False if False in statuses else True if statuses and all(statuses) else None,
        "bodies": bodies,
    }


def _wire_field_metrics(field, target, config) -> dict:
    """Global design-grid residuals; target zeros never create pointwise ratios."""
    actual = np.asarray(field, dtype=float)
    reference = np.asarray(target.b, dtype=float)
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[1] != 3:
        return {"available": False, "reason": "Wire field does not match the design-grid vector target."}
    weights = np.asarray(target.weights, dtype=float).reshape(-1)
    suppressed = np.asarray(target.suppressed, dtype=bool).reshape(-1)
    if len(weights) != len(reference) or len(suppressed) != len(reference):
        return {"available": False, "reason": "Wire field weights or suppression mask do not match the grid."}
    active = ~suppressed & np.isfinite(weights) & (weights > 0)
    if not np.any(active):
        return {"available": False, "reason": "No active positive-weight design samples."}
    if not np.all(np.isfinite(actual[active])) or not np.all(np.isfinite(reference[active])):
        return {"available": False, "reason": "Active wire fields or targets are nonfinite."}
    component = config.target_field_component
    actual = field_component_values(actual[active], component, b0_direction=config.b_0_direction)
    reference = field_component_values(reference[active], component, b0_direction=config.b_0_direction)
    point_count = int(np.count_nonzero(active))
    actual, reference = actual.reshape(point_count, -1), reference.reshape(point_count, -1)
    # Scale before squaring so a valid finite field/weight cannot overflow an
    # intermediate sum. Vector metrics use magnitude RMS, scalar metrics Bz etc.
    scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(reference))))
    weights = weights[active] / np.max(weights[active])
    weights /= np.sum(weights)
    residual = (actual / scale - reference / scale) if scale else np.zeros_like(actual)
    reference_scaled = reference / scale if scale else reference
    error_norm = np.hypot.reduce(residual, axis=1)
    reference_norm = np.hypot.reduce(reference_scaled, axis=1)
    rms = float(np.sqrt(weights @ (error_norm**2)))
    target_rms = float(np.sqrt(weights @ (reference_norm**2)))
    absolute = scale * rms
    peak = scale * float(np.max(error_norm))
    if not np.all(np.isfinite([absolute, peak])):
        return {"available": False, "reason": "Wire residual magnitude exceeds finite SI reporting range."}
    return {
        "available": True,
        "component": component,
        "sample_count": point_count,
        "rms_error_T": absolute,
        "peak_error_T": peak,
        "target_rms_T": scale * target_rms,
        "relative_rms_error": rms / target_rms if target_rms > 0 else None,
        "rms_convention": "vector_magnitude" if actual.shape[1] == 3 else "scalar",
    }


def _wire_field_evidence(result) -> tuple[dict, dict]:
    """Export evaluated wire maps without recalculation, current fits or acceptance.

    The pipeline layout field can use its realized centroid, its connected
    route, or copied contours. Only the first two qualify as routed evidence.
    Legacy evaluations can refit the reported current; original-current maps
    use the already computed per-amp operators when every part shares a drive.
    """
    from ..evaluation import field_evaluation_current_source, layout_field_geometry

    config = result.input_data
    current = float(result.needed_current_layout)
    policy = field_evaluation_current_source(config)
    refitted = policy.strip().lower().replace("-", "_") not in {"stream_function", "stream", "contour_step", "sf"}
    steps = [part.contour_step for part in result.coil_parts]
    common = bool(steps) and all(step is not None and np.isfinite(step) and step != 0 for step in steps)
    if common:
        common = all(np.isclose(step, steps[0], rtol=1e-12, atol=0.0) for step in steps)
    original_current = float(steps[0]) if common else None
    arrays, evidence = {}, {}
    geometry = [layout_field_geometry(part, config) for part in result.coil_parts]
    for key, field, per_amp in (
        ("contour", result.field_by_unconnected_loops, result.field_loops_per1Amp),
        ("routed", result.field_by_layout, result.field_layout_per1Amp),
    ):
        entry = {
            "scope": "design_grid_wire_field",
            "field_model": "thin_wire_biot_savart",
            "independent_sampling": False,
            "acceptance_claim": False,
            "current_A": current if np.isfinite(current) else None,
            "current_source": policy,
            "current_refitted": refitted,
            "geometry_sources": ["oriented_unconnected_contours"]
            if key == "contour"
            else [source for _, source in geometry],
        }
        if key == "routed" and (not geometry or any(points is None for points, _ in geometry)):
            entry.update(
                available=False, reason="A connected evaluated winding is unavailable; layout field copies contours."
            )
        elif not np.isfinite(current):
            entry.update(available=False, reason="The evaluated wire current is nonfinite.")
        else:
            entry.update(_wire_field_metrics(field, result.target_field, config))
            if entry["available"]:
                arrays[f"{key}_field_T"] = np.asarray(field)
                if key == "routed":
                    for index, (points, _) in enumerate(geometry):
                        arrays[f"evaluated_winding_{index}_points_m"] = points
                if original_current is not None:
                    original_field = np.asarray(per_amp) * original_current
                    original = _wire_field_metrics(original_field, result.target_field, config)
                    original.update(
                        current_A=original_current, current_source="original_contour_step", current_refitted=False
                    )
                    entry["at_stream_current"] = original
                    if original["available"]:
                        arrays[f"{key}_field_at_stream_current_T"] = original_field
                else:
                    entry["at_stream_current"] = {
                        "available": False,
                        "reason": "A common finite nonzero original contour current is unavailable.",
                    }
        evidence[key] = entry
    return arrays, evidence


def _constraint_rows(diagnostics) -> list[dict]:
    return [
        {
            "name": row["name"],
            "value": row.get("value"),
            "limit": row.get("limit"),
            "effective_limit": row.get("effective_limit"),
            "unit": row.get("unit", ""),
            "passed": row.get("met"),
            "scope": "continuous_sheet",
        }
        for row in diagnostics.get(
            "research_constraint_diagnostics", diagnostics.get("constrained_solver_contract_constraint_diagnostics", [])
        )
    ]


def _project_provenance(request: ResearchRequest) -> dict:
    return {
        "project_id": request.project_id,
        "project_revision": request.project_revision,
        "project_fingerprint": request.project_fingerprint,
        "project_model_fingerprint": request.project_model_fingerprint,
    }


def _target_analysis_recipe(request, target):
    """Retain the executed analytic scale for new-point analysis, without refitting."""
    if request.requirements is not None:
        return {}
    config = request.to_config()
    if str(config.target_field_definition_file).strip().lower() not in {"", "none"}:
        return {}
    diagnostics = getattr(target, "target_diagnostics", {})
    scale = diagnostics.get("target_gradient_normalization_scale")
    if scale is None or not np.isfinite(scale):
        return {}
    return {
        "target_recipe": {
            "field_shape_function": config.field_shape_function,
            "target_field_units": config.target_field_units,
            "normalization_scale": float(scale),
            "normalization_method": diagnostics.get("target_gradient_normalization_method"),
            "field_component": "Bz",
            "coordinate_frame": "world",
        }
    }


def _design_work_ranges(request):
    """Allocate only requested operations; percentages describe work, not an ETA."""
    stage = PipelineStage(request.pipeline_stage)
    full = stage is PipelineStage.EVALUATED
    weights = [("inputs", 2), ("pipeline", 80), ("checkpoint", 2)]
    if request.requirements is not None or stage.index >= PipelineStage.SHEET.index:
        weights.append(("validation", 10))
    if full and request.fabricate:
        weights.extend((("fabrication", 30), ("fabricated_physics", 5)))
        if request.engineering is not None:
            if request.engineering.get("independent_winding_validation", True) and request.requirements is None:
                weights.append(("winding_validation", 10))
            weights.append(("engineering", 5))
    if full and request.dynamic_analysis is not None:
        weights.append(("dynamic", 2))
    if full and request.fabricate and request.requirements is not None:
        weights.append(("fabricated_requirements", 10))
    if request.scene is not None and (full or stage is PipelineStage.ROUTED):
        weights.append(("scene", 8))
    if full:
        weights.append(("wire_evidence", 2))
    weights.append(("save", 8))
    total, position, ranges = sum(weight for _, weight in weights), 0.0, {}
    for index, (name, weight) in enumerate(weights):
        end = 98.0 if index == len(weights) - 1 else position + 98 * weight / total
        ranges[name] = (position, end)
        position = end
    return ranges


def _save_partial_result(
    request, checkpoint, directory, manifest, digest, started, stage_artifacts=(), work_ranges=None
) -> dict:
    """Incomplete workflows remain inspectable without claiming winding acceptance."""
    state = checkpoint.state
    work_ranges = _design_work_ranges(request) if work_ranges is None else work_ranges
    diagnostics = checkpoint.diagnostics
    arrays = _scientific_arrays(state.parts, state.target_field, state.combined_mesh, state.sf_b_field)
    pipeline_sweeps = _pipeline_sweep_evidence(state.parts, checkpoint.config)
    validation = {
        "available": False,
        "verified": False,
        "passed": None,
        "reason": "The continuous sheet has not been solved.",
    }
    constraints = _constraint_rows(diagnostics)
    relative = None
    requirements_review = None
    if request.requirements is not None:
        from ..application.research_requirements import evaluate_requirements, named_independent_sheet_validation

        with progress_scope(*work_ranges["validation"], "Reviewing regional checkpoint fields", overall=True):
            requirements_review, requirement_arrays = evaluate_requirements(
                state.parts, checkpoint.config, request.requirements, design_margin=request.design_margin
            )
        arrays.update(requirement_arrays)
    if checkpoint.stage.index >= PipelineStage.SHEET.index:
        sheet_view = SimpleNamespace(
            input_data=checkpoint.config,
            target_field=state.target_field,
            combined_mesh=state.combined_mesh,
            coil_parts=state.parts,
            regularization_diagnostics=diagnostics,
        )
        if request.requirements is None:
            with progress_scope(*work_ranges["validation"], "Checking an independent sheet grid", overall=True):
                validation = validate_on_independent_grid(
                    sheet_view, resolution=request.validation_resolution, design_margin=request.design_margin
                )
            relative = _relative_field_error(
                state.sf_b_field,
                state.target_field.b,
                ~np.asarray(state.target_field.suppressed, bool),
                checkpoint.config.target_field_component,
                checkpoint.config.b_0_direction,
            )
        else:
            validation = named_independent_sheet_validation(requirements_review, request.requirements)
        constraints.append(
            {
                "name": "Regional independent sheet validation"
                if request.requirements is not None
                else "Independent grid field error",
                "value": validation.get("value"),
                "limit": validation.get("limit"),
                "effective_limit": validation.get("effective_limit"),
                "unit": validation.get("unit", "T"),
                "passed": validation.get("passed"),
                "scope": "continuous_sheet",
            }
        )
        if validation["available"] and request.requirements is None:
            arrays.update(
                validation_points_m=validation["points_m"],
                validation_field_T=validation["field_T"],
                validation_target_T=validation["target_T"],
            )
    if requirements_review is not None:
        from ..application.research_requirements import requirements_constraint_rows

        constraints.extend(requirements_constraint_rows(requirements_review, include_wire=False))
    sheet_passed = (
        bool(diagnostics.get("constrained_solver_contract_met", False))
        and validation.get("passed") is True
        and all(row["passed"] is True for row in constraints)
    )
    if requirements_review is not None and checkpoint.stage is PipelineStage.ROUTED:
        constraints.extend(
            row for row in requirements_constraint_rows(requirements_review) if row["scope"] != "continuous_sheet"
        )
    values = diagnostics.get("research_physical_values", diagnostics)
    scene_review = None
    if request.scene is not None and checkpoint.stage is PipelineStage.ROUTED:
        from ..application.spatial_scene import SpatialScene, evaluate_spatial_scene

        with progress_scope(*work_ranges["scene"], "Checking winding against the spatial scene", overall=True):
            scene_review = evaluate_spatial_scene(sheet_view, SpatialScene.from_dict(request.scene))
        constraints.extend(scene_review["checks"])
    scope = {
        PipelineStage.PREPARED: "prepared_geometry",
        PipelineStage.SHEET: "continuous_sheet",
        PipelineStage.CONTOURS: "unconnected_contours",
        PipelineStage.ROUTED: "realized_winding",
    }[checkpoint.stage]
    summary = {
        "accepted": checkpoint.stage is PipelineStage.SHEET and sheet_passed,
        "sheet_accepted": sheet_passed,
        "pipeline_sweep_checks_passed": pipeline_sweeps["checks_passed"],
        "fabrication_review_performed": False,
        "scope": scope,
        "construction_released": False,
        "pipeline_stage": checkpoint.stage.value,
        "objective": diagnostics.get("research_objective"),
        "relative_field_error": relative,
        "validated_relative_field_error": validation.get("relative_field_error"),
        "field_source": validation.get("source", "unsolved"),
        "power_w": values.get("sheet_power_W"),
        "energy_j": values.get("energy_J"),
        "max_current_density_a_per_m": values.get(
            "max_surface_current_density_A_per_m", diagnostics.get("surface_current_density_max_A_per_m")
        ),
        "vertices": len(arrays["vertices_m"]),
        "triangles": len(arrays["faces"]),
        "contour_components": sum(len(part.contour_lines) for part in state.parts),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_start, save_end = work_ranges["save"]
    save_span = save_end - save_start
    with progress_scope(
        save_start, save_start + 0.65 * save_span, "Compressing scientific arrays and saving diagnostics", overall=True
    ):
        np.savez_compressed(directory / "scientific_result.npz", **arrays)
        atomic_json(directory / "solver_diagnostics.json", diagnostics)
    artifacts = [
        {"name": name, "kind": kind}
        for name, kind in (
            ("request.json", "request"),
            ("manifest.json", "provenance"),
            ("pipeline_checkpoint.cgstage", "checkpoint"),
            ("scientific_result.npz", "npz"),
            ("solver_diagnostics.json", "diagnostics"),
        )
    ]
    artifacts.extend(stage_artifacts)
    with progress_scope(
        save_start + 0.65 * save_span, save_start + 0.75 * save_span, "Rechecking immutable input files", overall=True
    ):
        verify_input_files(manifest)
    for index, artifact in enumerate(artifacts):
        with progress_scope(
            save_start + save_span * (0.75 + 0.2 * index / len(artifacts)),
            save_start + save_span * (0.75 + 0.2 * (index + 1) / len(artifacts)),
            f"Verifying checkpoint artifact {index + 1}/{len(artifacts)}: {artifact['name']}",
            overall=True,
        ):
            artifact["sha256"] = file_digest(directory / artifact["name"])
    payload = {
        "name": request.name,
        "software": dict(manifest["software"]),
        "config": request.to_dict()["config"],
        **_target_analysis_recipe(request, state.target_field),
        "summary": summary,
        "constraints": constraints,
        "artifacts": artifacts,
        "warnings": [
            "Stage checkpoint; final winding fields, electrical properties and manufacturing acceptance remain unverified."
        ],
        "validation": {
            key: value for key, value in validation.items() if key not in {"points_m", "field_T", "target_T"}
        },
        "pipeline_sweeps": pipeline_sweeps,
        **({"requirements_review": requirements_review} if requirements_review is not None else {}),
        **({"scene_review": scene_review} if scene_review is not None else {}),
        "request_sha256": digest,
        "pipeline_stage": checkpoint.stage.value,
        "checkpoint": "pipeline_checkpoint.cgstage",
        **_project_provenance(request),
    }
    with progress_scope(save_start + 0.95 * save_span, save_end, "Saving checkpoint result evidence", overall=True):
        atomic_json(directory / "result.json", payload)
    return json_value(payload)


def _recorded_contour_surface_spacing(parts, config):
    if config.skip_calculation_min_winding_distance:
        return None
    minima = []
    for part in parts:
        count = len(part.contour_lines)
        distances = np.asarray(part.contour_distance_matrix)
        if count < 2 or distances.shape != (count, count):
            continue
        pairs = distances[np.triu_indices(count, 1)]
        finite = pairs[np.isfinite(pairs) & (pairs >= 0)]
        if len(finite):
            minima.append(float(finite.min()))
    return min(minima, default=None)


def _relative_field_error(actual, target, mask, component, direction):
    a = field_component_values(np.asarray(actual)[mask], component, b0_direction=direction)
    b = field_component_values(np.asarray(target)[mask], component, b0_direction=direction)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    denominator = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denominator) if denominator > 0 else None


def _run_design(request: ResearchRequest | dict, output_dir: str | Path, *, threads=1) -> dict:
    """Run a resolved design and retain evidence from sheet through fabrication.

    Completion means a result was computed. ``summary.accepted`` is a separate
    engineering decision; invalid/unknown fabrication is never promoted to pass.
    """
    from .fabrication import compile_fabrication
    from .kernels import geometry_acceleration

    request = ResearchRequest.from_dict(request.to_dict() if isinstance(request, ResearchRequest) else request)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = request.to_config(threads=threads)
    # Runtime owns thread allocation even when expert settings request more.
    config = replace(
        config,
        parallel_worker_count=threads,
        constrained_optimization_threads=threads,
        plot_flag=False,
        save_stl_flag=False,
    )
    progress = ProgressReporter(label="Research")
    started = time.perf_counter()
    work_ranges = _design_work_ranges(request)
    with progress_scope(*work_ranges["inputs"], "Verifying and retaining immutable design inputs", overall=True):
        resolved = asdict(config)
        digest, manifest = request_identity(request.to_dict(), resolved)
        manifest["request_sha256"] = digest
        atomic_json(directory / "request.json", request.to_dict())
        atomic_json(directory / "manifest.json", manifest)
    stage_artifacts = []
    stage_evidence = []

    def retain_stage(completed):
        # Retain actual completed boundaries; an evaluated state cannot be rewound.
        name = f"checkpoints/{completed.stage.value}.cgstage"
        progress.report(None, f"Saving recovery checkpoint: {completed.stage.value}")
        completed.save(directory / name)
        stage_artifacts.append({"name": name, "kind": "stage_checkpoint", "stage": completed.stage.value})
        try:
            progress.report(None, f"Verifying recovery checkpoint and updating provenance: {completed.stage.value}")
            stage_evidence.append({**stage_artifacts[-1], "sha256": file_digest(directory / name)})
            atomic_json(
                directory / "checkpoints/evidence.json",
                {
                    "schema": "coilgen.checkpoint-evidence/1",
                    "request_sha256": digest,
                    **{
                        key: getattr(request, key)
                        for key in (
                            "project_id",
                            "project_revision",
                            "project_fingerprint",
                            "project_model_fingerprint",
                        )
                    },
                    "artifacts": stage_evidence,
                },
            )
        except OSError as exc:
            # Recovery metadata is optional. The last atomic evidence record
            # remains usable; computation and its final result are preserved.
            progress.report(
                None,
                f"Checkpoint recovery metadata unavailable for {completed.stage.value}; "
                f"only earlier committed boundaries can be resumed: {exc}",
            )
        publish_preview(completed)

    def publish_preview(completed):
        preview, error = None, None
        try:
            progress.report(None, f"Preparing saved viewport evidence: {completed.stage.value}")
            publish_stage_preview(directory, completed, request, digest)
            preview = STAGE_PREVIEW_MANIFEST
        except DesignCancelled:
            raise
        except Exception as exc:
            # Display data are optional. Preserve the numerical result and its
            # completed checkpoint if visualization storage is unavailable.
            error = f"{type(exc).__name__}: {exc}"
        progress.boundary(completed.stage.value, preview=preview, preview_error=error)

    with geometry_acceleration(request.kernel_backend):
        with progress_scope(*work_ranges["pipeline"], "Computing the requested numerical stages", overall=True):
            resumed = PipelineCheckpoint.load(request.resume_checkpoint) if request.resume_checkpoint else None
            pipeline_controls = {"requirements": request.requirements} if request.requirements is not None else {}
            checkpoint = CoilGenPipeline(config, project_root=directory, **pipeline_controls).run_stage(
                request.pipeline_stage, resumed, checkpoint_callback=retain_stage
            )
        if resumed is not None and checkpoint.stage is resumed.stage:
            # run_stage has now checked compatibility; no new callback ran.
            publish_preview(checkpoint)
        with progress_scope(*work_ranges["checkpoint"], "Saving the completed scientific checkpoint", overall=True):
            checkpoint.save(directory / "pipeline_checkpoint.cgstage")
    if checkpoint.stage is not PipelineStage.EVALUATED:
        return _save_partial_result(
            request, checkpoint, directory, manifest, digest, started, stage_artifacts, work_ranges
        )
    result = checkpoint.result
    pipeline_sweeps = _pipeline_sweep_evidence(result.coil_parts, config)
    diagnostics = result.regularization_diagnostics
    requirements_review, requirements_arrays = None, {}
    if request.requirements is not None:
        from ..application.research_requirements import evaluate_requirements, named_independent_sheet_validation

        with progress_scope(*work_ranges["validation"], "Checking named regions and physical poses", overall=True):
            requirements_review, requirements_arrays = evaluate_requirements(
                result.coil_parts, config, request.requirements, design_margin=request.design_margin
            )
            validation = named_independent_sheet_validation(requirements_review, request.requirements)
    else:
        with progress_scope(*work_ranges["validation"], "Checking an independent field grid", overall=True):
            validation = validate_on_independent_grid(
                result, resolution=request.validation_resolution, design_margin=request.design_margin
            )
    mask = ~np.asarray(result.target_field.suppressed, bool)
    relative = (
        None
        if request.requirements is not None
        else _relative_field_error(
            result.b_field_opt_sf, result.target_field.b, mask, config.target_field_component, config.b_0_direction
        )
    )
    constraints = _constraint_rows(diagnostics)
    validation_passed = validation.get("passed")
    constraints.append(
        {
            "name": "Regional independent sheet validation"
            if request.requirements is not None
            else "Independent grid field error",
            "value": validation.get("value"),
            "limit": validation.get("limit"),
            "effective_limit": validation.get("effective_limit"),
            "unit": validation.get("unit", "T"),
            "passed": validation_passed,
            "scope": "continuous_sheet",
        }
    )
    if requirements_review is not None:
        from ..application.research_requirements import requirements_constraint_rows

        constraints.extend(requirements_constraint_rows(requirements_review, include_wire=False))
    sheet_accepted = bool(diagnostics.get("constrained_solver_contract_met", False)) and all(
        row["passed"] is True for row in constraints
    )
    warnings = [
        "Magnetostatic sheet model; RF loading, SAR, transient eddy currents and thermal cooling are not modeled."
    ]
    if not validation["available"]:
        warnings.append(validation["reason"])
    if validation_passed is False:
        warnings.append(
            "Named regional validation exceeds a requested physical bound; review its limiting region, samples and pose."
            if request.requirements is not None
            else "The independent field grid exceeds the requested tolerance; refine the target sampling or design margin."
        )
    fabrication = None
    fabricated_physics = None
    winding_validation = None
    engineering_review = None
    engineering = None
    dynamic_analysis_review = None
    dynamic_analysis = getattr(request, "dynamic_analysis", None)
    if request.engineering is not None:
        from ..gradient_engineering import GradientEngineeringSpec, evaluate_gradient_engineering

        engineering = GradientEngineeringSpec.from_dict(request.engineering)
    artifacts = [
        {"name": "request.json", "kind": "request"},
        {"name": "manifest.json", "kind": "provenance"},
        {"name": "pipeline_checkpoint.cgstage", "kind": "checkpoint"},
    ]
    artifacts.extend(stage_artifacts)
    if request.fabricate:
        with progress_scope(
            *work_ranges["fabrication"], "Conditioning and checking the fabrication assembly", overall=True
        ):
            fabrication = compile_fabrication(
                result,
                output_dir=directory / "fabrication",
                strict=True,
                kernel_backend=request.kernel_backend,
                **({"circuits": engineering.circuit_map} if engineering else {}),
            )
        warnings.extend(fabrication.failure_reasons)
        if any(not plan.get("drive_feed_designed", False) for plan in fabrication.winding_steps):
            warnings.append(
                "Feed breaks, drive terminals and supply leads are not designed; closed turns are geometric circuits only."
            )
        constraints.append(
            {
                "name": "Fabrication geometry and complete assembly",
                "value": len(fabrication.failure_reasons),
                "limit": 0,
                "unit": "failures",
                "passed": fabrication.ready,
            }
        )
        validation_controls = {"check_default_field": False} if request.requirements is not None else {}
        with progress_scope(
            *work_ranges["fabricated_physics"], "Checking fabricated electrical and physical limits", overall=True
        ):
            fabricated_physics = validate_fabricated_constraints(
                result, fabrication, design_margin=request.design_margin, **validation_controls
            )
        constraints.extend(fabricated_physics["checks"])
        warnings.extend(fabricated_physics["unknown_reasons"])
        if engineering:
            if engineering.independent_winding_validation and request.requirements is None:
                with progress_scope(
                    *work_ranges["winding_validation"], "Checking winding on independent target samples", overall=True
                ):
                    winding_validation = validate_winding_on_independent_grid(
                        result,
                        fabrication,
                        resolution=request.validation_resolution,
                        design_margin=request.design_margin,
                    )
                constraints.extend(winding_validation["checks"])
                warnings.extend(winding_validation["unknown_reasons"])
                if not winding_validation["available"]:
                    constraints.append(
                        {
                            "name": "Independent winding validation unavailable",
                            "passed": None,
                            "verified": False,
                            "scope": "independent_realized_winding",
                            "value": None,
                            "limit": None,
                            "unit": "",
                            "reason": winding_validation.get("reason"),
                        }
                    )
            with progress_scope(
                *work_ranges["engineering"], "Reviewing the signed drive waveform and thermal model", overall=True
            ):
                circuit_references = {}
                if dynamic_analysis is not None:
                    from ..application.dynamic_studies import verified_circuit_references

                    with progress_scope(0, 10):
                        circuit_references = verified_circuit_references(dynamic_analysis, fabrication, engineering)
                with progress_scope(10 if dynamic_analysis is not None else 0, 100):
                    engineering_review = evaluate_gradient_engineering(engineering, fabrication, **circuit_references)
            drive = engineering_review["drive"]
            if engineering.drive is not None and not drive["available"]:
                constraints.append(
                    {
                        "name": "Requested drive review unavailable",
                        "passed": None,
                        "verified": False,
                        "scope": "fabricated_drive_waveform",
                        "value": None,
                        "limit": None,
                        "unit": "",
                        "reason": drive.get("reason"),
                    }
                )
            for body in drive.get("bodies", ()):
                for title, value, limit, unit, passed in (
                    ("Amplifier peak current", "peak_current_A", "current_limit_A", "A", "within_current_limit"),
                    ("Amplifier peak voltage", "peak_voltage_V", "voltage_limit_V", "V", "within_voltage_limit"),
                ):
                    verified = (
                        title != "Amplifier peak voltage"
                        or len(drive["bodies"]) == 1
                        or drive.get("mutual_inductance_included") is True
                    )
                    constraints.append(
                        {
                            "name": title,
                            "body_id": body["body_id"],
                            "value": body[value],
                            "limit": body[limit],
                            "effective_limit": body[limit],
                            "unit": unit,
                            "passed": body[passed] if verified else None,
                            "scope": "fabricated_drive_waveform",
                            "verified": verified,
                        }
                    )
            if len(drive.get("bodies", ())) > 1 and not drive.get("mutual_inductance_included"):
                warnings.append(
                    "Multi-body drive voltage estimates exclude mutual inductance; amplifier voltage acceptance remains unverified."
                )
            thermal = engineering_review["thermal"]
            if engineering.drive is not None and engineering.drive.thermal is not None:
                constraints.append(
                    {
                        "name": "Lumped thermal model peak temperature",
                        "scope": "fabricated_drive_thermal_model",
                        "value": thermal.get("peak_temperature_K"),
                        "limit": thermal.get("temperature_limit_K"),
                        "effective_limit": thermal.get("temperature_limit_K"),
                        "unit": "K",
                        "verified": thermal.get("available") is True,
                        "passed": thermal.get("within_model_limit") if thermal.get("available") else None,
                        "reason": thermal.get("reason"),
                        "model": thermal.get("model"),
                    }
                )
        for path in fabrication.artifacts:
            name = Path(path).relative_to(directory).as_posix()
            artifacts.append({"name": name, "kind": Path(path).suffix.lstrip(".")})
    else:
        warnings.append(
            "Continuous-sheet/contour study: routing, terminals and fabrication readiness have not been established."
        )
    if dynamic_analysis is not None:
        from ..application.dynamic_studies import review_dynamic_analyses

        with progress_scope(*work_ranges["dynamic"], "Reviewing external dynamic-analysis evidence", overall=True):
            dynamic_analysis_review = review_dynamic_analyses(
                dynamic_analysis, fabrication, engineering, engineering_review
            )
        constraints.extend(dynamic_analysis_review["checks"])
        warnings.extend(reason for item in dynamic_analysis_review["analyses"] for reason in item["reasons"])
    if request.requirements is not None:
        if fabrication is not None:
            bodies = {row["body_id"]: row for row in fabrication.electrical_metrics["bodies"]}
            plans = fabrication.winding_steps
            paths_complete = (
                bool(plans) and len(plans) == len(bodies) and {plan["body_id"] for plan in plans} == set(bodies)
            )
            for plan in plans:
                points = np.asarray(plan["electrical_points_m"])
                current = bodies.get(plan["body_id"], {}).get("current_A")
                paths_complete = (
                    paths_complete
                    and plan.get("circuit_complete") is True
                    and points.ndim == 2
                    and points.shape[1] == 3
                    and len(points) >= 3
                    and np.all(np.isfinite(points))
                )
                paths_complete = (
                    paths_complete
                    and current is not None
                    and np.isfinite(current)
                    and np.linalg.norm(points[-1] - points[0]) <= 1e-10
                )
            complete_paths = (
                {plan["body_id"]: (plan["electrical_points_m"], bodies[plan["body_id"]]["current_A"]) for plan in plans}
                if paths_complete
                else None
            )
            with progress_scope(
                *work_ranges["fabricated_requirements"],
                "Checking fabricated fields in named regions and poses",
                overall=True,
            ):
                requirements_review, requirements_arrays = evaluate_requirements(
                    result.coil_parts,
                    config,
                    request.requirements,
                    fabrication=fabrication,
                    complete_paths=complete_paths,
                    design_margin=request.design_margin,
                )
        from ..application.research_requirements import requirements_constraint_rows

        if not (result.input_data.skip_postprocessing and not request.fabricate):
            constraints.extend(
                row for row in requirements_constraint_rows(requirements_review) if row["scope"] != "continuous_sheet"
            )
    scene_review = None
    if request.scene is not None:
        from ..application.spatial_scene import SpatialScene, evaluate_spatial_scene

        with progress_scope(
            *work_ranges["scene"], "Checking conductor and feed clearances in the spatial scene", overall=True
        ):
            scene_review = evaluate_spatial_scene(
                result, SpatialScene.from_dict(request.scene), fabrication=fabrication
            )
        constraints.extend(scene_review["checks"])
    electrical = fabrication.electrical_metrics["bodies"] if fabrication else []
    # Report a complete circuit inductance only for one fully modeled body;
    # adding self-inductances of separate bodies would omit mutual coupling.
    complete_single = len(electrical) == 1 and electrical[0].get("circuit_complete", False)
    physical_ok = bool(diagnostics.get("constrained_solver_contract_met", False))
    accepted = physical_ok and validation_passed is True and all(c["passed"] is True for c in constraints)
    values = diagnostics.get("research_physical_values", diagnostics)
    summary = {
        "accepted": accepted,
        "sheet_accepted": sheet_accepted,
        "pipeline_sweep_checks_passed": pipeline_sweeps["checks_passed"],
        "fabrication_review_performed": fabrication is not None,
        "pipeline_stage": "evaluated",
        "scope": "fabrication_review" if fabrication else "continuous_sheet",
        "construction_released": False,
        "objective": diagnostics.get("research_objective"),
        "b0_t": values.get("b0_T"),
        "torque_origin_m": values.get("torque_origin_m"),
        "sheet_physics_scope": "continuous surface current before contouring and routing",
        "relative_field_error": relative,
        "validated_relative_field_error": validation.get("relative_field_error"),
        "field_source": validation.get("source")
        if request.requirements is not None
        else (
            "continuous sheet; independent grid without refit"
            if validation["available"]
            else "continuous sheet; independent validation unavailable"
        ),
        "power_w": values.get("sheet_power_W"),
        "energy_j": values.get("energy_J"),
        "max_current_density_a_per_m": values.get(
            "max_surface_current_density_A_per_m", diagnostics.get("surface_current_density_max_A_per_m")
        ),
        "force_n": values.get("force_N"),
        "torque_nm": values.get("torque_Nm"),
        "wire_length_m": sum(item["length_m"] for item in electrical) if electrical else None,
        "resistance_ohm": electrical[0]["resistance_ohm"] if complete_single else None,
        "inductance_h": electrical[0]["inductance_H"] if complete_single else None,
        "current_a": electrical[0]["current_A"] if complete_single else None,
        "fabricated_power_w": fabrication.electrical_metrics.get("total_joule_power_W") if fabrication else None,
        "fabricated_force_n": fabricated_physics.get("force_N") if fabricated_physics else None,
        "fabricated_torque_nm": fabricated_physics.get("torque_Nm") if fabricated_physics else None,
        "sheet_thickness_m": config.conductor_thickness,
        "conductor_resistivity_ohm_m": config.specific_conductivity_conductor,
        "minimum_spacing_m": _recorded_contour_surface_spacing(result.coil_parts, config),
        "minimum_spacing_scope": "recorded surface-geodesic spacing between unconnected contour centerlines; separate from routed/envelope clearance",
        "contour_components": sum(len(part.contour_lines) for part in result.coil_parts),
        "vertices": len(result.combined_mesh.vertices),
        "triangles": len(result.combined_mesh.faces),
        "solver": diagnostics.get("constrained_solver_name"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    array_file = directory / "scientific_result.npz"
    arrays = _scientific_arrays(result.coil_parts, result.target_field, result.combined_mesh, result.b_field_opt_sf)
    with progress_scope(
        *work_ranges["wire_evidence"], "Retaining evaluated wire-field maps and current provenance", overall=True
    ):
        wire_arrays, wire_evidence = _wire_field_evidence(result)
    arrays.update(wire_arrays)
    arrays.update(requirements_arrays)
    if validation["available"] and request.requirements is None:
        arrays.update(
            validation_points_m=validation["points_m"],
            validation_field_T=validation["field_T"],
            validation_target_T=validation["target_T"],
        )
    if fabricated_physics and fabricated_physics.get("field_T") is not None:
        arrays["fabricated_field_T"] = fabricated_physics["field_T"]
        # Wire-field validation uses the design samples, not the independent
        # sheet grid. Carry its own coordinates instead of implying a grid.
        arrays["fabricated_points_m"] = result.target_field.coords
    if winding_validation and winding_validation["available"]:
        for source, name in (
            ("points_m", "winding_validation_points_m"),
            ("field_T", "winding_validation_field_T"),
            ("target_T", "winding_validation_target_T"),
        ):
            arrays[name] = winding_validation[source]
    if fabrication is not None:
        # Review can intentionally re-realize on the FEM mesh while the native
        # pipeline uses an analytic surface. Keep both physical datasets.
        for index, record in enumerate(fabrication.paths):
            arrays[f"review_conductor_{index}_points_m"] = record.points_m
        for index, plan in enumerate(fabrication.winding_steps):
            if plan.get("electrical_points_m") is not None:
                arrays[f"review_electrical_{index}_points_m"] = np.asarray(plan["electrical_points_m"], dtype=float)
    save_start, save_end = work_ranges["save"]
    save_span = save_end - save_start
    with progress_scope(
        save_start, save_start + 0.6 * save_span, "Compressing scientific field and geometry arrays", overall=True
    ):
        np.savez_compressed(array_file, **arrays)
        artifacts.append({"name": array_file.name, "kind": "npz"})
    with progress_scope(
        save_start + 0.6 * save_span,
        save_start + 0.7 * save_span,
        "Saving solver diagnostics and design report",
        overall=True,
    ):
        atomic_json(directory / "solver_diagnostics.json", diagnostics)
        artifacts.append({"name": "solver_diagnostics.json", "kind": "diagnostics"})
        write_design_report(result, directory / "design_report.md")
        artifacts.append({"name": "design_report.md", "kind": "report"})
    with progress_scope(
        save_start + 0.7 * save_span, save_start + 0.75 * save_span, "Rechecking immutable input files", overall=True
    ):
        verify_input_files(manifest)
    for index, artifact in enumerate(artifacts):
        with progress_scope(
            save_start + save_span * (0.75 + 0.2 * index / len(artifacts)),
            save_start + save_span * (0.75 + 0.2 * (index + 1) / len(artifacts)),
            f"Verifying result artifact {index + 1}/{len(artifacts)}: {artifact['name']}",
            overall=True,
        ):
            artifact["sha256"] = file_digest(directory / artifact["name"])
    validation_summary = {k: v for k, v in validation.items() if k not in {"points_m", "field_T", "target_T"}}
    payload = {
        "name": request.name,
        "software": dict(manifest["software"]),
        "config": request.to_dict()["config"],
        **_target_analysis_recipe(request, result.target_field),
        "summary": summary,
        "constraints": constraints,
        "winding_steps": fabrication.winding_steps if fabrication else [],
        "artifacts": artifacts,
        "warnings": warnings,
        "validation": validation_summary,
        "wire_fields": wire_evidence,
        **({"requirements_review": requirements_review} if requirements_review is not None else {}),
        **({"scene_review": scene_review} if scene_review is not None else {}),
        "pipeline_sweeps": pipeline_sweeps,
        "fabrication": fabrication.summary() if fabrication else None,
        **(
            {
                "winding_validation": {
                    k: v for k, v in winding_validation.items() if k not in {"points_m", "field_T", "target_T"}
                }
            }
            if winding_validation
            else {}
        ),
        **({"engineering_review": engineering_review} if engineering_review else {}),
        **({"dynamic_analysis_review": dynamic_analysis_review} if dynamic_analysis_review is not None else {}),
        "fabricated_physics": {k: v for k, v in fabricated_physics.items() if k != "field_T"}
        if fabricated_physics
        else None,
        "request_sha256": digest,
        "pipeline_stage": "evaluated",
        "checkpoint": "pipeline_checkpoint.cgstage",
        **_project_provenance(request),
        "timing": result.timing_summary,
    }
    with progress_scope(save_start + 0.95 * save_span, save_end, "Saving research result evidence", overall=True):
        atomic_json(directory / "result.json", payload)
    return json_value(payload)


def execute_owned_design(request: ResearchRequest | dict, output_dir: str | Path, *, threads=1) -> dict:
    """Execute after the calling application has acquired ``DirectoryOwner``.

    This lets the application check/reuse immutable completed evidence under
    the same exclusive lock as execution. Ordinary callers use ``run_design``.
    """
    from threadpoolctl import threadpool_info, threadpool_limits
    from .jobs import RuntimePolicy

    RuntimePolicy(threads_per_worker=threads)
    with threadpool_limits(limits=threads):
        result = _run_design(request, output_dir, threads=threads)
        with progress_scope(98, 100, "Finalizing verified runtime and result metadata", overall=True):
            result["runtime"] = {"native_threads_requested": threads, "threadpools": threadpool_info()}
            result["metadata_sha256"] = result_metadata_digest(result)
            atomic_json(Path(output_dir) / "result.json", result)
        return result


def run_design(request: ResearchRequest | dict, output_dir: str | Path, *, threads=1) -> dict:
    """Run a bounded local research design and retain all validation evidence."""
    with DirectoryOwner(output_dir):
        return execute_owned_design(request, output_dir, threads=threads)
