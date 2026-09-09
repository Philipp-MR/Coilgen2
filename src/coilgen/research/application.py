from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np

from ..field_components import field_component_values
from ..pipeline import run_coilgen
from ..progress import ProgressReporter
from ..reporting import write_design_report
from .requests import ResearchRequest
from .storage import DirectoryOwner, atomic_json, file_digest, json_value, request_identity, verify_input_files
from .validation import validate_on_independent_grid, validate_fabricated_constraints


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
    resolved = asdict(config)
    digest, manifest = request_identity(request.to_dict(), resolved)
    manifest["request_sha256"] = digest
    atomic_json(directory / "request.json", request.to_dict())
    atomic_json(directory / "manifest.json", manifest)
    progress = ProgressReporter(label="Research")
    started = time.perf_counter()
    with geometry_acceleration(request.kernel_backend):
        result = run_coilgen(config, project_root=directory)
    diagnostics = result.regularization_diagnostics
    progress.report(86, "Checking an independent field grid")
    validation = validate_on_independent_grid(
        result, resolution=request.validation_resolution, design_margin=request.design_margin
    )
    mask = ~np.asarray(result.target_field.suppressed, bool)
    relative = _relative_field_error(
        result.b_field_opt_sf, result.target_field.b, mask, config.target_field_component, config.b_0_direction
    )
    constraints = [
        {
            "name": r["name"],
            "value": r.get("value"),
            "limit": r.get("limit"),
            "effective_limit": r.get("effective_limit"),
            "unit": r.get("unit", ""),
            "passed": r.get("met"),
        }
        for r in diagnostics.get(
            "research_constraint_diagnostics", diagnostics.get("constrained_solver_contract_constraint_diagnostics", [])
        )
    ]
    validation_passed = validation.get("passed")
    constraints.append(
        {
            "name": "Independent grid field error",
            "value": validation.get("value"),
            "limit": validation.get("limit"),
            "effective_limit": validation.get("effective_limit"),
            "unit": validation.get("unit", "T"),
            "passed": validation_passed,
        }
    )
    warnings = [
        "Magnetostatic sheet model; RF loading, SAR, transient eddy currents and thermal cooling are not modeled."
    ]
    if not validation["available"]:
        warnings.append(validation["reason"])
    if validation_passed is False:
        warnings.append(
            "The independent field grid exceeds the requested tolerance; refine the target sampling or design margin."
        )
    fabrication = None
    fabricated_physics = None
    artifacts = [{"name": "request.json", "kind": "request"}, {"name": "manifest.json", "kind": "provenance"}]
    if request.fabricate:
        progress.report(90, "Conditioning and checking the fabrication assembly")
        fabrication = compile_fabrication(
            result, output_dir=directory / "fabrication", strict=True, kernel_backend=request.kernel_backend
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
        fabricated_physics = validate_fabricated_constraints(result, fabrication, design_margin=request.design_margin)
        constraints.extend(fabricated_physics["checks"])
        warnings.extend(fabricated_physics["unknown_reasons"])
        for path in fabrication.artifacts:
            name = Path(path).relative_to(directory).as_posix()
            artifacts.append({"name": name, "kind": Path(path).suffix.lstrip(".")})
    else:
        warnings.append(
            "Continuous-sheet/contour study: routing, terminals and fabrication readiness have not been established."
        )
    electrical = fabrication.electrical_metrics["bodies"] if fabrication else []
    # Report a complete circuit inductance only for one fully modeled body;
    # adding self-inductances of separate bodies would omit mutual coupling.
    complete_single = len(electrical) == 1 and electrical[0].get("circuit_complete", False)
    physical_ok = bool(diagnostics.get("constrained_solver_contract_met", False))
    accepted = physical_ok and validation_passed is True and all(c["passed"] is True for c in constraints)
    values = diagnostics.get("research_physical_values", diagnostics)
    summary = {
        "accepted": accepted,
        "scope": "fabrication_review" if fabrication else "continuous_sheet",
        "construction_released": False,
        "objective": diagnostics.get("research_objective"),
        "b0_t": values.get("b0_T"),
        "torque_origin_m": values.get("torque_origin_m"),
        "sheet_physics_scope": "continuous surface current before contouring and routing",
        "relative_field_error": relative,
        "validated_relative_field_error": validation.get("relative_field_error"),
        "field_source": "continuous sheet; independent grid without refit"
        if validation["available"]
        else "continuous sheet; independent validation unavailable",
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
        "minimum_spacing_m": None,
        "contour_components": sum(len(part.contour_lines) for part in result.coil_parts),
        "vertices": len(result.combined_mesh.vertices),
        "triangles": len(result.combined_mesh.faces),
        "solver": diagnostics.get("constrained_solver_name"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    array_file = directory / "scientific_result.npz"
    arrays = {
        "vertices_m": result.combined_mesh.vertices,
        "faces": result.combined_mesh.faces,
        "stream_function_A": result.combined_mesh.stream_function,
        "target_points_m": result.target_field.coords,
        "target_field_T": result.target_field.b,
        "sheet_field_T": result.b_field_opt_sf,
    }
    if validation["available"]:
        arrays.update(
            validation_points_m=validation["points_m"],
            validation_field_T=validation["field_T"],
            validation_target_T=validation["target_T"],
        )
    if fabricated_physics and fabricated_physics.get("field_T") is not None:
        arrays["fabricated_field_T"] = fabricated_physics["field_T"]
    np.savez_compressed(array_file, **arrays)
    artifacts.append({"name": array_file.name, "kind": "npz"})
    atomic_json(directory / "solver_diagnostics.json", diagnostics)
    artifacts.append({"name": "solver_diagnostics.json", "kind": "diagnostics"})
    write_design_report(result, directory / "design_report.md")
    artifacts.append({"name": "design_report.md", "kind": "report"})
    verify_input_files(manifest)
    for artifact in artifacts:
        artifact["sha256"] = file_digest(directory / artifact["name"])
    validation_summary = {k: v for k, v in validation.items() if k not in {"points_m", "field_T", "target_T"}}
    payload = {
        "name": request.name,
        "config": request.to_dict()["config"],
        "summary": summary,
        "constraints": constraints,
        "winding_steps": fabrication.winding_steps if fabrication else [],
        "artifacts": artifacts,
        "warnings": warnings,
        "validation": validation_summary,
        "fabrication": fabrication.summary() if fabrication else None,
        "fabricated_physics": {k: v for k, v in fabricated_physics.items() if k != "field_T"}
        if fabricated_physics
        else None,
        "request_sha256": digest,
        "timing": result.timing_summary,
    }
    atomic_json(directory / "result.json", payload)
    progress.report(100, "Research result and evidence saved")
    return json_value(payload)


def run_design(request: ResearchRequest | dict, output_dir: str | Path, *, threads=1) -> dict:
    """Run a bounded local research design and retain all validation evidence."""
    from threadpoolctl import threadpool_info, threadpool_limits
    from .jobs import RuntimePolicy

    RuntimePolicy(threads_per_worker=threads)
    with DirectoryOwner(output_dir), threadpool_limits(limits=threads):
        result = _run_design(request, output_dir, threads=threads)
        result["runtime"] = {"native_threads_requested": threads, "threadpools": threadpool_info()}
        atomic_json(Path(output_dir) / "result.json", result)
        return result
