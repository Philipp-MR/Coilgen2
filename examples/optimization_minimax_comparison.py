"""Compare physical FEM objectives on one fixed cylinder and field-error budget."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

# Import numerical libraries only after selecting native threads and this checkout.
import numpy as np  # noqa: E402

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
    from contextlib import nullcontext

    def threadpool_limits(limits):
        return nullcontext()

    def threadpool_info():
        return {
            "control": "process environment set before NumPy import; threadpoolctl unavailable",
            "threads_requested": 1,
        }


import coilgen  # noqa: E402
from coilgen import CoilGenConfig, TargetField, run_coilgen  # noqa: E402
from coilgen.physics import calculate_sensitivity_matrix  # noqa: E402


def independent_face_currents(part):
    triangles = part.mesh.vertices[part.mesh.faces]
    double_area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(double_area_vectors, axis=1)
    normals = double_area_vectors / double_areas[:, None]
    opposite_edges = np.stack(
        (triangles[:, 2] - triangles[:, 1], triangles[:, 0] - triangles[:, 2], triangles[:, 1] - triangles[:, 0]),
        axis=1,
    )
    gradients = np.cross(normals[:, None, :], opposite_edges) / double_areas[:, None, None]
    stream_gradient = np.einsum("fva,fv->fa", gradients, part.stream_function[part.mesh.faces])
    return np.cross(stream_gradient, normals), 0.5 * double_areas


output = root / "tmp" / "optimization_review"
output.mkdir(parents=True, exist_ok=True)
base = CoilGenConfig(
    field_shape_function="y",
    target_gradient_strength=0.02,
    target_region_radius=0.06,
    target_region_resolution=11,
    coil_mesh_file="create cylinder mesh",
    cylinder_mesh_parameter_list=(0.3, 0.16, 32, 16, 1, 0, 0, 0),
    levels=8,
    pot_offset_factor=0.25,
    min_point_loop_number=3,
    gauss_order=2,
    sf_opt_method="constrained_quadratic",
    constrained_optimization_solver="clarabel",
    constrained_optimization_tolerance=1e-9,
    constrained_optimization_max_iterations=500,
    constrained_optimization_threads=1,
    constrained_quadratic_fallback_policy="strict",
    target_error_constraint_mode="separate",
    target_error_constraint_metric="norm",
    target_error_tolerance_percent=3.0,
    target_error_tolerance_absolute_T=0.0,
    target_error_constraint_acceptance_absolute_T=0.0,
    target_error_constraint_acceptance_tolerance=1e-7,
    target_error_auto_retry_enabled=False,
    stream_function_smoothness_factor=0.0,
    turn_spacing_regularization_factor=0.0,
    skip_postprocessing=True,
    skip_sweep=True,
    skip_inductance_calculation=True,
    skip_calculation_min_winding_distance=True,
    skip_unconnected_contour_wire_distance=True,
    plot_flag=False,
    save_stl_flag=False,
    progress_flag=False,
    parallel_worker_count=1,
    symmetry_reduction="off",
    retain_heavy_fields=True,
)
report = {
    "coilgen_import": coilgen.__file__,
    "numpy_version": np.__version__,
    "comparison": "same 3% relative weighted Bz error budget; continuous-sheet FEM only",
    "cases": [],
}
axis = np.linspace(-0.055, 0.055, 12)
validation_coordinates = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
validation_coordinates = validation_coordinates[np.linalg.norm(validation_coordinates, axis=1) <= 0.06]
validation_field = np.zeros_like(validation_coordinates)
validation_field[:, 2] = 0.02 * validation_coordinates[:, 1]
validation_target = TargetField(
    coords=validation_coordinates,
    b=validation_field,
    weights=np.ones(len(validation_coordinates)),
    suppressed=np.zeros(len(validation_coordinates), dtype=bool),
    target_field_group_inds=np.ones(len(validation_coordinates), dtype=int),
)
with threadpool_limits(limits=1):
    for mode in ("balanced", "minimax_current_density"):
        config = replace(base, constrained_quadratic_objective_mode=mode)
        (output / f"cylinder_{mode}_config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
        started = time.perf_counter()
        result = run_coilgen(config)
        elapsed = time.perf_counter() - started
        field = np.zeros_like(result.target_field.b)
        powers = []
        matrix_powers = []
        peak_currents = []
        current_operator_errors = []
        for index, part in enumerate(result.coil_parts):
            current, areas = independent_face_currents(part)
            magnitudes = np.linalg.norm(current, axis=1)
            peak_currents.append(float(np.max(magnitudes)))
            powers.append(
                float(config.specific_conductivity_conductor / config.conductor_thickness * (areas @ magnitudes**2))
            )
            matrix_powers.append(float(part.stream_function @ part.resistance_matrix @ part.stream_function))
            current_operator_errors.append(float(np.max(np.linalg.norm(current - part.current_density, axis=1))))
            field += np.einsum("atv,v->ta", part.sensitivity_matrix, part.stream_function)
            np.savez_compressed(
                output / f"cylinder_{mode}_part{index + 1}.npz",
                vertices=part.mesh.vertices,
                faces=part.mesh.faces,
                stream_function_A=part.stream_function,
                current_density_A_per_m=current,
                face_area_m2=areas,
                target_coordinates_m=result.target_field.coords,
                target_field_T=result.target_field.b,
                surface_field_T=field,
            )
        mask = ~result.target_field.suppressed
        weights = result.target_field.weights[mask]
        error = field[mask, 2] - result.target_field.b[mask, 2]
        error_norm = float(np.linalg.norm(np.sqrt(weights) * error))
        target_norm = float(np.linalg.norm(np.sqrt(weights) * result.target_field.b[mask, 2]))
        diagnostics = result.regularization_diagnostics
        validation_parts = [replace(part, sensitivity_matrix=None) for part in result.coil_parts]
        calculate_sensitivity_matrix(validation_parts, validation_target, config)
        validation_prediction = sum(
            np.einsum("atv,v->ta", part.sensitivity_matrix, part.stream_function) for part in validation_parts
        )
        validation_error = validation_prediction[:, 2] - validation_field[:, 2]
        np.savez_compressed(
            output / f"cylinder_{mode}_validation.npz",
            coordinates_m=validation_coordinates,
            target_field_T=validation_field,
            surface_field_T=validation_prediction,
        )
        row = {
            "mode": mode,
            "wall_seconds": elapsed,
            "vertices": sum(part.mesh.n_vertices for part in result.coil_parts),
            "faces": sum(part.mesh.n_faces for part in result.coil_parts),
            "unknowns": diagnostics.get("unknown_count"),
            "target_points": len(weights),
            "peak_surface_current_density_A_per_m": max(peak_currents),
            "sheet_power_W": sum(powers),
            "matrix_sheet_power_W": sum(matrix_powers),
            "independent_K_operator_max_error_A_per_m": max(current_operator_errors),
            "field_relative_error_percent": 100.0 * error_norm / target_norm,
            "field_rms_error_T": error_norm / np.sqrt(float(np.sum(weights))),
            "field_peak_error_T": float(np.max(np.abs(error))),
            "independent_validation_points": len(validation_coordinates),
            "independent_validation_relative_error_percent": float(
                100.0 * np.linalg.norm(validation_error) / np.linalg.norm(validation_field[:, 2])
            ),
            "independent_validation_rms_error_T": float(np.sqrt(np.mean(validation_error**2))),
            "independent_validation_peak_error_T": float(np.max(np.abs(validation_error))),
            "conic_status": diagnostics.get("constrained_solver_status"),
            "conic_solver": diagnostics.get("constrained_solver_name"),
            "conic_contract_met": diagnostics.get("constrained_solver_contract_met"),
            "solver_objective": diagnostics.get("constrained_solver_objective_value"),
            "solver_objective_unit": diagnostics.get("constrained_solver_objective_unit"),
            "minimax_epigraph_A_per_m": diagnostics.get("minimax_current_density_epigraph_A_per_m"),
            "density_scaling_applied": diagnostics.get("surface_current_density_scaling_applied"),
            "contours": sum(len(part.contour_lines) for part in result.coil_parts),
            "timing_summary": result.timing_summary,
        }
        report["cases"].append(row)
        report["native_threadpools"] = threadpool_info()
        (output / "cylinder_minimax_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (output / f"cylinder_{mode}_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
        )
        assert row["conic_contract_met"], row
        assert row["field_relative_error_percent"] <= 3.0 * (1.0 + 1e-7), row
        assert np.isclose(row["sheet_power_W"], row["matrix_sheet_power_W"], rtol=1e-10), row
        print(json.dumps(row, indent=2), flush=True)
    balanced, minimax = report["cases"]
    report["peak_current_reduction_percent"] = 100.0 * (
        1.0 - minimax["peak_surface_current_density_A_per_m"] / balanced["peak_surface_current_density_A_per_m"]
    )
    report["sheet_power_increase_percent"] = 100.0 * (minimax["sheet_power_W"] / balanced["sheet_power_W"] - 1.0)
    (output / "cylinder_minimax_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"cases", "native_threadpools"}}, indent=2
        )
    )
