from __future__ import annotations

from projects._shared.object_parameters import object_parameters

import csv
import time
from dataclasses import replace
from typing import Any

from projects._shared.example_utils import PROJECT_ROOT, example_output_dir, parse_example_parameters

from coilgen import CircularBiplanarCoilDesign, result_summary_dict


from projects.general_examples.parameters import BENCHMARK_PERFORMANCE_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS


################################################################################
# USER SETTINGS - EDIT THIS BLOCK FIRST IN SPYDER
################################################################################
PARAMETERS = object_parameters(
    {
        "design": {
            # Small generated biplanar problem. Increase radius/resolution for more
            # representative production benchmarks.
            "target_bz_expression": "y",
            "target_scale": 1.0,
            "target_region_radius_m": 0.08,
            "target_region_resolution": 5,
            "outer_radius_m": 0.25,
            "edge_length_m": 0.055,
            "plane_distance_m": 0.28,
            "contour_levels": 8,
            # Stream-function optimization strategy used by every benchmark case.
            # Choose exactly one active line:
            #   "tikhonov" benchmarks the standard regularized inverse solve.
            #   "target_error_constrained_efficiency" benchmarks the lowest-current
            #   solve at a fixed allowed target-field error.
            "optimization_method": "tikhonov",
            # "optimization_method": "target_error_constrained_efficiency",
            "target_error_tolerance_percent": 5.0,
            "target_error_tolerance_absolute_T": 0.0,
            "target_error_constraint_candidate_count": 61,
            "regularization_method": "fixed",
            "regularization_factor": 40.0,
            # Conductor used for resistance and equivalent round-wire radius in
            # the thin-wire Neumann/Biot-Savart inductance estimate.
            "conductor_width_m": 0.001,
            "conductor_height_m": 0.001,
            "conductor_resistivity_ohm_m": 0.018e-6,
            "skip_postprocessing": True,
            "skip_sweep": True,
            "skip_inductance": False,
            "progress": False,
        },
        "benchmark": {
            "repetitions": 1,
            "cases": (
                {"name": "serial_no_symmetry", "cpu_workers": 1, "symmetry_reduction": "off"},
                {"name": "serial_symmetry_auto", "cpu_workers": 1, "symmetry_reduction": "auto"},
                {"name": "parallel_symmetry_auto", "cpu_workers": 0, "symmetry_reduction": "auto"},
            ),
        },
        "numerics": {
            "gauss_order": 1,
            "parallel_min_work_size": 50000,
            "sensitivity_chunks_per_worker": 4,
            "sensitivity_calculation_method": "vectorized",
            "biot_savart_chunk_size": 512,
            "inductance_chunk_size": 256,
            "parallel_max_chunk_memory_mb": 64.0,
            "optimization_solver": "auto",
            "optimization_solver_residual_tolerance": 1e-8,
            "contour_symmetry_validation": "auto",
            "contour_symmetry_geometry_tolerance": 1e-6,
            "symmetry_reduction_target_tolerance": 1e-6,
        },
        "output": {
            "name": "performance_benchmark",
            "csv_filename": "performance_benchmark.csv",
        },
    }
)


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict[str, Any]:
    return parse_example_parameters(
        PARAMETERS,
        DEFAULT_PARAMETERS,
        example_name="benchmark_performance",
    )


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = get_parameters()
    base_design = build_base_design(parameters)
    rows: list[dict[str, Any]] = []
    repetitions = max(int(parameters["benchmark"]["repetitions"]), 1)
    for case in parameters["benchmark"]["cases"]:
        for repetition in range(1, repetitions + 1):
            design = configure_case(base_design, parameters, case)
            print(f"Benchmark {case['name']} repetition {repetition}/{repetitions}")
            started = time.perf_counter()
            result = design.run(project_root=PROJECT_ROOT)
            wall_time = time.perf_counter() - started
            row = benchmark_row(case, repetition, result, wall_time)
            rows.append(row)
            print(
                f"  wall={wall_time:.3f} s, pipeline={row['pipeline_execution_time_s']:.3f} s, "
                f"unknowns={row['unknown_count']}, symmetry={row['symmetry_reduction_active']}"
            )
    output_dir = example_output_dir(parameters["output"]["name"])
    csv_path = output_dir / parameters["output"]["csv_filename"]
    write_rows(csv_path, rows)
    print(f"Saved performance benchmark CSV: {csv_path}")
    print("Spyder variables available after run: rows, csv_path")
    return rows, csv_path


def build_base_design(parameters: dict[str, Any]) -> CircularBiplanarCoilDesign:
    design = parameters["design"]
    return CircularBiplanarCoilDesign(
        name="performance_benchmark",
        target_bz_expression=design["target_bz_expression"],
        target_scale=design["target_scale"],
        target_region_radius_m=design["target_region_radius_m"],
        target_region_resolution=design["target_region_resolution"],
        outer_radius_m=design["outer_radius_m"],
        edge_length_m=design["edge_length_m"],
        plane_distance_m=design["plane_distance_m"],
        contour_levels=design["contour_levels"],
        optimization_method=design["optimization_method"],
        target_error_tolerance_percent=design["target_error_tolerance_percent"],
        target_error_tolerance_absolute_T=design["target_error_tolerance_absolute_T"],
        target_error_constraint_candidate_count=design["target_error_constraint_candidate_count"],
        regularization_method=design["regularization_method"],
        regularization_factor=design["regularization_factor"],
        conductor_width_m=design["conductor_width_m"],
        conductor_height_m=design["conductor_height_m"],
        conductor_resistivity_ohm_m=design["conductor_resistivity_ohm_m"],
        cpu_workers=1,
        progress=design["progress"],
        skip_postprocessing=design["skip_postprocessing"],
        skip_sweep=design["skip_sweep"],
        skip_inductance=design["skip_inductance"],
        advanced_parameters=_advanced_parameters_for_design(parameters),
    )


def configure_case(
    base_design: CircularBiplanarCoilDesign,
    parameters: dict[str, Any],
    case: dict[str, Any],
) -> CircularBiplanarCoilDesign:
    advanced = _advanced_parameters_for_design(parameters)
    advanced["symmetry_reduction"] = case["symmetry_reduction"]
    return replace(
        base_design,
        name=str(case["name"]),
        cpu_workers=int(case["cpu_workers"]),
        advanced_parameters=advanced,
    )


def _advanced_parameters_for_design(parameters: dict[str, Any]) -> dict[str, Any]:
    design = parameters["design"]
    advanced = dict(parameters["numerics"])
    advanced["sf_opt_method"] = design["optimization_method"]
    advanced["target_error_tolerance_percent"] = design["target_error_tolerance_percent"]
    advanced["target_error_tolerance_absolute_T"] = design["target_error_tolerance_absolute_T"]
    advanced["target_error_constraint_factor_count"] = design["target_error_constraint_candidate_count"]
    return advanced


def benchmark_row(case: dict[str, Any], repetition: int, result: Any, wall_time: float) -> dict[str, Any]:
    summary = result_summary_dict(result)
    diagnostics = result.regularization_diagnostics
    return {
        "case": case["name"],
        "repetition": repetition,
        "cpu_workers": case["cpu_workers"],
        "symmetry_reduction": case["symmetry_reduction"],
        "wall_time_s": float(wall_time),
        "pipeline_execution_time_s": summary["execution_time_s"],
        "unknown_count": int(diagnostics.get("unknown_count", 0)),
        "symmetry_reduction_active": bool(diagnostics.get("symmetry_reduction_active", False)),
        "symmetry_reduction_factor": summary["symmetry_reduction_factor"],
        "contour_count": summary["contour_count"],
        "mean_layout_error_percent": summary.get("mean_rel_error_layout_vs_target_percent", float("nan")),
        "max_layout_error_percent": summary.get("max_rel_error_layout_vs_target_percent", float("nan")),
    }


def write_rows(path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rows, csv_path = main()
