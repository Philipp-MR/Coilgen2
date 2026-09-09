from __future__ import annotations

import csv
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterable

from projects._shared.example_utils import PROJECT_ROOT, example_output_dir, native_thread_budget

from coilgen import CoilGenConfig, result_summary_dict, run_coilgen


def base_benchmark_config(**overrides: Any) -> CoilGenConfig:
    values: dict[str, Any] = {
        "field_shape_function": "y",
        "target_gradient_strength": 1.0,
        "target_region_radius": 0.06,
        "target_region_resolution": 5,
        "coil_mesh_file": "create cylinder mesh",
        "cylinder_mesh_parameter_list": (0.3, 0.16, 10, 6, 1, 0, 0, 0),
        "levels": 4,
        "pot_offset_factor": 0.25,
        "min_point_loop_number": 3,
        "gauss_order": 1,
        "sf_opt_method": "tikhonov",
        "regularization_selection_method": "fixed",
        "tikonov_reg_factor": 40.0,
        "skip_postprocessing": True,
        "skip_sweep": True,
        "skip_inductance_calculation": True,
        "plot_flag": False,
        "save_stl_flag": False,
        "progress_flag": False,
        "low_memory_mode": True,
        "retain_heavy_fields": False,
        "parallel_worker_count": 1,
    }
    values.update(overrides)
    return CoilGenConfig(**values)


def run_timed_config(name: str, config: CoilGenConfig) -> dict[str, Any]:
    tracemalloc.start()
    start = time.perf_counter()
    try:
        with native_thread_budget():
            result = run_coilgen(config, project_root=PROJECT_ROOT)
        elapsed = time.perf_counter() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    summary = result_summary_dict(result)
    return {
        "case": name,
        "elapsed_s": elapsed,
        "tracemalloc_peak_MiB": peak_bytes / (1024.0 * 1024.0),
        "target_point_count": summary.get("target_point_count", ""),
        "mesh_vertex_count": summary.get("sensitivity_mesh_vertex_count", ""),
        "sensitivity_unknown_count": summary.get("sensitivity_unknown_count", ""),
        "sensitivity_matrix_MiB": summary.get("sensitivity_matrix_MiB", ""),
        "contour_count": summary.get("contour_count", ""),
        "layout_needed_current_A": summary.get("layout_needed_current_A", ""),
        "layout_mean_rel_error_percent": summary.get("layout_mean_rel_error_percent", ""),
        "execution_time_s": summary.get("execution_time_s", ""),
    }


def write_benchmark_outputs(name: str, rows: Iterable[dict[str, Any]], *, output_name: str | None = None) -> Path:
    output_dir = example_output_dir(output_name or name)
    csv_path = output_dir / f"{name}.csv"
    row_list = list(rows)
    keys: list[str] = []
    for row in row_list:
        for key in row:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(row_list)
    report_path = output_dir / f"{name}_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {name}",
                "",
                "This benchmark is a lightweight scaling screen. It records wall time, Python peak memory,",
                "and the sensitivity-matrix size reported by CoilGen.",
                "",
                f"Rows: {len(row_list)}",
                f"CSV: `{csv_path.name}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved benchmark CSV: {csv_path}")
    print(f"Saved benchmark report: {report_path}")
    return csv_path
