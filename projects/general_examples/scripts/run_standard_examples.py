from __future__ import annotations

from projects._shared.object_parameters import object_parameters

from pathlib import Path
from typing import Any

import numpy as np

from projects._shared.example_utils import (
    PROJECT_ROOT,
    example_output_dir,
    open_project_plot_book_after_run,
    parse_example_parameters,
)

from coilgen import (
    CoilGenConfig,
    CoilGenConfigDesign,
    CoilGenResult,
    CoilProject,
    contour_orientation_diagnostics,
    save_interconnect_debug_isometric,
    save_route_diagnostics,
    save_within_group_cut_choices,
)


from projects.general_examples.parameters import RUN_STANDARD_EXAMPLES_DEFAULT_PARAMETERS as DEFAULT_PARAMETERS


################################################################################
# USER SETTINGS
################################################################################
# Edit this object in Spyder. Leave run.selected_examples empty to run all standard examples.
PARAMETERS = object_parameters(
    {
        "run": {
            # Include the denser generated cylinder/planar/split-bore examples.
            "dense_examples": False,
            # Tuple of example names to run; empty means all selected standard/dense examples.
            "selected_examples": (),
        },
        "output": {
            # Output folder below outputs/.
            "name": "standard_examples",
            # Save the full SVG gallery for every generated example.
            "save_svg_galleries": True,
        },
        "plots": {
            # Open active Matplotlib windows for the first generated result.
            "show_active_for_first_result": False,
            # Close figures from previous runs before plotting the first result.
            "close_previous": True,
            # "book" opens one paged window; "windows" opens separate interactive figures.
            "mode": "book",
            # Plot groups to include if active plotting is enabled.
            "include": ("wire", "contour_groups", "field_pointclouds", "error_maps"),
        },
        "expectations": {
            # Regression checks applied by validate_generated_example_result().
            "default": {
                "min_contours": 1,
                "min_groups": 1,
                "max_route_failures": 0,
                "max_degenerate_opening_cuts": 0,
                "max_suspicious_connectors": 0,
                "max_clearance_fallbacks": 0,
                "require_route_quality_ok": True,
            }
        },
        "base_config": {
            # Shared CoilGenConfig values for standard generated examples.
            "target_region_radius": 0.08,
            "target_region_resolution": 5,
            "levels": 8,
            "pot_offset_factor": 0.25,
            "min_point_loop_number": 4,
            "gauss_order": 1,
            # Stream-function optimization strategy shared by generated examples.
            # Choose exactly one active line:
            #   "tikhonov" uses the regularization scan below.
            #   "target_error_constrained_efficiency" compares examples at a fixed
            #   allowed target-field error and chooses the lowest-current solution.
            "sf_opt_method": "tikhonov",
            # "sf_opt_method": "target_error_constrained_efficiency",
            "stream_function_cost_function": "field_error_plus_resistance",
            # L-curve regularization scan; tikonov_reg_factor is the scan center.
            "tikonov_reg_factor": 50.0,
            "regularization_selection_method": "l_curve",
            # Use exact normal-equation candidate solves and smooth curvature to avoid underregularized local knees.
            "l_curve_scan_method": "normal_equations",
            "l_curve_curvature_smoothing_window": 5,
            "l_curve_factor_count": 21,
            "l_curve_factor_min_multiplier": 1e-4,
            "l_curve_factor_max_multiplier": 1e4,
            "l_curve_regularization_factors": (),
            # Parallel CPU usage: 0 uses all logical cores, 1 is serial, -1 keeps one core free.
            "parallel_worker_count": 0,
            # Minimum estimated operation count before parallel workers are used.
            "parallel_min_work_size": 50000,
            # Sensitivity-matrix chunking; higher values create more, smaller worker tasks.
            "sensitivity_chunks_per_worker": 4,
            "sensitivity_calculation_method": "vectorized",
            # Segment chunk sizes for Biot-Savart field evaluation and Neumann inductance.
            "biot_savart_chunk_size": 512,
            "inductance_chunk_size": 256,
            "wire_distance_method": "auto",
            "wire_distance_chunk_size": 512,
            "wire_distance_spatial_min_segments": 3000,
            # Approximate memory cap per numerical chunk.
            "parallel_max_chunk_memory_mb": 64.0,
            "optimization_solver": "auto",
            "optimization_solver_residual_tolerance": 1e-8,
            # Used only by sf_opt_method="target_error_constrained_efficiency".
            # 5.0 means at most 5 percent weighted target-field residual.
            "target_error_tolerance_percent": 5.0,
            "target_error_tolerance_absolute_T": 0.0,
            "target_error_constraint_factor_count": 61,
            # Print progress percentages for long calculation stages.
            "progress_flag": True,
            "progress_min_interval_seconds": 1.0,
            "progress_percent_step": 5.0,
            "skip_postprocessing": False,
            "plot_flag": False,
            "save_stl_flag": False,
            "skip_sweep": True,
            "skip_inductance_calculation": False,
            "inductance_source": "auto",
            # Rectangular copper conductor for resistance and equivalent round-wire
            # radius in the Neumann/Biot-Savart inductance estimate.
            "conductor_cross_section_width": 0.001,
            "conductor_cross_section_height": 0.001,
            "specific_conductivity_conductor": 0.018e-6,
            "skip_calculation_min_winding_distance": False,
        },
        "dense_base_overrides": {
            # Overrides applied to dense_examples on top of base_config.
            "target_region_resolution": 6,
            "levels": 12,
            "min_point_loop_number": 5,
            "within_group_cut_candidate_count": 12,
        },
        "standard_examples": {
            # Each entry defines one generated case: target field, mesh builder, mesh parameters, and connector method.
            "cylinder_ygradient_direct3d": {
                "field_shape_function": "y",
                "coil_mesh_file": "create cylinder mesh",
                "mesh_parameter_key": "cylinder_mesh_parameter_list",
                "mesh_parameters": (0.5, 0.25, 12, 6, 1, 0, 0, 0),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {},
            },
            "cylinder_ygradient_geodesic": {
                "field_shape_function": "y",
                "coil_mesh_file": "create cylinder mesh",
                "mesh_parameter_key": "cylinder_mesh_parameter_list",
                "mesh_parameters": (0.5, 0.25, 12, 6, 1, 0, 0, 0),
                "inter_group_connection_path_method": "geodesic",
                "config_overrides": {},
            },
            "cylinder_xgradient_direct3d": {
                "field_shape_function": "x",
                "coil_mesh_file": "create cylinder mesh",
                "mesh_parameter_key": "cylinder_mesh_parameter_list",
                "mesh_parameters": (0.5, 0.25, 12, 6, 1, 0, 0, 0),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {},
            },
            "cylinder_zgradient_direct3d": {
                "field_shape_function": "z",
                "coil_mesh_file": "create cylinder mesh",
                "mesh_parameter_key": "cylinder_mesh_parameter_list",
                "mesh_parameters": (0.5, 0.25, 12, 6, 1, 0, 0, 0),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {},
            },
            "planary_xgradient_direct3d": {
                "field_shape_function": "x",
                "coil_mesh_file": "create planary mesh",
                "mesh_parameter_key": "planar_mesh_parameter_list",
                "mesh_parameters": (0.25, 0.25, 18, 18, 1, 0, 0, 0, 0, 0, 0),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {"surface_is_cylinder_flag": False},
            },
            "planary_xgradient_geodesic": {
                "field_shape_function": "x",
                "coil_mesh_file": "create planary mesh",
                "mesh_parameter_key": "planar_mesh_parameter_list",
                "mesh_parameters": (0.25, 0.25, 18, 18, 1, 0, 0, 0, 0, 0, 0),
                "inter_group_connection_path_method": "geodesic",
                "config_overrides": {"surface_is_cylinder_flag": False},
            },
            "planary_ygradient_direct3d": {
                "field_shape_function": "y",
                "coil_mesh_file": "create planary mesh",
                "mesh_parameter_key": "planar_mesh_parameter_list",
                "mesh_parameters": (0.25, 0.25, 18, 18, 1, 0, 0, 0, 0, 0, 0),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {"surface_is_cylinder_flag": False},
            },
            "biplanar_ygradient_direct3d": {
                "field_shape_function": "y",
                "coil_mesh_file": "create bi-planary mesh",
                "mesh_parameter_key": "biplanar_mesh_parameter_list",
                "mesh_parameters": (0.25, 0.25, 10, 10, 1, 0, 0, 0, 0, 0, 0.2),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {"surface_is_cylinder_flag": False},
            },
        },
        "dense_examples": {
            # Dense cases use dense_base_overrides unless an entry adds config_overrides.
            "dense_cylinder_ygradient_direct3d": {
                "field_shape_function": "y",
                "coil_mesh_file": "create cylinder mesh",
                "mesh_parameter_key": "cylinder_mesh_parameter_list",
                "mesh_parameters": (0.5, 0.25, 20, 10, 1, 0, 0, 0),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {},
            },
            "dense_planary_xgradient_direct3d": {
                "field_shape_function": "x",
                "coil_mesh_file": "create planary mesh",
                "mesh_parameter_key": "planar_mesh_parameter_list",
                "mesh_parameters": (0.25, 0.25, 28, 28, 1, 0, 0, 0, 0, 0, 0),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {"surface_is_cylinder_flag": False},
            },
            "dense_planary_ygradient_direct3d": {
                "field_shape_function": "y",
                "coil_mesh_file": "create planary mesh",
                "mesh_parameter_key": "planar_mesh_parameter_list",
                "mesh_parameters": (0.25, 0.25, 28, 28, 1, 0, 0, 0, 0, 0, 0),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {"surface_is_cylinder_flag": False},
            },
            "halbach_cylinder_ygradient_direct3d": {
                "field_shape_function": "y",
                "coil_mesh_file": "create cylinder mesh",
                "mesh_parameter_key": "cylinder_mesh_parameter_list",
                "mesh_parameters": (0.5, 0.25, 20, 10, 1, 0, 0, float(np.pi / 2.0)),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {},
            },
            "split_bore_flange_ygradient_direct3d": {
                "field_shape_function": "y",
                "coil_mesh_file": "create flanged split bore mesh",
                "mesh_parameter_key": "flanged_split_bore_mesh_parameter_list",
                "mesh_parameters": (0.12, 0.2, 0.28, 0.08, 0.2),
                "inter_group_connection_path_method": "direct_3d",
                "config_overrides": {
                    "surface_is_cylinder_flag": True,
                    "target_region_radius": 0.05,
                    "target_region_resolution": 5,
                    "levels": 4,
                    "pot_offset_factor": 0.25,
                    "min_point_loop_number": 3,
                    "gauss_order": 1,
                    "tikonov_reg_factor": 20.0,
                },
            },
        },
    }
)


STANDARD_EXAMPLE_EXPECTATIONS: dict[str, dict[str, float | int | bool]] = DEFAULT_PARAMETERS["expectations"]


################################################################################
# PARAMETER PARSING
################################################################################


def get_parameters() -> dict:
    return parse_example_parameters(PARAMETERS, DEFAULT_PARAMETERS, example_name="run_standard_examples")


################################################################################
# SCRIPT ENTRY POINT
################################################################################


def main():
    parameters = get_parameters()
    output_root = example_output_dir(parameters["output"]["name"])
    failures: list[str] = []
    results: dict[str, CoilGenResult] = {}
    project_runs: dict[str, object] = {}
    projects: dict[str, CoilProject] = {}
    figures = []
    selected = set(parameters["run"]["selected_examples"])
    configs = generated_example_configs(include_dense=parameters["run"]["dense_examples"], parameters=parameters)
    if selected:
        configs = [(name, config) for name, config in configs if name in selected]
    for example_index, (name, config) in enumerate(configs):
        print(f"Running {name}")
        output_dir = output_root / name
        project = CoilProject(
            name,
            output_dir=output_dir,
            project_root=PROJECT_ROOT,
            save_reports=False,
            save_gallery=False,
        )
        project.channel(
            "main",
            design=CoilGenConfigDesign(name=name, config=config),
            save_gallery=bool(parameters["output"]["save_svg_galleries"]),
        )
        project_run = project.solve()
        result = project_run.channel_runs[0].result
        results[name] = result
        projects[name] = project
        project_runs[name] = project_run
        files = list(project_run.files)
        example_failures, example_warnings = validate_generated_example_result(
            name,
            result,
            expectations=parameters["expectations"],
        )
        failures.extend(f"{name}: {failure}" for failure in example_failures)
        _print_example_summary(name, result, files, example_warnings)
        if example_index == 0:
            plots = parameters["plots"]
            figures = (
                project.plot(
                    include=plots["include"],
                    close_previous=plots["close_previous"],
                    mode=plots.get("mode", "book"),
                )
                if plots["show_active_for_first_result"]
                else []
            )

    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        raise SystemExit(1)
    return results, projects, project_runs, figures


def generated_example_configs(
    *,
    include_dense: bool = False,
    parameters: dict | None = None,
) -> list[tuple[str, CoilGenConfig]]:
    configs = standard_example_configs(parameters=parameters)
    if include_dense:
        configs.extend(dense_generated_example_configs(parameters=parameters))
    return configs


def standard_example_configs(parameters: dict | None = None) -> list[tuple[str, CoilGenConfig]]:
    parameters = get_parameters() if parameters is None else parameters
    return _configs_from_definitions(parameters["standard_examples"], _base_config(parameters))


def dense_generated_example_configs(parameters: dict | None = None) -> list[tuple[str, CoilGenConfig]]:
    parameters = get_parameters() if parameters is None else parameters
    return _configs_from_definitions(parameters["dense_examples"], _dense_base_config(parameters))


def validate_generated_example_result(
    name: str,
    result: CoilGenResult,
    expectations: dict[str, dict[str, float | int | bool]] | None = None,
) -> tuple[list[str], list[str]]:
    expectations = get_parameters()["expectations"] if expectations is None else expectations
    expectation = _example_expectation(name, expectations)
    failures: list[str] = []
    warnings: list[str] = []

    for part_ind, part in enumerate(result.coil_parts, start=1):
        diagnostics = part.route_diagnostics
        label = f"part {part_ind}"

        if part.wire_path is None or len(part.wire_path.v) < 2:
            failures.append(f"{label}: missing connected wire path")
        if len(part.contour_lines) < int(expectation["min_contours"]):
            failures.append(f"{label}: contour count {len(part.contour_lines)} below expectation")
        if len(part.groups) < int(expectation["min_groups"]):
            failures.append(f"{label}: group count {len(part.groups)} below expectation")
        if not np.isfinite(float(diagnostics.get("length", 0.0))):
            failures.append(f"{label}: non-finite route length")
        if "closed_uv" in diagnostics:
            failures.append(f"{label}: route diagnostics still contain UV state")
        for contour_row in contour_orientation_diagnostics(part):
            if not bool(contour_row["chirality_defined"]):
                warnings.append(
                    f"{label}: contour {contour_row['contour_index']} has undefined chirality "
                    f"(surface-normal coherence={float(contour_row['surface_normal_coherence']):.3g})"
                )

        _check_max_diagnostic(
            failures,
            label,
            diagnostics,
            "route_quality_failure_count",
            int(expectation["max_route_failures"]),
        )
        _check_max_diagnostic(
            failures,
            label,
            diagnostics,
            "degenerate_opening_cut_count",
            int(expectation["max_degenerate_opening_cuts"]),
        )
        _check_max_diagnostic(
            failures,
            label,
            diagnostics,
            "suspicious_inter_group_connector_count",
            int(expectation["max_suspicious_connectors"]),
        )
        _check_max_diagnostic(
            failures,
            label,
            diagnostics,
            "inter_group_clearance_fallback_count",
            int(expectation["max_clearance_fallbacks"]),
        )
        if bool(expectation["require_route_quality_ok"]) and not bool(diagnostics.get("route_quality_ok", False)):
            failures.append(f"{label}: route_quality_ok is false")

        warning_count = int(diagnostics.get("route_quality_warning_count", 0))
        if warning_count:
            warnings.append(
                f"{label}: {warning_count} route warning(s), status={diagnostics.get('route_quality_status', 'unknown')}"
            )

        if name.endswith("_geodesic") and int(diagnostics.get("inter_group_cut_count", 0)) > 0:
            if int(diagnostics.get("inter_group_geodesic_connector_count", 0)) == 0:
                failures.append(f"{label}: geodesic example has no geodesic connector")
        if name.endswith("_direct3d") and int(diagnostics.get("inter_group_cut_count", 0)) > 0:
            if int(diagnostics.get("inter_group_direct_connector_count", 0)) != int(
                diagnostics.get("inter_group_cut_count", 0)
            ):
                failures.append(f"{label}: direct-3D example did not use direct connectors for every inter-group cut")

    return failures, warnings


def save_generated_example_outputs(
    result: CoilGenResult,
    output_dir: str | Path,
    *,
    prefix: str = "generated",
) -> list[Path]:
    """Save compact debug outputs for a generated standard example."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    files.extend(save_within_group_cut_choices(result, output, prefix=prefix))
    for part_ind, part in enumerate(result.coil_parts, start=1):
        files.append(
            save_interconnect_debug_isometric(
                part,
                output / f"{prefix}_part{part_ind}_interconnect_debug_iso.svg",
            )
        )
    files.extend(save_route_diagnostics(result, output, prefix=prefix))
    return files


def _configs_from_definitions(
    definitions: dict[str, dict[str, Any]],
    base_config: dict[str, Any],
) -> list[tuple[str, CoilGenConfig]]:
    return [(name, _config_from_definition(definition, base_config)) for name, definition in definitions.items()]


def _config_from_definition(definition: dict[str, Any], base_config: dict[str, Any]) -> CoilGenConfig:
    config_values = dict(base_config)
    config_values.update(definition.get("config_overrides", {}))
    config_values.update(
        {
            "field_shape_function": definition["field_shape_function"],
            "coil_mesh_file": definition["coil_mesh_file"],
            "inter_group_connection_path_method": definition["inter_group_connection_path_method"],
            definition["mesh_parameter_key"]: definition["mesh_parameters"],
        }
    )
    return CoilGenConfig(**config_values)


def _base_config(parameters: dict) -> dict[str, Any]:
    return dict(parameters["base_config"])


def _dense_base_config(parameters: dict) -> dict[str, Any]:
    base = _base_config(parameters)
    base.update(parameters["dense_base_overrides"])
    return base


def _print_example_summary(name: str, result: CoilGenResult, files: list[Path], warnings: list[str]) -> None:
    part = result.coil_parts[0]
    diagnostics = part.route_diagnostics
    print(
        "  "
        f"status={diagnostics.get('route_quality_status', 'unknown')} "
        f"contours={len(part.contour_lines)} "
        f"groups={len(part.groups)} "
        f"wire_points={len(part.wire_path.v) if part.wire_path is not None else 0} "
        f"cuts={diagnostics['inter_group_cut_count']} "
        f"geodesic={diagnostics['inter_group_geodesic_connector_count']} "
        f"direct={diagnostics['inter_group_direct_connector_count']} "
        f"length={diagnostics['length']:.6g} "
        f"min_wire={diagnostics['min_wire_wire_distance']:.6g} "
        f"failures={diagnostics.get('route_quality_failure_count', 0)} "
        f"warnings={diagnostics.get('route_quality_warning_count', 0)} "
        f"bad_cuts={diagnostics['degenerate_opening_cut_count']} "
        f"max_connector_ratio={diagnostics['max_inter_group_connector_ratio']:.3g} "
        f"suspicious={diagnostics['suspicious_inter_group_connector_count']} "
        f"clearance_fallback={diagnostics['inter_group_clearance_fallback_count']} "
        f"cut_candidates={diagnostics.get('max_within_group_cut_candidate_count', 0):.0f} "
        f"reg_factor={result.regularization_factor:.3g} "
        f"files={len(files)}"
    )
    for warning in warnings:
        print(f"  WARN {name}: {warning}")


def _check_max_diagnostic(
    failures: list[str],
    label: str,
    diagnostics: dict[str, Any],
    key: str,
    maximum: int,
) -> None:
    value = int(diagnostics.get(key, 0))
    if value > maximum:
        failures.append(f"{label}: {key}={value} exceeds {maximum}")


def _example_expectation(
    name: str,
    expectations: dict[str, dict[str, float | int | bool]],
) -> dict[str, float | int | bool]:
    expectation = dict(expectations["default"])
    expectation.update(expectations.get(name, {}))
    return expectation


if __name__ == "__main__":
    results, projects, project_runs, figures = main()
    first_project_run = next(iter(project_runs.values()), None)
    plot_book = open_project_plot_book_after_run(
        first_project_run, figures=figures, include=get_parameters()["plots"].get("include")
    )
