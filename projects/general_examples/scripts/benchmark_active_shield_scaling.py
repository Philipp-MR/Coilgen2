from __future__ import annotations

from projects._shared.object_parameters import object_parameters

from projects.general_examples.scripts._benchmark_scaling_common import (
    base_benchmark_config,
    run_timed_config,
    write_benchmark_outputs,
)


PARAMETERS = object_parameters(
    {
        "output_name": "benchmark_active_shield_scaling",
        "surface_cases": (
            {"name": "primary_8x8_shield_8x8", "primary_radius_m": 0.22, "shield_radius_m": 0.28, "edge_m": 0.08},
            {"name": "primary_10x10_shield_10x10", "primary_radius_m": 0.28, "shield_radius_m": 0.34, "edge_m": 0.07},
            {"name": "primary_12x12_shield_12x12", "primary_radius_m": 0.34, "shield_radius_m": 0.42, "edge_m": 0.06},
        ),
        "plane_distance_m": 0.35,
        "shield_gap_m": 0.05,
        "target_region_resolution": 5,
    }
)


def main() -> list[dict]:
    rows = []
    for case in PARAMETERS["surface_cases"]:
        config = base_benchmark_config(
            coil_mesh_file="create shielded bi-planary mesh",
            surface_is_cylinder_flag=False,
            target_region_resolution=PARAMETERS["target_region_resolution"],
            shielded_circular_biplanar_mesh_parameter_list=(
                case["primary_radius_m"],
                case["edge_m"],
                case["shield_radius_m"],
                case["edge_m"],
                0,
                0,
                1,
                0,
                0,
                0,
                PARAMETERS["plane_distance_m"],
                PARAMETERS["shield_gap_m"],
            ),
        )
        rows.append(run_timed_config(case["name"], config))
    write_benchmark_outputs("active_shield_two_surface_scaling", rows, output_name=PARAMETERS["output_name"])
    return rows


if __name__ == "__main__":
    rows = main()
