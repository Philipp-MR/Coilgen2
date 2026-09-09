from __future__ import annotations

# Direct execution from Spyder or any working directory uses this checkout.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    for _path in (_Path(__file__).resolve().parents[3], _Path(__file__).resolve().parents[3] / "src"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))

from projects._shared.object_parameters import object_parameters

from projects.general_examples.scripts._benchmark_scaling_common import (
    base_benchmark_config,
    run_timed_config,
    write_benchmark_outputs,
)


PARAMETERS = object_parameters(
    {
        "output_name": "benchmark_mesh_vertex_scaling",
        "mesh_divisions": ((8, 4), (12, 6), (16, 8)),
        "radius_m": 0.16,
        "length_m": 0.3,
        "target_region_resolution": 5,
    }
)


def main() -> list[dict]:
    rows = []
    for angular_divisions, length_divisions in PARAMETERS["mesh_divisions"]:
        config = base_benchmark_config(
            target_region_resolution=PARAMETERS["target_region_resolution"],
            cylinder_mesh_parameter_list=(
                PARAMETERS["length_m"],
                PARAMETERS["radius_m"],
                int(angular_divisions),
                int(length_divisions),
                1,
                0,
                0,
                0,
            ),
        )
        rows.append(run_timed_config(f"mesh_{angular_divisions}x{length_divisions}", config))
    write_benchmark_outputs("mesh_vertex_scaling", rows, output_name=PARAMETERS["output_name"])
    return rows


if __name__ == "__main__":
    rows = main()
