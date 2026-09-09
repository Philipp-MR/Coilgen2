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
        "output_name": "benchmark_target_point_scaling",
        "target_region_resolutions": (3, 5, 7),
        "mesh": {
            "radius_m": 0.16,
            "length_m": 0.3,
            "angular_divisions": 10,
            "length_divisions": 6,
        },
    }
)


def main() -> list[dict]:
    mesh = PARAMETERS["mesh"]
    rows = []
    for resolution in PARAMETERS["target_region_resolutions"]:
        config = base_benchmark_config(
            target_region_resolution=int(resolution),
            cylinder_mesh_parameter_list=(
                mesh["length_m"],
                mesh["radius_m"],
                mesh["angular_divisions"],
                mesh["length_divisions"],
                1,
                0,
                0,
                0,
            ),
        )
        rows.append(run_timed_config(f"target_resolution_{resolution}", config))
    write_benchmark_outputs("target_point_scaling", rows, output_name=PARAMETERS["output_name"])
    return rows


if __name__ == "__main__":
    rows = main()
