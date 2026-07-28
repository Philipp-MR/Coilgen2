from __future__ import annotations

import time
from pathlib import Path

from projects._shared.object_parameters import object_parameters

from projects._shared.example_utils import example_output_dir

from coilgen import (
    CylinderCoilDesign,
    ParameterAxis,
)
from coilgen.workflow import (
    CoilCaseSpec,
    CoilVariantSpec,
    ParameterSweepSpec,
    ParameterSweepWorkflow,
)


PARAMETERS = object_parameters(
    {
        "output_name": "benchmark_sweep_case_cache_scaling",
        "radius_values_m": (0.14, 0.16),
        "resolution_values": (3, 5),
        "retain_case_results": False,
    }
)


def _case_factory(output_root: Path):
    def build(point: dict) -> CoilCaseSpec:
        name = f"r{point['radius_m']}_res{point['target_region_resolution']}"
        design = CylinderCoilDesign(
            name=name,
            target_bz_expression="y",
            radius_m=float(point["radius_m"]),
            length_m=0.3,
            angular_divisions=8,
            length_divisions=4,
            target_region_resolution=int(point["target_region_resolution"]),
            contour_levels=4,
            min_contour_points=3,
            regularization_method="fixed",
            regularization_factor=40.0,
            skip_postprocessing=True,
            skip_sweep=True,
            skip_inductance=True,
            progress=False,
            advanced_parameters={
                "low_memory_mode": True,
                "retain_heavy_fields": False,
                "contour_only_sweep_mode": True,
            },
        )
        return CoilCaseSpec(
            name=name,
            variants=(CoilVariantSpec(name="main", design=design, save_reports=False),),
            output_dir=output_root / name,
            parameters=point,
        )

    return build


def main():
    output_root = example_output_dir(PARAMETERS["output_name"])
    spec = ParameterSweepSpec(
        name="sweep_case_cache_scaling",
        axes=(
            ParameterAxis("radius_m", PARAMETERS["radius_values_m"]),
            ParameterAxis("target_region_resolution", PARAMETERS["resolution_values"]),
        ),
        case_factory=_case_factory(output_root),
        output_dir=output_root,
        resume_from_case_cache=True,
        retain_case_results=bool(PARAMETERS["retain_case_results"]),
    )
    workflow = ParameterSweepWorkflow()
    start = time.perf_counter()
    first = workflow.run(spec)
    first_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    second = workflow.run(spec)
    second_elapsed = time.perf_counter() - start
    report = output_root / "sweep_case_cache_scaling_report.md"
    report.write_text(
        "\n".join(
            [
                "# sweep_case_cache_scaling",
                "",
                f"First run rows: {len(first.rows())}",
                f"First run elapsed_s: {first_elapsed:.6g}",
                f"Second run cached rows: {len(second.cached_rows)}",
                f"Second run elapsed_s: {second_elapsed:.6g}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved sweep cache benchmark report: {report}")
    return first, second


if __name__ == "__main__":
    first, second = main()
