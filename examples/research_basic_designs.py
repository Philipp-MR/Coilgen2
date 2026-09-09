"""Two headless, independently validated magnetostatic designs.

Run from the repository root after installing ``.[research]``:
    python examples/research_basic_designs.py
Outputs use SI units. Accepted continuous sheets still require routing and
fabrication validation. These examples deliberately reserve a field margin.
"""

from dataclasses import replace
from pathlib import Path

from coilgen import CoilGenConfig
from coilgen.research.application import run_design
from coilgen.research.requests import ResearchRequest


def cylinder_config() -> CoilGenConfig:
    return CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.30, 0.12, 12, 6, 1, 0, 0, 0),
        surface_is_cylinder_flag=True,
        field_shape_function="y",
        target_gradient_strength=0.01,
        target_region_radius=0.045,
        target_region_resolution=15,
        sf_opt_method="constrained_quadratic",
        research_physics={"objective": "power", "b0_T": [0, 0, 3]},
        constrained_quadratic_fallback_policy="strict",
        constrained_optimization_tolerance=1e-8,
        target_error_tolerance_percent=4.0,
        target_error_constraint_acceptance_absolute_T=0,
        stream_function_smoothness_factor=0,
        turn_spacing_regularization_factor=0,
        symmetry_reduction="off",
        levels=8,
        min_point_loop_number=3,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
        plot_flag=False,
        save_stl_flag=False,
        progress_flag=False,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "outputs" / "research_basic"
    cylinder = cylinder_config()
    biplanar = replace(
        cylinder,
        coil_mesh_file="create bi-planary mesh",
        biplanar_mesh_parameter_list=(0.3, 0.3, 6, 6, 0, 0, 1, 0, 0, 0, 0.2),
        surface_is_cylinder_flag=False,
        field_shape_function="z",
        target_region_radius=0.04,
    )
    for name, config in (("cylinder_gradient", cylinder), ("biplanar_gradient", biplanar)):
        request = ResearchRequest.from_config(config, name=name, design_margin=0.8)
        payload = run_design(request, root / name)
        summary = payload["summary"]
        print(
            name,
            "accepted:",
            summary["accepted"],
            "independent relative field error:",
            summary["validated_relative_field_error"],
            "sheet power [W]:",
            summary["power_w"],
        )
        print("Evidence:", root / name / "result.json")


if __name__ == "__main__":
    main()
