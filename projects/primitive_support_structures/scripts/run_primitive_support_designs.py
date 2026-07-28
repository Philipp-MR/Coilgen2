from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
PROJECT_ROOT = REPOSITORY_ROOT

from projects._shared.example_utils import (
    open_project_plot_book_after_run,
    parse_example_parameters,
    print_result_summary,
)
from projects._shared.object_parameters import object_parameters
from projects._shared.paths import project_output_root


OUTPUT_ROOT = project_output_root("primitive_support_structures")

from coilgen import (
    AnnularDiscCoilDesign,
    CircularDiscCoilDesign,
    CoilDesign,
    CoilProject,
    ConicalFrustumCoilDesign,
    EllipsoidalCapCoilDesign,
    EllipticalCylinderCoilDesign,
    ParametricSurfaceCoilDesign,
    PartialCylinderCoilDesign,
    RuledSurfaceCoilDesign,
    ShieldedCoilDesign,
    SphericalCapCoilDesign,
    SuperellipseCylinderCoilDesign,
    SurfaceOfRevolutionCoilDesign,
    ToroidalSegmentCoilDesign,
)


DEFAULT_PARAMETERS = {
    "run": {
        # Choose one or more:
        # circular_disc, annular_disc, partial_cylinder, elliptical_cylinder,
        # superellipse_cylinder, spherical_cap, ellipsoidal_cap,
        # conical_frustum, toroidal_segment, ruled_surface,
        # surface_of_revolution, parametric_surface, shielded_annular_disc.
        "selected_cases": ("annular_disc",),
        "run_design": True,  # True runs CoilGen; False only validates the generated configs.
        "show_plots": True,  # Opens one interactive plot book per finished design in Spyder or a Python shell.
        "save_reports": True,  # Writes markdown/CSV summaries to this project's outputs folder.
    },
    "target": {
        "field_shape": "y",  # Target Bz expression. "y" gives a linear y-gradient test field.
        "target_scale": 1.0,  # Requested target-field scale in SI units used by the CoilGen target expression.
        "target_region_radius_m": 0.06,  # Radius of the spherical DSV target point cloud [m].
        "target_region_resolution": 5,  # Number of target samples per axis before spherical masking.
    },
    "optimization": {
        "method": "tikhonov",  # Regularized stream-function solve.
        # "method": "target_error_constrained_efficiency",  # Alternative: maximize efficiency for a fixed target-error limit.
        "regularization_method": "fixed",  # Use "l_curve" for automatic regularization scans; fixed is faster for primitive smoke tests.
        "regularization_factor": 20.0,  # Tikhonov factor when regularization_method="fixed".
        "target_error_tolerance_percent": 5.0,  # Used by target_error_constrained_efficiency.
    },
    "layout": {
        "contour_levels": 6,  # Number of contour levels in the stream function.
        "skip_postprocessing": True,  # True evaluates unconnected contours; set False to route a full connected wire path.
        "skip_sweep": True,  # True skips conductor-body sweep generation.
        "skip_inductance": True,  # Set False when inductance/resistance are needed for the primitive design.
    },
    "conductor": {
        "width_m": 0.0015,  # Rectangular copper conductor width [m].
        "height_m": 0.0015,  # Rectangular copper conductor height [m].
    },
    "mesh_resolution": {
        "circular_disc_radial": 18,  # Radial mesh divisions for the full circular disc.
        "annular_radial": 6,  # Radial mesh divisions through the annular support.
        "annular_circular": 48,  # Angular mesh divisions around the annulus.
        "cylindrical_angular": 48,  # Angular divisions for partial/elliptical cylinders.
        "cylindrical_length": 24,  # Axial divisions for partial/elliptical cylinders.
        "surface_of_revolution_angular": 48,  # Angular divisions around the axisymmetric surface.
        "cap_azimuthal": 48,  # Azimuthal divisions for spherical/ellipsoidal caps.
        "cap_polar": 16,  # Polar divisions from cap pole to boundary.
        "toroidal_major": 48,  # Divisions along the toroidal arc.
        "toroidal_minor": 16,  # Divisions around the tube section.
        "ruled_curve": 20,  # Samples along each ruled-surface boundary curve.
        "ruled_span": 12,  # Samples between the two ruled-surface boundary curves.
        "parametric_u": 24,  # u-direction divisions for the custom parametric surface.
        "parametric_v": 24,  # v-direction divisions for the custom parametric surface.
    },
}


PARAMETERS = object_parameters(
    {
        # Edit only this object for normal use. Missing entries are filled from DEFAULT_PARAMETERS.
        # "run": {"selected_cases": ("partial_cylinder", "shielded_annular_disc")},
    }
)


def build_designs(parameters: dict[str, Any]) -> dict[str, CoilDesign]:
    target = parameters["target"]
    optimization = parameters["optimization"]
    layout = parameters["layout"]
    conductor = parameters["conductor"]
    mesh = parameters["mesh_resolution"]

    common = {
        "target_bz_expression": target["field_shape"],
        "target_scale": target["target_scale"],
        "target_region_radius_m": target["target_region_radius_m"],
        "target_region_resolution": target["target_region_resolution"],
        "optimization_method": optimization["method"],
        "regularization_method": optimization["regularization_method"],
        "regularization_factor": optimization["regularization_factor"],
        "target_error_tolerance_percent": optimization["target_error_tolerance_percent"],
        "contour_levels": layout["contour_levels"],
        "skip_postprocessing": layout["skip_postprocessing"],
        "skip_sweep": layout["skip_sweep"],
        "skip_inductance": layout["skip_inductance"],
        "conductor_width_m": conductor["width_m"],
        "conductor_height_m": conductor["height_m"],
        "cpu_workers": 0,
        "progress": True,
        "show_internal_plots": False,
    }

    return {
        "circular_disc": CircularDiscCoilDesign(
            name="primitive_circular_disc_y_gradient",
            radius_m=0.25,
            radial_divisions=mesh["circular_disc_radial"],
            **common,
        ),
        "annular_disc": AnnularDiscCoilDesign(
            name="primitive_annular_disc_y_gradient",
            inner_radius_m=0.07,
            outer_radius_m=0.25,
            radial_divisions=mesh["annular_radial"],
            circular_divisions=mesh["annular_circular"],
            **common,
        ),
        "shielded_annular_disc": ShieldedCoilDesign(
            name="primitive_shielded_annular_disc_y_gradient",
            primary_design=AnnularDiscCoilDesign(
                name="primitive_annular_disc_primary_surface",
                inner_radius_m=0.07,
                outer_radius_m=0.25,
                radial_divisions=mesh["annular_radial"],
                circular_divisions=mesh["annular_circular"],
            ),
            shield_separation_m=0.03,
            **common,
        ),
        "partial_cylinder": PartialCylinderCoilDesign(
            name="primitive_partial_cylinder_y_gradient",
            radius_m=0.25,
            length_m=0.50,
            start_angle_rad=-1.35,
            end_angle_rad=1.35,
            angular_divisions=mesh["cylindrical_angular"],
            length_divisions=mesh["cylindrical_length"],
            **common,
        ),
        "elliptical_cylinder": EllipticalCylinderCoilDesign(
            name="primitive_elliptical_cylinder_y_gradient",
            radius_x_m=0.28,
            radius_y_m=0.18,
            length_m=0.50,
            angular_divisions=mesh["cylindrical_angular"],
            length_divisions=mesh["cylindrical_length"],
            **common,
        ),
        "superellipse_cylinder": SuperellipseCylinderCoilDesign(
            name="primitive_superellipse_cylinder_y_gradient",
            radius_x_m=0.28,
            radius_y_m=0.18,
            length_m=0.50,
            exponent=4.0,
            angular_divisions=mesh["cylindrical_angular"],
            length_divisions=mesh["cylindrical_length"],
            **common,
        ),
        "spherical_cap": SphericalCapCoilDesign(
            name="primitive_spherical_cap_y_gradient",
            radius_m=0.25,
            cap_polar_angle_rad=0.72 * np.pi,
            azimuthal_divisions=mesh["cap_azimuthal"],
            polar_divisions=mesh["cap_polar"],
            **common,
        ),
        "ellipsoidal_cap": EllipsoidalCapCoilDesign(
            name="primitive_ellipsoidal_cap_y_gradient",
            radius_x_m=0.28,
            radius_y_m=0.20,
            radius_z_m=0.18,
            cap_polar_angle_rad=0.72 * np.pi,
            azimuthal_divisions=mesh["cap_azimuthal"],
            polar_divisions=mesh["cap_polar"],
            **common,
        ),
        "conical_frustum": ConicalFrustumCoilDesign(
            name="primitive_conical_frustum_y_gradient",
            radius_bottom_m=0.24,
            radius_top_m=0.16,
            length_m=0.50,
            angular_divisions=mesh["cylindrical_angular"],
            length_divisions=mesh["cylindrical_length"],
            **common,
        ),
        "toroidal_segment": ToroidalSegmentCoilDesign(
            name="primitive_toroidal_segment_y_gradient",
            major_radius_m=0.32,
            minor_radius_m=0.07,
            start_angle_rad=-0.75 * np.pi,
            end_angle_rad=0.75 * np.pi,
            tube_start_angle_rad=0.0,
            tube_end_angle_rad=2.0 * np.pi,
            major_divisions=mesh["toroidal_major"],
            minor_divisions=mesh["toroidal_minor"],
            **common,
        ),
        "ruled_surface": RuledSurfaceCoilDesign(
            name="primitive_ruled_surface_y_gradient",
            curve_a_m=tuple(
                (-0.22 + 0.44 * t, -0.12, 0.04 * np.sin(np.pi * t)) for t in np.linspace(0.0, 1.0, mesh["ruled_curve"])
            ),
            curve_b_m=tuple(
                (-0.22 + 0.44 * t, 0.12, -0.04 * np.sin(np.pi * t)) for t in np.linspace(0.0, 1.0, mesh["ruled_curve"])
            ),
            span_divisions=mesh["ruled_span"],
            **common,
        ),
        "surface_of_revolution": SurfaceOfRevolutionCoilDesign(
            name="primitive_surface_of_revolution_y_gradient",
            circular_divisions=mesh["surface_of_revolution_angular"],
            z_radius_pairs_m=((-0.25, 0.18), (-0.08, 0.24), (0.08, 0.24), (0.25, 0.18)),
            **common,
        ),
        "parametric_surface": ParametricSurfaceCoilDesign(
            name="primitive_parametric_surface_y_gradient",
            surface=lambda u, v: (
                0.44 * (u - 0.5),
                0.24 * (v - 0.5),
                0.04 * np.sin(np.pi * u) * np.sin(np.pi * v),
            ),
            u_divisions=mesh["parametric_u"],
            v_divisions=mesh["parametric_v"],
            **common,
        ),
    }


def selected_designs(parameters: dict[str, Any]) -> dict[str, CoilDesign]:
    designs = build_designs(parameters)
    selected = tuple(parameters["run"]["selected_cases"])
    missing = sorted(set(selected) - set(designs))
    if missing:
        raise ValueError(f"Unknown primitive case(s): {', '.join(missing)}")
    return {name: designs[name] for name in selected}


def validate_selected_designs(parameters: dict[str, Any]) -> dict[str, CoilDesign]:
    designs = selected_designs(parameters)
    for name, design in designs.items():
        report = design.validate(project_root=PROJECT_ROOT)
        if not report.ok:
            raise ValueError(f"Invalid config for {name}:\n{report.format()}")
        print(f"{name}: configuration valid")
    return designs


def run_design(name: str, design: CoilDesign, parameters: dict[str, Any]) -> None:
    """Compatibility helper: run one primitive case through a single-channel project."""

    print(f"\nRunning {name}...")
    project = CoilProject(
        name=name,
        output_dir=OUTPUT_ROOT / name,
        project_root=PROJECT_ROOT,
        save_reports=bool(parameters["run"]["save_reports"]),
        open_plot_book=bool(parameters["run"]["show_plots"]),
        plot_include=("raw_contours", "contour_groups", "layout_field_overview", "performance_table"),
        max_plot_parts=4,
    )
    project.channel(name, label=name, design=design)
    project_run = project.solve()
    channel_run = project_run.channel_runs[0]
    if channel_run.result is not None:
        print_result_summary(channel_run.result, title=f"{name} result")


def build_configs(parameters: dict[str, Any] | None = None):
    """Return low-level CoilGenConfig objects for tests or external scripts."""

    params = parse_example_parameters(
        parameters or PARAMETERS, DEFAULT_PARAMETERS, example_name="primitive_support_structures"
    )
    return {name: design.to_config() for name, design in selected_designs(params).items()}


def main() -> None:
    parameters = parse_example_parameters(PARAMETERS, DEFAULT_PARAMETERS, example_name="primitive_support_structures")
    validate_selected_designs(parameters)
    if not parameters["run"]["run_design"]:
        print("run.run_design is False; stopping after validation.")
        return
    project = CoilProject(
        name="primitive_support_structures",
        output_dir=OUTPUT_ROOT,
        project_root=PROJECT_ROOT,
        save_reports=bool(parameters["run"]["save_reports"]),
        open_plot_book=bool(parameters["run"]["show_plots"]),
        plot_include=("raw_contours", "contour_groups", "layout_field_overview", "performance_table"),
        max_plot_parts=4,
        metadata={"selected_cases": tuple(parameters["run"]["selected_cases"])},
    )
    for name, design in selected_designs(parameters).items():
        project.channel(
            name,
            label=name,
            design=design,
            metadata={"primitive_case": name},
        )
    project_run = project.solve()
    for channel_run in project_run.channel_runs:
        if channel_run.result is not None:
            print_result_summary(channel_run.result, title=f"{channel_run.channel.name} result")
    print("Spyder variables available after run: project, project_run")
    return project, project_run


if __name__ == "__main__":
    run_outputs = main()
    if run_outputs is not None:
        project, project_run = run_outputs
        plot_book = open_project_plot_book_after_run(
            project_run, figures=project_run.figures, include=project.plot_include
        )
