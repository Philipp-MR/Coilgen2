"""Shared fixtures for pipeline stages."""

from coilgen import CoilGenConfig


def config():
    return CoilGenConfig(
        coil_mesh_file="create cylinder mesh",
        cylinder_mesh_parameter_list=(0.3, 0.12, 12, 6, 1, 0, 0, 0),
        field_shape_function="y",
        target_gradient_strength=0.01,
        target_region_radius=0.045,
        target_region_resolution=8,
        sf_opt_method="constrained_quadratic",
        research_physics={"objective": "power"},
        constrained_quadratic_fallback_policy="strict",
        target_error_tolerance_percent=5,
        stream_function_smoothness_factor=0,
        turn_spacing_regularization_factor=0,
        levels=6,
        min_point_loop_number=3,
        skip_postprocessing=True,
        skip_sweep=True,
        skip_inductance_calculation=True,
        progress_flag=False,
        plot_flag=False,
    )
