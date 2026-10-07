from __future__ import annotations

from projects.halbach_cylindrical_3axis_gradient.sandbox_cylinder_contour_smoothing import (
    MAX_CURVATURE_SECOND_DERIVATIVE_RMS_RATIO,
    MAX_WIGGLE_RMS_RATIO,
    run_sandbox,
)


def test_coarse_cylinder_bspline_is_smooth_in_developed_plane_and_stays_on_mesh() -> None:
    result = run_sandbox()

    assert result.smoothing_diagnostics["contour_path_smoothing_method"] == ("developed_cylinder_periodic_bspline")
    assert result.wiggle_rms_ratio < MAX_WIGGLE_RMS_RATIO
    assert result.curvature_second_derivative_rms_ratio < MAX_CURVATURE_SECOND_DERIVATIVE_RMS_RATIO
    assert result.max_surface_reconstruction_error_m < 1.0e-11
