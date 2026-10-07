"""Signed residuals use a true zero-centered mapper, without changing data."""

import numpy as np
import pytest

import test_native_workbench as workbench_tests
from coilgen.gui.viewport import _signed_residual_limits

application = workbench_tests.application
scientific_viewport = workbench_tests.scientific_viewport


def mapper_range(viewport):
    """Exercise PyVista's real mapper with the viewer's submitted mesh/options."""
    mesh, options, _actor = viewport.plotter.meshes[-1]
    plotter = viewport.pv.Plotter(off_screen=True)
    try:
        actor = plotter.add_mesh(mesh, **options)
        return actor.mapper.scalar_range
    finally:
        plotter.close()


@pytest.mark.parametrize("component,column", [("Bx", 0), ("By", 1), ("Bz", 2)])
def test_asymmetric_signed_residual_mapper_is_centered_on_zero(scientific_viewport, component, column):
    viewport = scientific_viewport
    target = np.zeros((2, 3))
    actual = np.zeros((2, 3))
    actual[:, column] = [-0.468e-6, 0.494e-6]
    arrays = {"target_points_m": [[0, 0, 0], [0, 0, 0.01]], "target_field_T": target, "routed_field_T": actual}
    viewport.display_run(arrays)
    viewport.units.setCurrentText("µT")
    viewport.component.setCurrentText(component)
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("routed_residual"))
    assert mapper_range(viewport) == pytest.approx((-0.494, 0.494))
    np.testing.assert_allclose(viewport._dataset_mesh.active_scalars, [-0.468, 0.494])
    np.testing.assert_array_equal(arrays["routed_field_T"], actual)
    # An explicit comparison range remains authoritative.
    viewport.set_colour_limits((-1, 2))
    assert mapper_range(viewport) == (-1, 2)


def test_magnitude_residual_and_regular_field_keep_actual_mapper_range(scientific_viewport):
    viewport = scientific_viewport
    arrays = {
        "target_points_m": [[0, 0, 0], [0, 0, 0.01]],
        "target_field_T": [[0, 0, 0], [0, 0, 0]],
        "routed_field_T": [[0, 0, -0.468e-6], [0, 0, 0.494e-6]],
    }
    viewport.display_run(arrays)
    viewport.units.setCurrentText("µT")
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("routed_residual"))
    viewport.component.setCurrentText("|B|")
    assert mapper_range(viewport) == pytest.approx((0.468, 0.494))
    viewport.component.setCurrentText("Bz")
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("routed_field"))
    assert mapper_range(viewport) == pytest.approx((-0.468, 0.494))


def test_zero_residual_has_finite_symmetric_mapper_and_unchanged_values(scientific_viewport):
    viewport = scientific_viewport
    viewport.display_run(
        {"target_points_m": [[0, 0, 0]], "target_field_T": [[0, 0, 0]], "contour_field_T": [[0, 0, 0]]}
    )
    viewport.dataset.setCurrentIndex(viewport.dataset.findData("contour_residual"))
    assert mapper_range(viewport) == (-1e-12, 1e-12)
    np.testing.assert_array_equal(viewport._dataset_mesh.active_scalars, [0])
    assert _signed_residual_limits([float("nan"), -2, float("inf"), 1]) == (-2, 2)
    assert _signed_residual_limits([float("nan"), float("inf")]) is None
