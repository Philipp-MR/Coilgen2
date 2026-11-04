"""Combined gradient targets (x+y, x-y) scale and report |G|, not a tied axis."""

from dataclasses import replace

import numpy as np
import pytest

from coilgen.application.presets import preset_config
from coilgen.geometry import load_mesh_from_config, split_disconnected_mesh
from coilgen.target import _fit_bz_linear_gradient, define_target_field


def _target(expression, axis="auto"):
    config = replace(
        preset_config("cylinder"),
        field_shape_function=expression,
        target_gradient_axis=axis,
        target_gradient_normalization="linear_fit",
        target_gradient_strength=0.02,
    )
    parts = split_disconnected_mesh(load_mesh_from_config(config, None))
    target = define_target_field(parts, None, None, config)
    gradient = _fit_bz_linear_gradient(target.coords, target.b, np.zeros(len(target.coords), dtype=bool))
    return target, gradient


@pytest.mark.parametrize(("expression", "signs"), [("x+y", (1, 1)), ("x-y", (1, -1))])
def test_combined_gradient_resolves_to_magnitude(expression, signs):
    target, gradient = _target(expression)
    assert target.target_diagnostics["target_gradient_axis"] == "magnitude"
    assert np.linalg.norm(gradient) == pytest.approx(0.02, rel=1e-9)
    np.testing.assert_allclose(gradient[:2], 0.02 / np.sqrt(2) * np.asarray(signs), rtol=1e-9)
    assert gradient[2] == pytest.approx(0.0, abs=1e-12)
    # The target itself is still the complete expression, sample by sample.
    x, y = target.coords[:, 0], target.coords[:, 1]
    np.testing.assert_allclose(target.b[:, 2], 0.02 / np.sqrt(2) * (x + signs[1] * y), atol=1e-15)


def test_explicit_axis_keeps_the_previous_per_axis_scaling():
    target, gradient = _target("x+y", axis="x")
    assert target.target_diagnostics["target_gradient_axis"] == "x"
    np.testing.assert_allclose(gradient[:2], (0.02, 0.02), rtol=1e-9)


def test_single_axis_target_is_unchanged():
    target, gradient = _target("y")
    assert target.target_diagnostics["target_gradient_axis"] == "y"
    np.testing.assert_allclose(gradient, (0.0, 0.02, 0.0), atol=1e-12)
    target, gradient = _target("0.5*y")
    assert target.target_diagnostics["target_gradient_axis"] == "y"
    assert gradient[1] == pytest.approx(0.02, rel=1e-9)
