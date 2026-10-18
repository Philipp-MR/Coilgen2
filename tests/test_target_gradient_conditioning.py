"""Affine target normalization is invariant to the world-coordinate origin."""

import numpy as np
import pytest

from coilgen import CoilGenConfig
from coilgen.models import Mesh
from coilgen.target import _fit_bz_linear_gradient, define_target_field


def grid(offset):
    return np.array(np.meshgrid(*([[-0.02, 0, 0.02]] * 3))).reshape(3, -1).T + offset


@pytest.mark.parametrize("offset", [np.zeros(3), np.array([0.2, -0.3, 0.7]), np.full(3, 1e6)])
def test_affine_target_fit_separates_world_origin_and_field_offset(offset):
    points = grid(offset)
    target = np.zeros_like(points)
    target[:, 2] = 0.03 * (points[:, 1] - offset[1]) + 0.004

    gradient = _fit_bz_linear_gradient(points, target, np.zeros(len(points), bool))

    np.testing.assert_allclose(gradient, [0, 0.03, 0], rtol=0, atol=1e-14)


def test_target_normalization_and_physical_fields_are_stable_under_large_translation():
    fields, diagnostics = [], []
    for offset in (np.zeros(3), np.full(3, 1e6)):
        points = grid(offset)
        mesh = Mesh(points, np.empty((0, 3), dtype=np.int64))
        config = CoilGenConfig(
            field_shape_function=f"0.03*(y-{offset[1]:.17g})",
            target_gradient_strength=0.01,
            target_gradient_normalization="linear_fit",
            target_gradient_axis="y",
            use_only_target_mesh_verts=True,
        )
        target = define_target_field([], mesh, None, config)
        fields.append(target.b)
        diagnostics.append(target.target_diagnostics)
        assert not target.target_diagnostics["target_gradient_normalization_fallback"]
        assert target.target_diagnostics["target_gradient_effective_T_per_m"] == pytest.approx(0.01)
        assert target.target_diagnostics["target_gradient_normalization_scale"] == pytest.approx(1 / 3)
    # Input coordinates at 1e6 m have finite representation precision; this
    # tolerance covers their quantization without allowing a gradient-scale error.
    np.testing.assert_allclose(fields[0], fields[1], rtol=1e-8, atol=1e-12)
    assert diagnostics[0]["target_gradient_raw_fit_y_T_per_m"] == pytest.approx(
        diagnostics[1]["target_gradient_raw_fit_y_T_per_m"]
    )


def test_rank_deficient_affine_fit_retains_minimum_norm_gradient():
    # Binary-exact increments keep the translated x/y columns truly identical;
    # decimal increments can acquire different quantization at offsets 2 and 3.
    coordinate = np.arange(-4, 5, dtype=float) / 256
    points = np.column_stack((coordinate, coordinate, np.zeros_like(coordinate))) + [2, 3, 4]
    target = np.zeros_like(points)
    target[:, 2] = 0.03 * (points[:, 0] - 2) + 0.004

    gradient = _fit_bz_linear_gradient(points, target, np.zeros(len(points), bool))

    np.testing.assert_allclose(gradient, [0.015, 0.015, 0], rtol=0, atol=1e-14)


@pytest.mark.parametrize("coordinate_scale", [1e-18, 1e-12, 1, 1e3])
def test_gradient_fit_rank_is_independent_of_coordinate_scale(coordinate_scale):
    points = coordinate_scale * grid(np.zeros(3))
    target = np.zeros_like(points)
    target[:, 2] = 0.03 * points[:, 1]

    gradient = _fit_bz_linear_gradient(points, target, np.zeros(len(points), bool))

    np.testing.assert_allclose(gradient, [0, 0.03, 0], rtol=0, atol=1e-14)
