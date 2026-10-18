"""Exterior finite-wire fields retain their small, nonzero analytic limit."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from coilgen.physics import biot_savart_wire_path


@pytest.mark.parametrize("axial", [-0.1, 0.2])
def test_nearly_collinear_exterior_samples_match_stable_analytic_field(axial):
    length = 0.1
    radii = np.array([1e-4, 1e-6, 1e-8, 1e-10])
    targets = np.column_stack((radii, np.zeros_like(radii), np.full_like(radii, axial)))
    # Independently evaluate the original signed endpoint formula at high
    # precision; its double-precision cancellation is the regression trigger.
    with localcontext() as context:
        context.prec = 60
        start = Decimal(str(axial))
        end = Decimal(str(length)) - start
        expected = []
        for radius in radii:
            rho = Decimal(str(radius))
            endpoint_sum = start / (rho * rho + start * start).sqrt() + end / (rho * rho + end * end).sqrt()
            expected.append(float(Decimal("1e-7") * endpoint_sum / rho))
    wire = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, length]])
    actual = biot_savart_wire_path(wire, targets)
    np.testing.assert_allclose(actual[:, 1], expected, rtol=5e-15, atol=0)
    np.testing.assert_array_equal(actual[:, [0, 2]], 0.0)
    np.testing.assert_allclose(biot_savart_wire_path(wire[::-1], targets), -actual, rtol=5e-15, atol=0)

    split = np.column_stack((np.zeros(101), np.zeros(101), np.linspace(0.0, length, 101)))
    np.testing.assert_allclose(biot_savart_wire_path(split, targets), actual, rtol=5e-15, atol=0)
