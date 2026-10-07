"""Longitudinal field ranking is distinct from vector safety bounds or current fitting."""

from copy import deepcopy

import numpy as np
import pytest

from coilgen.models import TargetField
from coilgen.physics import biot_savart_wire_path
from coilgen.routing_field_objective import longitudinal_field_distortion


def target(count, weights=None, suppressed=None):
    coords = np.column_stack((np.arange(count), np.ones((count, 2))))
    return TargetField(
        coords,
        np.zeros((count, 3)),
        np.ones(count) if weights is None else weights,
        np.zeros(count, bool) if suppressed is None else suppressed,
        np.ones(count, int),
    )


def test_weighted_signed_current_metric_keeps_offsets_and_ignores_transverse_field():
    context = target(3, [1, 3, 100], [False, False, True])
    reference = np.zeros((3, 3))
    candidate = np.array([[1e-6, 9e-3, 0], [3e-6, -9e-3, 0], [1, 0, 0]])
    result = longitudinal_field_distortion(reference, candidate, context, [2, 0, 0], -2)
    assert result["rms_T"] == pytest.approx(np.sqrt(28) * 1e-6)
    assert result["peak_T"] == pytest.approx(6e-6)
    assert result["current_a"] == -2
    assert result["active_sample_count"] == 2
    assert not result["current_refitted"]


def test_straight_b0_parallel_return_has_zero_longitudinal_but_nonzero_vector_field():
    context = target(3)
    points = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    field = biot_savart_wire_path(points, context.coords, worker_count=1)
    assert np.max(np.linalg.norm(field, axis=1)) > 0
    result = longitudinal_field_distortion(np.zeros_like(field), field, context, [1, 0, 0], 2)
    assert result["rms_T"] == result["peak_T"] == 0


def test_large_weights_field_amplitudes_and_b0_norm_do_not_overflow():
    context = target(2, [1e308, 1e308])
    field = np.array([[1e200, 1e200, 0], [-1e200, -1e200, 0]])
    result = longitudinal_field_distortion(np.zeros_like(field), field, context, [1e308, 1e308, 0], -2)
    assert result["rms_T"] == pytest.approx(2 * np.sqrt(2) * 1e200)
    assert result["peak_T"] == pytest.approx(result["rms_T"])


def test_inactive_bad_samples_do_not_poison_the_metric_or_mutate_inputs():
    context = target(5, [1, 0, -1, np.inf, np.nan])
    field = np.full((5, 3), np.nan)
    field[0] = [2e-6, 0, 0]
    context.coords[1:] = np.nan
    original = deepcopy((context, field))
    result = longitudinal_field_distortion(np.zeros_like(field), field, context, [1, 0, 0], 1)
    assert result["rms_T"] == result["peak_T"] == 2e-6
    np.testing.assert_equal(field, original[1])
    np.testing.assert_equal(context.coords, original[0].coords)


@pytest.mark.parametrize("direction", [[0, 0, 0], [np.nan, 0, 1], [0, np.inf, 1], [0, 1]])
def test_invalid_b0_direction_never_becomes_zero_error(direction):
    with pytest.raises(ValueError, match="B0"):
        longitudinal_field_distortion(np.zeros((1, 3)), np.zeros((1, 3)), target(1), direction, 1)


@pytest.mark.parametrize("current", [0, np.nan, np.inf, None])
def test_invalid_fixed_current_rejects(current):
    with pytest.raises(ValueError):
        longitudinal_field_distortion(np.zeros((1, 3)), np.zeros((1, 3)), target(1), [0, 0, 1], current)


@pytest.mark.parametrize("defect", ["empty", "all_suppressed", "field_shape", "weights_shape", "nonfinite_active"])
def test_invalid_or_empty_active_context_is_rejected(defect):
    context = target(2)
    field = np.zeros((2, 3))
    if defect == "empty":
        context.weights[:] = 0
    elif defect == "all_suppressed":
        context.suppressed[:] = True
    elif defect == "weights_shape":
        context.weights = np.ones((2, 1))
    elif defect == "field_shape":
        field = np.zeros((1, 3))
    else:
        field[0] = np.nan
    with pytest.raises(ValueError):
        longitudinal_field_distortion(np.zeros((2, 3)), field, context, [0, 0, 1], 1)
