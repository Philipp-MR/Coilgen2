"""Coordinate-independent preferences for added interconnections."""

import numpy as np
import pytest

from coilgen.routing_objective import routing_cost, routing_metrics


def test_axial_returns_have_no_alignment_penalty_and_transverse_returns_do():
    origin = np.zeros(3)
    assert routing_cost([origin, [0, 0, 2]], [0, 0, 1]) == 2
    assert routing_cost([origin, [2, 0, 0]], [0, 0, 1]) == 4
    assert routing_cost([origin, [2, 0, 0]], [0, 0, 1], 0) == 2


def test_objective_is_invariant_to_direction_scale_sign_rigid_rotation_and_subdivision():
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.4, -0.3], [0.6, 0.5, 1.0]])
    direction = np.array([0.5, 0.25, 1.0])
    expected = routing_cost(points, direction)
    q, _ = np.linalg.qr(np.array([[1.0, 4.0, 2.0], [5.0, 2.0, 1.0], [3.0, 1.0, 6.0]]))
    assert routing_cost(points @ q + [3, 4, 5], direction @ q) == pytest.approx(expected)
    assert routing_cost(points[::-1], -direction * 1e300) == pytest.approx(expected)
    assert routing_cost(points, direction * 1e-300) == pytest.approx(expected)
    subdivided = np.vstack([np.linspace(a, b, 7)[:-1] for a, b in zip(points[:-1], points[1:])] + [points[-1:]])
    assert routing_cost(subdivided, direction) == pytest.approx(expected)


def test_protected_turns_are_excluded_from_alignment_preference():
    values = routing_metrics([[0, 0, 0], [1, 0, 0], [1, 0, 2]], [0, 0, 1], mutable_segments=[False, True])
    assert values == {"length_m": 3, "transverse_length_m": 0, "cost_m": 3}


def test_duplicate_points_are_neutral_and_units_scale_linearly():
    points = np.array([[0, 0, 0], [0, 0, 0], [1, 2, 3]])
    assert routing_cost(points * 0.001, [1, 0, 0]) == pytest.approx(0.001 * routing_cost(points, [1, 0, 0]))


@pytest.mark.parametrize("direction", [[0, 0, 0], [0, np.nan, 1], [np.inf, 0, 0], [1, 2]])
def test_invalid_b0_rejected(direction):
    with pytest.raises(ValueError):
        routing_cost([[0, 0, 0], [1, 0, 0]], direction)


@pytest.mark.parametrize("weight", [-1, np.inf, np.nan, True, [1]])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError):
        routing_cost([[0, 0, 0], [1, 0, 0]], [0, 0, 1], weight)


def test_incomplete_mutable_mask_rejected():
    with pytest.raises(ValueError, match="mask"):
        routing_metrics([[0, 0, 0], [1, 0, 0]], [0, 0, 1], mutable_segments=[])
