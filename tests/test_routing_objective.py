"""Coordinate-independent preferences for added interconnections."""

import numpy as np
import pytest

from coilgen.routing_objective import (
    added_route_regularity,
    route_is_better,
    route_order_key,
    routing_cost,
    routing_metrics,
)


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


def test_added_backtracking_measures_excursions_and_ignores_protected_turns():
    points = [[0, 0, 0], [100, 0, 0], [2, 0, 0], [4, 0, 0], [3, 0, 0], [5, 0, 0]]
    values = added_route_regularity(points, [False, False, True, True, True])
    assert values["added_length_m"] == 5
    assert values["added_backtracking_m"] == 1
    assert values["added_excess_length_m"] == 2
    assert values["added_turning_rad"] == pytest.approx(2 * np.pi)


def test_regularity_is_invariant_to_rotation_reversal_and_linear_subdivision():
    points = np.array([[0.0, 0, 0], [0.3, 0.2, 0.1], [0.1, 0.4, 0.2], [0.8, 0.5, 0.3]])
    expected = added_route_regularity(points, [True] * 3)
    q, _ = np.linalg.qr(np.array([[1.0, 4.0, 2.0], [5.0, 2.0, 1.0], [3.0, 1.0, 6.0]]))
    variants = [points @ q + [3, 4, 5], points[::-1]]
    variants.append(np.vstack([np.linspace(a, b, 9)[:-1] for a, b in zip(points[:-1], points[1:])] + [points[-1:]]))
    for variant in variants:
        assert added_route_regularity(variant, [True] * (len(variant) - 1)) == pytest.approx(expected)
    scaled = added_route_regularity(points * 0.001, [True] * 3)
    for key in ("added_length_m", "added_backtracking_m", "added_excess_length_m"):
        assert scaled[key] == pytest.approx(expected[key] * 0.001)
    assert scaled["added_turning_rad"] == pytest.approx(expected["added_turning_rad"])


def test_closed_added_span_is_independent_of_cycle_seam():
    points = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    mask = np.array([True, False, False, True])
    ids = ("return", "turn", "turn", "return")
    expected = added_route_regularity(points, mask, span_ids=ids, closed=True)
    for offset in range(4):
        unique = np.roll(points[:-1], -offset, axis=0)
        rotated = np.vstack((unique, unique[:1]))
        assert added_route_regularity(
            rotated,
            np.roll(mask, -offset),
            span_ids=np.roll(ids, -offset),
            closed=True,
        ) == pytest.approx(expected)
    all_added = added_route_regularity(points, [True] * 4, closed=True)
    assert all_added["added_backtracking_m"] == 0
    assert all_added["added_excess_length_m"] == 4
    assert all_added["added_turning_rad"] == pytest.approx(2 * np.pi)


def test_regular_policy_has_one_transitive_key_and_preserves_physical_policy():
    from itertools import permutations

    candidates = [
        dict(
            jump_count=0,
            longitudinal_rms_T=e,
            length_m=length,
            added_backtracking_m=b,
            added_excess_length_m=2 * b,
            added_turning_rad=1.0,
        )
        for e, length, b in ((1e-6, 3.0, 0.1), (2e-6, 4.0, 0.0), (3e-6, 2.0, 0.1))
    ]
    assert min(candidates, key=route_order_key) is candidates[0]
    for order in permutations(candidates):
        winner = order[0]
        for candidate in order[1:]:
            if route_is_better(candidate, winner, policy="regular"):
                winner = candidate
        assert winner is candidates[1]
    incomplete = {**candidates[0], "regularity_is_lower_bound": True}
    with pytest.raises(ValueError, match="provenance"):
        route_order_key(incomplete, policy="regular")
