"""Preferences distinguish attachment kinks and concentrated mutable bends."""

import numpy as np
import pytest

from coilgen.routing_objective import (
    RoutingSearchCost,
    added_route_regularity,
    regularity_order_key,
    route_order_key,
    search_cost_key,
    transition_regularity,
    join_bend_consistency,
)


def test_straight_connection_still_charges_two_right_angle_attachments():
    points = [[-1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]]
    values = added_route_regularity(points, [False, True, False])
    assert values["added_turning_rad"] == 0
    assert values["added_attachment_turning_rad"] == pytest.approx(np.pi)
    assert values["added_maximum_attachment_angle_rad"] == pytest.approx(np.pi / 2)
    assert values["added_bending_energy"] == 0
    assert transition_regularity(points[1:3], [1, 0, 0], [1, 0, 0]) == pytest.approx(values)


def test_protected_corner_does_not_charge_added_bending_or_attachments():
    points = [[0, -1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    values = added_route_regularity(points, [False, False, True, False])
    assert values["added_turning_rad"] == 0
    assert values["added_attachment_turning_rad"] == 0
    assert values["added_bending_energy"] == 0


def test_curvature_energy_distinguishes_gentle_arc_from_concentrated_corner():
    corner = added_route_regularity([[1, 0, 0], [1, 1, 0], [0, 1, 0]], [True, True])
    theta = np.linspace(0, np.pi / 2, 257)
    arc = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    gentle = added_route_regularity(arc, [True] * (len(arc) - 1))
    assert corner["added_bending_energy"] == pytest.approx(np.pi**2 / 2)
    assert gentle["added_bending_energy"] == pytest.approx(np.pi**2 / 4, rel=0.005)
    assert gentle["added_bending_energy"] < corner["added_bending_energy"]


def test_attachment_and_energy_are_scale_rotation_reversal_and_subdivision_invariant():
    points = np.array([[-1.0, 0, 0], [0, 0, 0], [0.3, 0.2, 0], [0.8, 0.5, 0.1], [1, 0.5, 0.1]])
    mask = [False, True, True, False]
    expected = added_route_regularity(points, mask)
    q, _ = np.linalg.qr(np.array([[1.0, 4.0, 2.0], [5.0, 2.0, 1.0], [3.0, 1.0, 6.0]]))
    transformed = added_route_regularity(points @ q + [3, 4, 5], mask)
    assert transformed == pytest.approx(expected)
    assert added_route_regularity(points[::-1], mask[::-1]) == pytest.approx(expected)
    subdivided = np.vstack([np.linspace(a, b, 7)[:-1] for a, b in zip(points[:-1], points[1:])] + [points[-1:]])
    assert added_route_regularity(subdivided, np.repeat(mask, 6)) == pytest.approx(expected)
    scaled = added_route_regularity(points * 0.001, mask)
    for key in ("added_attachment_turning_rad", "added_maximum_attachment_angle_rad", "added_bending_energy"):
        assert scaled[key] == pytest.approx(expected[key])


def test_regular_policy_and_additive_graph_guide_prefer_attachment_and_curvature():
    shared = dict(
        jump_count=0,
        longitudinal_rms_T=1e-6,
        length_m=2.0,
        added_backtracking_m=0.0,
        added_excess_length_m=0.2,
        added_turning_rad=2.0,
    )
    smooth = {**shared, "added_attachment_turning_rad": 0.0, "added_bending_energy": 3.0}
    kinked = {**shared, "length_m": 1.8, "added_attachment_turning_rad": 0.5, "added_bending_energy": 1.0}
    assert route_order_key(smooth, policy="regular") < route_order_key(kinked, policy="regular")
    assert route_order_key(kinked, policy="physical") < route_order_key(smooth, policy="physical")
    first = RoutingSearchCost(0, 2.0, regularity_order_key(smooth))
    second = RoutingSearchCost(0, 1.8, regularity_order_key(kinked))
    assert search_cost_key(first) < search_cost_key(second)
    combined = first + first + 1.0
    assert combined.regularity == pytest.approx(np.asarray(first.regularity) * 2)
    assert float(combined) == 5.0


def test_invalid_attachment_tangent_and_regularity_guide_rejected():
    with pytest.raises(ValueError, match="tangents"):
        transition_regularity([[0, 0, 0], [1, 0, 0]], [0, 0, 0])
    with pytest.raises(ValueError, match="regularity"):
        RoutingSearchCost(0, 1.0, (np.nan,))


def test_repeated_join_coherence_couples_only_same_group_normalized_bends():
    assert join_bend_consistency([2.0, 2.0, 10.0], [0, 0, 1]) == 0
    assert join_bend_consistency([2.0, 5.0, 10.0], [0, 0, None]) > 0
    assert join_bend_consistency([1e300, 1e300], [0, 0]) == 0
    theta = np.linspace(0, np.pi / 2, 65)
    arc = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    owner = ("contour_group", 0, -1, -1, -1, "joint_spiral_join")
    points = np.vstack((arc, 2 * arc + [4, 0, 0]))
    mask = [True] * 64 + [False] + [True] * 64
    ids = [owner] * 64 + [None] + [owner] * 64
    values = added_route_regularity(points, mask, span_ids=ids)
    assert values["added_join_bend_consistency"] == pytest.approx(0, abs=1e-20)
