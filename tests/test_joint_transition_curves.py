"""Whole return/connector proposals preserve the actual directed attachments."""

import numpy as np
import pytest

from coilgen.connector_bridges import _evaluate, _maximum_distance_to_polyline
from coilgen.joint_transition_curves import (
    _DistanceBudgetExceeded,
    _SegmentDistanceIndex,
    _local_clearance,
    _minimum_speed,
    _self_contact_reason,
    generate_joint_transition_curves,
    generate_axial_return_curves,
)
from coilgen.routing_objective import routing_cost


LIMITS = dict(
    maximum_height_m=0.002,
    minimum_radius_m=0.0012,
    maximum_segment_length_m=0.001,
    chord_tolerance_m=1e-5,
    maximum_tangent_angle_deg=2.0,
)


def _facing_transition():
    # Frozen corners of the actual rank-48 facing-corridor source span327:342.
    # Omitted samples are collinear. These tangents are the neighbouring
    # protected polygon chords, not ideal circle tangents at the attachments.
    source = np.array(
        [
            [-0.010370264394014185, -0.0024056463960341907, 0],
            [-0.010370264394014183, 0.0024056463960341855, 0],
            [-0.008656266414471595, 0.0029460678831249873, 0],
            [0.006555396591021275, -0.003608469594051277, 0],
        ]
    )
    incoming = np.array([8.831778041720924e-5, 0.00026400099270389946, 0])
    outgoing = np.array([0.00013247667062581125, -0.000396001489055844, 0])
    normal = np.array([0.0, 0, 1])
    obstacles = np.array([[[0, -0.01, 0], [0, 0.01, 0]]], dtype=float)
    return source, incoming, outgoing, normal, obstacles


def test_frozen_facing_transition_has_finite_radius_options_with_exact_tangents():
    source, incoming, outgoing, normal, obstacles = args = _facing_transition()
    snapshots = [value.copy() for value in args]
    generation = generate_joint_transition_curves(*args, **LIMITS)
    assert generation.reason == "candidates_available" and 1 < len(generation.candidates) <= 8
    heights = {candidate.height_m for candidate in generation.candidates}
    assert {0.0, 0.001, 0.0015, 0.002} <= heights
    for candidate in generation.candidates:
        np.testing.assert_array_equal(candidate.points[[0, -1]], source[[0, -1]])
        _, derivatives, _, _ = _evaluate(
            candidate.control_points_m, normal, candidate.height_m, candidate.peak_parameter, [0, 1]
        )
        np.testing.assert_allclose(
            derivatives / np.linalg.norm(derivatives, axis=1)[:, None],
            np.array([incoming / np.linalg.norm(incoming), outgoing / np.linalg.norm(outgoing)]),
            atol=2e-15,
        )
        # Independent dense derivative grid, deliberately unrelated to output
        # samples and the generator's shoulder grids.
        dense, first, second, _ = _evaluate(
            candidate.control_points_m, normal, candidate.height_m, candidate.peak_parameter, np.linspace(0, 1, 20003)
        )
        curvature = np.linalg.norm(np.cross(first, second), axis=1) / np.linalg.norm(first, axis=1) ** 3
        assert 1 / curvature.max() >= LIMITS["minimum_radius_m"] * 1.004
        assert _maximum_distance_to_polyline(dense, candidate.points) <= LIMITS["chord_tolerance_m"] * 1.001
        assert np.max(candidate.points[:, 2]) <= LIMITS["maximum_height_m"] + 1e-15
        assert np.min(candidate.points[:, 2]) >= -1e-15
        assert np.max(np.linalg.norm(np.diff(candidate.points, axis=0), axis=1)) <= 0.001
        assert np.all(np.diff(candidate.parameters) > 1e-12)
        assert np.all(np.diff(candidate.source_fractions) > 0)
    for value, snapshot in zip(args, snapshots):
        np.testing.assert_array_equal(value, snapshot)


def test_reentrant_tangent_can_be_a_regular_simple_curve():
    source = np.array([[0, 0, 0], [-0.003, 0.008, 0], [0.02, 0, 0]])
    incoming, outgoing = np.array([-1.0, 1, 0]), np.array([1.0, -1, 0])
    assert incoming @ (source[-1] - source[0]) < 0
    result = generate_joint_transition_curves(
        source, incoming, outgoing, [0, 0, 1], np.empty((0, 2, 3)), **{**LIMITS, "minimum_radius_m": 0.001}
    )
    assert result.candidates, result.rejected_reasons
    for candidate in result.candidates:
        assert _self_contact_reason(candidate.points, 1e-12) == ""
        assert (
            _minimum_speed(candidate.control_points_m, candidate.normal, candidate.height_m, candidate.peak_parameter)
            > 1e-5
        )


def test_hidden_analytic_zero_speed_is_rejected_without_sampling_luck():
    controls = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0.5, 0, 0]], dtype=float)
    assert _minimum_speed(controls, np.array([0, 0, 1]), 0, 0.5) < 1e-12
    result = generate_joint_transition_curves(
        [[0, 0, 0], [0.02, 0, 0]],
        [-1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        np.empty((0, 2, 3)),
        **{**LIMITS, "maximum_height_m": 0},
    )
    assert not result.candidates and "curve_cusp_or_zero_speed" in result.rejected_reasons


def test_nonincident_loop_contact_is_rejected():
    points = np.array([[-1, -1, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0]], dtype=float)
    assert _self_contact_reason(points, 1e-12) == "curve_self_intersection"
    assert _self_contact_reason(points + np.arange(4)[:, None] * [0, 0, 0.1], 1e-12) == ""


def test_distance_broad_phase_keeps_the_actual_nearest_obstacle():
    from coilgen.connector_bridges import _curve_clearance

    source, incoming, outgoing, normal, _ = args = _facing_transition()
    candidate = generate_joint_transition_curves(*args, **LIMITS, maximum_candidates=1).candidates[0]
    rng = np.random.default_rng(31)
    obstacles = rng.uniform(-0.05, 0.05, size=(300, 2, 3))
    measured = _local_clearance(candidate.points, candidate.control_points_m, normal, candidate.height_m, obstacles)
    assert measured == pytest.approx(_curve_clearance(candidate.points, obstacles), abs=1e-15)


def test_scaled_rotated_inputs_preserve_radius_and_directed_geometry():
    source, incoming, outgoing, normal, obstacles = args = _facing_transition()
    first = generate_joint_transition_curves(*args, **LIMITS, maximum_candidates=1).candidates[0]
    rotation = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    scale, translation = 3.0, np.array([0.1, -0.2, 0.4])
    transformed = (
        scale * source @ rotation.T + translation,
        incoming @ rotation.T,
        outgoing @ rotation.T,
        normal @ rotation.T,
        scale * obstacles @ rotation.T + translation,
    )
    limits = {key: value if key.endswith("deg") else value * scale for key, value in LIMITS.items()}
    second = generate_joint_transition_curves(*transformed, **limits, maximum_candidates=1).candidates[0]
    assert second.minimum_radius_m == pytest.approx(first.minimum_radius_m * scale, rel=1e-10)
    np.testing.assert_allclose(second.points, scale * first.points @ rotation.T + translation, atol=1e-14)


def test_source_collinear_subdivision_does_not_change_proposals():
    source, incoming, outgoing, normal, obstacles = args = _facing_transition()
    subdivided = np.vstack(
        [*(a + np.arange(8)[:, None] / 8 * (b - a) for a, b in zip(source[:-1], source[1:])), source[-1:]]
    )
    first = generate_joint_transition_curves(*args, **LIMITS, maximum_candidates=1).candidates[0]
    second = generate_joint_transition_curves(
        subdivided, incoming, outgoing, normal, obstacles, **LIMITS, maximum_candidates=1
    ).candidates[0]
    np.testing.assert_array_equal(second.points, first.points)
    assert second.maximum_deviation_m == pytest.approx(first.maximum_deviation_m, abs=1e-15)


def test_generated_options_follow_b0_preference_with_sign_and_rotation_invariance():
    source, incoming, outgoing, normal, _ = _facing_transition()
    obstacles = np.empty((0, 2, 3))
    args = source, incoming, outgoing, normal, obstacles
    direction = np.array([0.0, 1, 0])
    first = generate_joint_transition_curves(*args, **LIMITS, maximum_candidates=1, b0_weight=0).candidates[0]
    weighted = generate_joint_transition_curves(
        *args, **LIMITS, maximum_candidates=1, b0_direction=direction, b0_weight=4
    ).candidates[0]
    assert routing_cost(weighted.points, direction, 4) < routing_cost(first.points, direction, 4) - 1e-4
    rotation = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    transformed = (source @ rotation.T, incoming @ rotation.T, outgoing @ rotation.T, normal @ rotation.T, obstacles)
    reversed_rotated = generate_joint_transition_curves(
        *transformed, **LIMITS, maximum_candidates=1, b0_direction=-3 * (rotation @ direction), b0_weight=4
    ).candidates[0]
    np.testing.assert_allclose(reversed_rotated.points, weighted.points @ rotation.T, atol=1e-14)
    assert reversed_rotated.minimum_radius_m == pytest.approx(weighted.minimum_radius_m, rel=1e-12)


def test_spatial_index_matches_exhaustive_segment_distance_and_fails_before_large_allocation():
    from coilgen.connector_bridges import _curve_clearance

    rng = np.random.default_rng(734)
    points = rng.normal(size=(17, 3))
    obstacles = rng.normal(size=(41, 2, 3))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    index = _SegmentDistanceIndex(obstacles)
    assert index.gap(segments, 0.1) == pytest.approx(min(0.1, _curve_clearance(points, obstacles)), abs=1e-15)
    with pytest.raises(_DistanceBudgetExceeded, match="curve_distance_pair_budget_exceeded"):
        index.gap(segments, 0.1, maximum_pairs=1)


def test_numpy_only_spatial_index_preserves_distances_without_scipy(monkeypatch):
    import builtins
    from coilgen.connector_bridges import _curve_clearance

    real_import = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name == "scipy.spatial":
            raise ImportError("Fixture without optional SciPy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    rng = np.random.default_rng(345)
    points = rng.normal(size=(19, 3))
    obstacles = rng.normal(size=(300, 2, 3))
    index = _SegmentDistanceIndex(obstacles)
    assert index.tree is None and index.fallback_tree is not None
    segments = np.stack((points[:-1], points[1:]), axis=1)
    assert index.gap(segments, 0.1) == pytest.approx(min(0.1, _curve_clearance(points, obstacles)), abs=1e-15)
    with pytest.raises(_DistanceBudgetExceeded, match="curve_distance_pair_budget_exceeded"):
        index.gap(segments, 0.1, maximum_pairs=1)


@pytest.mark.parametrize("direction,weight", [([0, 0, 0], 1), ([0, np.nan, 1], 1), ([0, 0, 1], -1)])
def test_invalid_b0_objective_cannot_authorize_curves(direction, weight):
    result = generate_joint_transition_curves(*_facing_transition(), **LIMITS, b0_direction=direction, b0_weight=weight)
    assert result.reason == "invalid_b0_objective" and not result.candidates


def test_oversized_source_reports_input_budget_without_geometry_search():
    source = np.linspace([0.0, 0, 0], [0.02, 0, 0], 4097)
    result = generate_joint_transition_curves(source, [1, 0, 0], [1, 0, 0], [0, 0, 1], np.empty((0, 2, 3)), **LIMITS)
    assert result.reason == "curve_input_budget_exceeded" and not result.candidates


@pytest.mark.parametrize(
    "setting,value",
    [("maximum_height_m", -1), ("minimum_radius_m", 0), ("maximum_candidates", 0), ("chord_tolerance_m", np.nan)],
)
def test_invalid_limits_cannot_authorize_a_curve(setting, value):
    result = generate_joint_transition_curves(*_facing_transition(), **{**LIMITS, setting: value})
    assert result.reason == "invalid_limits" and not result.candidates


def test_unavailable_bend_radius_or_corridor_is_explicit():
    args = _facing_transition()
    radius_failure = generate_joint_transition_curves(*args, **{**LIMITS, "minimum_radius_m": 1.0})
    assert not radius_failure.candidates and "insufficient_bend_radius" in radius_failure.rejected_reasons
    deviation_failure = generate_joint_transition_curves(*args, **LIMITS, maximum_deviation_m=1e-8)
    assert not deviation_failure.candidates and "deviation_limit" in deviation_failure.rejected_reasons


def test_waypoint_corridor_avoids_obstacle_that_blocks_every_zero_lift_direct_arch():
    from coilgen.joint_transition_curves import generate_joint_corridor_curves

    source = np.array([[-0.02, 0, 0], [-0.01, 0.012, 0], [0.01, 0.012, 0], [0.02, 0, 0]])
    obstacle = np.array([[[0, -0.005, 0], [0, 0.005, 0]]])
    args = source, [1, 0, 0], [1, 0, 0], [0, 0, 1], obstacle
    limits = {**LIMITS, "maximum_height_m": 0, "clearance_score_cap_m": 0.001}
    direct = generate_joint_transition_curves(*args, **limits)
    assert direct.candidates and all(curve.local_clearance_m < 1e-10 for curve in direct.candidates)
    corridors = generate_joint_corridor_curves(*args, **limits)
    assert corridors.candidates, corridors.rejected_reasons
    for curve in corridors.candidates:
        assert curve.profile_family == "support_waypoints"
        assert curve.local_clearance_m == pytest.approx(0.001)
        assert curve.minimum_radius_m >= limits["minimum_radius_m"] * 1.005
        np.testing.assert_array_equal(curve.points[[0, -1]], source[[0, -1]])
        np.testing.assert_array_equal(curve.points[:, 2], 0)
        derivatives = []
        for controls in curve.control_points_m:
            _, first, second, _ = _evaluate(controls, curve.normal, 0, 0.5, np.linspace(0, 1, 15001))
            derivatives.append(first[[0, -1]] / np.linalg.norm(first[[0, -1]], axis=1)[:, None])
            curvature = np.linalg.norm(np.cross(first, second), axis=1) / np.linalg.norm(first, axis=1) ** 3
            assert 1 / curvature.max() >= limits["minimum_radius_m"] * 1.004
        np.testing.assert_allclose(derivatives[0][0], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(derivatives[-1][-1], [1, 0, 0], atol=1e-12)
        for before, after in zip(derivatives[:-1], derivatives[1:]):
            np.testing.assert_allclose(before[1], after[0], atol=1e-12)


def test_b0_corridors_preserve_inputs_and_are_sign_rotation_invariant():
    from coilgen.joint_transition_curves import _corridor_waypoints

    source = np.array([[-0.02, 0, 0], [0.02, 0.03, 0]])
    direction = np.array([0.0, 9.0, 0.0])
    original = direction.copy()
    first = _corridor_waypoints(source, np.array([0.0, 0, 1]), direction, 0.0001)
    np.testing.assert_array_equal(direction, original)
    assert len(first) == 2 and {family for family, _ in first} == {"b0_corridor"}
    rotation = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    second = _corridor_waypoints(source @ rotation.T, rotation @ [0, 0, 1], -rotation @ direction, 0.0001)
    for (family1, points1), (family2, points2) in zip(first, second):
        assert family1 == family2
        np.testing.assert_allclose(points2, points1 @ rotation.T, atol=1e-15)


def test_corridor_family_respects_impossible_bend_radius():
    from coilgen.joint_transition_curves import generate_joint_corridor_curves

    source = np.array([[-0.02, 0, 0], [-0.01, 0.012, 0], [0.01, 0.012, 0], [0.02, 0, 0]])
    result = generate_joint_corridor_curves(
        source, [1, 0, 0], [1, 0, 0], [0, 0, 1], np.empty((0, 2, 3)), **{**LIMITS, "minimum_radius_m": 0.02}
    )
    assert not result.candidates and "insufficient_bend_radius" in result.rejected_reasons


def test_direct_curve_self_contact_budget_propagates_as_exhaustion(monkeypatch):
    from coilgen import joint_transition_curves as curves

    monkeypatch.setattr(curves, "_self_contact_reason", lambda *_: "self_contact_pair_budget_exceeded")
    result = curves.generate_joint_transition_curves(*_facing_transition(), **LIMITS)
    assert result.reason == "self_contact_pair_budget_exceeded" and not result.candidates


def test_corridor_invalid_obstacle_shape_is_rejected_without_reshape_exception():
    from coilgen.joint_transition_curves import generate_joint_corridor_curves

    source, incoming, outgoing, normal, _ = _facing_transition()
    result = generate_joint_corridor_curves(source, incoming, outgoing, normal, np.zeros((2, 2)), **LIMITS)
    assert result.reason == "invalid_geometry_or_input_budget" and not result.candidates


def test_axial_return_has_actual_3d_b0_span_and_keeps_both_directed_bends():
    from coilgen.connector_bridges import _baseline
    from coilgen.physics import biot_savart_wire_path

    source = np.array([[-0.03, 0, 0], [0.03, 0.004, 0]])
    b0 = np.array([1.0, 0.2, 0.3])
    b0 /= np.linalg.norm(b0)
    limits = {**LIMITS, "maximum_height_m": 0.008}
    result = generate_axial_return_curves(
        source, [1, 0, 0], [1, 0, 0], [0, 0, 1], np.empty((0, 2, 3)), **limits, b0_direction=b0
    )
    assert result.candidates, result.rejected_reasons
    for candidate in result.candidates:
        assert candidate.profile_family == "b0_straight_return"
        np.testing.assert_array_equal(candidate.points[[0, -1]], source)
        controls = candidate.control_points_m
        assert controls.shape == (3, 4, 3)
        middle = controls[1]
        delta = middle[-1] - middle[0]
        direction = delta / np.linalg.norm(delta)
        assert abs(direction @ b0) == pytest.approx(1.0, abs=1e-14)
        assert abs(direction[2]) > 0.2  # Not B0 projected into the support plane.
        np.testing.assert_allclose(middle, middle[0] + np.linspace(0, 1, 4)[:, None] * delta, atol=1e-15)
        derivatives = [_baseline(control, np.linspace(0, 1, 20003))[1:] for control in controls]
        np.testing.assert_allclose(derivatives[0][0][0] / np.linalg.norm(derivatives[0][0][0]), [1, 0, 0], atol=1e-14)
        np.testing.assert_allclose(
            derivatives[-1][0][-1] / np.linalg.norm(derivatives[-1][0][-1]), [1, 0, 0], atol=1e-14
        )
        for left, right in zip(derivatives[:-1], derivatives[1:]):
            np.testing.assert_allclose(
                left[0][-1] / np.linalg.norm(left[0][-1]), right[0][0] / np.linalg.norm(right[0][0]), atol=1e-14
            )
        for first, second in derivatives:
            curvature = np.linalg.norm(np.cross(first, second), axis=1) / np.linalg.norm(first, axis=1) ** 3
            assert np.max(curvature) <= 1 / (limits["minimum_radius_m"] * 1.004)
        assert _self_contact_reason(candidate.points, 1e-12) == ""
        assert len(candidate.points) > 10
        # A straight span contributes no longitudinal Biot--Savart component;
        # neither curved end is removed from the complete returned geometry.
        observation = np.array([0.003, -0.005, 0.02])
        assert abs(np.cross(delta, observation - middle[0]) @ b0) < 1e-18
        assert np.linalg.norm(np.cross(controls[0][1] - controls[0][0], b0)) > 1e-4
        middle_field = biot_savart_wire_path(middle[[0, -1]], observation[None])
        complete_field = biot_savart_wire_path(candidate.points, observation[None])
        assert abs(float(middle_field[0] @ b0)) < 1e-12 * np.linalg.norm(middle_field[0])
        assert abs(float(complete_field[0] @ b0)) > 1e-10


def test_axial_returns_are_b0_sign_and_rigid_rotation_invariant():
    source = np.array([[-0.03, 0, 0], [0.03, 0.004, 0]])
    b0 = np.array([1.0, 0.2, 0.3])
    empty = np.empty((0, 2, 3))
    limits = {**LIMITS, "maximum_height_m": 0.008}
    first = generate_axial_return_curves(source, [1, 0, 0], [1, 0, 0], [0, 0, 1], empty, **limits, b0_direction=b0)
    rotation = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    second = generate_axial_return_curves(
        source @ rotation.T,
        rotation @ [1, 0, 0],
        rotation @ [1, 0, 0],
        rotation @ [0, 0, 1],
        empty,
        **limits,
        b0_direction=-7 * (rotation @ b0),
    )
    assert len(first.candidates) == len(second.candidates) > 0
    for a, b in zip(first.candidates, second.candidates):
        np.testing.assert_allclose(b.points, a.points @ rotation.T, atol=1e-13)
        assert b.minimum_radius_m == pytest.approx(a.minimum_radius_m, rel=1e-12)


def test_axial_family_does_not_relax_an_impossible_bend_radius():
    result = generate_axial_return_curves(
        [[-0.03, 0, 0], [0.03, 0.004, 0]],
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        np.empty((0, 2, 3)),
        **{**LIMITS, "maximum_height_m": 0.008, "minimum_radius_m": 1.0},
        b0_direction=[1, 0.2, 0.3],
    )
    assert not result.candidates
    assert "insufficient_bend_radius" in result.rejected_reasons


def test_axial_family_can_retain_both_signed_straight_span_directions():
    result = generate_axial_return_curves(
        [[-0.2, -0.2, 0], [0.2, 0.2, 0]],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        np.empty((0, 2, 3)),
        **{**LIMITS, "maximum_height_m": 0.02, "maximum_segment_length_m": 0.01, "chord_tolerance_m": 1e-4},
        b0_direction=[1, 0, 0],
    )
    directions = {
        np.sign(curve.control_points_m[1, -1, 0] - curve.control_points_m[1, 0, 0]) for curve in result.candidates
    }
    assert directions == {-1, 1}
