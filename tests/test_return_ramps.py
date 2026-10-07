"""Actual-attachment return ramps are geometric proposals, not body certificates."""

import numpy as np
import pytest

from coilgen.connector_bridges import generate_return_ramps


def _saved_return(site):
    # Metres, exact actual trial endpoints and neighboring tangent vectors from
    # the Sept16 GY replay. These short fixtures do not depend on saved outputs.
    if site == 8506:
        points = [
            [-0.00315269454809086, -0.06009280555795769, 0.09069411052527152],
            [0.00788177088873071, -0.05985031520895855, 0.09198243764271836],
            [0.00922749385754682, -0.05172457499632435, 0.09142501914149416],
        ]
        before = [1.1736660759629125e-4, 3.2155052523408223e-6, 4.8614864123913692e-5]
        after = [9.573299438347619e-4, 5.722285384999498e-3, -3.965384714170739e-4]
        normal = [-3.3399355753547716e-7, -3.08263484855139e-7, 0.9999999999998967]
        under = [
            [-0.00154737313901161, -0.05178611124526117, 0.09135905661229213],
            [0.00673371940917333, -0.05985031520895855, 0.08921079953497664],
        ]
    else:
        points = [
            [-0.00685883834609783, 0.05986538396767294, -0.0891589759030842],
            [0.00382684632680724, 0.06008546706864154, -0.09363975432772954],
            [0.00159139471871958, 0.05180632918196029, -0.09446132605739964],
        ]
        before = [2.410385571536668e-4, 1.8377730512525448e-5, -9.984128933938907e-5]
        after = [-1.7046331580793676e-3, -5.6274541529156e-3, -5.015955510939557e-4]
        normal = [5.8472982735647115e-7, 4.1272738482312433e-7, -0.9999999999997439]
        under = [
            [-0.00830828992547309, 0.05169354614036564, -0.08855859141492652],
            [0.00301087760492621, 0.06008546686787972, -0.09075285373281607],
        ]
    return dict(
        source_points=np.asarray(points),
        tangent_before=np.asarray(before),
        tangent_after=np.asarray(after),
        normal=np.asarray(normal),
        underpass_segments=np.asarray([under]),
        maximum_height_m=0.003,
        minimum_radius_m=0.002715436306690595,
        maximum_segment_length_m=0.005,
        chord_tolerance_m=1e-5,
    )


def _positions(candidate, t):
    controls = candidate.control_points_m
    weights = np.array([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t**3]).T
    peak = candidate.peak_parameter
    if candidate.profile_family == "quartic":
        u = (1 - peak) * t / (peak + (1 - 2 * peak) * t)
        lift = 16 * candidate.height_m * u * u * (1 - u) ** 2
    else:
        u = np.where(t <= peak, t / peak, (1 - t) / (1 - peak))
        lift = candidate.height_m * (6 * u**5 - 15 * u**4 + 10 * u**3)
    return weights @ controls + lift[:, None] * candidate.normal


@pytest.mark.parametrize("site", [8506, 18766])
def test_saved_actual_endpoint_return_has_bounded_early_ramp(site):
    inputs = _saved_return(site)
    generated = generate_return_ramps(**inputs)
    assert 0 < len(generated.candidates) <= 64
    expected_height, expected_handles, family = (
        (0.001, (0.35, 0.45), "quartic") if site == 8506 else (0.0024, (0.45, 0.65), "quintic")
    )
    span = np.linalg.norm(inputs["source_points"][-1] - inputs["source_points"][0])
    matching = []
    for candidate in generated.candidates:
        controls = candidate.control_points_m
        handles = [np.linalg.norm(controls[1] - controls[0]) / span, np.linalg.norm(controls[-1] - controls[-2]) / span]
        if (
            candidate.height_m == pytest.approx(expected_height)
            and np.allclose(handles, expected_handles)
            and candidate.peak_parameter == pytest.approx(0.35)
            and candidate.profile_family == family
        ):
            matching.append(candidate)
    assert len(matching) == 1
    candidate = matching[0]
    np.testing.assert_array_equal(candidate.points[[0, -1]], inputs["source_points"][[0, -1]])
    assert np.all(np.diff(candidate.source_fractions) > 0)
    grid = np.linspace(0, 1, 20001)
    position = _positions(candidate, grid)
    first = np.gradient(position, grid, axis=0, edge_order=2)
    second = np.gradient(first, grid, axis=0, edge_order=2)
    curvature = np.linalg.norm(np.cross(first, second), axis=1) / np.linalg.norm(first, axis=1) ** 3
    assert 1 / curvature.max() >= inputs["minimum_radius_m"]
    for actual, expected in [(first[0], inputs["tangent_before"]), (first[-1], inputs["tangent_after"])]:
        np.testing.assert_allclose(actual / np.linalg.norm(actual), expected / np.linalg.norm(expected), atol=1e-6)
    # The hump vanishes at the endpoints; the actual lifted endpoint is retained
    # in the baseline. These numbers must never be mislabeled total support lift.
    np.testing.assert_allclose(candidate.lift_distances_m[[0, -1]], 0, atol=1e-15)


def test_return_ramp_set_is_invariant_to_path_reversal():
    inputs = _saved_return(8506)
    forward = generate_return_ramps(**inputs)
    reverse_inputs = dict(
        inputs,
        source_points=inputs["source_points"][::-1],
        tangent_before=-inputs["tangent_after"],
        tangent_after=-inputs["tangent_before"],
    )
    backward = generate_return_ramps(**reverse_inputs)
    assert len(forward.candidates) == len(backward.candidates)
    # Match analytic shapes, independent of tiny clearance sorting roundoff.
    unmatched = list(backward.candidates)
    for candidate in forward.candidates:
        matched = next(
            (
                other
                for other in unmatched
                if other.profile_family == candidate.profile_family
                and other.height_m == pytest.approx(candidate.height_m)
                and np.allclose(other.control_points_m[::-1], candidate.control_points_m, atol=1e-13)
                and other.peak_parameter == pytest.approx(1 - candidate.peak_parameter)
            ),
            None,
        )
        assert matched is not None
        np.testing.assert_allclose(
            _positions(candidate, np.linspace(0, 1, 129)), _positions(matched, np.linspace(1, 0, 129)), atol=1e-13
        )
        unmatched = [other for other in unmatched if other is not matched]
