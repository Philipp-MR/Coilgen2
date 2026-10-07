"""Saved connector and independent geometric checks for bounded bridge proposals."""

import numpy as np
import pytest

from coilgen.connector_bridges import generate_connector_bridges


def _saved_gz():
    # Frozen pre-lift connector 0 closing, with its two untouched contour neighbors.
    # Metres, from the 2026-09-14 GZ provenance; no output-directory dependency.
    neighbors = np.array(
        [
            [0.08387906698195316, 0.029974521551022263, 0.03650515970337379],
            [0.08387734188330911, 0.030279507089369388, 0.03650932444320956],
            [0.0801581097810807, 0.04007697594782451, 0.045488345912875865],
            [0.08015406373548829, 0.0401555649884186, 0.04549811387967778],
            [0.08012142743593897, 0.040757863384129, 0.045576904490308996],
        ]
    )
    underpass = np.array([[[0.0806022838, 0.0301445586, 0.04441601545], [0.08381019733, 0.04027796716, 0.03667142031]]])
    return {
        "source_points": neighbors[1:4],
        "tangent_before": neighbors[1] - neighbors[0],
        "tangent_after": neighbors[-1] - neighbors[-2],
        "normal": np.array([0.9238795540451957, 7.471430119899305e-8, 0.3826833803776234]),
        "underpass_segments": underpass,
        "maximum_height_m": 0.003,
        "minimum_radius_m": 0.00271544,
        "maximum_segment_length_m": 0.001,
        "chord_tolerance_m": 1e-5,
    }


def _straight(crossing=0.5):
    return {
        "source_points": np.array([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0]]),
        "tangent_before": [1.0, 0.0, 0.0],
        "tangent_after": [1.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "underpass_segments": np.array([[[0.04 * crossing, -0.02, 0.0], [0.04 * crossing, 0.02, 0.0]]]),
        "maximum_height_m": 0.003,
        "minimum_radius_m": 0.00271544,
        "maximum_segment_length_m": 0.001,
        "chord_tolerance_m": 1e-5,
    }


def _sample_independently(candidate, parameters):
    """Evaluate position only; derivatives in tests use independent differences."""
    t = np.asarray(parameters)
    controls = candidate.control_points_m
    weights = np.array([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t**3]).T
    if candidate.profile_family == "quartic":
        u = (1 - candidate.peak_parameter) * t / (candidate.peak_parameter + (1 - 2 * candidate.peak_parameter) * t)
        h = 16 * candidate.height_m * u * u * (1 - u) ** 2
    else:
        u = np.where(
            t <= candidate.peak_parameter, t / candidate.peak_parameter, (1 - t) / (1 - candidate.peak_parameter)
        )
        h = candidate.height_m * (6 * u**5 - 15 * u**4 + 10 * u**3)
    return weights @ controls + h[:, None] * candidate.normal


def test_saved_gz_rebuilds_whole_span_with_fixed_endpoints_and_bounded_curvature():
    inputs = _saved_gz()
    result = generate_connector_bridges(**inputs)
    assert result.reason == "candidates_available"
    assert 1 <= len(result.candidates) <= 8
    candidate = result.candidates[0]
    np.testing.assert_array_equal(candidate.points[[0, -1]], inputs["source_points"][[0, -1]])
    assert candidate.minimum_radius_m >= inputs["minimum_radius_m"]
    assert 0.45 < candidate.peak_parameter < 0.55  # Actual crossing, not the 79 um landing.
    assert candidate.local_clearance_m < 0.006  # Never claim the configured 6 mm is met.
    assert candidate.local_clearance_m > 0.0029
    assert np.max(candidate.lift_distances_m) <= 0.003 + 1e-15
    assert np.min(candidate.lift_distances_m) >= 0
    assert np.max(candidate.lift_distances_m) == pytest.approx(0.003)
    assert np.all(np.diff(candidate.source_fractions) > 0)
    np.testing.assert_array_equal(candidate.source_fractions[[0, -1]], [0.0, 1.0])

    grid = np.linspace(0.0, 1.0, 20001)
    positions = _sample_independently(candidate, grid)
    first = np.gradient(positions, grid, axis=0, edge_order=2)
    second = np.gradient(first, grid, axis=0, edge_order=2)
    curvature = np.linalg.norm(np.cross(first, second), axis=1) / np.linalg.norm(first, axis=1) ** 3
    assert 1 / curvature.max() >= inputs["minimum_radius_m"]
    chord = candidate.points[-1] - candidate.points[0]
    assert np.min(first @ chord) > 0
    for actual, expected in [(first[0], inputs["tangent_before"]), (first[-1], inputs["tangent_after"])]:
        np.testing.assert_allclose(actual / np.linalg.norm(actual), expected / np.linalg.norm(expected), atol=1e-6)


def test_short_asymmetric_saved_gz_crossing_has_lower_radius_valid_proposals():
    # Independent GZ connector at source6061:6063, frozen from the Sept16 run.
    # The crossing sits early in the approach; every full-height candidate
    # misses the required radius. Lower proposals do not certify body clearance.
    inputs = {
        "source_points": np.array(
            [
                [0.0839685493692387, 0.01144875990052689, -0.03628912212147978],
                [0.08015745425459582, 0.00137561479540945, -0.04548992773950448],
                [0.07859034566800104, -0.00472950538868443, -0.04927326588759361],
            ]
        ),
        "tangent_before": [-5.9625047384159746e-7, -2.9397851370466244e-4, -1.4399613409016299e-6],
        "tangent_after": [-9.7240137580945074e-8, -8.9499034028614824e-6, -2.3477331866084139e-7],
        "normal": [0.9238795418750072, 3.502969414171817e-7, -0.38268340975890786],
        "underpass_segments": np.array(
            [
                [
                    [0.08109578533737723, 0.01107034727696758, -0.04322459273172976],
                    [0.08507900791150827, 0.00187902642757678, -0.03360824710765734],
                ]
            ]
        ),
        "maximum_height_m": 0.003,
        "minimum_radius_m": 0.002715436306690595,
        "maximum_segment_length_m": 0.001,
        "chord_tolerance_m": 1e-5,
        "planarity_tolerance_m": 1e-7,
    }
    result = generate_connector_bridges(**inputs)
    assert 1 <= len(result.candidates) <= 8
    assert "insufficient_bend_radius" in result.rejected_reasons
    assert result.candidates[0].height_m == pytest.approx(0.0021)
    assert result.candidates[0].minimum_radius_m == pytest.approx(0.002770899, abs=1e-8)
    for candidate in result.candidates:
        assert 0.002 <= candidate.height_m <= 0.0021 + 1e-15
        np.testing.assert_array_equal(candidate.points[[0, -1]], inputs["source_points"][[0, -1]])
        grid = np.linspace(0, 1, 20001)
        positions = _sample_independently(candidate, grid)
        first = np.gradient(positions, grid, axis=0, edge_order=2)
        second = np.gradient(first, grid, axis=0, edge_order=2)
        curvature = np.linalg.norm(np.cross(first, second), axis=1) / np.linalg.norm(first, axis=1) ** 3
        assert 1 / curvature.max() >= inputs["minimum_radius_m"]
        for actual, key in [(first[0], "tangent_before"), (first[-1], "tangent_after")]:
            expected = np.asarray(inputs[key])
            np.testing.assert_allclose(actual / np.linalg.norm(actual), expected / np.linalg.norm(expected), atol=1e-6)
        assert np.all(np.diff(candidate.source_fractions) > 0)
        assert 0 < candidate.local_clearance_m < 0.002661128  # Still needs the exact finite-body check.

    reversed_inputs = dict(inputs)
    reversed_inputs.update(
        source_points=inputs["source_points"][::-1],
        tangent_before=-np.asarray(inputs["tangent_after"]),
        tangent_after=-np.asarray(inputs["tangent_before"]),
    )
    reversed_result = generate_connector_bridges(**reversed_inputs)
    assert len(reversed_result.candidates) == len(result.candidates)
    for forward, reverse in zip(result.candidates, reversed_result.candidates):
        np.testing.assert_allclose(forward.points, reverse.points[::-1], atol=1e-13)
        assert forward.minimum_radius_m == pytest.approx(reverse.minimum_radius_m, rel=1e-11)


def test_adaptive_samples_bound_segment_length_and_between_sample_chord_error():
    inputs = _saved_gz()
    candidate = generate_connector_bridges(**inputs).candidates[0]
    assert np.max(np.linalg.norm(np.diff(candidate.points, axis=0), axis=1)) <= inputs["maximum_segment_length_m"]
    # Eleven interior checkpoints are independent of the generator's five-point test.
    for index, (start, end) in enumerate(zip(candidate.parameters[:-1], candidate.parameters[1:])):
        parameters = np.linspace(start, end, 13)
        positions = _sample_independently(candidate, parameters)
        linear = candidate.points[index] + np.linspace(0, 1, 13)[:, None] * (
            candidate.points[index + 1] - candidate.points[index]
        )
        assert np.max(np.linalg.norm(positions - linear, axis=1)) <= inputs["chord_tolerance_m"] * 1.01
        epsilon = 1e-6
        tangents = (
            _sample_independently(candidate, parameters + epsilon)
            - _sample_independently(candidate, parameters - epsilon)
        ) / (2 * epsilon)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        frame = candidate.normal - (tangents @ candidate.normal)[:, None] * tangents
        frame /= np.linalg.norm(frame, axis=1)[:, None]
        assert np.min(tangents @ tangents.T) >= np.cos(np.radians(2.01))
        assert np.min(frame @ frame.T) >= np.cos(np.radians(2.01))


def test_collinear_source_and_underpass_subdivision_does_not_change_candidates():
    inputs = _saved_gz()
    first = generate_connector_bridges(**inputs)
    source = inputs["source_points"]
    inputs["source_points"] = np.vstack(
        [np.linspace(a, b, 11)[:-1] for a, b in zip(source[:-1], source[1:])] + [source[-1:]]
    )
    original = inputs["underpass_segments"][0]
    subdivided = np.linspace(original[0], original[1], 9)
    inputs["underpass_segments"] = np.stack((subdivided[:-1], subdivided[1:]), axis=1)
    second = generate_connector_bridges(**inputs)
    assert len(first.candidates) == len(second.candidates)
    for left, right in zip(first.candidates, second.candidates):
        np.testing.assert_allclose(left.points, right.points, atol=1e-14)
        np.testing.assert_allclose(left.source_fractions, right.source_fractions, atol=1e-13)
        assert left.minimum_radius_m == pytest.approx(right.minimum_radius_m, rel=1e-12)


def test_asymmetric_crossing_sets_continuous_peak_and_preserves_normal_sign():
    inputs = _straight(0.30)
    positive = generate_connector_bridges(**inputs)
    assert positive.candidates
    inputs["normal"] = [0.0, 0.0, -1.0]
    negative = generate_connector_bridges(**inputs)
    assert len(positive.candidates) == len(negative.candidates)
    for up, down in zip(positive.candidates, negative.candidates):
        peak = _sample_independently(up, [up.peak_parameter])[0]
        assert peak[0] == pytest.approx(0.012, abs=1e-12)
        assert peak[2] == pytest.approx(0.003)
        np.testing.assert_allclose(up.points[:, :2], down.points[:, :2], atol=1e-14)
        np.testing.assert_allclose(up.points[:, 2], -down.points[:, 2], atol=1e-14)


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"normal": [0.0, 0.0, 0.0]}, "degenerate_span_or_tangent"),
        ({"tangent_before": [-1.0, 0.0, 0.0]}, "unsupported_reentrant_tangent"),
        ({"tangent_before": [1.0, 0.0, 1.0]}, "unsupported_nonplanar_tangent"),
        ({"source_points": [[0.0, 0.0, 0.0], [0.02, 0.0, 0.001], [0.04, 0.0, 0.0]]}, "unsupported_nonplanar_span"),
        ({"maximum_height_m": -1}, "invalid_limits"),
        ({"normal": [0.0, 0.0, np.nan]}, "invalid_geometry"),
    ],
)
def test_unsupported_inputs_have_explicit_reasons(change, reason):
    inputs = _straight()
    inputs.update(change)
    result = generate_connector_bridges(**inputs)
    assert not result.candidates
    assert result.reason == reason


def test_multiple_distinct_crossings_are_not_hidden_under_one_hump():
    inputs = _straight(0.25)
    inputs["underpass_segments"] = np.concatenate((inputs["underpass_segments"], _straight(0.75)["underpass_segments"]))
    result = generate_connector_bridges(**inputs)
    assert not result.candidates
    assert "unsupported_multiple_crossings" in result.rejected_reasons


def test_insufficient_radius_and_deviation_limits_reject_instead_of_relaxing_constraints():
    inputs = _saved_gz()
    inputs["minimum_radius_m"] = 0.01
    result = generate_connector_bridges(**inputs)
    assert not result.candidates
    assert "insufficient_bend_radius" in result.rejected_reasons
    inputs["minimum_radius_m"] = 0.00271544
    inputs["maximum_deviation_m"] = 0.001
    result = generate_connector_bridges(**inputs)
    assert not result.candidates
    assert "deviation_limit" in result.rejected_reasons


def test_crossing_at_attachment_is_reported_without_lifting_the_attachment():
    result = generate_connector_bridges(**_straight(0.0))
    assert not result.candidates
    assert "crossing_too_close_to_attachment" in result.rejected_reasons


def _saved_spatial_gx():
    # Frozen GX connector 2 spans two physical octagonal facets. The endpoint
    # tangents and coordinates must remain in 3D rather than being flattened.
    neighbors = np.array(
        [
            [-0.0027742521789489724, -0.023481021544517686, 0.08385086671134907],
            [-0.0029256645825873244, -0.026782114658867522, 0.08378815016741095],
            [0.0033876890679826197, -0.0364197285690289, 0.08359677087189074],
            [0.0033944387595374425, -0.03654780325364583, 0.08359397503359264],
            [0.0034241065635583104, -0.03713979646867373, 0.08358168611446047],
        ]
    )
    inputs = _straight()
    inputs.update(
        source_points=neighbors[1:4],
        tangent_before=neighbors[1] - neighbors[0],
        tangent_after=neighbors[-1] - neighbors[-2],
        normal=[2.460349578824817e-7, 2.0745190458852083e-9, 0.9999999999999697],
        underpass_segments=np.array(
            [
                [
                    [0.002957255177470401, -0.026430572947924294, 0.08377506323505506],
                    [-0.0033445821116361656, -0.036771839141410245, 0.08361462926601368],
                ]
            ]
        ),
        spatial_mode=True,
    )
    return inputs


def test_spatial_saved_connector_keeps_three_dimensional_tangents_and_bounds_arch():
    inputs = _saved_spatial_gx()
    result = generate_connector_bridges(**inputs)
    assert 1 <= len(result.candidates) <= 64
    candidate = result.candidates[0]
    assert candidate.profile_family == "quartic"
    assert candidate.height_m < inputs["maximum_height_m"]  # Radius requires a lower arch here.
    assert candidate.local_clearance_m < 0.00266  # Proposal does not certify finite-profile clearance.
    np.testing.assert_array_equal(candidate.points[[0, -1]], inputs["source_points"][[0, -1]])
    assert np.max(candidate.lift_distances_m) <= inputs["maximum_height_m"]
    assert np.min(candidate.lift_distances_m) >= 0
    grid = np.linspace(0, 1, 20001)
    positions = _sample_independently(candidate, grid)
    first = np.gradient(positions, grid, axis=0, edge_order=2)
    second = np.gradient(first, grid, axis=0, edge_order=2)
    curvature = np.linalg.norm(np.cross(first, second), axis=1) / np.linalg.norm(first, axis=1) ** 3
    assert 1 / curvature.max() >= inputs["minimum_radius_m"]
    for actual, expected in [(first[0], inputs["tangent_before"]), (first[-1], inputs["tangent_after"])]:
        np.testing.assert_allclose(actual / np.linalg.norm(actual), expected / np.linalg.norm(expected), atol=1e-6)
    inputs["spatial_mode"] = False
    assert generate_connector_bridges(**inputs).reason == "unsupported_nonplanar_span"


def test_spatial_curvature_derivatives_and_sampling_are_independent_of_source_subdivision():
    inputs = _saved_spatial_gx()
    original = generate_connector_bridges(**inputs)
    source = inputs["source_points"]
    inputs["source_points"] = np.vstack(
        (np.linspace(source[0], source[1], 7)[:-1], np.linspace(source[1], source[2], 3))
    )
    subdivided = generate_connector_bridges(**inputs)
    assert len(original.candidates) == len(subdivided.candidates)
    for before, after in zip(original.candidates, subdivided.candidates):
        np.testing.assert_allclose(before.points, after.points, atol=1e-14)
        np.testing.assert_allclose(before.source_fractions, after.source_fractions, atol=1e-14)


def test_spatial_grid_includes_lower_arch_for_endpoint_at_a_support_crease():
    # GY connector 0 has an outgoing tangent on the adjacent octagonal facet.
    # The full-route/support replay accepts its 2 mm arch; 2.25 mm already
    # exceeds the actual 3 mm support-distance limit because the base is 3D.
    neighbors = np.array(
        [
            [-0.06919773002598414, 0.0031179462885864997, -0.05504953289285599],
            [-0.06904880317790739, 0.003111752059642158, -0.05540907395743513],
            [-0.06518475804803468, -0.0028511917123463407, -0.06473770040020325],
            [-0.06505381774902344, -0.0028532596794864705, -0.06505381774902344],
            [-0.06454569194149778, -0.002834580290283595, -0.06526429019522746],
        ]
    )
    inputs = _straight()
    inputs.update(
        source_points=neighbors[1:4],
        tangent_before=neighbors[1] - neighbors[0],
        tangent_after=neighbors[-1] - neighbors[-2],
        normal=[-0.7071066368930343, -7.701090928977409e-8, -0.7071069254800271],
        underpass_segments=np.array(
            [
                [
                    [-0.06901080660517503, -0.0030540637760453175, -0.05550080425846684],
                    [-0.06522284938068365, 0.00289738032207803, -0.06464573955370127],
                ]
            ]
        ),
        spatial_mode=True,
    )
    result = generate_connector_bridges(**inputs)
    assert result.candidates
    lower = [candidate for candidate in result.candidates if abs(candidate.height_m - 0.002) < 1e-12]
    assert lower
    span = np.linalg.norm(neighbors[3] - neighbors[1])
    selected = [
        candidate
        for candidate in lower
        if abs(np.linalg.norm(candidate.control_points_m[1] - neighbors[1]) / span - 0.55) < 1e-12
        and abs(np.linalg.norm(neighbors[3] - candidate.control_points_m[2]) / span - 0.35) < 1e-12
    ]
    assert len(selected) == 1
    assert selected[0].minimum_radius_m > 0.003
    assert selected[0].local_clearance_m < 0.00266


@pytest.mark.parametrize("spatial_mode", [False, True])
def test_peak_close_to_regular_sampling_knot_never_creates_a_duplicate_segment(spatial_mode):
    result = generate_connector_bridges(**_straight(), spatial_mode=spatial_mode)
    assert result.candidates
    for candidate in result.candidates:
        assert np.min(np.diff(candidate.parameters)) > 1e-12
        assert np.min(np.linalg.norm(np.diff(candidate.points, axis=0), axis=1)) > 1e-10
