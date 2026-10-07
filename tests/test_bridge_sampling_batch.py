"""Frozen scalar sampling checks protect exact dyadic knots and admission limits."""

import numpy as np
import pytest

from coilgen.connector_bridges import _adaptive_parameters, _evaluate


def _scalar_reference(controls, normal, height, peak, segment_length, chord_error, angle_deg, family):
    regular = np.linspace(0, 1, 17)
    knots = np.sort(np.r_[regular[np.abs(regular - peak) > 1e-12], peak])
    intervals = [(a, b, 0) for a, b in zip(knots[:-1], knots[1:])][::-1]
    result, cosine = [0.0], np.cos(np.radians(angle_deg))
    while intervals:
        low, high, depth = intervals.pop()
        parameters = np.linspace(low, high, 5)
        points, first, _, _ = _evaluate(controls, normal, height, peak, parameters, family)
        tangents = first / np.linalg.norm(first, axis=1)[:, None]
        frames = normal - (tangents @ normal)[:, None] * tangents
        lengths = np.linalg.norm(frames, axis=1)
        if np.min(lengths) <= 1e-8:
            return None
        frames /= lengths[:, None]
        linear = points[0] + np.linspace(0, 1, 5)[:, None] * (points[-1] - points[0])
        split = (
            np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > segment_length
            or np.linalg.norm(points - linear, axis=1).max() > chord_error
            or np.min(tangents @ tangents.T) < cosine
            or np.min(frames @ frames.T) < cosine
        )
        if split:
            if depth >= 20 or len(result) + len(intervals) >= 4096:
                return None
            middle = (low + high) / 2
            intervals.extend(((middle, high, depth + 1), (low, middle, depth + 1)))
        else:
            result.append(high)
    return np.asarray(result)


@pytest.mark.parametrize("family", ["quintic", "quartic"])
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e3])
@pytest.mark.parametrize("peak", [0.35, 0.5, 0.651234567])
def test_batched_sampling_preserves_scalar_knots_and_curve_points(family, scale, peak):
    controls = scale * np.array([[0.0, 0, 0], [-0.003, 0.006, 0], [0.008, -0.007, 0], [0.016, 0, 0]])
    normal = np.array([0.0, 0, 1])
    args = (controls, normal, 0.002 * scale, peak, 0.001 * scale, 1e-5 * scale, 2.0, family)
    expected, actual = _scalar_reference(*args), _adaptive_parameters(*args)
    assert expected is not None and actual is not None
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        _evaluate(controls, normal, args[2], peak, actual, family)[0],
        _evaluate(controls, normal, args[2], peak, expected, family)[0],
    )


@pytest.mark.parametrize("defect", ["point_budget", "depth_budget", "singular_guide"])
def test_batched_sampling_keeps_fail_closed_scalar_admission(defect):
    controls = np.array([[0.0, 0, 0], [0.005, 0, 0], [0.01, 0, 0], [0.015, 0, 0]])
    normal = np.array([1.0, 0, 0]) if defect == "singular_guide" else np.array([0.0, 0, 1])
    length = 1e-9 if defect == "depth_budget" else 1e-6 if defect == "point_budget" else 0.001
    args = (controls, normal, 0.0, 0.5, length, 1e-5, 2.0, "quintic")
    assert _scalar_reference(*args) is None
    assert _adaptive_parameters(*args) is None
