import numpy as np
import pytest

from coilgen.wire_provenance import ordered_source_stations


@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_identical_directed_visits_survive_resampling_and_closed_seam_rotation(reverse):
    v = np.array(
        [[-2.0, 0], [-1, 0], [1, 0], [2, 1], [0, 2], [-2, 1], [-1, 0], [1, 0], [2, -1], [0, -2], [-2, -1], [-2, 0]]
    )
    v = np.column_stack((v, np.zeros(len(v))))
    # Intervals 1 and 6 cover the exact same spatial segment in the same
    # direction. Neither a nearest-point nor tangent test distinguishes them.
    source = np.insert(v, 4, v[4], axis=0)
    starts = v[:-1]
    diff = np.diff(v, axis=0)
    subdivisions = starts[:, None, :] + diff[:, None, :] * np.array([0.0, 0.33, 0.67])[None, :, None]
    target = np.roll(subdivisions.reshape(-1, 3), -19, axis=0)
    target = np.vstack((target, target[0]))
    expected = np.repeat(np.arange(len(v) - 1), 3)
    expected[expected >= 4] += 1
    expected = np.roll(expected, -19)
    if reverse:
        target = target[::-1]
        expected = expected[::-1]
    i, t = ordered_source_stations(source, target)
    np.testing.assert_array_equal(i, expected)
    np.testing.assert_allclose(
        source[i] + t[:, None] * (source[i + 1] - source[i]), (target[1:] + target[:-1]) / 2, atol=1e-12
    )
