"""Sub-nanometre route slivers must not be reported as wire crossings."""

import numpy as np

from coilgen import CoilGenConfig, find_min_wire_segment_distance_3d, find_wire_segment_contacts_3d


def _helix(turns=3, samples=120, pitch=0.006, radius=0.05):
    angle = np.linspace(0.0, 2.0 * np.pi * turns, samples)
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), pitch * angle / (2.0 * np.pi)))


def _with_sliver(points, index=57, offset=1e-10):
    sliver = points[index] + np.array([offset, 0.0, 0.0])
    return np.vstack((points[: index + 1], sliver, points[index + 1 :]))


def test_sliver_is_not_a_crossing():
    clean = _helix()
    sliver = _with_sliver(clean)
    config = CoilGenConfig(wire_distance_method="all_pairs")
    assert not find_wire_segment_contacts_3d(sliver, 1e-6, config=config)
    a, b, distance = find_min_wire_segment_distance_3d(sliver, config=config)
    _a, _b, clean_distance = find_min_wire_segment_distance_3d(clean, config=config)
    assert distance == clean_distance > 0.005
    # Reported indices are segment starts of the original array.
    assert 0 <= a < b < len(sliver) - 1


def test_real_crossing_is_still_detected():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 1.0, 0.0], [0.5, -1.0, 0.0]])
    contacts = find_wire_segment_contacts_3d(_with_sliver(points, index=1), 1e-6)
    assert contacts and min(distance for _a, _b, distance in contacts) < 1e-9
