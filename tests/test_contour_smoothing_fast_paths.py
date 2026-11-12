"""The pruned contour distance and the batched chart mapping must match the plain code bit for bit."""

import numpy as np
import pytest

from coilgen.contour_smoothing import _harmonic_surface_chart, _locate_uv_point, _map_uv_points
from coilgen.contours import _contour_pair_min_distance, _contour_pair_min_distance_grid
from coilgen.geometry import build_planar_mesh


def _polyline_pairs():
    rng = np.random.default_rng(3)
    turns = np.linspace(0.0, 6.0 * np.pi, 900)
    helix = np.c_[np.cos(turns), np.sin(turns), 0.01 * turns]
    yield "adjacent_helices", helix, np.c_[1.003 * np.cos(turns), 1.003 * np.sin(turns), 0.01 * turns]
    yield (
        "overlapping_parallel_lines",
        np.c_[np.linspace(0.0, 1.0, 500), np.zeros(500), np.zeros(500)],
        np.c_[np.linspace(0.3, 1.4, 400), np.full(400, 1.0e-3), np.zeros(400)],
    )
    curve = np.c_[np.linspace(0.0, 1.0, 300), np.linspace(0.0, 1.0, 300) ** 2, np.zeros(300)]
    yield (
        "repeated_points",
        np.repeat(curve, 2, axis=0),
        np.c_[np.full(300, 0.5), np.linspace(-1, 1, 300), np.zeros(300)],
    )
    yield "crossing_curves", curve, np.c_[np.linspace(0.0, 1.0, 400), 1.0 - np.linspace(0.0, 1.0, 400), np.zeros(400)]
    walks = [np.cumsum(rng.normal(scale=0.01, size=(int(count), 3)), axis=0) for count in (450, 520)]
    yield "random_walks", walks[0], walks[1]


@pytest.mark.parametrize(
    "name,points_a,points_b", list(_polyline_pairs()), ids=lambda value: value if isinstance(value, str) else ""
)
def test_pruned_pair_distance_matches_full_grid(name, points_a, points_b) -> None:
    assert (len(points_a) - 1) * (len(points_b) - 1) > 20_000, "must exercise the pruned path"
    assert _contour_pair_min_distance(points_a, points_b) == _contour_pair_min_distance_grid(points_a, points_b)
    assert _contour_pair_min_distance(points_b, points_a) == _contour_pair_min_distance_grid(points_b, points_a)


def test_batched_chart_mapping_matches_point_by_point_location() -> None:
    mesh = build_planar_mesh(0.2, 0.3, 14, 18, 1, 0, 0, 0, 0, 0, 0)
    chart = _harmonic_surface_chart(mesh, np.arange(len(mesh.faces)))
    low, high = np.min(chart.uv, axis=0), np.max(chart.uv, axis=0)
    centre, radius = 0.5 * (low + high), 0.35 * (high - low)
    angles = np.linspace(0.0, 2.0 * np.pi, 1500)
    points_uv = centre + radius * np.c_[np.cos(angles), np.sin(3.0 * angles) * 0.6 + 0.4 * np.sin(angles)]
    # Put some samples exactly on chart vertices, where neighbouring triangles tie.
    points_uv[::97] = chart.uv[np.random.default_rng(2).integers(0, len(chart.uv), len(points_uv[::97]))]

    mapped = _map_uv_points(chart, points_uv)

    expected = []
    hint = None
    for point in points_uv:
        hint, barycentric = _locate_uv_point(chart, point, hint)
        expected.append(barycentric @ mesh.vertices[mesh.faces[int(chart.global_faces[hint])]])
    expected = np.asarray(expected)
    expected[-1] = expected[0]
    np.testing.assert_array_equal(mapped, expected)
