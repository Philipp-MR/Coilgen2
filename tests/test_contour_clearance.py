"""Indexed connector checks preserve the original scalar geometric decisions."""

import math

import numpy as np
import pytest

from coilgen import layout
from coilgen.config import CoilGenConfig
from coilgen.contour_clearance import ContourClearanceIndex
from coilgen.geometry_kernels import segment_distance
from coilgen.models import CoilPart, ContourLine, InterGroupCut, Mesh


def reference(contours, points, excluded=()):
    best, loop = math.inf, -1
    for label, contour in enumerate(contours):
        if label in excluded:
            continue
        for start, end in zip(points[:-1], points[1:]):
            if np.linalg.norm(end - start) <= 1e-12:
                continue
            for first, last in zip(contour[:-1], contour[1:]):
                if np.linalg.norm(last - first) <= 1e-12:
                    continue
                distance = segment_distance(start, end, first, last)
                if distance < best:
                    best, loop = distance, label
    return best, loop


@pytest.mark.parametrize("seed", range(8))
def test_random_curves_match_scalar_reference_exactly(seed):
    rng = np.random.default_rng(seed)
    contours = [np.cumsum(rng.normal(size=(8, 3)), axis=0) for _ in range(5)]
    points = np.cumsum(rng.normal(size=(7, 3)), axis=0)
    index = ContourClearanceIndex(contours)
    for excluded in ((), (0, 2), range(5)):
        assert index.distance_to_path(points, excluded) == reference(contours, points, excluded)


@pytest.mark.parametrize("scale,offset", [(1e-6, 0), (1.0, 0), (1e6, 0), (1.0, 1e9)])
def test_long_segments_near_parallel_and_interior_crossings(scale, offset):
    contours = [
        np.array([[-100.0, 0, 0], [100, 0, 0]]),
        np.array([[-100.0, -2, 1], [100, 2, 1]]),
        np.array([[-100.0, -2, -1], [100, 2, -1]]),
    ]
    contours = [points * scale + offset for points in contours]
    index = ContourClearanceIndex(contours)
    for points in (
        np.array([[0.0, -100, 0], [0, 100, 0]]),
        np.array([[-100.0, 1e-11, 0], [100, 2e-11, 0]]),
        np.array([[100.0, 0, 0], [101, 0, 0]]),
    ):
        transformed = points * scale + offset
        assert index.distance_to_path(transformed) == reference(contours, transformed)


def test_tied_contours_retain_original_loop_order_after_tree_reordering():
    contours = [np.array([[0.0, y, 0], [1, y, 0]]) for y in (1, -1, *range(2, 50))]
    query = np.array([[0.2, 0, 0], [0.8, 0, 0]])
    index = ContourClearanceIndex(contours)
    assert index.distance_to_path(query) == (1.0, 0)
    assert index.distance_to_path(query, (0,)) == (1.0, 1)


def test_degenerate_empty_and_nonfinite_segments_follow_legacy_admission():
    contours = [
        np.zeros((1, 3)),
        np.zeros((2, 3)),
        np.array([[0.0, 0, 0], [1e-13, 0, 0]]),
        np.array([[0.0, 1, 0], [1, 1, 0]]),
        np.array([[np.nan, 0, 0], [np.nan, 1, 0]]),
    ]
    query = np.array([[0.0, 0, 0], [1, 0, 0]])
    assert ContourClearanceIndex(contours).distance_to_path(query) == (1.0, 3)
    assert ContourClearanceIndex(contours[:3]).distance_to_path(query) == (math.inf, -1)
    assert ContourClearanceIndex([]).distance_to_path(query) == (math.inf, -1)
    assert ContourClearanceIndex(contours).distance_to_path(query[:1]) == (math.inf, -1)
    assert layout._polyline_polyline_distance_3d(query[:1], contours[3]) == 1.0


def test_operation_snapshot_and_fresh_index_after_contour_change():
    contour = np.array([[0.0, 1, 0], [1, 1, 0]])
    query = np.array([[0.0, 0, 0], [1, 0, 0]])
    index = ContourClearanceIndex([contour])
    contour[:, 1] = 2
    assert index.distance_to_path(query) == (1.0, 0)
    assert ContourClearanceIndex([contour]).distance_to_path(query) == (2.0, 0)


def test_separated_contours_prune_pairs_without_a_resolution_approximation():
    x = np.linspace(0, 1, 201)
    contours = [np.column_stack((x, np.full_like(x, y), np.zeros_like(x))) for y in range(32)]
    query = np.column_stack((x, np.full_like(x, -0.003), np.zeros_like(x)))
    index = ContourClearanceIndex(contours)
    assert index.distance_to_path(query) == (0.003, 0)
    exhaustive_pairs = 200 * 200 * 32
    assert index.segment_pair_evaluations < exhaustive_pairs // 100


def test_minimum_and_violation_count_share_queries_and_preserve_strict_threshold(monkeypatch):
    mesh = Mesh(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
    contour = ContourLine(np.array([[0.0, 0, 0], [1, 0, 0]]), potential=0.0)
    first = np.array([[0.0, 1, 0], [1, 1, 0]])
    closing = np.array([[0.0, 0.125, 0], [1, 0.125, 0]])
    part = CoilPart(mesh=mesh, contour_lines=[contour])
    part.opening_cuts_among_groups = [
        InterGroupCut(first, closing, connection_path_v=first, closing_connection_path_v=closing)
    ]
    built = []

    def factory(contours):
        index = ContourClearanceIndex(contours)
        built.append(index)
        return index

    monkeypatch.setattr(layout, "ContourClearanceIndex", factory)
    result = layout._route_quality_diagnostics(
        part, -1, -1, 0.0, CoilGenConfig(min_inter_group_connector_clearance=0.125)
    )
    assert result["min_inter_group_connector_to_contour_distance"] == 0.125
    assert result["min_inter_group_connector_to_contour_cut_index"] == 0
    assert result["min_inter_group_connector_to_contour_loop_index"] == 0
    assert result["inter_group_connector_clearance_violation_count"] == 0
    assert len(built) == 1 and built[0].query_count == 2
    assert layout._connector_clearance_summary(part, np.nextafter(0.125, math.inf)) == (0.125, 0, 0, 1)


def test_zero_candidate_clearance_threshold_skips_inactive_work(monkeypatch):
    from _group_routing_cases import make_group_routing_case

    case = make_group_routing_case("facing_corridor")
    case.config.min_inter_group_connector_clearance = 0.0
    monkeypatch.setattr(
        layout, "ContourClearanceIndex", lambda *_a: pytest.fail("Inactive candidate clearance indexed")
    )
    monkeypatch.setattr(
        layout,
        "_connector_path_clearance_to_unrelated_contours",
        lambda *_a, **_k: pytest.fail("Inactive check queried"),
    )
    selected = layout._select_group_connection(
        case.part, case.part.connected_groups, [0, 1], np.empty((0, 3)), case.config
    )
    assert selected is not None
