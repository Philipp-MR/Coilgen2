"""Unsupported cyclic return repairs must remain explicit without moving the seam."""

import numpy as np

from coilgen.config import CoilGenConfig
from coilgen.geometry import normalize_mesh
from coilgen.layout import _NormalShiftRoute, _return_crossing_ramp_spans, shift_return_paths
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry, WireSegmentOwner


def _fixture():
    points = 0.001 * np.array(
        [
            [-5, 0, 0],
            [0, 0, 0],
            [10, 0, 0],
            [20, 0, 0],
            [30, 0, 0],
            [40, 0, 0],
            [50, 30, 0],
            [8, 30, 0],
            [8, -30, 0],
            [-5, -30, 0],
            [-5, 0, 0],
        ],
        dtype=float,
    )
    owner = WireSegmentOwner("contour_group", group_index=0)
    path = WirePath(points, segment_owners=[owner] * 10, segment_geometry=[WirePathGeometry.SURFACE] * 10)
    route = _NormalShiftRoute(("within_group_return", 0, 0), 2, 0, (1, 2, 3, 4))
    routes = [[route] if index in route.segment_indices else [] for index in range(10)]
    factors = np.zeros(11)
    factors[2:5] = 1.0
    trial = points.copy()
    trial[:, 2] = factors * 0.003
    config = CoilGenConfig(
        normal_shift_length=0.003,
        conductor_cross_section_width=0.00196,
        conductor_cross_section_height=0.0018,
        sweep_min_bend_radius=0.0025,
    )
    return path, routes, [(1, 7, 1.8, 7.5, 0.0)], trial, factors, config


def test_ordinary_return_approach_is_selected_without_seam_diagnostic():
    unresolved = []
    assert _return_crossing_ramp_spans(*_fixture(), unresolved_spans=unresolved) == [(1, 3, (7,))]
    assert unresolved == []


def test_closed_seam_return_approach_is_explicit_and_preserves_all_input():
    path, routes, _anchors, trial, factors, config = _fixture()
    indices = np.r_[np.arange(2, 10), np.arange(2)]
    point_indices = np.r_[indices, indices[0]]
    rotated = WirePath(
        path.v[point_indices],
        segment_owners=[path.segment_owners[i] for i in indices],
        segment_geometry=[path.segment_geometry[i] for i in indices],
    )
    trial, factors = trial[point_indices], factors[point_indices]
    source_copy, trial_copy, factors_copy = rotated.v.copy(), trial.copy(), factors.copy()
    owners, geometry = list(rotated.segment_owners), list(rotated.segment_geometry)
    unresolved = []
    result = _return_crossing_ramp_spans(
        rotated,
        [routes[i] for i in indices],
        [(9, 5, 9.8, 5.5, 0.0)],
        trial,
        factors,
        config,
        unresolved_spans=unresolved,
    )
    assert result == []
    assert len(unresolved) == 1
    event = unresolved[0]
    assert event["reason"] == "unsupported_return_seam"
    assert event["kind"] == "return_approach"
    assert event["accepted"] is False
    assert event["source_segment_ranges"] == [[9, 10], [0, 1]]
    assert event["underpass_source_segments"] == [5]
    np.testing.assert_array_equal(rotated.v, source_copy)
    np.testing.assert_array_equal(trial, trial_copy)
    np.testing.assert_array_equal(factors, factors_copy)
    assert rotated.segment_owners == owners
    assert rotated.segment_geometry == geometry


def test_pipeline_marks_required_seam_return_repair_unresolved():
    path, _routes, _anchors, _trial, _factors, config = _fixture()
    contour = WireSegmentOwner("contour_loop", group_index=0, loop_index=0)
    returning = WireSegmentOwner("within_group_return", group_index=1)
    original_owners = [returning if 1 <= index <= 4 else contour for index in range(10)]
    indices = np.r_[np.arange(2, 10), np.arange(2)]
    source = path.v[np.r_[indices, indices[0]]].copy()
    owners = [original_owners[index] for index in indices]
    mesh = normalize_mesh(
        Mesh(
            np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]]), np.array([[0, 1, 2], [0, 2, 3]])
        )
    )
    part = CoilPart(mesh=mesh, wire_path=WirePath(source.copy(), segment_owners=owners))
    shift_return_paths([part], config)
    assert part.normal_shift_diagnostics["normal_shift_status"] == "unresolved"
    assert part.normal_shift_diagnostics["normal_shift_quality_failure"]
    assert part.normal_shift_diagnostics["normal_shift_rejected_bridge_count"] >= 1
    assert any(event["reason"] == "unsupported_return_seam" for event in part.normal_shift_provenance["bridge_events"])
    np.testing.assert_array_equal(part.normal_shift_provenance["points_before_m"], source)
    assert len(part.wire_path.v) == len(source)  # Only the existing vertex lift was applied.
    protected = np.zeros(len(source), dtype=bool)
    for index, owner in enumerate(owners):
        if owner.kind == "contour_loop":
            protected[index : index + 2] = True
    np.testing.assert_array_equal(part.wire_path.v[protected], source[protected])
