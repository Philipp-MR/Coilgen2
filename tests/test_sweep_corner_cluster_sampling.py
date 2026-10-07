"""Configured compound sampling and cumulative two-pass work limits."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import coilgen.sweep_corner_clusters as NEW
from coilgen.config import CoilGenConfig
from coilgen.manufacturing import _AnalyticCylinderSurface
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry, WireSegmentOwner


@pytest.mark.parametrize("radius", [0.002, 0.082])
@pytest.mark.parametrize("angle", [0.01, 0.8, np.pi])
def test_configured_interior_bounds_and_fine_endpoint_tangents(radius, angle):
    origin, tangent, across = np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    maximum_length, maximum_chord, rotation = 0.001, 3e-5, np.deg2rad(5.0)
    points = NEW._endpoint_sampled_cluster_arc(
        origin, tangent, (angle, across, None), radius, (maximum_length, maximum_chord, rotation)
    )
    expected_end = origin + radius * (np.sin(angle) * tangent + (1 - np.cos(angle)) * across)
    np.testing.assert_allclose(points[0], origin, atol=1e-16)
    np.testing.assert_allclose(points[-1], expected_end, atol=1e-16)
    parameters = np.unwrap(np.arctan2(points[:, 0] / radius, 1 - points[:, 1] / radius))
    steps = np.diff(parameters)
    assert np.all(steps > 0)
    assert steps.max() <= rotation + 1e-12
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() <= maximum_length + 1e-12
    assert (radius * (1 - np.cos(steps / 2))).max() <= maximum_chord + 1e-12
    assert max(steps[0], steps[-1]) <= np.deg2rad(1.0) + 1e-12
    assert np.all(np.isfinite(points)) and len(points) <= 4096


def test_arc_sampling_refuses_work_exhaustion_instead_of_returning_coarse_points():
    points = NEW._endpoint_sampled_cluster_arc(
        np.zeros(3), np.array([1.0, 0, 0]), (np.pi, np.array([0, 1.0, 0]), None), 0.082, (0.001, 1e-15, 0.1)
    )
    assert points is None


@pytest.mark.parametrize("custom_limit", [16, 17])
def test_global_sixteen_cluster_count_is_hard_and_preserves_completed_work(custom_limit):
    work = NEW._ClusterWork(maximum_clusters=custom_limit)
    for _ in range(16):
        work.consume_cluster()
    with pytest.raises(NEW._ClusterBudgetExceeded, match="count_budget"):
        work.consume_cluster()
    assert work.clusters == 16


def tip_context():
    root = Path(__file__).resolve().parents[1]
    chart = json.loads((root / "tests/fixtures/gradient_tip_clusters.json").read_text())[0]
    surface = _AnalyticCylinderSurface(np.zeros(3), np.array([0, 0, 1]), 0.066, -0.1, 0.1)
    points = surface.exp_map(np.array([0.066, 0, 0]), np.asarray(chart))
    geometry = [WirePathGeometry.SURFACE] * (len(points) - 1)
    owners = [WireSegmentOwner("contour_loop", 1, 2)] * len(geometry)
    wire = WirePath(points.copy(), segment_geometry=geometry, segment_owners=owners)
    part = CoilPart(Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int)), wire_path=wire)
    config = CoilGenConfig(
        coil_mesh_file="none",
        sweep_bend_mode="fillet",
        sweep_corner_cluster_max_source_deviation_m=0.001,
        conductor_cross_section_width=0.0014,
        conductor_cross_section_height=0.0014,
        sweep_max_chord_error=5e-5,
        sweep_max_frame_rotation_deg=1.5,
    )
    return part, points, geometry, surface, config


def test_real_source_cluster_two_passes_share_count_and_numerical_budgets():
    part, points, geometry, surface, config = tip_context()
    before = deepcopy(part.wire_path)
    work = NEW._ClusterWork(maximum_clusters=1)
    realized, labels, first = NEW.resolve_source_corner_clusters(
        part, points, geometry, points, geometry, surface, [], {9, 10}, 0.0014285714285714288, config, work=work
    )
    assert first["sweep_corner_cluster_resolved_count"] == 1
    assert first["sweep_corner_cluster_certificates"][0]["accepted"]
    assert len(realized) == len(labels) + 1
    completed = work.trials, work.pairs, work.curvature_points
    assert min(completed) > 0
    rejected, unchanged, second = NEW.resolve_source_corner_clusters(
        part, points, geometry, points, geometry, surface, [], {9, 10}, 0.0014285714285714288, config, work=work
    )
    assert second["sweep_corner_cluster_budget_exhausted"]
    assert second["sweep_corner_cluster_budget_reason"] == "corner_cluster_count_budget"
    assert second["sweep_corner_cluster_global_certified_count"] == 1
    assert second["sweep_corner_cluster_resolved_count"] == 0
    assert (work.trials, work.pairs, work.curvature_points) == completed
    np.testing.assert_array_equal(rejected, points)
    assert unchanged == geometry
    np.testing.assert_array_equal(part.wire_path.v, before.v)
    assert part.wire_path.segment_owners == before.segment_owners
    assert part.wire_path.segment_geometry == before.segment_geometry


def test_custom_shared_work_cannot_enable_a_seventeenth_cluster_on_a_later_pass():
    part, points, geometry, surface, config = tip_context()
    work = NEW._ClusterWork(maximum_clusters=17, clusters=16)
    returned, labels, report = NEW.resolve_source_corner_clusters(
        part, points, geometry, points, geometry, surface, [], {9, 10}, 0.0014285714285714288, config, work=work
    )
    assert report["sweep_corner_cluster_budget_exhausted"]
    assert report["sweep_corner_cluster_global_count_limit"] == 16
    assert report["sweep_corner_cluster_pass_remaining_limit"] == 0
    assert work.clusters == 16 and work.trials == work.pairs == work.curvature_points == 0
    np.testing.assert_array_equal(returned, points)
    assert labels == geometry


@pytest.mark.parametrize("invalid", [True, -1, 17, 1.5])
def test_pass_cluster_limit_cannot_expand_global_bound_or_accept_noninteger_value(invalid):
    part, points, labels, surface, config = tip_context()
    with pytest.raises(ValueError, match="integer"):
        NEW.resolve_source_corner_clusters(
            part,
            points,
            labels,
            points,
            labels,
            surface,
            [],
            {9, 10},
            0.0014285714285714288,
            config,
            maximum_clusters=invalid,
        )
