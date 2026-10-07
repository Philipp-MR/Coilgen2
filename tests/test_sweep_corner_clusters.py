"""Hard source, provenance, and budget checks for optional cylinder clusters."""

import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

from coilgen.sweep_corner_clusters import _ClusterWork, _ClusterBudgetExceeded, _certify_cluster  # noqa: E402
from coilgen.manufacturing import _AnalyticCylinderSurface  # noqa: E402
from coilgen.models import CoilPart, Mesh, WirePath, WirePathGeometry, WireSegmentOwner  # noqa: E402
from coilgen.config import CoilGenConfig
import coilgen.sweep_corner_clusters as clusters
from coilgen.manufacturing import _resolve_sweep_centerline_bends


SURFACE = _AnalyticCylinderSurface(np.zeros(3), np.array([0, 0, 1]), 0.066, -0.1, 0.1)
ORIGIN = np.array([0.066, 0, 0])


def check(source, realized, *, bound=0.001, owners=None, geometry=None, maximum_points=2048, surface=SURFACE):
    return _certify_cluster(
        SURFACE.exp_map(ORIGIN, np.asarray(source)),
        SURFACE.exp_map(ORIGIN, np.asarray(realized)),
        surface,
        required_radius_m=0.001,
        maximum_source_deviation_m=bound,
        maximum_chord_error_m=5e-5,
        tangent_tolerance_deg=0.5,
        source_owner_keys=owners or [("contour_loop", 1, 2)] * (len(source) - 1),
        source_geometry=geometry or [WirePathGeometry.SURFACE] * (len(source) - 1),
        maximum_validation_points=maximum_points,
        work=_ClusterWork(),
    )


def regular_fillet():
    radius = 0.002
    source = radius * np.array([[-3, 0], [0, 0], [0, 3]])
    angle = np.linspace(-np.pi / 2, 0, 361)
    arc = np.array([-radius, radius]) + radius * np.column_stack((np.cos(angle), np.sin(angle)))
    realized = np.vstack((source[0], arc, source[-1]))
    return source, realized


@pytest.mark.parametrize("reflection", [-1, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_legitimate_fillet_has_explicit_two_way_corridor_and_reflection_invariant_certificate(reflection, reverse):
    source, realized = regular_fillet()
    source[:, 0] *= reflection
    realized[:, 0] *= reflection
    if reverse:
        source, realized = source[::-1], realized[::-1]
    result = check(source, realized)
    assert result["accepted"], result["reasons"]
    assert result["metrics"]["maximum_two_way_source_deviation_bound_m"] < 0.00084
    assert result["metrics"]["minimum_physical_radius_m"] > 0.00199
    assert not result["metrics"]["whole_body_certified"]


def test_same_valid_fillet_rejects_when_the_explicit_source_bound_is_too_small():
    result = check(*regular_fillet(), bound=0.0008)
    assert result["reasons"] == ["cluster_source_deviation_bound"]


def test_compact_shared_trim_ninety_degree_s_cannot_be_certified_by_flattening():
    source = 0.001 * np.array([[0, 0], [5, 0], [5, 5], [10, 5]])
    realized = 0.001 * np.array([[0, 0], [5, 2.5], [10, 5]])
    result = check(source, realized, bound=0.0005)
    assert not result["accepted"]
    assert "cluster_endpoint_tangent_changed" in result["reasons"]
    assert "cluster_source_deviation_bound" in result["reasons"]


def test_true_source_backtrack_cannot_be_silently_flattened():
    source = 0.001 * np.array([[-10, 0], [0, 0], [-5, 0], [-15, 0]])
    realized = 0.001 * np.array([[-10, 0], [-12, 0], [-15, 0]])
    result = check(source, realized, bound=0.0005)
    assert not result["accepted"]
    assert "cluster_endpoint_tangent_changed" in result["reasons"]
    assert "cluster_source_deviation_bound" in result["reasons"]


@pytest.mark.parametrize("defect", ["geometry", "owner", "unsupported", "sample_budget"])
def test_local_realization_cannot_cross_contracts_or_exhaust_its_certificate_budget(defect):
    kwargs = {}
    if defect == "geometry":
        kwargs["geometry"] = [WirePathGeometry.SURFACE, WirePathGeometry.FREE_3D]
    elif defect == "owner":
        kwargs["owners"] = [("contour_loop", 1, 2), ("contour_loop", 1, 3)]
    elif defect == "unsupported":
        kwargs["surface"] = None
    else:
        kwargs["maximum_points"] = 3
    result = check(*regular_fillet(), **kwargs)
    assert not result["accepted"]


@pytest.mark.parametrize("kind", ["pairs", "trials", "curvature_points"])
def test_explicit_work_limits_refuse_the_next_operation_and_preserve_completed_work(kind):
    work = _ClusterWork(maximum_trials=1, maximum_pairs=3, maximum_curvature_points=2)
    if kind == "pairs":
        work.consume_pairs(2)
        with pytest.raises(_ClusterBudgetExceeded, match="distance_pair_budget"):
            work.consume_pairs(2)
        assert work.pairs == 2
        assert work.attempted_pairs == 4
    elif kind == "trials":
        work.consume_trial()
        with pytest.raises(_ClusterBudgetExceeded, match="trial_budget"):
            work.consume_trial()
        assert work.trials == 1
    else:
        work.consume_curvature_points(2)
        with pytest.raises(_ClusterBudgetExceeded, match="curvature_point_budget"):
            work.consume_curvature_points(1)
        assert work.curvature_points == 2


# Three paired protected tip geometries from the standard Y gradient; intrinsic
# coordinates reproduce the small kinks without requiring a saved whole coil.
TIP_CLUSTERS = json.loads((Path(__file__).parent / "fixtures/gradient_tip_clusters.json").read_text())


def tip_context(family=0, bound=0.001):
    points = SURFACE.exp_map(ORIGIN, np.asarray(TIP_CLUSTERS[family]))
    labels = [WirePathGeometry.SURFACE] * (len(points) - 1)
    part = CoilPart(
        Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)),
        wire_path=WirePath(
            points.copy(),
            segment_geometry=labels,
            segment_owners=[WireSegmentOwner("contour_loop", 1, 2) for _ in labels],
        ),
    )
    config = CoilGenConfig(
        coil_mesh_file="none",
        sweep_bend_mode="fillet",
        sweep_corner_cluster_max_source_deviation_m=bound,
        conductor_cross_section_width=0.0014,
        conductor_cross_section_height=0.0014,
        sweep_max_chord_error=5e-5,
        sweep_max_frame_rotation_deg=1.5,
    )
    return part, points, labels, config


@pytest.mark.parametrize("family", [0, 1, 2])
def test_protected_gradient_tip_cluster_has_bounded_physical_realization_and_keeps_source(family):
    part, points, labels, config = tip_context(family)
    before = deepcopy(part.wire_path)
    realized, geometry, diagnostics = clusters.resolve_source_corner_clusters(
        part, points, labels, points, labels, SURFACE, [], {9, 10}, 0.0014285714285714288, config
    )
    assert diagnostics["sweep_corner_cluster_resolved_count"] == 1
    assert not diagnostics["sweep_corner_cluster_budget_exhausted"]
    certificate = diagnostics["sweep_corner_cluster_certificates"][0]
    assert certificate["accepted"]
    assert certificate["metrics"]["maximum_two_way_source_deviation_bound_m"] < 0.001
    assert certificate["metrics"]["minimum_physical_radius_m"] >= 0.001428
    assert len(geometry) == len(realized) - 1
    np.testing.assert_array_equal(part.wire_path.v, before.v)
    assert part.wire_path.segment_owners == before.segment_owners
    assert part.wire_path.segment_geometry == before.segment_geometry
    assert (
        diagnostics["sweep_corner_cluster_distance_pair_count"]
        <= diagnostics["sweep_corner_cluster_distance_pair_limit"]
    )


@pytest.mark.parametrize("defect", ["source_bound", "owner_boundary", "geometry_boundary", "work_exhausted"])
def test_cluster_search_cannot_waive_source_contract_or_incomplete_work(defect, monkeypatch):
    part, points, labels, config = tip_context(bound=0.0001 if defect == "source_bound" else 0.001)
    if defect == "owner_boundary":
        for index in range(10, len(labels)):
            part.wire_path.segment_owners[index] = WireSegmentOwner("contour_loop", 1, 3)
    elif defect == "geometry_boundary":
        labels[9] = WirePathGeometry.FREE_3D
        part.wire_path.segment_geometry = labels.copy()
    elif defect == "work_exhausted":
        factory = clusters._ClusterWork
        monkeypatch.setattr(clusters, "_ClusterWork", lambda: factory(maximum_trials=1))
    realized, geometry, diagnostics = clusters.resolve_source_corner_clusters(
        part, points, labels, points, labels, SURFACE, [], {9, 10}, 0.0014285714285714288, config
    )
    assert diagnostics["sweep_corner_cluster_resolved_count"] == 0
    np.testing.assert_array_equal(realized, points)
    assert geometry == labels
    if defect == "work_exhausted":
        assert diagnostics["sweep_corner_cluster_budget_exhausted"]
        assert diagnostics["sweep_corner_cluster_trial_count"] == 1
        assert diagnostics["sweep_corner_cluster_budget_reason"] == "corner_cluster_trial_budget"


@pytest.mark.parametrize("mode", ["miter", "fillet"])
def test_default_disabled_cluster_path_keeps_existing_mode_diagnostics_and_bypasses_search(mode, monkeypatch):
    part, points, labels, config = tip_context()
    config.sweep_bend_mode = mode
    config.sweep_corner_cluster_max_source_deviation_m = 0
    monkeypatch.setattr(clusters, "resolve_source_corner_clusters", lambda *args: pytest.fail("disabled search called"))
    _result, _labels, diagnostics = _resolve_sweep_centerline_bends(
        part,
        points,
        labels,
        np.array([[-0.0001, -0.0001], [0.0001, -0.0001], [0.0001, 0.0001], [-0.0001, 0.0001]]),
        config,
        closed=False,
    )
    assert not any("corner_cluster" in key for key in diagnostics)


def test_enabled_cluster_search_rejects_compact_ninety_degree_shared_trim_on_cylinder():
    chart = 0.001 * np.array([[-5, 0], [0, 0], [5, 0], [5, 5], [10, 5], [15, 5]])
    points = SURFACE.exp_map(ORIGIN, chart)
    labels = [WirePathGeometry.SURFACE] * (len(points) - 1)
    part = CoilPart(
        Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)),
        wire_path=WirePath(
            points.copy(),
            segment_geometry=labels,
            segment_owners=[WireSegmentOwner("contour_loop", 0, 0) for _ in labels],
        ),
    )
    config = CoilGenConfig(
        coil_mesh_file="none", sweep_bend_mode="fillet", sweep_corner_cluster_max_source_deviation_m=0.001
    )
    realized, geometry, diagnostics = clusters.resolve_source_corner_clusters(
        part, points, labels, points, labels, SURFACE, [], {2, 3}, 0.01, config
    )
    assert diagnostics["sweep_corner_cluster_resolved_count"] == 0
    assert not diagnostics["sweep_corner_cluster_budget_exhausted"]
    np.testing.assert_array_equal(realized, points)
    assert geometry == labels
