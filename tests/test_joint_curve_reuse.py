"""Completed local curve certificates survive wider grids without extra work."""

from types import SimpleNamespace

import numpy as np
import pytest

from _group_routing_cases import make_group_routing_case
from coilgen import joint_route_shapes as shapes
from coilgen.connector_bridges import BridgeGeneration
from coilgen.geometry import build_cylinder_mesh
from coilgen.joint_cylinder_curves import _cylinder_chart
from coilgen.manufacturing import _SurfaceProjector
from coilgen.models import CoilPart, WirePathGeometry
from coilgen.routing_jumps import raised_span_metrics


def _evaluator():
    case = make_group_routing_case("facing_corridor")
    return shapes.JointTransitionEvaluator(case.part, case.config, _protected_segments={}, search_mode="progressive")


def _points():
    return np.array([[-0.01, 0, 0], [0, 0, 0.001], [0.01, 0, 0]])


@pytest.mark.parametrize("reasons", [(), ("lift_limit_violation",)])
def test_repeated_grid_geometry_reuses_completed_screens_without_changing_candidates(monkeypatch, reasons):
    evaluator, points, calls = _evaluator(), _points(), []
    monkeypatch.setattr(shapes, "_validate_support", lambda *_args: calls.append(True) or reasons)
    monkeypatch.setattr(evaluator, "jump_count", lambda _curve: 1)
    first, second = SimpleNamespace(points=points), SimpleNamespace(points=points.copy())
    expected = [first] if not reasons else []
    selected, rejected = evaluator._supported(BridgeGeneration((first,)), points, [0, 0, 1])
    assert selected == expected and rejected == set(reasons)
    used_pairs, used_scores = evaluator.work.spatial_pairs, evaluator.work.interaction_scores
    # Reuse is valid even once the geometric work caps have been reached.
    evaluator.work.spatial_pairs = evaluator.limits.maximum_spatial_pairs
    evaluator.work.interaction_scores = evaluator.limits.maximum_interaction_scores
    selected, rejected = evaluator._supported(BridgeGeneration((second,)), points, [0, 0, 1])
    assert selected == ([second] if not reasons else []) and rejected == set(reasons)
    assert calls == [True]
    assert evaluator.diagnostics["support_screen_cache_hits"] == 1
    assert evaluator.diagnostics["generated_curve_count"] == 2
    assert evaluator.diagnostics["curve_sample_count"] == 2 * len(points)
    assert evaluator.work.spatial_pairs == evaluator.limits.maximum_spatial_pairs >= used_pairs
    assert evaluator.work.interaction_scores == evaluator.limits.maximum_interaction_scores >= used_scores


@pytest.mark.parametrize("change", ["chord", "ports", "normal", "lift_sign", "clearance"])
def test_screen_reuse_requires_the_exact_support_and_continuation_contract(monkeypatch, change):
    evaluator, points, calls = _evaluator(), _points(), []
    monkeypatch.setattr(shapes, "_validate_support", lambda *_args: calls.append(True) or ())
    monkeypatch.setattr(evaluator, "jump_count", lambda _curve: 1)
    evaluator._supported(BridgeGeneration((SimpleNamespace(points=points),)), points, [0, 0, 1])
    source, normal, proposed = points.copy(), [0, 0, 1], points.copy()
    if change == "chord":
        proposed[1, 2] += 1e-6
    elif change == "ports":
        source[0, 0] += 1e-6
    elif change == "normal":
        normal = [0, 0, -1]
    elif change == "lift_sign":
        evaluator.allowed_shift *= -1
    else:
        evaluator.clearance += 1e-6
    evaluator._supported(BridgeGeneration((SimpleNamespace(points=proposed),)), source, normal)
    assert calls == [True, True]


def test_external_geometry_reuse_does_not_transfer_support_certificates(monkeypatch):
    points, calls = _points(), []
    generation = BridgeGeneration((SimpleNamespace(points=points),))
    monkeypatch.setattr(shapes, "_validate_support", lambda *_args: calls.append(True) or ())
    for evaluator in (_evaluator(), _evaluator()):
        monkeypatch.setattr(evaluator, "jump_count", lambda _curve: 1)
        evaluator._supported(generation, points, [0, 0, 1])
    assert calls == [True, True]


def test_uncertified_budget_failure_is_retried_and_remains_explicit(monkeypatch):
    evaluator, points, calls = _evaluator(), _points(), []
    generation = BridgeGeneration((SimpleNamespace(points=points),))
    monkeypatch.setattr(
        shapes,
        "_validate_support",
        lambda *_args: calls.append(True) or ("support_certificate_budget_exceeded",),
    )
    for _attempt in range(2):
        with pytest.raises(shapes.RoutingEvaluationBudgetExceeded, match="support_certificate_budget_exceeded"):
            evaluator._supported(generation, points, [0, 0, 1])
    assert calls == [True, True] and not evaluator._curve_screen_results


def test_jump_reuse_requires_exact_geometry_and_attachment_labels(monkeypatch):
    evaluator, points, calls = _evaluator(), _points(), []

    def metrics(_points, _projector, *, segment_geometry, **_kwargs):
        calls.append(tuple(segment_geometry))
        return {"jump_count": int(WirePathGeometry.FREE_3D in segment_geometry)}

    monkeypatch.setattr(shapes, "raised_span_metrics", metrics)
    first = SimpleNamespace(points=points, segment_geometry=[WirePathGeometry.FREE_3D] * 2)
    same = SimpleNamespace(points=points.copy(), segment_geometry=["free_3d"] * 2)
    surface = SimpleNamespace(points=points.copy(), segment_geometry=[WirePathGeometry.SURFACE] * 2)
    assert evaluator.jump_count(first) == evaluator.jump_count(same) == 1
    assert evaluator.jump_count(surface) == 0
    assert len(calls) == 2 and evaluator.diagnostics["jump_geometry_cache_hits"] == 1
    first.points[1, 2] += 1e-6
    assert evaluator.jump_count(first) == 1 and len(calls) == 3


def test_reused_screen_still_allows_live_cancellation(monkeypatch):
    evaluator, points = _evaluator(), _points()
    generation = BridgeGeneration((SimpleNamespace(points=points),))
    monkeypatch.setattr(shapes, "_validate_support", lambda *_args: ())
    monkeypatch.setattr(evaluator, "jump_count", lambda _curve: 1)
    evaluator._supported(generation, points, [0, 0, 1])

    def cancel(_diagnostics):
        raise RuntimeError("cancelled")

    evaluator.progress_callback = cancel
    with pytest.raises(RuntimeError, match="cancelled"):
        evaluator._supported(generation, points, [0, 0, 1])


@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("height", [0.0, 0.5e-6, 2e-6, 0.006])
def test_faceted_jump_distances_preserve_counts_at_physical_height_thresholds(transformed, height):
    mesh = build_cylinder_mesh(0.25, 0.066, 28, 20, 1, 0, 0, 0)
    if transformed:
        rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [2, -1, 4], [3, 4, -1]]))
        mesh.vertices = mesh.vertices @ rotation.T + np.array([0.19, -0.42, 0.063])
    case = make_group_routing_case("facing_corridor")
    evaluator = shapes.JointTransitionEvaluator(CoilPart(mesh=mesh), case.config, _protected_segments={})
    chart = _cylinder_chart(evaluator.projector)
    parameter = np.linspace(0, 1, 101)
    developed = np.column_stack(
        ((0.1 + 0.8 * parameter) * chart.lengths[0], -0.025 + 0.05 * parameter, np.zeros(len(parameter)))
    )
    normal = np.cross(chart.directions[0], chart.axis)
    points = chart.map(developed) + height * np.sin(2 * np.pi * parameter)[:, None] ** 2 * normal
    curve = SimpleNamespace(points=points)
    expected = raised_span_metrics(
        points,
        evaluator.projector,
        segment_geometry=shapes._curve_geometry(curve),
        tolerance_m=evaluator.jump_tolerance,
        maximum_sample_spacing_m=evaluator.jump_spacing,
    )
    assert evaluator.jump_count(curve) == expected["jump_count"]
    assert not isinstance(evaluator._jump_projector, _SurfaceProjector)


def _coupling_choices():
    horizontal = np.array([[-0.01, 0, 0], [0.01, 0, 0]])
    vertical = np.array([[0, -0.01, 0], [0, 0.01, 0]])
    return [
        [SimpleNamespace(points=points), SimpleNamespace(points=points + [0, 0, 0.002])]
        for points in (horizontal, vertical)
    ]


def _rank(choices, *, evaluator=None, diagnostics=None):
    return shapes._rank_curve_combinations(
        choices,
        np.empty((0, 2, 3)),
        0.001,
        0.003,
        2,
        work=evaluator.coupling_work if evaluator is not None else None,
        diagnostics=diagnostics,
        protected_clearance_certificate=lambda *_args: True,
        coupling_evaluator=evaluator,
    )


def test_cross_topology_coupling_reuses_exact_scores_and_indexes_without_changing_rank():
    evaluator, choices, diagnostics = _evaluator(), _coupling_choices(), {}
    expected = _rank(choices)
    assert _rank(choices, evaluator=evaluator, diagnostics=diagnostics) == expected
    assert diagnostics["new_transition_option_pair_scores"] == 4
    assert diagnostics["reused_transition_option_pair_scores"] == 0
    records = [evaluator.coupling_geometry(curve) for domain in choices for curve in domain]
    assert all(not geometry.flags.writeable for geometry, _index, _key in records)
    # Fresh curve objects from a different topology can have identical chords.
    repeated = [[SimpleNamespace(points=curve.points.copy()) for curve in domain] for domain in choices]
    evaluator.coupling_work.interaction_scores = evaluator.limits.maximum_interaction_scores
    evaluator.coupling_work.spatial_pairs = evaluator.limits.maximum_coupling_spatial_pairs
    assert _rank(repeated, evaluator=evaluator, diagnostics=diagnostics) == expected
    assert diagnostics["new_transition_option_pair_scores"] == 0
    assert diagnostics["reused_transition_option_pair_scores"] == 4
    assert evaluator.coupling_work.interaction_scores == evaluator.limits.maximum_interaction_scores
    assert evaluator.coupling_work.spatial_pairs == evaluator.limits.maximum_coupling_spatial_pairs
    for curve, record in zip((curve for domain in repeated for curve in domain), records):
        assert evaluator.coupling_geometry(curve) is record


@pytest.mark.parametrize("change", ["geometry", "cap", "query_direction"])
def test_coupling_reuse_requires_exact_geometry_cap_and_query_direction(change):
    evaluator, choices = _evaluator(), _coupling_choices()
    first, second = (evaluator.coupling_geometry(domain[0]) for domain in choices)
    evaluator.coupling_gap(first, second, 0.001, evaluator.coupling_work)
    used = evaluator.coupling_work.interaction_scores
    if change == "geometry":
        choices[0][0].points[0, 2] += 1e-6
        first = evaluator.coupling_geometry(choices[0][0])
    elif change == "query_direction":
        first, second = second, first
    cap = 0.002 if change == "cap" else 0.001
    gap = evaluator.coupling_gap(first, second, cap, evaluator.coupling_work)
    assert evaluator.coupling_work.interaction_scores == used + 1
    reference = shapes._ShapeWork(evaluator.limits).score(first[1], second[0], cap)
    assert gap == reference


@pytest.mark.parametrize("failure", ["score_budget", "spatial_budget", "cancelled"])
def test_failed_or_interrupted_coupling_queries_are_retried_without_cached_scores(failure):
    evaluator, choices = _evaluator(), _coupling_choices()
    first, second = (evaluator.coupling_geometry(domain[0]) for domain in choices)
    work = shapes._ShapeWork(evaluator.limits)
    if failure == "score_budget":
        work.interaction_scores = work.limits.maximum_interaction_scores
        expected_error = shapes._DistanceBudgetExceeded
    elif failure == "spatial_budget":
        work.spatial_pairs = work.limits.maximum_spatial_pairs
        expected_error = shapes._DistanceBudgetExceeded
    else:
        def cancel(_diagnostics):
            raise RuntimeError("cancelled")

        evaluator.progress_callback = cancel
        expected_error = RuntimeError
    with pytest.raises(expected_error):
        evaluator.coupling_gap(first, second, 0.001, work)
    assert not evaluator._coupling_pair_scores
    evaluator.progress_callback = None
    assert evaluator.coupling_gap(first, second, 0.001, evaluator.coupling_work) == 0.0
    assert evaluator.coupling_work.interaction_scores == 1


def test_coupling_memos_remain_bounded_without_changing_uncached_queries():
    evaluator, choices = _evaluator(), _coupling_choices()
    evaluator.limits = shapes.JointShapeLimits(maximum_curve_samples=4, maximum_interaction_scores=1)
    records = [evaluator.coupling_geometry(curve) for domain in choices for curve in domain]
    assert evaluator._coupling_geometry_samples == 4 and len(evaluator._coupling_geometries) == 2
    for first, second in ((records[0], records[1]), (records[2], records[3])):
        work = shapes._ShapeWork(shapes.JointShapeLimits())
        gap = evaluator.coupling_gap(first, second, 0.001, work)
        assert gap == work.score(first[1], second[0], 0.001)
    assert len(evaluator._coupling_pair_scores) == 1
