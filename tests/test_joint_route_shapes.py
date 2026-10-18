"""Coupled transition selection and a real, default-budget physical repair."""

from copy import deepcopy
from dataclasses import replace
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from _group_routing_cases import make_group_routing_case
from coilgen import joint_route_shapes as shapes
from coilgen.connector_bridges import BridgeGeneration
from coilgen.geometry import build_cylinder_mesh
from coilgen.group_route_validation import (
    GroupRouteValidationLimits,
    _sweep_preview,
    prepare_group_route_baseline,
    validate_group_route_candidate,
)
from coilgen.group_route_support import certify_group_route_support
from coilgen.group_routing import GroupRouteCandidate, GroupRoutingLimits, propose_joint_group_routes
from coilgen.joint_group_adoption import improve_group_routes
from coilgen.joint_cylinder_curves import _cylinder_chart
from coilgen.layout import cleanup_wire_paths
from coilgen.manufacturing import _SurfaceProjector, realize_coil_paths
from coilgen.models import CoilPart, TargetField, WirePath, WirePathGeometry, WireSegmentOwner
from coilgen.physics import biot_savart_wire_path


def _mock_direct_only(monkeypatch, generate):
    """Isolate existing orchestration tests from every analytic curve family."""
    monkeypatch.setattr(shapes, "generate_joint_transition_curves", generate)
    monkeypatch.setattr(shapes, "generate_axial_return_curves", lambda *_args, **_kwargs: BridgeGeneration())
    monkeypatch.setattr(shapes, "generate_planar_return_curves", lambda *_args, **_kwargs: BridgeGeneration())
    monkeypatch.setattr(shapes, "generate_cylindrical_transition_curves", lambda *_args, **_kwargs: BridgeGeneration())


def test_progress_callback_preserves_serializable_transition_cache(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    messages = []
    evaluator = shapes.JointTransitionEvaluator(
        case.part, case.config, progress_callback=lambda values: messages.append(dict(values))
    )
    source = np.array([[-0.005, 0, 0], [0.005, 0, 0]])
    tangent = np.array([1.0, 0, 0])
    _mock_direct_only(
        monkeypatch,
        lambda points, *_a, **_k: BridgeGeneration((SimpleNamespace(points=points.copy()),), "fixture"),
    )
    monkeypatch.setattr(shapes, "_validate_support", lambda *_a: ())
    expected = evaluator.search_cost(source, tangent, tangent)
    assert messages
    restored = pickle.loads(pickle.dumps(evaluator))
    assert restored.search_cost(source, tangent, tangent) == expected
    assert restored.diagnostics["transition_cache_hits"] == 1
    assert restored.progress_callback is None
    # Loading a checkpoint made before progress support does not run __init__.
    del restored.progress_callback
    assert restored.search_cost(source + [0, 0.001, 0], tangent, tangent) == expected


def test_shape_trial_admission_precedes_any_geometry_work(monkeypatch):
    count = shapes.JointShapeLimits().maximum_shape_trials // shapes._TRANSITION_TRIALS + 1
    identifiers = tuple(item for index in range(count) for item in (index, None))
    candidate = SimpleNamespace(segment_ids=identifiers)
    monkeypatch.setattr(shapes, "_SurfaceProjector", lambda *_: pytest.fail("Over-budget support projected"))
    monkeypatch.setattr(shapes, "generate_joint_transition_curves", lambda *_: pytest.fail("Over-budget generated"))
    result = shapes.shape_joint_group_route(None, candidate, None)
    assert result.reason == "joint_shape_trial_budget_exceeded"
    assert result.diagnostics == {
        "transition_count": count,
        "shape_trial_upper_bound": count * shapes._TRANSITION_TRIALS,
        "maximum_shape_trials": shapes.JointShapeLimits().maximum_shape_trials,
        "infeasibility_proven": False,
        "work_budget_exhausted": True,
        "search_failure_scope": "work_budget_admission",
    }
    assert not result.candidates


def test_fifty_two_transitions_reach_shaping_with_protected_edges_exact(monkeypatch):
    """Test scalable admission/assembly, not certification of mocked curves."""
    case = make_group_routing_case("facing_corridor")
    angles = np.linspace(0, 2 * np.pi, 105)
    points = np.column_stack((0.025 * np.cos(angles), 0.025 * np.sin(angles), np.zeros(len(angles))))
    points[-1] = points[0]
    identities = tuple((0, 0, index) if index % 2 == 0 else None for index in range(104))
    owner = WireSegmentOwner(kind="within_group_return", group_index=0)
    path = WirePath(points, [owner] * 104, [WirePathGeometry.FREE_3D] * 104)
    protected = {
        identity: points[index : index + 2].copy() for index, identity in enumerate(identities) if identity is not None
    }
    candidate = GroupRouteCandidate(path, identities, protected, (0,), ("fixture",), [], 1.0, 0.0)
    generated = []

    def generate(source, *_args, **_kwargs):
        generated.append(source.copy())
        return BridgeGeneration((SimpleNamespace(points=source.copy()),), "fixture")

    _mock_direct_only(monkeypatch, generate)
    monkeypatch.setattr(shapes, "_validate_support", lambda *_args: ())
    progress = []
    result = shapes.shape_joint_group_route(case.part, candidate, case.config, progress_callback=progress.append)
    assert result.candidates and len(generated) == 52, result.diagnostics
    assert result.diagnostics["transition_count"] == 52
    assert result.diagnostics["interacting_transition_pairs"] < 52
    np.testing.assert_array_equal(result.candidates[0].path.v, points)
    assert result.candidates[0].segment_ids == identities
    assert all(np.array_equal(result.candidates[0].protected_segments[key], value) for key, value in protected.items())
    completed = [
        event["completed"] for event in progress if event["phase"] == "curve transitions" and "completed" in event
    ]
    assert completed == list(range(53))
    assert progress[-2:] == [
        {"phase": "curve coupling", "completed": 0, "total": None},
        {"phase": "curve coupling", "status": "finished"},
    ]


def test_sparse_coupled_search_only_scores_spatial_neighbors():
    horizontal = np.array([[-0.01, 0, 0], [0.01, 0, 0]])
    vertical = np.array([[0, -0.01, 0], [0, 0.01, 0]])
    choices = [
        [SimpleNamespace(points=points), SimpleNamespace(points=points + [0, 0, 0.002])]
        for points in (horizontal, vertical)
    ]
    choices.extend([[SimpleNamespace(points=horizontal + [index * 0.1, 0, 0])] for index in range(1, 51)])
    diagnostics = {}
    results = shapes._rank_curve_combinations(choices, np.empty((0, 2, 3)), 0.001, 0.003, 2, diagnostics=diagnostics)
    assert all(selected[0] != selected[1] and gap == 0.001 for selected, gap, _ in results)
    assert diagnostics["interacting_transition_pairs"] == 1
    assert diagnostics["scored_transition_option_pairs"] == 4
    assert diagnostics["possible_transition_pairs"] == 1326


def _certified_protected_curve(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    protected = {
        (0, 0, 0): np.array([[0.05, 0.05, 0.0], [0.06, 0.05, 0.0]]),
        (0, 0, 1): np.array([[0.06, 0.05, 0.0], [0.07, 0.05, 0.0]]),
    }
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config, _protected_segments=protected)
    points = np.array([[-0.02, -0.02, 0.0], [0.0, -0.02, 0.0], [0.02, -0.02, 0.0]])
    curve = SimpleNamespace(points=points)
    monkeypatch.setattr(shapes, "_validate_support", lambda *_args: ())
    supported, _reasons = evaluator._supported(BridgeGeneration((curve,)), points, np.array([0, 0, 1.0]))
    assert supported == [curve]
    return evaluator, curve


def test_coupling_reuses_exact_protected_clearance_without_resetting_spent_work(monkeypatch):
    evaluator, curve = _certified_protected_curve(monkeypatch)
    evaluator.work.spatial_pairs = evaluator.limits.maximum_spatial_pairs
    evaluator.work.interaction_scores = evaluator.limits.maximum_interaction_scores
    diagnostics = {}
    result = shapes._rank_curve_combinations(
        [[curve]],
        evaluator.obstacles[::-1],
        evaluator.clearance,
        3 * max(evaluator.clearance, evaluator.minimum_radius),
        1,
        work=evaluator.work,
        diagnostics=diagnostics,
        protected_clearance_certificate=evaluator.protected_clearance_certificate,
    )
    assert result[0][1] == evaluator.clearance
    assert diagnostics["reused_protected_clearance_certificates"] == 1
    assert evaluator.work.spatial_pairs == evaluator.limits.maximum_spatial_pairs
    assert evaluator.work.interaction_scores == evaluator.limits.maximum_interaction_scores


@pytest.mark.parametrize("change", ["curve", "ports", "protected", "clearance", "continuation", "uncertified"])
def test_changed_protected_clearance_contract_consumes_fresh_bounded_work(monkeypatch, change):
    evaluator, curve = _certified_protected_curve(monkeypatch)
    protected = evaluator.obstacles.copy()
    clearance = evaluator.clearance
    continuation = 3 * max(clearance, evaluator.minimum_radius)
    if change == "curve":
        curve.points[1, 2] += 1e-6
    elif change == "ports":
        curve.points[0, 0] += 1e-6
    elif change == "protected":
        protected[0, 0, 0] += 1e-6
    elif change == "clearance":
        clearance *= 0.9
    elif change == "continuation":
        continuation += 1e-6
    elif change == "uncertified":
        curve = SimpleNamespace(points=curve.points.copy())
    evaluator.work.interaction_scores = evaluator.limits.maximum_interaction_scores
    with pytest.raises(shapes._DistanceBudgetExceeded, match="joint_interaction_score_budget_exceeded"):
        shapes._rank_curve_combinations(
            [[curve]],
            protected,
            clearance,
            continuation,
            1,
            work=evaluator.work,
            protected_clearance_certificate=evaluator.protected_clearance_certificate,
        )


def test_mutable_interactions_have_a_separate_hard_budget_from_spent_support(monkeypatch):
    evaluator, first = _certified_protected_curve(monkeypatch)
    second = SimpleNamespace(points=np.array([[0.0, -0.03, 0.0], [0.0, -0.01, 0.0], [0.0, 0.01, 0.0]]))
    supported, _reasons = evaluator._supported(BridgeGeneration((second,)), second.points, np.array([0, 0, 1.0]))
    assert supported == [second]
    evaluator.work.spatial_pairs = evaluator.limits.maximum_spatial_pairs

    def rank():
        return shapes._rank_curve_combinations(
            [[first], [second]],
            evaluator.obstacles,
            evaluator.clearance,
            3 * max(evaluator.clearance, evaluator.minimum_radius),
            1,
            work=evaluator.coupling_work,
            protected_clearance_certificate=evaluator.protected_clearance_certificate,
        )

    result = rank()
    assert result[0][1] == 0.0
    assert evaluator.coupling_work.spatial_pairs > 0
    diagnostics = evaluator.work_diagnostics()
    assert diagnostics["support_spatial_candidate_pairs"] == 8000000
    assert diagnostics["coupling_spatial_candidate_pairs"] == evaluator.coupling_work.spatial_pairs
    assert diagnostics["maximum_total_spatial_pairs"] == 16000000
    # The separate coupling phase is also hard bounded across repeated shapes.
    evaluator.coupling_work.spatial_pairs = evaluator.limits.maximum_coupling_spatial_pairs
    with pytest.raises(shapes._DistanceBudgetExceeded, match="joint_spatial_pair_budget_exceeded"):
        rank()
    assert evaluator.work.spatial_pairs == 8000000


def test_b0_preference_is_unsigned_rotation_invariant_and_clearance_secondary():
    transverse = np.array([[-1.0, 0, 0], [0, 1, 0], [1, 0, 0]])
    axial = np.array([[-1.0, 0, 0], [0, 0, 1], [1, 0, 0]])
    choices = [[SimpleNamespace(points=points) for points in (transverse, axial)]]
    empty = np.empty((0, 2, 3))

    def rank(values, direction, weight, obstacles=empty):
        return shapes._rank_curve_combinations(values, obstacles, 0.1, 0, 1, b0_direction=direction, b0_weight=weight)[
            0
        ]

    assert rank(choices, [0, 0, 1], 0)[0] == (0,)
    weighted = rank(choices, [0, 0, 1], 1)
    assert weighted[0] == (1,)
    assert rank(choices, [0, 0, -3], 1) == weighted
    rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [2, -1, 4], [3, 4, -1]]))
    rotated = [[SimpleNamespace(points=item.points @ rotation.T) for item in choices[0]]]
    result = rank(rotated, rotation @ [0, 0, 1], 1)
    assert result[0] == weighted[0] and result[2] == pytest.approx(weighted[2])
    obstacle = np.array([[[-0.1, 0, 1], [0.1, 0, 1]]])
    assert rank(choices, [0, 0, 1], 100, obstacle)[0] == (0,)


def test_coupled_selection_prefers_fewer_jumps_before_any_length_guide():
    short = SimpleNamespace(points=np.array([[0.0, 0, 0], [1, 0, 0]]))
    long = SimpleNamespace(points=np.array([[0.0, 0, 0], [0, 100, 0], [1, 0, 0]]))
    result = shapes._rank_curve_combinations([[short, long]], np.empty((0, 2, 3)), 0.1, 0, 1, jump_counts=[[2, 1]])
    assert result[0][0] == (1,)
    assert result[0][2] > 100


def test_equal_jump_axial_alternative_survives_scalar_guide_shortlisting():
    choices = [
        [
            SimpleNamespace(
                points=np.array([[0.0, 0, 0], [1.0, offset, 0], [2.0, 0, 0]]),
                profile_family="b0_straight_return" if offset == 10 else "quintic",
            )
            for offset in (0.0, 0.1, 0.2, 10.0)
        ]
    ]
    result = shapes._rank_curve_combinations(choices, np.empty((0, 2, 3)), 0.1, 0, 2, jump_counts=[[1, 1, 1, 1]])
    assert [state[0] for state in result] == [(0,), (3,)]


def test_transition_shortlist_keeps_full_height_needed_to_clear_another_return(monkeypatch):
    """Same-jump half-height siblings must not erase the feasible full lift."""
    case = make_group_routing_case("facing_corridor")
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config)
    evaluator.b0_weight = 0.0

    def curve(height, offset=0.0, jumps=1):
        return SimpleNamespace(
            points=np.array([[-0.02, 0, 0], [-0.015, offset, height], [0.015, offset, height], [0.02, 0, 0]]),
            profile_family="cylinder_compact_lifted_return",
            height_m=height,
            jumps=jumps,
        )

    half_height = [curve(0.003, offset) for offset in np.linspace(0.0, 0.0001, 32)]
    full_height = curve(0.006)
    lower_jump = curve(0.0, jumps=0)
    higher_jump = curve(0.006, jumps=2)
    monkeypatch.setattr(evaluator, "jump_count", lambda value: value.jumps)
    shortlisted = evaluator._shortlist([*half_height, full_height, higher_jump, lower_jump])
    assert len(shortlisted) == 32  # Both regular and physical guide branches stay represented.
    assert shortlisted[0] is lower_jump
    assert any(value is full_height for value in shortlisted)
    assert [value.jumps for value in shortlisted] == sorted(value.jumps for value in shortlisted)
    # Every cheap 3 mm rail passes within 0.6 mm of this other return. The
    # retained 6 mm rail meets 3.5 mm clearance without adding another jump.
    obstacle = np.array([[[-0.01, 0, 0.0025], [0.01, 0, 0.0025]]])
    result = shapes._rank_curve_combinations(
        [shortlisted], obstacle, 0.0035, 0, 1, jump_counts=[[value.jumps for value in shortlisted]]
    )
    assert len(result) == 1
    assert shortlisted[result[0][0][0]] is full_height
    assert result[0][1] == pytest.approx(0.0035)


def test_regular_combination_ranking_uses_bend_distribution_and_keeps_alternatives():
    theta = np.linspace(0, np.pi / 2, 65)
    gentle = SimpleNamespace(points=np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta))))
    corner = SimpleNamespace(points=np.array([[1.0, 0, 0], [1, 1, 0], [0, 1, 0]]))
    # The two candidates have the same endpoints and no protected obstacles;
    # the more distributed bend is preferred without erasing the other family.
    result = shapes._rank_curve_combinations(
        [[corner, gentle]],
        np.empty((0, 2, 3)),
        0.1,
        0,
        2,
        b0_direction=[1, 0, 0],
        b0_weight=10.0,
        selection_policy="regular",
    )
    assert [state[0] for state in result] == [(1,), (0,)]


def test_complete_feasible_families_survive_same_axial_count_with_three_slots():
    choices = []
    for offset in (0.0, 0.5):
        choices.append(
            [
                SimpleNamespace(
                    points=np.array([[offset, 0, 0], [offset - deviation, 0, 0.05], [offset + 0.01, 0, 0.1]]),
                    profile_family=family,
                )
                for deviation, family in (
                    (0.005, "planar_biarc_return"),
                    (0.05, "cylinder_compact_lifted_return"),
                    (0.1, "b0_straight_return"),
                )
            ]
        )
    result = shapes._rank_curve_combinations(choices, np.empty((0, 2, 3)), 0.001, 0, 3, jump_counts=[[1] * 3, [1] * 3])
    assert [state[0] for state in result] == [(0, 0), (1, 1), (2, 2)]
    higher_jump = shapes._rank_curve_combinations(
        choices, np.empty((0, 2, 3)), 0.001, 0, 3, jump_counts=[[1, 1, 2], [1, 1, 2]]
    )
    assert all(2 not in state[0] for state in higher_jump)


def test_different_infeasible_gaps_do_not_erase_distinct_complete_families():
    families = ("planar_biarc_lifted_return", "cylinder_compact_lifted_return", "b0_straight_return")
    choices = [
        [
            SimpleNamespace(points=np.array([[0, 0, 0], [0.01, gap, 0.05], [0.02, 0, 0.1]]), profile_family=family)
            for family, gap in zip(families, (0.001, 0.0005, 0.0002))
        ]
    ]
    obstacle = np.array([[[0.009, 0, 0.05], [0.011, 0, 0.05]]])
    result = shapes._rank_curve_combinations(choices, obstacle, 0.0035, 0, 3, jump_counts=[[1] * 3])
    assert [state[0] for state in result] == [(0,), (1,), (2,)]
    assert all(state[1] < 0.0035 for state in result)
    assert result[0][1] > result[1][1] > result[2][1]


@pytest.mark.parametrize(
    "family",
    [
        "b0_straight_return",
        "planar_biarc_return",
        "planar_biarc_lifted_return",
        "cylinder_compact_return",
        "cylinder_compact_lifted_return",
    ],
)
def test_equal_jump_coupled_routes_keep_complete_compact_axial_spans_for_field_evaluation(family):
    """A longer complete pair reaches field checks instead of only shorter cubics."""
    choices = []
    for offset in (0.0, 0.5):
        short = [
            SimpleNamespace(
                points=np.array([[offset, 0, 0], [offset + deviation, 0, 0.05], [offset + 0.01, 0, 0.1]]),
                profile_family="quintic",
            )
            for deviation in (0.005, 0.007, 0.009)
        ]
        compact = SimpleNamespace(
            points=np.array([[offset, 0, 0], [offset - 0.05, 0, 0], [offset - 0.05, 0, 0.1], [offset + 0.01, 0, 0.1]]),
            profile_family=family,
        )
        choices.append([*short, compact])
    result = shapes._rank_curve_combinations(choices, np.empty((0, 2, 3)), 0.001, 0, 2, jump_counts=[[1] * 4, [1] * 4])
    assert [state[0] for state in result] == [(0, 0), (3, 3)]
    assert result[1][2] > result[0][2]
    # Diversity is restricted to the best feasible jump class. It cannot send
    # a higher-jump return ahead of equally feasible lower-jump candidates.
    higher_jump = shapes._rank_curve_combinations(
        choices, np.empty((0, 2, 3)), 0.001, 0, 2, jump_counts=[[1, 1, 1, 2], [1, 1, 1, 2]]
    )
    assert all(3 not in selected for selected, _gap, _cost in higher_jump)


def test_axial_family_runs_even_with_a_feasible_direct_arch(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    source = np.array([[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0]])
    direct = SimpleNamespace(points=source.copy(), profile_family="quintic")
    axial = SimpleNamespace(
        points=np.array([source[0], [0, -0.02, 0.001], source[-1]]), profile_family="b0_straight_return"
    )
    calls = []
    monkeypatch.setattr(
        shapes, "generate_joint_transition_curves", lambda *_args, **_kwargs: BridgeGeneration((direct,), "fixture")
    )

    def generate(*_args, **_kwargs):
        calls.append(True)
        return BridgeGeneration((axial,), "fixture")

    monkeypatch.setattr(shapes, "generate_axial_return_curves", generate)
    monkeypatch.setattr(shapes, "_validate_support", lambda *_: ())
    monkeypatch.setattr(shapes._ShapeWork, "score", lambda self, index, points, clearance, excluded=(): clearance)
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config)
    options = evaluator.supported_curves(source, [1, 0, 0], [1, 0, 0])
    assert len(calls) == 1
    assert {curve.profile_family for curve in options} == {"quintic", "b0_straight_return"}
    assert [evaluator.jump_count(curve) for curve in options] == [0, 1]


@pytest.mark.parametrize("transformed", [False, True])
def test_antipodal_cylinder_ports_keep_intrinsic_surface_routes_without_an_arch_frame(monkeypatch, transformed):
    mesh = build_cylinder_mesh(0.25, 0.066, 28, 20, 1, 0, 0, 0)
    chart = _cylinder_chart(_SurfaceProjector(mesh))
    developed = np.array(
        [[0.3 * chart.lengths[0], -0.005, 0], [0.3 * chart.lengths[0] + chart.perimeter / 2, 0.005, 0]]
    )
    source = chart.map(developed)
    tangents = [-chart.directions[0], -chart.directions[14]]
    config = deepcopy(make_group_routing_case("facing_corridor").config)
    config.skip_normal_shift = True
    if transformed:
        rotation, _ = np.linalg.qr(np.array([[1.0, 2, 3], [2, -1, 4], [3, 4, -1]]))
        shift = np.array([0.19, -0.42, 0.063])
        mesh.vertices = mesh.vertices @ rotation.T + shift
        source = source @ rotation.T + shift
        tangents = [value @ rotation.T for value in tangents]
        config.b_0_direction = rotation @ np.asarray(config.b_0_direction)
    part = CoilPart(mesh=mesh)
    projector = _SurfaceProjector(mesh)
    assert np.linalg.norm(projector.project_with_face_normals(source)[2].sum(axis=0)) < 1e-12

    def no_cartesian_frame(*_args, **_kwargs):
        pytest.fail("Opposite facets must not generate a Cartesian arch or corridor.")

    for name in (
        "generate_joint_transition_curves",
        "generate_axial_return_curves",
        "generate_planar_return_curves",
        "generate_joint_corridor_curves",
    ):
        monkeypatch.setattr(shapes, name, no_cartesian_frame)
    evaluator = shapes.JointTransitionEvaluator(part, config, _protected_segments={})
    curves = evaluator.supported_curves(source, *tangents)
    assert curves, evaluator.last_rejected_reasons
    assert evaluator.expand_corridors(source, *tangents) == curves
    for curve in curves:
        np.testing.assert_array_equal(curve.points[[0, -1]], source)
        assert curve.profile_family == "cylinder_surface"
        assert evaluator.jump_count(curve) == 0
        assert set(curve.segment_geometry) == {WirePathGeometry.SURFACE}
        midpoints = (curve.points[:-1] + curve.points[1:]) / 2
        np.testing.assert_allclose(projector.project(midpoints)[0], midpoints, atol=1e-12)
        for chord, tangent in (
            (curve.points[1] - curve.points[0], tangents[0]),
            (curve.points[-1] - curve.points[-2], tangents[1]),
        ):
            assert chord @ tangent >= np.cos(np.radians(2)) * np.linalg.norm(chord) * np.linalg.norm(tangent)


def test_graph_cost_uses_measured_jumps_while_scalar_call_stays_compatible(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    source = np.array([[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0]])
    long_surface = SimpleNamespace(points=np.array([source[0], [0.0, 0.025, 0.0], source[-1]]))
    short_raised = SimpleNamespace(points=np.array([source[0], [0.0, -0.02, 0.001], source[-1]]))
    _mock_direct_only(monkeypatch, lambda *_args, **_kwargs: BridgeGeneration((short_raised, long_surface), "fixture"))
    monkeypatch.setattr(shapes, "_validate_support", lambda *_: ())
    monkeypatch.setattr(shapes._ShapeWork, "score", lambda self, index, points, clearance, excluded=(): clearance)
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config)
    score = evaluator.search_cost(source, [1, 0, 0], [1, 0, 0])
    assert score.jumps == 0
    assert float(score) == pytest.approx(
        shapes.routing_cost(long_surface.points, case.config.b_0_direction, case.config.joint_group_routing_b0_weight)
    )
    assert evaluator(source, [1, 0, 0], [1, 0, 0]) < float(score)


def test_jump_sampling_exhaustion_is_explicit_and_not_cached_as_infeasible(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    source = np.array([[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0]])
    _mock_direct_only(monkeypatch, lambda *_args, **_kwargs: BridgeGeneration((SimpleNamespace(points=source),)))
    monkeypatch.setattr(shapes, "_validate_support", lambda *_: ())
    monkeypatch.setattr(shapes._ShapeWork, "score", lambda self, index, points, clearance, excluded=(): clearance)

    def fail(*_args, **_kwargs):
        raise ValueError("jump_sampling_budget_exceeded")

    monkeypatch.setattr(shapes, "raised_span_metrics", fail)
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config)
    with pytest.raises(shapes.RoutingEvaluationBudgetExceeded, match="jump_sampling_budget_exceeded"):
        evaluator.search_cost(source, [1, 0, 0], [1, 0, 0])
    assert not evaluator._evaluated


def test_spatial_score_budget_exhaustion_is_explicit():
    points = np.array([[0.0, 0, 0], [1, 0, 0]])
    choices = [[SimpleNamespace(points=points)], [SimpleNamespace(points=points + [0, 0.01, 0])]]
    limits = shapes.JointShapeLimits(maximum_interaction_scores=2)
    with pytest.raises(shapes._DistanceBudgetExceeded, match="joint_interaction_score_budget_exceeded"):
        shapes._rank_curve_combinations(choices, np.empty((0, 2, 3)), 0.1, 0, 1, work=shapes._ShapeWork(limits))


def test_spatial_distance_index_keeps_long_obstacle_with_remote_midpoint():
    first = np.array([[[0.0, -1, 0], [0, 1, 0]]])
    obstacles = np.array([[[-100.0, 0, 0.03], [1, 0, 0.03]], [[20.0, 2, 0], [21, 2, 0]]])
    assert shapes._capped_curve_gap(first, obstacles, 0.1) == pytest.approx(0.03)


def test_coupled_ranking_avoids_two_individually_clear_crossing_curves():
    horizontal = np.array([[-0.01, 0, 0], [0.01, 0, 0]])
    vertical = np.array([[0, -0.01, 0], [0, 0.01, 0]])
    choices = [
        [SimpleNamespace(points=points), SimpleNamespace(points=points + [0, 0, 0.002])]
        for points in (horizontal, vertical)
    ]
    results = shapes._rank_curve_combinations(choices, np.empty((0, 2, 3)), 0.001, 0.003, 2)
    assert len(results) == 2
    assert all(selected[0] != selected[1] and gap == 0.001 for selected, gap, _ in results)


def test_scoring_exclusion_follows_connectivity_not_geometric_proximity():
    segments = np.array(
        [[[0.0, 0, 0], [1, 0, 0]], [[1, 0, 0], [2, 0, 0]], [[2, 0, 0], [3, 0, 0]], [[0.1, -1, 0], [0.1, 1, 0]]]
    )
    remote = shapes._remote_protected_segments(segments, [[0, 0, 0]], 1.5)
    np.testing.assert_array_equal(remote, segments[2:])


def test_skip_normal_shift_forbids_added_arch_height_and_support_offset(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    case.config.joint_group_routing_selection_policy = "physical"
    case.config.skip_normal_shift = True
    proposal = propose_joint_group_routes(case.part, case.config).candidates[0]
    heights, offsets = [], []

    def generate(source, *_args, **kwargs):
        heights.append(kwargs["maximum_height_m"])
        return BridgeGeneration((SimpleNamespace(points=source.copy()),), "fixture")

    def support(_projector, _point, _segments, _normal, maximum_lift, *_args):
        offsets.append(maximum_lift)
        return ()

    _mock_direct_only(monkeypatch, generate)
    monkeypatch.setattr(shapes, "_validate_support", support)
    result = shapes.shape_joint_group_route(case.part, proposal, case.config, maximum_candidates=1)
    assert result.candidates and heights and offsets
    assert set(heights) == set(offsets) == {0.0}


def test_ephemeral_proposal_cache_preserves_geometry_and_rechecks_support(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    case.config.joint_group_routing_selection_policy = "physical"
    proposal = propose_joint_group_routes(case.part, case.config).candidates[0]
    generated, checked = [], []

    def generate(source, *_args, **kwargs):
        generated.append(kwargs)
        return BridgeGeneration((SimpleNamespace(points=source.copy()),), "fixture")

    def support(*_args):
        checked.append(True)
        return ()

    _mock_direct_only(monkeypatch, generate)
    monkeypatch.setattr(shapes, "_validate_support", support)
    cache = {}
    first = shapes.shape_joint_group_route(case.part, proposal, case.config, transition_cache=cache)
    calls = len(generated)
    second = shapes.shape_joint_group_route(case.part, proposal, case.config, transition_cache=cache)
    assert len(generated) == calls > 0
    assert len(checked) == 2 * calls
    np.testing.assert_array_equal(first.candidates[0].path.v, second.candidates[0].path.v)
    case.config.normal_shift_length *= 2
    shapes.shape_joint_group_route(case.part, proposal, case.config, transition_cache=cache)
    assert len(generated) == 2 * calls
    identity = next(iter(proposal.protected_segments))
    proposal.protected_segments[identity] = proposal.protected_segments[identity] + [0, 0, 1e-8]
    shapes.shape_joint_group_route(case.part, proposal, case.config, transition_cache=cache)
    assert len(generated) == 3 * calls
    case.config.joint_group_routing_b0_weight = 2.0
    shapes.shape_joint_group_route(case.part, proposal, case.config, transition_cache=cache)
    assert len(generated) == 4 * calls
    case.config.b_0_direction = (0.0, 1.0, 0.0)
    shapes.shape_joint_group_route(case.part, proposal, case.config, transition_cache=cache)
    assert len(generated) == 5 * calls


def test_budget_limited_generation_does_not_poison_ephemeral_cache(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    proposal = propose_joint_group_routes(case.part, case.config).candidates[0]
    cache = {}
    monkeypatch.setattr(
        shapes,
        "generate_joint_transition_curves",
        lambda *_args, **_kwargs: BridgeGeneration(reason="joint_spatial_pair_budget_exceeded"),
    )
    result = shapes.shape_joint_group_route(case.part, proposal, case.config, transition_cache=cache)
    assert result.reason == "joint_spatial_pair_budget_exceeded"
    assert not result.candidates and not cache


@pytest.fixture(scope="module")
def qualified_route():
    case = make_group_routing_case("facing_corridor")
    # Retain the former return-prefix graph as a physical regression. Settings are
    # unchanged: 0.6x0.4mm profile, 1.2mm radius, 2mm lift, 5 degree frame cap.
    proposal = propose_joint_group_routes(
        case.part, case.config, limits=GroupRoutingLimits(return_fractions=(0.75,), candidate_count=6)
    ).candidates[0]
    generation = shapes.shape_joint_group_route(case.part, proposal, case.config, maximum_candidates=3)
    assert generation.candidates, generation.reason
    candidate = generation.candidates[0]
    part = deepcopy(case.part)
    part.wire_path, part.opening_cuts_among_groups = deepcopy(candidate.path), deepcopy(candidate.cuts)
    cleanup_wire_paths([part], config=case.config)
    diagnostics, reasons = _sweep_preview(part, case.config, GroupRouteValidationLimits())
    assert not reasons, reasons
    return case, candidate, part, diagnostics


def test_real_facing_fixture_passes_unchanged_full_body_constraints(qualified_route):
    case, candidate, part, diagnostics = qualified_route
    assert diagnostics["manufacturing_ok"] and diagnostics["mesh_solid_ok"]
    assert diagnostics["exact_collision_completed"] and diagnostics["exact_intersection_count"] == 0
    assert diagnostics["sweep_final_min_bend_radius_m"] >= case.config.sweep_min_bend_radius - 1e-12
    assert diagnostics["sweep_final_max_frame_rotation_deg"] <= case.config.sweep_max_frame_rotation_deg + 1e-6
    assert (
        diagnostics["sweep_capsule_min_centerline_distance_m"] >= diagnostics["sweep_required_centerline_clearance_m"]
    )
    assert diagnostics["sweep_triangle_edge_ratio_max"] <= diagnostics["sweep_triangle_edge_ratio_limit"]
    assert diagnostics["sweep_ring_cross_section_area_max_relative_error"] < 1e-10
    actual = {(tuple(a), tuple(b)) for a, b in zip(part.wire_path.v[:-1], part.wire_path.v[1:])}
    assert all((tuple(a), tuple(b)) in actual for a, b in candidate.protected_segments.values())
    assert candidate.retained_return_length_m == 0
    for cut in candidate.cuts:
        assert cut.direct_distance == pytest.approx(
            np.linalg.norm(cut.connection_path_v[-1] - cut.connection_path_v[0])
        )
        assert np.isinf(cut.geodesic_distance) and not cut.used_direct_fallback
    assert case.part.wire_path is None  # Isolated candidate construction.


def test_progressive_real_route_still_requires_full_body_and_fixed_current_field_validation(qualified_route):
    """A compact-grid route certifies physically but cannot bypass a zero field budget."""
    case, source, original, _diagnostics = qualified_route
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config, search_mode="progressive")
    proposed = shapes.shape_joint_group_route(
        case.part,
        source,
        case.config,
        maximum_candidates=3,
        transition_evaluator=evaluator,
    )
    assert proposed.candidates, proposed.reason
    roi = np.array([[x, y, 0.03] for x in (-0.012, 0, 0.012) for y in (-0.006, 0, 0.006)])
    original_field = biot_savart_wire_path(original.wire_path.v, roi, worker_count=1)
    candidate = next(
        value
        for value in proposed.candidates
        if np.max(np.linalg.norm(biot_savart_wire_path(value.path.v, roi, worker_count=1) - original_field, axis=1)) > 0
    )
    prepared = deepcopy(case.part)
    prepared.wire_path, prepared.opening_cuts_among_groups = deepcopy(candidate.path), deepcopy(candidate.cuts)
    cleanup_wire_paths([prepared], config=case.config)
    diagnostics, reasons = _sweep_preview(prepared, case.config, GroupRouteValidationLimits())
    assert not reasons and diagnostics["manufacturing_ok"]
    assert diagnostics["mesh_solid_ok"] and diagnostics["exact_collision_completed"]
    assert diagnostics["exact_intersection_count"] == 0
    assert diagnostics["sweep_final_min_bend_radius_m"] >= case.config.sweep_min_bend_radius - 1e-12
    kwargs = dict(
        protected_segments=candidate.protected_segments,
        candidate_segment_ids=candidate.segment_ids,
        group_sequence=candidate.group_sequence,
        required_groups=tuple(range(len(prepared.groups))),
        candidate_cuts=candidate.cuts,
        target_points_m=roi,
        maximum_field_delta_t_per_a=0.0,
        apply_normal_shift=False,
    )
    matching = prepare_group_route_baseline(prepared, case.config, roi, reuse_existing_sweep=True)
    accepted = validate_group_route_candidate(prepared, candidate.path, case.config, baseline=matching, **kwargs)
    assert accepted.accepted, accepted.reasons
    support = certify_group_route_support(accepted.prepared_part, case.config)
    assert support.accepted, support.reasons
    reference = prepare_group_route_baseline(original, case.config, roi, reuse_existing_sweep=True)
    rejected = validate_group_route_candidate(original, candidate.path, case.config, baseline=reference, **kwargs)
    assert not rejected.accepted and "field_delta_budget_exceeded" in rejected.reasons
    assert rejected.diagnostics["route_maximum_field_delta_t_per_a"] > 0


@pytest.fixture(scope="module")
def qualified_evaluated_route():
    """Frozen physical seed for the historical pointwise field-budget regression."""
    case = make_group_routing_case("facing_corridor")
    # The dedicated 1 um perturbation below was chosen against this physical
    # profile. Default-regular manufacture/field certification uses qualified_route.
    case = replace(case, config=replace(case.config, joint_group_routing_selection_policy="physical"))
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config)
    proposal = propose_joint_group_routes(
        case.part,
        case.config,
        limits=GroupRoutingLimits(return_fractions=(0.0,), candidate_count=6),
        transition_evaluator=evaluator,
    ).candidates[0]
    generation = shapes.shape_joint_group_route(
        case.part,
        proposal,
        case.config,
        maximum_candidates=3,
        transition_evaluator=evaluator,
    )
    assert generation.candidates, generation.reason
    candidate = generation.candidates[0]
    part = deepcopy(case.part)
    part.wire_path, part.opening_cuts_among_groups = deepcopy(candidate.path), deepcopy(candidate.cuts)
    cleanup_wire_paths([part], config=case.config)
    diagnostics, reasons = _sweep_preview(part, case.config, GroupRouteValidationLimits())
    assert not reasons, reasons
    # Independently certify the seed, including fixed-current fields and final
    # support. Provenance lives on the introduced join owners; transient search
    # status must not authorize objective classification.
    roi = np.array([[x, y, 0.03] for x in (-0.012, 0, 0.012) for y in (-0.006, 0, 0.006)])
    baseline = prepare_group_route_baseline(part, case.config, roi, reuse_existing_sweep=True)
    validation = validate_group_route_candidate(
        part,
        candidate.path,
        case.config,
        protected_segments=candidate.protected_segments,
        candidate_segment_ids=candidate.segment_ids,
        group_sequence=candidate.group_sequence,
        required_groups=tuple(range(len(part.groups))),
        candidate_cuts=candidate.cuts,
        target_points_m=roi,
        maximum_field_delta_t_per_a=0.0,
        apply_normal_shift=False,
        baseline=baseline,
    )
    assert validation.accepted, validation.reasons
    part = validation.prepared_part
    support = certify_group_route_support(part, case.config)
    assert support.accepted, support.reasons
    part.joint_route_diagnostics = {
        "seed_validation": validation.diagnostics,
        "seed_support": support.diagnostics,
    }
    return case, candidate, part, diagnostics


def _perturbed_repair(qualified_route):
    case, candidate, qualified, _ = qualified_route
    baseline = deepcopy(qualified)
    identifiers = candidate.segment_ids
    interior = [i for i in range(1, len(identifiers)) if identifiers[i - 1] is None and identifiers[i] is None]
    index = interior[len(interior) // 2]
    points = baseline.wire_path.v
    first, last = points[index] - points[index - 1], points[index] - points[index + 1]
    length_gradient = first / np.linalg.norm(first) + last / np.linalg.norm(last)
    assert np.linalg.norm(length_gradient) > 1e-8
    points[index] += 1e-6 * length_gradient / np.linalg.norm(length_gradient)
    cleanup_wire_paths([baseline], config=case.config)
    _sweep_preview(baseline, case.config, GroupRouteValidationLimits())
    original_points = baseline.wire_path.v.copy()
    xy = np.array([(x, y) for x in (-0.012, 0, 0.012) for y in (-0.006, 0, 0.006)])
    roi = np.column_stack((xy, np.full(len(xy), 0.03)))
    target = TargetField(
        roi,
        biot_savart_wire_path(original_points, roi) * baseline.contour_step,
        np.ones(len(roi)),
        np.zeros(len(roi), dtype=bool),
        np.ones(len(roi), dtype=int),
    )
    assert case.config.joint_group_routing_max_field_change_rel == 0.001
    return case, candidate, baseline, target, original_points


def test_default_field_budget_rejects_jump_first_shapes_outside_fixed_current_budget(qualified_evaluated_route):
    """Jump priority never overrides the independent 0.1% field gate."""
    case, candidate, baseline, target, original_points = _perturbed_repair(qualified_evaluated_route)
    # This frozen physical pattern was selected by the exhaustive curve graph;
    # keep that search coverage while checking its unchanged 0.1% field gate.
    config = replace(case.config, joint_group_routing_search_mode="thorough")
    selected = improve_group_routes([baseline], config, target)[0]
    assert selected is baseline, selected.joint_route_diagnostics
    assert selected.joint_route_diagnostics["status"] == "baseline_retained", selected.joint_route_diagnostics
    assert selected.manufacturing_diagnostics["manufacturing_ok"]
    attempts = selected.joint_route_diagnostics["attempts"]
    assert attempts and not any(attempt["selected"] for attempt in attempts)
    assert selected.joint_route_diagnostics["baseline_routing_metrics"]["jump_count"] == 2
    field_rejections = 0
    for attempt in attempts:
        validation = attempt["validation"]
        if validation["route_maximum_pointwise_vector_delta_t_per_a"] > validation["route_maximum_delta_t_per_a"]:
            assert "pointwise_field_delta_budget" in attempt["reasons"], attempt
            field_rejections += 1
        else:
            # A near-incumbent alternative can pass the field gate while still
            # failing the unchanged objective/manufacturing improvement gate.
            assert attempt["reasons"] == ["no_routing_objective_or_manufacturing_improvement"], attempt
            assert attempt["routing_metrics"] is not None
            assert (
                validation["physical_maximum_pointwise_vector_delta_t_per_a"]
                <= validation["physical_maximum_delta_t_per_a"]
            )
    assert field_rejections > 0
    actual = {(tuple(a), tuple(b)) for a, b in zip(selected.wire_path.v[:-1], selected.wire_path.v[1:])}
    assert all((tuple(a), tuple(b)) in actual for a, b in candidate.protected_segments.values())
    np.testing.assert_array_equal(baseline.wire_path.v, original_points)


def test_whole_transition_evaluator_scores_generated_geometry_and_reuses_it(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    source = np.array([[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0]])
    shaped = np.array([source[0], [0, -0.025, 0], source[-1]])
    calls = []

    def generate(*args, **kwargs):
        calls.append(True)
        return BridgeGeneration((SimpleNamespace(points=shaped.copy()),), "fixture")

    _mock_direct_only(monkeypatch, generate)
    monkeypatch.setattr(shapes, "_validate_support", lambda *_: ())
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config)
    tangent = np.array([1.0, 0, 0])
    cost = evaluator(source, tangent, tangent)
    assert cost == pytest.approx(
        shapes.routing_cost(shaped, case.config.b_0_direction, case.config.joint_group_routing_b0_weight)
    )
    assert cost > shapes.routing_cost(source, case.config.b_0_direction, case.config.joint_group_routing_b0_weight)
    assert evaluator(source.copy(), tangent.copy(), tangent.copy()) == cost
    assert len(calls) == 1 and evaluator.diagnostics["transition_cache_hits"] == 1
    np.testing.assert_array_equal(evaluator.supported_curves(source, tangent, tangent)[0].points, shaped)


def test_evaluator_distinguishes_unsupported_family_from_exhausted_work(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    source = np.array([[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0]])
    tangent = np.array([1.0, 0, 0])

    def failure(*_args, **_kwargs):
        return BridgeGeneration(reason="no_feasible_joint_transition", rejected_reasons=("insufficient_bend_radius",))

    _mock_direct_only(monkeypatch, failure)
    monkeypatch.setattr(shapes, "generate_joint_corridor_curves", failure)
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config)
    assert np.isinf(evaluator(source, tangent, tangent))
    assert evaluator.diagnostics["unsupported_transition_count"] == 1
    assert evaluator.diagnostics["unsupported_transition_reasons"]["insufficient_bend_radius"] == 1
    limited = shapes.JointTransitionEvaluator(
        case.part, case.config, limits=shapes.JointShapeLimits(maximum_shape_trials=64)
    )
    with pytest.raises(shapes.RoutingEvaluationBudgetExceeded, match="joint_shape_trial_budget_exceeded"):
        limited(source, tangent, tangent)
    assert not limited._evaluated


def test_evaluator_support_budget_is_not_cached_as_infeasibility(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    source = np.array([[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0]])
    tangent = np.array([1.0, 0, 0])
    monkeypatch.setattr(
        shapes,
        "generate_joint_transition_curves",
        lambda *_args, **_kwargs: BridgeGeneration((SimpleNamespace(points=source),), "fixture"),
    )
    monkeypatch.setattr(shapes, "_validate_support", lambda *_: ("support_certificate_budget_exceeded",))
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config)
    with pytest.raises(shapes.RoutingEvaluationBudgetExceeded, match="support_certificate_budget_exceeded"):
        evaluator(source, tangent, tangent)
    assert not evaluator._evaluated


@pytest.mark.parametrize("component", ["vertices", "faces", "face_normals", "vertex_normals"])
def test_evaluator_rejects_changed_support_context(component):
    case = make_group_routing_case("facing_corridor")
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config)
    snapshot = getattr(evaluator.projector, component).copy()
    values = getattr(case.part.mesh, component)
    values.flat[0] += 1 if component == "faces" else 1e-5
    with pytest.raises(ValueError, match="physical context changed"):
        evaluator.check_context(case.part, case.config, evaluator.protected)
    np.testing.assert_array_equal(getattr(evaluator.projector, component), snapshot)


def test_evaluator_corridor_fallback_is_reused_by_final_shaping(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    proposal = propose_joint_group_routes(
        case.part, case.config, limits=GroupRoutingLimits(return_fractions=(0.0,))
    ).candidates[0]
    direct_calls, corridor_calls = [], []

    def direct(*_args, **_kwargs):
        direct_calls.append(True)
        return BridgeGeneration(reason="no_feasible_joint_transition")

    def corridor(source, *_args, **_kwargs):
        corridor_calls.append(True)
        return BridgeGeneration((SimpleNamespace(points=source.copy()),), "fixture")

    _mock_direct_only(monkeypatch, direct)
    monkeypatch.setattr(shapes, "generate_joint_corridor_curves", corridor)
    monkeypatch.setattr(shapes, "_validate_support", lambda *_: ())
    monkeypatch.setattr(shapes._ShapeWork, "score", lambda self, index, points, clearance, excluded=(): clearance)
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config)
    result = shapes.shape_joint_group_route(case.part, proposal, case.config, transition_evaluator=evaluator)
    assert result.candidates
    calls = len(direct_calls)
    again = shapes.shape_joint_group_route(case.part, proposal, case.config, transition_evaluator=evaluator)
    assert len(direct_calls) == len(corridor_calls) == calls
    assert evaluator.diagnostics["transition_cache_hits"] == calls
    np.testing.assert_array_equal(result.candidates[0].path.v, again.candidates[0].path.v)


def test_coupled_conflict_lazily_expands_corridors_even_when_direct_spans_are_locally_valid(monkeypatch):
    # Geometry-only orchestration fixture: the independent body tests above
    # cover fabrication. Two individually supported spans cross each other.
    case = make_group_routing_case("facing_corridor")
    points = np.array(
        [
            [-0.02, 0, 0],
            [-0.01, 0, 0],
            [0.01, 0, 0],
            [0.02, 0, 0],
            [0, -0.02, 0],
            [0, -0.01, 0],
            [0, 0.01, 0],
            [0, 0.02, 0],
            [-0.02, 0, 0],
        ]
    )
    identities = tuple(None if i in (1, 5) else (0, 0, i) for i in range(8))
    protected = {key: points[i : i + 2] for i, key in enumerate(identities) if key is not None}
    owner = WireSegmentOwner(kind="within_group_return", group_index=0)
    path = WirePath(points, [owner] * 8, [WirePathGeometry.FREE_3D] * 8)
    candidate = GroupRouteCandidate(path, identities, protected, (0,), ("fixture",), [], 1.0, 0.0)
    expanded = []

    def direct(source, *_args, **_kwargs):
        return BridgeGeneration((SimpleNamespace(points=source.copy()),), "fixture")

    def corridor(source, *_args, **_kwargs):
        expanded.append(source.copy())
        return BridgeGeneration(
            (SimpleNamespace(points=np.array([source[0], source.mean(axis=0) + [0, 0, 0.002], source[-1]])),), "fixture"
        )

    _mock_direct_only(monkeypatch, direct)
    monkeypatch.setattr(shapes, "generate_joint_corridor_curves", corridor)
    monkeypatch.setattr(shapes, "_validate_support", lambda *_: ())
    result = shapes.shape_joint_group_route(case.part, candidate, case.config, maximum_candidates=2)
    assert result.candidates and len(expanded) == 2
    assert result.diagnostics["corridor_conflict_expansion_indices"] == [0, 1]
    assert result.diagnostics["conflicting_transition_pairs"] == []
    assert result.diagnostics["proposed_remote_gap_scores_m"][0] >= 0.001
    assert result.diagnostics["transition_evaluator"]["corridor_expansion_count"] == 2


@pytest.mark.parametrize("has_baseline_mesh", [True, False])
def test_new_family_repairs_invalid_legacy_bend_within_unchanged_field_and_body_gates(
    qualified_route, has_baseline_mesh
):
    """A certified repair may worsen E/L when the incumbent violates radius."""
    case, candidate, baseline, target, original_points = _perturbed_repair(qualified_route)
    if not has_baseline_mesh:
        realize_coil_paths([baseline], case.config)
        assert baseline.swept_body is None and not baseline.swept_bodies
    original_centroid = baseline.fabrication_path.wire_points_m.copy()
    # Reproduce the known exhaustive-family repair independently of the new
    # default's intentionally bounded proposal coverage.
    config = replace(case.config, joint_group_routing_search_mode="thorough")
    selected = improve_group_routes([baseline], config, target)[0]
    diagnostics = selected.joint_route_diagnostics
    assert selected is not baseline, diagnostics
    assert diagnostics["status"] == "accepted" and diagnostics["selection_basis"] == "baseline_repair", diagnostics
    feasibility = diagnostics["baseline_feasibility"]
    assert not feasibility["accepted"] and "minimum_bend_radius_violation" in feasibility["reasons"]
    before, after = diagnostics["baseline_routing_metrics"], diagnostics["selected_routing_metrics"]
    assert before["jump_count"] == after["jump_count"] == 2
    assert after["longitudinal_rms_T"] > before["longitudinal_rms_T"]
    assert after["length_m"] > before["length_m"]
    body = selected.manufacturing_diagnostics
    assert body["manufacturing_ok"] and body["mesh_solid_ok"]
    assert body["exact_collision_completed"] and body["exact_intersection_count"] == 0
    assert body["sweep_final_min_bend_radius_m"] >= case.config.sweep_min_bend_radius - 1e-12
    assert body["sweep_final_max_frame_rotation_deg"] <= case.config.sweep_max_frame_rotation_deg + 1e-6
    assert body["sweep_capsule_min_centerline_distance_m"] >= body["sweep_required_centerline_clearance_m"]
    accepted = next(attempt for attempt in diagnostics["attempts"] if attempt["selected"])
    assert not accepted["reasons"]
    field = accepted["validation"]
    assert field["route_maximum_pointwise_vector_delta_t_per_a"] <= field["route_maximum_delta_t_per_a"]
    assert field["physical_maximum_pointwise_vector_delta_t_per_a"] <= field["physical_maximum_delta_t_per_a"]
    actual = {(tuple(a), tuple(b)) for a, b in zip(selected.wire_path.v[:-1], selected.wire_path.v[1:])}
    assert all((tuple(a), tuple(b)) in actual for a, b in candidate.protected_segments.values())
    np.testing.assert_array_equal(baseline.wire_path.v, original_points)
    np.testing.assert_array_equal(baseline.fabrication_path.wire_points_m, original_centroid)
    if not has_baseline_mesh:
        assert baseline.swept_body is None and not baseline.swept_bodies


def test_supported_waypoint_corridor_routes_around_a_real_support_hole():
    from coilgen.geometry import normalize_mesh
    from coilgen.models import Mesh

    case = make_group_routing_case("facing_corridor")
    case.config.skip_normal_shift = True
    coordinates = [-0.04, -0.01, 0.01, 0.04]
    vertices = np.array([[x, y, 0] for y in coordinates for x in coordinates])
    faces = []
    for row in range(3):
        for column in range(3):
            if (row, column) == (1, 1):
                continue
            first = 4 * row + column
            faces.extend(((first, first + 1, first + 5), (first, first + 5, first + 4)))
    part = SimpleNamespace(mesh=normalize_mesh(Mesh(vertices=vertices, faces=np.asarray(faces))), groups=[])
    evaluator = shapes.JointTransitionEvaluator(part, case.config)
    source = np.array([[-0.03, 0, 0], [-0.025, 0.022, 0], [0.025, 0.022, 0], [0.03, 0, 0]])
    tangent = np.array([1.0, 0, 0])
    assert np.isinf(evaluator(source[[0, -1]], tangent, tangent))
    assert np.isfinite(evaluator(source, tangent, tangent)), evaluator.last_rejected_reasons
    curves = evaluator.supported_curves(source, tangent, tangent)
    assert curves and all(curve.profile_family == "support_waypoints" for curve in curves)
    assert all(np.all(curve.points[:, 2] == 0) for curve in curves)


def test_shaping_inherits_shared_evaluator_limits_and_rejects_explicit_mismatch(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    proposal = propose_joint_group_routes(case.part, case.config).candidates[0]
    limits = shapes.JointShapeLimits(maximum_shape_trials=63)
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config, limits=limits)
    monkeypatch.setattr(
        shapes,
        "generate_joint_transition_curves",
        lambda *_args, **_kwargs: pytest.fail("Admission must precede geometry"),
    )
    inherited = shapes.shape_joint_group_route(case.part, proposal, case.config, transition_evaluator=evaluator)
    assert inherited.reason == "joint_shape_trial_budget_exceeded"
    assert inherited.diagnostics["maximum_shape_trials"] == 63
    conflict = shapes.shape_joint_group_route(
        case.part,
        proposal,
        case.config,
        transition_evaluator=evaluator,
        limits=shapes.JointShapeLimits(maximum_shape_trials=64),
    )
    assert conflict.reason == "transition_evaluator_limits_mismatch"
    assert not conflict.candidates and evaluator.diagnostics["generated_transition_count"] == 0


def test_cached_supported_proposal_shapes_at_spent_generation_limit_without_new_work(monkeypatch):
    """Reusing paid-for geometry is allowed; new generation still consumes budget."""
    case = make_group_routing_case("facing_corridor")
    proposal = propose_joint_group_routes(
        case.part, case.config, limits=GroupRoutingLimits(return_fractions=(0.0,))
    ).candidates[0]
    transition_count = sum(
        identity is None and (index == 0 or proposal.segment_ids[index - 1] is not None)
        for index, identity in enumerate(proposal.segment_ids)
    )
    limits = shapes.JointShapeLimits(maximum_shape_trials=transition_count * shapes._TRANSITION_TRIALS)
    generated = []

    def generate(source, *_args, **_kwargs):
        generated.append(source.copy())
        return BridgeGeneration((SimpleNamespace(points=source.copy()),), "fixture")

    _mock_direct_only(monkeypatch, generate)
    monkeypatch.setattr(shapes, "_validate_support", lambda *_args: ())
    monkeypatch.setattr(shapes._ShapeWork, "score", lambda _self, _index, _points, clearance, excluded=(): clearance)
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config, limits=limits)
    first = shapes.shape_joint_group_route(case.part, proposal, case.config, transition_evaluator=evaluator)
    assert first.candidates, first.reason
    assert evaluator.diagnostics["shape_trial_upper_bound"] == limits.maximum_shape_trials
    before_samples = evaluator.diagnostics["curve_sample_count"]
    second = shapes.shape_joint_group_route(case.part, proposal, case.config, transition_evaluator=evaluator)
    assert second.candidates, second.reason
    assert len(generated) == transition_count
    assert evaluator.diagnostics["curve_sample_count"] == before_samples
    np.testing.assert_array_equal(first.candidates[0].path.v, second.candidates[0].path.v)
    with pytest.raises(shapes.RoutingEvaluationBudgetExceeded, match="joint_shape_trial_budget_exceeded"):
        evaluator.supported_curves(np.array([[-0.01, 0, 0], [0.01, 0, 0]]), [1, 0, 0], [1, 0, 0])
    assert len(generated) == transition_count


def _paired_detour_curve(offset, winding, extent):
    return SimpleNamespace(
        points=np.array([[offset, 0, 0], [offset + 0.5, 0, extent], [offset + 1, 0, 0]]),
        profile_family="cylinder_detour_lifted_return",
        control_points_m=np.array([[0, 0, 0], [0, -extent, 0], [winding, -extent, 0], [winding, 0, 0]]),
    )


def test_wide_counter_directed_pair_survives_partial_beam_and_three_complete_slots():
    # Fifty cheap intermediate siblings erase the far prefix from a scalar
    # beam48. The pair diversity must reserve it before the second return.
    choices = [[_paired_detour_curve(0, sign, extent) for extent in (0.01, 0.5) for sign in (-1, 1)]]
    for offset in (10, 20):
        choices.append(
            [
                SimpleNamespace(
                    points=np.array([[offset, 0, 0], [offset + 0.5, value, 0], [offset + 1, 0, 0]]),
                    profile_family="quintic",
                )
                for value in np.linspace(0, 0.001, 50)
            ]
        )
    choices.append([_paired_detour_curve(30, sign, extent) for extent in (0.01, 0.5) for sign in (1, -1)])
    jumps = [[1] * 4, [0] * 50, [0] * 50, [1] * 4]
    limits = shapes.JointShapeLimits(maximum_interaction_scores=1024, maximum_spatial_pairs=64)
    work = shapes._ShapeWork(limits)
    result = shapes._rank_curve_combinations(
        choices,
        np.empty((0, 2, 3)),
        0.001,
        0,
        3,
        jump_counts=jumps,
        return_blocks=(True, False, False, True),
        work=work,
    )
    assert len(result) == 3
    assert result[0][0] == (0, 0, 0, 0)
    assert {(state[0][0], state[0][-1]) for state in result[1:]} == {(2, 2), (3, 3)}
    assert work.interaction_scores <= limits.maximum_interaction_scores
    assert work.spatial_pairs <= limits.maximum_spatial_pairs
    # Preserving a wide pair cannot promote a higher-jump class.
    result = shapes._rank_curve_combinations(
        choices,
        np.empty((0, 2, 3)),
        0.001,
        0,
        3,
        jump_counts=[[0, 0, 1, 1], *jumps[1:3], [0, 0, 1, 1]],
        return_blocks=(True, False, False, True),
    )
    assert all(state[0][0] < 2 and state[0][-1] < 2 for state in result)


def test_wide_spiral_joins_do_not_activate_return_pair_diversity():
    choices = [
        [_paired_detour_curve(offset, sign, extent) for extent in (0.01, 0.5) for sign in (-1, 1)] for offset in (0, 10)
    ]
    result = shapes._rank_curve_combinations(
        choices, np.empty((0, 2, 3)), 0.001, 0, 3, jump_counts=[[1] * 4] * 2, return_blocks=(False, True)
    )
    assert all(state[0][0] < 2 and state[0][1] < 2 for state in result)
