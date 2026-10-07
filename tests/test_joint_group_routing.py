"""Small topology tests precede adoption of any joint return geometry."""

from copy import deepcopy
from collections import Counter
from dataclasses import replace
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen.group_routing import GroupRoutingLimits, propose_joint_group_routes
from coilgen.routing_objective import routing_metrics
from _group_routing_cases import make_group_routing_case, directed_port_length_witness


@pytest.mark.parametrize("maximum_expanded_states", [250_000, 1])
def test_phase_progress_preserves_transition_order_and_search_results(maximum_expanded_states):
    case = make_group_routing_case("greedy_trap")
    limits = GroupRoutingLimits(return_fractions=(0.0,), maximum_expanded_states=maximum_expanded_states)
    calls = []

    def evaluate(points, before, after):
        calls.append((points.tobytes(), before.tobytes(), after.tobytes()))
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    original = propose_joint_group_routes(case.part, case.config, limits=limits, transition_evaluator=evaluate)
    original_calls = list(calls)
    calls.clear()
    events = []
    reported = propose_joint_group_routes(
        case.part, case.config, limits=limits, transition_evaluator=evaluate, progress_callback=events.append
    )
    assert calls == original_calls
    assert reported.reason == original.reason and reported.diagnostics == original.diagnostics
    assert len(reported.candidates) == len(original.candidates)
    for actual, expected in zip(reported.candidates, original.candidates):
        np.testing.assert_array_equal(actual.path.v, expected.path.v)
        assert actual.path.segment_owners == expected.path.segment_owners
        assert actual.segment_ids == expected.segment_ids
        assert actual.group_sequence == expected.group_sequence
        assert actual.variant_names == expected.variant_names
    states = [event for event in events if event["phase"] == "winding states"]
    assert states[0]["completed"] == 0 and states[0]["total"] == 6
    assert states[-2]["completed"] == states[-2]["total"] == 6
    assert states[-1]["status"] == "finished"
    ports = [event for event in events if event["phase"] == "port connections"]
    assert ports[0]["completed"] == 0 and ports[0]["total"] == 24
    assert all(event["upper_bound"] for event in ports if "completed" in event)
    assert ports[-1]["status"] == ("stopped" if original.diagnostics["budget_exhausted"] else "finished")


def test_phase_progress_preserves_partial_count_when_first_group_is_unsupported():
    case = make_group_routing_case("greedy_trap")
    events = []
    result = propose_joint_group_routes(
        case.part,
        case.config,
        limits=GroupRoutingLimits(return_fractions=(0.0,)),
        transition_evaluator=lambda *_args: np.inf,
        progress_callback=events.append,
    )
    assert result.reason == "no_supported_group_state_in_transition_family"
    assert events[0]["total"] == 6
    assert events[-2]["completed"] == 2
    assert events[-1]["status"] == "stopped"
    assert all(event["phase"] == "winding states" for event in events)


@pytest.mark.parametrize(
    "name,count", [("shared_corridor", 2), ("shared_corridor", 3), ("shared_corridor", 4), ("opposed_direction", 2)]
)
def test_proposals_keep_every_directed_contour_interval_exactly_once(name, count):
    case = make_group_routing_case(name, group_count=count)
    original = deepcopy(case.part)
    result = propose_joint_group_routes(case.part, case.config)
    assert result.reason == "proposals_require_validation"
    assert 0 < len(result.candidates) <= 6
    for candidate in result.candidates:
        assert len(candidate.path.segment_owners) == len(candidate.path.v) - 1
        assert len(candidate.path.segment_geometry) == len(candidate.segment_ids)
        np.testing.assert_array_equal(candidate.path.v[0], candidate.path.v[-1])
        assert sorted(candidate.group_sequence) == list(range(count))
        assert len(candidate.cuts) == count
        counts = Counter(identity for identity in candidate.segment_ids if identity is not None)
        assert set(counts) == set(candidate.protected_segments)
        assert set(counts.values()) == {1}
        for index, identity in enumerate(candidate.segment_ids):
            if identity is not None:
                np.testing.assert_array_equal(
                    candidate.path.v[index : index + 2], candidate.protected_segments[identity]
                )
        for index, cut in enumerate(candidate.cuts):
            assert (cut.group1_index, cut.group2_index) == (
                candidate.group_sequence[index],
                candidate.group_sequence[(index + 1) % count],
            )
    for before, after in zip(original.connected_groups, case.part.connected_groups):
        np.testing.assert_array_equal(before.v, after.v)
        np.testing.assert_array_equal(before.return_path_v, after.return_path_v)
    assert case.part.wire_path is None
    assert case.part.opening_cuts_among_groups == []


def test_bounded_search_is_deterministic_and_records_its_scope():
    case = make_group_routing_case(group_count=4)
    limits = GroupRoutingLimits(beam_width=3, candidate_count=2)
    first = propose_joint_group_routes(case.part, case.config, limits=limits)
    second = propose_joint_group_routes(case.part, case.config, limits=limits)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["candidate_count"] == 2
    assert first.diagnostics["search_method"] == "complete_small_graph"
    assert first.diagnostics["completed_states"] == 6 * 8**4
    assert first.diagnostics["port_graph_search_exhaustive"]
    assert first.diagnostics["expanded_states"] <= limits.maximum_expanded_states
    for a, b in zip(first.candidates, second.candidates):
        assert a.group_sequence == b.group_sequence
        assert a.variant_names == b.variant_names
        np.testing.assert_array_equal(a.path.v, b.path.v)


def test_invalid_or_unexpanded_provenance_never_produces_a_proposal():
    case = make_group_routing_case()
    case.part.groups[0].opened_loops[0].turn_multiplicity = 2
    assert propose_joint_group_routes(case.part, case.config).reason == "invalid_group_provenance"
    case = make_group_routing_case()
    case.part.connected_groups[0].spiral_in_v = case.part.connected_groups[0].spiral_in_v[::-1]
    case.part.connected_groups[0].spiral_out_v = case.part.connected_groups[0].spiral_out_v[::-1]
    assert propose_joint_group_routes(case.part, case.config).reason == "invalid_group_provenance"


def test_group_count_admission_rejects_before_generating_connections():
    case = make_group_routing_case(group_count=3)
    result = propose_joint_group_routes(case.part, case.config, limits=GroupRoutingLimits(maximum_groups=2))
    assert result.reason == "unsupported_group_count_or_missing_groups"
    assert not result.candidates
    assert "cached_connections" not in result.diagnostics


def test_three_group_fixture_exposes_a_local_nearest_connection_trap():
    case = make_group_routing_case("greedy_trap")
    witness = directed_port_length_witness(case)
    assert witness["best_connector_length_m"] < witness["greedy_connector_length_m"]
    assert witness["best_order"] != witness["greedy_order"]


@pytest.mark.parametrize("fractions", [(), (1.0,), (-0.1,), (np.nan,)])
def test_return_fractions_cannot_close_a_group_or_extrapolate(fractions):
    with pytest.raises(ValueError):
        GroupRoutingLimits(return_fractions=fractions)


def _assert_directed_continuity(candidate, group_count):
    points = candidate.path.v
    np.testing.assert_array_equal(points[0], points[-1])
    assert sorted(candidate.group_sequence) == list(range(group_count))
    counts = Counter(identity for identity in candidate.segment_ids if identity is not None)
    assert set(counts) == set(candidate.protected_segments)
    assert set(counts.values()) == {1}
    for index, identity in enumerate(candidate.segment_ids):
        if identity is not None:
            np.testing.assert_array_equal(points[index : index + 2], candidate.protected_segments[identity])
    for index, cut in enumerate(candidate.cuts):
        locations = [
            i
            for i, owner in enumerate(candidate.path.segment_owners)
            if owner.kind == "inter_group_connector" and owner.connector_index == index
        ]
        assert locations == list(range(locations[0], locations[-1] + 1))
        np.testing.assert_array_equal(points[locations[0] : locations[-1] + 2], cut.connection_path_v)
        np.testing.assert_array_equal(cut.selected_point1_v[0], cut.connection_path_v[0])
        np.testing.assert_array_equal(cut.selected_point2_v[0], cut.connection_path_v[-1])
        assert (cut.group1_index, cut.group2_index) == (
            candidate.group_sequence[index],
            candidate.group_sequence[(index + 1) % group_count],
        )


def _many_group_case(count):
    """Repeat directed winding geometry without relying on support certificates."""
    case = make_group_routing_case()
    template_group = deepcopy(case.part.groups[0])
    template_connected = deepcopy(case.part.connected_groups[0])
    case.part.groups = []
    case.part.connected_groups = []
    case.part.contour_lines = []
    for index in range(count):
        # Asymmetric placement avoids arbitrary ordering from geometric ties.
        shift = np.array([index * 0.039, (index % 3) * 0.031, (index % 2) * 0.021])
        group, connected = deepcopy(template_group), deepcopy(template_connected)
        for loop in (*group.loops, *group.opened_loops):
            loop.v = loop.v + shift
        group.loop_indices = [2 * index, 2 * index + 1]
        for name in ("v", "return_path_v", "spiral_in_v", "spiral_out_v"):
            setattr(connected, name, getattr(connected, name) + shift)
        case.part.groups.append(group)
        case.part.connected_groups.append(connected)
        case.part.contour_lines.extend(group.loops)
    return case


@pytest.mark.parametrize("count", [9, 16])
def test_more_than_eight_groups_form_complete_directed_proposals_with_bounded_work(count):
    case = _many_group_case(count)
    limits = GroupRoutingLimits(beam_width=8, candidate_count=2, maximum_expanded_states=10_000)
    result = propose_joint_group_routes(case.part, case.config, limits=limits)
    assert result.reason == "proposals_require_validation"
    assert 0 < len(result.candidates) <= 2
    assert result.diagnostics["expanded_states"] <= limits.maximum_expanded_states
    assert result.diagnostics["cached_connections"] <= limits.maximum_connector_evaluations
    assert result.diagnostics["effective_beam_width"] <= limits.beam_width
    assert not result.diagnostics["manufacturing_feasibility_proven"]
    assert not result.diagnostics["objective_optimality_proven"]
    for candidate in result.candidates:
        _assert_directed_continuity(candidate, count)


def test_default_limits_retain_complete_sixty_four_group_candidates_within_work_budgets():
    case = _many_group_case(64)
    result = propose_joint_group_routes(case.part, case.config)
    assert len(result.candidates) == GroupRoutingLimits().candidate_count
    assert result.diagnostics["search_method"] == "beam_orders_with_fixed_order_state_dp"
    assert result.diagnostics["budget_exhausted"]
    assert result.diagnostics["budget_exhaustion_reason"] == "expanded_state_budget"
    assert not result.diagnostics["objective_optimality_proven"]
    assert result.diagnostics["expanded_states"] <= GroupRoutingLimits().maximum_expanded_states
    assert result.diagnostics["cached_connections"] <= GroupRoutingLimits().maximum_connector_evaluations
    for candidate in result.candidates:
        _assert_directed_continuity(candidate, 64)


def _objective(candidate, direction, weight):
    return routing_metrics(
        candidate.path.v,
        direction,
        weight,
        mutable_segments=[identity is None for identity in candidate.segment_ids],
    )["cost_m"]


def test_b0_preference_improves_mutable_objective_without_charging_protected_turns():
    case = make_group_routing_case("greedy_trap")
    limits = GroupRoutingLimits(beam_width=512, candidate_count=4, return_fractions=(0.0, 0.75))
    direction = (1, 1, 0)
    shortest = propose_joint_group_routes(case.part, case.config, limits=limits, b0_direction=direction, b0_weight=0)
    aligned = propose_joint_group_routes(case.part, case.config, limits=limits, b0_direction=direction, b0_weight=5)
    assert _objective(aligned.candidates[0], direction, 5) < _objective(shortest.candidates[0], direction, 5) - 1e-4
    for candidate, cost in zip(aligned.candidates, aligned.diagnostics["candidate_objective_costs_m"]):
        assert cost == pytest.approx(_objective(candidate, direction, 5), abs=1e-12)
        assert cost < routing_metrics(candidate.path.v, direction, 5)["cost_m"]
        _assert_directed_continuity(candidate, 3)
    for candidate, cost in zip(shortest.candidates, shortest.diagnostics["candidate_objective_costs_m"]):
        assert cost == pytest.approx(candidate.length_m, abs=1e-12)


def test_zero_weight_recovers_global_length_minimum_on_small_exhaustive_port_graph():
    case = make_group_routing_case("greedy_trap")
    lengths = []
    for tail in itertools.permutations((1, 2)):
        order = (0, *tail)
        for directions in itertools.product(("spiral_in_v", "spiral_out_v"), repeat=3):
            paths = [getattr(case.part.connected_groups[g], directions[g]) for g in order]
            total = sum(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) for path in paths)
            total += sum(float(np.linalg.norm(paths[i][-1] - paths[(i + 1) % 3][0])) for i in range(3))
            lengths.append(total)
    result = propose_joint_group_routes(
        case.part,
        case.config,
        limits=GroupRoutingLimits(beam_width=512, return_fractions=(0.0,)),
        b0_weight=0,
    )
    assert result.candidates[0].length_m == pytest.approx(min(lengths), abs=1e-12)
    assert result.diagnostics["port_graph_search_exhaustive"]
    assert result.diagnostics["objective_optimality_proven"]


def test_b0_sign_and_rigid_rotation_leave_directed_search_invariant():
    angles = (-63.0, 21.0, 137.0)
    case = make_group_routing_case(group_count=3, gap_angles_degrees=angles, current_directions=(1, -1, 1))
    theta = 0.73
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    translation = np.array([0.11, -0.03, 0.05])
    rotated = make_group_routing_case(
        group_count=3,
        gap_angles_degrees=angles,
        current_directions=(1, -1, 1),
        rotation=rotation,
        translation=translation,
    )
    limits = GroupRoutingLimits(beam_width=24, candidate_count=4, return_fractions=(0.0, 0.75))
    direction = np.array([1.0, 2.0, 3.0])
    results = [
        propose_joint_group_routes(case.part, case.config, limits=limits, b0_direction=direction, b0_weight=3),
        propose_joint_group_routes(case.part, case.config, limits=limits, b0_direction=-direction, b0_weight=3),
        propose_joint_group_routes(
            rotated.part, rotated.config, limits=limits, b0_direction=rotation @ direction, b0_weight=3
        ),
    ]
    for index, result in enumerate(results[1:], 1):
        assert len(result.candidates) == len(results[0].candidates)
        np.testing.assert_allclose(
            result.diagnostics["candidate_objective_costs_m"],
            results[0].diagnostics["candidate_objective_costs_m"],
            atol=1e-12,
            rtol=0,
        )
        for original, changed in zip(results[0].candidates, result.candidates):
            assert original.group_sequence == changed.group_sequence
            assert original.variant_names == changed.variant_names
            expected = original.path.v if index == 1 else original.path.v @ rotation.T + translation
            np.testing.assert_allclose(changed.path.v, expected, atol=1e-12, rtol=0)
            _assert_directed_continuity(changed, 3)


def test_objective_defaults_come_from_config():
    case = make_group_routing_case()
    config = replace(case.config, b_0_direction=(1, 2, 0), joint_group_routing_b0_weight=7.0)
    implicit = propose_joint_group_routes(case.part, config)
    explicit = propose_joint_group_routes(case.part, config, b0_direction=(1, 2, 0), b0_weight=7.0)
    assert implicit.diagnostics == explicit.diagnostics
    for a, b in zip(implicit.candidates, explicit.candidates):
        np.testing.assert_array_equal(a.path.v, b.path.v)


@pytest.mark.parametrize(
    "limits,reason",
    [
        (GroupRoutingLimits(maximum_expanded_states=1), "expanded_state_budget"),
        (GroupRoutingLimits(maximum_connector_evaluations=1), "connector_evaluation_budget"),
    ],
)
def test_exhausted_budget_does_not_return_partial_circuit_or_claim_infeasibility(limits, reason):
    case = make_group_routing_case()
    result = propose_joint_group_routes(case.part, case.config, limits=limits)
    assert not result.candidates
    assert result.reason == "search_budget_exhausted"
    assert result.diagnostics["budget_exhaustion_reason"] == reason
    assert not result.diagnostics["port_graph_search_exhaustive"]
    assert not result.diagnostics["port_graph_infeasibility_proven"]
    assert not result.diagnostics["manufacturing_feasibility_proven"]


def test_completed_circuit_can_survive_later_budget_exhaustion_but_is_not_certified():
    case = make_group_routing_case()
    limits = GroupRoutingLimits(return_fractions=(0.0,), maximum_expanded_states=5)
    result = propose_joint_group_routes(case.part, case.config, limits=limits)
    assert result.reason == "proposals_require_validation"
    assert len(result.candidates) == 2
    assert result.diagnostics["budget_exhausted"]
    assert not result.diagnostics["objective_optimality_proven"]
    assert not result.diagnostics["manufacturing_feasibility_proven"]
    _assert_directed_continuity(result.candidates[0], 2)


def test_exhaustive_disconnected_port_graph_reports_only_graph_infeasibility(monkeypatch):
    from coilgen import layout

    monkeypatch.setattr(
        layout,
        "_inter_group_connector_result",
        lambda *args, **kwargs: SimpleNamespace(path_v=np.empty((0, 3)), used_direct_fallback=False),
    )
    case = make_group_routing_case()
    result = propose_joint_group_routes(case.part, case.config, limits=GroupRoutingLimits(return_fractions=(0.0,)))
    assert result.reason == "no_closed_connection"
    assert result.diagnostics["port_graph_search_exhaustive"]
    assert result.diagnostics["port_graph_infeasibility_proven"]
    assert not result.diagnostics["manufacturing_feasibility_proven"]


@pytest.mark.parametrize("direction,weight", [((0, 0, 0), 1), ((1, 2), 1), ((0, 0, 1), -1), ((0, 0, 1), np.inf)])
def test_invalid_objective_fails_closed(direction, weight):
    case = make_group_routing_case()
    result = propose_joint_group_routes(case.part, case.config, b0_direction=direction, b0_weight=weight)
    assert result.reason == "invalid_routing_objective"
    assert not result.candidates


@pytest.mark.parametrize(
    "name",
    ["maximum_groups", "beam_width", "candidate_count", "maximum_expanded_states", "maximum_connector_evaluations"],
)
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_search_budget_fields_require_positive_integers(name, value):
    with pytest.raises(ValueError):
        GroupRoutingLimits(**{name: value})


def test_duplicate_loop_identity_fails_closed_without_overwriting_protected_geometry():
    case = make_group_routing_case()
    case.part.groups[0].loop_indices = [0, 0]
    result = propose_joint_group_routes(case.part, case.config)
    assert result.reason == "invalid_group_provenance"
    assert "Duplicate protected contour identity" in result.diagnostics["detail"]


def _single_group_case(loop_count=3, current_direction=1):
    from coilgen.layout import interconnect_within_groups
    from coilgen.models import ContourLine, LoopGroup

    case = make_group_routing_case()
    angles = current_direction * np.linspace(0, 2 * np.pi, 181)
    loops = []
    for index, radius in enumerate(np.linspace(0.03, 0.01, loop_count)):
        points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.zeros(len(angles))))
        points[-1] = points[0]
        loops.append(
            ContourLine(
                points, potential=current_direction * float(index + 1), current_orientation=float(current_direction)
            )
        )
    case.part.groups = [LoopGroup(loops=loops, loop_indices=list(range(loop_count)), outer_loop_indices=(0,))]
    case.part.contour_lines = loops
    case.part.loop_groups = [list(range(loop_count))]
    case.part.group_levels = [[0]]
    case.part.level_positions = [[]]
    case.part.group_centers_v = np.zeros((1, 3))
    case.part.connected_groups = []
    interconnect_within_groups([case.part], case.config)
    return case


@pytest.mark.parametrize("loop_count,current_direction", [(3, 1), (3, -1), (4, 1), (4, -1)])
def test_one_multiloop_group_closes_its_own_return_without_reversing_contours(loop_count, current_direction):
    case = _single_group_case(loop_count, current_direction)
    source = deepcopy(case.part)
    result = propose_joint_group_routes(case.part, case.config)
    assert result.reason == "proposals_require_validation"
    assert result.candidates
    assert result.diagnostics["group_count"] == 1
    assert result.diagnostics["port_graph_search_exhaustive"]
    assert not result.diagnostics["manufacturing_feasibility_proven"]
    for candidate in result.candidates:
        _assert_directed_continuity(candidate, 1)
        assert len(candidate.cuts) == 1
        assert candidate.cuts[0].group1_index == candidate.cuts[0].group2_index == 0
        assert candidate.cuts[0].is_return_connection
    for original, after in zip(source.groups[0].opened_loops, case.part.groups[0].opened_loops):
        np.testing.assert_array_equal(after.v, original.v)
        assert after.current_orientation == current_direction
        assert after.potential == original.potential


def test_one_three_loop_group_can_shape_its_complete_return_with_exact_protected_coverage():
    from coilgen.joint_route_shapes import shape_joint_group_route
    from coilgen.group_route_validation import _structural_reasons
    from coilgen.models import WirePathGeometry

    case = _single_group_case()
    limits = GroupRoutingLimits(return_fractions=(0.75,), candidate_count=1)
    proposal = propose_joint_group_routes(case.part, case.config, limits=limits).candidates[0]
    shapes = shape_joint_group_route(case.part, proposal, case.config, maximum_candidates=1)
    assert shapes.reason == "shaped_proposals_require_validation"
    assert len(shapes.candidates) == 1
    candidate = shapes.candidates[0]
    assert shapes.diagnostics["transition_count"] == 3  # Two spiral joins and the complete return.
    assert not shapes.diagnostics["clearance_score_is_finite_body_certificate"]
    _assert_directed_continuity(candidate, 1)
    assert (
        _structural_reasons(
            candidate.path,
            candidate.segment_ids,
            candidate.protected_segments,
            candidate.group_sequence,
            (0,),
            candidate.cuts,
        )
        == ()
    )
    assert all(
        geometry is WirePathGeometry.FREE_3D
        for geometry, identity in zip(candidate.path.segment_geometry, candidate.segment_ids)
        if identity is None
    )
    assert np.max(candidate.path.v[:, 2]) > 0
    assert np.max(candidate.path.v[:, 2]) <= abs(case.config.normal_shift_length) + 1e-12


def test_single_closed_loop_without_return_ports_does_not_invent_cuts():
    case = _single_group_case(loop_count=1)
    before = deepcopy(case.part.groups[0])
    result = propose_joint_group_routes(case.part, case.config)
    assert not result.candidates
    assert result.reason == "single_closed_loop_has_no_free_ports"
    np.testing.assert_array_equal(case.part.groups[0].loops[0].v, before.loops[0].v)
    assert case.part.groups[0].opening_cuts_v == []
    assert "cached_connections" not in result.diagnostics


def test_zero_groups_are_rejected_before_route_search():
    case = make_group_routing_case()
    case.part.groups = []
    case.part.connected_groups = []
    result = propose_joint_group_routes(case.part, case.config)
    assert result.reason == "unsupported_group_count_or_missing_groups"
    assert not result.candidates
    assert "cached_connections" not in result.diagnostics


def test_shaped_transition_search_evaluates_internal_joins_and_complete_ports_with_exact_tangents():
    case = make_group_routing_case(group_count=4)
    incoming, outgoing = {}, {}
    for group in case.part.groups:
        for loop in group.opened_loops:
            for a, b in zip(loop.v[:-1], loop.v[1:]):
                incoming[tuple(b)] = b - a
                outgoing[tuple(a)] = b - a
    calls = []

    def evaluate(points, before, after):
        np.testing.assert_array_equal(before, incoming[tuple(points[0])])
        np.testing.assert_array_equal(after, outgoing[tuple(points[-1])])
        calls.append((points.copy(), before.copy(), after.copy()))
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    result = propose_joint_group_routes(
        case.part,
        case.config,
        limits=GroupRoutingLimits(beam_width=1, return_fractions=(0.0,)),
        transition_evaluator=evaluate,
    )
    assert result.reason == "proposals_require_validation"
    assert result.diagnostics["completed_states"] == 96
    assert result.diagnostics["expanded_states"] == 252
    assert result.diagnostics["cached_connections"] == 48
    assert len(calls) == 8 + 48  # Two internal joins per group plus each directed state edge once.
    assert result.diagnostics["transition_cost_basis"] == "shaped_whole_transition"
    assert result.diagnostics["port_graph_search_exhaustive"]
    for candidate, cost in zip(result.candidates, result.diagnostics["candidate_objective_costs_m"]):
        assert candidate.retained_return_length_m == 0
        assert cost == pytest.approx(candidate.length_m, abs=1e-12)
        _assert_directed_continuity(candidate, 4)


def test_whole_shaped_transition_cost_can_reverse_provisional_spiral_ranking():
    case = make_group_routing_case("greedy_trap")
    limits = GroupRoutingLimits(return_fractions=(0.0,), beam_width=1, candidate_count=1)
    provisional = propose_joint_group_routes(case.part, case.config, limits=limits, b0_weight=0).candidates[0]
    preferred = {
        group: "spiral_out" if name.startswith("spiral_in") else "spiral_in"
        for group, name in zip(provisional.group_sequence, provisional.variant_names)
    }
    exits, entries = {}, {}
    for group, connected in enumerate(case.part.connected_groups):
        for name in ("spiral_in", "spiral_out"):
            points = getattr(connected, name + "_v")
            exits[tuple(points[-1])] = group, name
            entries[tuple(points[0])] = group, name

    def evaluate(points, _before, _after):
        first, last = exits[tuple(points[0])], entries[tuple(points[-1])]
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if first[0] == last[0]:
            return length
        # A declared shaped corridor cost makes a previously short port choice
        # expensive. The search must use this geometry cost, not the old chord.
        return length + float(first[1] != preferred[first[0]]) + float(last[1] != preferred[last[0]])

    result = propose_joint_group_routes(case.part, case.config, limits=limits, transition_evaluator=evaluate)
    candidate = result.candidates[0]
    assert all(
        name.startswith(preferred[group]) for group, name in zip(candidate.group_sequence, candidate.variant_names)
    )
    assert (
        candidate.variant_names != provisional.variant_names or candidate.group_sequence != provisional.group_sequence
    )
    assert result.diagnostics["objective_optimality_proven"]
    _assert_directed_continuity(candidate, 3)


def test_jump_search_cost_is_preserved_and_precedes_the_geometric_guide():
    from coilgen.routing_objective import RoutingSearchCost

    case = make_group_routing_case("greedy_trap")
    limits = GroupRoutingLimits(return_fractions=(0.0,), candidate_count=6)
    preferred = {0: "spiral_out", 1: "spiral_in", 2: "spiral_out"}
    exits, entries = {}, {}
    for group, connected in enumerate(case.part.connected_groups):
        for name in ("spiral_in", "spiral_out"):
            points = getattr(connected, name + "_v")
            exits[tuple(points[-1])] = group, name
            entries[tuple(points[0])] = group, name

    class Evaluator:
        def __call__(self, *_args):
            pytest.fail("The scalar guide must not discard the jump count.")

        def search_cost(self, points, _before, _after):
            first, last = exits[tuple(points[0])], entries[tuple(points[-1])]
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if first[0] == last[0]:
                return RoutingSearchCost(0, length)
            jumps = int(first[1] != preferred[first[0]]) + int(last[1] != preferred[last[0]])
            # Deliberately expensive zero-jump transitions must still win.
            return RoutingSearchCost(jumps, length + (100.0 if jumps == 0 else 0.0))

    result = propose_joint_group_routes(case.part, case.config, limits=limits, transition_evaluator=Evaluator())
    candidate = result.candidates[0]
    assert result.diagnostics["objective"] == "jump_count_then_geometric_guide"
    assert result.diagnostics["candidate_jump_counts"][0] == 0
    assert result.diagnostics["candidate_objective_costs_m"][0] > 300.0
    assert any(count > 0 for count in result.diagnostics["candidate_jump_counts"])
    assert all(
        name.startswith(preferred[group]) for group, name in zip(candidate.group_sequence, candidate.variant_names)
    )
    _assert_directed_continuity(candidate, 3)


@pytest.mark.parametrize("value", [-1.0, np.nan, -np.inf, True, [1.0], None])
def test_invalid_transition_cost_rejects_without_mutating_part(value):
    case = make_group_routing_case()
    original = case.part.connected_groups[0].spiral_in_v.copy()
    result = propose_joint_group_routes(
        case.part,
        case.config,
        limits=GroupRoutingLimits(return_fractions=(0.0,)),
        transition_evaluator=lambda *_args: value,
    )
    assert not result.candidates
    assert result.reason == "invalid_group_provenance"
    assert "Transition evaluator" in result.diagnostics["detail"]
    np.testing.assert_array_equal(case.part.connected_groups[0].spiral_in_v, original)


def test_unsupported_internal_join_reports_only_the_declared_transition_family():
    case = make_group_routing_case()
    result = propose_joint_group_routes(
        case.part,
        case.config,
        limits=GroupRoutingLimits(return_fractions=(0.0,)),
        transition_evaluator=lambda *_args: np.inf,
    )
    assert not result.candidates
    assert result.reason == "no_supported_group_state_in_transition_family"
    assert not result.diagnostics["physical_feasibility_proven"]
    assert not result.diagnostics["objective_optimality_proven"]


def test_whole_transition_evaluator_rejects_retained_return_prefixes():
    case = make_group_routing_case()
    result = propose_joint_group_routes(case.part, case.config, transition_evaluator=lambda *_args: 0.0)
    assert result.reason == "transition_evaluator_requires_zero_return_fraction"
    assert not result.candidates


def test_transition_generation_budget_is_explicit_even_while_building_group_states():
    from coilgen.group_routing import RoutingEvaluationBudgetExceeded

    def exhausted(*_args):
        raise RoutingEvaluationBudgetExceeded("transition_curve_budget")

    case = make_group_routing_case()
    result = propose_joint_group_routes(
        case.part, case.config, limits=GroupRoutingLimits(return_fractions=(0.0,)), transition_evaluator=exhausted
    )
    assert not result.candidates
    assert result.reason == "search_budget_exhausted"
    assert result.diagnostics["budget_exhausted"]
    assert result.diagnostics["budget_exhaustion_reason"] == "transition_curve_budget"
    assert not result.diagnostics["objective_optimality_proven"]


def test_unsupported_internal_join_does_not_spend_budget_on_later_joins_of_same_state():
    from coilgen.group_routing import _mutable_cost

    calls = []

    def evaluate(*_args):
        calls.append(True)
        if len(calls) > 1:
            pytest.fail("Spent transition budget after this group state was already unsupported")
        return np.inf

    points = np.column_stack((np.arange(6, dtype=float), np.zeros((6, 2))))
    identities = ((0, 0, 0), None, (0, 1, 0), None, (0, 2, 0))
    assert _mutable_cost(points, identities, (0, 0, 1), 1.0, evaluate) == np.inf
    assert calls == [True]


def test_introduced_spiral_joins_have_persistent_explicit_provenance():
    case = make_group_routing_case(group_count=3)
    result = propose_joint_group_routes(
        case.part, case.config, limits=GroupRoutingLimits(return_fractions=(0.0, 0.75), candidate_count=6)
    )
    assert result.candidates
    for candidate in result.candidates:
        joins = 0
        for identity, owner in zip(candidate.segment_ids, candidate.path.segment_owners):
            if identity is not None:
                assert owner.kind == "contour_loop"
                assert owner.connector_role == ""
            elif owner.kind == "contour_group":
                assert owner.connector_role == "joint_spiral_join"
                joins += 1
            else:
                assert owner.connector_role != "joint_spiral_join"
        assert joins == 3  # Each two-contour group has one introduced internal join.
    assert not case.part.joint_route_diagnostics  # Provenance is independent of adoption status.
