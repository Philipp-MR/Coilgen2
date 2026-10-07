"""Cheap directed graph proposals precede all detailed transition geometry."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from coilgen.group_routing import GroupRoutingLimits, propose_joint_group_topologies
from coilgen.joint_group_adoption import _candidate_fingerprint
from coilgen.models import WirePathGeometry
from _group_routing_cases import make_group_routing_case


def _directions(candidate):
    return {group: name.split(":")[0] for group, name in zip(candidate.group_sequence, candidate.variant_names)}


def test_cheap_batch_keeps_opposite_winding_and_order_without_evaluating_curves(monkeypatch):
    case = make_group_routing_case("shared_corridor", group_count=4)
    original = deepcopy(case.part)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Cheap graph planning must never evaluate detailed curves.")

    monkeypatch.setattr("coilgen.group_routing._evaluate_transition", forbidden)
    result = propose_joint_group_topologies(case.part, case.config)
    assert result.reason == "proposals_require_validation"
    assert len(result.candidates) == 6
    assert result.diagnostics["cheap_topology_pool_count"] == 96
    assert result.diagnostics["detailed_transition_evaluations"] == 0
    assert result.diagnostics["candidate_cheap_rank"][0] == 1
    assert not result.diagnostics["objective_optimality_proven"]
    assert not result.diagnostics["physical_feasibility_proven"]
    first, opposite, reversed_first, reversed_opposite = result.candidates[:4]
    assert all(_directions(first)[group] != _directions(opposite)[group] for group in range(4))
    reverse = first.group_sequence[:1] + first.group_sequence[:0:-1]
    assert reversed_first.group_sequence == reversed_opposite.group_sequence == reverse
    assert _directions(reversed_first) == _directions(first)
    assert _directions(reversed_opposite) == _directions(opposite)
    for candidate in result.candidates:
        np.testing.assert_array_equal(candidate.path.v[0], candidate.path.v[-1])
        counts = Counter(identity for identity in candidate.segment_ids if identity is not None)
        assert set(counts) == set(candidate.protected_segments)
        assert set(counts.values()) == {1}
        assert sorted(candidate.group_sequence) == list(range(4))
        assert candidate.retained_return_length_m == 0
        for index, identity in enumerate(candidate.segment_ids):
            if identity is not None:
                np.testing.assert_array_equal(
                    candidate.path.v[index : index + 2], candidate.protected_segments[identity]
                )
    for before, after in zip(original.connected_groups, case.part.connected_groups):
        np.testing.assert_array_equal(before.v, after.v)
        np.testing.assert_array_equal(before.return_path_v, after.return_path_v)


def test_cheap_progress_and_batch_are_deterministic_and_respect_work_limits():
    case = make_group_routing_case("greedy_trap")
    limits = GroupRoutingLimits(candidate_count=3, maximum_expanded_states=1)
    events = []
    first = propose_joint_group_topologies(case.part, case.config, limits=limits, progress_callback=events.append)
    second = propose_joint_group_topologies(case.part, case.config, limits=limits)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["expanded_states"] <= limits.maximum_expanded_states
    assert len(first.candidates) <= 3
    assert first.diagnostics["budget_exhausted"]
    assert all(event["phase"].startswith("topology ") for event in events)
    assert events[-1]["status"] == "stopped"


def test_cheap_proposals_do_not_discard_symmetric_routes_or_claim_field_equivalence():
    case = make_group_routing_case("shared_corridor", group_count=2)
    ordinary = propose_joint_group_topologies(case.part, case.config)
    evidence = {"mesh_verified": True, "sf_verified": True, "contours_verified": True}
    symmetric = propose_joint_group_topologies(case.part, case.config, symmetry=evidence)
    assert len(ordinary.candidates) == len(symmetric.candidates) == 4
    assert symmetric.diagnostics["symmetry_evidence_provided"]
    assert symmetric.diagnostics["symmetry_routes_discarded"] == 0
    assert not symmetric.diagnostics["physical_feasibility_proven"]
    for before, after in zip(ordinary.candidates, symmetric.candidates):
        np.testing.assert_array_equal(before.path.v, after.path.v)


def test_cheap_graph_preserves_invalid_provenance_rejection():
    case = make_group_routing_case()
    case.part.groups[0].opened_loops[0].turn_multiplicity = 2
    result = propose_joint_group_topologies(case.part, case.config)
    assert result.reason == "invalid_group_provenance"
    assert not result.candidates
    assert result.diagnostics["detailed_transition_evaluations"] == 0


def test_verified_symmetry_prioritises_existing_counterpart_without_reusing_evaluations():
    case = make_group_routing_case("shared_corridor", group_count=4)
    limits = GroupRoutingLimits(candidate_count=3)
    ordinary = propose_joint_group_topologies(case.part, case.config, limits=limits)
    permutation = [0, 2, 1, 3]
    evidence = {"transforms": [{"verified": True, "group_permutation": permutation}]}
    symmetric = propose_joint_group_topologies(case.part, case.config, limits=limits, symmetry=evidence)
    expected = tuple(permutation[group] for group in ordinary.candidates[0].group_sequence)
    assert symmetric.candidates[2].group_sequence == expected
    assert symmetric.diagnostics["symmetry_priority_group_orders"] == [list(expected)]
    assert symmetric.diagnostics["symmetry_priority_candidate_count"] == 1
    assert not symmetric.diagnostics["symmetry_evaluation_reuse"]
    for before, after in zip(ordinary.candidates[:2], symmetric.candidates[:2]):
        assert before.group_sequence == after.group_sequence
        assert before.variant_names == after.variant_names
    for transform in (
        {"verified": False, "group_permutation": permutation},
        {"verified": True, "group_permutation": [0, 0, 2, 3]},
        {"verified": True, "group_permutation": [True, 0, 2, 3]},
    ):
        result = propose_joint_group_topologies(
            case.part, case.config, limits=limits, symmetry={"transforms": [transform]}
        )
        assert not result.diagnostics["symmetry_priority_group_orders"]
        for before, after in zip(ordinary.candidates, result.candidates):
            assert before.group_sequence == after.group_sequence
            assert before.variant_names == after.variant_names


def test_validation_dedup_ignores_search_annotations_and_identifies_an_unchanged_copy():
    case = make_group_routing_case()
    candidate = propose_joint_group_topologies(case.part, case.config).candidates[0]
    copied = deepcopy(candidate)
    expected = _candidate_fingerprint("existing_cuts", candidate)
    assert _candidate_fingerprint("existing_cuts", copied) == expected
    # These rank proposals but are never arguments to the physical validator.
    copied.variant_names = tuple("different_search_annotation" for _ in copied.variant_names)
    copied.length_m += 1.0
    copied.joint_return_family_counts = {"intrinsic_compact": 2}
    assert _candidate_fingerprint("existing_cuts", copied) == expected


@pytest.mark.parametrize(
    "change", ["points", "geometry", "owner", "identities", "protected", "order", "cut", "context"]
)
def test_validation_dedup_preserves_changed_physical_or_provenance_inputs(change):
    case = make_group_routing_case()
    candidate = propose_joint_group_topologies(case.part, case.config).candidates[0]
    changed = deepcopy(candidate)
    context = "existing_cuts"
    if change == "points":
        # Deduplication is exact; no geometric tolerance may conceal a change.
        changed.path.v[1, 0] = np.nextafter(changed.path.v[1, 0], np.inf)
    elif change == "geometry":
        changed.path.segment_geometry[0] = WirePathGeometry.FREE_3D
    elif change == "owner":
        changed.path.segment_owners[0].loop_index += 1
    elif change == "identities":
        changed.segment_ids = (None, *changed.segment_ids[1:])
    elif change == "protected":
        first = next(iter(changed.protected_segments.values()))
        first[0, 0] = np.nextafter(first[0, 0], np.inf)
    elif change == "order":
        changed.group_sequence = changed.group_sequence[::-1]
    elif change == "cut":
        changed.cuts[0].selected_point1_v[0, 0] += 0.001
    else:
        context = "alternate_cuts_1"
    assert _candidate_fingerprint(context, changed) != _candidate_fingerprint("existing_cuts", candidate)
