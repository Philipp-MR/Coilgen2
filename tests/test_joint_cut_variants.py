"""Alternative openings preserve source currents and isolate the baseline."""

from copy import deepcopy
from dataclasses import is_dataclass, replace
import pickle

import numpy as np
import pytest

from _group_routing_cases import make_group_routing_case
from coilgen import joint_cut_variants as cuts
from coilgen.group_routing import propose_joint_group_routes
from coilgen.layout import interconnect_within_groups
from coilgen.models import FabricationPath, SweptBody, WirePath


def _assert_same_values(first, second):
    """Exact state content; candidate arrays intentionally become read-only."""
    if isinstance(first, np.ndarray):
        assert first.dtype == second.dtype and first.shape == second.shape
        assert first.tobytes() == second.tobytes()
    elif is_dataclass(first):
        _assert_same_values(vars(first), vars(second))
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            _assert_same_values(first[name], second[name])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second):
            _assert_same_values(a, b)
    else:
        assert first == second


def _case(*, force=(), known_sides=True):
    case = make_group_routing_case("opposed_direction")
    case.config = replace(case.config, force_cut_selection=force)
    if known_sides:
        interconnect_within_groups([case.part], case.config)
    # State sentinels, not a claim that this illustrative baseline is a solid.
    points = np.array([[0.0, 0, 0], [0.05, 0, 0], [0.05, 0.05, 0], [0, 0.05, 0], [0, 0, 0]])
    case.part.wire_path = WirePath(points)
    tangents = np.array([[1.0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    normal = np.tile([0.0, 0, 1], (4, 1))
    case.part.fabrication_path = FabricationPath(
        points_m=points[:-1] + [0, 0, 0.0002],
        tangents=tangents,
        normals=normal,
        transverse=np.cross(tangents, normal),
        cross_section_m=np.array([[-0.0003, 0], [0.0003, 0], [0, 0.0004]]),
        source_points_m=points,
        closed=True,
    )
    case.part.swept_body = deepcopy(case.part.mesh)
    case.part.swept_bodies = [
        SweptBody(
            deepcopy(case.part.swept_body),
            diagnostics={"sentinel": "body"},
            fabrication_path=case.part.fabrication_path,
        )
    ]
    case.part.manufacturing_diagnostics = {"sentinel": "original"}
    case.part.normal_shift_provenance = {"points_before_m": points.copy(), "schema_version": 2}
    case.part.normal_shift_diagnostics = {"sentinel": "lift"}
    case.part.field_by_layout = np.ones((2, 3))
    return case


def _mixed_group_case(sizes):
    from coilgen.layout import interconnect_among_groups

    case = make_group_routing_case("opposed_direction", group_count=len(sizes))
    case.part.contour_lines = []
    case.part.loop_groups = []
    for group, size in zip(case.part.groups, sizes):
        group.loops = group.loops[:size]
        start = len(case.part.contour_lines)
        group.loop_indices = list(range(start, start + size))
        group.outer_loop_indices = (start,)
        case.part.contour_lines.extend(group.loops)
        case.part.loop_groups.append(group.loop_indices)
    interconnect_within_groups([case.part], case.config)
    interconnect_among_groups([case.part], case.config)
    return case


def _assert_baseline_values(context, original):
    _assert_same_values(context.mesh.vertices, original.mesh.vertices)
    _assert_same_values(context.mesh.faces, original.mesh.faces)
    for name in (
        "wire_path",
        "fabrication_path",
        "swept_body",
        "swept_bodies",
        "manufacturing_diagnostics",
        "normal_shift_provenance",
        "normal_shift_diagnostics",
        "shift_array",
        "points_to_shift",
        "field_by_layout",
    ):
        _assert_same_values(getattr(context, name), getattr(original, name))


@pytest.mark.parametrize("sizes", [(2, 1), (1, 1), (2, 1, 1), (1, 1, 1)])
def test_singletons_get_explicit_isolated_cuts_and_join_one_directed_cycle(sizes):
    from coilgen.group_route_validation import _structural_reasons

    case = _mixed_group_case(sizes)
    before = pickle.dumps(case.part, protocol=5)
    assert propose_joint_group_routes(case.part, case.config).reason == "singleton_cut_context_required"
    context = cuts.prepare_joint_group_context(case.part, case.config)
    assert context is not case.part
    assert cuts.prepare_joint_group_context(context, case.config) is context
    _assert_baseline_values(context, case.part)
    assert cuts._full_contour_signature(context) == cuts._full_contour_signature(case.part)
    for index, (original, group) in enumerate(zip(case.part.groups, context.groups)):
        if sizes[index] > 1:
            _assert_same_values(group, original)
            continue
        assert len(group.opening_cuts_v) == 1
        assert group.within_group_cut_side in {"high", "low"}
        assert cuts._opened_contour_follows_source(group.loops[0], group.opened_loops[0])
        assert not np.array_equal(group.opened_loops[0].v[0], group.opened_loops[0].v[-1])
    result = propose_joint_group_routes(context, case.config)
    assert result.candidates
    for candidate in result.candidates:
        assert (
            _structural_reasons(
                candidate.path,
                candidate.segment_ids,
                candidate.protected_segments,
                candidate.group_sequence,
                tuple(range(len(sizes))),
                candidate.cuts,
            )
            == ()
        )
    assert pickle.dumps(case.part, protocol=5) == before


@pytest.mark.parametrize("sizes", [(2, 1), (1, 1), (2, 1, 1)])
def test_alternate_singleton_cuts_are_directed_bounded_and_deterministic(sizes):
    case = _mixed_group_case(sizes)
    before = pickle.dumps(case.part, protocol=5)
    result = cuts.propose_joint_cut_variants(case.part, case.config)
    repeated = cuts.propose_joint_cut_variants(case.part, case.config)
    assert 0 < len(result.variants) <= 2
    assert result.diagnostics == repeated.diagnostics
    for context, again in zip(result.variants, repeated.variants):
        assert cuts._cut_signature(context) == cuts._cut_signature(again)
        _assert_baseline_values(context, case.part)
        assert cuts._full_contour_signature(context) == cuts._full_contour_signature(case.part)
        assert propose_joint_group_routes(context, case.config).candidates
        for group in context.groups:
            for source, opened in zip(group.loops, group.opened_loops):
                assert cuts._opened_contour_follows_source(source, opened)
    assert pickle.dumps(case.part, protocol=5) == before


@pytest.mark.parametrize("side", ["high", "low"])
def test_singleton_forced_side_is_used_then_preserved_by_alternative_search(side):
    case = _mixed_group_case((2, 1))
    case.config = replace(case.config, force_cut_selection=("auto", side))
    context = cuts.prepare_joint_group_context(case.part, case.config)
    assert context.groups[1].within_group_cut_side == side
    alternatives = cuts.propose_joint_cut_variants(context, case.config)
    assert alternatives.variants
    for alternative in alternatives.variants:
        _assert_same_values(alternative.groups[1], context.groups[1])
        _assert_same_values(alternative.connected_groups[1], context.connected_groups[1])


def test_singleton_preparation_rejects_reversed_arc_without_changing_baseline(monkeypatch):
    case = _mixed_group_case((2, 1))
    before = pickle.dumps(case.part, protocol=5)
    production = cuts.interconnect_within_groups

    def reverse_singleton(parts, config, **options):
        production(parts, config, **options)
        loop = parts[0].groups[1].opened_loops[0]
        loop.v = loop.v[::-1].copy()

    monkeypatch.setattr(cuts, "interconnect_within_groups", reverse_singleton)
    with pytest.raises(ValueError, match="opened_contour_direction_or_geometry_changed"):
        cuts.prepare_joint_group_context(case.part, case.config)
    assert pickle.dumps(case.part, protocol=5) == before


def test_real_alternate_rails_are_bounded_deterministic_and_preserve_source_and_baseline():
    case = _case()
    before = pickle.dumps(case.part, protocol=5)
    config_before = pickle.dumps(case.config, protocol=5)
    result = cuts.propose_joint_cut_variants(case.part, case.config)
    again = cuts.propose_joint_cut_variants(case.part, case.config)
    assert result.reasons == ()
    assert len(result.variants) == 2
    assert result.diagnostics == again.diagnostics
    assert not result.diagnostics["global_search_exhaustive"]
    assert not result.diagnostics["manufacturing_feasibility_proven"]
    assert pickle.dumps(case.part, protocol=5) == before
    assert pickle.dumps(case.config, protocol=5) == config_before
    baseline_sides = [group.within_group_cut_side for group in case.part.groups]
    for original_side, new_group in zip(baseline_sides, result.variants[0].groups):
        assert new_group.within_group_cut_side == {"high": "low", "low": "high"}[original_side]
    for context, repeated in zip(result.variants, again.variants):
        assert cuts._full_contour_signature(context) == cuts._full_contour_signature(case.part)
        assert cuts._cut_signature(context) == cuts._cut_signature(repeated)
        for name in (
            "wire_path",
            "fabrication_path",
            "swept_body",
            "swept_bodies",
            "manufacturing_diagnostics",
            "normal_shift_provenance",
            "normal_shift_diagnostics",
            "field_by_layout",
        ):
            _assert_same_values(getattr(context, name), getattr(case.part, name))
        proposal = propose_joint_group_routes(context, case.config)
        assert proposal.candidates
        for group in context.groups:
            for source, opened in zip(group.loops, group.opened_loops):
                assert cuts._opened_contour_follows_source(source, opened)


def test_contexts_share_large_baseline_buffers_readonly_but_isolate_objects_and_metadata():
    case = _case()
    context, sibling = cuts.propose_joint_cut_variants(case.part, case.config).variants
    for original, candidate in (
        (case.part.mesh.vertices, context.mesh.vertices),
        (case.part.wire_path.v, context.wire_path.v),
        (case.part.fabrication_path.points_m, context.fabrication_path.points_m),
        (case.part.swept_body.vertices, context.swept_body.vertices),
        (case.part.swept_bodies[0].mesh.vertices, context.swept_bodies[0].mesh.vertices),
        (case.part.normal_shift_provenance["points_before_m"], context.normal_shift_provenance["points_before_m"]),
    ):
        assert np.shares_memory(original, candidate)
        assert not candidate.flags.writeable
        with pytest.raises(ValueError):
            candidate[0, 0] += 1.0
    context.mesh.metadata["only_this_context"] = 1
    context.manufacturing_diagnostics["sentinel"] = "modified"
    context.swept_bodies[0].diagnostics["sentinel"] = "modified"
    context.groups[0].opened_loops[0].v = context.groups[0].opened_loops[0].v + 1
    assert "only_this_context" not in case.part.mesh.metadata
    assert "only_this_context" not in sibling.mesh.metadata
    assert case.part.manufacturing_diagnostics == sibling.manufacturing_diagnostics == {"sentinel": "original"}
    assert case.part.swept_bodies[0].diagnostics == sibling.swept_bodies[0].diagnostics == {"sentinel": "body"}
    assert not np.array_equal(context.groups[0].opened_loops[0].v, sibling.groups[0].opened_loops[0].v)


@pytest.mark.parametrize("forced_index,side", [(0, "low"), (1, "high")])
def test_per_group_forced_side_and_its_exact_cut_geometry_never_change(forced_index, side):
    selections = ["auto", "auto"]
    selections[forced_index] = side
    case = _case(force=tuple(selections))
    forced = deepcopy((case.part.groups[forced_index], case.part.connected_groups[forced_index]))
    result = cuts.propose_joint_cut_variants(case.part, case.config)
    assert len(result.variants) == 1
    assert result.diagnostics["forced_group_indices"] == [forced_index]
    assert result.diagnostics["variants"][0]["changed_group_indices"] == [1 - forced_index]
    context = result.variants[0]
    assert context.groups[forced_index].within_group_cut_side == side
    _assert_same_values((context.groups[forced_index], context.connected_groups[forced_index]), forced)


@pytest.mark.parametrize("force", [("high",), ("LOW",), ("high", "low")])
def test_all_forced_sides_skip_generation(force, monkeypatch):
    case = _case(force=force)
    monkeypatch.setattr(
        cuts, "interconnect_within_groups", lambda *args, **kwargs: pytest.fail("unexpected reconstruction")
    )
    result = cuts.propose_joint_cut_variants(case.part, case.config)
    assert not result.variants
    assert result.reasons == ("no_unforced_multi_loop_groups",)


def test_second_plan_targets_worst_free_group_and_restores_untouched_geometry_exactly():
    case = _case()
    case.part.groups[0].within_group_cut_score = 1.0
    case.part.groups[1].within_group_cut_score = 9.0
    result = cuts.propose_joint_cut_variants(case.part, case.config)
    assert result.diagnostics["variants"][1]["changed_group_indices"] == [1]
    context = result.variants[1]
    _assert_same_values(context.groups[0], case.part.groups[0])
    _assert_same_values(context.connected_groups[0], case.part.connected_groups[0])


def test_unknown_side_uses_isolated_production_inference_without_mutating_original():
    case = _case(known_sides=False)
    before = pickle.dumps(case.part, protocol=5)
    result = cuts.propose_joint_cut_variants(case.part, case.config)
    assert len(result.variants) == 2
    assert result.diagnostics["canonical_side_inference"]
    assert pickle.dumps(case.part, protocol=5) == before


@pytest.mark.parametrize(
    "change,reason",
    [
        ("reverse_opened", "opened_contour_direction_or_geometry_changed"),
        ("skip_corner", "opened_contour_direction_or_geometry_changed"),
        ("change_current", "original_full_contours_changed"),
        ("move_source", "original_full_contours_changed"),
        ("change_baseline", "baseline_geometry_or_state_changed"),
        ("change_membership", "source_group_membership_changed"),
        ("fallback", "requested_cut_side_unavailable"),
    ],
)
def test_reconstruction_invariant_violation_is_rejected_without_live_mutation(change, reason, monkeypatch):
    case = _case()
    before = pickle.dumps(case.part, protocol=5)
    production = cuts.interconnect_within_groups

    def corrupt(parts, config):
        production(parts, config)
        for group in parts[0].groups:
            if change == "reverse_opened":
                group.opened_loops[0].v = group.opened_loops[0].v[::-1]
            elif change == "skip_corner":
                group.opened_loops[0].v = np.delete(group.opened_loops[0].v, 10, axis=0)
            elif change == "change_current":
                group.loops[0].current_orientation *= -1
            elif change == "move_source":
                group.loops[0].v = group.loops[0].v + 1e-6
            elif change == "change_membership":
                group.loop_indices = list(reversed(group.loop_indices))
            elif change == "fallback":
                group.within_group_cut_side = "fallback"
        if change == "change_baseline":
            parts[0].wire_path.v = parts[0].wire_path.v + 1e-6

    monkeypatch.setattr(cuts, "interconnect_within_groups", corrupt)
    result = cuts.propose_joint_cut_variants(case.part, case.config)
    assert not result.variants
    assert reason in result.reasons
    assert pickle.dumps(case.part, protocol=5) == before


def test_reconstruction_error_is_explicit_and_does_not_discard_other_valid_plan(monkeypatch):
    case = _case()
    production = cuts.interconnect_within_groups
    calls = []

    def fail_first(parts, config):
        calls.append(tuple(config.force_cut_selection))
        if len(calls) == 1:
            raise ValueError("fixture unavailable rail")
        return production(parts, config)

    monkeypatch.setattr(cuts, "interconnect_within_groups", fail_first)
    result = cuts.propose_joint_cut_variants(case.part, case.config)
    assert len(result.variants) == 1
    assert result.reasons == ()
    assert result.diagnostics["attempts"][0]["reasons"] == ["within_group_reconstruction_failed"]
    assert "fixture unavailable rail" in result.diagnostics["attempts"][0]["detail"]


def test_identical_geometry_with_changed_side_label_is_not_a_new_context(monkeypatch):
    case = _case()

    def relabel(parts, config):
        for index, group in enumerate(parts[0].groups):
            group.within_group_cut_side = config.force_cut_selection[index]

    monkeypatch.setattr(cuts, "interconnect_within_groups", relabel)
    result = cuts.propose_joint_cut_variants(case.part, case.config)
    assert not result.variants
    assert result.reasons == ("duplicate_cut_geometry",)


@pytest.mark.parametrize("maximum,expected", [(0, 0), (1, 1), (2, 2), (10, 2)])
def test_cut_candidate_budget_is_bounded(maximum, expected):
    case = _case()
    case.config = replace(case.config, joint_group_routing_selection_policy="physical")
    result = cuts.propose_joint_cut_variants(case.part, case.config, maximum_candidates=maximum)
    assert len(result.variants) == expected
    assert len(result.diagnostics["attempts"]) <= min(maximum, 2)


@pytest.mark.parametrize("value", [-1, 0.5, True, np.inf])
def test_invalid_candidate_budget_is_rejected(value):
    case = _case()
    with pytest.raises(ValueError):
        cuts.propose_joint_cut_variants(case.part, case.config, maximum_candidates=value)


def test_disabled_search_and_weighted_contours_fail_before_reconstruction(monkeypatch):
    case = _case()
    monkeypatch.setattr(
        cuts, "interconnect_within_groups", lambda *args, **kwargs: pytest.fail("unexpected reconstruction")
    )
    disabled = cuts.propose_joint_cut_variants(case.part, replace(case.config, joint_group_routing_search_cuts=False))
    assert disabled.reasons == ("cut_search_disabled",)
    case.part.groups[0].loops[0].turn_multiplicity = 2
    weighted = cuts.propose_joint_cut_variants(case.part, case.config)
    assert weighted.reasons == ("invalid_or_unexpanded_full_contours",)


def test_no_shift_or_sweep_is_called_for_cut_proposals(monkeypatch):
    from coilgen import layout, manufacturing

    case = _case()
    monkeypatch.setattr(layout, "shift_return_paths", lambda *args, **kwargs: pytest.fail("unexpected shift"))
    monkeypatch.setattr(
        manufacturing, "create_swept_coil_meshes", lambda *args, **kwargs: pytest.fail("unexpected sweep")
    )
    assert cuts.propose_joint_cut_variants(case.part, case.config).variants
