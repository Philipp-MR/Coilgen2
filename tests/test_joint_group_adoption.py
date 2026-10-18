"""Transaction and control-flow checks; manufacturing arithmetic is tested separately."""

from copy import deepcopy
import pickle
import re
from types import SimpleNamespace

import numpy as np
import pytest

from coilgen import joint_group_adoption as adoption, routing_progress
from coilgen.config import CoilGenConfig
from coilgen.config_validation import validate_config
from coilgen.group_route_validation import GroupRouteValidation
from coilgen.group_routing import GroupRouteCandidate, GroupRouteProposals
from coilgen.models import (
    CoilPart,
    ConnectedGroup,
    ContourLine,
    LoopGroup,
    Mesh,
    TargetField,
    WirePath,
    WireSegmentOwner,
)
from coilgen.progress import progress_context


@pytest.fixture(autouse=True)
def isolate_jump_geometry(monkeypatch):
    # This suite tests transactions/selection. Actual surface detachment,
    # mounting and cyclic geometry are exercised in test_routing_jumps.py.
    monkeypatch.setattr(
        adoption,
        "fabricated_jump_metrics",
        lambda part, _config: {
            "jump_count": getattr(part.fabrication_path, "test_jump_count", 0),
        },
    )


def make_part(index=0, *, groups=2):
    points = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]) + [3 * index, 0, 0]
    mesh = Mesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) + [3 * index, 0, 0],
        np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [0, 2, 3]]),
    )
    loops = [ContourLine(points.copy(), potential=float(i + 1)) for i in range(groups)]
    part = CoilPart(
        mesh=mesh,
        wire_path=WirePath(points.copy(), segment_owners=[WireSegmentOwner("contour_loop")] * 4),
        contour_lines=loops,
        groups=[LoopGroup([loop], opened_loops=[deepcopy(loop)]) for loop in loops],
        connected_groups=[
            ConnectedGroup(points.copy(), points[:2].copy(), points.copy(), points.copy()) for _ in loops
        ],
        contour_step=2.0,
        swept_body=deepcopy(mesh),
        manufacturing_diagnostics={"manufacturing_ok": True},
    )
    part.fabrication_path = SimpleNamespace(
        wire_points_m=points + [0, 0, 0.01], segment_owners=deepcopy(part.wire_path.segment_owners)
    )
    part.pcb_layout = {"old": "layout"}
    return part


def candidate_for(part):
    points = part.wire_path.v
    points = points[0] + 0.8 * (points - points[0])
    return GroupRouteCandidate(
        path=WirePath(points, segment_owners=deepcopy(part.wire_path.segment_owners)),
        segment_ids=(None,) * 4,
        protected_segments={},
        group_sequence=tuple(range(len(part.groups))),
        variant_names=("fixture",) * len(part.groups),
        cuts=[],
        length_m=adoption._length(points),
        retained_return_length_m=0.0,
    )


def test_cut_context_fingerprint_keeps_distinct_return_seed_with_equal_ports():
    original = make_part()
    identical = deepcopy(original)
    alternate = deepcopy(original)
    alternate.connected_groups[0].return_path_v[1] += [0, 0, 0.01]
    np.testing.assert_array_equal(original.groups[0].opened_loops[0].v, alternate.groups[0].opened_loops[0].v)
    assert adoption._cut_context_fingerprint(original) == adoption._cut_context_fingerprint(identical)
    assert adoption._cut_context_fingerprint(original) != adoption._cut_context_fingerprint(alternate)


def prepared_for(part, path):
    result = deepcopy(part)
    result.wire_path = deepcopy(path)
    result.fabrication_path = SimpleNamespace(
        wire_points_m=path.v + [0, 0, 0.02], segment_owners=deepcopy(path.segment_owners)
    )
    return result


def distinct_shaped_proposals(monkeypatch):
    """Different geometry, so a physical-check budget is not spent on repeats."""
    serial = iter(range(1000))

    def shape(_part, candidate, _config, **_kwargs):
        values = [deepcopy(candidate) for _ in range(3)]
        for value in values:
            origin = value.path.v[0]
            value.path.v = origin + (1.0 - 0.0001 * next(serial)) * (value.path.v - origin)
            value.length_m = adoption._length(value.path.v)
        return GroupRouteProposals(values)

    monkeypatch.setattr(adoption, "shape_joint_group_route", shape)


def state_snapshot(part):
    names = (
        "wire_path",
        "swept_body",
        "fabrication_path",
        "groups",
        "connected_groups",
        "contour_lines",
        "opening_cuts_among_groups",
        "manufacturing_diagnostics",
        "pcb_layout",
    )
    return tuple(id(getattr(part, name)) for name in names), pickle.dumps(tuple(getattr(part, name) for name in names))


@pytest.fixture
def setup_adoption(monkeypatch):
    target = TargetField(
        np.array([[0.0, 0, 0.1], [0.01, 0, 0.1]]),
        np.array([[0.0, 0, 0.004], [0, 0, 0.002]]),
        np.ones(2),
        np.zeros(2, dtype=bool),
        np.ones(2, dtype=int),
    )
    calls = SimpleNamespace(
        propose=[], shape=[], baseline=[], validate=[], support=[], domain=[], collision=[], field=[]
    )

    def field(points, targets, **_kwargs):
        calls.field.append(points.copy())
        return np.tile(np.array([points[0, 0], points[0, 2], 1.0]) * 1e-6, (len(targets), 1))

    def propose(part, _config, **_kwargs):
        calls.propose.append(part)
        return GroupRouteProposals([candidate_for(part), candidate_for(part)], diagnostics={"fixture": True})

    def shape(part, candidate, _config, **_kwargs):
        calls.shape.append(part)
        return GroupRouteProposals([deepcopy(candidate) for _ in range(3)])

    def baseline(part, _config, targets, limits, **kwargs):
        calls.baseline.append((part, targets, limits, kwargs))
        points = part.fabrication_path.wire_points_m
        return SimpleNamespace(
            fingerprint="fixture",
            physical_field_per_amp=np.tile(np.array([points[0, 0], points[0, 2], 1.0]) * 1e-6, (len(targets), 1)),
            diagnostics={},
            reasons=(),
        )

    def reject(part, path, _config, **kwargs):
        calls.validate.append((part, path, kwargs))
        return GroupRouteValidation(False, ("fixture_rejected",), {"complete": True})

    def support(part, _config):
        calls.support.append(part)
        return SimpleNamespace(accepted=True, reasons=(), diagnostics={"completed": True})

    def collision(bodies, gap, _limits):
        calls.collision.append((bodies, gap))
        return {"interpart_clearance_completed": True}, ""

    def domain(mesh, _config):
        calls.domain.append(mesh)
        return SimpleNamespace(accepted=True, reasons=(), diagnostics={"completed": True})

    monkeypatch.setattr(adoption, "certify_group_route_support", support)
    monkeypatch.setattr(adoption, "check_group_route_support_domain", domain)
    monkeypatch.setattr(adoption, "biot_savart_wire_path", field)
    monkeypatch.setattr(adoption, "propose_joint_group_routes", propose)
    monkeypatch.setattr(
        adoption,
        "propose_joint_group_topologies",
        lambda *args, **kwargs: adoption.propose_joint_group_routes(*args, **kwargs),
    )
    monkeypatch.setattr(
        adoption,
        "propose_joint_cut_variants",
        lambda *_a, **_k: SimpleNamespace(variants=[], reasons=(), diagnostics={}),
    )
    monkeypatch.setattr(adoption, "shape_joint_group_route", shape)
    monkeypatch.setattr(adoption, "prepare_group_route_baseline", baseline)
    monkeypatch.setattr(adoption, "validate_group_route_candidate", reject)
    monkeypatch.setattr(adoption, "_SurfaceProjector", lambda _mesh: object())
    monkeypatch.setattr(
        adoption, "JointTransitionEvaluator", lambda *_a, **_k: SimpleNamespace(set_search_level=lambda _level: None)
    )
    monkeypatch.setattr(adoption, "prepare_joint_group_context", lambda part, _config: part)
    monkeypatch.setattr(adoption, "raised_span_metrics", lambda *_a, **_k: {"jump_count": 0})
    monkeypatch.setattr(
        adoption,
        "assess_routing_baseline",
        lambda part, *_a, **_k: {
            "accepted": part.manufacturing_diagnostics.get("manufacturing_ok", False),
            "reasons": [],
        },
    )
    monkeypatch.setattr(adoption, "audit_interpart_clearance", collision)
    return target, calls, adoption


def test_default_enabled_and_explicit_off_performs_no_work(setup_adoption):
    target, calls, _ = setup_adoption
    assert CoilGenConfig().joint_group_routing is True
    assert CoilGenConfig().joint_group_routing_max_candidates == 6
    assert CoilGenConfig().joint_group_routing_max_field_change_rel == 0.001
    assert CoilGenConfig().joint_group_routing_field_acceptance == "absolute"
    assert CoilGenConfig().joint_group_routing_selection_policy == "regular"
    assert CoilGenConfig().joint_group_routing_regular_field_tolerance_rel == 0.001
    parts = [make_part()]
    before = state_snapshot(parts[0])
    assert adoption.improve_group_routes(parts, CoilGenConfig(joint_group_routing=False), target) is parts
    assert state_snapshot(parts[0]) == before
    assert not calls.field and not calls.propose


@pytest.mark.parametrize(
    "policy,increase_T,validator_accepts,adopted",
    [
        ("regular", 0.3e-6, True, True),
        (" REGULAR ", 0.3e-6, True, True),
        ("regular", 0.5e-6, True, False),
        ("regular", 0.0, False, False),
        ("physical", 0.3e-6, True, False),
        (" PhYsIcAl ", 0.3e-6, True, False),
    ],
)
def test_regular_preference_requires_baseline_anchored_field_budget_and_all_gates(
    setup_adoption,
    monkeypatch,
    policy,
    increase_T,
    validator_accepts,
    adopted,
):
    target, _, _ = setup_adoption
    part = make_part()
    owner = WireSegmentOwner("within_group_return", group_index=0)
    owners = [owner] * 3 + [WireSegmentOwner("contour_loop")] * 2
    part.wire_path = WirePath(
        np.array([[0.0, 0, 0], [2, 0, 0], [1, 0, 0], [3, 0, 0], [3, 1, 0], [0, 0, 0]]),
        owners,
    )
    part.fabrication_path = prepared_for(part, part.wire_path).fabrication_path
    candidate = candidate_for(part)
    candidate.path = WirePath(
        np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 1, 0], [0, 0, 0]]),
        deepcopy(owners),
    )
    candidate.segment_ids = (None,) * 5
    candidate.length_m = adoption._length(candidate.path.v)
    monkeypatch.setattr(adoption, "propose_joint_group_routes", lambda *_a, **_k: GroupRouteProposals([candidate]))

    def validate(context, path, *_a, **_k):
        if not validator_accepts:
            return GroupRouteValidation(False, ("route_fixed_current_target",), {})
        # Both unchanged contours contribute 1 uT/A. The original series route
        # contributes 1 uT/A at 2 A; this candidate's contour-reference RMS
        # increases by precisely increase_T. Geometry improves independently.
        cached = np.tile([0.0, 0.0, 1e-6 - increase_T / 2], (len(target.coords), 1))
        return GroupRouteValidation(True, (), {}, prepared_for(context, path), physical_field_per_amp=cached)

    monkeypatch.setattr(adoption, "validate_group_route_candidate", validate)
    config = CoilGenConfig(
        joint_group_routing_selection_policy=policy,
        joint_group_routing_regular_field_tolerance_rel=0.0001,
        joint_group_routing_max_candidates=1,
    )
    selected = adoption.improve_group_routes([part], config, target)[0]
    assert (selected is not part) is adopted
    diagnostics = selected.joint_route_diagnostics
    assert diagnostics["selection_policy"] == policy.strip().lower()
    assert diagnostics["regularity_part_rms_tolerance_T"] == pytest.approx(0.4e-6)
    assert diagnostics["regularity_maximum_longitudinal_rms_T"] == pytest.approx(2.4e-6)
    attempt = diagnostics["attempts"][0]
    if adopted:
        assert attempt["routing_metrics"]["added_backtracking_m"] == 0
        assert diagnostics["baseline_routing_metrics"]["added_backtracking_m"] == 1
    elif not validator_accepts:
        assert attempt["reasons"] == ["route_fixed_current_target"]
    elif policy == "regular":
        assert "regularity_field_rms_tolerance_exceeded" in attempt["reasons"]
    else:
        assert "no_routing_objective_or_manufacturing_improvement" in attempt["reasons"]


@pytest.mark.parametrize(
    "policy,tolerance,field",
    [
        ("beauty", 0.001, "joint_group_routing_selection_policy"),
        ("regular", -0.001, "joint_group_routing_regular_field_tolerance_rel"),
        ("regular", np.nan, "joint_group_routing_regular_field_tolerance_rel"),
        ("regular", True, "joint_group_routing_regular_field_tolerance_rel"),
    ],
)
def test_regular_policy_validation_rejects_undefined_ranking_or_budget(policy, tolerance, field):
    report = validate_config(
        CoilGenConfig(
            joint_group_routing_selection_policy=policy,
            joint_group_routing_regular_field_tolerance_rel=tolerance,
        )
    )
    assert any(message.field == field for message in report.errors)


def test_joint_progress_reports_measured_phase_eta_and_real_work_counts(setup_adoption, monkeypatch, capsys):
    target, calls, _ = setup_adoption
    part = make_part()
    before = state_snapshot(part)
    propose = adoption.propose_joint_group_routes
    clock = SimpleNamespace(seconds=0.0)
    monkeypatch.setattr(routing_progress, "perf_counter", lambda: clock.seconds)

    def evaluator(*_args, progress_callback, **_kwargs):
        return SimpleNamespace(progress_callback=progress_callback, diagnostics={})

    def search(context, config, *, transition_evaluator, progress_callback, **kwargs):
        progress_callback({"phase": "winding states", "completed": 0, "total": 4})
        for completed in range(1, 5):
            clock.seconds += 1.0
            progress_callback({"phase": "winding states", "completed": completed, "total": 4})
            if completed <= 2:
                samples, checks = ((8, 2), (20, 5))[completed - 1]
                transition_evaluator.progress_callback(
                    {
                        "generated_transition_count": completed,
                        "curve_sample_count": samples,
                        "support_checks": checks,
                    }
                )
        progress_callback({"phase": "winding states", "status": "finished"})
        progress_callback({"phase": "port connections", "completed": 0, "total": 6, "upper_bound": True})
        for completed in range(1, 4):
            clock.seconds += 20.0
            progress_callback(
                {
                    "phase": "port connections",
                    "completed": completed,
                    "total": 6,
                    "upper_bound": True,
                }
            )
        progress_callback({"phase": "port connections", "status": "finished", "message": "fixture graph pruned"})
        return propose(context, config, **kwargs)

    monkeypatch.setattr(adoption, "JointTransitionEvaluator", evaluator)
    monkeypatch.setattr(adoption, "propose_joint_group_routes", search)
    config = CoilGenConfig(
        progress_flag=True,
        joint_group_routing_search_mode="thorough",
        progress_min_interval_seconds=0.0,
        joint_group_routing_search_cuts=False,
        joint_group_routing_max_candidates=1,
    )
    events = []
    with progress_context(events.append):
        selected = adoption.improve_group_routes([part], config, target)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Joint routing")]
    assert any("completed parts 0/1" in line for line in lines)
    assert any("transitions=1; curve samples=8; support checks=2" in line for line in lines)
    assert any("transitions=2; curve samples=20; support checks=5" in line for line in lines)
    for percentage in ("25.0%", "50.0%", "100.0%"):
        assert any(percentage in line and "winding states:" in line for line in lines)
    assert all("%" not in line for line in lines if "port connections:" in line)
    assert "completed parts 1/1" in lines[-1]
    joint_events = [event for event in events if event["label"] == "Joint routing"]
    states = [event for event in joint_events if "winding states:" in event["message"]]
    connections = [event for event in joint_events if "port connections:" in event["message"]]
    assert {25.0, 50.0, 100.0} <= {event["percent"] for event in states}
    assert "0/4" in states[0]["message"] and "phase ETA estimating" in states[0]["message"]
    assert "0 of at most 6" in connections[0]["message"]
    assert "full-coverage ETA estimating" in connections[0]["message"]
    assert all(event["percent"] is None for event in connections)
    assert "3 of at most 6" in connections[-1]["message"]
    assert any("fixture graph pruned" in event["message"] for event in connections)
    assert "ETA" not in connections[-1]["message"]
    state_measured = next(event["message"] for event in states if "2/4" in event["message"])
    connection_measured = next(event["message"] for event in connections if "2 of at most 6" in event["message"])

    def eta_seconds(message):
        estimate = re.search(r"ETA ~(?:(\d+(?:\.\d+)?) min )?(\d+(?:\.\d+)?)\s*s", message)
        assert estimate is not None, message
        return 60 * float(estimate[1] or 0) + float(estimate[2])

    assert eta_seconds(state_measured) > 0
    assert eta_seconds(connection_measured) > 10 * eta_seconds(state_measured)
    assert "pruning may finish sooner" in connection_measured
    part_events = [event for event in joint_events if "completed parts" in event["message"]]
    assert part_events and all(event["percent"] is None and "ETA" not in event["message"] for event in part_events)
    assert "completed parts 1/1" in joint_events[-1]["message"]
    assert selected[0] is part and state_snapshot(part) == before
    assert len(calls.validate) == 1
    assert part.joint_route_diagnostics["status"] == "baseline_retained"


def test_unswept_route_with_no_supported_proposal_never_tessellates(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    part = make_part()
    part.swept_body = None
    part.manufacturing_diagnostics = {"swept_body_created": False}
    before = state_snapshot(part)
    monkeypatch.setattr(adoption, "propose_joint_group_routes", lambda *_a, **_k: GroupRouteProposals())
    monkeypatch.setattr(
        adoption, "create_swept_coil_meshes", lambda *_a: pytest.fail("No finalist needs a body preview")
    )
    selected = adoption.improve_group_routes([part], CoilGenConfig(), target)
    assert selected[0] is part and state_snapshot(part) == before
    assert not calls.field and not calls.baseline and not calls.validate
    assert part.joint_route_diagnostics["candidate_count"] == 0
    assert part.joint_route_diagnostics["selected_route_status"] == "unverified"


def test_finalists_materialize_isolated_baselines_and_all_backgrounds(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    parts = [make_part(0), make_part(1), make_part(2, groups=1)]
    for part in parts:
        part.swept_body = None
        part.manufacturing_diagnostics = {"swept_body_created": False}
    before = [state_snapshot(part) for part in parts]
    previews = []

    def sweep(isolated, config):
        # Baselines keep the original production spacing. The isolated
        # candidate's routing override must not alter the comparison reference.
        assert config.min_wire_distance_m == 0.005
        assert config.min_turn_distance == 0.005
        for preview in isolated:
            assert all(preview is not source for source in parts)
            preview.swept_body = deepcopy(preview.mesh)
            preview.manufacturing_diagnostics = {"manufacturing_ok": True}
            preview.fabrication_path = SimpleNamespace(
                wire_points_m=preview.wire_path.v + [0, 0, 0.01],
                segment_owners=deepcopy(preview.wire_path.segment_owners),
            )
            previews.append(preview)
        return isolated

    monkeypatch.setattr(adoption, "create_swept_coil_meshes", sweep)
    config = CoilGenConfig(
        min_wire_distance_m=0.005,
        min_turn_distance=0.005,
        joint_group_routing_clearance_m=0.003,
        joint_group_routing_max_candidates=1,
        joint_group_routing_search_cuts=False,
    )
    selected = adoption.improve_group_routes(parts, config, target)
    assert all(result is original for result, original in zip(selected, parts))
    assert [state_snapshot(part) for part in parts] == before
    assert len(previews) == 3  # Includes the ineligible single-loop background.
    assert len(calls.baseline) == len(calls.validate) == 2
    assert all(context.swept_body is not None for context, *_ in calls.baseline)
    assert all(context.fabrication_path is not None for context, *_ in calls.baseline)
    np.testing.assert_allclose(
        calls.validate[0][2]["other_physical_field_per_amp"],
        np.tile([9e-6, 0.02e-6, 2e-6], (2, 1)),
    )


@pytest.mark.parametrize("manufacturing_ok", [True, False])
def test_final_mesh_refreshes_retained_status_without_erasing_search_reason(manufacturing_ok):
    part = make_part()
    part.manufacturing_diagnostics = {}
    part.joint_route_diagnostics = adoption._retained_diagnostics(part, "no_supported_group_state")
    part.manufacturing_diagnostics = {
        "manufacturing_ok": manufacturing_ok,
        "manufacturing_failure_reasons": "" if manufacturing_ok else "bend_radius_violation",
    }
    adoption.refresh_retained_route_diagnostics([part])
    values = part.joint_route_diagnostics
    assert values["reason"] == "no_supported_group_state"
    assert values["selected_route_status"] == ("manufacturing_checks_passed" if manufacturing_ok else "invalid")
    assert (
        values["selected_manufacturing_failure_reasons"]
        == part.manufacturing_diagnostics["manufacturing_failure_reasons"]
    )


def test_final_manufacturing_refresh_preserves_independent_baseline_assessment():
    part = make_part()
    part.joint_route_diagnostics = {
        "status": "baseline_retained",
        "reason": "no_candidate_passed",
        "selected_route_status": "invalid",
        "validation_basis": "independent_baseline_assessment",
        "selected_manufacturing_failure_reasons": "",
    }
    part.manufacturing_diagnostics = {
        "manufacturing_ok": True,
        "manufacturing_failure_reasons": "reported_bend_warning",
    }
    adoption.refresh_retained_route_diagnostics([part])
    assert part.joint_route_diagnostics == {
        "status": "baseline_retained",
        "reason": "no_candidate_passed",
        "selected_route_status": "invalid",
        "validation_basis": "independent_baseline_assessment",
        "selected_manufacturing_failure_reasons": "reported_bend_warning",
    }


def test_route_acceptance_retains_explicit_absolute_field_failures_in_diagnostics(setup_adoption, monkeypatch):
    target, _calls, _adoption = setup_adoption
    part = make_part()
    part.manufacturing_diagnostics["manufacturing_ok"] = False

    def accept(context, path, _config, **_kwargs):
        return GroupRouteValidation(
            True,
            (),
            {
                "route_absolute_field_contract_met": False,
                "physical_absolute_field_contract_met": False,
                "route_absolute_field_contract_reasons": "gradient_strength_contract",
                "physical_absolute_field_contract_reasons": "gradient_strength_contract",
                "route_inherited_field_contract_violations": "gradient_strength_contract",
                "physical_inherited_field_contract_violations": "gradient_strength_contract",
            },
            prepared_for(context, path),
        )

    monkeypatch.setattr(adoption, "validate_group_route_candidate", accept)
    config = CoilGenConfig(joint_group_routing_field_acceptance="non_regression", joint_group_routing_max_candidates=1)
    selected = adoption.improve_group_routes([part], config, target)[0]
    diagnostics = selected.joint_route_diagnostics
    assert diagnostics["status"] == "accepted"
    assert diagnostics["field_acceptance_policy"] == "non_regression"
    assert diagnostics["selected_absolute_field_contract_met"] is False
    assert diagnostics["selected_inherited_field_contract_violations"] == {
        "route": "gradient_strength_contract",
        "physical": "gradient_strength_contract",
    }
    assert (
        diagnostics["selected_absolute_field_contract_reasons"]
        == diagnostics["selected_inherited_field_contract_violations"]
    )


@pytest.mark.parametrize("policy", ["", "regression", "assume_valid", True, None])
def test_field_acceptance_config_rejects_unknown_policies(policy):
    report = validate_config(CoilGenConfig(joint_group_routing_field_acceptance=policy))
    assert any(message.field == "joint_group_routing_field_acceptance" for message in report.errors)


@pytest.mark.parametrize("policy", ["absolute", "non_regression"])
def test_declared_field_acceptance_policies_validate(policy):
    report = validate_config(CoilGenConfig(joint_group_routing_field_acceptance=policy))
    assert not any(message.field == "joint_group_routing_field_acceptance" for message in report.errors)


@pytest.mark.parametrize("flag", ["skip_sweep", "skip_postprocessing", "contour_only_sweep_mode"])
def test_explicit_pipeline_skip_modes_remain_authoritative(setup_adoption, flag):
    target, calls, _ = setup_adoption
    config = CoilGenConfig()
    setattr(config, flag, True)
    parts = [make_part()]
    assert adoption.improve_group_routes(parts, config, target) is parts
    assert not calls.field and not calls.propose


@pytest.mark.parametrize("budget", [1, 2, 4, 5, 6])
def test_hard_candidate_budget_spans_multiple_proposals(setup_adoption, monkeypatch, budget):
    target, calls, _ = setup_adoption
    distinct_shaped_proposals(monkeypatch)
    parts = [make_part()]
    result = adoption.improve_group_routes(
        parts,
        CoilGenConfig(joint_group_routing_search_mode="thorough", joint_group_routing_max_candidates=budget),
        target,
    )
    assert result[0] is parts[0]
    assert len(calls.validate) == budget
    assert len(calls.baseline) == 1
    assert calls.baseline[0][3] == {"reuse_existing_sweep": True, "audit_geometry": False}
    assert result[0].joint_route_diagnostics["candidate_count"] == budget


def test_all_group_orders_receive_first_shape_before_later_alternatives(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    part = make_part()
    proposals = [candidate_for(part) for _ in range(4)]
    for index, candidate in enumerate(proposals):
        candidate.variant_names = (str(index),)
    monkeypatch.setattr(adoption, "propose_joint_group_routes", lambda *_a, **_k: GroupRouteProposals(proposals))

    def ordered_shapes(_part, proposal, _config, **_kwargs):
        candidates = [deepcopy(proposal) for _ in range(3)]
        for rank, candidate in enumerate(candidates):
            candidate.variant_names = (*proposal.variant_names, str(rank))
        return GroupRouteProposals(candidates)

    monkeypatch.setattr(adoption, "shape_joint_group_route", ordered_shapes)
    adoption.improve_group_routes(
        [part],
        CoilGenConfig(
            joint_group_routing_max_candidates=6,
            joint_group_routing_search_mode="thorough",
            joint_group_routing_selection_policy="physical",
        ),
        target,
    )
    assert len(calls.validate) == 6
    assert [row["variants"] for row in part.joint_route_diagnostics["attempts"]] == [
        ("0", "0"),
        ("1", "0"),
        ("2", "0"),
        ("3", "0"),
        ("0", "1"),
        ("1", "1"),
    ]


def test_pcb_mode_skips_before_support_or_field_work(setup_adoption):
    target, calls, _ = setup_adoption
    parts = [make_part()]
    before = state_snapshot(parts[0])
    result = adoption.improve_group_routes(parts, CoilGenConfig(make_cylndrical_pcb=True), target)
    assert result is parts and state_snapshot(parts[0]) == before
    assert parts[0].joint_route_diagnostics["reason"] == "pcb_layer_routing"
    assert not calls.domain and not calls.field and not calls.propose


def test_unsupported_domain_skips_before_target_or_field_work(setup_adoption, monkeypatch):
    _, calls, _ = setup_adoption
    parts = [make_part()]
    before = state_snapshot(parts[0])
    monkeypatch.setattr(
        adoption,
        "check_group_route_support_domain",
        lambda *_: SimpleNamespace(
            accepted=False, reasons=("unsupported_nonplanar_support",), diagnostics={"completed": True}
        ),
    )
    result = adoption.improve_group_routes(parts, CoilGenConfig(), None)
    assert result is parts and state_snapshot(parts[0]) == before
    assert parts[0].joint_route_diagnostics["reason"] == "support_domain_not_certified"
    assert not calls.field and not calls.propose


def test_all_rejections_preserve_route_body_group_objects_and_bytes(setup_adoption):
    target, calls, _ = setup_adoption
    parts = [make_part()]
    before = state_snapshot(parts[0])
    result = adoption.improve_group_routes(parts, CoilGenConfig(), target)
    assert result[0] is parts[0]
    assert state_snapshot(parts[0]) == before
    assert parts[0].joint_route_diagnostics["status"] == "baseline_retained"
    assert not calls.support and not calls.collision


def test_fixed_current_separate_backgrounds_and_cumulative_budget(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    parts = [make_part(0), make_part(1), make_part(2, groups=1)]
    accepted = []

    def accept(part, path, _config, **kwargs):
        calls.validate.append((part, path, kwargs))
        prepared = prepared_for(part, path)
        accepted.append(prepared)
        return GroupRouteValidation(True, (), {}, prepared)

    monkeypatch.setattr(adoption, "validate_group_route_candidate", accept)
    config = CoilGenConfig(joint_group_routing_max_field_change_rel=0.02, joint_group_routing_max_candidates=1)
    result = adoption.improve_group_routes(parts, config, target)
    assert len(calls.validate) == 2
    whole_allowance = 0.02 * 0.004 / 2.0
    for _, _, kwargs in calls.validate:
        assert kwargs["current_a"] == 2.0
        assert kwargs["apply_normal_shift"] is False
        assert kwargs["maximum_field_delta_t_per_a"] == whole_allowance / 2
        assert kwargs["target_field"] is target
    assert sum(row[2]["maximum_field_delta_t_per_a"] for row in calls.validate) == whole_allowance
    # The second candidate sees the first accepted part's physical centroid,
    # while the unchanged third (ineligible) part still contributes to both fields.
    second = calls.validate[1][2]
    expected_route = np.tile([6e-6, 0, 2e-6], (2, 1))
    expected_physical = np.tile([6e-6, 0.03e-6, 2e-6], (2, 1))
    np.testing.assert_allclose(second["other_field_per_amp"], expected_route)
    np.testing.assert_allclose(second["other_physical_field_per_amp"], expected_physical)
    assert result[:2] == accepted
    assert result[2] is parts[2]
    assert result[0].pcb_layout is None and parts[0].pcb_layout == {"old": "layout"}


@pytest.mark.parametrize("failure", ["target", "current", "centroid", "no_active_dsv"])
def test_missing_context_retains_baseline_before_search(setup_adoption, failure):
    target, calls, _ = setup_adoption
    parts = [make_part()]
    if failure == "target":
        target = None
    elif failure == "current":
        parts[0].contour_step = 0.0
    elif failure == "centroid":
        parts[0].fabrication_path = None
    else:
        target.weights[:] = 0.0
    before = state_snapshot(parts[0])
    assert adoption.improve_group_routes(parts, CoilGenConfig(), target) is parts
    assert not calls.field and not calls.propose
    assert state_snapshot(parts[0]) == before
    assert parts[0].joint_route_diagnostics["status"] == "baseline_retained"


@pytest.mark.parametrize("defect", ["weight_shape", "suppression_shape", "target_shape", "nonfinite_coords"])
def test_malformed_target_context_fails_closed_before_field_generation(setup_adoption, defect):
    target, calls, _ = setup_adoption
    parts = [make_part()]
    if defect == "weight_shape":
        target.weights = np.ones(3)
    elif defect == "suppression_shape":
        target.suppressed = np.zeros(3, dtype=bool)
    elif defect == "target_shape":
        target.b = np.zeros((3, 3))
    else:
        target.coords[0, 0] = np.nan
    before = state_snapshot(parts[0])
    result = adoption.improve_group_routes(parts, CoilGenConfig(), target)
    assert result is parts and state_snapshot(parts[0]) == before
    assert not calls.field and not calls.propose
    assert parts[0].joint_route_diagnostics["status"] == "baseline_retained"


@pytest.mark.parametrize("failure", ["support", "multipart_collision"])
def test_postvalidation_rejections_do_not_adopt(setup_adoption, monkeypatch, failure):
    target, calls, support_module = setup_adoption
    parts = [make_part(), make_part(1, groups=1)]
    before = [state_snapshot(part) for part in parts]
    monkeypatch.setattr(
        adoption,
        "validate_group_route_candidate",
        lambda part, path, *_a, **_k: GroupRouteValidation(True, (), {}, prepared_for(part, path)),
    )
    if failure == "support":
        monkeypatch.setattr(
            support_module,
            "certify_group_route_support",
            lambda *_: SimpleNamespace(
                accepted=False, reasons=("support_penetration",), diagnostics={"completed": True}
            ),
        )
        reason = "support_penetration"
    else:
        monkeypatch.setattr(
            adoption,
            "audit_interpart_clearance",
            lambda *_: (
                {"exact_collision_completed": True, "exact_intersection_count": 1},
                "swept_mesh_self_intersection",
            ),
        )
        reason = "swept_mesh_self_intersection"
    result = adoption.improve_group_routes(parts, CoilGenConfig(joint_group_routing_max_candidates=1), target)
    assert all(new is old for new, old in zip(result, parts))
    assert [state_snapshot(part) for part in parts] == before
    assert result[0].joint_route_diagnostics["attempts"][0]["reasons"] == [reason]
    if failure == "support":
        assert not calls.collision


def test_only_accepted_part_is_replaced_and_source_remains_intact(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    parts = [make_part(), make_part(1)]
    before = [state_snapshot(part) for part in parts]

    def decide(part, path, _config, **kwargs):
        calls.validate.append((part, path, kwargs))
        if part is parts[0]:
            return GroupRouteValidation(True, (), {}, prepared_for(part, path))
        return GroupRouteValidation(False, ("manufacturing_not_certified",))

    monkeypatch.setattr(adoption, "validate_group_route_candidate", decide)
    result = adoption.improve_group_routes(parts, CoilGenConfig(joint_group_routing_max_candidates=1), target)
    assert result[0] is not parts[0] and result[1] is parts[1]
    assert result[0].mesh is parts[0].mesh
    assert result[0].mesh.vertices.flags.writeable
    assert result[0].joint_route_diagnostics["status"] == "accepted"
    assert result[1].joint_route_diagnostics["status"] == "baseline_retained"
    assert [state_snapshot(part) for part in parts] == before


def test_impossible_geometry_does_not_trigger_full_grid_field_work(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    part = make_part()
    monkeypatch.setattr(
        adoption, "shape_joint_group_route", lambda *_a, **_k: GroupRouteProposals(reason="no_supported_transition")
    )
    selected = adoption.improve_group_routes([part], CoilGenConfig(), target)
    assert selected[0] is part
    assert not calls.field and not calls.baseline and not calls.validate
    assert part.joint_route_diagnostics["reason"] == "no_validated_improvement"


def test_valid_baseline_requires_real_improvement(setup_adoption, monkeypatch):
    target, _, _ = setup_adoption
    part = make_part()

    def no_improvement(part, _config, **_kwargs):
        candidate = candidate_for(part)
        candidate.path = deepcopy(part.wire_path)
        candidate.length_m = adoption._length(part.wire_path.v)
        return GroupRouteProposals([candidate])

    monkeypatch.setattr(adoption, "propose_joint_group_routes", no_improvement)
    monkeypatch.setattr(
        adoption,
        "validate_group_route_candidate",
        lambda part, path, *_a, **_k: GroupRouteValidation(True, (), {}, prepared_for(part, path)),
    )
    result = adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=1), target)
    assert result[0] is part
    assert part.joint_route_diagnostics["attempts"][0]["reasons"] == [
        "no_routing_objective_or_manufacturing_improvement"
    ]


@pytest.mark.parametrize("has_field_cache", [True, False])
def test_validated_physical_field_is_reused_for_ranking_and_commit(setup_adoption, monkeypatch, has_field_cache):
    target, calls, _ = setup_adoption
    part = make_part()
    prepared = []
    original_field = adoption.biot_savart_wire_path
    physical_calls = []

    def field(points, targets, **kwargs):
        if prepared and np.array_equal(points, prepared[0].fabrication_path.wire_points_m):
            physical_calls.append(points)
        return original_field(points, targets, **kwargs)

    def validate(context, path, *_a, **_kwargs):
        result = prepared_for(context, path)
        prepared.append(result)
        if not has_field_cache:
            return SimpleNamespace(accepted=True, reasons=(), diagnostics={}, prepared_part=result)
        first = result.fabrication_path.wire_points_m[0]
        cached = np.tile(np.array([first[0], first[2], 1.0]) * 1e-6, (len(target.coords), 1))
        cached.flags.writeable = False
        return GroupRouteValidation(True, (), {}, result, physical_field_per_amp=cached)

    monkeypatch.setattr(adoption, "biot_savart_wire_path", field)
    monkeypatch.setattr(adoption, "validate_group_route_candidate", validate)
    selected = adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=1), target)[0]
    assert selected is prepared[0]
    assert selected.joint_route_diagnostics["status"] == "accepted"
    assert len(physical_calls) == (0 if has_field_cache else 1)


def test_later_fabricated_winner_beats_first_valid_and_shortest_raw_route(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    part = make_part()
    before = state_snapshot(part)
    proposals = [candidate_for(part) for _ in range(3)]
    for index, (candidate, scale) in enumerate(zip(proposals, (0.5, 0.7, 0.6))):
        candidate.path.v = part.wire_path.v * scale
        candidate.variant_names = (str(index),)
    monkeypatch.setattr(adoption, "propose_joint_group_routes", lambda *_a, **_k: GroupRouteProposals(proposals))
    monkeypatch.setattr(adoption, "shape_joint_group_route", lambda _p, c, *_a, **_k: GroupRouteProposals([c]))
    prepared = []

    def validate(context, path, _config, **kwargs):
        calls.validate.append((context, path, kwargs))
        result = prepared_for(context, path)
        scale = (0.9, 0.6, 0.8)[len(prepared)]
        result.fabrication_path.wire_points_m = part.wire_path.v * scale
        prepared.append(result)
        return GroupRouteValidation(True, (), {}, result)

    monkeypatch.setattr(adoption, "validate_group_route_candidate", validate)
    selected = adoption.improve_group_routes(
        [part], CoilGenConfig(joint_group_routing_search_mode="thorough", joint_group_routing_max_candidates=3), target
    )[0]
    assert selected is prepared[1]
    assert len(calls.validate) == 3 and len(calls.baseline) == 2
    assert all(call[0] is part for call in calls.validate)
    assert state_snapshot(part) == before
    diagnostics = selected.joint_route_diagnostics
    assert diagnostics["selected_length_m"] == pytest.approx(2.4)
    assert diagnostics["selected_routing_metrics"]["geometry"] == "fabricated_centroid"
    assert diagnostics["selected_attempt_index"] == 1
    assert [row["selected"] for row in diagnostics["attempts"]] == [False, True, False]
    assert all(row["eligible_for_adoption"] for row in diagnostics["attempts"])


def test_shorter_raw_route_cannot_hide_worse_fabricated_route(setup_adoption, monkeypatch):
    target, _, _ = setup_adoption
    part = make_part()

    def validate(context, path, *_a, **_k):
        result = prepared_for(context, path)
        result.fabrication_path.wire_points_m *= 2
        return GroupRouteValidation(True, (), {}, result)

    monkeypatch.setattr(adoption, "validate_group_route_candidate", validate)
    selected = adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=1), target)[0]
    assert selected is part
    assert selected.joint_route_diagnostics["attempts"][0]["reasons"] == [
        "no_routing_objective_or_manufacturing_improvement"
    ]


def test_winner_is_committed_once_with_constant_backgrounds_within_each_part(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    distinct_shaped_proposals(monkeypatch)
    parts = [make_part(0), make_part(1), make_part(2, groups=1)]
    counts = {id(part): 0 for part in parts}
    prepared = []

    def validate(context, path, _config, **kwargs):
        calls.validate.append((context, path, kwargs))
        index = counts[id(context)]
        counts[id(context)] += 1
        result = prepared_for(context, path)
        origin = path.v[0]
        result.fabrication_path.wire_points_m = origin + (0.9 if index == 0 else 0.7) * (path.v - origin)
        result.fabrication_path.wire_points_m[:, 2] += 0.02 + 0.01 * index
        prepared.append(result)
        return GroupRouteValidation(True, (), {}, result)

    monkeypatch.setattr(adoption, "validate_group_route_candidate", validate)
    selected = adoption.improve_group_routes(
        parts, CoilGenConfig(joint_group_routing_search_mode="thorough", joint_group_routing_max_candidates=2), target
    )
    assert len(calls.validate) == 4
    assert selected[0] is prepared[1] and selected[1] is prepared[3]
    for first in (0, 2):
        for key in ("other_field_per_amp", "other_physical_field_per_amp"):
            np.testing.assert_array_equal(calls.validate[first][2][key], calls.validate[first + 1][2][key])
        assert calls.validate[first][2]["baseline"] is calls.validate[first + 1][2]["baseline"]
    np.testing.assert_allclose(
        calls.validate[2][2]["other_physical_field_per_amp"], np.tile([6e-6, 0.04e-6, 2e-6], (2, 1))
    )
    assert all(call[2]["maximum_field_delta_t_per_a"] == pytest.approx(1e-6) for call in calls.validate)


def test_candidate_missing_fabricated_ownership_is_not_scored_as_protected(setup_adoption, monkeypatch):
    target, _, _ = setup_adoption
    part = make_part()

    def validate(context, path, *_a, **_k):
        result = prepared_for(context, path)
        result.fabrication_path.segment_owners = ()
        return GroupRouteValidation(True, (), {}, result)

    monkeypatch.setattr(adoption, "validate_group_route_candidate", validate)
    assert adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=1), target)[0] is part
    assert part.joint_route_diagnostics["attempts"][0]["reasons"] == ["missing_candidate_objective_provenance"]


def test_unknown_legacy_fabricated_ownership_uses_explicit_length_lower_bound():
    part = make_part()
    part.fabrication_path.segment_owners = ()
    metrics = adoption._route_metrics(part, CoilGenConfig())
    assert metrics["length_m"] == metrics["cost_m"] == 4
    assert metrics["transverse_length_m"] == 0
    assert metrics["cost_is_lower_bound"] is True


def test_missing_previously_accepted_fabricated_ownership_retains_baseline(setup_adoption):
    target, calls, _ = setup_adoption
    part = make_part()
    part.joint_route_diagnostics = {"status": "accepted"}
    part.fabrication_path.segment_owners = ()
    before = state_snapshot(part)
    assert adoption.improve_group_routes([part], CoilGenConfig(), target)[0] is part
    assert state_snapshot(part) == before
    assert part.joint_route_diagnostics["reason"] == "baseline_objective_geometry_unavailable"
    assert not calls.field and not calls.propose


@pytest.mark.parametrize("weight,adopted", [(0.0, False), (1.0, False)])
def test_alignment_guide_cannot_override_equal_jumps_and_actual_field_error(
    setup_adoption, monkeypatch, weight, adopted
):
    target, _, _ = setup_adoption
    part = make_part()
    owner = WireSegmentOwner("within_group_return", group_index=0)
    part.wire_path = WirePath(np.array([[0.0, 0, 0], [1, 0, 0], [1, 0, 0.5], [0, 0, 0.5], [0, 0, 0]]), [owner] * 4)
    part.fabrication_path = prepared_for(part, part.wire_path).fabrication_path
    candidate = candidate_for(part)
    candidate.path = WirePath(
        np.array([[0.0, 0, 0], [0.5, 0, 0], [0.5, 0, 1.25], [0, 0, 1.25], [0, 0, 0]]), [owner] * 4
    )
    candidate.length_m = adoption._length(candidate.path.v)
    monkeypatch.setattr(adoption, "propose_joint_group_routes", lambda *_a, **_k: GroupRouteProposals([candidate]))
    monkeypatch.setattr(
        adoption,
        "validate_group_route_candidate",
        lambda original, path, *_a, **_k: GroupRouteValidation(True, (), {}, prepared_for(original, path)),
    )
    config = CoilGenConfig(joint_group_routing_b0_weight=weight, joint_group_routing_max_candidates=1)
    selected = adoption.improve_group_routes([part], config, target)[0]
    assert (selected is not part) is adopted
    assert selected.joint_route_diagnostics["baseline_routing_metrics"]["length_m"] == 3


def test_longer_fewer_jump_route_wins_after_independent_validation(setup_adoption, monkeypatch):
    target, _, _ = setup_adoption
    part = make_part()
    part.fabrication_path.test_jump_count = 3

    def validate(context, path, *_a, **_k):
        result = prepared_for(context, path)
        result.fabrication_path.wire_points_m *= 3
        result.fabrication_path.test_jump_count = 1
        return GroupRouteValidation(True, (), {}, result)

    monkeypatch.setattr(adoption, "validate_group_route_candidate", validate)
    selected = adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=1), target)[0]
    assert selected is not part
    values = selected.joint_route_diagnostics["selected_routing_metrics"]
    assert values["jump_count"] == 1
    assert values["length_m"] > selected.joint_route_diagnostics["baseline_length_m"]


def test_uncertified_baseline_cannot_veto_a_valid_more_jump_repair(setup_adoption, monkeypatch):
    target, _, _ = setup_adoption
    part = make_part()
    monkeypatch.setattr(
        adoption,
        "assess_routing_baseline",
        lambda *_a, **_k: {
            "accepted": False,
            "reasons": ["baseline_support_not_certified"],
        },
    )

    def validate(context, path, *_a, **_k):
        result = prepared_for(context, path)
        result.fabrication_path.test_jump_count = 2
        return GroupRouteValidation(True, (), {}, result)

    monkeypatch.setattr(adoption, "validate_group_route_candidate", validate)
    selected = adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=1), target)[0]
    assert selected is not part
    assert selected.joint_route_diagnostics["selected_routing_metrics"]["jump_count"] == 2
    assert not selected.joint_route_diagnostics["baseline_feasibility"]["accepted"]
    assert selected.joint_route_diagnostics["selection_basis"] == "baseline_repair"


@pytest.mark.parametrize("policy,adopted", [("physical", True), ("regular", False)])
def test_actual_longitudinal_field_improvement_beats_shorter_equal_jump_route_under_physical_policy(
    setup_adoption,
    monkeypatch,
    policy,
    adopted,
):
    target, _, _ = setup_adoption
    part = make_part()

    # Original two contours each produce 1 uT/A. Baseline has 1 uT/A;
    # the longer candidate has 2 uT/A and therefore zero routing distortion.
    def field(points, targets, **_kwargs):
        value = 2e-6 if points[0, 2] > 0.015 else 1e-6
        return np.tile([0.0, 0, value], (len(targets), 1))

    def validate(context, path, *_a, **_k):
        result = prepared_for(context, path)
        result.fabrication_path.wire_points_m[:, :2] *= 3
        return GroupRouteValidation(True, (), {}, result)

    monkeypatch.setattr(adoption, "biot_savart_wire_path", field)
    monkeypatch.setattr(adoption, "validate_group_route_candidate", validate)
    selected = adoption.improve_group_routes(
        [part], CoilGenConfig(joint_group_routing_max_candidates=1, joint_group_routing_selection_policy=policy), target
    )[0]
    assert (selected is not part) is adopted
    if not adopted:
        assert selected.joint_route_diagnostics["attempts"][0]["reasons"] == [
            "no_routing_objective_or_manufacturing_improvement"
        ]
        return
    values = selected.joint_route_diagnostics["selected_routing_metrics"]
    assert values["longitudinal_rms_T"] == 0
    assert values["length_m"] > 4
    assert selected.joint_route_diagnostics["baseline_routing_metrics"]["longitudinal_rms_T"] == pytest.approx(2e-6)


def test_lowest_proposed_jump_budget_is_validated_first_across_cut_contexts(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    part = make_part()
    proposals = [candidate_for(part), candidate_for(part)]
    proposals[0].path.v *= 0.5  # Shortest, but represents three transfers.
    proposals[1].path.v *= 1.5  # Longer, one transfer.
    monkeypatch.setattr(adoption, "propose_joint_group_routes", lambda *_a, **_k: GroupRouteProposals(proposals))
    monkeypatch.setattr(adoption, "shape_joint_group_route", lambda _p, c, *_a, **_k: GroupRouteProposals([c]))
    monkeypatch.setattr(
        adoption,
        "raised_span_metrics",
        lambda points, *_a, **_k: {
            "jump_count": 3 if adoption._length(points) < 3 else 1,
        },
    )
    adoption.improve_group_routes(
        [part], CoilGenConfig(joint_group_routing_search_mode="thorough", joint_group_routing_max_candidates=1), target
    )
    assert len(calls.validate) == 1
    assert calls.validate[0][1] is proposals[1].path
    assert part.joint_route_diagnostics["physical_candidate_budget_exhausted"]


def test_mutable_intra_group_joins_are_scored_without_biasing_protected_contours():
    part = make_part()
    part.wire_path = WirePath(
        np.array([[0.0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]]),
        [
            WireSegmentOwner("contour_group"),
            WireSegmentOwner("contour_loop"),
            WireSegmentOwner("within_group_return"),
            WireSegmentOwner("contour_loop"),
        ],
    )
    part.fabrication_path = prepared_for(part, part.wire_path).fabrication_path
    config = CoilGenConfig()
    exact = adoption._route_metrics(part, config, complete_provenance=True)
    conservative = adoption._route_metrics(part, config)
    assert exact["transverse_length_m"] == 2
    assert exact["cost_m"] == 6
    assert exact["cost_is_lower_bound"] is False
    assert conservative["cost_m"] <= exact["cost_m"]
    assert conservative["cost_is_lower_bound"] is True


def test_explicit_join_objective_survives_acceptance_and_later_retention():
    part = make_part()
    part.wire_path = WirePath(
        np.array([[0.0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]]),
        [
            WireSegmentOwner("contour_group", connector_role="joint_spiral_join"),
            WireSegmentOwner("contour_loop"),
            WireSegmentOwner("within_group_return"),
            WireSegmentOwner("contour_loop"),
        ],
    )
    part.fabrication_path = prepared_for(part, part.wire_path).fabrication_path
    config = CoilGenConfig()
    original = adoption._route_metrics(part, config)
    assert original["cost_m"] == 6
    assert original["cost_is_lower_bound"] is False
    for status in ("accepted", "baseline_retained", "baseline_retained"):
        part.joint_route_diagnostics = {"status": status}
        assert adoption._route_metrics(part, config) == original


def test_invalid_baseline_retention_is_never_reported_as_valid(setup_adoption):
    target, _, _ = setup_adoption
    part = make_part()
    part.manufacturing_diagnostics = {"manufacturing_ok": False, "manufacturing_failure_reasons": "folded_sweep_quad"}
    selected = adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=1), target)[0]
    assert selected is part
    assert selected.joint_route_diagnostics["status"] == "baseline_retained"
    assert selected.joint_route_diagnostics["selected_route_status"] == "invalid"
    assert selected.joint_route_diagnostics["selected_manufacturing_failure_reasons"] == "folded_sweep_quad"


def test_rejected_candidates_never_certify_the_incumbent(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    part = make_part()
    before = state_snapshot(part)
    monkeypatch.setattr(
        adoption, "assess_routing_baseline", lambda *_a, **_k: pytest.fail("Rejected route certified incumbent")
    )
    selected = adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=2), target)[0]
    assert selected is part and state_snapshot(part) == before
    assert all(row[3].get("audit_geometry") is False for row in calls.baseline)
    assert selected.joint_route_diagnostics["baseline_certification_deferred"]
    assert selected.joint_route_diagnostics["validation_basis"] == "existing_production_diagnostics"
    assert "baseline_feasibility" not in selected.joint_route_diagnostics


@pytest.mark.parametrize("failure", ["missing_field", "changed_field", "stale_source"])
def test_deferred_incumbent_certification_failure_blocks_adoption(setup_adoption, monkeypatch, failure):
    target, calls, _ = setup_adoption
    part = make_part()
    before = state_snapshot(part)
    prepare = adoption.prepare_group_route_baseline

    def baseline(*args, **kwargs):
        result = prepare(*args, **kwargs)
        if kwargs.get("audit_geometry", True):
            if failure == "missing_field":
                result.physical_field_per_amp = None
            elif failure == "changed_field":
                result.physical_field_per_amp += 1e-6
            else:
                result.fingerprint = "changed_source"
        return result

    monkeypatch.setattr(adoption, "prepare_group_route_baseline", baseline)
    monkeypatch.setattr(
        adoption,
        "validate_group_route_candidate",
        lambda context, path, *_a, **_k: GroupRouteValidation(True, (), {}, prepared_for(context, path)),
    )
    monkeypatch.setattr(
        adoption, "assess_routing_baseline", lambda *_a, **_k: pytest.fail("Failed baseline became repair incumbent")
    )
    selected = adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=2), target)[0]
    assert selected is part and state_snapshot(part) == before
    expected = {
        "missing_field": "baseline_physical_field_unavailable",
        "changed_field": "baseline_physical_field_changed",
        "stale_source": "stale_baseline",
    }[failure]
    assert all(expected in attempt["reasons"] for attempt in selected.joint_route_diagnostics["attempts"])
    assert len(calls.baseline) == 2  # Fields once and one full certification attempt.


def test_candidate_support_failure_skips_incumbent_certification(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    part = make_part()
    monkeypatch.setattr(
        adoption,
        "validate_group_route_candidate",
        lambda context, path, *_a, **_k: GroupRouteValidation(True, (), {}, prepared_for(context, path)),
    )
    monkeypatch.setattr(
        adoption,
        "certify_group_route_support",
        lambda *_a: SimpleNamespace(accepted=False, reasons=("unsupported_candidate",), diagnostics={}),
    )
    monkeypatch.setattr(
        adoption, "assess_routing_baseline", lambda *_a, **_k: pytest.fail("Unsupported candidate certified incumbent")
    )
    selected = adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=1), target)[0]
    assert selected is part and len(calls.baseline) == 1
    assert selected.joint_route_diagnostics["attempts"][0]["reasons"] == ["unsupported_candidate"]


def test_deferred_incumbent_reference_keeps_inactive_nan_samples(setup_adoption, monkeypatch):
    target, _calls, _ = setup_adoption
    part = make_part()
    target.weights[1] = 0.0
    prepare = adoption.prepare_group_route_baseline

    def baseline(*args, **kwargs):
        result = prepare(*args, **kwargs)
        result.physical_field_per_amp[1] = np.nan
        return result

    monkeypatch.setattr(adoption, "prepare_group_route_baseline", baseline)
    monkeypatch.setattr(
        adoption,
        "validate_group_route_candidate",
        lambda context, path, *_a, **_k: GroupRouteValidation(True, (), {}, prepared_for(context, path)),
    )
    selected = adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=1), target)[0]
    assert selected is not part
    assert not selected.joint_route_diagnostics["baseline_certification_deferred"]


def test_alternate_cut_context_is_independently_validated_and_adopted_atomically(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    original = make_part()
    alternative = deepcopy(original)
    alternative.groups[0].within_group_cut_side = "low"
    before = state_snapshot(original)
    monkeypatch.setattr(
        adoption,
        "propose_joint_cut_variants",
        lambda *_a, **_k: SimpleNamespace(variants=[alternative], reasons=(), diagnostics={"fixture": True}),
    )

    def decide(context, path, _config, **kwargs):
        calls.validate.append((context, path, kwargs))
        if context is original:
            return GroupRouteValidation(False, ("incompatible_existing_ports",))
        assert context is alternative
        return GroupRouteValidation(True, (), {}, prepared_for(context, path))

    monkeypatch.setattr(adoption, "validate_group_route_candidate", decide)
    selected = adoption.improve_group_routes([original], CoilGenConfig(joint_group_routing_max_candidates=2), target)[0]
    assert selected is not original
    assert selected.groups[0].within_group_cut_side == "low"
    assert state_snapshot(original) == before
    assert [row[0] for row in calls.baseline] == [original, alternative, alternative]
    assert calls.baseline[-1][3]["reuse_existing_sweep"]
    assert "audit_geometry" not in calls.baseline[-1][3]
    assert selected.joint_route_diagnostics["selected_cut_context"] == "alternate_cuts_1"


def test_explicit_cut_search_disable_preserves_fixed_cut_scope(setup_adoption, monkeypatch):
    target, _, _ = setup_adoption
    monkeypatch.setattr(adoption, "propose_joint_cut_variants", lambda *_a, **_k: pytest.fail("cut search disabled"))
    adoption.improve_group_routes([make_part()], CoilGenConfig(joint_group_routing_search_cuts=False), target)


def test_inconsistent_series_current_rejected_before_geometry_and_field_search(setup_adoption):
    target, calls, _ = setup_adoption
    parts = [make_part(), make_part(1)]
    parts[1].contour_step *= 2
    assert adoption.improve_group_routes(parts, CoilGenConfig(), target) is parts
    assert not calls.propose and not calls.field
    assert all(part.joint_route_diagnostics["reason"] == "inconsistent_series_current" for part in parts)


def test_single_multi_loop_group_is_admitted_for_its_own_return(setup_adoption):
    target, calls, _ = setup_adoption
    part = make_part(groups=1)
    part.groups[0].loops.append(deepcopy(part.groups[0].loops[0]))
    part.groups[0].opened_loops.append(deepcopy(part.groups[0].opened_loops[0]))
    adoption.improve_group_routes([part], CoilGenConfig(joint_group_routing_max_candidates=1), target)
    assert calls.propose == [part]
    assert len(calls.validate) == 1


def test_multipart_gate_uses_full_insulation_and_tolerance_gap(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    monkeypatch.setattr(
        adoption,
        "validate_group_route_candidate",
        lambda part, path, *_a, **_k: GroupRouteValidation(True, (), {}, prepared_for(part, path)),
    )
    config = CoilGenConfig(
        sweep_insulation_thickness=0.001,
        sweep_manufacturing_tolerance=0.0002,
        sweep_bend_clearance_margin=0.0003,
        joint_group_routing_max_candidates=1,
    )
    adoption.improve_group_routes([make_part(), make_part(1, groups=1)], config, target)
    assert calls.collision[0][1] == pytest.approx(0.0025)


def test_invalid_other_body_cannot_supply_a_clearance_certificate(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    parts = [make_part(), make_part(1, groups=1)]
    parts[1].manufacturing_diagnostics["manufacturing_ok"] = False
    monkeypatch.setattr(
        adoption,
        "validate_group_route_candidate",
        lambda part, path, *_a, **_k: GroupRouteValidation(True, (), {}, prepared_for(part, path)),
    )
    selected = adoption.improve_group_routes(parts, CoilGenConfig(joint_group_routing_max_candidates=1), target)
    assert selected[0] is parts[0]
    assert selected[0].joint_route_diagnostics["attempts"][0]["reasons"] == ["other_part_manufacturing_not_certified"]
    assert not calls.collision


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, "bad"])
def test_b0_weight_config_requires_finite_nonnegative(value):
    report = validate_config(CoilGenConfig(joint_group_routing_b0_weight=value))
    assert any(message.field == "joint_group_routing_b0_weight" for message in report.errors)


@pytest.mark.parametrize("value", [0, -1, 65, 1.5, True, np.nan, np.inf, "6"])
def test_candidate_budget_config_requires_bounded_integer(value):
    report = validate_config(CoilGenConfig(joint_group_routing_max_candidates=value))
    assert any(message.field == "joint_group_routing_max_candidates" for message in report.errors)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_relative_field_budget_config_requires_finite_nonnegative(value):
    report = validate_config(CoilGenConfig(joint_group_routing_max_field_change_rel=value))
    assert any(message.field == "joint_group_routing_max_field_change_rel" for message in report.errors)


@pytest.mark.parametrize("value", [1, "yes"])
def test_routing_flag_config_requires_boolean(value):
    report = validate_config(CoilGenConfig(joint_group_routing=value))
    assert any(message.field == "joint_group_routing" for message in report.errors)


@pytest.mark.parametrize("field", ["routing_jump_height_tolerance_m", "routing_jump_sample_spacing_m"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, "bad"])
def test_jump_resolution_requires_finite_positive_measurement(field, value):
    report = validate_config(CoilGenConfig(**{field: value}))
    assert any(message.field == field for message in report.errors)


@pytest.mark.parametrize("value", ["", "auto", "terminal", None])
def test_routing_circuit_scope_requires_explicit_supported_mode(value):
    report = validate_config(CoilGenConfig(routing_circuit_scope=value))
    assert any(message.field == "routing_circuit_scope" for message in report.errors)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, "bad"])
def test_joint_routing_clearance_requires_finite_positive_override(value):
    report = validate_config(CoilGenConfig(joint_group_routing_clearance_m=value))
    assert any(message.field == "joint_group_routing_clearance_m" for message in report.errors)


@pytest.mark.parametrize("value", [None, 0.0035])
def test_joint_routing_clearance_projection_and_valid_override(value):
    config = CoilGenConfig(joint_group_routing_clearance_m=value)
    assert config.routing_settings.joint_group_routing_clearance_m == value
    assert not any(message.field == "joint_group_routing_clearance_m" for message in validate_config(config).errors)


def test_legacy_slotted_config_without_clearance_override_keeps_existing_settings(setup_adoption):
    target, _calls, module = setup_adoption
    config = CoilGenConfig(min_wire_distance_m=0.005, min_turn_distance=0.005)
    del config.joint_group_routing_clearance_m
    del config.joint_group_routing_field_acceptance
    part = make_part()
    assert module.improve_group_routes([part], config, target)[0] is part
    assert part.joint_route_diagnostics["routing_clearance_override_m"] is None
    assert part.joint_route_diagnostics["field_acceptance_policy"] == "absolute"
    assert part.joint_route_diagnostics["routing_clearance_target_m"] == 0.005
    assert config.min_wire_distance_m == config.min_turn_distance == 0.005


def test_global_six_physical_slots_keep_complete_families_without_promoting_extra_jumps():
    def candidate(jumps, guide, family):
        return (
            jumps,
            guide,
            "fixture",
            None,
            SimpleNamespace(joint_return_family_counts={family: 2}),
        )

    cheap = [candidate(1, 10 + index * 0.01, "planar") for index in range(8)]
    compact = candidate(1, 20, "intrinsic_compact")
    b0 = candidate(1, 30, "b0_straight")
    ordered = adoption._order_complete_return_families([*cheap, compact, b0])
    assert ordered[0] is cheap[0]
    assert any(item is compact for item in ordered[:6])
    assert any(item is b0 for item in ordered[:6])
    lower = [candidate(0, guide, "planar") for guide in range(7)]
    ordered = adoption._order_complete_return_families([*lower, *cheap, compact, b0])
    assert all(item[0] == 0 for item in ordered[:6])


@pytest.mark.parametrize("override, turn_floor", [(None, 0.005), (0.0035, 0.005), (0.0005, 0.005)])
def test_route_clearance_override_is_isolated_and_keeps_physical_floors(
    setup_adoption, monkeypatch, override, turn_floor
):
    from coilgen.manufacturing import _required_centerline_clearance, conductor_cross_section_points

    target, _calls, module = setup_adoption
    part = make_part()
    config = CoilGenConfig(
        min_wire_distance_m=0.005,
        min_turn_distance=turn_floor,
        joint_group_routing_clearance_m=override,
        joint_group_routing_max_candidates=1,
    )
    before = pickle.dumps(config)
    physical_floor = _required_centerline_clearance(conductor_cross_section_points(config), config)
    configs = []

    def accept(original, path, routing_config, **_kwargs):
        configs.append(routing_config)
        prepared = prepared_for(original, path)
        prepared.manufacturing_diagnostics["sweep_capsule_min_centerline_distance_m"] = 0.0036
        return GroupRouteValidation(True, (), {}, prepared)

    monkeypatch.setattr(module, "validate_group_route_candidate", accept)
    selected = module.improve_group_routes([part], config, target)[0]
    assert selected is not part
    assert pickle.dumps(config) == before
    routing = configs[0]
    if override is None:
        assert routing is config
        expected = max(config.min_wire_distance_m, turn_floor, physical_floor)
    else:
        assert routing is not config
        expected = max(override, physical_floor)
        assert routing.min_wire_distance_m == override
        assert _required_centerline_clearance(conductor_cross_section_points(routing), routing) == pytest.approx(
            expected
        )
    assert routing.min_turn_distance == (turn_floor if override is None else override)
    assert selected.joint_route_diagnostics["routing_clearance_target_m"] == pytest.approx(expected)
    assert selected.joint_route_diagnostics["source_contour_clearance_m"] == 0.005
    assert selected.joint_route_diagnostics["selected_physical_minimum_centerline_distance_m"] == 0.0036


def test_physical_routing_clearance_checks_remote_legs_and_excludes_smooth_continuations():
    from coilgen.manufacturing import _capsule_clearance_diagnostics

    continuation = np.column_stack((np.linspace(0, 0.025, 200), np.zeros((200, 2))))
    local = _capsule_clearance_diagnostics(continuation, 0.0035, 0.0014, closed=False)
    assert not local["sweep_capsule_clearance_violation"]
    for gap, fails in [(0.003, True), (0.004, False)]:
        radius = gap / 2
        angle = np.linspace(-np.pi / 2, np.pi / 2, 100)
        arc = np.column_stack((0.020 + radius * np.cos(angle), radius + radius * np.sin(angle), np.zeros(100)))
        first = np.column_stack((np.linspace(0, 0.020, 101), np.zeros((101, 2))))
        last = first[::-1] + [0, gap, 0]
        points = np.vstack((first, arc[1:], last[1:]))
        audit = _capsule_clearance_diagnostics(points, 0.0035, 0.0014, closed=False)
        assert audit["sweep_capsule_clearance_violation"] is fails
        assert not audit["sweep_capsule_self_intersection"]


@pytest.mark.parametrize("value", [-0.1, 0, np.nan, np.inf, True, "invalid"])
def test_config_rejects_invalid_optional_longitudinal_budget(value):
    config = CoilGenConfig(joint_group_routing_max_longitudinal_field_change_rel=value)
    report = validate_config(config)
    assert any(message.field == "joint_group_routing_max_longitudinal_field_change_rel" for message in report.errors)


@pytest.mark.parametrize("value", [None, 0.05])
def test_config_exposes_optional_longitudinal_budget_in_routing_settings(value):
    config = CoilGenConfig(
        joint_group_routing_field_acceptance="budget_only",
        joint_group_routing_max_longitudinal_field_change_rel=value,
    )
    assert config.routing_settings.joint_group_routing_max_longitudinal_field_change_rel == value
    assert not any(
        message.field
        in {"joint_group_routing_field_acceptance", "joint_group_routing_max_longitudinal_field_change_rel"}
        for message in validate_config(config).errors
    )


def test_longitudinal_budget_allocation_includes_only_eligible_parts_and_uses_requested_target_peak(setup_adoption):
    target, calls, _ = setup_adoption
    parts = [make_part(0), make_part(1), make_part(2, groups=1)]
    config = CoilGenConfig(
        joint_group_routing_field_acceptance="budget_only",
        joint_group_routing_max_field_change_rel=0.3,
        joint_group_routing_max_longitudinal_field_change_rel=0.05,
        joint_group_routing_max_candidates=1,
    )
    adoption.improve_group_routes(parts, config, target)
    assert len(calls.validate) == 2
    whole_longitudinal_budget = 0.05 * 0.004 / 2.0
    whole_vector_budget = 0.3 * 0.004 / 2.0
    for part, _, kwargs in calls.validate:
        assert kwargs["maximum_longitudinal_field_delta_t_per_a"] == whole_longitudinal_budget / 2
        assert kwargs["maximum_field_delta_t_per_a"] == whole_vector_budget / 2
        assert part.joint_route_diagnostics["maximum_longitudinal_field_delta_t_per_a"] == whole_longitudinal_budget / 2
    assert (
        sum(row[2]["maximum_longitudinal_field_delta_t_per_a"] for row in calls.validate) == whole_longitudinal_budget
    )


def test_budget_only_selected_diagnostics_preserve_new_and_inherited_field_failures(setup_adoption, monkeypatch):
    target, _calls, _adoption = setup_adoption
    part = make_part()
    part.manufacturing_diagnostics["manufacturing_ok"] = False

    def accept(context, path, _config, **_kwargs):
        diagnostics = {}
        for stage in ("route", "physical"):
            diagnostics.update(
                {
                    f"{stage}_absolute_field_contract_met": False,
                    f"{stage}_absolute_field_contract_reasons": "gradient_strength_contract; linearity_rms_contract",
                    f"{stage}_inherited_field_contract_violations": "gradient_strength_contract",
                    f"{stage}_budget_waived_field_contract_violations": "gradient_strength_contract; linearity_rms_contract",
                    f"{stage}_newly_failed_field_contract_violations": "linearity_rms_contract",
                    f"{stage}_regressed_inherited_field_contract_violations": "gradient_strength_contract",
                }
            )
        return GroupRouteValidation(True, (), diagnostics, prepared_for(context, path))

    monkeypatch.setattr(adoption, "validate_group_route_candidate", accept)
    config = CoilGenConfig(joint_group_routing_field_acceptance="budget_only", joint_group_routing_max_candidates=1)
    selected = adoption.improve_group_routes([part], config, target)[0]
    diagnostics = selected.joint_route_diagnostics
    assert diagnostics["status"] == "accepted"
    assert not diagnostics["selected_absolute_field_contract_met"]
    assert diagnostics["selected_newly_failed_field_contract_violations"] == {
        "route": "linearity_rms_contract",
        "physical": "linearity_rms_contract",
    }
    assert diagnostics["selected_regressed_inherited_field_contract_violations"] == {
        "route": "gradient_strength_contract",
        "physical": "gradient_strength_contract",
    }


def test_global_six_slots_keep_wide_balanced_pairs_and_lower_jump_priority():
    def candidate(jumps, cost, key=None):
        value = SimpleNamespace(joint_return_family_counts={})
        if key is not None:
            value.joint_detour_rail_key = key
        return jumps, cost, "fixture", None, value

    near = [candidate(2, 1 + value / 100, ((-1, 1), 0, 0.01, 0.02)) for value in range(8)]
    negative = candidate(2, 3, ((-1, 1), 0, 0.05, 0.1))
    positive = candidate(2, 4, ((1, -1), 0, 0.05, 0.1))
    same_winding = candidate(2, 2, ((-1, -1), 2, 0.05, 0.1))
    result = adoption._order_complete_return_families([*near, same_winding, negative, positive])
    assert result[0] is near[0]
    assert any(value is negative for value in result[:6])
    assert any(value is positive for value in result[:6])
    lower = [candidate(1, value) for value in range(6)]
    result = adoption._order_complete_return_families([*lower, *near, negative, positive])
    assert all(value[0] == 1 for value in result[:6])


@pytest.mark.parametrize("mode", ["progressive", "thorough"])
def test_search_mode_config_is_explicit(mode):
    config = CoilGenConfig(joint_group_routing_search_mode=mode)
    assert config.routing_settings.joint_group_routing_search_mode == mode
    assert not any(message.field == "joint_group_routing_search_mode" for message in validate_config(config).errors)


@pytest.mark.parametrize("mode", ["fast", "", None, True])
def test_search_mode_rejects_unrecognised_values(mode):
    assert any(
        message.field == "joint_group_routing_search_mode"
        for message in validate_config(CoilGenConfig(joint_group_routing_search_mode=mode)).errors
    )


def test_progressive_keeps_physical_budget_and_expands_only_after_rejection(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    levels = []
    monkeypatch.setattr(
        adoption,
        "JointTransitionEvaluator",
        lambda *_a, **_k: SimpleNamespace(set_search_level=levels.append),
    )
    count = iter(range(100))

    def unique_shapes(_part, candidate, *_args, **_kwargs):
        values = [deepcopy(candidate) for _ in range(3)]
        for value in values:
            value.path.v = value.path.v * (1.0 + 0.001 * next(count))
        return GroupRouteProposals(values)

    monkeypatch.setattr(adoption, "shape_joint_group_route", unique_shapes)
    part = make_part()
    before = state_snapshot(part)
    adoption.improve_group_routes([part], CoilGenConfig(), target)
    assert levels == [1]
    assert len(calls.validate) == 6
    assert [row["physical_candidate_budget"] for row in part.joint_route_diagnostics["search_stages"]] == [3, 3]
    assert state_snapshot(part) == before
    assert all(row[2]["current_a"] == 2.0 for row in calls.validate)
    assert len({row[2]["maximum_field_delta_t_per_a"] for row in calls.validate}) == 1


def test_progressive_acceptance_stops_before_expanding_curves(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    monkeypatch.setattr(
        adoption,
        "JointTransitionEvaluator",
        lambda *_a, **_k: SimpleNamespace(set_search_level=lambda _level: pytest.fail("unneeded expansion")),
    )

    def accept(part, path, _config, **kwargs):
        calls.validate.append((part, path, kwargs))
        return GroupRouteValidation(True, (), {}, prepared_for(part, path))

    monkeypatch.setattr(adoption, "validate_group_route_candidate", accept)
    part = make_part()
    selected = adoption.improve_group_routes([part], CoilGenConfig(), target)[0]
    assert selected is not part
    assert len(calls.validate) == 1
    assert len(selected.joint_route_diagnostics["search_stages"]) == 1
    assert selected.joint_route_diagnostics["search_mode"] == "progressive"
    assert selected.joint_route_diagnostics["jump_optimality_proven"] is False


def test_cheap_field_rejections_leave_body_slot_for_later_candidate(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    distinct_shaped_proposals(monkeypatch)

    def decide(part, path, _config, **kwargs):
        calls.validate.append((part, path, kwargs))
        if len(calls.validate) < 4:
            return GroupRouteValidation(False, ("pointwise_field_delta_budget",), {"body_preview_started": False})
        return GroupRouteValidation(True, (), {"body_preview_started": True}, prepared_for(part, path))

    monkeypatch.setattr(adoption, "validate_group_route_candidate", decide)
    original = make_part()
    selected = adoption.improve_group_routes([original], CoilGenConfig(joint_group_routing_max_candidates=1), target)[0]
    assert selected is not original
    diagnostics = selected.joint_route_diagnostics
    assert diagnostics["field_candidate_count"] == 4
    assert diagnostics["body_preview_candidate_count"] == diagnostics["candidate_count"] == 1
    assert len({row[2]["maximum_field_delta_t_per_a"] for row in calls.validate}) == 1
    assert all(row[2]["current_a"] == original.contour_step for row in calls.validate)


def test_field_screen_work_cap_is_distinct_from_body_and_physical_field_limits(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    distinct_shaped_proposals(monkeypatch)
    real_limits = adoption.GroupRouteValidationLimits
    monkeypatch.setattr(adoption, "GroupRouteValidationLimits", lambda: real_limits(maximum_field_candidates=4))

    def reject(part, path, _config, **kwargs):
        calls.validate.append((part, path, kwargs))
        return GroupRouteValidation(False, ("pointwise_field_delta_budget",), {"body_preview_started": False})

    monkeypatch.setattr(adoption, "validate_group_route_candidate", reject)
    original = make_part()
    before = state_snapshot(original)
    selected = adoption.improve_group_routes([original], CoilGenConfig(joint_group_routing_max_candidates=1), target)[0]
    assert selected is original and state_snapshot(original) == before
    diagnostics = selected.joint_route_diagnostics
    assert diagnostics["field_candidate_count"] == len(calls.validate) == 4
    assert diagnostics["body_preview_candidate_count"] == diagnostics["candidate_count"] == 0
    assert diagnostics["field_screen_budget_exhausted"]
    assert "field_screen_candidate_budget_exhausted" in diagnostics["work_budget_exhaustion_reasons"]
    assert diagnostics["work_budget_exhausted"] and not diagnostics["infeasibility_proven"]
    assert not diagnostics.get("physical_candidate_budget_exhausted", False)
    assert len({row[2]["maximum_field_delta_t_per_a"] for row in calls.validate}) == 1


def test_progressive_shape_failure_expands_without_sweeping_failed_geometry(setup_adoption, monkeypatch):
    target, calls, _ = setup_adoption
    stage = SimpleNamespace(level=0)
    monkeypatch.setattr(
        adoption,
        "JointTransitionEvaluator",
        lambda *_a, **_k: SimpleNamespace(set_search_level=lambda level: setattr(stage, "level", level)),
    )
    old_shape = adoption.shape_joint_group_route
    monkeypatch.setattr(
        adoption,
        "shape_joint_group_route",
        lambda *args, **kwargs: (
            GroupRouteProposals(reason="no_supported_transition") if stage.level == 0 else old_shape(*args, **kwargs)
        ),
    )
    part = make_part()
    adoption.improve_group_routes([part], CoilGenConfig(), target)
    assert stage.level == 1
    assert [row["physical_candidate_budget"] for row in part.joint_route_diagnostics["search_stages"]] == [0, 1]
    assert len(calls.validate) == 1
    assert len(calls.baseline) == 1


@pytest.mark.parametrize("policy,expected_cut_calls", [("regular", 2), ("physical", 1)])
def test_failed_transition_drives_bounded_port_context_search_without_changing_body_budget(
    setup_adoption,
    monkeypatch,
    policy,
    expected_cut_calls,
):
    target, calls, _ = setup_adoption
    stage = SimpleNamespace(level=0)
    monkeypatch.setattr(
        adoption,
        "JointTransitionEvaluator",
        lambda *_a, **_k: SimpleNamespace(set_search_level=lambda level: setattr(stage, "level", level)),
    )
    cut_calls = []

    def cuts(part, _config, **kwargs):
        cut_calls.append(kwargs)
        alternate = deepcopy(part)
        alternate.groups[0].opened_loops[0].v[0] += [0.001, 0, 0]
        return SimpleNamespace(
            variants=[alternate] if kwargs.get("failure_witnesses") else [],
            reasons=(),
            diagnostics={},
        )

    monkeypatch.setattr(adoption, "propose_joint_cut_variants", cuts)
    old_shape = adoption.shape_joint_group_route
    witness = dict(
        endpoints_m=[[0.0, 0, 0], [0.01, 0, 0]],
        owner={"group_index": 0},
        tangent_before=[1.0, 0, 0],
        tangent_after=[1.0, 0, 0],
        reasons=["insufficient_bend_radius"],
    )
    monkeypatch.setattr(
        adoption,
        "shape_joint_group_route",
        lambda *args, **kwargs: (
            GroupRouteProposals(reason="no_supported_transition", diagnostics={"failed_transition_witness": witness})
            if stage.level == 0
            else old_shape(*args, **kwargs)
        ),
    )
    part = make_part()
    before = state_snapshot(part)
    config = CoilGenConfig(joint_group_routing_selection_policy=policy, joint_group_routing_max_candidates=3)
    adoption.improve_group_routes([part], config, target)
    assert len(cut_calls) == expected_cut_calls
    assert len(calls.validate) <= config.joint_group_routing_max_candidates
    assert state_snapshot(part) == before
    assert part.joint_route_diagnostics["infeasibility_proven"] is False
    if policy == "regular":
        assert cut_calls[1]["maximum_candidates"] <= 7
        assert all(value == witness for value in cut_calls[1]["failure_witnesses"])
        assert part.joint_route_diagnostics["failure_directed_cut_search"]["context_count_after_search"] == 2
    else:
        assert "failure_directed_cut_search" not in part.joint_route_diagnostics


def test_progressive_uses_budget_only_for_distinct_validation_inputs(setup_adoption):
    target, calls, _ = setup_adoption
    part = make_part()
    adoption.improve_group_routes([part], CoilGenConfig(), target)
    assert len(calls.validate) == 1
    assert part.joint_route_diagnostics["candidate_count"] == 1
    assert part.joint_route_diagnostics["duplicate_candidate_count"] > 0
    assert [row["physical_candidate_budget"] for row in part.joint_route_diagnostics["search_stages"]] == [1, 0]


def test_candidate_fingerprint_preserves_cut_and_interval_semantics():
    part = make_part()
    candidate = candidate_for(part)
    original = adoption._candidate_fingerprint("existing", candidate)
    assert adoption._candidate_fingerprint("existing", deepcopy(candidate)) == original
    assert adoption._candidate_fingerprint("other_cut", candidate) != original
    candidate.path.segment_owners[0] = WireSegmentOwner("within_group_return")
    assert adoption._candidate_fingerprint("existing", candidate) != original
    candidate = candidate_for(part)
    candidate.segment_ids = ((0, 0, 0), *candidate.segment_ids[1:])
    assert adoption._candidate_fingerprint("existing", candidate) != original


def test_regular_topology_search_keeps_order_diversity_and_all_pending_proposals():
    candidates = [
        SimpleNamespace(group_sequence=(0, 2, 1, 3), variant_names=("in",) * 4),
        SimpleNamespace(group_sequence=(0, 2, 1, 3), variant_names=("out",) * 4),
        SimpleNamespace(group_sequence=(0, 1, 2, 3), variant_names=("in",) * 4),
    ]
    assert adoption._topology_search_order(candidates)[0] == 0
    assert adoption._topology_search_order(candidates, attempted_indices=(0,)) == [2, 1]


def test_regular_topology_search_softly_avoids_witnessed_directed_cycle_return():
    candidates = [
        SimpleNamespace(group_sequence=(0, 2, 1, 3)),
        SimpleNamespace(group_sequence=(0, 2, 1, 3)),
        SimpleNamespace(group_sequence=(0, 1, 2, 3)),
    ]
    witness = {
        "owner": {
            "kind": "inter_group_connector",
            "connector_role": "joint_return",
            "group1_index": 0,
            "group2_index": 2,
        }
    }
    ordered = adoption._topology_search_order(candidates, failure_witnesses=(witness,))
    assert ordered == [2, 0, 1]
    assert sorted(ordered) == list(range(len(candidates)))
    # A failure in the closing directed edge is considered, too. Its reversed
    # direction is not automatically excluded or declared infeasible.
    reverse = SimpleNamespace(group_sequence=(0, 3, 1, 2))
    assert adoption._topology_search_order([reverse, candidates[2]], failure_witnesses=(witness,)) == [0, 1]
    closing = {"owner": {**witness["owner"], "group1_index": 2, "group2_index": 0}}
    assert adoption._topology_search_order([reverse, candidates[2]], failure_witnesses=(closing,)) == [1, 0]


@pytest.mark.parametrize(
    "reason",
    [
        "pointwise_field_delta_budget",
        "field_delta_budget_exceeded",
        "regularity_field_rms_budget_exceeded",
        "physical_candidate_budget_exhausted",
    ],
)
def test_physical_field_budget_is_not_computational_work_exhaustion(reason):
    assert not adoption._work_budget_reason(reason)


@pytest.mark.parametrize(
    "reason",
    [
        "joint_shape_trial_budget_exceeded",
        "joint_spatial_pair_budget_exceeded",
        "connector_evaluation_budget",
        "expanded_state_budget",
        "search_budget_exhausted",
        "transition_work_budget_exhausted",
    ],
)
def test_computational_work_budget_exhaustion_is_reported(reason):
    assert adoption._work_budget_reason(reason)
