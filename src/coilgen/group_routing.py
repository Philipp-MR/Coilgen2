"""Bounded proposals for joining directed winding groups through their returns.

This module proposes one closed series circuit. It does not certify a conductor
or change a live CoilPart; adoption requires the independent route validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import numpy as np

from .models import CoilPart, InterGroupCut, WirePath, WirePathGeometry, WireSegmentOwner
from .joint_state_search import RoutingEvaluationBudgetExceeded, search_group_states
from .routing_objective import (
    RoutingSearchCost,
    search_cost_key,
    routing_selection_policy,
    regularity_order_key,
    transition_regularity,
    REGULARITY_NAMES,
)


SegmentId = tuple[int, int, int]


@dataclass(frozen=True)
class GroupRoutingLimits:
    """Deterministic admission and work limits for the directed port search.

    Defaults admit 64 groups and keep at most 48 partial circuits per starting
    variant, reduced automatically for large group sets. The 250,000 expansion
    cap counts attempted transitions including closure; 50,000 caps uncached
    connector solves. Six complete proposals may advance to physical checks.
    These limits bound this search, not the mathematical feasible design set.
    """

    maximum_groups: int = 64
    beam_width: int = 48
    candidate_count: int = 6
    return_fractions: tuple[float, ...] = (0.0, 0.4, 0.75, 0.9)
    maximum_expanded_states: int = 250_000
    maximum_connector_evaluations: int = 50_000

    def __post_init__(self):
        budgets = (
            self.maximum_groups,
            self.beam_width,
            self.candidate_count,
            self.maximum_expanded_states,
            self.maximum_connector_evaluations,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in budgets):
            raise ValueError("Group routing budgets must be positive integers.")
        if not self.return_fractions or any(not np.isfinite(f) or not 0 <= f < 1 for f in self.return_fractions):
            raise ValueError("Return fractions must be finite and in [0, 1).")


@dataclass
class GroupRouteCandidate:
    path: WirePath
    segment_ids: tuple[SegmentId | None, ...]
    protected_segments: dict[SegmentId, np.ndarray]
    group_sequence: tuple[int, ...]
    variant_names: tuple[str, ...]
    cuts: list[InterGroupCut]
    length_m: float
    retained_return_length_m: float


@dataclass
class GroupRouteProposals:
    candidates: list[GroupRouteCandidate] = field(default_factory=list)
    reason: str = "no_candidate"
    diagnostics: dict = field(default_factory=dict)


@dataclass
class _GroupVariant:
    group_index: int
    name: str
    path: WirePath
    segment_ids: tuple[SegmentId | None, ...]
    length_m: float
    return_length_m: float
    routing_cost_m: float | RoutingSearchCost


def _length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _edge_key(a, b):
    return tuple(a), tuple(b)


def _evaluate_transition(evaluator, points, tangent_before, tangent_after):
    evaluate = getattr(evaluator, "search_cost", evaluator)
    value = evaluate(points, tangent_before, tangent_after)
    if isinstance(value, RoutingSearchCost):
        return value
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError("Transition evaluator must return a nonnegative scalar cost or positive infinity.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Transition evaluator must return a nonnegative scalar cost or positive infinity.") from exc
    if math.isnan(value) or value < 0:
        raise ValueError("Transition evaluator must return a nonnegative scalar cost or positive infinity.")
    return value


def _mutable_cost(points, identities, b0_direction, b0_weight, transition_evaluator=None, *, policy="physical"):
    """Charge alignment only to mutable intervals; protected turns stay fixed."""
    from .routing_objective import routing_cost

    cost = RoutingSearchCost(0, 0.0, (0.0,) * len(REGULARITY_NAMES)) if policy == "regular" else 0.0
    start = 0
    while start < len(identities):
        mutable = identities[start] is None
        end = start + 1
        while end < len(identities) and (identities[end] is None) == mutable:
            end += 1
        section = points[start : end + 1]
        if mutable and transition_evaluator is not None:
            if start == 0 or end == len(identities):
                raise ValueError("A group state must expose protected directed intervals at both ports.")
            mutable_cost = _evaluate_transition(
                transition_evaluator, section, points[start] - points[start - 1], points[end + 1] - points[end]
            )
            if not math.isfinite(mutable_cost):
                return math.inf
            cost += mutable_cost
        else:
            guide = routing_cost(section, b0_direction, b0_weight) if mutable else _length(section)
            if mutable and policy == "regular":
                measured = transition_regularity(
                    section,
                    points[start] - points[start - 1] if start else None,
                    points[end + 1] - points[end] if end < len(identities) else None,
                )
                guide = RoutingSearchCost(0, guide, regularity_order_key(measured))
            cost += guide
        start = end
    return cost


def _cost_key(cost):
    # Ignore sub-picometre floating-point changes when resolving geometrically
    # tied paths, so rigid rotations do not arbitrarily change their ordering.
    return search_cost_key(cost)


def _prefix(points, fraction, step):
    """Truncate a return by arclength, retaining every intervening corner."""
    points = np.asarray(points, dtype=float)
    if fraction == 0 or len(points) < 2:
        return points[:1].copy()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    stop = fraction * float(lengths.sum())
    result = [points[0].copy()]
    traversed = 0.0
    for index, length in enumerate(lengths):
        if length <= 1e-14:
            continue
        travel = min(length, stop - traversed)
        if travel <= 1e-14:
            break
        end = points[index] + (travel / length) * (points[index + 1] - points[index])
        count = max(1, int(math.ceil(travel / step)))
        result.extend(np.linspace(points[index], end, count + 1)[1:])
        traversed += travel
        if traversed >= stop - 1e-14:
            break
    return np.asarray(result)


def _protected_group_segments(part):
    protected = {}
    lookup = {}
    for group_index, group in enumerate(part.groups):
        if not group.opened_loops or len(group.opened_loops) != len(group.loops):
            raise ValueError("Every group must expose its original opened contours.")
        for local_index, loop in enumerate(group.opened_loops):
            if loop.turn_multiplicity != 1 or group.loops[local_index].turn_multiplicity != 1:
                raise ValueError("Weighted contours require explicit physical turn expansion.")
            if len(loop.v) < 2 or not np.all(np.isfinite(loop.v)):
                raise ValueError("Opened contours must contain finite directed intervals.")
            loop_index = group.loop_indices[local_index] if local_index < len(group.loop_indices) else local_index
            for segment_index, (a, b) in enumerate(zip(loop.v[:-1], loop.v[1:])):
                if np.linalg.norm(b - a) <= 1e-14:
                    raise ValueError("Degenerate protected contour interval.")
                key = _edge_key(a, b)
                if key in lookup:
                    raise ValueError("Coincident protected intervals require unambiguous source provenance.")
                identity = group_index, int(loop_index), segment_index
                if identity in protected:
                    raise ValueError("Duplicate protected contour identity.")
                protected[identity] = np.asarray([a, b]).copy()
                lookup[key] = identity
    return protected, lookup


class _UnsupportedGroupState(ValueError):
    pass


def _group_variants(
    part,
    config,
    limits,
    protected,
    lookup,
    b0_direction,
    b0_weight,
    transition_evaluator=None,
    progress_callback=None,
):
    from .layout import _group_route_variants
    from .manufacturing import conductor_cross_section_points, _cross_section_envelope_radius

    envelope = _cross_section_envelope_radius(conductor_cross_section_points(config))
    step = max(envelope, abs(float(config.normal_shift_length)) / 4, 1e-6)
    groups = []
    planned_bases = None
    completed = 0
    if progress_callback is not None:
        # Planning reads the same directed paths; it never scores an extra
        # transition or changes the order in which candidates are evaluated.
        planned_bases = [_group_route_variants(group) for group in part.connected_groups]
        for index, bases in enumerate(planned_bases):
            if len(part.groups[index].loops) == 1:
                planned_bases[index] = bases[:1]
        total = sum(map(len, planned_bases)) * len(set(limits.return_fractions))
        progress_callback({"phase": "winding states", "completed": 0, "total": total})
    for group_index, group in enumerate(part.connected_groups):
        required = {identity for identity in protected if identity[0] == group_index}
        variants = []
        unsupported = False
        bases = _group_route_variants(group) if planned_bases is None else planned_bases[group_index]
        # A cut singleton has one directed arc, not two different radial
        # winding orders. Reversing its provisional chord creates no new ports.
        if len(part.groups[group_index].loops) == 1:
            bases = bases[:1]
        for base in bases:
            spiral = np.asarray(base.spiral_v, dtype=float)
            identities = tuple(lookup.get(_edge_key(a, b)) for a, b in zip(spiral[:-1], spiral[1:]))
            present = [identity for identity in identities if identity is not None]
            if set(present) != required or len(present) != len(required):
                if progress_callback is not None:
                    completed += len(set(limits.return_fractions))
                    progress_callback(
                        {
                            "phase": "winding states",
                            "completed": completed,
                            "total": total,
                            "message": f"group {group_index + 1}/{len(part.connected_groups)}: rejected provenance",
                        }
                    )
                continue  # Never infer or reverse missing contour geometry.
            spiral_owners = [
                WireSegmentOwner("contour_loop", group_index=group_index, loop_index=identity[1])
                if identity is not None
                else WireSegmentOwner("contour_group", group_index=group_index, connector_role="joint_spiral_join")
                for identity in identities
            ]
            return_points = np.asarray(base.return_path_v, dtype=float)
            if np.linalg.norm(spiral[-1] - return_points[0]) > 1e-12:
                return_points = np.vstack((spiral[-1], return_points))
            else:
                return_points = return_points.copy()
                return_points[0] = spiral[-1]
            for fraction in dict.fromkeys(limits.return_fractions):
                returning = _prefix(return_points, fraction, step)
                points = np.vstack((spiral, returning[1:]))
                if np.linalg.norm(points[-1] - points[0]) < 2 * envelope:
                    if progress_callback is not None:
                        completed += 1
                        progress_callback(
                            {
                                "phase": "winding states",
                                "completed": completed,
                                "total": total,
                                "message": f"group {group_index + 1}/{len(part.connected_groups)}: skipped closed port",
                            }
                        )
                    continue  # A closed local group cannot also be a series port.
                count = len(returning) - 1
                path_ids = identities + (None,) * count
                owners = spiral_owners + [WireSegmentOwner("within_group_return", group_index=group_index)] * count
                if transition_evaluator is not None and (path_ids[0] is None or path_ids[-1] is None):
                    raise ValueError("A group state must expose protected directed intervals at both ports.")
                cost = _mutable_cost(
                    points,
                    path_ids,
                    b0_direction,
                    b0_weight,
                    transition_evaluator,
                    policy=routing_selection_policy(config),
                )
                if progress_callback is not None:
                    completed += 1
                    progress_callback(
                        {
                            "phase": "winding states",
                            "completed": completed,
                            "total": total,
                            "message": f"group {group_index + 1}/{len(part.connected_groups)}: {base.name}",
                        }
                    )
                if not math.isfinite(cost):
                    unsupported = True
                    continue
                variants.append(
                    _GroupVariant(
                        group_index,
                        f"{base.name}:return_{fraction:g}",
                        WirePath(points, owners, [WirePathGeometry.SURFACE] * (len(points) - 1)),
                        path_ids,
                        _length(points),
                        _length(returning),
                        cost,
                    )
                )
        if not variants:
            if unsupported:
                raise _UnsupportedGroupState(f"Group {group_index} has no state supported by the transition family.")
            raise ValueError(f"Group {group_index} has no complete directed open variant.")
        groups.append(variants)
    if progress_callback is not None:
        progress_callback({"phase": "winding states", "status": "finished"})
    return groups


def propose_joint_group_routes(
    part: CoilPart,
    config,
    *,
    limits: GroupRoutingLimits | None = None,
    b0_direction=None,
    b0_weight: float | None = None,
    transition_evaluator=None,
    progress_callback=None,
):
    """Enumerate bounded whole-circuit proposals without touching live geometry.

    Existing within-group cuts and contour directions stay fixed. Search varies
    inward/outward winding order, the retained return prefix, and group order.
    Geometry and electromagnetic acceptance are intentionally separate. The
    B0 alignment cost applies to mutable joins/returns only. Zero weight gives
    the original length objective. Beam pruning or a work limit never proves
    that no feasible physical route exists.

    An optional evaluator scores whole shaped mutable spans from their exact
    directed endpoint tangents. In that mode return fractions must be zero:
    the final transition replaces the local return and inter-group bridge.
    Small state graphs are fully enumerated within the declared work budget.
    """
    from .layout import (
        _inter_group_connector_result,
        _strict_geodesic_inter_group_paths,
        _use_geodesic_inter_group_paths,
    )
    from .surface_distance import SurfaceGeodesicSolver
    from .routing_objective import routing_cost

    limits = limits or GroupRoutingLimits()
    count = len(part.connected_groups)
    result = GroupRouteProposals(
        diagnostics={
            "group_count": count,
            "beam_width": limits.beam_width,
            "maximum_groups": limits.maximum_groups,
            "maximum_expanded_states": limits.maximum_expanded_states,
            "maximum_connector_evaluations": limits.maximum_connector_evaluations,
            "fixed_cuts": True,
            "cut_relocation_supported": False,
            "manufacturing_feasibility_proven": False,
            "physical_feasibility_proven": False,
            "transition_cost_basis": "shaped_whole_transition" if transition_evaluator is not None else "raw_polyline",
        }
    )
    if transition_evaluator is not None and any(fraction != 0 for fraction in limits.return_fractions):
        result.reason = "transition_evaluator_requires_zero_return_fraction"
        return result
    if not 1 <= count <= limits.maximum_groups or len(part.groups) != count:
        result.reason = "unsupported_group_count_or_missing_groups"
        result.diagnostics["admission_reason"] = (
            "group_count_budget_exceeded" if count > limits.maximum_groups else "insufficient_or_missing_groups"
        )
        return result
    if count == 1 and len(part.groups[0].loops) == 1 and len(part.connected_groups[0].return_path_v) < 2:
        result.reason = "single_closed_loop_has_no_free_ports"
        return result
    if any(
        len(group.loops) == 1 and len(connected.return_path_v) < 2
        for group, connected in zip(part.groups, part.connected_groups)
    ):
        result.reason = "singleton_cut_context_required"
        return result
    if b0_direction is None:
        b0_direction = getattr(config, "b_0_direction", (0.0, 0.0, 1.0))
    if b0_weight is None:
        b0_weight = getattr(config, "joint_group_routing_b0_weight", 1.0)
    # Validate the objective even if there is no mutable interval in a group.
    # This also keeps invalid context from being accepted through a degenerate
    # candidate or hidden by an exhausted search budget.
    try:
        routing_cost(np.zeros((2, 3)), b0_direction, b0_weight)
    except (ValueError, TypeError) as exc:
        result.reason = "invalid_routing_objective"
        result.diagnostics["detail"] = str(exc)
        return result
    try:
        protected, lookup = _protected_group_segments(part)
        variants = _group_variants(
            part,
            config,
            limits,
            protected,
            lookup,
            b0_direction,
            b0_weight,
            transition_evaluator,
            progress_callback,
        )
    except RoutingEvaluationBudgetExceeded as exc:
        if progress_callback is not None:
            progress_callback({"phase": "winding states", "status": "stopped", "message": str(exc)})
        result.reason = "search_budget_exhausted"
        result.diagnostics.update(
            budget_exhausted=True,
            budget_exhaustion_reason=str(exc) or "transition_evaluation_budget_exceeded",
            port_graph_search_exhaustive=False,
            objective_optimality_proven=False,
        )
        return result
    except _UnsupportedGroupState as exc:
        if progress_callback is not None:
            progress_callback({"phase": "winding states", "status": "stopped", "message": str(exc)})
        result.reason = "no_supported_group_state_in_transition_family"
        result.diagnostics.update(
            detail=str(exc), objective_optimality_proven=False, port_graph_search_exhaustive=False
        )
        return result
    except ValueError as exc:
        if progress_callback is not None:
            progress_callback({"phase": "winding states", "status": "stopped", "message": str(exc)})
        result.reason = "invalid_group_provenance"
        result.diagnostics["detail"] = str(exc)
        return result
    solver = (
        SurfaceGeodesicSolver(
            part.mesh,
            refine_routes=routing_selection_policy(config) == "regular",
        )
        if _use_geodesic_inter_group_paths(config)
        else None
    )
    strict = _strict_geodesic_inter_group_paths(config)
    connections = {}
    possible_connections = (
        len(variants[0])
        if count == 1
        else sum(
            len(first) * len(second) for i, first in enumerate(variants) for j, second in enumerate(variants) if i != j
        )
    )
    connection_total = min(possible_connections, limits.maximum_connector_evaluations)
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "port connections",
                "completed": 0,
                "total": connection_total,
                "upper_bound": True,
            }
        )

    def connector(first, second):
        key = first, second
        if key not in connections:
            if len(connections) >= limits.maximum_connector_evaluations:
                raise RoutingEvaluationBudgetExceeded("connector_evaluation_budget")
            a = variants[first[0]][first[1]].path.v[-1]
            b = variants[second[0]][second[1]].path.v[0]
            route = _inter_group_connector_result(a, b, solver, allow_direct_fallback=not strict)
            points = np.asarray(route.path_v, dtype=float)
            valid = len(points) >= 2 and np.all(np.isfinite(points)) and np.linalg.norm(b - a) > 1e-12
            if valid:
                points = points.copy()
                points[[0, -1]] = a, b
            cost = math.inf
            if valid:
                if transition_evaluator is None:
                    cost = routing_cost(points, b0_direction, b0_weight)
                    if routing_selection_policy(config) == "regular":
                        first_points = variants[first[0]][first[1]].path.v
                        second_points = variants[second[0]][second[1]].path.v
                        measured = transition_regularity(
                            points,
                            first_points[-1] - first_points[-2],
                            second_points[1] - second_points[0],
                        )
                        cost = RoutingSearchCost(0, cost, regularity_order_key(measured))
                else:
                    first_points = variants[first[0]][first[1]].path.v
                    second_points = variants[second[0]][second[1]].path.v
                    cost = _evaluate_transition(
                        transition_evaluator,
                        points,
                        first_points[-1] - first_points[-2],
                        second_points[1] - second_points[0],
                    )
            connections[key] = (points, cost, route.used_direct_fallback)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "port connections",
                        "completed": len(connections),
                        "total": connection_total,
                        "upper_bound": True,
                    }
                )
        return connections[key]

    # The same bounded engine handles geometric guides and shaped jump costs.
    # Closure participates in exact small-graph enumeration and state DP alike.
    try:
        searched = search_group_states(
            [[variant.routing_cost_m for variant in choices] for choices in variants],
            lambda first, second: connector(first, second)[1],
            beam_width=limits.beam_width,
            candidate_count=limits.candidate_count,
            maximum_expanded_states=limits.maximum_expanded_states,
        )
    except ValueError as exc:
        if progress_callback is not None:
            progress_callback({"phase": "port connections", "status": "stopped", "message": str(exc)})
        result.reason = "invalid_transition_evaluation"
        result.diagnostics["detail"] = str(exc)
        return result
    search_diagnostics = searched.diagnostics
    complete = searched.circuits
    budget_reason = search_diagnostics["budget_exhaustion_reason"]
    if progress_callback is not None:
        progress_callback(
            {
                "phase": "port connections",
                "status": "stopped" if budget_reason else "finished",
                "message": budget_reason or "connection search finished; unused connections need no evaluation",
            }
        )
    complete.sort(key=lambda item: (_cost_key(item[0]), item[1]))
    # Avoid spending every expensive validation on the same group order and
    # spiral directions with tiny changes to only the retained return length.
    buckets = {}
    for cost, sequence in complete:
        signature = tuple((g, variants[g][v].name.split(":")[0]) for g, v in sequence)
        buckets.setdefault(signature, []).append((cost, sequence))
    finalists = []
    while buckets and len(finalists) < limits.candidate_count:
        heads = sorted((_cost_key(values[0][0]), key) for key, values in buckets.items())
        for _cost, key in heads:
            finalists.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
            if len(finalists) == limits.candidate_count:
                break
    for _cost, sequence in finalists:
        sections, owners, geometry, identities, cuts = [], [], [], [], []
        retained_length = 0.0
        for position, selected in enumerate(sequence):
            group_index, variant_index = selected
            variant = variants[group_index][variant_index]
            sections.append(variant.path.v if position == 0 else variant.path.v[1:])
            owners.extend(variant.path.segment_owners)
            geometry.extend(variant.path.segment_geometry)
            identities.extend(variant.segment_ids)
            retained_length += variant.return_length_m
            following = sequence[(position + 1) % count]
            path, _distance, fallback = connector(selected, following)
            sections.append(path[1:])
            owner = WireSegmentOwner(
                "inter_group_connector",
                connector_index=position,
                group1_index=group_index,
                group2_index=following[0],
                connector_role="joint_return",
            )
            owners.extend([owner] * (len(path) - 1))
            geometry.extend(
                [WirePathGeometry.SURFACE if solver is not None and not fallback else WirePathGeometry.FREE_3D]
                * (len(path) - 1)
            )
            identities.extend([None] * (len(path) - 1))
            cuts.append(
                InterGroupCut(
                    cut1_v=variant.path.v[-2:].copy(),
                    cut2_v=variants[following[0]][following[1]].path.v[:2].copy(),
                    connection_path_v=path.copy(),
                    group1_index=group_index,
                    group2_index=following[0],
                    selected_point1_v=path[:1].copy(),
                    selected_point2_v=path[-1:].copy(),
                    direct_distance=float(np.linalg.norm(path[-1] - path[0])),
                    connector_distance=_length(path),
                    geodesic_distance=_length(path) if solver is not None and not fallback else math.inf,
                    connection_method="geodesic" if solver is not None and not fallback else "direct_3d",
                    used_direct_fallback=fallback,
                    is_return_connection=True,
                    group1_variant_name=variant.name,
                    group2_variant_name=variants[following[0]][following[1]].name,
                )
            )
        path = WirePath(np.vstack(sections), owners, geometry)
        result.candidates.append(
            GroupRouteCandidate(
                path,
                tuple(identities),
                protected,
                tuple(g for g, _v in sequence),
                tuple(variants[g][v].name for g, v in sequence),
                cuts,
                _length(path.v),
                retained_length,
            )
        )
    if result.candidates:
        result.reason = "proposals_require_validation"
    elif budget_reason:
        result.reason = "search_budget_exhausted"
    elif search_diagnostics["pruned_states"]:
        result.reason = "no_closed_connection_within_search_limits"
    else:
        result.reason = "no_closed_connection"
    result.diagnostics.update(
        variant_counts=[len(values) for values in variants],
        cached_connections=len(connections),
        candidate_count=len(result.candidates),
        bounded_search=True,
        objective=(
            "jump_count_then_geometric_guide"
            if any(isinstance(cost, RoutingSearchCost) for cost, _sequence in finalists)
            else "length_plus_mutable_transverse_b0_cost"
        ),
        objective_comparison_resolution_m=1e-12,
        b0_direction=np.asarray(b0_direction, dtype=float).tolist(),
        b0_weight=float(b0_weight),
        candidate_objective_costs_m=[float(cost) for cost, _sequence in finalists],
        candidate_jump_counts=[cost.jumps if isinstance(cost, RoutingSearchCost) else None for cost, _ in finalists],
        protected_interval_count=len(protected),
    )
    result.diagnostics.update(search_diagnostics)
    return result


def _diverse_topology_indices(candidates, count, symmetry_orders=()):
    """Keep inexpensive leaders and opposite winding/order representatives.

    Diversity concerns graph states, not physical symmetry. In particular an
    opposite winding order retains every prescribed directed contour interval;
    it does not reverse the current carried by those intervals.
    """
    if not candidates:
        return []
    directions = [
        {group: name.split(":")[0] for group, name in zip(candidate.group_sequence, candidate.variant_names)}
        for candidate in candidates
    ]
    orders = [candidate.group_sequence for candidate in candidates]
    ports = [
        {
            (group, directions[index][group], following, directions[index][following])
            for group, following in zip(order, order[1:] + order[:1])
        }
        for index, order in enumerate(orders)
    ]
    selected = [0]
    remaining = set(range(1, len(candidates)))

    def winding_distance(first, second):
        return sum(directions[first][group] != directions[second][group] for group in directions[first])

    def add(index):
        if index in remaining and len(selected) < count:
            selected.append(index)
            remaining.remove(index)

    # Complementary winding choices can carry different return-field errors
    # despite equal protected turns. Do not spend the initial batch entirely
    # on small port-length differences in one winding state.
    if remaining and len(selected) < count:
        opposite = min(remaining, key=lambda index: (-winding_distance(0, index), index))
        if winding_distance(0, opposite):
            add(opposite)
    for order in symmetry_orders:
        counterparts = [index for index in remaining if orders[index] == order]
        if counterparts:
            add(min(counterparts))
    reverse_order = orders[0][:1] + orders[0][:0:-1]
    for reference in tuple(selected):
        reverse = [
            index
            for index in remaining
            if orders[index] == reverse_order and directions[index] == directions[reference]
        ]
        if reverse:
            add(min(reverse))
    while remaining and len(selected) < count:
        seen_orders = {orders[index] for index in selected}
        seen_winding = {tuple(sorted(directions[index].items())) for index in selected}
        seen_ports = set().union(*(ports[index] for index in selected))
        following = min(
            remaining,
            key=lambda index: (
                -(orders[index] not in seen_orders),
                -(tuple(sorted(directions[index].items())) not in seen_winding),
                -len(ports[index] - seen_ports),
                index,
            ),
        )
        add(following)
    return selected


def _verified_symmetry_orders(symmetry, order):
    """Priority hints only; cut ports and fields need not share the symmetry."""
    orders = []
    for transform in (symmetry or {}).get("transforms", []):
        permutation = transform.get("group_permutation")
        if transform.get("verified") is not True or permutation is None:
            continue
        if (
            len(permutation) != len(order)
            or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in permutation)
            or set(permutation) != set(order)
        ):
            continue
        mapped = tuple(int(permutation[group]) for group in order)
        start = mapped.index(order[0])
        mapped = mapped[start:] + mapped[:start]
        if mapped != order and mapped not in orders:
            orders.append(mapped)
    return orders


def propose_joint_group_topologies(
    part: CoilPart,
    config,
    *,
    limits: GroupRoutingLimits | None = None,
    b0_direction=None,
    b0_weight: float | None = None,
    progress_callback=None,
    symmetry: dict | None = None,
):
    """Rank a small diverse batch using raw ports, before shaping any curves.

    The cheap pool includes every topology for small two-state graphs with at
    most 256 circuits, otherwise at least eight times the requested batch (up
    to 256, unless the batch itself is larger). Existing graph/connector work
    limits still apply. Local returns are replaced as whole transitions, so
    this interface always exposes the zero-return states.

    No detailed transition evaluator is accepted or called. These geometric
    guides rank proposals only: the batch may miss a feasible or better route.
    Verified source-symmetry group permutations can prioritise existing order
    counterparts after the best/opposite winding choices. They never discard
    routes, reuse physical evaluations or imply electromagnetic invariance.
    """
    limits = limits or GroupRoutingLimits(return_fractions=(0.0,))
    group_count = len(part.connected_groups)
    circuit_count = math.prod(1 if len(group.loops) == 1 else 2 for group in part.groups)
    circuit_count *= math.factorial(max(0, group_count - 1))
    pool_count = max(limits.candidate_count, min(256, max(48, limits.candidate_count * 8)))
    if 1 <= group_count <= limits.maximum_groups and len(part.groups) == group_count and circuit_count <= 256:
        pool_count = max(limits.candidate_count, circuit_count)

    def report(event):
        if progress_callback is not None:
            event = dict(event)
            event["phase"] = "topology " + event["phase"]
            progress_callback(event)

    result = propose_joint_group_routes(
        part,
        config,
        limits=replace(limits, candidate_count=pool_count, return_fractions=(0.0,)),
        b0_direction=b0_direction,
        b0_weight=b0_weight,
        progress_callback=report if progress_callback is not None else None,
    )
    pool = result.candidates
    symmetry_orders = _verified_symmetry_orders(symmetry, pool[0].group_sequence) if pool else []
    selected = _diverse_topology_indices(pool, limits.candidate_count, symmetry_orders)
    costs = result.diagnostics.get("candidate_objective_costs_m", [])
    jumps = result.diagnostics.get("candidate_jump_counts", [])
    result.candidates = [pool[index] for index in selected]
    result.diagnostics.update(
        proposal_basis="cheap_complete_topology_then_directed_port_diversity",
        detailed_transition_evaluations=0,
        cheap_topology_pool_count=len(pool),
        cheap_topology_pool_limit=pool_count,
        candidate_count=len(selected),
        candidate_objective_costs_m=[costs[index] for index in selected],
        candidate_jump_counts=[jumps[index] for index in selected],
        candidate_cheap_rank=[index + 1 for index in selected],
        objective_optimality_proven=False,
        optimality_scope="bounded_geometric_topology_proposals_require_physical_validation",
        physical_feasibility_proven=False,
        symmetry_evidence_provided=symmetry is not None,
        symmetry_priority_group_orders=[list(order) for order in symmetry_orders],
        symmetry_priority_candidate_count=sum(pool[index].group_sequence in symmetry_orders for index in selected),
        symmetry_evaluation_reuse=False,
        symmetry_routes_discarded=0,
    )
    return result
