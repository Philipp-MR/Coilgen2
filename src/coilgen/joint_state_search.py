"""Bounded directed group-state search with exact small-graph enumeration.

Costs describe an additive graph only. Remote conductor conflicts and fields
remain the responsibility of independent physical validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from .routing_objective import search_cost_key


class RoutingEvaluationBudgetExceeded(RuntimeError):
    """A declared geometry or graph work budget has been exhausted."""


@dataclass
class StateSearchResult:
    circuits: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _key(item):
    return search_cost_key(item[0]), item[1]


def _enumeration_work(sizes, limit):
    """Count every prefix edge and closing edge, without visiting the graph."""
    elementary = [1]
    for size in sizes[1:]:
        elementary.append(0)
        for index in range(len(elementary) - 1, 0, -1):
            elementary[index] += size * elementary[index - 1]
    total = sizes[0] * math.factorial(len(sizes) - 1) * elementary[-1]
    for depth in range(1, len(sizes)):
        total += sizes[0] * math.factorial(depth) * elementary[depth]
        if total > limit:
            return limit + 1
    return total


def search_group_states(costs, transition_cost, *, beam_width, candidate_count, maximum_expanded_states):
    """Return the best complete circuits found in the declared additive graph.

    Small graphs are completely enumerated whenever their worst-case attempted
    edge count fits the budget, independently of beam width. Larger searches
    seed complete circuits with exact fixed-order state DP, then explore orders
    by a bounded beam and re-optimize states for its selected orders.
    """
    count = len(costs)
    sizes = [len(values) for values in costs]
    complete = []
    completed = expanded = pruned = starts_completed = dp_orders = 0
    seen_complete = set()
    budget_reason = ""
    exact = _enumeration_work(sizes, maximum_expanded_states) <= maximum_expanded_states
    effective_width = beam_width

    def edge(first, second):
        nonlocal expanded
        if expanded >= maximum_expanded_states:
            raise RoutingEvaluationBudgetExceeded("expanded_state_budget")
        expanded += 1
        return transition_cost(first, second)

    def retain(cost, sequence):
        nonlocal completed
        if sequence in seen_complete:
            return
        # Only retained circuits need deduplication; counting a discarded
        # rediscovered circuit has no bearing on the optimality claim.
        completed += 1
        complete.append((cost, sequence))
        complete.sort(key=_key)
        del complete[candidate_count:]
        seen_complete.clear()
        seen_complete.update(sequence for _cost, sequence in complete)

    def close(cost, sequence):
        final = edge(sequence[-1], sequence[0])
        if math.isfinite(final):
            retain(cost + final, sequence)

    def enumerate_from(cost, sequence, visited):
        if len(sequence) == count:
            close(cost, sequence)
            return
        for group, values in enumerate(costs):
            if visited & (1 << group):
                continue
            for state, own_cost in enumerate(values):
                following = group, state
                distance = edge(sequence[-1], following)
                if math.isfinite(distance):
                    enumerate_from(cost + distance + own_cost, sequence + (following,), visited | (1 << group))

    def optimize_order(order):
        """K-best cyclic state DP for one order, including each starting state."""
        nonlocal dp_orders
        for initial in range(sizes[order[0]]):
            first = order[0], initial
            states = {initial: [(costs[first[0]][initial], (first,))]}
            for group in order[1:]:
                following_states = {}
                for state, own_cost in enumerate(costs[group]):
                    options = []
                    for paths in states.values():
                        for cost, sequence in paths:
                            distance = edge(sequence[-1], (group, state))
                            if math.isfinite(distance):
                                options.append((cost + distance + own_cost, sequence + ((group, state),)))
                    if options:
                        following_states[state] = sorted(options, key=_key)[:candidate_count]
                states = following_states
                if not states:
                    break
            for paths in states.values():
                for cost, sequence in paths:
                    if len(sequence) == count:
                        close(cost, sequence)
        dp_orders += 1

    try:
        if exact:
            for state, own_cost in enumerate(costs[0]):
                enumerate_from(own_cost, ((0, state),), 1)
                starts_completed += 1
        else:
            orders_seen = set()
            for order in (tuple(range(count)), (0, *range(count - 1, 0, -1))):
                if order not in orders_seen:
                    optimize_order(order)
                    orders_seen.add(order)
            # Reserve one quarter of the remaining work for fixed-order DP
            # refinement; order search must not consume the entire allowance.
            work_per_beam = max(sizes) * count * (count - 1) // 2 + 1
            remaining = max(1, (maximum_expanded_states - expanded) * 3 // 4)
            effective_width = min(beam_width, max(1, remaining // (sizes[0] * work_per_beam)))
            for initial, own_cost in enumerate(costs[0]):
                first = 0, initial
                beam = [(own_cost, (first,), 1)]
                for _depth in range(1, count):
                    options = []
                    for cost, sequence, visited in beam:
                        for group, values in enumerate(costs):
                            if visited & (1 << group):
                                continue
                            for state, state_cost in enumerate(values):
                                following = group, state
                                distance = edge(sequence[-1], following)
                                if math.isfinite(distance):
                                    options.append(
                                        (cost + distance + state_cost, sequence + (following,), visited | (1 << group))
                                    )
                    pruned += max(0, len(options) - effective_width)
                    beam = sorted(options, key=lambda item: _key(item[:2]))[:effective_width]
                    if not beam:
                        break
                for cost, sequence, _visited in beam:
                    if len(sequence) == count:
                        close(cost, sequence)
                starts_completed += 1
            for _cost, sequence in tuple(complete):
                order = tuple(group for group, _state in sequence)
                if order not in orders_seen:
                    optimize_order(order)
                    orders_seen.add(order)
    except RoutingEvaluationBudgetExceeded as exc:
        budget_reason = str(exc) or "transition_evaluation_budget_exceeded"

    exhaustive = exact and not budget_reason
    return StateSearchResult(
        complete,
        {
            "search_method": "complete_small_graph" if exact else "beam_orders_with_fixed_order_state_dp",
            "expanded_states": expanded,
            "completed_states": completed,
            "effective_beam_width": effective_width,
            "pruned_states": pruned,
            "starting_variants_completed": starts_completed,
            "fixed_orders_state_optimized": dp_orders,
            "budget_exhausted": bool(budget_reason),
            "budget_exhaustion_reason": budget_reason,
            "port_graph_search_exhaustive": exhaustive,
            "port_graph_infeasibility_proven": exhaustive and not complete,
            "objective_optimality_proven": exhaustive and bool(complete),
            "optimality_scope": "enumerated_additive_state_graph_at_objective_resolution",
            "physical_feasibility_proven": False,
        },
    )
