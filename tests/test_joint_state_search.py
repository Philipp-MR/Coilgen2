"""Independent exhaustive oracles for directed spiral-state optimization."""

import itertools
import math

import numpy as np
import pytest

from coilgen.joint_state_search import RoutingEvaluationBudgetExceeded, search_group_states


def _oracle(costs, transitions, orders=None):
    count = len(costs)
    if orders is None:
        orders = ((0, *tail) for tail in itertools.permutations(range(1, count)))
    candidates = []
    for order in orders:
        for choices in itertools.product(*(range(len(costs[group])) for group in order)):
            sequence = tuple(zip(order, choices))
            cost = sum(costs[group][state] for group, state in sequence)
            cost += sum(transitions[a, b] for a, b in zip(sequence, (*sequence[1:], sequence[0])))
            if math.isfinite(cost):
                candidates.append((cost, sequence))
    return sorted(candidates, key=lambda item: (round(item[0], 12), item[1]))


def _graph(count, seed=13):
    rng = np.random.default_rng(seed)
    costs = rng.uniform(0, 0.5, (count, 2)).tolist()
    transitions = {
        ((a, sa), (b, sb)): float(rng.uniform(0, 2))
        for a, b in itertools.product(range(count), repeat=2)
        if a != b or count == 1
        for sa, sb in itertools.product(range(2), repeat=2)
    }
    return costs, transitions


@pytest.mark.parametrize("seed", [0, 2, 13, 701])
def test_all_96_four_group_circuits_match_exhaustive_oracle_even_with_beam_one(seed):
    costs, transitions = _graph(4, seed)
    expected = _oracle(costs, transitions)
    result = search_group_states(
        costs, lambda a, b: transitions[a, b], beam_width=1, candidate_count=6, maximum_expanded_states=252
    )
    assert result.diagnostics["search_method"] == "complete_small_graph"
    assert result.diagnostics["expanded_states"] == 252
    assert result.diagnostics["completed_states"] == len(expected) == 96
    assert result.diagnostics["port_graph_search_exhaustive"]
    assert result.diagnostics["objective_optimality_proven"]
    assert not result.diagnostics["physical_feasibility_proven"]
    assert [sequence for _cost, sequence in result.circuits] == [sequence for _cost, sequence in expected[:6]]
    np.testing.assert_allclose([cost for cost, _ in result.circuits], [cost for cost, _ in expected[:6]], atol=1e-12)


def test_last_to_first_connection_changes_the_best_spiral_states():
    costs = [[0.0, 0.0]] * 3
    transitions = {
        ((a, sa), (b, sb)): (100.0 if b == 0 and sb == 0 else float(sa + sb))
        for a, b in itertools.permutations(range(3), 2)
        for sa, sb in itertools.product(range(2), repeat=2)
    }
    result = search_group_states(
        costs, lambda a, b: transitions[a, b], beam_width=1, candidate_count=3, maximum_expanded_states=1000
    )
    assert result.circuits[0][1][0] == (0, 1)
    assert result.circuits == _oracle(costs, transitions)[:3]


def test_large_fixed_order_dynamic_programming_keeps_multiple_best_state_sequences():
    costs, transitions = _graph(8, 42)
    # Only this order has supported transitions. Its 256 state combinations
    # can be checked independently, although enumerating all orders cannot fit.
    for (a, _sa), (b, _sb) in transitions:
        if b != (a + 1) % 8:
            transitions[((a, _sa), (b, _sb))] = math.inf
    expected = _oracle(costs, transitions, [tuple(range(8))])
    result = search_group_states(
        costs, lambda a, b: transitions[a, b], beam_width=1, candidate_count=4, maximum_expanded_states=5000
    )
    assert result.diagnostics["search_method"] == "beam_orders_with_fixed_order_state_dp"
    assert result.diagnostics["fixed_orders_state_optimized"] >= 1
    assert not result.diagnostics["objective_optimality_proven"]
    assert [sequence for _cost, sequence in result.circuits] == [sequence for _cost, sequence in expected[:4]]
    np.testing.assert_allclose([cost for cost, _ in result.circuits], [cost for cost, _ in expected[:4]], atol=1e-12)


def test_geometry_budget_preserves_only_complete_incumbents_and_does_not_claim_optimality():
    costs, transitions = _graph(3)
    calls = 0

    def evaluate(a, b):
        nonlocal calls
        calls += 1
        if calls == 8:
            raise RoutingEvaluationBudgetExceeded("curve_generation_budget")
        return transitions[a, b]

    result = search_group_states(costs, evaluate, beam_width=1, candidate_count=6, maximum_expanded_states=1000)
    assert result.circuits
    assert result.diagnostics["budget_exhaustion_reason"] == "curve_generation_budget"
    assert not result.diagnostics["objective_optimality_proven"]
    for cost, sequence in result.circuits:
        assert len(sequence) == 3
        expected = sum(costs[g][s] for g, s in sequence)
        expected += sum(transitions[a, b] for a, b in zip(sequence, (*sequence[1:], sequence[0])))
        assert cost == pytest.approx(expected)


def test_disconnected_supported_graph_is_not_a_physical_infeasibility_certificate():
    result = search_group_states(
        [[0, 0]] * 3, lambda a, b: math.inf, beam_width=1, candidate_count=2, maximum_expanded_states=1000
    )
    assert not result.circuits
    assert result.diagnostics["port_graph_infeasibility_proven"]
    assert not result.diagnostics["physical_feasibility_proven"]
    assert result.diagnostics["optimality_scope"].startswith("enumerated_additive_state_graph")


def test_single_group_state_optimization_includes_its_own_closing_return():
    costs, transitions = _graph(1)
    result = search_group_states(
        costs, lambda a, b: transitions[a, b], beam_width=1, candidate_count=6, maximum_expanded_states=2
    )
    assert result.circuits == _oracle(costs, transitions)
    assert result.diagnostics["expanded_states"] == 2
