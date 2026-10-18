"""Progressive grids reduce proposal work without changing physical screens."""

from types import SimpleNamespace

import numpy as np
import pytest

from _group_routing_cases import make_group_routing_case
from coilgen import joint_route_shapes as shapes
from coilgen import joint_transition_curves as transitions
from coilgen.connector_bridges import BridgeGeneration
from coilgen.joint_curve_search import representative_values


@pytest.mark.parametrize("level, expected", [(0, 4), (1, 18), (None, 64)])
def test_direct_grid_is_reduced_before_any_curve_sampling(monkeypatch, level, expected):
    tried = []

    def reject_cusp(controls, normal, height, peak):
        tried.append((controls.copy(), height, peak))
        return 0.0

    monkeypatch.setattr(transitions, "_minimum_speed", reject_cusp)
    monkeypatch.setattr(transitions, "_evaluate", lambda *_args: pytest.fail("Rejected grid was sampled"))
    result = transitions.generate_joint_transition_curves(
        [[0, 0, 0], [0.1, 0, 0]],
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        np.empty((0, 2, 3)),
        maximum_height_m=0.006,
        minimum_radius_m=0.0014,
        maximum_segment_length_m=0.001,
        chord_tolerance_m=1e-5,
        search_level=level,
    )
    assert len(tried) == expected
    assert not result.candidates
    assert "curve_cusp_or_zero_speed" in result.rejected_reasons
    assert {height for _, height, _ in tried} >= {0.0, 0.006}


def test_representative_parameters_retain_both_extremes():
    values = tuple(range(16))
    initial = representative_values(values, 0)
    broader = representative_values(values, 1)
    assert initial == (0, 15)
    assert set(initial) <= set(broader) <= set(values)
    assert representative_values(values, None) == values


def test_widening_keeps_cumulative_work_and_separates_external_cache(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    cache, levels, charts = {}, [], []

    def generate(points, *_args, **kwargs):
        levels.append(kwargs["search_level"])
        return BridgeGeneration((SimpleNamespace(points=points.copy()),), "fixture")

    monkeypatch.setattr(shapes, "generate_joint_transition_curves", generate)
    for name in (
        "generate_axial_return_curves",
        "generate_planar_return_curves",
        "generate_cylindrical_transition_curves",
    ):
        monkeypatch.setattr(shapes, name, lambda *_args, **_kwargs: BridgeGeneration())
    monkeypatch.setattr(shapes, "_validate_support", lambda *_args: ())

    def no_chart(_projector):
        charts.append(True)
        raise ValueError("fixture support is planar")

    monkeypatch.setattr(shapes, "_developable_chart", no_chart)
    evaluator = shapes.JointTransitionEvaluator(
        case.part,
        case.config,
        search_mode="progressive",
        transition_cache=cache,
        _protected_segments={},
    )
    monkeypatch.setattr(evaluator, "jump_count", lambda _curve: 0)
    source, tangent = np.array([[-0.005, 0, 0], [0.005, 0, 0]]), np.array([1.0, 0, 0])
    first = evaluator.supported_curves(source, tangent, tangent)
    assert evaluator.supported_curves(source, tangent, tangent) is first
    previous_trials = evaluator.diagnostics["shape_trial_upper_bound"]
    previous_scores = evaluator.work.interaction_scores
    evaluator.set_search_level(1)
    widened = evaluator.supported_curves(source, tangent, tangent)
    assert any(curve is first[0] for curve in widened)
    assert levels == [0, 1] and len(cache) == 2 and len(charts) == 1
    assert evaluator.diagnostics["shape_trial_upper_bound"] > previous_trials
    assert evaluator.work.interaction_scores == previous_scores
    assert evaluator.diagnostics["support_screen_cache_hits"] == 1
    assert evaluator.diagnostics["search_level"] == 1
    with pytest.raises(ValueError, match="only widen"):
        evaluator.set_search_level(0)


def test_exact_protected_collision_rejects_before_support_projection(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    protected = {(0, 0, 0): np.array([[0, -0.02, 0], [0, 0.02, 0]])}
    evaluator = shapes.JointTransitionEvaluator(
        case.part,
        case.config,
        _protected_segments=protected,
        search_mode="progressive",
    )
    points = np.array([[-0.015, 0, 0], [0.015, 0, 0]])
    monkeypatch.setattr(shapes, "_validate_support", lambda *_args: pytest.fail("Colliding curve projected"))
    selected, reasons = evaluator._supported(
        BridgeGeneration((SimpleNamespace(points=points),)),
        points,
        np.array([0, 0, 1.0]),
    )
    assert selected == [] and "protected_corridor_clearance" in reasons
    assert evaluator.diagnostics["support_checks"] == 0
    assert evaluator.diagnostics["protected_clearance_rejections_before_support"] == 1


def test_progressive_shortlist_retains_both_windings_and_near_far_detour_rails(monkeypatch):
    case = make_group_routing_case("facing_corridor")
    evaluator = shapes.JointTransitionEvaluator(case.part, case.config, search_mode="progressive")
    monkeypatch.setattr(evaluator, "jump_count", lambda _curve: 1)
    detours = []
    for winding in (-1, 1):
        for side in (-1, 1):
            for height in (0.0, 0.006):
                for extent in (0.01, 0.03):
                    controls = np.array(
                        [
                            [0, 0, 0],
                            [0, 0.001 * side, 0],
                            [0, extent * side, 0],
                            [0.05 * winding, extent * side, 0],
                            [0.1 * winding, 0, 0],
                        ]
                    )
                    detours.append(
                        SimpleNamespace(
                            points=controls.copy(),
                            control_points_m=controls,
                            height_m=height,
                            profile_family="cylinder_detour_return" if height == 0 else "cylinder_detour_lifted_return",
                        )
                    )
    cheap = [SimpleNamespace(points=np.array([[0, 0, 0], [length, 0, 0]])) for length in np.linspace(0.001, 0.02, 32)]
    selected = evaluator._shortlist((*cheap, *detours))
    assert len(selected) <= 32
    assert all(any(curve is value for value in selected) for curve in detours)
